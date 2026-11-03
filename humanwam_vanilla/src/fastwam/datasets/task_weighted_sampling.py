"""Fail-closed task-weighted sampling contracts for a single dataset corpus.

The contract changes only how existing global frame-anchor indices are sampled.
It does not duplicate roots, alter examples, create source-specific normalization,
or attach a task-dependent loss weight.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any


TASK_WEIGHTED_SAMPLING_MODE = "task_weighted_replacement_v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_fraction(value: Any, *, location: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{location} must be a positive finite number, not bool")
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            f"{location} must be a positive finite rational value, got {value!r}"
        ) from exc
    if fraction <= 0:
        raise ValueError(f"{location} must be positive, got {value!r}")
    # Keep the integer mass representation bounded and auditable. This is far
    # above the intended small integer weights (for example hard=2/easy=1).
    if fraction.numerator > 1_000_000 or fraction.denominator > 1_000_000:
        raise ValueError(
            f"{location} is too precise/large for the bounded exact contract: "
            f"{fraction.numerator}/{fraction.denominator}"
        )
    return fraction


def build_task_weighted_sampling_contract(
    dataset_dirs: Sequence[str | Path],
    root_capacities: Sequence[int],
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return an exact-root, integer-mass sampling contract or ``None``.

    A configured weight applies to every existing frame anchor under its exact
    root. Sampling is with replacement from the expanded integer mass. Thus an
    anchor with integer weight 2 has exactly twice the probability of an anchor
    with integer weight 1. The effective stream length is the native corpus size
    rounded up to a complete configured global batch, preventing Accelerate from
    adding resume-dependent ``even_batches`` samples.
    """

    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError("task_weighted_sampling must be a mapping or null")
    expected_top = {
        "mode",
        "expected_batch_size_per_rank",
        "expected_num_processes",
        "global_batch_size",
        "rules",
    }
    unexpected_top = set(config) - expected_top
    missing_top = expected_top - set(config)
    if unexpected_top or missing_top:
        raise ValueError(
            "task_weighted_sampling fields mismatch: "
            f"missing={sorted(missing_top)}, unexpected={sorted(unexpected_top)}"
        )
    if config.get("mode") != TASK_WEIGHTED_SAMPLING_MODE:
        raise ValueError(
            "task_weighted_sampling.mode must be "
            f"{TASK_WEIGHTED_SAMPLING_MODE!r}"
        )
    raw_rules = config.get("rules")
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
        raise TypeError("task_weighted_sampling.rules must be a list")
    topology = {}
    for field in (
        "expected_batch_size_per_rank",
        "expected_num_processes",
        "global_batch_size",
    ):
        raw_value = config.get(field)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise TypeError(f"task_weighted_sampling.{field} must be an integer")
        if raw_value <= 0:
            raise ValueError(
                f"task_weighted_sampling.{field} must be positive, got {raw_value}"
            )
        topology[field] = raw_value
    global_batch_size = topology["global_batch_size"]
    topology_product = (
        topology["expected_batch_size_per_rank"]
        * topology["expected_num_processes"]
    )
    if topology_product != global_batch_size:
        raise ValueError(
            "task_weighted_sampling topology mismatch: "
            "expected_batch_size_per_rank*expected_num_processes="
            f"{topology_product}, global_batch_size={global_batch_size}"
        )

    roots = [str(Path(path).expanduser().resolve()) for path in dataset_dirs]
    capacities = [int(value) for value in root_capacities]
    if not roots:
        raise ValueError("task-weighted sampling requires at least one dataset root")
    if len(roots) != len(capacities):
        raise ValueError(
            f"root/capacity count mismatch: roots={len(roots)} "
            f"capacities={len(capacities)}"
        )
    if len(set(roots)) != len(roots):
        raise ValueError("task-weighted sampling dataset roots must be unique")
    if any(value <= 0 for value in capacities):
        raise ValueError(
            f"task-weighted sampling root capacities must be positive: {capacities}"
        )

    by_root: dict[str, tuple[str, Fraction, str]] = {}
    task_ids: set[str] = set()
    for position, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, Mapping):
            raise TypeError(
                f"task_weighted_sampling.rules[{position}] must be a mapping"
            )
        unexpected = set(raw_rule) - {"dataset_root", "task_id", "weight"}
        missing = {"dataset_root", "task_id", "weight"} - set(raw_rule)
        if unexpected or missing:
            raise ValueError(
                f"task_weighted_sampling.rules[{position}] fields mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        root = str(Path(str(raw_rule["dataset_root"])).expanduser().resolve())
        task_id = str(raw_rule["task_id"]).strip()
        if not task_id:
            raise ValueError(
                f"task_weighted_sampling.rules[{position}].task_id is empty"
            )
        if root in by_root:
            raise ValueError(
                f"Duplicate task-weighted sampling rule for root {root!r}"
            )
        if task_id in task_ids:
            raise ValueError(
                f"Duplicate task_id in task-weighted sampling rules: {task_id!r}"
            )
        fraction = _positive_fraction(
            raw_rule["weight"],
            location=f"task_weighted_sampling.rules[{position}].weight",
        )
        by_root[root] = (task_id, fraction, str(raw_rule["weight"]))
        task_ids.add(task_id)

    expected_roots = set(roots)
    configured_roots = set(by_root)
    if expected_roots != configured_roots:
        raise ValueError(
            "task_weighted_sampling rules must match dataset roots exactly; "
            f"missing={sorted(expected_roots - configured_roots)}, "
            f"extra={sorted(configured_roots - expected_roots)}"
        )

    denominator_lcm = 1
    ordered_fractions = []
    for root in roots:
        fraction = by_root[root][1]
        ordered_fractions.append(fraction)
        denominator_lcm = math.lcm(denominator_lcm, fraction.denominator)
    integer_weights = [
        fraction.numerator * (denominator_lcm // fraction.denominator)
        for fraction in ordered_fractions
    ]
    common_gcd = math.gcd(*integer_weights)
    integer_weights = [value // common_gcd for value in integer_weights]

    entries = []
    start = 0
    native_epoch_samples = sum(capacities)
    total_mass = sum(
        capacity * weight
        for capacity, weight in zip(capacities, integer_weights, strict=True)
    )
    if total_mass <= 0 or total_mass >= 1 << 63:
        raise ValueError(
            f"task-weighted integer mass must be in [1, 2^63), got {total_mass}"
        )
    for root, capacity, integer_weight in zip(
        roots, capacities, integer_weights, strict=True
    ):
        task_id, configured_fraction, configured_text = by_root[root]
        end = start + capacity
        probability_numerator = capacity * integer_weight
        entries.append(
            {
                "dataset_root": root,
                "task_id": task_id,
                "start": start,
                "end": end,
                "capacity": capacity,
                "configured_weight": configured_text,
                "configured_weight_fraction": (
                    f"{configured_fraction.numerator}/{configured_fraction.denominator}"
                ),
                "integer_weight": integer_weight,
                "expected_task_probability_numerator": probability_numerator,
                "expected_task_probability_denominator": total_mass,
                "expected_task_probability": probability_numerator / total_mass,
                "per_anchor_probability_numerator": integer_weight,
                "per_anchor_probability_denominator": total_mass,
                "per_anchor_expected_dose_ratio_vs_uniform_same_effective_slots": (
                    integer_weight * native_epoch_samples / total_mass
                ),
            }
        )
        start = end

    effective_epoch_samples = (
        (native_epoch_samples + global_batch_size - 1) // global_batch_size
    ) * global_batch_size
    payload = {
        "mode": TASK_WEIGHTED_SAMPLING_MODE,
        "replacement": True,
        "expected_batch_size_per_rank": topology[
            "expected_batch_size_per_rank"
        ],
        "expected_num_processes": topology["expected_num_processes"],
        "global_batch_size": global_batch_size,
        "native_epoch_samples": native_epoch_samples,
        "effective_epoch_samples": effective_epoch_samples,
        "padding_equivalent_extra": effective_epoch_samples - native_epoch_samples,
        "total_integer_mass": total_mass,
        "weighting": "uniform within exact task root; per-anchor mass proportional to integer_weight",
        "tasks": entries,
    }
    return {**payload, "sha256": _canonical_sha256(payload)}
