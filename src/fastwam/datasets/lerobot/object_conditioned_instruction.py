"""Runtime instruction sampling conditioned on per-episode object bindings.

The manifest deliberately lives outside the LeRobot parquet files.  A trajectory
keeps one audited object binding, while every dataset access independently draws
one compatible natural-language instruction.  This avoids turning ``task_index``
into a permanent frame-to-caption assignment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


MANIFEST_SCHEMA = "fastwam_object_conditioned_instruction_v1"
MANIFEST_SCHEMA_V2 = "fastwam_object_conditioned_instruction_v2"
SUPPORTED_MANIFEST_SCHEMAS = {MANIFEST_SCHEMA, MANIFEST_SCHEMA_V2}
PER_DRAW_SAMPLER_SCHEMA = "fastwam_hanging_mug_prompt_per_draw_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected an object at {path}:{line_number}, got "
                    f"{type(value).__name__}"
                )
            records.append(value)
    return records


class ObjectConditionedInstructionSampler:
    """Load an audited prompt inventory and sample by dataset/episode binding."""

    # RobotVideoDataset uses this capability marker to pass an explicit draw
    # ordinal.  Keeping the marker opt-in preserves compatibility with older
    # episode samplers and small test doubles.
    supports_draw_ordinal = True

    def __init__(self, manifest_path: str | Path, dataset_dirs: Sequence[str | Path]):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Object-conditioned instruction manifest does not exist: "
                f"{self.manifest_path}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise TypeError(f"Instruction manifest must be an object: {self.manifest_path}")
        if manifest.get("schema") not in SUPPORTED_MANIFEST_SCHEMAS:
            raise ValueError(
                f"Unsupported instruction manifest schema at {self.manifest_path}: "
                f"{manifest.get('schema')!r}"
            )
        if manifest.get("status") != "READY":
            raise ValueError(f"Instruction manifest is not READY: {self.manifest_path}")

        inventory_value = manifest.get("prompt_inventory")
        if not isinstance(inventory_value, dict):
            raise TypeError("Instruction manifest needs object field `prompt_inventory`")
        inventory_path_value = inventory_value.get("path")
        if not isinstance(inventory_path_value, str) or not inventory_path_value:
            raise ValueError("prompt_inventory.path must be a non-empty string")
        inventory_path = Path(inventory_path_value).expanduser()
        if not inventory_path.is_absolute():
            inventory_path = self.manifest_path.parent / inventory_path
        self.inventory_path = inventory_path.resolve()
        if not self.inventory_path.is_file():
            raise FileNotFoundError(self.inventory_path)
        declared_inventory_sha = inventory_value.get("sha256")
        observed_inventory_sha = _sha256_file(self.inventory_path)
        if declared_inventory_sha != observed_inventory_sha:
            raise ValueError(
                "Prompt inventory SHA256 mismatch: "
                f"declared={declared_inventory_sha} observed={observed_inventory_sha}"
            )

        pools: dict[str, list[str]] = {}
        tasks_by_prompt_id: dict[int, str] = {}
        unique_tasks: set[str] = set()
        inventory = _read_jsonl(self.inventory_path)
        for row_number, record in enumerate(inventory, start=1):
            task = record.get("task")
            bindings = record.get("bindings")
            if not isinstance(task, str) or not task:
                raise ValueError(
                    f"Prompt inventory row {row_number} needs non-empty string `task`"
                )
            if task in unique_tasks:
                raise ValueError(f"Duplicate task in prompt inventory: {task!r}")
            unique_tasks.add(task)
            prompt_id = int(record.get("prompt_id", row_number - 1))
            if prompt_id in tasks_by_prompt_id:
                raise ValueError(f"Duplicate prompt_id in prompt inventory: {prompt_id}")
            tasks_by_prompt_id[prompt_id] = task
            if not isinstance(bindings, list) or not bindings:
                raise ValueError(
                    f"Prompt inventory row {row_number} needs non-empty list `bindings`"
                )
            for binding in bindings:
                if not isinstance(binding, str) or not binding:
                    raise ValueError(
                        f"Invalid binding at prompt inventory row {row_number}: {binding!r}"
                    )
                pools.setdefault(binding, []).append(task)

        expected_unique = int(inventory_value.get("unique_prompts", len(inventory)))
        if len(inventory) != expected_unique:
            raise ValueError(
                f"Prompt inventory count mismatch: expected={expected_unique} "
                f"observed={len(inventory)}"
            )
        expected_per_binding = inventory_value.get("prompts_per_binding")
        if expected_per_binding is not None:
            expected_per_binding = int(expected_per_binding)
            bad = {
                binding: len(tasks)
                for binding, tasks in pools.items()
                if len(tasks) != expected_per_binding
            }
            if bad:
                raise ValueError(
                    "Object-conditioned prompt pool sizes differ from the declared "
                    f"{expected_per_binding}: {bad}"
                )
        self.pools = {binding: tuple(tasks) for binding, tasks in pools.items()}

        sources_value = manifest.get("sources")
        if not isinstance(sources_value, list) or not sources_value:
            raise ValueError("Instruction manifest needs non-empty list `sources`")
        sources_by_fingerprint: dict[tuple[str, str], dict[str, Any]] = {}
        for source in sources_value:
            if not isinstance(source, dict):
                raise TypeError("Each instruction manifest source must be an object")
            fingerprint = (
                str(source.get("info_sha256", "")),
                str(source.get("episodes_sha256", "")),
            )
            if not all(fingerprint):
                raise ValueError(
                    f"Source {source.get('id')!r} is missing metadata fingerprints"
                )
            if fingerprint in sources_by_fingerprint:
                raise ValueError(f"Duplicate source fingerprint: {fingerprint}")
            raw_bindings = source.get("episode_bindings")
            if not isinstance(raw_bindings, dict) or not raw_bindings:
                raise ValueError(
                    f"Source {source.get('id')!r} needs non-empty episode_bindings"
                )
            episode_bindings: dict[int, str] = {}
            episode_prompt_pools: dict[int, tuple[str, ...]] = {}
            for episode, value in raw_bindings.items():
                episode_index = int(episode)
                binding = value.get("binding") if isinstance(value, dict) else value
                explicit_prompt_ids = (
                    value.get("prompt_ids") if isinstance(value, dict) else None
                )
                if explicit_prompt_ids is not None:
                    if (
                        not isinstance(explicit_prompt_ids, list)
                        or not explicit_prompt_ids
                        or len(explicit_prompt_ids) != len(set(explicit_prompt_ids))
                    ):
                        raise ValueError(
                            "Explicit episode prompt_ids must be a non-empty unique list: "
                            f"source={source.get('id')!r} episode={episode_index}"
                        )
                    try:
                        candidates = tuple(
                            tasks_by_prompt_id[int(prompt_id)]
                            for prompt_id in explicit_prompt_ids
                        )
                    except KeyError as error:
                        raise ValueError(
                            "Unknown explicit prompt_id for "
                            f"source={source.get('id')!r} episode={episode_index}: "
                            f"{error.args[0]}"
                        ) from error
                    if not isinstance(binding, str) or not binding:
                        raise ValueError(
                            "Explicit episode prompt pools still require a stable "
                            f"binding label: source={source.get('id')!r} "
                            f"episode={episode_index}"
                        )
                else:
                    if not isinstance(binding, str) or binding not in self.pools:
                        raise ValueError(
                            f"Unknown prompt binding for source={source.get('id')!r} "
                            f"episode={episode_index}: {binding!r}"
                        )
                    candidates = self.pools[binding]
                episode_bindings[episode_index] = binding
                episode_prompt_pools[episode_index] = candidates
            normalized_source = dict(source)
            normalized_source["episode_bindings"] = episode_bindings
            normalized_source["episode_prompt_pools"] = episode_prompt_pools
            sources_by_fingerprint[fingerprint] = normalized_source

        self._sources_by_dataset_index: tuple[dict[str, Any], ...] = tuple(
            self._match_dataset_root(Path(root), sources_by_fingerprint)
            for root in dataset_dirs
        )
        self.manifest_sha256 = _sha256_file(self.manifest_path)
        self.unique_prompt_count = len(inventory)
        self.runtime_episode_pool_sizes_by_source = {
            str(source.get("id", "")): tuple(
                sorted(
                    {
                        len(candidates)
                        for candidates in source["episode_prompt_pools"].values()
                    }
                )
            )
            for source in self._sources_by_dataset_index
        }

    @staticmethod
    def _match_dataset_root(
        root: Path,
        sources_by_fingerprint: dict[tuple[str, str], dict[str, Any]],
    ) -> dict[str, Any]:
        root = root.expanduser().resolve()
        info_path = root / "meta" / "info.json"
        episodes_path = root / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(
                f"Cannot fingerprint object-conditioned dataset root: {root}"
            )
        fingerprint = (_sha256_file(info_path), _sha256_file(episodes_path))
        source = sources_by_fingerprint.get(fingerprint)
        if source is None:
            raise ValueError(
                "No object-conditioned instruction source matches dataset root "
                f"{root}; metadata fingerprint={fingerprint}"
            )
        episode_rows = _read_jsonl(episodes_path)
        observed_episodes = {int(record["episode_index"]) for record in episode_rows}
        declared_episodes = set(source["episode_bindings"])
        if observed_episodes != declared_episodes:
            raise ValueError(
                f"Episode binding coverage mismatch for {root}: "
                f"missing={sorted(observed_episodes - declared_episodes)[:10]} "
                f"extra={sorted(declared_episodes - observed_episodes)[:10]}"
            )
        return source

    def sample_task(
        self,
        dataset_index: int,
        episode_index: int,
        *,
        rng: Any | None = None,
        draw_ordinal: int | None = None,
        epoch: int = 0,
        prompt_seed: int = 0,
    ) -> str:
        """Draw one compatible task string for this training access.

        ``draw_ordinal`` is the runtime training-draw identity.  When supplied,
        selection is stateless and keyed by epoch, seed, source, episode, and
        manifest, so a trajectory never owns one permanent caption.  The legacy
        ``rng`` path remains available for focused tests and older callers.
        """

        try:
            source = self._sources_by_dataset_index[int(dataset_index)]
        except IndexError as error:
            raise IndexError(f"Unknown dataset_index={dataset_index}") from error
        try:
            binding = source["episode_bindings"][int(episode_index)]
        except KeyError as error:
            raise KeyError(
                f"No object binding for source={source.get('id')!r} "
                f"episode={episode_index}"
            ) from error
        candidates = source["episode_prompt_pools"][int(episode_index)]
        if not candidates:
            raise ValueError(
                f"Empty object-conditioned prompt pool for source={source.get('id')!r} "
                f"episode={episode_index} binding={binding!r}"
            )
        if draw_ordinal is None:
            generator = np.random if rng is None else rng
            position = int(generator.randint(len(candidates)))
        else:
            # Hashing the draw identity avoids dependence on DataLoader worker
            # RNG state and makes the per-draw contract auditable/replayable.
            # The ordinal is intentionally not an episode-level value.
            key = "\0".join(
                (
                    PER_DRAW_SAMPLER_SCHEMA,
                    self.manifest_sha256,
                    str(int(prompt_seed)),
                    str(int(epoch)),
                    str(int(draw_ordinal)),
                    str(source.get("id", dataset_index)),
                    str(int(dataset_index)),
                    str(int(episode_index)),
                )
            )
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            position = int.from_bytes(digest[:8], "big") % len(candidates)
        return candidates[position]

    def binding_for_episode(self, dataset_index: int, episode_index: int) -> str:
        source = self._sources_by_dataset_index[int(dataset_index)]
        return source["episode_bindings"][int(episode_index)]

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(str(source.get("id", "")) for source in self._sources_by_dataset_index)
