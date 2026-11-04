# Modified by Open-HumanScale contributors: removed internal development references.
"""Logical episode-segment manifests shared by training and statistics.

Large ego recordings often contain several temporally annotated substeps.  Rewriting
or duplicating their parquet/video payloads is unnecessary: a segment manifest can
declare half-open frame ranges that the loader treats as independent episodes.  The
same parser is used by the training loader, text-cache precomputation, and sampled
normalization statistics so caption and boundary semantics cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
CODEBOOK_SCHEMA_VERSION = 2
CODEBOOK_AXES = frozenset(
    {
        "visual_background",
        "background_layout_geometry",
        "background_appearance_illumination",
        "background_illumination_luminance",
        "background_palette_chroma",
        "background_contrast",
        "background_bright_area_fraction",
        "background_cool_chroma_fraction",
        "background_appearance_coarse",
        "background_appearance_fine_residual",
        "lighting_level",
        "lighting_temperature",
        "dominant_background_palette",
        "workspace_surface_type",
        "workspace_surface_color",
        "workspace_surface_pattern",
        "support_surface_base",
        "workspace_overlay",
        "fixed_fixtures",
        "coarse_layout_geometry",
        "persistent_clutter_level",
        "persistent_fixture_tokens",
        "workspace_surface_material",
        "persistent_layout_clutter",
        "semantic_place",
        "skill",
        "object",
        "viewpoint",
        "capture_domain",
    }
)
VALID_SOURCE_HAND_TAGS = {"left", "right", "both"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAMESPACED_ID_RE = re.compile(r"^[a-z][a-z0-9_]*:[^:\s|]+:[^:\s|].*$")


def _canonical_content_sha256(value: dict[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"created_at_utc", "content_sha256"}
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_legacy_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest fields visible before codebook schema v2.

    The projection intentionally retains segment order, scalar ``environment_id``,
    and the complete legacy sampling contract while removing only v2 annotations and
    hash/timestamp fields that necessarily change when a new document is emitted.
    """

    projection = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "created_at_utc",
            "content_sha256",
            "codebook_schema_version",
            "codebooks",
            "codebook_annotation_provenance",
        }
    }
    roots = projection.get("roots")
    if isinstance(roots, Mapping):
        projected_roots: dict[str, Any] = {}
        for root, raw_entry in roots.items():
            if not isinstance(raw_entry, Mapping):
                projected_roots[str(root)] = raw_entry
                continue
            entry = dict(raw_entry)
            segments = entry.get("segments")
            if isinstance(segments, list):
                entry["segments"] = [
                    {
                        key: item
                        for key, item in segment.items()
                        if key != "codebook_axes"
                    }
                    if isinstance(segment, Mapping)
                    else segment
                    for segment in segments
                ]
            projected_roots[str(root)] = entry
        projection["roots"] = projected_roots
    return projection


def canonical_legacy_projection_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_legacy_projection(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return text


def _looks_like_factorized_environment_id(value: str) -> bool:
    components = [component.strip() for component in value.split("|")]
    return len(components) > 1 and all(
        _NAMESPACED_ID_RE.fullmatch(component) is not None
        for component in components
    )


def _validate_codebook_registry(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_registry = value.get("codebooks")
    if not isinstance(raw_registry, dict):
        raise TypeError("codebook schema v2 requires top-level object `codebooks`")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_axis, raw_entry in raw_registry.items():
        axis = str(raw_axis).strip()
        if axis not in CODEBOOK_AXES:
            raise ValueError(f"Invalid codebook axis {axis!r}; expected {sorted(CODEBOOK_AXES)}")
        if not isinstance(raw_entry, dict):
            raise TypeError(f"codebooks.{axis} must be an object")
        entry = dict(raw_entry)
        for field in ("method", "version", "fit_population", "fit_group_field"):
            entry[field] = _require_nonempty_string(
                entry.get(field), f"codebooks.{axis}.{field}"
            )
        entry["assignment_artifact_sha256"] = _require_sha256(
            entry.get("assignment_artifact_sha256"),
            f"codebooks.{axis}.assignment_artifact_sha256",
        )
        raw_ids = entry.get("ids", [])
        if not isinstance(raw_ids, list):
            raise TypeError(f"codebooks.{axis}.ids must be a list")
        ids = [_require_nonempty_string(item, f"codebooks.{axis}.ids") for item in raw_ids]
        if len(set(ids)) != len(ids):
            raise ValueError(f"codebooks.{axis}.ids contains duplicate IDs")
        invalid_ids = [item for item in ids if _NAMESPACED_ID_RE.fullmatch(item) is None]
        if invalid_ids:
            raise ValueError(
                f"codebooks.{axis}.ids contains non-namespaced IDs: {invalid_ids[:3]}"
            )
        entry["ids"] = ids
        encoder_revisions = entry.get("encoder_revisions")
        if encoder_revisions is not None and not isinstance(encoder_revisions, dict):
            raise ValueError(f"codebooks.{axis}.encoder_revisions must be an object")
        capacity_prior = entry.get("capacity_prior")
        if capacity_prior is not None and not isinstance(capacity_prior, dict):
            raise ValueError(f"codebooks.{axis}.capacity_prior must be an object")
        normalized[axis] = entry
    return normalized


def _validate_segment_codebook_axes(
    raw_axes: Any,
    registry: Mapping[str, Mapping[str, Any]],
    *,
    location: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_axes, dict):
        raise TypeError(f"{location}.codebook_axes must be an object")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_axis, raw_record in raw_axes.items():
        axis = str(raw_axis).strip()
        if axis not in CODEBOOK_AXES:
            raise ValueError(f"Invalid codebook axis {axis!r} at {location}")
        if axis not in registry:
            raise ValueError(f"{location} uses axis {axis!r} absent from codebooks registry")
        if not isinstance(raw_record, dict):
            raise TypeError(f"{location}.codebook_axes.{axis} must be an object")
        record = dict(raw_record)
        atom_id = _require_nonempty_string(
            record.get("id"), f"{location}.codebook_axes.{axis}.id"
        )
        if _NAMESPACED_ID_RE.fullmatch(atom_id) is None:
            raise ValueError(
                f"{location}.codebook_axes.{axis}.id must be namespaced as "
                "`namespace:version:value`"
            )
        declared_ids = registry[axis].get("ids", [])
        if atom_id not in declared_ids:
            raise ValueError(
                f"{location}.codebook_axes.{axis}.id={atom_id!r} is absent from "
                "the registry"
            )
        for field in ("method", "version"):
            record[field] = _require_nonempty_string(
                record.get(field), f"{location}.codebook_axes.{axis}.{field}"
            )
            if record[field] != registry[axis][field]:
                raise ValueError(
                    f"{location}.codebook_axes.{axis}.{field} disagrees with registry"
                )
        confidence = record.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(f"{location}.codebook_axes.{axis}.confidence must be numeric")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"{location}.codebook_axes.{axis}.confidence must be in [0, 1]"
            )
        record["confidence"] = confidence
        record["artifact_sha256"] = _require_sha256(
            record.get("artifact_sha256"),
            f"{location}.codebook_axes.{axis}.artifact_sha256",
        )
        if record["artifact_sha256"] != registry[axis]["assignment_artifact_sha256"]:
            raise ValueError(
                f"{location}.codebook_axes.{axis}.artifact_sha256 disagrees with registry"
            )
        record["provenance_sha256"] = _require_sha256(
            record.get("provenance_sha256"),
            f"{location}.codebook_axes.{axis}.provenance_sha256",
        )
        if "source_value" in record and not isinstance(
            record["source_value"], (str, int, float, bool, type(None))
        ):
            raise ValueError(
                f"{location}.codebook_axes.{axis}.source_value must be scalar"
            )
        normalized[axis] = record
    return normalized


def _validate_active_axis_projection(
    value: Mapping[str, Any],
    normalized_roots: Mapping[str, Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    if provenance.get("projection_schema_version") != 1:
        raise ValueError("active-axis projection requires projection_schema_version=1")
    active_axis = _require_nonempty_string(
        provenance.get("active_axis"),
        "codebook_annotation_provenance.active_axis",
    )
    if active_axis not in CODEBOOK_AXES or active_axis not in registry:
        raise ValueError(
            f"Projected active_axis={active_axis!r} is absent from the codebook registry"
        )
    for field in (
        "source_annotation_manifest_file_sha256",
        "source_annotation_manifest_content_sha256",
        "registry_sha256",
        "assignment_sha256",
        "assignment_artifact_sha256",
        "projected_population_sha256",
        "projection_policy_sha256",
        "environment_ids_sha256",
        "legacy_projection_before_sha256",
        "legacy_projection_after_sha256",
    ):
        _require_sha256(
            provenance.get(field),
            f"codebook_annotation_provenance.{field}",
        )

    source_provenance = provenance.get("source_annotation_provenance")
    if not isinstance(source_provenance, dict) or source_provenance.get("mode") != (
        "annotation_only"
    ):
        raise ValueError(
            "active-axis projection must retain its annotation-only source provenance"
        )
    source_legacy_sha = _require_sha256(
        source_provenance.get("legacy_projection_sha256"),
        "source_annotation_provenance.legacy_projection_sha256",
    )
    if source_legacy_sha != provenance["legacy_projection_before_sha256"]:
        raise ValueError(
            "Active-axis legacy before hash disagrees with source annotation provenance"
        )
    if provenance.get("legacy_projection_unchanged") is not False:
        raise ValueError(
            "active-axis projections must declare legacy_projection_unchanged=false"
        )
    if (
        provenance["assignment_artifact_sha256"]
        != registry[active_axis]["assignment_artifact_sha256"]
    ):
        raise ValueError(
            "Active-axis assignment artifact hash disagrees with the registry"
        )
    observed_registry_sha = _canonical_value_sha256(registry[active_axis])
    if provenance["registry_sha256"] != observed_registry_sha:
        raise ValueError(
            "Active-axis registry hash mismatch: "
            f"declared={provenance['registry_sha256']}, "
            f"observed={observed_registry_sha}"
        )

    policy = provenance.get("projection_policy")
    if not isinstance(policy, dict):
        raise TypeError("active-axis projection requires object projection_policy")
    if policy.get("active_axis") != active_axis:
        raise ValueError("projection_policy.active_axis disagrees with active_axis")
    observed_policy_sha = _canonical_value_sha256(policy)
    if provenance["projection_policy_sha256"] != observed_policy_sha:
        raise ValueError("Active-axis projection policy hash mismatch")

    contract = value.get("environment_sampling")
    if not isinstance(contract, dict) or contract.get("mode") != "uniform":
        raise ValueError(
            "active-axis projection requires a uniform environment_sampling contract"
        )
    if contract.get("active_axis") != active_axis:
        raise ValueError(
            "environment_sampling.active_axis disagrees with projection provenance"
        )
    if contract.get("projection_policy_sha256") != observed_policy_sha:
        raise ValueError(
            "environment_sampling projection policy hash disagrees with provenance"
        )
    raw_order = contract.get("environment_order")
    if not isinstance(raw_order, list) or not raw_order:
        raise ValueError(
            "active-axis environment_sampling.environment_order must be non-empty"
        )
    environment_order = [str(item).strip() for item in raw_order]
    if len(set(environment_order)) != len(environment_order):
        raise ValueError("active-axis environment_order contains duplicate IDs")
    undeclared = [
        item for item in environment_order if item not in registry[active_axis]["ids"]
    ]
    if undeclared:
        raise ValueError(
            f"active-axis environment_order contains unregistered IDs: {undeclared[:3]}"
        )
    observed_environment_ids_sha = hashlib.sha256(
        "\n".join(environment_order).encode("utf-8")
    ).hexdigest()
    if (
        contract.get("environment_ids_sha256") != observed_environment_ids_sha
        or provenance["environment_ids_sha256"] != observed_environment_ids_sha
    ):
        raise ValueError("Active-axis environment ID hash mismatch")

    projected_population = []
    for root, entry in normalized_roots.items():
        segments = entry.get("segments")
        if segments is None:
            raise ValueError("active-axis projections cannot contain passthrough roots")
        for position, segment in enumerate(segments):
            axes = segment.get("codebook_axes") or {}
            record = axes.get(active_axis)
            if record is None:
                raise ValueError(
                    f"Projected segment {root}:{position} is missing active axis "
                    f"{active_axis!r}"
                )
            environment_id = str(segment.get("environment_id", "")).strip()
            if environment_id != record["id"]:
                raise ValueError(
                    f"Projected environment_id={environment_id!r} does not match "
                    f"{active_axis} code={record['id']!r} at {root}:{position}"
                )
            if environment_id not in environment_order:
                raise ValueError(
                    f"Projected segment uses environment ID outside environment_order: "
                    f"{environment_id!r}"
                )
            projected_population.append(
                {
                    "root": str(Path(root).expanduser().resolve()),
                    "episode_index": int(segment["episode_index"]),
                    "segment_index": int(segment.get("segment_index", 0)),
                    "start_frame": int(segment["start_frame"]),
                    "end_frame": int(segment["end_frame"]),
                    "environment_id": environment_id,
                }
            )
    observed_population_sha = _canonical_value_sha256(projected_population)
    if provenance["projected_population_sha256"] != observed_population_sha:
        raise ValueError("Active-axis projected population hash mismatch")

    observed_legacy_after = canonical_legacy_projection_sha256(value)
    if provenance["legacy_projection_after_sha256"] != observed_legacy_after:
        raise ValueError(
            "Active-axis legacy after hash mismatch: "
            f"declared={provenance['legacy_projection_after_sha256']}, "
            f"observed={observed_legacy_after}"
        )
    if provenance["legacy_projection_before_sha256"] == observed_legacy_after:
        raise ValueError("Active-axis projection did not change the legacy projection")


def load_episode_segment_manifest(
    manifest: str | Path | None,
    *,
    include_document: bool = False,
) -> (
    tuple[Path | None, dict[str, dict[str, Any]]]
    | tuple[Path | None, dict[str, dict[str, Any]], dict[str, Any] | None]
):
    """Load and validate a legacy or annotation-only logical-segment manifest.

    Segment ranges are always half-open on the *materialized LeRobot timeline*:
    ``[start_frame, end_frame)``.  Extra segment fields are retained verbatim because
    object/environment studies consume metadata such as ``environment_id``,
    ``scene_desc``, and ``objects``.  ``source_hand_tag`` may explicitly override the
    physical episode tag for a logical segment (for example, a per-hand payload
    reused as a verified paired ``both``-arm sample).

    A mixed corpus may additionally list ``passthrough_roots``. Those roots use their
    physical episodes/tasks unchanged while roots in ``roots`` use logical segments.
    The normalized mapping contains both kinds and marks passthrough entries with
    ``passthrough=true`` and ``segments=None`` so every downstream consumer can enforce
    the same exact configured-root inventory.
    """

    if manifest is None or str(manifest).strip().lower() in {"", "none", "null"}:
        if include_document:
            return None, {}, None
        return None, {}

    manifest_path = Path(str(manifest)).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Episode segment manifest does not exist: {manifest_path}"
        )

    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("version", -1)) != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported episode segment manifest at {manifest_path}; "
            f"expected version={MANIFEST_VERSION}"
        )
    if value.get("complete") is not True:
        raise ValueError(f"Episode segment manifest {manifest_path} is not marked complete")
    if value.get("frame_convention") != "materialized_half_open":
        raise ValueError(
            f"Episode segment manifest {manifest_path} must declare "
            "frame_convention='materialized_half_open'"
        )

    declared_sha = value.get("content_sha256")
    if not isinstance(declared_sha, str) or len(declared_sha) != 64:
        raise ValueError(
            f"Episode segment manifest {manifest_path} needs a 64-character "
            "content_sha256"
        )
    observed_sha = _canonical_content_sha256(value)
    if declared_sha != observed_sha:
        raise ValueError(
            f"Episode segment manifest content hash mismatch at {manifest_path}: "
            f"declared={declared_sha}, observed={observed_sha}"
        )

    has_v2_fields = any(
        key in value
        for key in ("codebook_schema_version", "codebooks", "codebook_annotation_provenance")
    )
    if has_v2_fields:
        if value.get("codebook_schema_version") != CODEBOOK_SCHEMA_VERSION:
            raise ValueError(
                "Codebook-annotated manifests require "
                f"codebook_schema_version={CODEBOOK_SCHEMA_VERSION}"
            )
        codebook_registry = _validate_codebook_registry(value)
        value["codebooks"] = codebook_registry
    else:
        codebook_registry = {}

    roots = value.get("roots")
    if not isinstance(roots, dict):
        raise TypeError(
            f"Episode segment manifest {manifest_path} needs object field `roots`"
        )

    normalized: dict[str, dict[str, Any]] = {}
    observed_segments = 0
    observed_frames = 0
    for raw_root, raw_entry in roots.items():
        if not isinstance(raw_entry, dict):
            raise TypeError(f"Segment manifest root entry must be an object: {raw_root!r}")
        raw_segments = raw_entry.get("segments")
        if not isinstance(raw_segments, list):
            raise TypeError(
                f"Segment manifest root {raw_root!r} needs list field `segments`"
            )

        segments: list[dict[str, Any]] = []
        previous_key: tuple[int, int, int, int] | None = None
        previous_end_by_episode: dict[int, int] = {}
        for position, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                raise TypeError(
                    f"Segment {position} for root {raw_root!r} must be an object"
                )
            episode_index = int(raw_segment.get("episode_index", -1))
            start_frame = int(raw_segment.get("start_frame", -1))
            end_frame = int(raw_segment.get("end_frame", -1))
            segment_index = int(raw_segment.get("segment_index", position))
            task = str(raw_segment.get("task", "")).strip()
            raw_source_hand_tag = raw_segment.get("source_hand_tag")
            source_hand_tag = None
            if raw_source_hand_tag is not None:
                source_hand_tag = str(raw_source_hand_tag).strip().lower()
                if source_hand_tag not in VALID_SOURCE_HAND_TAGS:
                    raise ValueError(
                        f"Invalid source_hand_tag={raw_source_hand_tag!r} at root "
                        f"{raw_root!r}, episode {episode_index}, segment "
                        f"{segment_index}; expected one of "
                        f"{sorted(VALID_SOURCE_HAND_TAGS)}"
                    )
            if episode_index < 0 or segment_index < 0:
                raise ValueError(
                    f"Negative episode/segment index at root {raw_root!r}, "
                    f"segment position {position}"
                )
            if start_frame < 0 or end_frame <= start_frame:
                raise ValueError(
                    f"Invalid half-open segment [{start_frame}, {end_frame}) at "
                    f"root {raw_root!r}, episode {episode_index}"
                )
            if not task:
                raise ValueError(
                    f"Empty task at root {raw_root!r}, episode {episode_index}, "
                    f"segment {segment_index}"
                )
            raw_environment_id = raw_segment.get("environment_id")
            if raw_environment_id is not None and _looks_like_factorized_environment_id(
                str(raw_environment_id)
            ):
                raise ValueError(
                    "Legacy environment_id must remain scalar and cannot contain a "
                    f"multi-axis tuple at root {raw_root!r}, episode {episode_index}, "
                    f"segment {segment_index}"
                )
            raw_codebook_axes = raw_segment.get("codebook_axes")
            if raw_codebook_axes is not None:
                if not has_v2_fields:
                    raise ValueError(
                        "Segment codebook_axes requires top-level "
                        f"codebook_schema_version={CODEBOOK_SCHEMA_VERSION}"
                    )
                normalized_axes = _validate_segment_codebook_axes(
                    raw_codebook_axes,
                    codebook_registry,
                    location=(
                        f"roots[{raw_root!r}].segments[{position}]"
                    ),
                )
            else:
                normalized_axes = None

            sort_key = (episode_index, start_frame, end_frame, segment_index)
            if previous_key is not None and sort_key <= previous_key:
                raise ValueError(
                    f"Segments for root {raw_root!r} must be strictly sorted by "
                    "(episode_index, start_frame, end_frame, segment_index)"
                )
            previous_key = sort_key
            previous_end = previous_end_by_episode.get(episode_index)
            if previous_end is not None and start_frame < previous_end:
                raise ValueError(
                    f"Overlapping segments for root {raw_root!r}, episode "
                    f"{episode_index}: previous end={previous_end}, "
                    f"next start={start_frame}"
                )
            previous_end_by_episode[episode_index] = end_frame

            segment = dict(raw_segment)
            segment.update(
                {
                    "episode_index": episode_index,
                    "segment_index": segment_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "task": task,
                }
            )
            if source_hand_tag is not None:
                segment["source_hand_tag"] = source_hand_tag
            if normalized_axes is not None:
                segment["codebook_axes"] = normalized_axes
            segments.append(segment)
            observed_segments += 1
            observed_frames += end_frame - start_frame

        declared_root_segments = raw_entry.get("total_segments")
        if declared_root_segments is not None and int(declared_root_segments) != len(segments):
            raise ValueError(
                f"Segment count mismatch for root {raw_root!r}: "
                f"declared={declared_root_segments}, observed={len(segments)}"
            )
        root_frames = sum(
            int(segment["end_frame"]) - int(segment["start_frame"])
            for segment in segments
        )
        declared_root_frames = raw_entry.get("total_segment_frames")
        if declared_root_frames is not None and int(declared_root_frames) != root_frames:
            raise ValueError(
                f"Segment-frame mismatch for root {raw_root!r}: "
                f"declared={declared_root_frames}, observed={root_frames}"
            )

        root = str(Path(str(raw_root)).expanduser().resolve())
        if root in normalized:
            raise ValueError(f"Manifest contains duplicate resolved root {root!r}")
        entry = dict(raw_entry)
        entry["segments"] = segments
        entry["total_segments"] = len(segments)
        entry["total_segment_frames"] = root_frames
        entry["passthrough"] = False
        normalized[root] = entry

    raw_passthrough_roots = value.get("passthrough_roots", [])
    if not isinstance(raw_passthrough_roots, list):
        raise TypeError(
            f"Episode segment manifest {manifest_path} field "
            "`passthrough_roots` must be a list"
        )
    seen_passthrough: set[str] = set()
    for raw_root in raw_passthrough_roots:
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise ValueError(
                f"Episode segment manifest {manifest_path} contains an invalid "
                f"passthrough root: {raw_root!r}"
            )
        root = str(Path(str(raw_root)).expanduser().resolve())
        if root in normalized:
            raise ValueError(
                f"Manifest root {root!r} cannot be both segmented and passthrough"
            )
        if root in seen_passthrough:
            raise ValueError(f"Duplicate passthrough root in manifest: {root!r}")
        seen_passthrough.add(root)
        normalized[root] = {
            "passthrough": True,
            "segments": None,
            "total_segments": None,
            "total_segment_frames": None,
        }

    declared_segments = value.get("total_segments")
    if declared_segments is not None and int(declared_segments) != observed_segments:
        raise ValueError(
            f"Episode segment manifest segment total mismatch at {manifest_path}: "
            f"declared={declared_segments}, observed={observed_segments}"
        )
    declared_frames = value.get("total_segment_frames")
    if declared_frames is not None and int(declared_frames) != observed_frames:
        raise ValueError(
            f"Episode segment manifest frame total mismatch at {manifest_path}: "
            f"declared={declared_frames}, observed={observed_frames}"
        )

    if has_v2_fields:
        provenance = value.get("codebook_annotation_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                "codebook schema v2 requires object `codebook_annotation_provenance`"
            )
        provenance_mode = provenance.get("mode")
        if provenance_mode == "annotation_only":
            if provenance.get("legacy_projection_unchanged") is not True:
                raise ValueError(
                    "annotation-only manifests must declare legacy_projection_unchanged=true"
                )
            declared_legacy_hash = _require_sha256(
                provenance.get("legacy_projection_sha256"),
                "codebook_annotation_provenance.legacy_projection_sha256",
            )
            observed_legacy_hash = canonical_legacy_projection_sha256(value)
            if declared_legacy_hash != observed_legacy_hash:
                raise ValueError(
                    "Annotation-only legacy projection hash mismatch: "
                    f"declared={declared_legacy_hash}, observed={observed_legacy_hash}"
                )
        elif provenance_mode == "active_axis_projection":
            _validate_active_axis_projection(
                value,
                normalized,
                codebook_registry,
                provenance,
            )
        else:
            raise ValueError(
                "codebook_annotation_provenance.mode must be 'annotation_only' or "
                "'active_axis_projection'"
            )

    if include_document:
        return manifest_path, normalized, value
    return manifest_path, normalized


def validate_segment_manifest_roots(
    manifest_path: Path | None,
    manifest_roots: dict[str, dict[str, Any]],
    configured_roots: set[str],
    *,
    configured_label: str = "dataset_dirs",
) -> None:
    """Require an exact root match, including explicitly declared passthrough roots."""

    if manifest_path is None:
        return
    manifest_set = set(manifest_roots)
    missing = sorted(configured_roots - manifest_set)
    unexpected = sorted(manifest_set - configured_roots)
    if missing or unexpected:
        raise ValueError(
            f"Episode segment manifest {manifest_path} roots must exactly match "
            f"{configured_label}; missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
