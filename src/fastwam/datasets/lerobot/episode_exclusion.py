"""Shared episode-exclusion manifest helpers.

The joint ego corpus is split across hundreds of self-contained LeRobot roots.  A
curation pass therefore records bad episodes as a non-destructive, per-root denylist
instead of rewriting the source trees.  Both training and standalone statistics must
interpret that denylist identically; keeping the parser here avoids subtle corpus
drift between those two paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


MANIFEST_VERSION = 1


def load_episode_exclusion_manifest(
    manifest: Optional[str | Path],
) -> tuple[Optional[Path], dict[str, dict[str, Any]]]:
    """Load and normalize a version-1 per-root episode denylist.

    The returned mapping is keyed by resolved absolute root paths.  Unknown fields are
    intentionally tolerated so a producer can attach provenance without requiring a
    synchronized loader update; the filtering contract itself remains the two fields
    normalized below.
    """

    if manifest is None or str(manifest).strip().lower() in {"", "none", "null"}:
        return None, {}

    manifest_path = Path(str(manifest)).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Episode exclusion manifest does not exist: {manifest_path}"
        )

    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("version", -1)) != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported episode exclusion manifest at {manifest_path}; "
            f"expected version={MANIFEST_VERSION}"
        )
    if value.get("complete") is not True:
        raise ValueError(
            f"Episode exclusion manifest {manifest_path} is not marked complete"
        )
    declared_content_sha256 = value.get("content_sha256")
    if not isinstance(declared_content_sha256, str) or len(declared_content_sha256) != 64:
        raise ValueError(
            f"Episode exclusion manifest {manifest_path} needs a 64-character "
            "content_sha256"
        )
    hash_payload = {
        key: item
        for key, item in value.items()
        if key not in {"created_at_utc", "content_sha256"}
    }
    encoded = json.dumps(
        hash_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_content_sha256 = hashlib.sha256(encoded).hexdigest()
    if declared_content_sha256 != observed_content_sha256:
        raise ValueError(
            f"Episode exclusion manifest content hash mismatch at {manifest_path}: "
            f"declared={declared_content_sha256}, observed={observed_content_sha256}"
        )
    roots = value.get("roots")
    if not isinstance(roots, dict):
        raise ValueError(
            f"Episode exclusion manifest {manifest_path} needs object field `roots`"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for root, entry in roots.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest root entry must be an object: {root!r}")
        raw_indices = entry.get("excluded_episode_indices", [])
        if not isinstance(raw_indices, list):
            raise ValueError(
                "Manifest excluded_episode_indices must be a list for root "
                f"{root!r}"
            )
        indices = [int(index) for index in raw_indices]
        if any(index < 0 for index in indices):
            raise ValueError(
                f"Manifest contains a negative episode index for root {root!r}"
            )
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"Manifest contains duplicate episode indices for root {root!r}"
            )

        excluded_frames = entry.get("excluded_frames")
        if excluded_frames is not None:
            excluded_frames = int(excluded_frames)
            if excluded_frames < 0:
                raise ValueError(
                    f"Manifest excluded_frames must be non-negative for root {root!r}"
                )

        normalized_root = str(Path(str(root)).expanduser().resolve())
        if normalized_root in normalized:
            raise ValueError(
                f"Manifest contains duplicate resolved root {normalized_root!r}"
            )
        normalized[normalized_root] = {
            "excluded_episode_indices": frozenset(indices),
            "excluded_frames": excluded_frames,
        }

    declared_episodes = value.get("total_excluded_episodes")
    observed_episodes = sum(
        len(entry["excluded_episode_indices"]) for entry in normalized.values()
    )
    if declared_episodes is not None and int(declared_episodes) != observed_episodes:
        raise ValueError(
            f"Episode exclusion manifest excluded-episode total mismatch at "
            f"{manifest_path}: declared={declared_episodes}, observed={observed_episodes}"
        )
    declared_frames = value.get("total_excluded_frames")
    root_frames = [entry["excluded_frames"] for entry in normalized.values()]
    if declared_frames is not None and all(item is not None for item in root_frames):
        observed_frames = sum(int(item) for item in root_frames)
        if int(declared_frames) != observed_frames:
            raise ValueError(
                f"Episode exclusion manifest excluded-frame total mismatch at "
                f"{manifest_path}: declared={declared_frames}, observed={observed_frames}"
            )

    return manifest_path, normalized


def validate_manifest_roots(
    manifest_path: Optional[Path],
    manifest_roots: dict[str, dict[str, Any]],
    configured_roots: set[str],
    *,
    configured_label: str = "dataset_dirs",
) -> None:
    """Reject denylist roots that are not part of the configured corpus."""

    unknown_roots = sorted(set(manifest_roots) - configured_roots)
    if unknown_roots:
        raise ValueError(
            f"Episode exclusion manifest {manifest_path} contains roots absent from "
            f"{configured_label}: {unknown_roots[:10]}"
        )
