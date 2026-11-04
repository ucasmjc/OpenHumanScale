# Modified by Open-HumanScale contributors: removed unused release runtime code.
"""Normalize per-episode pinhole intrinsics by warping RGB frames.

The action/state tensors deliberately do not pass through this module.  For a
source calibration ``K_s`` and virtual calibration ``K_v`` on the same camera
axes, the inverse sampling map is::

    p_s ~ K_s K_v^-1 p_v

Dense sidecars are loaded once at dataset construction.  The hot path performs
one table lookup and reuses a sampling grid for every repeated calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .eef_anchor_overlay import _cover_resize_center_crop_table, _load_dense_camera_intrinsics, camera_intrinsics_from_record


def _plain_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")


class CameraIntrinsicsWarp:
    """Warp a single-camera video to a fixed virtual pinhole calibration."""

    def __init__(
        self,
        *,
        dataset_dirs: Sequence[str],
        selected_episode_indices: Sequence[Iterable[int]],
        video_size: Sequence[int],
        source_intrinsics: Mapping[str, Any],
        target_intrinsics: Mapping[str, Any],
        interpolation: str = "bilinear",
        fill_value: float = -1.0,
        resize_mode: str = "cover_crop",
        source_routes: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if len(video_size) != 2:
            raise ValueError(f"video_size must be [height,width], got {video_size}")
        self.output_height, self.output_width = map(int, video_size)
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError(f"video_size must be positive, got {video_size}")

        interpolation = str(interpolation).strip().lower()
        if interpolation not in {"bilinear", "nearest"}:
            raise ValueError(
                "camera intrinsic warp interpolation must be bilinear or nearest, "
                f"got {interpolation!r}"
            )
        self.interpolation = interpolation
        self.fill_value = float(fill_value)
        if not torch.isfinite(torch.tensor(self.fill_value)):
            raise ValueError(f"fill_value must be finite, got {fill_value}")

        target = camera_intrinsics_from_record(target_intrinsics)
        if (target.height, target.width) != (
            self.output_height,
            self.output_width,
        ):
            raise ValueError(
                "target_intrinsics must already be expressed on the model-input "
                f"raster {self.output_width}x{self.output_height}, got "
                f"{target.width}x{target.height}"
            )
        self.target_intrinsics = target
        self._target = target.as_tensor()

        roots = [Path(path).expanduser().resolve() for path in dataset_dirs]
        if resize_mode not in {"cover_crop", "direct_resize"}:
            raise ValueError(f"Unknown camera intrinsics resize_mode={resize_mode!r}")
        self.resize_mode = resize_mode
        enabled = []
        for path in dataset_dirs:
            if source_routes is None:
                enabled.append(True)
                continue
            matches = [route for route in source_routes
                       if str(route.get("match", "")) and str(route["match"]) in str(path)]
            if len(matches) != 1 or not isinstance(matches[0].get("warp"), bool):
                raise ValueError(f"Expected exactly one explicit boolean warp route for {path}")
            enabled.append(matches[0]["warp"])
        self._enabled = tuple(enabled)
        if len(roots) != len(selected_episode_indices):
            raise ValueError(
                "selected_episode_indices must have one entry per dataset root, got "
                f"{len(selected_episode_indices)} for {len(roots)} roots"
            )
        spec = _plain_mapping(source_intrinsics, context="source_intrinsics")
        mode = str(spec.get("mode", "")).strip().lower()
        self._episode_intrinsics: tuple[torch.Tensor | None, ...]
        self._fixed_intrinsics: tuple[torch.Tensor | None, ...]
        if mode == "fixed":
            calibration = camera_intrinsics_from_record(spec)
            if resize_mode == "direct_resize":
                fixed = calibration.as_tensor() * torch.tensor([
                    self.output_width / calibration.width,
                    self.output_height / calibration.height,
                    self.output_width / calibration.width,
                    self.output_height / calibration.height,
                ])
            else:
                fixed = calibration.cover_resize_center_crop(
                    self.output_height, self.output_width
                ).as_tensor()
            self._fixed_intrinsics = tuple(fixed for _ in roots)
            self._episode_intrinsics = tuple(None for _ in roots)
        elif mode == "sidecar":
            relative_path = str(spec.get("relative_path", "meta/camera_intrinsics.npy"))
            episode_tables: list[torch.Tensor | None] = []
            for root, selected, warp_enabled in zip(roots, selected_episode_indices, self._enabled, strict=True):
                if not warp_enabled:
                    episode_tables.append(None)
                    continue
                sidecar = root / relative_path
                raw_table = _load_dense_camera_intrinsics(sidecar)
                if resize_mode == "direct_resize":
                    table = raw_table[:, :4].clone()
                    table[:, [0, 2]] *= self.output_width / raw_table[:, 4, None]
                    table[:, [1, 3]] *= self.output_height / raw_table[:, 5, None]
                    valid = torch.isfinite(raw_table).all(dim=1)
                    valid &= (raw_table[:, [0, 1, 4, 5]] > 0).all(dim=1)
                    table[~valid] = torch.nan
                else:
                    table = _cover_resize_center_crop_table(
                        raw_table, self.output_height, self.output_width
                    )
                selected_tensor = torch.as_tensor(list(selected), dtype=torch.long)
                out_of_range = selected_tensor[
                    (selected_tensor < 0) | (selected_tensor >= table.shape[0])
                ]
                in_range = selected_tensor[
                    (selected_tensor >= 0) & (selected_tensor < table.shape[0])
                ]
                invalid = in_range[~torch.isfinite(table[in_range]).all(dim=1)]
                if out_of_range.numel() or invalid.numel():
                    preview = torch.cat([out_of_range, invalid])[:10].tolist()
                    raise ValueError(
                        f"Camera sidecar {sidecar} is missing "
                        f"{out_of_range.numel() + invalid.numel()} selected episodes; "
                        f"first={preview}"
                    )
                episode_tables.append(table)
            self._fixed_intrinsics = tuple(None for _ in roots)
            self._episode_intrinsics = tuple(episode_tables)
        else:
            raise ValueError(
                "source_intrinsics.mode must be 'fixed' or 'sidecar', "
                f"got {mode!r}"
            )

        # Per-worker cache.  Current MECKA has many episodes but only a small set
        # of repeated calibrations, so caching by exact float32 K avoids allocating
        # a full HxW grid per episode.
        self._grid_cache: dict[tuple[float, float, float, float], torch.Tensor] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_grid_cache"] = {}
        return state

    def intrinsics_for(self, dataset_index: int, episode_index: int) -> torch.Tensor:
        dataset_index = int(dataset_index)
        if not 0 <= dataset_index < len(self._fixed_intrinsics):
            raise IndexError(f"dataset_index={dataset_index} is out of range")
        if not self._enabled[dataset_index]:
            raise ValueError("Passthrough sources have no warp calibration")
        fixed = self._fixed_intrinsics[dataset_index]
        if fixed is not None:
            return fixed
        table = self._episode_intrinsics[dataset_index]
        assert table is not None
        episode_index = int(episode_index)
        if episode_index < 0 or episode_index >= table.shape[0]:
            raise KeyError(
                f"No camera calibration for dataset={dataset_index}, "
                f"episode={episode_index}"
            )
        result = table[episode_index]
        if not bool(torch.isfinite(result).all()):
            raise KeyError(
                f"Invalid camera calibration for dataset={dataset_index}, "
                f"episode={episode_index}"
            )
        return result

    def _grid(self, source: torch.Tensor) -> torch.Tensor:
        source = source.to(dtype=torch.float32, device="cpu").reshape(4)
        key = tuple(float(value) for value in source)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        fx_s, fy_s, cx_s, cy_s = source
        fx_v, fy_v, cx_v, cy_v = self._target
        output_x = torch.arange(self.output_width, dtype=torch.float32)
        output_y = torch.arange(self.output_height, dtype=torch.float32)
        source_x = (fx_s / fx_v) * (output_x - cx_v) + cx_s
        source_y = (fy_s / fy_v) * (output_y - cy_v) + cy_s
        grid_y, grid_x = torch.meshgrid(source_y, source_x, indexing="ij")
        # align_corners=False maps pixel center x to 2*(x+0.5)/W - 1.
        grid = torch.stack(
            [
                2.0 * (grid_x + 0.5) / self.output_width - 1.0,
                2.0 * (grid_y + 0.5) / self.output_height - 1.0,
            ],
            dim=-1,
        ).unsqueeze(0).contiguous()
        self._grid_cache[key] = grid
        return grid

    def apply(
        self,
        video: torch.Tensor,
        *,
        dataset_index: int,
        episode_index: int,
    ) -> torch.Tensor:
        """Return ``video`` warped to ``target_intrinsics``; action is untouched."""

        if video.ndim != 4:
            raise ValueError(
                "camera intrinsic warp currently supports one camera with video "
                f"[C,T,H,W], got {tuple(video.shape)}"
            )
        if video.shape[0] != 3:
            raise ValueError(f"camera intrinsic warp expects RGB, got {video.shape[0]} channels")
        if tuple(video.shape[-2:]) != (self.output_height, self.output_width):
            raise ValueError(
                "camera intrinsic warp raster mismatch: expected "
                f"{self.output_height}x{self.output_width}, got {tuple(video.shape[-2:])}"
            )
        if not video.is_floating_point():
            raise TypeError(f"camera intrinsic warp expects floating video, got {video.dtype}")

        if not 0 <= int(dataset_index) < len(self._enabled):
            raise IndexError(f"dataset_index={dataset_index} is out of range")
        if not self._enabled[int(dataset_index)]:
            return video
        source = self.intrinsics_for(dataset_index, episode_index)
        grid = self._grid(source).to(device=video.device)
        frames = video.permute(1, 0, 2, 3).contiguous()
        output_dtype = frames.dtype
        if frames.dtype not in {torch.float32, torch.float64}:
            frames = frames.float()
        grid = grid.to(dtype=frames.dtype).expand(frames.shape[0], -1, -1, -1)
        # Shift by the desired fill so grid_sample's zero padding becomes the
        # correct value (-1 is black for the current normalized video contract).
        warped = F.grid_sample(
            frames - self.fill_value,
            grid,
            mode=self.interpolation,
            padding_mode="zeros",
            align_corners=False,
        ).add_(self.fill_value)
        if warped.dtype != output_dtype:
            warped = warped.to(output_dtype)
        return warped.permute(1, 0, 2, 3).contiguous()
