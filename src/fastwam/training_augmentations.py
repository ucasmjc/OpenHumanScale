"""Batch-level pixel augmentations used by the training loop.

These transforms run after the DataLoader has stacked a batch.  That placement
is intentional: every sample and frame in one microbatch must have the same
spatial shape, while the shape may vary between microbatches.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BatchSharedCenterCrop:
    """Stateless, reproducible center crop shared by a whole video batch."""

    min_long_edge: int
    max_long_edge: int
    long_edge_step: int
    spatial_multiple: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        min_edge = int(self.min_long_edge)
        max_edge = int(self.max_long_edge)
        step = int(self.long_edge_step)
        multiple = int(self.spatial_multiple)
        if min_edge <= 0 or max_edge <= 0:
            raise ValueError(
                "batch-shared center-crop edges must be positive, got "
                f"min={min_edge} max={max_edge}"
            )
        if min_edge > max_edge:
            raise ValueError(
                "batch-shared center-crop min_long_edge must be <= max_long_edge, "
                f"got {min_edge} > {max_edge}"
            )
        if step <= 0:
            raise ValueError(
                f"batch-shared center-crop long_edge_step must be positive, got {step}"
            )
        if multiple <= 0:
            raise ValueError(
                f"batch-shared center-crop spatial_multiple must be positive, got {multiple}"
            )
        invalid = [edge for edge in self.allowed_long_edges if edge % multiple != 0]
        if invalid:
            raise ValueError(
                "Every batch-shared center-crop width must be divisible by the "
                f"model spatial multiple {multiple}; invalid={invalid}"
            )

    @property
    def allowed_long_edges(self) -> tuple[int, ...]:
        return tuple(
            range(
                int(self.min_long_edge),
                int(self.max_long_edge) + 1,
                int(self.long_edge_step),
            )
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        default_seed: int,
    ) -> "BatchSharedCenterCrop":
        return cls(
            min_long_edge=int(config.get("min_long_edge", 320)),
            max_long_edge=int(config.get("max_long_edge", 448)),
            long_edge_step=int(config.get("long_edge_step", 32)),
            spatial_multiple=int(config.get("spatial_multiple", 32)),
            seed=int(config.get("seed", default_seed)),
        )

    def long_edge_for_batch(self, *, epoch: int, batch_in_epoch: int) -> int:
        """Choose one width without consuming any process-local RNG state.

        The stateless hash makes exact-state resume reproducible and gives every
        distributed rank the same width as long as their epoch/batch counters
        agree, which is already a training-loop invariant.
        """

        payload = (
            f"fastwam-batch-shared-center-crop-v1:{int(self.seed)}:"
            f"{int(epoch)}:{int(batch_in_epoch)}"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        index = int.from_bytes(digest[:8], byteorder="little", signed=False)
        edges = self.allowed_long_edges
        return edges[index % len(edges)]

    def apply(self, video: torch.Tensor, *, long_edge: int) -> torch.Tensor:
        """Center-crop ``[B,C,T,H,W]`` along its landscape long edge."""

        if video.ndim != 5:
            raise ValueError(
                "batch-shared center crop expects video [B,C,T,H,W], got "
                f"{tuple(video.shape)}"
            )
        height, width = (int(video.shape[-2]), int(video.shape[-1]))
        if width < height:
            raise ValueError(
                "batch-shared long-edge crop currently requires landscape video, "
                f"got HxW={height}x{width}"
            )
        long_edge = int(long_edge)
        if long_edge not in self.allowed_long_edges:
            raise ValueError(
                f"Requested crop width {long_edge} is not in {self.allowed_long_edges}"
            )
        if long_edge > width:
            raise ValueError(
                f"Requested crop width {long_edge} exceeds input width {width}"
            )
        left = (width - long_edge) // 2
        cropped = video[..., left : left + long_edge]
        if tuple(cropped.shape[-2:]) != (height, long_edge):
            raise RuntimeError(
                "batch-shared center crop produced an unexpected shape: "
                f"input={tuple(video.shape)} output={tuple(cropped.shape)}"
            )
        return cropped.contiguous()
