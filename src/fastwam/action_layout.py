"""Explicit action components and validity for heterogeneous supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import torch


EEF_CAMERA22_COMPONENTS = {
    "position": (0, 1, 2, 8, 9, 10),
    "rotation": (3, 4, 5, 6, 11, 12, 13, 14),
    "gripper": (7, 15),
    "camera_translation": (16, 17, 18),
    "camera_rotation": (19, 20, 21),
}


def resolve_action_components(
    action_dim: int,
    num_arms: int = 2,
    component_indices: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Validate a complete, disjoint partition, or recover the legacy arm layout."""
    if component_indices is None:
        if num_arms <= 0 or action_dim % num_arms:
            raise ValueError(
                f"action_dim={action_dim} must be divisible by num_arms={num_arms}"
            )
        arm_dim = action_dim // num_arms
        if arm_dim not in (7, 8, 10):
            raise ValueError(
                "Implicit EEF components require 7, 8, or 10 channels per arm "
                "(xyz3 + axis-angle/euler3, quaternion4, or rotation6d6 + gripper1); "
                f"action_dim={action_dim} requires explicit action_component_indices"
            )
        return {
            "position": tuple(a * arm_dim + i for a in range(num_arms) for i in range(3)),
            "rotation": tuple(a * arm_dim + i for a in range(num_arms) for i in range(3, arm_dim - 1)),
            "gripper": tuple((a + 1) * arm_dim - 1 for a in range(num_arms)),
        }
    result = {}
    seen = set()
    for name, indices in component_indices.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"Invalid action component name: {name!r}")
        values = tuple(indices)
        if not values or any(isinstance(i, bool) or not isinstance(i, Integral) for i in values):
            raise ValueError(f"Action component {name!r} must have nonempty integer indices")
        for index in values:
            if index < 0 or index >= action_dim or index in seen:
                raise ValueError(f"Invalid or overlapping action component index {index} in {name!r}")
            seen.add(index)
        result[name] = tuple(int(i) for i in values)
    if seen != set(range(action_dim)):
        raise ValueError(f"Action components must cover all {action_dim} dimensions; missing {sorted(set(range(action_dim)) - seen)}")
    return result


def action_valid_mask(
    action: torch.Tensor,
    action_dim_valid: torch.Tensor | None = None,
    action_is_pad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [B,T,D] validity; accept static [B,D] and per-step [B,T,D] masks."""
    if action.ndim != 3:
        raise ValueError(f"Action must be [B,T,D], got {tuple(action.shape)}")
    batch, steps, dims = action.shape
    valid = torch.ones_like(action, dtype=torch.bool)
    if action_dim_valid is not None:
        dim_valid = torch.as_tensor(action_dim_valid, device=action.device, dtype=torch.bool)
        if tuple(dim_valid.shape) == (batch, dims):
            dim_valid = dim_valid[:, None, :]
        elif tuple(dim_valid.shape) != (batch, steps, dims):
            raise ValueError(f"action_dim_valid must be [B,D] or [B,T,D], got {tuple(dim_valid.shape)}")
        valid = valid & dim_valid
    if action_is_pad is not None:
        padding = torch.as_tensor(action_is_pad, device=action.device, dtype=torch.bool)
        if tuple(padding.shape) != (batch, steps):
            raise ValueError(f"action_is_pad must be [B,T], got {tuple(padding.shape)}")
        valid = valid & ~padding[..., None]
    return valid


def masked_action_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over valid dimensions, then valid time steps, independently per sample."""
    token_valid = valid.any(dim=2)
    token_mean = torch.where(valid, values, 0.0).sum(dim=2) / valid.sum(dim=2).clamp_min(1)
    return token_mean.sum(dim=1) / token_valid.sum(dim=1).clamp_min(1)
