"""Action supervision follows logical annotations, never dataset path names."""
import torch


BAD_SEGMENT_REASONS = frozenset({"action_idle", "body_locomotion", "action_instability"})


def bad_segment_action_mask(action, reason, existing_mask=None):
    if not isinstance(reason, str) or reason not in BAD_SEGMENT_REASONS | {""}:
        raise ValueError(f"Unknown logical bad_reason: {reason!r}")
    if action.ndim != 2:
        raise ValueError("Expected per-sample action [T,C]")
    mask = torch.ones(action.shape[-1], dtype=torch.bool, device=action.device)
    if existing_mask is not None:
        if tuple(existing_mask.shape) != tuple(mask.shape):
            raise ValueError("Existing action dimension mask does not match action")
        mask &= existing_mask.to(device=action.device, dtype=torch.bool)
    if reason:
        mask.zero_()
    return mask
