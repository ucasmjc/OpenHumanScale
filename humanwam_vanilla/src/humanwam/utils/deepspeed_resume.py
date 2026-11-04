"""Synchronize DeepSpeed master parameters after a weights-only model load."""
import torch


@torch.no_grad()
def refresh_deepspeed_master_params(engine):
    # Accelerate's optimizer wrapper is not the owner of the master tensors.
    optimizer = engine.optimizer
    refresh = getattr(optimizer, "refresh_fp32_params", None)
    if not callable(refresh):
        raise RuntimeError(
            "DeepSpeed weights-only load requires refresh_fp32_params(); "
            f"unsupported optimizer {type(optimizer).__name__}. "
            "Refusing to train with potentially stale master parameters."
        )
    refresh()
    checked = 0
    # BF16_Optimizer (the production ZeRO1 path) exposes the local low/high
    # precision partitions. Check ALL elements in bounded chunks after refresh.
    masters = getattr(optimizer, "fp32_groups_flat_partition", None)
    partitions = getattr(optimizer, "bf16_partitioned_groups", None)
    if masters is not None and partitions is not None:
        if len(masters) != len(partitions):
            raise RuntimeError("DeepSpeed master/parameter group count mismatch")
        for i, (master, group) in enumerate(zip(masters, partitions)):
            rank = torch.distributed.get_rank(optimizer.real_dp_process_group[i])
            low = group[rank].view(-1)
            high = master.view(-1)
            if low.numel() != high.numel():
                raise RuntimeError("DeepSpeed master/parameter partition size mismatch")
            for start in range(0, low.numel(), 1 << 20):
                expected = low[start:start + (1 << 20)].to(high.dtype)
                if not torch.equal(high[start:start + (1 << 20)], expected):
                    raise RuntimeError("DeepSpeed FP32 refresh did not match loaded model")
            checked += low.numel()
    return {"optimizer": type(optimizer).__name__, "verified_local_elements": checked}
