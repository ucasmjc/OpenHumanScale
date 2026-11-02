import logging
import json
import inspect
import os
import random
import re
from collections.abc import Iterable, Mapping
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import send_to_device
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .checkpoint_storage import (
    AsyncCheckpointMirror,
    OssCheckpointStore,
    prune_training_states,
    prune_uploaded_weights,
)
from .models.bottleneck_adapter import (
    inject_bottleneck_adapters,
    is_bottleneck_adapter_checkpoint,
    load_bottleneck_adapter,
    load_bottleneck_adapter_payload,
    resolve_bottleneck_base_checkpoint,
    save_bottleneck_adapter,
    set_bottleneck_adapter_trainable,
    validate_bottleneck_training_contract,
)
from .models.lora import (
    DEFAULT_MOT_LORA_TARGETS,
    inject_lora_adapters,
    is_lora_adapter_checkpoint,
    load_lora_adapter,
    load_lora_adapter_payload,
    resolve_lora_base_checkpoint,
    save_lora_adapter,
    set_lora_trainable,
    validate_lora_training_contract,
)
from .training_augmentations import BatchSharedCenterCrop
from .utils.fs import ensure_dir
from .utils.deepspeed_resume import refresh_deepspeed_master_params
from .utils.logging_config import get_logger, setup_logging
from .utils.lr_schedule import FloorCosineLR
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import build_resumable_sampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


_MANUAL_FSDP_STATE_FORMAT = "fastwam_manual_fsdp_rank_shards_v1"


def _parse_bool_config(value, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _parse_auto_bool_config(value, *, name: str) -> bool | None:
    if value is None or str(value).strip().lower() in {"", "auto", "null", "none"}:
        return None
    return _parse_bool_config(value, name=name)


def _validate_checkpoint_steps(raw_steps, *, max_steps: int) -> tuple[int, ...]:
    """Normalize a fail-closed sparse checkpoint schedule.

    Step zero is intentionally not accepted here: it has different ordering
    semantics and is controlled only by ``save_initial_checkpoint``.
    """

    if raw_steps is None:
        return ()
    if isinstance(raw_steps, (str, bytes, Mapping)) or not isinstance(
        raw_steps, Iterable
    ):
        raise ValueError(
            "checkpoint_steps must be a list of strictly increasing integers"
        )

    steps: list[int] = []
    for index, value in enumerate(raw_steps):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "checkpoint_steps must contain integers; "
                f"item {index} is {value!r}"
            )
        steps.append(int(value))

    if any(step <= 0 for step in steps):
        raise ValueError(
            "checkpoint_steps must contain only positive steps; use "
            "save_initial_checkpoint for step 0"
        )
    if any(left >= right for left, right in zip(steps, steps[1:])):
        raise ValueError(
            "checkpoint_steps must be unique and strictly increasing, got "
            f"{steps}"
        )
    if any(step > int(max_steps) for step in steps):
        raise ValueError(
            f"checkpoint_steps must be <= max_steps={int(max_steps)}, got {steps}"
        )
    return tuple(steps)


def _checkpoint_step_is_scheduled(
    step: int, *, save_every: int, checkpoint_steps: Iterable[int]
) -> bool:
    """Return whether one post-optimizer checkpoint is due at ``step``."""

    step = int(step)
    return (
        int(save_every) > 0 and step > 0 and step % int(save_every) == 0
    ) or step in checkpoint_steps


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.dataloader_timeout = float(cfg.get("dataloader_timeout", 0) or 0)
        self.dataloader_prefetch_factor = int(cfg.get("dataloader_prefetch_factor", 2) or 2)
        self.dataloader_persistent_workers = _parse_auto_bool_config(
            cfg.get("dataloader_persistent_workers", "auto"),
            name="dataloader_persistent_workers",
        )
        self.dataloader_in_order = _parse_bool_config(
            cfg.get("dataloader_in_order", True),
            name="dataloader_in_order",
        )
        self.dataloader_non_blocking = _parse_bool_config(
            cfg.get("dataloader_non_blocking", True),
            name="dataloader_non_blocking",
        )
        self.dataloader_rank_aware_strict = _parse_bool_config(
            cfg.get("dataloader_rank_aware_strict", False),
            name="dataloader_rank_aware_strict",
        )
        self.dataloader_require_merged_cache = _parse_bool_config(
            cfg.get("dataloader_require_merged_cache", False),
            name="dataloader_require_merged_cache",
        )
        self.dataloader_fail_on_cache_miss = _parse_bool_config(
            cfg.get("dataloader_fail_on_cache_miss", False),
            name="dataloader_fail_on_cache_miss",
        )
        self.dataloader_require_video_latent_cache = _parse_bool_config(
            cfg.get("dataloader_require_video_latent_cache", False),
            name="dataloader_require_video_latent_cache",
        )
        self.dataloader_fail_on_latent_cache_miss = _parse_bool_config(
            cfg.get("dataloader_fail_on_latent_cache_miss", False),
            name="dataloader_fail_on_latent_cache_miss",
        )
        self.dataloader_pin_memory = torch.cuda.is_available()
        if self.dataloader_timeout < 0:
            raise ValueError(f"dataloader_timeout must be >= 0, got {self.dataloader_timeout}")
        if self.dataloader_prefetch_factor <= 0:
            raise ValueError(
                f"dataloader_prefetch_factor must be > 0, got {self.dataloader_prefetch_factor}"
            )
        if self.num_workers == 0 and self.dataloader_timeout != 0:
            raise ValueError("dataloader_timeout must be 0 when num_workers=0")
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_initial_checkpoint = _parse_bool_config(
            cfg.get("save_initial_checkpoint", False),
            name="save_initial_checkpoint",
        )
        raw_checkpoint_steps = cfg.get("checkpoint_steps", []) or []
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)

        crop_cfg = cfg.get("batch_shared_center_crop", {}) or {}
        self.batch_shared_center_crop_enabled = _parse_bool_config(
            crop_cfg.get("enabled", False),
            name="batch_shared_center_crop.enabled",
        )
        self.batch_shared_center_crop_verify_across_ranks = _parse_bool_config(
            crop_cfg.get("verify_across_ranks", True),
            name="batch_shared_center_crop.verify_across_ranks",
        )
        self.batch_shared_center_crop = None
        if self.batch_shared_center_crop_enabled:
            self.batch_shared_center_crop = BatchSharedCenterCrop.from_config(
                crop_cfg,
                default_seed=self.seed,
            )
            if getattr(self.train_dataset, "_video_latent_cache", None) is not None:
                raise ValueError(
                    "batch_shared_center_crop requires decoded pixels and is "
                    "incompatible with a video latent cache"
                )
            logger.info(
                "Batch-shared center crop enabled: widths=%s seed=%d "
                "verify_across_ranks=%s",
                list(self.batch_shared_center_crop.allowed_long_edges),
                self.batch_shared_center_crop.seed,
                self.batch_shared_center_crop_verify_across_ranks,
            )

        self.resume = cfg.resume
        self._exact_state_resume = bool(
            self.resume and Path(str(self.resume)).is_dir()
        )
        # Weights-only resume that inherits the LR (see configs/train.yaml `resume_step`).
        self.resume_step = int(cfg.get("resume_step", 0) or 0)
        self.resume_mot_skip_prefixes = tuple(
            str(prefix) for prefix in (cfg.get("resume_mot_skip_prefixes", []) or [])
        )
        self.resume_skip_proprio_encoder = _parse_bool_config(
            cfg.get("resume_skip_proprio_encoder", False),
            name="resume_skip_proprio_encoder",
        )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        # LoRA must be injected before freezing, optimizer construction, and
        # FSDP wrapping.  LoRALinear preserves the historical ``weight``/``bias``
        # state-dict keys, so a normal full checkpoint remains a valid
        # weights-only initialization (the new B=0 adapters are simply missing).
        lora_cfg = cfg.get("lora", {}) or {}
        self.lora_enabled = _parse_bool_config(
            lora_cfg.get("enabled", False),
            name="lora.enabled",
        )
        self.lora_rank = int(lora_cfg.get("rank", 16))
        self.lora_alpha = float(lora_cfg.get("alpha", 16.0))
        self.lora_dropout = float(lora_cfg.get("dropout", 0.0))
        self.lora_experts = tuple(
            str(name) for name in (lora_cfg.get("experts", ["video", "action"]) or [])
        )
        self.lora_target_modules = tuple(
            str(name)
            for name in (
                lora_cfg.get("target_modules", list(DEFAULT_MOT_LORA_TARGETS)) or []
            )
        )
        self.lora_trainable_modules = tuple(
            str(name) for name in (lora_cfg.get("trainable_modules", []) or [])
        )
        self.lora_checkpoint_mode = str(
            lora_cfg.get("checkpoint_mode", "adapter")
        ).strip().lower()
        if self.lora_checkpoint_mode not in {"adapter", "full"}:
            raise ValueError(
                "lora.checkpoint_mode must be 'adapter' or 'full', got "
                f"{self.lora_checkpoint_mode!r}."
            )
        self.lora_resume_adapter = str(
            lora_cfg.get("resume_adapter", "") or ""
        ).strip() or None
        self.lora_base_checkpoint = str(
            lora_cfg.get("base_checkpoint", "") or ""
        ).strip() or None
        self.lora_base_checkpoint_sha256 = str(
            lora_cfg.get("base_checkpoint_sha256", "") or ""
        ).strip().lower() or None
        if self.lora_base_checkpoint_sha256 is not None and (
            len(self.lora_base_checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.lora_base_checkpoint_sha256)
        ):
            raise ValueError(
                "lora.base_checkpoint_sha256 must be a 64-character lowercase hex digest."
            )
        self.lora_verify_base_sha256 = _parse_bool_config(
            lora_cfg.get("verify_base_sha256", True),
            name="lora.verify_base_sha256",
        )

        # Bottleneck adapters are a separate PEFT method and artifact format.
        # Keep them parallel to (rather than hidden under) the historical LoRA
        # `.adapter.pt` contract so resume dispatch cannot confuse the two.
        bottleneck_cfg = cfg.get("bottleneck_adapter", {}) or {}
        self.bottleneck_adapter_enabled = _parse_bool_config(
            bottleneck_cfg.get("enabled", False),
            name="bottleneck_adapter.enabled",
        )
        self.bottleneck_adapter_experts = tuple(
            str(name)
            for name in (
                bottleneck_cfg.get("experts", ["video", "action"]) or []
            )
        )
        self.bottleneck_adapter_sites = tuple(
            str(name)
            for name in (
                bottleneck_cfg.get("sites", ["self_attn", "ffn"]) or []
            )
        )
        raw_bottleneck_dims = bottleneck_cfg.get(
            "bottleneck_dims", {"video": 77, "action": 103}
        )
        if isinstance(raw_bottleneck_dims, Mapping):
            self.bottleneck_adapter_dims = {
                str(name): int(value)
                for name, value in raw_bottleneck_dims.items()
            }
        else:
            shared_dim = int(raw_bottleneck_dims)
            self.bottleneck_adapter_dims = {
                expert: shared_dim for expert in self.bottleneck_adapter_experts
            }
        self.bottleneck_adapter_activation = str(
            bottleneck_cfg.get("activation", "gelu_tanh")
        ).strip().lower()
        self.bottleneck_adapter_dropout = float(
            bottleneck_cfg.get("dropout", 0.0)
        )
        self.bottleneck_adapter_residual_scale = float(
            bottleneck_cfg.get("residual_scale", 1.0)
        )
        self.bottleneck_adapter_init_seed = int(
            bottleneck_cfg.get("init_seed", 0)
        )
        self.bottleneck_adapter_trainable_modules = tuple(
            str(name)
            for name in (bottleneck_cfg.get("trainable_modules", []) or [])
        )
        self.bottleneck_adapter_checkpoint_mode = str(
            bottleneck_cfg.get("checkpoint_mode", "delta")
        ).strip().lower()
        if self.bottleneck_adapter_checkpoint_mode not in {"delta", "full"}:
            raise ValueError(
                "bottleneck_adapter.checkpoint_mode must be 'delta' or 'full', got "
                f"{self.bottleneck_adapter_checkpoint_mode!r}."
            )
        self.bottleneck_adapter_resume_delta = str(
            bottleneck_cfg.get("resume_delta", "") or ""
        ).strip() or None
        self.bottleneck_adapter_base_checkpoint = str(
            bottleneck_cfg.get("base_checkpoint", "") or ""
        ).strip() or None
        self.bottleneck_adapter_base_checkpoint_sha256 = str(
            bottleneck_cfg.get("base_checkpoint_sha256", "") or ""
        ).strip().lower() or None
        if self.bottleneck_adapter_base_checkpoint_sha256 is not None and (
            len(self.bottleneck_adapter_base_checkpoint_sha256) != 64
            or any(
                c not in "0123456789abcdef"
                for c in self.bottleneck_adapter_base_checkpoint_sha256
            )
        ):
            raise ValueError(
                "bottleneck_adapter.base_checkpoint_sha256 must be a "
                "64-character lowercase hex digest."
            )
        self.bottleneck_adapter_verify_base_sha256 = _parse_bool_config(
            bottleneck_cfg.get("verify_base_sha256", True),
            name="bottleneck_adapter.verify_base_sha256",
        )
        if self.lora_enabled and self.bottleneck_adapter_enabled:
            raise ValueError(
                "LoRA and bottleneck_adapter are mutually exclusive PEFT methods."
            )
        if self.lora_resume_adapter is not None and not self.lora_enabled:
            raise ValueError("lora.resume_adapter requires lora.enabled=true.")
        if (
            self.bottleneck_adapter_resume_delta is not None
            and not self.bottleneck_adapter_enabled
        ):
            raise ValueError(
                "bottleneck_adapter.resume_delta requires "
                "bottleneck_adapter.enabled=true."
            )
        if (
            self.resume
            and is_bottleneck_adapter_checkpoint(self.resume)
            and not self.bottleneck_adapter_enabled
        ):
            raise ValueError(
                "Resuming a `.bottleneck.pt` delta requires "
                "bottleneck_adapter.enabled=true."
            )
        if (
            self.lora_resume_adapter is not None
            and self.resume
            and Path(str(self.resume)).is_dir()
        ):
            raise ValueError(
                "An exact state-directory resume cannot be combined with "
                "lora.resume_adapter. Resume the state directory by itself."
            )
        if (
            self.bottleneck_adapter_resume_delta is not None
            and self.resume
            and Path(str(self.resume)).is_dir()
        ):
            raise ValueError(
                "An exact state-directory resume cannot be combined with "
                "bottleneck_adapter.resume_delta. Resume the state directory by itself."
            )
        if (
            self.lora_enabled
            and self.resume
            and is_bottleneck_adapter_checkpoint(self.resume)
        ):
            raise ValueError(
                "A `.bottleneck.pt` delta cannot be resumed with lora.enabled=true."
            )
        if (
            self.bottleneck_adapter_enabled
            and self.resume
            and is_lora_adapter_checkpoint(self.resume)
        ):
            raise ValueError(
                "A LoRA `.adapter.pt` checkpoint cannot be resumed with "
                "bottleneck_adapter.enabled=true."
            )
        if (
            self.lora_enabled
            and self.resume
            and not Path(str(self.resume)).is_dir()
            and is_lora_adapter_checkpoint(self.resume)
        ):
            if self.lora_resume_adapter is not None:
                raise ValueError(
                    "Specify an adapter either as `resume` or `lora.resume_adapter`, not both."
                )
            self.lora_resume_adapter = str(self.resume)
        elif (
            self.lora_enabled
            and self.resume
            and not Path(str(self.resume)).is_dir()
            and self.lora_base_checkpoint is None
        ):
            # The formal post-training contract uses cfg.resume for the frozen
            # head-only base and starts the local LoRA schedule at step zero.
            self.lora_base_checkpoint = str(self.resume)
        if (
            self.bottleneck_adapter_enabled
            and self.resume
            and not Path(str(self.resume)).is_dir()
            and is_bottleneck_adapter_checkpoint(self.resume)
        ):
            if self.bottleneck_adapter_resume_delta is not None:
                raise ValueError(
                    "Specify a bottleneck delta either as `resume` or "
                    "`bottleneck_adapter.resume_delta`, not both."
                )
            self.bottleneck_adapter_resume_delta = str(self.resume)
        elif (
            self.bottleneck_adapter_enabled
            and self.resume
            and not Path(str(self.resume)).is_dir()
            and self.bottleneck_adapter_base_checkpoint is None
        ):
            self.bottleneck_adapter_base_checkpoint = str(self.resume)
        if self.lora_base_checkpoint is not None:
            base_path = Path(self.lora_base_checkpoint).expanduser()
            if base_path.exists():
                self.lora_base_checkpoint = str(base_path.resolve())
        if self.bottleneck_adapter_base_checkpoint is not None:
            base_path = Path(self.bottleneck_adapter_base_checkpoint).expanduser()
            if base_path.exists():
                self.bottleneck_adapter_base_checkpoint = str(base_path.resolve())
        self.lora_injection_report = None
        if self.lora_enabled:
            self.lora_injection_report = inject_lora_adapters(
                self.model.dit,
                experts=self.lora_experts,
                target_modules=self.lora_target_modules,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            logger.info(
                "Injected LoRA adapters | experts=%s targets=%s modules=%d "
                "rank=%d alpha=%.4f dropout=%.4f trainable_params=%d "
                "tiny_modules=%s",
                list(self.lora_experts),
                list(self.lora_target_modules),
                self.lora_injection_report.module_count,
                self.lora_rank,
                self.lora_alpha,
                self.lora_dropout,
                self.lora_injection_report.trainable_parameters,
                list(self.lora_trainable_modules),
            )
        self.bottleneck_adapter_injection_report = None
        if self.bottleneck_adapter_enabled:
            self.bottleneck_adapter_injection_report = inject_bottleneck_adapters(
                self.model.dit,
                experts=self.bottleneck_adapter_experts,
                sites=self.bottleneck_adapter_sites,
                bottleneck_dims=self.bottleneck_adapter_dims,
                activation=self.bottleneck_adapter_activation,
                dropout=self.bottleneck_adapter_dropout,
                residual_scale=self.bottleneck_adapter_residual_scale,
                init_seed=self.bottleneck_adapter_init_seed,
            )
            logger.info(
                "Injected bottleneck adapters | experts=%s sites=%s dims=%s "
                "activation=%s dropout=%.4f residual_scale=%.4f init_seed=%d "
                "modules=%d trainable_params=%d tiny_modules=%s",
                list(self.bottleneck_adapter_experts),
                list(self.bottleneck_adapter_sites),
                dict(self.bottleneck_adapter_dims),
                self.bottleneck_adapter_activation,
                self.bottleneck_adapter_dropout,
                self.bottleneck_adapter_residual_scale,
                self.bottleneck_adapter_init_seed,
                self.bottleneck_adapter_injection_report.module_count,
                self.bottleneck_adapter_injection_report.trainable_parameters,
                list(self.bottleneck_adapter_trainable_modules),
            )
        # Keep the resolved contract on the unwrapped model: the static train-mode
        # helper is also called after evaluation and through distributed wrappers.
        self.model.lora_enabled = self.lora_enabled
        self.model.lora_rank = self.lora_rank
        self.model.lora_alpha = self.lora_alpha
        self.model.lora_dropout = self.lora_dropout
        self.model.lora_trainable_modules = self.lora_trainable_modules
        self.model.bottleneck_adapter_enabled = self.bottleneck_adapter_enabled
        self.model.bottleneck_adapter_trainable_modules = (
            self.bottleneck_adapter_trainable_modules
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            dataloader_config=self._accelerator_dataloader_config(),
        )

        _ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        self._is_deepspeed = _ds_plugin is not None
        _zero_stage = (
            _ds_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
            if _ds_plugin is not None else "n/a"
        )
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            _zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional source-routed state encoders) as trainable
        # when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)

        # Manual FSDP: wrap the whole model as a single FSDP unit. FastWAM's
        # MoT does cross-expert mixed attention via direct sub-param access, so
        # per-block auto-wrap is impossible -> wrap the whole model. This also
        # sidesteps DeepSpeed ZeRO-2's O(N^2) per-step backward hook in
        # deepspeed==0.18.5, and FSDP overlaps gradient comm with backward.
        self._manual_fsdp = (
            os.environ.get("FASTWAM_MANUAL_FSDP", "0") == "1"
            and self.accelerator.num_processes > 1
        )
        if self.dataloader_rank_aware_strict and not self._manual_fsdp:
            raise ValueError(
                "dataloader_rank_aware_strict requires multi-process manual FSDP"
            )
        if self._manual_fsdp:
            if (
                os.environ.get("TORCH_COMPILE", "0") == "1"
                and bool(getattr(getattr(self.model, "mot", None), "compile_blocks", False))
            ):
                raise ValueError(
                    "TORCH_COMPILE=1 (whole-model compile) and "
                    "model.mot_compile_blocks=true are mutually exclusive. Use the "
                    "RoboTwin static block compile; whole-loss compile captures the "
                    "stateful VAE/schedulers and has prohibitive startup latency."
                )
            self._maybe_preload_weights_for_fsdp()
            self._wrap_model_with_fsdp()
            # Static compilation AFTER FSDP wrap (documented FSDP+compile order;
            # top-level compile so accelerate.unwrap_model handles it via _orig_mod
            # -- compiling a submodule trips has_compiled_regions). Video shapes are
            # static. Opt-in via TORCH_COMPILE=1.
            if os.environ.get("TORCH_COMPILE", "0") == "1":
                mode = os.environ.get("TORCH_COMPILE_MODE", "default")
                logger.info("torch.compile: compiling FSDP model (mode=%s)", mode)
                self.model = torch.compile(self.model, mode=mode)
        custom_bridge_optimizer = any(
            (
                float(cfg.get("optimizer_video_lr_scale", 1.0)) != 1.0,
                float(cfg.get("optimizer_action_lr_scale", 1.0)) != 1.0,
                float(cfg.get("optimizer_view_lr_scale", 1.0)) != 1.0,
                int(cfg.get("optimizer_video_freeze_steps", 0)) > 0,
            )
        )
        if self._manual_fsdp and custom_bridge_optimizer:
            raise ValueError(
                "Bridge optimizer parameter groups require non-manual FSDP/ZeRO1; "
                "the whole-model manual-FSDP wrapper flattens video/action "
                "parameters before optimizer construction. Launch this experiment "
                "with FASTWAM_MANUAL_FSDP=0."
            )
        optimizer_param_groups = self._build_optimizer_param_groups(
            cfg=cfg,
            model=self.model,
        )
        optimizer_fused = _parse_bool_config(
            os.environ.get(
                "FASTWAM_ADAMW_FUSED",
                cfg.get("optimizer_fused", False),
            ),
            name="optimizer_fused/FASTWAM_ADAMW_FUSED",
        )
        optimizer_kwargs = {"fused": True} if optimizer_fused else {}
        self.optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            **optimizer_kwargs,
        )
        logger.info("AdamW optimizer: fused=%s", optimizer_fused)

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        if self.dataloader_require_merged_cache and getattr(
            self.train_dataset, "_merged_cache", None
        ) is None:
            raise RuntimeError(
                "dataloader_require_merged_cache=true but the training dataset "
                "did not activate a merged video cache"
            )
        if self.dataloader_require_video_latent_cache and getattr(
            self.train_dataset, "_video_latent_cache", None
        ) is None:
            raise RuntimeError(
                "dataloader_require_video_latent_cache=true but the training "
                "dataset did not activate a video latent cache"
            )
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        self.checkpoint_steps = _validate_checkpoint_steps(
            raw_checkpoint_steps,
            max_steps=self.max_steps,
        )
        warmup_ratio = float(getattr(cfg, "warmup_ratio", 0.05))
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}.")
        warmup_steps = int(total_train_steps * warmup_ratio)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            floor_lr=cfg.get("lr_floor", None),
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._optimizer_bridge_freeze_handles = (
            self._register_optimizer_bridge_freeze_hooks()
        )

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        self.checkpoint_state_keep_last = int(
            os.environ.get(
                "FASTWAM_CHECKPOINT_STATE_KEEP_LAST",
                cfg.get("checkpoint_state_keep_last", 2),
            )
        )
        self.checkpoint_weights_keep_local = int(
            os.environ.get(
                "FASTWAM_CHECKPOINT_WEIGHTS_KEEP_LOCAL",
                cfg.get("checkpoint_weights_keep_local", 1),
            )
        )
        checkpoint_oss_root = str(
            os.environ.get(
                "FASTWAM_CHECKPOINT_OSS_ROOT",
                cfg.get("checkpoint_oss_root", "") or "",
            )
        ).strip()
        self.checkpoint_store = None
        if checkpoint_oss_root and checkpoint_oss_root.lower() != "null":
            self.checkpoint_store = OssCheckpointStore(
                root_uri=checkpoint_oss_root,
                output_dir=self.output_dir,
                ossutil_bin=os.environ.get(
                    "FASTWAM_OSSUTIL_BIN",
                    str(cfg.get("checkpoint_ossutil_bin", "ossutil")),
                ),
                config_file=os.environ.get(
                    "FASTWAM_OSSUTIL_CONFIG",
                    str(cfg.get("checkpoint_ossutil_config", "~/.ossutilconfig")),
                ),
                timeout_seconds=int(
                    os.environ.get(
                        "FASTWAM_CHECKPOINT_UPLOAD_TIMEOUT",
                        cfg.get("checkpoint_upload_timeout_seconds", 900),
                    )
                ),
                retry_times=int(cfg.get("checkpoint_upload_retry_times", 3)),
                parallel=int(cfg.get("checkpoint_upload_parallel", 16)),
            )

        mirror_cfg = cfg.get("checkpoint_mirror", {}) or {}
        self.checkpoint_mirror_enabled = _parse_bool_config(
            os.environ.get(
                "FASTWAM_CHECKPOINT_MIRROR_ENABLED",
                mirror_cfg.get("enabled", False),
            ),
            name="checkpoint_mirror.enabled",
        )
        self.checkpoint_mirror_strict = _parse_bool_config(
            os.environ.get(
                "FASTWAM_CHECKPOINT_MIRROR_STRICT",
                mirror_cfg.get("strict", True),
            ),
            name="checkpoint_mirror.strict",
        )
        self.checkpoint_mirror_start_step = int(
            os.environ.get(
                "FASTWAM_CHECKPOINT_MIRROR_START_STEP",
                mirror_cfg.get("start_step", 0),
            )
        )
        self.checkpoint_mirror_interval_steps = int(
            os.environ.get(
                "FASTWAM_CHECKPOINT_MIRROR_INTERVAL_STEPS",
                mirror_cfg.get("interval_steps", 1),
            )
        )
        if self.checkpoint_mirror_start_step < 0:
            raise ValueError(
                "checkpoint_mirror.start_step must be non-negative, got "
                f"{self.checkpoint_mirror_start_step}"
            )
        if self.checkpoint_mirror_interval_steps <= 0:
            raise ValueError(
                "checkpoint_mirror.interval_steps must be positive, got "
                f"{self.checkpoint_mirror_interval_steps}"
            )
        self.checkpoint_mirror_drain_timeout_seconds = float(
            os.environ.get(
                "FASTWAM_CHECKPOINT_MIRROR_DRAIN_TIMEOUT_SECONDS",
                mirror_cfg.get("drain_timeout_seconds", 3600),
            )
        )
        if self.checkpoint_mirror_drain_timeout_seconds <= 0:
            raise ValueError(
                "checkpoint_mirror.drain_timeout_seconds must be positive, got "
                f"{self.checkpoint_mirror_drain_timeout_seconds}"
            )
        self.checkpoint_mirror = None
        self.checkpoint_mirror_backend = None
        if self.checkpoint_mirror_enabled and self.checkpoint_store is not None:
            raise ValueError(
                "checkpoint_mirror and synchronous checkpoint_oss_root cannot both "
                "be enabled; their independent retention/publish lifecycles would race"
            )
        if self.checkpoint_mirror_enabled and self.accelerator.is_main_process:
            mirror_root = str(
                os.environ.get(
                    "FASTWAM_CHECKPOINT_MIRROR_ROOT",
                    mirror_cfg.get("filesystem_root", "") or "",
                )
            ).strip()
            fallback_oss_root = str(
                os.environ.get(
                    "FASTWAM_CHECKPOINT_MIRROR_OSS_ROOT",
                    mirror_cfg.get("fallback_oss_root", "") or "",
                )
            ).strip()
            max_attempts = int(
                os.environ.get(
                    "FASTWAM_CHECKPOINT_MIRROR_MAX_ATTEMPTS",
                    mirror_cfg.get("max_attempts", 3),
                )
            )
            retry_backoff_seconds = float(
                os.environ.get(
                    "FASTWAM_CHECKPOINT_MIRROR_RETRY_BACKOFF_SECONDS",
                    mirror_cfg.get("retry_backoff_seconds", 2.0),
                )
            )
            mirror_path = Path(mirror_root).expanduser() if mirror_root else None
            if mirror_path is not None and mirror_path.is_dir():
                self.checkpoint_mirror = AsyncCheckpointMirror.from_filesystem(
                    mirror_path,
                    self.output_dir,
                    max_attempts=max_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                self.checkpoint_mirror_backend = "filesystem"
                logger.info(
                    "[ckpt-mirror] enabled backend=filesystem root=%s",
                    mirror_path,
                )
            elif fallback_oss_root:
                ossutil_bin = str(
                    os.environ.get(
                        "FASTWAM_CHECKPOINT_MIRROR_OSSUTIL_BIN",
                        mirror_cfg.get("ossutil_bin", "ossutil"),
                    )
                )
                ossutil_config = str(
                    os.environ.get(
                        "FASTWAM_CHECKPOINT_MIRROR_OSSUTIL_CONFIG",
                        mirror_cfg.get("ossutil_config", "~/.ossutilconfig"),
                    )
                )
                resolved_ossutil = Path(ossutil_bin).expanduser()
                resolved_config = Path(ossutil_config).expanduser()
                if os.sep in ossutil_bin and not resolved_ossutil.is_file():
                    raise FileNotFoundError(
                        f"Checkpoint mirror ossutil is unavailable: {resolved_ossutil}"
                    )
                if not resolved_config.is_file():
                    raise FileNotFoundError(
                        "Checkpoint mirror OSS config is unavailable: "
                        f"{resolved_config}"
                    )
                if resolved_config.stat().st_mode & 0o077:
                    raise PermissionError(
                        "Checkpoint mirror OSS config must have mode 600: "
                        f"{resolved_config}"
                    )
                mirror_store = OssCheckpointStore(
                    root_uri=fallback_oss_root,
                    output_dir=self.output_dir,
                    ossutil_bin=str(resolved_ossutil),
                    config_file=str(resolved_config),
                    timeout_seconds=int(
                        mirror_cfg.get("upload_timeout_seconds", 1800)
                    ),
                    retry_times=max_attempts,
                    parallel=int(mirror_cfg.get("upload_parallel", 4)),
                    force_overwrite=False,
                )
                self.checkpoint_mirror = AsyncCheckpointMirror.from_oss(
                    mirror_store,
                    probe=True,
                    max_attempts=max_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                self.checkpoint_mirror_backend = "oss"
                if mirror_path is not None:
                    logger.info(
                        "[ckpt-mirror] requested filesystem root is not mounted: %s; "
                        "using equivalent OSS root=%s",
                        mirror_path,
                        fallback_oss_root,
                    )
                else:
                    logger.info(
                        "[ckpt-mirror] enabled backend=oss root=%s",
                        fallback_oss_root,
                    )
            else:
                requested = str(mirror_path) if mirror_path is not None else "<unset>"
                raise FileNotFoundError(
                    "Checkpoint mirror is enabled but its filesystem root is not "
                    f"mounted ({requested}) and no OSS fallback is configured"
                )

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        if self._manual_fsdp and self.dataloader_rank_aware_strict:
            # The DataLoader is already rank-local. Preparing it through
            # Accelerate would shard it a second time and reintroduce padding.
            self.optimizer, self.scheduler = self.accelerator.prepare(
                self.optimizer, self.scheduler
            )
        elif self._manual_fsdp:
            # Model is already FSDP-wrapped + on device; prepare only the rest.
            self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
                self.optimizer, self.train_loader, self.scheduler
            )
        else:
            self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler
            )
        if self._is_deepspeed:
            logger.info(
                "DeepSpeed scheduler ownership: wrapper=%s engine_scheduler=%s",
                type(self.scheduler).__name__,
                type(getattr(self.model, "lr_scheduler", None)).__name__,
            )
        self.optimizer.zero_grad(set_to_none=True)
        self._gradient_debug_handles = self._register_gradient_debug_hooks()
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        self._apply_resume_step()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _register_gradient_debug_hooks(self):
        """Fail inside backward at the first parameter receiving a non-finite gradient.

        DeepSpeed's BF16 optimizer computes and validates the global norm inside
        ``engine.step()``, which Accelerate invokes from ``backward()``.  A normal
        post-backward check therefore runs too late to identify the offending
        parameter.  Keep this opt-in because checking every parameter introduces
        a device synchronization per gradient tensor.
        """
        if os.environ.get("FASTWAM_GRADIENT_DEBUG", "0") != "1":
            return []

        handles = []
        hooked = 0

        def make_hook(parameter_name):
            def check_gradient(grad):
                if grad is None:
                    return grad
                finite = torch.isfinite(grad)
                if bool(finite.all().item()):
                    return grad
                bad = int((~finite).sum().item())
                finite_values = grad.detach()[finite].float()
                finite_min = (
                    float(finite_values.min().item()) if finite_values.numel() else float("nan")
                )
                finite_max = (
                    float(finite_values.max().item()) if finite_values.numel() else float("nan")
                )
                raise FloatingPointError(
                    "FASTWAM_GRADIENT_DEBUG non-finite parameter gradient: "
                    f"name={parameter_name} rank={self.accelerator.process_index} "
                    f"shape={tuple(grad.shape)} dtype={grad.dtype} "
                    f"bad={bad}/{grad.numel()} finite_min={finite_min:.9g} "
                    f"finite_max={finite_max:.9g}"
                )

            return check_gradient

        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                handles.append(parameter.register_hook(make_hook(name)))
                hooked += 1
        logger.warning(
            "FASTWAM_GRADIENT_DEBUG=1: registered non-finite checks on %d trainable parameters.",
            hooked,
        )
        return handles

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = build_resumable_sampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
            process_index=getattr(self.accelerator, "process_index", 0),
            rank_aware_strict=getattr(self, "dataloader_rank_aware_strict", False),
        )
        persistent_workers, epoch_sensitive = self._resolve_persistent_workers(dataset)
        loader_kwargs = {}
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=getattr(self, "dataloader_pin_memory", torch.cuda.is_available()),
            worker_init_fn=worker_init_fn,
            timeout=self.dataloader_timeout,
            persistent_workers=persistent_workers,
            in_order=self.dataloader_in_order,
            **loader_kwargs,
        )
        logger.info(
            "DataLoader config: workers=%d batch_size=%d timeout=%.1fs "
            "prefetch_factor=%s persistent_workers=%s persistent_setting=%s "
            "epoch_sensitive=%s in_order=%s pin_memory=%s "
            "accelerate_non_blocking=%s",
            self.num_workers,
            self.batch_size,
            self.dataloader_timeout,
            self.dataloader_prefetch_factor if self.num_workers > 0 else "n/a",
            persistent_workers,
            (
                "auto"
                if self.dataloader_persistent_workers is None
                else self.dataloader_persistent_workers
            ),
            epoch_sensitive,
            self.dataloader_in_order,
            loader.pin_memory,
            self.dataloader_non_blocking,
        )
        return loader

    def _accelerator_dataloader_config(self) -> DataLoaderConfiguration:
        return DataLoaderConfiguration(non_blocking=self.dataloader_non_blocking)

    def _resolve_persistent_workers(self, dataset) -> tuple[bool, bool]:
        epoch_sensitive = callable(getattr(dataset, "set_epoch", None))
        configured = self.dataloader_persistent_workers
        if self.num_workers <= 0:
            if configured is True:
                raise ValueError(
                    "dataloader_persistent_workers=true requires num_workers > 0"
                )
            return False, epoch_sensitive
        if epoch_sensitive:
            if configured is True:
                raise ValueError(
                    "dataloader_persistent_workers=true is unsafe for a dataset "
                    "that exposes set_epoch(); use auto/false until epoch state is "
                    "shared with persistent worker copies"
                )
            return False, True
        if configured is None:
            return True, False
        return bool(configured), False

    def _set_data_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        self.train_sampler.set_epoch(epoch)
        dataset_set_epoch = getattr(self.train_dataset, "set_epoch", None)
        if callable(dataset_set_epoch):
            dataset_set_epoch(epoch)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0, floor_lr=None):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "cosine_floor":
            if floor_lr is None:
                raise ValueError("cosine_floor requires an explicit lr_floor")
            main_scheduler = FloorCosineLR(
                self.optimizer, T_max=remaining_steps, floor_lr=floor_lr,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'cosine_floor', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _paired_weights_checkpoint(self, state_dir: Path) -> Path:
        """Resolve the weights artifact paired with an exact state directory."""

        weights_dir = state_dir.parent.parent / "weights"
        adapter_path = weights_dir / f"{state_dir.name}.adapter.pt"
        bottleneck_path = weights_dir / f"{state_dir.name}.bottleneck.pt"
        full_path = weights_dir / f"{state_dir.name}.pt"
        if self.lora_enabled and self.lora_checkpoint_mode == "adapter":
            if adapter_path.is_file():
                return adapter_path
            raise FileNotFoundError(
                "LoRA training state is complete but its paired adapter is missing: "
                f"{adapter_path}"
            )
        if (
            bool(getattr(self, "bottleneck_adapter_enabled", False))
            and getattr(self, "bottleneck_adapter_checkpoint_mode", None) == "delta"
        ):
            if bottleneck_path.is_file():
                return bottleneck_path
            raise FileNotFoundError(
                "Bottleneck-adapter training state is complete but its paired "
                f"delta is missing: {bottleneck_path}"
            )
        if full_path.is_file():
            return full_path
        raise FileNotFoundError(
            f"Training state is complete but its paired model weights are missing: {full_path}"
        )

    def _resolve_lora_adapter_contract(
        self, adapter_path: str | Path
    ) -> tuple[dict[str, object], dict[str, object], Path]:
        if not self.lora_enabled:
            raise ValueError("Cannot load a LoRA adapter when lora.enabled=false.")
        payload = load_lora_adapter_payload(adapter_path)
        metadata = validate_lora_training_contract(
            payload,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            experts=self.lora_experts,
            target_modules=self.lora_target_modules,
            trainable_modules=self.lora_trainable_modules,
        )
        binding = metadata["base_checkpoint"]
        recorded_sha256 = str(binding.get("sha256", "")).strip().lower() or None
        if (
            self.lora_base_checkpoint_sha256 is not None
            and recorded_sha256 != self.lora_base_checkpoint_sha256
        ):
            raise ValueError(
                "Configured and adapter-recorded base SHA-256 differ: "
                f"configured={self.lora_base_checkpoint_sha256} "
                f"adapter={recorded_sha256}."
            )
        base_path = resolve_lora_base_checkpoint(
            payload,
            override=self.lora_base_checkpoint,
            allow_path_override=False,
            verify_sha256=self.lora_verify_base_sha256,
        )
        self.lora_base_checkpoint = str(base_path.resolve())
        self.lora_base_checkpoint_sha256 = recorded_sha256
        return payload, dict(metadata), base_path

    def _load_lora_base_and_adapter(self, model, adapter_path: str | Path) -> None:
        """Load the bound full base first, then the small adapter, with audits."""

        payload, metadata, base_path = self._resolve_lora_adapter_contract(adapter_path)
        recorded_sha256 = self.lora_base_checkpoint_sha256
        logger.info(
            "Loading LoRA base then adapter | base=%s adapter=%s adapter_step=%s sha256=%s",
            base_path,
            adapter_path,
            metadata.get("step"),
            recorded_sha256 or "not-recorded",
        )
        model.load_checkpoint(
            str(base_path),
            optimizer=None,
            mot_skip_prefixes=self.resume_mot_skip_prefixes,
            skip_proprio_encoder=self.resume_skip_proprio_encoder,
        )
        load_lora_adapter(model, payload)

    def _resolve_bottleneck_adapter_contract(
        self,
        adapter_path: str | Path,
        *,
        expected_step: int | None = None,
    ) -> tuple[dict[str, object], dict[str, object], Path]:
        if not self.bottleneck_adapter_enabled:
            raise ValueError(
                "Cannot load a bottleneck delta when "
                "bottleneck_adapter.enabled=false."
            )
        payload = load_bottleneck_adapter_payload(adapter_path)
        metadata = validate_bottleneck_training_contract(
            payload,
            experts=self.bottleneck_adapter_experts,
            sites=self.bottleneck_adapter_sites,
            bottleneck_dims=self.bottleneck_adapter_dims,
            activation=self.bottleneck_adapter_activation,
            dropout=self.bottleneck_adapter_dropout,
            residual_scale=self.bottleneck_adapter_residual_scale,
            init_seed=self.bottleneck_adapter_init_seed,
            trainable_modules=self.bottleneck_adapter_trainable_modules,
        )
        if expected_step is not None:
            observed_step = int(metadata.get("step", -1))
            if observed_step != int(expected_step):
                raise ValueError(
                    "Paired bottleneck delta/training-state step mismatch: "
                    f"delta={observed_step} state={int(expected_step)}."
                )
        binding = metadata["base_checkpoint"]
        recorded_sha256 = str(binding.get("sha256", "")).strip().lower() or None
        if (
            self.bottleneck_adapter_base_checkpoint_sha256 is not None
            and recorded_sha256
            != self.bottleneck_adapter_base_checkpoint_sha256
        ):
            raise ValueError(
                "Configured and bottleneck-delta-recorded base SHA-256 differ: "
                f"configured={self.bottleneck_adapter_base_checkpoint_sha256} "
                f"adapter={recorded_sha256}."
            )
        base_path = resolve_bottleneck_base_checkpoint(
            payload,
            override=self.bottleneck_adapter_base_checkpoint,
            allow_path_override=False,
            verify_sha256=self.bottleneck_adapter_verify_base_sha256,
        )
        self.bottleneck_adapter_base_checkpoint = str(base_path.resolve())
        self.bottleneck_adapter_base_checkpoint_sha256 = recorded_sha256
        return payload, dict(metadata), base_path

    def _load_bottleneck_base_and_adapter(
        self,
        model,
        adapter_path: str | Path,
        *,
        expected_step: int | None = None,
    ) -> None:
        """Load the immutable full base followed by its bound bottleneck delta."""

        payload, metadata, base_path = self._resolve_bottleneck_adapter_contract(
            adapter_path,
            expected_step=expected_step,
        )
        logger.info(
            "Loading bottleneck base then delta | base=%s delta=%s "
            "adapter_step=%s sha256=%s",
            base_path,
            adapter_path,
            metadata.get("step"),
            self.bottleneck_adapter_base_checkpoint_sha256 or "not-recorded",
        )
        model.load_checkpoint(
            str(base_path),
            optimizer=None,
            mot_skip_prefixes=self.resume_mot_skip_prefixes,
            skip_proprio_encoder=self.resume_skip_proprio_encoder,
        )
        load_bottleneck_adapter(model, payload)

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if self.lora_resume_adapter is not None:
            self._load_lora_base_and_adapter(
                self._unwrap_model(), self.lora_resume_adapter
            )
            self._refresh_masters_after_weights_only_load()
            logger.warning(
                "Loaded LoRA weights only; optimizer/scheduler/local step were not restored."
            )
            return
        if self.bottleneck_adapter_resume_delta is not None:
            self._load_bottleneck_base_and_adapter(
                self._unwrap_model(),
                self.bottleneck_adapter_resume_delta,
                expected_step=(self.resume_step if self.resume_step > 0 else None),
            )
            self._refresh_masters_after_weights_only_load()
            logger.warning(
                "Loaded bottleneck-adapter weights only; "
                "optimizer/scheduler/local step were not restored."
            )
            return
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            if (
                self.lora_enabled
                and self.lora_checkpoint_mode == "adapter"
                and not getattr(self, "_manual_fsdp", False)
            ):
                # Accelerate state may contain its own model shard, but future
                # adapter saves still require the immutable base binding.
                paired_adapter = self._paired_weights_checkpoint(resume_path)
                self._resolve_lora_adapter_contract(paired_adapter)
            if (
                self.bottleneck_adapter_enabled
                and self.bottleneck_adapter_checkpoint_mode == "delta"
                and not getattr(self, "_manual_fsdp", False)
            ):
                paired_delta = self._paired_weights_checkpoint(resume_path)
                trainer_state_path = resume_path / "trainer_state.json"
                if not trainer_state_path.is_file():
                    raise FileNotFoundError(
                        "Exact bottleneck-adapter resume requires trainer_state.json: "
                        f"{trainer_state_path}"
                    )
                expected_step = int(
                    json.loads(trainer_state_path.read_text())["global_step"]
                )
                self._resolve_bottleneck_adapter_contract(
                    paired_delta,
                    expected_step=expected_step,
                )
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self._unwrap_model().load_checkpoint(
            str(resume_path),
            optimizer=None,
            mot_skip_prefixes=self.resume_mot_skip_prefixes,
            skip_proprio_encoder=self.resume_skip_proprio_encoder,
        )
        self._refresh_masters_after_weights_only_load()
        logger.warning("Loaded .pt weights only; optimizer moments/scheduler/step were not restored.")

    def _refresh_masters_after_weights_only_load(self):
        if not getattr(self, "_is_deepspeed", False):
            return
        result = refresh_deepspeed_master_params(self.model)
        logger.info("DeepSpeed weights-only master refresh: %s", result)
        audit_dir = os.environ.get("FASTWAM_WEIGHTS_FIRST_STEP_AUDIT_DIR")
        if audit_dir:
            # Opt-in launch verification for a fresh, low-LR transfer. Samples
            # span every trainable tensor, not only the action IO interfaces.
            probes = {}
            for name, param in self._unwrap_model().named_parameters():
                if param.requires_grad and param.numel():
                    indices = torch.linspace(0, param.numel() - 1, min(64, param.numel()),
                                             device=param.device, dtype=torch.float64).long()
                    probes[name] = (indices, param.detach().reshape(-1)[indices].float().cpu())
            self._weights_first_step_probes = probes
            self._weights_first_step_refresh = result

    def _verify_weights_after_first_step(self):
        probes = getattr(self, "_weights_first_step_probes", None)
        if not probes:
            return
        parameters = dict(self._unwrap_model().named_parameters())
        deltas = {}
        for name, (indices, before) in probes.items():
            after = parameters[name].detach().reshape(-1)[indices].float().cpu()
            deltas[name] = float((after - before).abs().max())
        threshold = float(os.environ.get("FASTWAM_WEIGHTS_FIRST_STEP_MAX_DELTA", "0.01"))
        maximum = max(deltas.values())
        passed = all(np.isfinite(v) and v <= threshold for v in deltas.values())
        flags = self.accelerator.gather(torch.tensor([int(passed)], device=self.accelerator.device))
        record = {"status": "PASS" if passed else "FAIL", "step": self.global_step,
                  "rank": self.accelerator.process_index, "tensor_probes": len(probes),
                  "max_abs_delta": maximum, "threshold": threshold,
                  "master_refresh": self._weights_first_step_refresh, "deltas": deltas}
        out = Path(os.environ["FASTWAM_WEIGHTS_FIRST_STEP_AUDIT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rank_{self.accelerator.process_index:02d}.json").write_text(json.dumps(record, indent=2) + "\n")
        del self._weights_first_step_probes
        if not bool(flags.all().item()):
            raise RuntimeError("Loaded weights changed excessively on first optimizer step; see first-step audit")
        logger.info("Weights first optimizer step audit PASS: rank=%d tensors=%d max_delta=%g",
                    self.accelerator.process_index, len(probes), maximum)

    def _apply_resume_step(self):
        """Weights-only resume that INHERITS the LR schedule.

        A normal resume restores optimizer/scheduler/global_step from the `state/` dir.
        When that dir is lost, `resume` still loads the `.pt` weights but global_step stays
        0, so the cosine LR would restart from the top. If `resume_step > 0`, set
        global_step to it and fast-forward the LR scheduler so training continues the SAME
        cosine curve from that step to max_steps (the last few % of training) instead of
        re-warming up. Optimizer moment estimates are NOT restored (fresh Adam state); at
        the low tail-end LR this is a negligible perturbation.
        """
        step = int(getattr(self, "resume_step", 0) or 0)
        if step <= 0:
            return
        if self.max_steps is not None and step >= self.max_steps:
            raise ValueError(f"resume_step={step} must be < max_steps={self.max_steps}.")
        self.global_step = step
        self.epoch = 0
        # A weights-only continuation must advance the deterministic sampler as
        # well as the LR schedule.  Each optimizer step consumes
        # `gradient_accumulation_steps` global micro-batches.
        self.batch_in_epoch = step * self.gradient_accumulation_steps
        self.train_sampler.set_epoch_offset(0)
        self.train_sampler.set_epoch(self.epoch)
        self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
        # Fast-forward the LR scheduler to `step`. Step the UNDERLYING scheduler directly:
        # the Accelerate wrapper's step() no-ops before any optimizer.step() (its
        # gradient-accumulation guard), which would leave the LR unmoved. The wrapped and
        # raw optimizers share param_groups, so advancing the raw scheduler sets the live LR.
        raw = getattr(self.scheduler, "scheduler", self.scheduler)
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # silence "step() before optimizer.step()"
            for _ in range(step):
                raw.step()
        lr_now = self.optimizer.param_groups[0]["lr"]
        logger.info(
            "resume_step=%d: set global_step, restored sampler batch offset=%d, and "
            "fast-forwarded LR scheduler -> lr=%.4e (fresh optimizer moments; %d "
            "steps remain to max_steps=%s).",
            step,
            self.batch_in_epoch,
            lr_now,
            (self.max_steps - step) if self.max_steps else -1,
            self.max_steps,
        )

    def _maybe_preload_weights_for_fsdp(self):
        """Load resume weights before FSDP wraps and flattens parameters."""
        resume = self.resume
        if self.lora_resume_adapter is not None:
            self._load_lora_base_and_adapter(self.model, self.lora_resume_adapter)
            self.resume = None  # weights already in place; skip post-wrap load
            self.lora_resume_adapter = None
            return
        if self.bottleneck_adapter_resume_delta is not None:
            self._load_bottleneck_base_and_adapter(
                self.model,
                self.bottleneck_adapter_resume_delta,
                expected_step=(self.resume_step if self.resume_step > 0 else None),
            )
            self.resume = None  # weights already in place; skip post-wrap load
            self.bottleneck_adapter_resume_delta = None
            return
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            manifest = self._validate_manual_fsdp_state_dir(resume_path)
            weights_path = self._paired_weights_checkpoint(resume_path)
            logger.info(
                "Manual FSDP full-state resume: pre-wrap model load weights=%s "
                "state=%s step=%d world_size=%d",
                weights_path,
                resume_path,
                int(manifest["step"]),
                int(manifest["world_size"]),
            )
            if is_lora_adapter_checkpoint(weights_path):
                self._load_lora_base_and_adapter(self.model, weights_path)
            elif is_bottleneck_adapter_checkpoint(weights_path):
                self._load_bottleneck_base_and_adapter(
                    self.model,
                    weights_path,
                    expected_step=int(manifest["step"]),
                )
            else:
                self.model.load_checkpoint(
                    str(weights_path),
                    optimizer=None,
                    mot_skip_prefixes=self.resume_mot_skip_prefixes,
                    skip_proprio_encoder=self.resume_skip_proprio_encoder,
                )
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("FSDP pre-wrap weights load: %s", resume_path)
        self.model.load_checkpoint(
            str(resume_path),
            optimizer=None,
            mot_skip_prefixes=self.resume_mot_skip_prefixes,
            skip_proprio_encoder=self.resume_skip_proprio_encoder,
        )
        self.resume = None  # weights already in place; skip post-wrap load

    def _manual_fsdp_optimizer_path(self, state_dir: Path, rank: int | None = None) -> Path:
        if rank is None:
            rank = int(self.accelerator.process_index)
        return state_dir / f"optimizer_rank_{int(rank):05d}.bin"

    def _manual_fsdp_optimizer_meta_path(self, state_dir: Path, rank: int | None = None) -> Path:
        if rank is None:
            rank = int(self.accelerator.process_index)
        return state_dir / f"optimizer_rank_{int(rank):05d}.json"

    def _optimizer_layout(self) -> list[list[dict[str, object]]]:
        raw_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        return [
            [
                {
                    "shape": list(parameter.shape),
                    "numel": int(parameter.numel()),
                    "dtype": str(parameter.dtype),
                }
                for parameter in group["params"]
            ]
            for group in raw_optimizer.param_groups
        ]

    @staticmethod
    def _atomic_torch_save(payload, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json_save(payload: dict, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _validate_manual_fsdp_state_dir(self, state_dir: Path) -> dict:
        manifest_path = state_dir / "manual_fsdp_state.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                "Manual FSDP state predates per-rank optimizer checkpointing and is "
                f"not exactly resumable: {state_dir}. Missing {manifest_path.name}."
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != _MANUAL_FSDP_STATE_FORMAT:
            raise RuntimeError(
                f"Unsupported manual FSDP state format in {manifest_path}: "
                f"{manifest.get('format')!r}"
            )
        expected_world_size = int(self.accelerator.num_processes)
        observed_world_size = int(manifest.get("world_size", -1))
        if observed_world_size != expected_world_size:
            raise RuntimeError(
                "Manual FSDP optimizer shards require the same world size: "
                f"checkpoint={observed_world_size} current={expected_world_size}"
            )
        expected_sharding = os.environ.get("FSDP_SHARDING", "SHARD_GRAD_OP")
        if manifest.get("sharding_strategy") != expected_sharding:
            raise RuntimeError(
                "Manual FSDP optimizer shards require the same sharding strategy: "
                f"checkpoint={manifest.get('sharding_strategy')!r} "
                f"current={expected_sharding!r}"
            )
        required = [
            state_dir / "scheduler.bin",
            state_dir / "trainer_state.json",
        ]
        if bool(manifest.get("has_scaler", False)):
            required.append(state_dir / "scaler.bin")
        for rank in range(expected_world_size):
            required.extend(
                [
                    self._manual_fsdp_optimizer_path(state_dir, rank),
                    self._manual_fsdp_optimizer_meta_path(state_dir, rank),
                    state_dir / f"random_states_{rank}.pkl",
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Incomplete manual FSDP full state; missing: " + ", ".join(missing)
            )
        return manifest

    def _wrap_model_with_fsdp(self):
        """Wrap the whole FastWAM model as a single FSDP unit.

        FastWAM.forward() == training_loss() encloses the full DiT compute, so a
        single flat-param unit gathers all params on forward entry. The frozen
        VAE / text encoder go in `ignored_modules` so their plain-tensor forward
        keeps working and their (absent) grads never enter the reduce-scatter.
        No MixedPrecision policy: the model is already bf16, so FSDP's native
        dtype handling gives bf16 params/grads/reduce (an explicit policy would
        trigger FSDP's fp32 upcast path). use_orig_params=True keeps original
        param views alive for MoT's direct sub-param access.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

        # FULL_SHARD all-gathers before forward and backward. SHARD_GRAD_OP
        # all-gathers once before forward, retains the unsharded parameters
        # through backward, and reshards afterwards. It therefore saves one
        # all-gather versus FULL_SHARD, but it does not keep parameters replicated
        # between steps. Override with FSDP_SHARDING=FULL_SHARD if memory-bound.
        strategy = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }[os.environ.get("FSDP_SHARDING", "SHARD_GRAD_OP")]

        ignored = [
            m for m in (getattr(self.model, a, None) for a in ("vae", "text_encoder"))
            if isinstance(m, torch.nn.Module)
        ]
        self.model = FSDP(
            self.model,
            sharding_strategy=strategy,
            mixed_precision=None,
            ignored_modules=ignored if ignored else None,
            device_id=self.accelerator.device,
            use_orig_params=True,
            sync_module_states=False,
            limit_all_gathers=True,
            forward_prefetch=True,
        )
        logger.info("Manual FSDP: wrapped whole model as single %s unit; ignored=%d module(s)", strategy, len(ignored))

    def _unwrap_model(self, model=None):
        """Unwrap local PyTorch wrappers without importing unused DeepSpeed.

        Accelerate probes every installed optional backend in ``unwrap_model``.
        Some runtime images contain the DeepSpeed Python package but not ``nvcc``;
        importing that unused package then fails before an FSDP job can train.
        Keep Accelerate's implementation for real DeepSpeed jobs and handle the
        PyTorch wrappers used by FastWAM directly otherwise.
        """
        model = self.model if model is None else model
        if self._is_deepspeed:
            return self.accelerator.unwrap_model(model)

        wrappers = (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            wrappers = wrappers + (FSDP,)
        except (ImportError, RuntimeError):
            pass

        while True:
            compiled_model = getattr(model, "_orig_mod", None)
            if compiled_model is not None:
                model = compiled_model
                continue
            if isinstance(model, wrappers):
                model = model.module
                continue
            return model

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self._unwrap_model()
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        if bool(getattr(model, "lora_enabled", False)):
            trainable_names = set_lora_trainable(
                model,
                trainable_modules=tuple(
                    getattr(model, "lora_trainable_modules", ())
                ),
            )
            logger.info(
                "Setting LoRA train mode: tensors=%d tiny_modules=%s",
                len(trainable_names),
                list(getattr(model, "lora_trainable_modules", ())),
            )
            return
        if bool(getattr(model, "bottleneck_adapter_enabled", False)):
            trainable_names = set_bottleneck_adapter_trainable(
                model,
                trainable_modules=tuple(
                    getattr(model, "bottleneck_adapter_trainable_modules", ())
                ),
            )
            logger.info(
                "Setting bottleneck-adapter train mode: tensors=%d tiny_modules=%s",
                len(trainable_names),
                list(
                    getattr(model, "bottleneck_adapter_trainable_modules", ())
                ),
            )
            return
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        source_proprio_encoders = getattr(model, "source_proprio_encoders", None)
        if source_proprio_encoders is not None:
            source_proprio_encoders.train()
            source_proprio_encoders.requires_grad_(True)

    def _build_optimizer_param_groups(self, cfg, model):
        video_scale = float(cfg.get("optimizer_video_lr_scale", 1.0))
        action_scale = float(cfg.get("optimizer_action_lr_scale", 1.0))
        view_scale = float(cfg.get("optimizer_view_lr_scale", 1.0))
        video_freeze_steps = int(cfg.get("optimizer_video_freeze_steps", 0))
        for name, value in (
            ("optimizer_video_lr_scale", video_scale),
            ("optimizer_action_lr_scale", action_scale),
            ("optimizer_view_lr_scale", view_scale),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}.")
        if video_freeze_steps < 0:
            raise ValueError(
                "optimizer_video_freeze_steps must be >= 0, got "
                f"{video_freeze_steps}."
            )

        named_trainable = []
        seen = set()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            named_trainable.append((name, parameter))
        if not named_trainable:
            raise ValueError("No trainable parameters remain after DiT-only freezing.")

        custom_groups = any(
            value != 1.0 for value in (video_scale, action_scale, view_scale)
        ) or video_freeze_steps > 0
        if not custom_groups:
            return [parameter for _, parameter in named_trainable]

        buckets = {
            "video_backbone": [],
            "action": [],
            "view": [],
            "other": [],
        }
        for name, parameter in named_trainable:
            normalized = name.removeprefix("_orig_mod.").removeprefix(
                "_fsdp_wrapped_module."
            )
            if "view_embedding" in normalized:
                buckets["view"].append(parameter)
            elif normalized.startswith("video_expert.") or "mixtures.video." in normalized:
                buckets["video_backbone"].append(parameter)
            elif normalized.startswith("action_expert.") or "mixtures.action." in normalized:
                buckets["action"].append(parameter)
            else:
                buckets["other"].append(parameter)

        definitions = (
            ("video_backbone", video_scale, video_freeze_steps),
            ("action", action_scale, 0),
            ("view", view_scale, 0),
            ("other", action_scale, 0),
        )
        groups = []
        for group_name, lr_scale, freeze_until in definitions:
            parameters = buckets[group_name]
            if not parameters:
                continue
            groups.append(
                {
                    "params": parameters,
                    "lr": self.learning_rate * lr_scale,
                    "weight_decay": self.weight_decay,
                    "bridge_group_name": group_name,
                    "bridge_lr_scale": lr_scale,
                    "bridge_freeze_until_step": freeze_until,
                    "bridge_base_weight_decay": self.weight_decay,
                }
            )
            logger.info(
                "Optimizer bridge group | name=%s tensors=%d lr_scale=%.4f "
                "freeze_until_step=%d",
                group_name,
                len(parameters),
                lr_scale,
                freeze_until,
            )
        return groups

    def _register_optimizer_bridge_freeze_hooks(self):
        handles = []
        for group in self.optimizer.param_groups:
            freeze_until = int(group.get("bridge_freeze_until_step", 0))
            if freeze_until <= 0:
                continue
            for parameter in group["params"]:
                handles.append(
                    parameter.register_hook(
                        lambda grad, until=freeze_until: (
                            torch.zeros_like(grad)
                            if self.global_step < until
                            else grad
                        )
                    )
                )
        return handles

    def _apply_optimizer_bridge_runtime_state(self) -> None:
        # DeepSpeed ZeRO-1 may execute engine.step() inside backward(), so the
        # freeze must already be active before forward/backward begins. Gradient
        # hooks zero the update; disabling decoupled weight decay prevents AdamW
        # from drifting frozen weights despite a zero gradient.
        for group in self.optimizer.param_groups:
            freeze_until = int(group.get("bridge_freeze_until_step", 0))
            base_weight_decay = float(
                group.get("bridge_base_weight_decay", group.get("weight_decay", 0.0))
            )
            group["weight_decay"] = (
                0.0 if self.global_step < freeze_until else base_weight_decay
            )

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        video_layout = sample.get("video_layout", "pixel")
        if isinstance(video_layout, (list, tuple)):
            if len(video_layout) != 1:
                raise ValueError(
                    f"Expected one eval video layout, got {video_layout!r}."
                )
            video_layout = video_layout[0]
        video_layout = str(video_layout)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video_layout in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
        }:
            if video.ndim == 5:
                video = video.unsqueeze(0)
            if video.ndim != 6 or video.shape[1] != 3 or video.shape[2] != 3:
                raise ValueError(
                    "Expected separate-view video [3views,3,T,H,W] or "
                    f"[B,3views,3,T,H,W], got {tuple(video.shape)}."
                )
            num_video_frames = video.shape[3]
        else:
            if video.ndim == 4:
                video = video.unsqueeze(0)
            if video.ndim != 5:
                raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
            num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")

        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        goal_image = sample.get("goal_image", None)
        if goal_image is not None and goal_image.ndim == 3:
            goal_image = goal_image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            "goal_image": goal_image,
            "video_layout": video_layout,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None
        # Under manual FSDP the params are sharded 1-D outside FSDP.forward;
        # eval reaches into sub-params directly, so un-shard for the eval body.
        if getattr(self, "_manual_fsdp", False):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            _fsdp = getattr(self.model, "_orig_mod", self.model)  # unwrap torch.compile
            with FSDP.summon_full_params(_fsdp, writeback=False, recurse=True):
                return self._evaluate_inner()
        return self._evaluate_inner()

    def _evaluate_inner(self):
        model = self._unwrap_model()
        # In-training eval runs under FSDP.summon_full_params: params are un-sharded into
        # transient buffers that are freed/reallocated each eval, so capturing the inference
        # CUDA graphs here would bind to stale addresses. Force eager for the eval body
        # (deploy/standalone inference keeps the default-on graphs).
        if getattr(model, "_use_cuda_graph", False) or getattr(model, "_use_cuda_graph_prefill", False):
            model._use_cuda_graph = False
            model._use_cuda_graph_prefill = False
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        if sample["video_layout"] in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
        }:
            input_image = video0[:, :, 0].unsqueeze(0)  # [1,V,C,H,W]
            num_frames = int(video0.shape[2])
            gt_video0 = (
                model._compose_robotwin_pixel_multiview(video0)
                if sample["video_layout"] == "robotwin_multiview_tokens"
                else model._compose_robotwin_pixel_tshape(video0)
            )
        else:
            input_image = video0[:, 0].unsqueeze(0)
            _, num_frames, _, _ = video0.shape
            gt_video0 = video0

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )

        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((gt_video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _lora_checkpoint_metadata(self) -> dict[str, object]:
        if not self.lora_base_checkpoint:
            raise RuntimeError(
                "Refusing to save an unbound LoRA adapter: no full base checkpoint "
                "was recorded. Initialize LoRA post-training through cfg.resume or "
                "set lora.base_checkpoint explicitly."
            )
        base_path = Path(self.lora_base_checkpoint).expanduser()
        if not base_path.is_file():
            raise FileNotFoundError(
                f"Refusing to save LoRA adapter; bound base checkpoint is missing: {base_path}"
            )
        if is_lora_adapter_checkpoint(base_path):
            raise ValueError(
                f"LoRA base binding must point to a full checkpoint, not an adapter: {base_path}"
            )
        base_binding: dict[str, object] = {"path": str(base_path.resolve())}
        if self.lora_base_checkpoint_sha256 is not None:
            base_binding["sha256"] = self.lora_base_checkpoint_sha256
        return {
            "base_checkpoint": base_binding,
            "rank": int(self.lora_rank),
            "alpha": float(self.lora_alpha),
            "dropout": float(self.lora_dropout),
            "experts": list(self.lora_experts),
            "target_modules": list(self.lora_target_modules),
            "trainable_modules": list(self.lora_trainable_modules),
            "step": int(self.global_step),
            "world_size": int(self.accelerator.num_processes),
        }

    def _bottleneck_adapter_checkpoint_metadata(self) -> dict[str, object]:
        if not self.bottleneck_adapter_base_checkpoint:
            raise RuntimeError(
                "Refusing to save an unbound bottleneck delta: no full base "
                "checkpoint was recorded. Initialize post-training through "
                "cfg.resume or set bottleneck_adapter.base_checkpoint explicitly."
            )
        base_path = Path(self.bottleneck_adapter_base_checkpoint).expanduser()
        if not base_path.is_file():
            raise FileNotFoundError(
                "Refusing to save bottleneck delta; bound base checkpoint is "
                f"missing: {base_path}"
            )
        if is_lora_adapter_checkpoint(base_path) or is_bottleneck_adapter_checkpoint(
            base_path
        ):
            raise ValueError(
                "Bottleneck base binding must point to a full checkpoint, not a "
                f"PEFT delta: {base_path}"
            )
        base_binding: dict[str, object] = {"path": str(base_path.resolve())}
        if self.bottleneck_adapter_base_checkpoint_sha256 is not None:
            base_binding["sha256"] = (
                self.bottleneck_adapter_base_checkpoint_sha256
            )
        return {
            "base_checkpoint": base_binding,
            "experts": list(self.bottleneck_adapter_experts),
            "sites": list(self.bottleneck_adapter_sites),
            "bottleneck_dims": dict(self.bottleneck_adapter_dims),
            "activation": self.bottleneck_adapter_activation,
            "dropout": float(self.bottleneck_adapter_dropout),
            "residual_scale": float(self.bottleneck_adapter_residual_scale),
            "init_seed": int(self.bottleneck_adapter_init_seed),
            "trainable_modules": list(
                self.bottleneck_adapter_trainable_modules
            ),
            "step": int(self.global_step),
            "world_size": int(self.accelerator.num_processes),
        }

    def _save_weights_checkpoint(self, step_tag: str):
        lora_adapter_only = (
            self.lora_enabled and self.lora_checkpoint_mode == "adapter"
        )
        bottleneck_delta_only = (
            bool(getattr(self, "bottleneck_adapter_enabled", False))
            and getattr(self, "bottleneck_adapter_checkpoint_mode", None) == "delta"
        )
        if lora_adapter_only and bottleneck_delta_only:
            raise RuntimeError("Multiple PEFT delta checkpoint modes are active.")
        if lora_adapter_only:
            suffix = ".adapter.pt"
        elif bottleneck_delta_only:
            suffix = ".bottleneck.pt"
        else:
            suffix = ".pt"
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}{suffix}")

        def save_unwrapped() -> None:
            unwrapped = self._unwrap_model()
            if lora_adapter_only:
                save_lora_adapter(
                    unwrapped,
                    ckpt_path,
                    trainable_modules=self.lora_trainable_modules,
                    metadata=self._lora_checkpoint_metadata(),
                )
            elif bottleneck_delta_only:
                save_bottleneck_adapter(
                    unwrapped,
                    ckpt_path,
                    trainable_modules=self.bottleneck_adapter_trainable_modules,
                    metadata=self._bottleneck_adapter_checkpoint_metadata(),
                )
            else:
                unwrapped.save_checkpoint(
                    ckpt_path, optimizer=None, step=self.global_step
                )

        if getattr(self, "_manual_fsdp", False):
            # Collective: every rank must enter summon_full_params to un-shard the
            # flat params; only rank 0 writes. PEFT delta modes collect only
            # trainable tensors after summoning, avoiding a 12 GB checkpoint.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            _fsdp = getattr(self.model, "_orig_mod", self.model)  # unwrap torch.compile
            with FSDP.summon_full_params(_fsdp, writeback=False, recurse=True):
                if self.accelerator.is_main_process:
                    save_unwrapped()
        elif self.accelerator.is_main_process:
            save_unwrapped()
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _save_manual_fsdp_training_state(self, state_path: str) -> None:
        """Save an exact, same-topology manual-FSDP training state.

        ``Accelerator.save_state`` writes optimizers only on the global main
        process.  That is invalid for this manually wrapped SHARD_GRAD_OP model:
        every rank owns a different Adam shard.  Save each rank's optimizer and
        RNG state to a unique file instead, while rank 0 saves the shared
        scheduler/scaler/trainer metadata.
        """
        state_dir = Path(state_path)
        rank = int(self.accelerator.process_index)
        world_size = int(self.accelerator.num_processes)

        optimizer_path = self._manual_fsdp_optimizer_path(state_dir, rank)
        optimizer_meta_path = self._manual_fsdp_optimizer_meta_path(state_dir, rank)
        optimizer_state = self.optimizer.state_dict()
        self._atomic_torch_save(optimizer_state, optimizer_path)
        self._atomic_json_save(
            {
                "rank": rank,
                "world_size": world_size,
                "layout": self._optimizer_layout(),
                "bytes": int(optimizer_path.stat().st_size),
            },
            optimizer_meta_path,
        )
        del optimizer_state

        rng_state = {
            "step": int(self.accelerator.step),
            "random_state": random.getstate(),
            "numpy_random_seed": np.random.get_state(),
            "torch_manual_seed": torch.get_rng_state(),
            "torch_cuda_manual_seed": torch.cuda.get_rng_state_all(),
        }
        self._atomic_torch_save(rng_state, state_dir / f"random_states_{rank}.pkl")

        if self.accelerator.is_main_process:
            self._atomic_torch_save(self.scheduler.state_dict(), state_dir / "scheduler.bin")
            scaler = self.accelerator.scaler
            if scaler is not None:
                self._atomic_torch_save(scaler.state_dict(), state_dir / "scaler.bin")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_trainer_state(str(state_dir))
            manifest = {
                "format": _MANUAL_FSDP_STATE_FORMAT,
                "step": int(self.global_step),
                "world_size": world_size,
                "sharding_strategy": os.environ.get("FSDP_SHARDING", "SHARD_GRAD_OP"),
                "mixed_precision": self.mixed_precision,
                "optimizer": type(getattr(self.optimizer, "optimizer", self.optimizer)).__name__,
                "has_scaler": self.accelerator.scaler is not None,
                "optimizer_files": [
                    {
                        "rank": shard_rank,
                        "path": self._manual_fsdp_optimizer_path(state_dir, shard_rank).name,
                        "bytes": int(
                            self._manual_fsdp_optimizer_path(state_dir, shard_rank).stat().st_size
                        ),
                    }
                    for shard_rank in range(world_size)
                ],
            }
            self._atomic_json_save(manifest, state_dir / "manual_fsdp_state.json")
        self.accelerator.wait_for_everyone()

    def _load_manual_fsdp_training_state(self, state_dir: Path) -> None:
        manifest = self._validate_manual_fsdp_state_dir(state_dir)
        rank = int(self.accelerator.process_index)

        optimizer_meta = json.loads(
            self._manual_fsdp_optimizer_meta_path(state_dir, rank).read_text()
        )
        current_layout = self._optimizer_layout()
        if optimizer_meta.get("layout") != current_layout:
            raise RuntimeError(
                "Manual FSDP optimizer parameter layout changed; exact resume is unsafe: "
                f"rank={rank} checkpoint={state_dir}"
            )

        optimizer_state = torch.load(
            self._manual_fsdp_optimizer_path(state_dir, rank),
            map_location="cpu",
            weights_only=False,
        )
        self.optimizer.load_state_dict(optimizer_state)
        del optimizer_state

        raw_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        for parameter, parameter_state in raw_optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                tensor = parameter_state.get(key)
                if tensor is not None and tensor.shape != parameter.shape:
                    raise RuntimeError(
                        "Loaded Adam moment does not match its local FSDP parameter shard: "
                        f"rank={rank} key={key} moment={tuple(tensor.shape)} "
                        f"parameter={tuple(parameter.shape)}"
                    )

        scheduler_state = torch.load(
            state_dir / "scheduler.bin", map_location="cpu", weights_only=False
        )
        self.scheduler.load_state_dict(scheduler_state)
        if bool(manifest.get("has_scaler", False)):
            if self.accelerator.scaler is None:
                raise RuntimeError("Checkpoint has a GradScaler but the current run does not.")
            scaler_state = torch.load(
                state_dir / "scaler.bin", map_location="cpu", weights_only=False
            )
            self.accelerator.scaler.load_state_dict(scaler_state)

        rng_state = torch.load(
            state_dir / f"random_states_{rank}.pkl",
            map_location="cpu",
            weights_only=False,
        )
        random.setstate(rng_state["random_state"])
        np.random.set_state(rng_state["numpy_random_seed"])
        torch.set_rng_state(rng_state["torch_manual_seed"])
        torch.cuda.set_rng_state_all(rng_state["torch_cuda_manual_seed"])
        self.accelerator.step = int(rng_state.get("step", 0))

        logger.info(
            "Loaded exact manual-FSDP state rank=%d/%d optimizer=%s scheduler=ok rng=ok",
            rank,
            int(manifest["world_size"]),
            self._manual_fsdp_optimizer_path(state_dir, rank),
        )

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        # Called on ALL ranks: under FSDP the weights save is a collective
        # (summon_full_params); the method writes only on rank 0.
        ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        # The file is closed and every FSDP rank has left the full-parameter
        # gather before rank zero queues it.  Large filesystem copies/uploads
        # now overlap state saving and subsequent optimizer steps.
        mirror_path = None
        checkpoint_mirror = getattr(self, "checkpoint_mirror", None)
        mirror_step_selected = (
            self.global_step >= self.checkpoint_mirror_start_step
            and (
                self.global_step - self.checkpoint_mirror_start_step
            )
            % self.checkpoint_mirror_interval_steps
            == 0
        )
        if (
            self.accelerator.is_main_process
            and checkpoint_mirror is not None
            and mirror_step_selected
        ):
            mirror_path = checkpoint_mirror.enqueue(ckpt_path)
            logger.info(
                "[ckpt-mirror] queued step=%d backend=%s source=%s destination=%s",
                self.global_step,
                getattr(self, "checkpoint_mirror_backend", "unknown"),
                ckpt_path,
                mirror_path,
            )
        elif self.accelerator.is_main_process and checkpoint_mirror is not None:
            logger.info(
                "[ckpt-mirror] skipped step=%d schedule=start:%d interval:%d",
                self.global_step,
                self.checkpoint_mirror_start_step,
                self.checkpoint_mirror_interval_steps,
            )

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        if getattr(self, "_manual_fsdp", False):
            self._save_manual_fsdp_training_state(state_path)
        else:
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        published_weights_path = ckpt_path
        upload_error_path = Path(f"{ckpt_path}.upload_failed.json")
        if self.checkpoint_store is not None:
            if self.accelerator.is_main_process:
                upload_error_path.unlink(missing_ok=True)
                try:
                    upload_result = self.checkpoint_store.upload(ckpt_path)
                    published_weights_path = upload_result.oss_uri
                    logger.info(
                        "[ckpt-oss] step=%d bytes=%d uri=%s",
                        self.global_step,
                        upload_result.size,
                        upload_result.oss_uri,
                    )
                except Exception as exc:
                    upload_error_path.write_text(
                        json.dumps(
                            {
                                "step": self.global_step,
                                "local_path": ckpt_path,
                                "error": repr(exc),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
            self.accelerator.wait_for_everyone()
            if upload_error_path.exists():
                raise RuntimeError(
                    f"Checkpoint OSS upload failed; local weight was retained: {upload_error_path}"
                )
            published_weights_path = self.checkpoint_store.oss_uri(ckpt_path)

        if self.accelerator.is_main_process:
            removed_states = prune_training_states(
                self.state_dir, self.checkpoint_state_keep_last
            )
            if removed_states:
                logger.info(
                    "[ckpt-retention] removed %d old training states; keep_last=%d",
                    len(removed_states),
                    self.checkpoint_state_keep_last,
                )
            if self.checkpoint_store is not None:
                removed_weights = prune_uploaded_weights(
                    self.weights_dir, self.checkpoint_weights_keep_local
                )
                if removed_weights:
                    logger.info(
                        "[ckpt-retention] removed %d verified local weights; keep_last=%d",
                        len(removed_weights),
                        self.checkpoint_weights_keep_local,
                    )
        self.accelerator.wait_for_everyone()

        return {
            "weights_path": published_weights_path,
            "local_weights_path": ckpt_path,
            "mirror_path": mirror_path,
            "state_path": state_path,
        }

    def _maybe_save_initial_checkpoint(self):
        """Save the real initialized state before the first batch is requested.

        Exact-state resume is excluded even when it points at step zero: that
        state already owns its step-0 artifact and must not be overwritten.
        Weights-only initialization is intentionally eligible because its
        loaded weights are the true local step-0 state for that training run.
        """

        if (
            not self.save_initial_checkpoint
            or self.global_step != 0
            or self._exact_state_resume
        ):
            return None
        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[ckpt-initial] step=0 weights=%s state=%s",
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        return ckpt_info

    def finish_checkpoint_mirror(self, *, strict: bool | None = None):
        """Drain and close rank-zero asynchronous checkpoint publishing.

        Normal training exits use strict mode so a run cannot report success
        while its requested backup failed.  During an existing training error,
        runtime calls this with ``strict=False`` to preserve the original
        exception while still giving completed checkpoints a bounded chance to
        finish publishing.
        """
        checkpoint_mirror = getattr(self, "checkpoint_mirror", None)
        if checkpoint_mirror is None:
            return []
        strict = (
            bool(getattr(self, "checkpoint_mirror_strict", True))
            if strict is None
            else bool(strict)
        )
        results = checkpoint_mirror.close(
            strict=strict,
            timeout_seconds=float(
                getattr(self, "checkpoint_mirror_drain_timeout_seconds", 3600)
            ),
        )
        logger.info(
            "[ckpt-mirror] drained backend=%s completed=%d strict=%s",
            getattr(self, "checkpoint_mirror_backend", "unknown"),
            len(results),
            strict,
        )
        return results

    def load_training_state(self, state_dir: str):
        if getattr(self, "_manual_fsdp", False):
            self._load_manual_fsdp_training_state(Path(state_dir))
        else:
            self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(0)
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self._unwrap_model()

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        self._maybe_save_initial_checkpoint()

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        self._set_data_epoch(self.epoch)
        data_iter = iter(self.train_loader)
        from fastwam.utils.training_gc import configure_from_env
        _training_gc = configure_from_env(int(self.accelerator.process_index))
        if _training_gc is not None:
            self.accelerator.wait_for_everyone()
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        # --- step timing instrumentation (diagnostic) ---
        _prof = {
            "data": 0.0,
            "h2d": 0.0,
            "ready_max": 0.0,
            "ready_spread": 0.0,
            "fwd": 0.0,
            "bwd": 0.0,
            "opt": 0.0,
            "n": 0,
        }
        # SourceMix uses a deterministic 1-in-N period. When log_every equals that
        # period (the replay runs use 10 and 0.1), logging only the current step
        # aliases the source schedule and can misleadingly show long all-robot/all-Ego
        # stretches. Keep the instantaneous metrics for compatibility and also report
        # source-aware means over the complete logging window.
        _source_log_window = {
            "steps": 0,
            "ego_fraction": 0.0,
            "ego_steps": 0.0,
            "robot_steps": 0.0,
            "ego_loss_action": 0.0,
            "robot_loss_action": 0.0,
            "ego_loss_video": 0.0,
            "robot_loss_video": 0.0,
        }
        # STEP_PROFILE=1 adds per-phase torch.cuda.synchronize() for accurate
        # data/fwd/bwd/opt attribution (small overhead). Default off.
        _PROF = os.environ.get("STEP_PROFILE", "0") == "1"
        _RANK_PHASE_DIR = os.environ.get("FASTWAM_RANK_PHASE_DIR", "").strip()
        _rank_phase_path = None
        _rank_phase_summary_path = None
        _optimizer_step_wall0 = None
        _optimizer_step_profile = {
            "data_fetch_ms": 0.0,
            "h2d_ms": 0.0,
            "pre_forward_wait_ms": 0.0,
            "forward_wall_ms": 0.0,
            "forward_cuda_ms": 0.0,
            "backward_wall_ms": 0.0,
            "backward_cuda_ms": 0.0,
        }
        if _PROF and _RANK_PHASE_DIR:
            phase_root = Path(_RANK_PHASE_DIR)
            phase_root.mkdir(parents=True, exist_ok=True)
            _rank_phase_path = (
                phase_root
                / f"rank_{int(self.accelerator.process_index):05d}.jsonl"
            )
            _rank_phase_summary_path = phase_root / "rank_phase_summary.jsonl"

        def _append_phase(path: Path | None, payload: dict) -> None:
            if path is None:
                return
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
                stream.flush()

        _FAIL_ON_NONFINITE = os.environ.get("FASTWAM_FAIL_ON_NONFINITE", "1") == "1"
        _SYNC_AFTER_BACKWARD = os.environ.get(
            "FASTWAM_SYNC_AFTER_BACKWARD",
            "1" if self._manual_fsdp else "0",
        ) == "1"
        _STOP_AFTER_RUN_STEPS = int(os.environ.get("FASTWAM_STOP_AFTER_RUN_STEPS", "0"))
        _SKIP_DIAGNOSTIC_CHECKPOINT = os.environ.get(
            "FASTWAM_DIAGNOSTIC_SKIP_CHECKPOINT", "0"
        ) == "1"
        if _SKIP_DIAGNOSTIC_CHECKPOINT and _STOP_AFTER_RUN_STEPS <= 0:
            raise ValueError(
                "FASTWAM_DIAGNOSTIC_SKIP_CHECKPOINT=1 requires "
                "FASTWAM_STOP_AFTER_RUN_STEPS to be positive"
            )
        _diagnostic_stop_step = (
            self.run_start_step + _STOP_AFTER_RUN_STEPS
            if _STOP_AFTER_RUN_STEPS > 0
            else None
        )
        _merged_cache_hits = 0
        _merged_cache_misses = 0
        _latent_cache_hits = 0
        _latent_cache_misses = 0
        _pyav_calls = 0

        while self.global_step < self.max_steps:
            try:
                if _training_gc is not None:
                    _training_gc.tick(self.global_step, "train")
                _step_wall0 = time.perf_counter()
                _td0 = time.perf_counter()
                sample = next(data_iter)
                if self.dataloader_rank_aware_strict:
                    sample = send_to_device(
                        sample,
                        self.accelerator.device,
                        non_blocking=self.dataloader_non_blocking,
                    )
                cache_marker = sample.get("_fastwam_merged_video_cache")
                if cache_marker is None:
                    cache_hits = 0
                    cache_misses = int(self.batch_size)
                else:
                    marker_tensor = torch.as_tensor(cache_marker, dtype=torch.bool)
                    cache_hits = int(marker_tensor.sum().item())
                    cache_misses = int(marker_tensor.numel() - cache_hits)
                _merged_cache_hits += cache_hits
                _merged_cache_misses += cache_misses
                latent_cache_marker = sample.get("_fastwam_video_latent_cache")
                if latent_cache_marker is None:
                    latent_cache_hits = 0
                    latent_cache_misses = int(self.batch_size)
                else:
                    latent_marker_tensor = torch.as_tensor(
                        latent_cache_marker, dtype=torch.bool
                    )
                    latent_cache_hits = int(latent_marker_tensor.sum().item())
                    latent_cache_misses = int(
                        latent_marker_tensor.numel() - latent_cache_hits
                    )
                _latent_cache_hits += latent_cache_hits
                _latent_cache_misses += latent_cache_misses
                pyav_calls = int(
                    torch.as_tensor(
                        sample.get("_fastwam_pyav_calls", 0),
                        dtype=torch.int64,
                    )
                    .sum()
                    .item()
                )
                _pyav_calls += pyav_calls
                if self.dataloader_fail_on_cache_miss and cache_misses:
                    raise RuntimeError(
                        "merged-cache production gate failed: "
                        f"hits={cache_hits} misses={cache_misses} "
                        f"global_step={self.global_step}"
                    )
                if self.dataloader_fail_on_cache_miss and pyav_calls:
                    raise RuntimeError(
                        "zero-PyAV production gate failed: "
                        f"calls={pyav_calls} global_step={self.global_step}"
                    )
                if (
                    self.dataloader_fail_on_latent_cache_miss
                    and latent_cache_misses
                ):
                    raise RuntimeError(
                        "video-latent-cache production gate failed: "
                        f"hits={latent_cache_hits} misses={latent_cache_misses} "
                        f"global_step={self.global_step}"
                    )
                _data_elapsed = time.perf_counter() - _td0
                _prof["data"] += _data_elapsed
                _h2d_elapsed = 0.0
                _pre_forward_wait = 0.0
                _forward_cuda_ms = 0.0
                _backward_cuda_ms = 0.0
                _optimizer_cuda_ms = 0.0
                _post_step_barrier = 0.0
                self.batch_in_epoch += 1
                batch_crop_width = None
                if self.batch_shared_center_crop is not None:
                    video = sample.get("video")
                    if video is None:
                        if sample.get("video_latents") is not None:
                            raise RuntimeError(
                                "batch_shared_center_crop cannot crop precomputed "
                                "video latents"
                            )
                        raise RuntimeError(
                            "batch_shared_center_crop is enabled but the batch has "
                            "no `video` tensor"
                        )
                    batch_crop_width = self.batch_shared_center_crop.long_edge_for_batch(
                        epoch=self.epoch,
                        batch_in_epoch=self.batch_in_epoch,
                    )
                    sample["video"] = self.batch_shared_center_crop.apply(
                        video,
                        long_edge=batch_crop_width,
                    )
                    if self.batch_shared_center_crop_verify_across_ranks:
                        local_crop_width = torch.tensor(
                            [batch_crop_width],
                            device=self.accelerator.device,
                            dtype=torch.int64,
                        )
                        crop_widths = self.accelerator.gather(local_crop_width).reshape(-1)
                        if not bool(torch.all(crop_widths == crop_widths[0]).item()):
                            raise RuntimeError(
                                "batch-shared center-crop width diverged across ranks: "
                                f"{crop_widths.detach().cpu().tolist()}"
                            )
                if _PROF and _optimizer_step_wall0 is None:
                    # Preserve the first microstep's data-fetch start so GA>1
                    # summaries cover the complete optimizer step.
                    _optimizer_step_wall0 = _step_wall0
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self._set_data_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self._unwrap_model()
                progress_model = self._unwrap_model()
                set_training_progress = getattr(
                    progress_model, "set_training_progress", None
                )
                if callable(set_training_progress):
                    set_training_progress(self.global_step, self.max_steps)
                self._apply_optimizer_bridge_runtime_state()

                if _PROF:
                    # Accelerate moves batches to CUDA from DataLoader.__next__.
                    # With non_blocking=True the CPU-side next() timer only sees
                    # the enqueue; synchronize here to attribute the actual copy
                    # latency separately instead of silently charging it to the
                    # following forward pass.
                    _th2d0 = time.perf_counter()
                    torch.cuda.synchronize()
                    _h2d_elapsed = time.perf_counter() - _th2d0
                    _prof["h2d"] += _h2d_elapsed
                    # FSDP's first forward all-gather is also an implicit
                    # cross-rank barrier.  Without an explicit data-ready
                    # measurement, a slow decoder rank is charged to another
                    # rank's forward timer.  Gather the ready latency here so
                    # the diagnostic can separate input stragglers from model
                    # compute/collectives.  This is enabled only by STEP_PROFILE.
                    _ready_local = torch.tensor(
                        [_data_elapsed + _h2d_elapsed],
                        device=self.accelerator.device,
                        dtype=torch.float64,
                    )
                    _tready0 = time.perf_counter()
                    _ready_by_rank = self.accelerator.gather(_ready_local).reshape(-1)
                    _pre_forward_wait = time.perf_counter() - _tready0
                    _ready_max = float(_ready_by_rank.max().item())
                    _ready_min = float(_ready_by_rank.min().item())
                    _prof["ready_max"] += _ready_max
                    _prof["ready_spread"] += _ready_max - _ready_min
                    _forward_start_event = torch.cuda.Event(enable_timing=True)
                    _forward_end_event = torch.cuda.Event(enable_timing=True)
                    _forward_start_event.record()
                _tf0 = time.perf_counter()
                with self.accelerator.autocast():
                    if self._manual_fsdp:
                        # Route through FSDP.forward (== FastWAM.forward ==
                        # training_loss) so FSDP all-gathers sharded params on entry.
                        loss, loss_dict = self.model(sample)
                    else:
                        loss, loss_dict = train_model.training_loss(sample)
                if _FAIL_ON_NONFINITE:
                    loss_finite = torch.isfinite(loss.detach()).all().to(
                        device=loss.device, dtype=torch.int32
                    ).reshape(1)
                    loss_finite_by_rank = self.accelerator.gather(loss_finite)
                    if not bool(loss_finite_by_rank.bool().all().item()):
                        raise FloatingPointError(
                            "Non-finite training loss before backward at "
                            f"global_step={self.global_step}, epoch={self.epoch}, "
                            f"finite_by_rank={loss_finite_by_rank.detach().cpu().tolist()}, "
                            f"local_loss={float(loss.detach().float().item())}"
                        )
                if _PROF:
                    _forward_end_event.record()
                    _forward_end_event.synchronize()
                    _forward_cuda_ms = float(
                        _forward_start_event.elapsed_time(_forward_end_event)
                    )
                    torch.cuda.synchronize()
                _forward_wall = time.perf_counter() - _tf0
                _prof["fwd"] += _forward_wall
                _tb0 = time.perf_counter()
                if _PROF:
                    _backward_start_event = torch.cuda.Event(enable_timing=True)
                    _backward_end_event = torch.cuda.Event(enable_timing=True)
                    _backward_start_event.record()
                self.accelerator.backward(loss)
                if _SYNC_AFTER_BACKWARD:
                    # Selective activation recomputation inside the manually
                    # wrapped FSDP model can leave work on auxiliary CUDA
                    # streams.  Join them before FSDP reads/clips gradients.
                    torch.cuda.synchronize()
                if _PROF:
                    _backward_end_event.record()
                    _backward_end_event.synchronize()
                    _backward_cuda_ms = float(
                        _backward_start_event.elapsed_time(_backward_end_event)
                    )
                    torch.cuda.synchronize()
                _backward_wall = time.perf_counter() - _tb0
                _prof["bwd"] += _backward_wall
                _prof["n"] += 1
                if _PROF:
                    _optimizer_step_profile["data_fetch_ms"] += (
                        _data_elapsed * 1000.0
                    )
                    _optimizer_step_profile["h2d_ms"] += _h2d_elapsed * 1000.0
                    _optimizer_step_profile["pre_forward_wait_ms"] += (
                        _pre_forward_wait * 1000.0
                    )
                    _optimizer_step_profile["forward_wall_ms"] += (
                        _forward_wall * 1000.0
                    )
                    _optimizer_step_profile["forward_cuda_ms"] += _forward_cuda_ms
                    _optimizer_step_profile["backward_wall_ms"] += (
                        _backward_wall * 1000.0
                    )
                    _optimizer_step_profile["backward_cuda_ms"] += _backward_cuda_ms
                _topt0 = time.perf_counter()
                if _PROF:
                    _optimizer_start_event = torch.cuda.Event(enable_timing=True)
                    _optimizer_end_event = torch.cuda.Event(enable_timing=True)
                    _optimizer_start_event.record()

                if self.accelerator.sync_gradients:
                    if self._is_deepspeed:
                        # Accelerate's DeepSpeedEngineWrapper calls engine.step()
                        # from backward() at accumulation boundaries.  DeepSpeed
                        # has therefore already clipped/reduced gradients, stepped
                        # Adam and the LR scheduler, and cleared gradients here.
                        # This call only retrieves DeepSpeed's recorded global norm.
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    elif self._manual_fsdp:
                        # FSDP reduces the grad norm across shards internally.
                        grad_norm = getattr(self.model, "_orig_mod", self.model).clip_grad_norm_(self.max_grad_norm)
                    else:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    grad_norm_tensor = torch.as_tensor(
                        grad_norm, device=loss.device, dtype=torch.float32
                    ).detach().reshape(1)
                    if _FAIL_ON_NONFINITE:
                        grad_finite_by_rank = self.accelerator.gather(
                            torch.isfinite(grad_norm_tensor).to(dtype=torch.int32)
                        )
                        if not bool(grad_finite_by_rank.bool().all().item()):
                            self.optimizer.zero_grad(set_to_none=True)
                            raise FloatingPointError(
                                "Non-finite gradient norm before optimizer step at "
                                f"global_step={self.global_step}, epoch={self.epoch}, "
                                f"finite_by_rank={grad_finite_by_rank.detach().cpu().tolist()}, "
                                f"local_grad_norm={float(grad_norm_tensor.item())}"
                            )
                    if self._is_deepspeed:
                        # Accelerate only hands DeepSpeed schedulers that are in
                        # DeepSpeed's native VALID_LR_SCHEDULES set to
                        # deepspeed.initialize().  Our SequentialLR remains an
                        # AcceleratedScheduler, so engine.step() cannot advance it.
                        # Calling the wrapper is safe in both cases: a native
                        # DeepSpeedSchedulerWrapper implements step() as a no-op,
                        # while AcceleratedScheduler advances the external schedule.
                        self.scheduler.step()
                    else:
                        self.optimizer.step()
                        if not self.accelerator.optimizer_step_was_skipped:
                            self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    if _PROF:
                        _optimizer_end_event.record()
                        _optimizer_end_event.synchronize()
                        _optimizer_cuda_ms = float(
                            _optimizer_start_event.elapsed_time(_optimizer_end_event)
                        )
                        torch.cuda.synchronize()
                    _optimizer_wall = time.perf_counter() - _topt0
                    _prof["opt"] += _optimizer_wall
                    _tcollect0 = time.perf_counter()
                    self.global_step += 1
                    self._verify_weights_after_first_step()
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    globally_reduced_source_metrics = {
                        "loss_video_ego",
                        "loss_video_robot",
                        "loss_action_ego",
                        "loss_action_robot",
                    }
                    for key, value in loss_dict.items():
                        if key in globally_reduced_source_metrics:
                            # FastWAM._source_balanced_loss already all-reduces
                            # these means, so every rank holds the same value.
                            global_loss_metrics[key] = float(value)
                        else:
                            metric_tensor = torch.tensor(
                                float(value),
                                device=loss.device,
                                dtype=torch.float32,
                            ).reshape(1)
                            global_loss_metrics[key] = float(
                                self.accelerator.gather(metric_tensor).mean().item()
                            )
                    if "source_is_ego" in sample:
                        source_fraction = sample["source_is_ego"].float().mean().reshape(1)
                        global_loss_metrics["ego_replay_fraction"] = float(
                            self.accelerator.gather(source_fraction).mean().item()
                        )
                        if self.accelerator.is_main_process:
                            source_value = global_loss_metrics["ego_replay_fraction"]
                            _source_log_window["steps"] += 1
                            _source_log_window["ego_fraction"] += source_value
                            # Mixed batches expose source-balanced means directly;
                            # use those metrics instead of attributing the total
                            # batch loss by sample fraction. Fall back to the old
                            # estimate for checkpoints/configs without the new
                            # model metrics.
                            if "loss_action_ego" in global_loss_metrics:
                                _source_log_window["ego_steps"] += float(source_value > 0.0)
                                _source_log_window["ego_loss_action"] += global_loss_metrics["loss_action_ego"]
                            else:
                                _source_log_window["ego_steps"] += source_value
                                _source_log_window["ego_loss_action"] += source_value * global_loss_metrics.get("loss_action", 0.0)
                            if "loss_action_robot" in global_loss_metrics:
                                _source_log_window["robot_steps"] += float(source_value < 1.0)
                                _source_log_window["robot_loss_action"] += global_loss_metrics["loss_action_robot"]
                            else:
                                robot_weight = 1.0 - source_value
                                _source_log_window["robot_steps"] += robot_weight
                                _source_log_window["robot_loss_action"] += robot_weight * global_loss_metrics.get("loss_action", 0.0)
                            if "loss_video_ego" in global_loss_metrics:
                                _source_log_window["ego_loss_video"] += global_loss_metrics["loss_video_ego"]
                            else:
                                _source_log_window["ego_loss_video"] += source_value * global_loss_metrics.get("loss_video", 0.0)
                            if "loss_video_robot" in global_loss_metrics:
                                _source_log_window["robot_loss_video"] += global_loss_metrics["loss_video_robot"]
                            else:
                                _source_log_window["robot_loss_video"] += (1.0 - source_value) * global_loss_metrics.get("loss_video", 0.0)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                    _metric_collective_wall = time.perf_counter() - _tcollect0
                    if _PROF:
                        _tbarrier0 = time.perf_counter()
                        self.accelerator.wait_for_everyone()
                        _post_step_barrier = time.perf_counter() - _tbarrier0

                        local_phase = torch.tensor(
                            [
                                _optimizer_step_profile["data_fetch_ms"],
                                _optimizer_step_profile["h2d_ms"],
                                _optimizer_step_profile["pre_forward_wait_ms"],
                                _optimizer_step_profile["forward_wall_ms"],
                                _optimizer_step_profile["forward_cuda_ms"],
                                _optimizer_step_profile["backward_wall_ms"],
                                _optimizer_step_profile["backward_cuda_ms"],
                                _optimizer_wall * 1000.0,
                                _optimizer_cuda_ms,
                                _metric_collective_wall * 1000.0,
                                _post_step_barrier * 1000.0,
                                (
                                    time.perf_counter()
                                    - (
                                        _optimizer_step_wall0
                                        if _optimizer_step_wall0 is not None
                                        else _step_wall0
                                    )
                                )
                                * 1000.0,
                                float(torch.cuda.max_memory_allocated()),
                                float(torch.cuda.max_memory_reserved()),
                            ],
                            device=self.accelerator.device,
                            dtype=torch.float64,
                        ).reshape(1, -1)
                        phase_by_rank = self.accelerator.gather(local_phase)
                        phase_names = (
                            "data_fetch_ms",
                            "h2d_ms",
                            "pre_forward_wait_ms",
                            "forward_wall_ms",
                            "forward_cuda_ms",
                            "backward_wall_ms",
                            "backward_cuda_ms",
                            "optimizer_wall_ms",
                            "optimizer_cuda_ms",
                            "metric_collective_ms",
                            "post_step_barrier_ms",
                            "total_wall_ms",
                            "peak_allocated_bytes",
                            "peak_reserved_bytes",
                        )
                        local_payload = {
                            "schema": "fastwam_rank_phase_v1",
                            "step": int(self.global_step),
                            "epoch": int(self.epoch),
                            "rank": int(self.accelerator.process_index),
                            **{
                                name: float(local_phase[0, index].item())
                                for index, name in enumerate(phase_names)
                            },
                            "loss": global_loss,
                            "grad_norm": global_grad_norm,
                            "cache_hits_cumulative": int(_merged_cache_hits),
                            "cache_misses_cumulative": int(_merged_cache_misses),
                            "cache_hits_step": int(cache_hits),
                            "cache_misses_step": int(cache_misses),
                            "latent_cache_hits_cumulative": int(
                                _latent_cache_hits
                            ),
                            "latent_cache_misses_cumulative": int(
                                _latent_cache_misses
                            ),
                            "latent_cache_hits_step": int(latent_cache_hits),
                            "latent_cache_misses_step": int(
                                latent_cache_misses
                            ),
                            "pyav_calls_cumulative": int(_pyav_calls),
                            "sample_shapes": {
                                key: list(value.shape)
                                for key, value in sample.items()
                                if key
                                in {
                                    "video",
                                    "video_latents",
                                    "pixel_values",
                                    "action",
                                    "proprio",
                                    "context",
                                }
                                and isinstance(value, torch.Tensor)
                            },
                        }
                        local_payload["forward_wall_minus_cuda_ms"] = (
                            local_payload["forward_wall_ms"]
                            - local_payload["forward_cuda_ms"]
                        )
                        _append_phase(_rank_phase_path, local_payload)
                        if self.accelerator.is_main_process:
                            summaries = {}
                            for index, name in enumerate(phase_names):
                                values = phase_by_rank[:, index].detach().cpu()
                                ordered = torch.sort(values).values
                                p95_index = max(
                                    0,
                                    min(
                                        len(ordered) - 1,
                                        ceil(0.95 * len(ordered)) - 1,
                                    ),
                                )
                                summaries[name] = {
                                    "min": float(ordered[0].item()),
                                    "median": float(ordered.median().item()),
                                    "max": float(ordered[-1].item()),
                                    "p95": float(ordered[p95_index].item()),
                                    "straggler_rank": int(
                                        torch.argmax(values).item()
                                    ),
                                }
                            forward_gap = (
                                phase_by_rank[:, phase_names.index("forward_wall_ms")]
                                - phase_by_rank[
                                    :, phase_names.index("forward_cuda_ms")
                                ]
                            )
                            ordered_gap = torch.sort(forward_gap.cpu()).values
                            summaries["forward_wall_minus_cuda_ms"] = {
                                "min": float(ordered_gap[0].item()),
                                "median": float(ordered_gap.median().item()),
                                "max": float(ordered_gap[-1].item()),
                                "p95": float(ordered_gap[p95_index].item()),
                                "straggler_rank": int(
                                    torch.argmax(forward_gap).item()
                                ),
                            }
                            _append_phase(
                                _rank_phase_summary_path,
                                {
                                    "schema": "fastwam_rank_phase_summary_v1",
                                    "step": int(self.global_step),
                                    "epoch": int(self.epoch),
                                    "world_size": int(
                                        self.accelerator.num_processes
                                    ),
                                    "phases": summaries,
                                    "loss": global_loss,
                                    "grad_norm": global_grad_norm,
                                    "cache_hits_cumulative": int(_merged_cache_hits),
                                    "cache_misses_cumulative": int(
                                        _merged_cache_misses
                                    ),
                                    "latent_cache_hits_cumulative": int(
                                        _latent_cache_hits
                                    ),
                                    "latent_cache_misses_cumulative": int(
                                        _latent_cache_misses
                                    ),
                                    "pyav_calls_cumulative": int(_pyav_calls),
                                    "sample_shapes": {
                                        key: list(value.shape)
                                        for key, value in sample.items()
                                        if key
                                        in {
                                            "video",
                                            "video_latents",
                                            "pixel_values",
                                            "action",
                                            "proprio",
                                            "context",
                                        }
                                        and isinstance(value, torch.Tensor)
                                    },
                                    "compile_counters": {
                                        group: {
                                            str(key): int(value)
                                            for key, value in counters.items()
                                        }
                                        for group, counters in (
                                            torch._dynamo.utils.counters.items()
                                        )
                                        if group
                                        in {
                                            "frames",
                                            "stats",
                                            "inductor",
                                            "aot_autograd",
                                        }
                                    },
                                },
                            )
                        _optimizer_step_wall0 = None
                        for phase_name in _optimizer_step_profile:
                            _optimizer_step_profile[phase_name] = 0.0

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and (self.global_step <= int(self.cfg.get("log_first_steps", 0)) or self.global_step % self.log_every == 0) and self.accelerator.is_main_process:
                        if _source_log_window["steps"] > 0:
                            window_steps = float(_source_log_window["steps"])
                            ego_fraction = float(_source_log_window["ego_fraction"])
                            ego_steps = float(_source_log_window["ego_steps"])
                            robot_steps = float(_source_log_window["robot_steps"])
                            global_loss_metrics["ego_replay_fraction_window"] = (
                                ego_fraction / window_steps
                            )
                            if ego_steps > 0.0:
                                global_loss_metrics["ego_loss_action_window"] = (
                                    _source_log_window["ego_loss_action"] / ego_steps
                                )
                                global_loss_metrics["ego_loss_video_window"] = (
                                    _source_log_window["ego_loss_video"] / ego_steps
                                )
                            if robot_steps > 0.0:
                                global_loss_metrics["robot_loss_action_window"] = (
                                    _source_log_window["robot_loss_action"] / robot_steps
                                )
                                global_loss_metrics["robot_loss_video_window"] = (
                                    _source_log_window["robot_loss_video"] / robot_steps
                                )
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if batch_crop_width is not None:
                            description += f"crop_width={batch_crop_width} "
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        _n = max(_prof["n"], 1)
                        description += (
                            "prof_ms(data=%.0f h2d=%.0f ready_max=%.0f "
                            "ready_spread=%.0f fwd=%.0f bwd=%.0f opt=%.0f) "
                            % (
                                1000 * _prof["data"] / _n,
                                1000 * _prof["h2d"] / _n,
                                1000 * _prof["ready_max"] / _n,
                                1000 * _prof["ready_spread"] / _n,
                                1000 * _prof["fwd"] / _n,
                                1000 * _prof["bwd"] / _n,
                                1000 * _prof["opt"] / _n,
                            )
                        )
                        description += (
                            "cache_hits=%d cache_misses=%d "
                            "latent_hits=%d latent_misses=%d "
                            % (
                                _merged_cache_hits,
                                _merged_cache_misses,
                                _latent_cache_hits,
                                _latent_cache_misses,
                            )
                        )
                        _prof = {
                            "data": 0.0,
                            "h2d": 0.0,
                            "ready_max": 0.0,
                            "ready_spread": 0.0,
                            "fwd": 0.0,
                            "bwd": 0.0,
                            "opt": 0.0,
                            "n": 0,
                        }
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)
                        if os.environ.get("FASTWAM_PLAIN_TRAIN_LOG", "0") == "1":
                            print(description, flush=True)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        if batch_crop_width is not None:
                            wandb_payload["train/batch_crop_width"] = int(
                                batch_crop_width
                            )
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)
                        for key in _source_log_window:
                            _source_log_window[key] = 0

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    ckpt_info = None
                    if _checkpoint_step_is_scheduled(
                        self.global_step,
                        save_every=self.save_every,
                        checkpoint_steps=self.checkpoint_steps,
                    ):
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if (
                        _diagnostic_stop_step is not None
                        and self.global_step >= _diagnostic_stop_step
                    ):
                        if ckpt_info is None and not _SKIP_DIAGNOSTIC_CHECKPOINT:
                            ckpt_info = self.save_checkpoint()
                        if _SKIP_DIAGNOSTIC_CHECKPOINT:
                            self.accelerator.wait_for_everyone()
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[done] diagnostic stop reached step=%d checkpoint=%s",
                                    self.global_step,
                                    "periodic" if ckpt_info is not None else "skipped",
                                )
                            return
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] diagnostic stop reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

                    if self.global_step >= self.max_steps:
                        # max_steps is commonly divisible by save_every.  Reuse
                        # the periodic checkpoint instead of gathering and
                        # overwriting the same full FSDP weights twice.
                        if ckpt_info is None:
                            ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
