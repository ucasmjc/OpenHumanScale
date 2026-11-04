# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from humanwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


class ActionDiT(nn.Module):
    # Internal route marker used when a legacy dual-adapter checkpoint is
    # continued with a sample-mixed Ego/robot batch.
    LEGACY_EGO_ROUTE = "__ego__"
    ACTION_BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "head.",
        "ego_action_encoder.",
        "ego_head.",
        "source_action_encoders.",
        "source_heads.",
    )
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
        separate_ego_action_tokenizer: bool = False,
        action_tokenizer_sources: Optional[list[str]] = None,
        default_action_tokenizer_source: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.separate_ego_action_tokenizer = bool(separate_ego_action_tokenizer)
        if self.separate_ego_action_tokenizer:
            # Robot and Ego share the 30-block ActionDiT trunk, but keep separate
            # continuous-space I/O adapters.  The two sources both happen to be
            # 20D after preprocessing, yet their embodiment, temporal statistics,
            # gripper occupancy, and per-dataset normalization are not identical.
            self.ego_action_encoder = nn.Linear(action_dim, hidden_dim)
            self.ego_head = nn.Linear(hidden_dim, action_dim)
            self.initialize_ego_action_tokenizer_from_default()
        else:
            self.ego_action_encoder = None
            self.ego_head = None

        raw_sources = action_tokenizer_sources or []
        self.action_tokenizer_sources = tuple(str(source) for source in raw_sources)
        if len(set(self.action_tokenizer_sources)) != len(self.action_tokenizer_sources):
            raise ValueError(
                "action_tokenizer_sources must not contain duplicates: "
                f"{self.action_tokenizer_sources}"
            )
        for source in self.action_tokenizer_sources:
            if not source or not source.isidentifier():
                raise ValueError(
                    "Action tokenizer source names must be non-empty Python-style "
                    f"identifiers, got {source!r}"
                )
        if self.action_tokenizer_sources:
            if default_action_tokenizer_source is None:
                raise ValueError(
                    "default_action_tokenizer_source is required when "
                    "action_tokenizer_sources are configured"
                )
            default_action_tokenizer_source = str(default_action_tokenizer_source)
            if default_action_tokenizer_source not in self.action_tokenizer_sources:
                raise ValueError(
                    "default_action_tokenizer_source must be one of "
                    f"action_tokenizer_sources, got {default_action_tokenizer_source!r}"
                )
        elif default_action_tokenizer_source is not None:
            raise ValueError(
                "default_action_tokenizer_source requires action_tokenizer_sources"
            )
        self.default_action_tokenizer_source = default_action_tokenizer_source
        extra_sources = [
            source
            for source in self.action_tokenizer_sources
            if source != self.default_action_tokenizer_source
        ]
        self.source_action_encoders = nn.ModuleDict(
            {source: nn.Linear(action_dim, hidden_dim) for source in extra_sources}
        )
        self.source_heads = nn.ModuleDict(
            {source: nn.Linear(hidden_dim, action_dim) for source in extra_sources}
        )
        self.initialize_source_action_tokenizers_from_default()
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    @torch.no_grad()
    def initialize_ego_action_tokenizer_from_default(self) -> None:
        """Initialize the Ego I/O adapter from the default robot adapter.

        Ego-pretraining checkpoints produced before the dual-interface change only
        contain ``action_encoder`` and ``head``.  Starting both interfaces from those
        learned weights preserves the pretrained action mapping, after which replay
        and robot batches can specialize their own adapters without direct I/O
        interference.
        """
        if not self.separate_ego_action_tokenizer:
            return
        if self.ego_action_encoder is None or self.ego_head is None:
            raise RuntimeError(
                "Separate Ego action tokenizer is enabled but its modules are missing."
            )
        self.ego_action_encoder.load_state_dict(
            self.action_encoder.state_dict(), strict=True
        )
        self.ego_head.load_state_dict(self.head.state_dict(), strict=True)

    @torch.no_grad()
    def initialize_source_action_tokenizers_from_default(self) -> None:
        """Clone the default I/O weights into every additional source adapter."""
        for source in self.source_action_encoders:
            self.source_action_encoders[source].load_state_dict(
                self.action_encoder.state_dict(), strict=True
            )
            self.source_heads[source].load_state_dict(
                self.head.state_dict(), strict=True
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Backward-compatible weights-only resume: old checkpoints have one action
        # tokenizer.  Populate the new Ego adapter from that checkpoint's default
        # adapter before PyTorch descends into the Linear child modules.  New dual-
        # interface checkpoints already contain the Ego keys and are left untouched.
        if self.separate_ego_action_tokenizer:
            aliases = {
                "ego_action_encoder.weight": "action_encoder.weight",
                "ego_action_encoder.bias": "action_encoder.bias",
                "ego_head.weight": "head.weight",
                "ego_head.bias": "head.bias",
            }
            for target_suffix, source_suffix in aliases.items():
                target_key = prefix + target_suffix
                source_key = prefix + source_suffix
                if target_key not in state_dict and source_key in state_dict:
                    state_dict[target_key] = state_dict[source_key]
        for source in self.source_action_encoders:
            aliases = {
                f"source_action_encoders.{source}.weight": "action_encoder.weight",
                f"source_action_encoders.{source}.bias": "action_encoder.bias",
                f"source_heads.{source}.weight": "head.weight",
                f"source_heads.{source}.bias": "head.bias",
            }
            for target_suffix, source_suffix in aliases.items():
                target_key = prefix + target_suffix
                source_key = prefix + source_suffix
                if target_key not in state_dict and source_key in state_dict:
                    state_dict[target_key] = state_dict[source_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _action_tokenizer_modules(
        self,
        action_tokenizer_source: Optional[str] = None,
        use_ego_action_tokenizer: bool = False,
    ) -> tuple[nn.Linear, nn.Linear]:
        if action_tokenizer_source is not None:
            source = str(action_tokenizer_source)
            if source == self.LEGACY_EGO_ROUTE:
                if self.ego_action_encoder is None or self.ego_head is None:
                    raise RuntimeError(
                        "Legacy Ego action-tokenizer route requested but the Ego adapter is disabled."
                    )
                return self.ego_action_encoder, self.ego_head
            if source not in self.action_tokenizer_sources:
                raise ValueError(
                    f"Unknown action tokenizer source {source!r}; configured="
                    f"{self.action_tokenizer_sources}"
                )
            if source == self.default_action_tokenizer_source:
                return self.action_encoder, self.head
            return self.source_action_encoders[source], self.source_heads[source]

        if use_ego_action_tokenizer:
            if not self.separate_ego_action_tokenizer:
                return self.action_encoder, self.head
            if self.ego_action_encoder is None or self.ego_head is None:
                raise RuntimeError(
                    "Separate Ego action tokenizer is enabled but its modules are missing."
                )
            return self.ego_action_encoder, self.ego_head
        return self.action_encoder, self.head

    @staticmethod
    def _normalize_action_tokenizer_route(
        action_tokenizer_source,
        batch_size: int,
    ) -> tuple[Optional[str], ...]:
        if action_tokenizer_source is None:
            return (None,) * batch_size
        if isinstance(action_tokenizer_source, str):
            return (action_tokenizer_source,) * batch_size
        if isinstance(action_tokenizer_source, (list, tuple)):
            values = tuple(
                None if value is None else str(value)
                for value in action_tokenizer_source
            )
            if len(values) == 1 and batch_size > 1:
                values = values * batch_size
            if len(values) != batch_size:
                raise ValueError(
                    "action_tokenizer_source must be scalar or have one value per "
                    f"sample; got {len(values)} values for batch_size={batch_size}"
                )
            return values
        raise TypeError(
            "action_tokenizer_source must be None, a string, or a sequence of strings; "
            f"got {type(action_tokenizer_source).__name__}"
        )

    def _encode_action_tokens(
        self,
        action_tokens: torch.Tensor,
        route: tuple[Optional[str], ...],
        use_ego_action_tokenizer: bool,
    ) -> torch.Tensor:
        unique_routes = tuple(dict.fromkeys(route))
        if len(unique_routes) == 1:
            encoder, _ = self._action_tokenizer_modules(
                unique_routes[0], use_ego_action_tokenizer
            )
            return encoder(action_tokens)

        encoded = action_tokens.new_empty(
            action_tokens.shape[0], action_tokens.shape[1], self.hidden_dim
        )
        for source in unique_routes:
            indices = torch.tensor(
                [i for i, value in enumerate(route) if value == source],
                dtype=torch.long,
                device=action_tokens.device,
            )
            encoder, _ = self._action_tokenizer_modules(source, False)
            source_tokens = encoder(action_tokens.index_select(0, indices))
            encoded.index_copy_(0, indices, source_tokens)
        return encoded

    def _decode_action_tokens(
        self,
        tokens: torch.Tensor,
        route: tuple[Optional[str], ...],
        use_ego_action_tokenizer: bool,
    ) -> torch.Tensor:
        unique_routes = tuple(dict.fromkeys(route))
        if len(unique_routes) == 1:
            _, head = self._action_tokenizer_modules(
                unique_routes[0], use_ego_action_tokenizer
            )
            return head(tokens)

        output = tokens.new_empty(tokens.shape[0], tokens.shape[1], self.action_dim)
        for source in unique_routes:
            indices = torch.tensor(
                [i for i, value in enumerate(route) if value == source],
                dtype=torch.long,
                device=tokens.device,
            )
            _, head = self._action_tokenizer_modules(source, False)
            source_output = head(tokens.index_select(0, indices))
            output.index_copy_(0, indices, source_output)
        return output

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )

        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        context_embedding_cache: Optional[torch.Tensor] = None,
        use_ego_action_tokenizer: bool = False,
        action_tokenizer_source=None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        if use_ego_action_tokenizer and action_tokenizer_source is not None:
            raise ValueError(
                "Legacy use_ego_action_tokenizer and action_tokenizer_source routing "
                "cannot be enabled together."
            )
        action_tokenizer_route = self._normalize_action_tokenizer_route(
            action_tokenizer_source, batch_size
        )
        tokens = self._encode_action_tokens(
            action_tokens,
            action_tokenizer_route,
            use_ego_action_tokenizer,
        )
        context_emb = context_embedding_cache if context_embedding_cache is not None else self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "use_ego_action_tokenizer": bool(use_ego_action_tokenizer),
                "action_tokenizer_route": action_tokenizer_route,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        use_ego_action_tokenizer = bool(
            pre_state.get("meta", {}).get("use_ego_action_tokenizer", False)
        )
        route = pre_state.get("meta", {}).get("action_tokenizer_route")
        if route is None:
            route = (None,) * tokens.shape[0]
        return self._decode_action_tokens(tokens, route, use_ego_action_tokenizer)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        use_ego_action_tokenizer: bool = False,
        action_tokenizer_source=None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            use_ego_action_tokenizer=use_ego_action_tokenizer,
            action_tokenizer_source=action_tokenizer_source,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)
