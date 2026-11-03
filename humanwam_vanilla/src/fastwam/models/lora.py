"""Small, dependency-free LoRA utilities for FastWAM's MoT experts.

The implementation deliberately subclasses :class:`torch.nn.Linear`.  The
pretrained ``weight`` and ``bias`` therefore keep their historical state-dict
keys when a projection is adapted; only ``lora_A`` and ``lora_B`` are new.
This is important for loading a normal FastWAM checkpoint before a LoRA
continuation and for exporting a merged checkpoint for the existing runtime.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_MOT_LORA_TARGETS = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "ffn.0",
    "ffn.2",
)
LORA_ADAPTER_FORMAT = "fastwam_lora_adapter_v1"


def checkpoint_sha256(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    """Return the SHA-256 of a local checkpoint without loading it into memory."""

    checkpoint = Path(path)
    if int(chunk_bytes) <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}.")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        while chunk := handle.read(int(chunk_bytes)):
            digest.update(chunk)
    return digest.hexdigest()


def load_lora_adapter_payload(path: str | Path) -> dict[str, object]:
    """Load and minimally validate a FastWAM adapter payload.

    This deliberately does not accept an arbitrary state dict.  Keeping the
    production envelope mandatory prevents a small adapter from being applied
    to an unrecorded base model or a different LoRA topology by accident.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"LoRA adapter must contain a mapping, got {type(payload).__name__}.")
    payload = dict(payload)
    if payload.get("format") != LORA_ADAPTER_FORMAT:
        raise ValueError(
            f"Unsupported LoRA adapter format: {payload.get('format')!r}."
        )
    if not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("LoRA adapter payload is missing a mapping `state_dict`.")
    if not isinstance(payload.get("metadata"), Mapping):
        raise ValueError("LoRA adapter payload is missing a mapping `metadata`.")
    return payload


def is_lora_adapter_checkpoint(path: str | Path) -> bool:
    """Cheaply identify trainer-produced adapters by their explicit suffix."""

    return str(path).endswith(".adapter.pt")


@dataclass(frozen=True)
class LoRAInjectionReport:
    module_names: tuple[str, ...]
    trainable_parameters: int
    base_parameters: int

    @property
    def module_count(self) -> int:
        return len(self.module_names)


class LoRALinear(nn.Linear):
    """A state-dict-compatible ``nn.Linear`` with a low-rank residual.

    ``lora_B`` is initialized to zero, so injection is function preserving.
    The pretrained ``weight`` and ``bias`` objects are reused by
    :meth:`from_linear` instead of copied.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        if int(rank) <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError(f"LoRA alpha must be finite and positive, got {alpha}.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}.")

        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=device, dtype=dtype)
        )
        self.merged = False
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        # Same fan-in initialization convention as the common PEFT LoRA path;
        # B=0 is the property that makes the injected model exactly identical.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        if isinstance(linear, LoRALinear):
            raise ValueError("Cannot inject LoRA into an existing LoRALinear.")
        adapted = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # Preserve Parameter identity as well as values.  This matters if a
        # caller injects before constructing an optimizer (the trainer does).
        adapted.weight = linear.weight
        adapted.bias = linear.bias
        adapted.train(linear.training)
        return adapted

    def lora_delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.weight.add_(
            self.lora_delta_weight().to(
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.weight.sub_(
            self.lora_delta_weight().to(
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.merged = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        if self.merged:
            return output
        low_rank = F.linear(F.linear(self.lora_dropout(input), self.lora_A), self.lora_B)
        return output + low_rank * self.scaling


def _get_child(module: nn.Module, path: str) -> nn.Module:
    if not path:
        return module
    current = module
    for part in path.split("."):
        if part.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential)):
            current = current[int(part)]
        else:
            try:
                current = getattr(current, part)
            except AttributeError as error:
                raise ValueError(
                    f"Module path {path!r} does not exist at component {part!r}."
                ) from error
        if not isinstance(current, nn.Module):
            raise ValueError(f"Module path {path!r} resolves to non-module {current!r}.")
    return current


def _replace_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def inject_mot_lora(
    mot: nn.Module,
    *,
    experts: Sequence[str] = ("video", "action"),
    target_modules: Sequence[str] = DEFAULT_MOT_LORA_TARGETS,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRAInjectionReport:
    """Inject LoRA into selected projections of every block in MoT experts.

    ``target_modules`` contains paths relative to each transformer block, for
    example ``self_attn.q`` or ``ffn.0``.  Missing experts/targets and duplicate
    requests fail closed so a typo cannot silently change an ablation.
    """

    if not experts:
        raise ValueError("At least one LoRA expert is required.")
    if len(set(experts)) != len(experts):
        raise ValueError(f"Duplicate LoRA experts are not allowed: {list(experts)!r}.")
    if not target_modules:
        raise ValueError("At least one LoRA target module is required.")
    if len(set(target_modules)) != len(target_modules):
        raise ValueError(
            f"Duplicate LoRA target modules are not allowed: {list(target_modules)!r}."
        )
    mixtures = getattr(mot, "mixtures", None)
    if not isinstance(mixtures, nn.ModuleDict):
        raise ValueError("LoRA injection expects MoT.mixtures to be an nn.ModuleDict.")

    injected: list[str] = []
    trainable_parameters = 0
    base_parameters = 0
    for expert_name in experts:
        if expert_name not in mixtures:
            raise ValueError(
                f"Unknown LoRA expert {expert_name!r}; available={list(mixtures.keys())!r}."
            )
        expert = mixtures[expert_name]
        blocks = getattr(expert, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
            raise ValueError(f"MoT expert {expert_name!r} has no non-empty ModuleList blocks.")
        for block_index, block in enumerate(blocks):
            for target_path in target_modules:
                parent_path, separator, child_name = str(target_path).rpartition(".")
                if not separator:
                    parent_path, child_name = "", str(target_path)
                parent = _get_child(block, parent_path)
                child = _get_child(parent, child_name)
                if isinstance(child, LoRALinear):
                    raise ValueError(
                        "LoRA target was already injected: "
                        f"mixtures.{expert_name}.blocks.{block_index}.{target_path}"
                    )
                if not isinstance(child, nn.Linear):
                    raise ValueError(
                        "LoRA targets must be nn.Linear modules; "
                        f"{expert_name}.blocks.{block_index}.{target_path} is {type(child).__name__}."
                    )
                adapted = LoRALinear.from_linear(
                    child,
                    rank=int(rank),
                    alpha=float(alpha),
                    dropout=float(dropout),
                )
                _replace_child(parent, child_name, adapted)
                full_name = (
                    f"mixtures.{expert_name}.blocks.{block_index}.{target_path}"
                )
                injected.append(full_name)
                trainable_parameters += adapted.lora_A.numel() + adapted.lora_B.numel()
                base_parameters += adapted.weight.numel()
                if adapted.bias is not None:
                    base_parameters += adapted.bias.numel()

    return LoRAInjectionReport(
        module_names=tuple(injected),
        trainable_parameters=int(trainable_parameters),
        base_parameters=int(base_parameters),
    )


def inject_lora_adapters(
    mot: nn.Module,
    *,
    experts: Sequence[str] = ("video", "action"),
    target_modules: Sequence[str] = DEFAULT_MOT_LORA_TARGETS,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRAInjectionReport:
    """Stable public alias for :func:`inject_mot_lora`."""

    return inject_mot_lora(
        mot,
        experts=experts,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


def iter_lora_modules(module: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            yield name, child


def set_lora_trainable(
    model: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
) -> tuple[str, ...]:
    """Freeze the model, then enable LoRA tensors and exact module allowlist paths."""

    model.requires_grad_(False)
    trainable_names: set[str] = set()
    for module_name, adapted in iter_lora_modules(model):
        adapted.lora_A.requires_grad_(True)
        adapted.lora_B.requires_grad_(True)
        trainable_names.add(f"{module_name}.lora_A")
        trainable_names.add(f"{module_name}.lora_B")

    if not trainable_names:
        raise ValueError("LoRA train mode requested, but the model has no LoRALinear modules.")
    if len(set(trainable_modules)) != len(trainable_modules):
        raise ValueError(
            f"Duplicate trainable module paths are not allowed: {list(trainable_modules)!r}."
        )
    if any(not str(path).strip() for path in trainable_modules):
        raise ValueError("Trainable module paths must be non-empty strings.")
    for module_path in trainable_modules:
        allowed = _get_child(model, str(module_path))
        allowed.train()
        allowed.requires_grad_(True)
        prefix = f"{module_path}." if module_path else ""
        for parameter_name, parameter in allowed.named_parameters():
            if parameter.requires_grad:
                trainable_names.add(f"{prefix}{parameter_name}")
    return tuple(sorted(trainable_names))


def lora_state_dict(
    module: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
    cpu: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Return an adapter-only state dict, optionally including tiny modules."""

    if any(not str(path).strip() for path in trainable_modules):
        raise ValueError("Trainable module paths must be non-empty strings.")
    full_state = module.state_dict()
    lora_keys = {
        f"{name}.{suffix}" if name else suffix
        for name, _ in iter_lora_modules(module)
        for suffix in ("lora_A", "lora_B")
    }
    prefixes = tuple(f"{str(path)}." for path in trainable_modules)
    selected = OrderedDict()
    for name, tensor in full_state.items():
        if name not in lora_keys and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        value = tensor.detach().clone()
        if cpu:
            value = value.cpu()
        selected[name] = value
    missing = sorted(lora_keys.difference(selected))
    if missing:
        raise RuntimeError(f"Failed to collect LoRA state keys: {missing[:10]!r}.")
    return selected


def adapter_payload(
    module: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    adapted = list(iter_lora_modules(module))
    ranks = sorted({child.rank for _, child in adapted})
    alphas = sorted({child.alpha for _, child in adapted})
    return {
        "format": LORA_ADAPTER_FORMAT,
        "state_dict": lora_state_dict(
            module,
            trainable_modules=trainable_modules,
            cpu=True,
        ),
        "metadata": {
            "ranks": ranks,
            "alphas": alphas,
            "module_count": len(adapted),
            "trainable_modules": list(trainable_modules),
            **dict(metadata or {}),
        },
    }


def save_lora_adapter(
    module: nn.Module,
    path: str | Path,
    *,
    trainable_modules: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(
            adapter_payload(
                module,
                trainable_modules=trainable_modules,
                metadata=metadata,
            ),
            temporary,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_lora_adapter(
    module: nn.Module,
    adapter: str | Path | Mapping[str, object],
) -> object:
    payload = (
        load_lora_adapter_payload(adapter)
        if isinstance(adapter, (str, Path))
        else dict(adapter)
    )
    if payload.get("format") != LORA_ADAPTER_FORMAT:
        raise ValueError(
            f"Unsupported LoRA adapter format: {payload.get('format')!r}."
        )
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("LoRA adapter payload is missing a mapping `state_dict`.")
    adapted = list(iter_lora_modules(module))
    metadata = payload.get("metadata", {})
    if isinstance(metadata, Mapping):
        expected_ranks = sorted({child.rank for _, child in adapted})
        expected_alphas = sorted({child.alpha for _, child in adapted})
        if "ranks" in metadata and list(metadata["ranks"]) != expected_ranks:
            raise ValueError(
                "LoRA adapter rank mismatch: "
                f"adapter={list(metadata['ranks'])!r}, model={expected_ranks!r}."
            )
        if "alphas" in metadata and list(metadata["alphas"]) != expected_alphas:
            raise ValueError(
                "LoRA adapter alpha mismatch: "
                f"adapter={list(metadata['alphas'])!r}, model={expected_alphas!r}."
            )
    expected_lora_keys = set(lora_state_dict(module, cpu=False))
    provided_lora_keys = {
        name
        for name in state
        if str(name).split(".")[-1] in {"lora_A", "lora_B"}
    }
    missing = sorted(expected_lora_keys.difference(provided_lora_keys))
    if missing:
        raise ValueError(f"LoRA adapter is incomplete; missing keys: {missing[:10]!r}.")
    incompatible = module.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "LoRA adapter contains unexpected keys: "
            f"{incompatible.unexpected_keys[:10]!r}."
        )
    return incompatible


def _adapter_base_binding(metadata: Mapping[str, object]) -> tuple[str, str | None]:
    binding = metadata.get("base_checkpoint")
    if not isinstance(binding, Mapping):
        raise ValueError("LoRA adapter metadata is missing `base_checkpoint` binding.")
    path = str(binding.get("path", "")).strip()
    if not path:
        raise ValueError("LoRA adapter base checkpoint binding has an empty path.")
    sha256 = str(binding.get("sha256", "")).strip().lower() or None
    if sha256 is not None and (len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256)):
        raise ValueError(f"Invalid base checkpoint SHA-256 in adapter metadata: {sha256!r}.")
    return path, sha256


def validate_lora_training_contract(
    payload: Mapping[str, object],
    *,
    rank: int,
    alpha: float,
    experts: Sequence[str],
    target_modules: Sequence[str],
    trainable_modules: Sequence[str],
) -> Mapping[str, object]:
    """Fail closed when an adapter and the configured module topology differ."""

    if payload.get("format") != LORA_ADAPTER_FORMAT:
        raise ValueError(
            f"Unsupported LoRA adapter format: {payload.get('format')!r}."
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("LoRA adapter payload is missing a mapping `metadata`.")
    _adapter_base_binding(metadata)

    expected = {
        "rank": int(rank),
        "alpha": float(alpha),
        "experts": list(experts),
        "target_modules": list(target_modules),
        "trainable_modules": list(trainable_modules),
    }
    missing = [name for name in expected if name not in metadata]
    if missing:
        raise ValueError(
            "LoRA adapter metadata predates the production training contract; "
            f"missing fields: {missing!r}."
        )
    observed = {
        "rank": int(metadata["rank"]),
        "alpha": float(metadata["alpha"]),
        "experts": list(metadata["experts"]),
        "target_modules": list(metadata["target_modules"]),
        "trainable_modules": list(metadata["trainable_modules"]),
    }
    mismatches = {
        name: {"adapter": observed[name], "configured": value}
        for name, value in expected.items()
        if observed[name] != value
    }
    if mismatches:
        raise ValueError(f"LoRA adapter training contract mismatch: {mismatches!r}.")
    return metadata


def resolve_lora_base_checkpoint(
    payload: Mapping[str, object],
    *,
    override: str | Path | None = None,
    allow_path_override: bool = False,
    verify_sha256: bool = True,
) -> Path:
    """Resolve and optionally verify the full checkpoint bound to an adapter."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("LoRA adapter payload is missing a mapping `metadata`.")
    recorded_path, recorded_sha256 = _adapter_base_binding(metadata)
    recorded = Path(recorded_path).expanduser()
    chosen = Path(override).expanduser() if override is not None else recorded
    if override is not None:
        same_path = chosen.resolve(strict=False) == recorded.resolve(strict=False)
        if not same_path and not allow_path_override:
            raise ValueError(
                "Explicit base checkpoint does not match the adapter binding: "
                f"adapter={recorded} override={chosen}. Pass an explicit path-override "
                "flag only after checking provenance."
            )
    if not chosen.is_file():
        raise FileNotFoundError(f"LoRA base checkpoint not found: {chosen}")
    if verify_sha256 and recorded_sha256 is not None:
        actual_sha256 = checkpoint_sha256(chosen)
        if actual_sha256 != recorded_sha256:
            raise ValueError(
                "LoRA base checkpoint SHA-256 mismatch: "
                f"expected={recorded_sha256} actual={actual_sha256} path={chosen}"
            )
    return chosen


def _checkpoint_state_target(
    checkpoint: Mapping[str, object], adapter_key: str
) -> tuple[Mapping[str, torch.Tensor], str]:
    """Map a full-model adapter key onto the legacy checkpoint envelope."""

    if adapter_key.startswith("mot."):
        state = checkpoint.get("mot")
        key = adapter_key[len("mot.") :]
    elif adapter_key.startswith("dit."):
        # ``FastWAM.dit`` aliases ``FastWAM.mot``.  Trainer-produced adapters
        # normally use the first registered name (``mot``), but accept the alias.
        state = checkpoint.get("mot")
        key = adapter_key[len("dit.") :]
    elif adapter_key.startswith("proprio_encoder."):
        state = checkpoint.get("proprio_encoder")
        key = adapter_key[len("proprio_encoder.") :]
    elif adapter_key.startswith("source_proprio_encoders."):
        state = checkpoint.get("source_proprio_encoders")
        key = adapter_key[len("source_proprio_encoders.") :]
    else:
        raise ValueError(
            "Adapter state key is outside the supported FastWAM checkpoint roots: "
            f"{adapter_key!r}."
        )
    if not isinstance(state, Mapping):
        raise ValueError(
            f"Base checkpoint has no state mapping for adapter key {adapter_key!r}."
        )
    return state, key


@torch.no_grad()
def merge_lora_adapter_checkpoint(
    base_checkpoint: str | Path,
    adapter: str | Path | Mapping[str, object],
    output: str | Path,
    *,
    verify_base_sha256: bool = True,
    allow_base_path_override: bool = False,
) -> Path:
    """Merge an adapter into a legacy runtime-compatible FastWAM checkpoint.

    The implementation operates directly on the checkpoint dictionaries, so
    exporting does not require constructing the 6B-parameter model or loading
    VAE/text weights.  Every adapter key is accounted for and all LoRA tensors
    are removed from the exported envelope.
    """

    payload = (
        load_lora_adapter_payload(adapter)
        if isinstance(adapter, (str, Path))
        else dict(adapter)
    )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("LoRA adapter payload is missing a mapping `metadata`.")
    resolved_base = resolve_lora_base_checkpoint(
        payload,
        override=base_checkpoint,
        allow_path_override=allow_base_path_override,
        verify_sha256=verify_base_sha256,
    )
    rank = int(metadata.get("rank", 0))
    alpha = float(metadata.get("alpha", 0.0))
    if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            f"Adapter metadata has invalid rank/alpha: rank={rank} alpha={alpha}."
        )
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("LoRA adapter payload is missing a mapping `state_dict`.")

    # mmap avoids eagerly materializing a second 12 GB copy.  Modified target
    # tensors use private copy-on-write pages before the atomic torch.save.
    checkpoint = torch.load(
        resolved_base,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("mot"), Mapping):
        raise ValueError(f"Base checkpoint is not a standard FastWAM MoT checkpoint: {resolved_base}")
    checkpoint = dict(checkpoint)
    if any("lora_" in str(name) for name in checkpoint["mot"]):
        raise ValueError("Base checkpoint already contains LoRA tensors; expected an unadapted base.")

    lora_a = {str(name)[: -len(".lora_A")]: tensor for name, tensor in state.items() if str(name).endswith(".lora_A")}
    lora_b = {str(name)[: -len(".lora_B")]: tensor for name, tensor in state.items() if str(name).endswith(".lora_B")}
    if not lora_a or set(lora_a) != set(lora_b):
        raise ValueError(
            "Adapter has incomplete LoRA A/B pairs: "
            f"only_A={sorted(set(lora_a) - set(lora_b))[:10]!r} "
            f"only_B={sorted(set(lora_b) - set(lora_a))[:10]!r}."
        )
    declared_module_count = int(metadata.get("module_count", -1))
    if declared_module_count != len(lora_a):
        raise ValueError(
            "Adapter module count mismatch: "
            f"metadata={declared_module_count} state={len(lora_a)}."
        )

    scaling = alpha / rank
    consumed: set[str] = set()
    for module_name in sorted(lora_a):
        a = lora_a[module_name]
        b = lora_b[module_name]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError(f"LoRA pair for {module_name!r} is not tensor-valued.")
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(
                f"LoRA pair shape mismatch for {module_name}: A={tuple(a.shape)} "
                f"B={tuple(b.shape)} rank={rank}."
            )
        target_state, target_key = _checkpoint_state_target(
            checkpoint, f"{module_name}.weight"
        )
        weight = target_state.get(target_key)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                f"Base checkpoint is missing LoRA target {module_name}.weight."
            )
        if tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(
                f"Base/adapter shape mismatch for {module_name}: "
                f"weight={tuple(weight.shape)} A={tuple(a.shape)} B={tuple(b.shape)}."
            )
        delta = torch.matmul(b.float(), a.float()).mul_(scaling)
        weight.add_(delta.to(dtype=weight.dtype))
        consumed.update({f"{module_name}.lora_A", f"{module_name}.lora_B"})

    declared_trainable = tuple(str(name) for name in metadata.get("trainable_modules", ()))
    for adapter_key, tensor in state.items():
        adapter_key = str(adapter_key)
        if adapter_key in consumed:
            continue
        if not any(
            adapter_key == prefix or adapter_key.startswith(f"{prefix}.")
            for prefix in declared_trainable
        ):
            raise ValueError(
                f"Undeclared non-LoRA tensor in adapter state: {adapter_key!r}."
            )
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Adapter value for {adapter_key!r} is not a tensor.")
        target_state, target_key = _checkpoint_state_target(checkpoint, adapter_key)
        target = target_state.get(target_key)
        if not isinstance(target, torch.Tensor) or target.shape != tensor.shape:
            raise ValueError(
                f"Base/adapter tiny-module mismatch for {adapter_key}: "
                f"base={getattr(target, 'shape', None)} adapter={tuple(tensor.shape)}."
            )
        target.copy_(tensor.to(dtype=target.dtype))
        consumed.add(adapter_key)
    if consumed != {str(name) for name in state}:
        raise RuntimeError("Internal error: not every adapter tensor was consumed during merge.")

    if "step" in metadata:
        checkpoint["step"] = int(metadata["step"])
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@torch.no_grad()
def merge_lora_weights(module: nn.Module, *, unload: bool = False) -> tuple[str, ...]:
    """Merge every adapter into base weights, optionally remove LoRA modules."""

    adapted = list(iter_lora_modules(module))
    for _, child in adapted:
        child.merge()
    if unload:
        # Deepest-first is defensive for arbitrary nested LoRA modules.
        for name, child in sorted(adapted, key=lambda pair: pair[0].count("."), reverse=True):
            parent_path, _, child_name = name.rpartition(".")
            parent = _get_child(module, parent_path)
            plain = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            plain.weight = child.weight
            plain.bias = child.bias
            plain.train(child.training)
            _replace_child(parent, child_name, plain)
    return tuple(name for name, _ in adapted)


@torch.no_grad()
def unmerge_lora_weights(module: nn.Module) -> tuple[str, ...]:
    adapted = list(iter_lora_modules(module))
    for _, child in adapted:
        child.unmerge()
    return tuple(name for name, _ in adapted)


__all__ = [
    "DEFAULT_MOT_LORA_TARGETS",
    "LORA_ADAPTER_FORMAT",
    "LoRAInjectionReport",
    "LoRALinear",
    "adapter_payload",
    "checkpoint_sha256",
    "inject_lora_adapters",
    "inject_mot_lora",
    "is_lora_adapter_checkpoint",
    "iter_lora_modules",
    "load_lora_adapter",
    "load_lora_adapter_payload",
    "lora_state_dict",
    "merge_lora_adapter_checkpoint",
    "merge_lora_weights",
    "resolve_lora_base_checkpoint",
    "save_lora_adapter",
    "set_lora_trainable",
    "unmerge_lora_weights",
    "validate_lora_training_contract",
]
