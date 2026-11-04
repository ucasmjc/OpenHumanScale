"""Function-preserving bottleneck adapters for FastWAM's MoT experts.

The adapters in this module are sequential residual adapters attached after a
transformer block's mixed self-attention update and/or its FFN update.  They are
token-wise: no operation mixes the sequence dimension, so the MoT attention
mask remains the sole authority for cross-view and cross-expert information
flow.

Injection happens before loading a post-training base checkpoint.  The final
``up`` projection is exactly zero initialized, which makes every injected
adapter an identity while allowing a normal checkpoint (with no adapter keys)
to initialize the frozen backbone.
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

from .lora import checkpoint_sha256


BOTTLENECK_ADAPTER_FORMAT = "fastwam_bottleneck_adapter_v1"
BOTTLENECK_ADAPTER_SUFFIX = ".bottleneck.pt"
DEFAULT_BOTTLENECK_ADAPTER_EXPERTS = ("video", "action")
DEFAULT_BOTTLENECK_ADAPTER_SITES = ("self_attn", "ffn")
_SITE_ATTRIBUTES = {
    "self_attn": "post_self_attn_adapter",
    "ffn": "post_ffn_adapter",
}
_ADAPTER_PARAMETER_SUFFIXES = (
    "down.weight",
    "down.bias",
    "up.weight",
    "up.bias",
)


def _join_module_key(module_name: str, parameter_name: str) -> str:
    return f"{module_name}.{parameter_name}" if module_name else parameter_name


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _derived_parameter_seed(init_seed: int, *parts: object) -> int:
    """Derive a stable per-module seed without Python's randomized ``hash``."""

    identity = ":".join([str(int(init_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    # torch.Generator.manual_seed accepts a signed 64-bit seed on every backend.
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


class BottleneckAdapter(nn.Module):
    """A pre-normalized bottleneck residual initialized as the identity.

    ``LayerNorm`` is deliberately affine-free.  Consequently each instance has
    exactly four parameter tensors: down/up weights and biases.  ``residual_scale``
    is a fixed Python float rather than a trainable gate.
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        *,
        activation: str = "gelu_tanh",
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        eps: float = 1e-6,
        init_seed: int = 42,
        parameter_seed: int | None = None,
        expert_name: str = "",
        site: str = "",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError(f"Adapter hidden_dim must be positive, got {hidden_dim}.")
        if int(bottleneck_dim) <= 0:
            raise ValueError(
                f"Adapter bottleneck_dim must be positive, got {bottleneck_dim}."
            )
        activation = str(activation).strip().lower()
        if activation != "gelu_tanh":
            raise ValueError(
                "Bottleneck adapter activation must be 'gelu_tanh', "
                f"got {activation!r}."
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(
                f"Bottleneck adapter dropout must be in [0, 1), got {dropout}."
            )
        if not math.isfinite(float(residual_scale)) or float(residual_scale) < 0.0:
            raise ValueError(
                "Bottleneck adapter residual_scale must be finite and non-negative, "
                f"got {residual_scale}."
            )
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError(f"Bottleneck adapter eps must be positive, got {eps}.")

        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.activation_name = activation
        self.dropout_probability = float(dropout)
        self.residual_scale = float(residual_scale)
        self.init_seed = int(init_seed)
        self.parameter_seed = int(
            self.init_seed if parameter_seed is None else parameter_seed
        )
        self.expert_name = str(expert_name)
        self.site = str(site)

        self.norm = nn.LayerNorm(
            self.hidden_dim,
            eps=float(eps),
            elementwise_affine=False,
            device=device,
            dtype=dtype,
        )
        self.down = nn.Linear(
            self.hidden_dim,
            self.bottleneck_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(self.dropout_probability)
        self.up = nn.Linear(
            self.bottleneck_dim,
            self.hidden_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.reset_adapter_parameters()

    def reset_adapter_parameters(self) -> None:
        generator = torch.Generator(device=self.down.weight.device)
        generator.manual_seed(self.parameter_seed)
        nn.init.kaiming_uniform_(
            self.down.weight,
            a=math.sqrt(5),
            generator=generator,
        )
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(
            self.dropout(
                self.activation(
                    self.down(self.norm(x))
                )
            )
        )
        return x + delta * self.residual_scale

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, bottleneck_dim={self.bottleneck_dim}, "
            f"activation={self.activation_name!r}, dropout={self.dropout_probability}, "
            f"residual_scale={self.residual_scale}, init_seed={self.init_seed}"
        )


@dataclass(frozen=True)
class BottleneckAdapterInjectionReport:
    module_names: tuple[str, ...]
    bottleneck_dims: Mapping[str, int]
    trainable_parameters: int
    parameter_tensor_count: int

    @property
    def module_count(self) -> int:
        return len(self.module_names)


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
            raise ValueError(
                f"Module path {path!r} resolves to non-module {current!r}."
            )
    return current


def _block_hidden_dim(block: nn.Module) -> int:
    hidden_dim = getattr(block, "hidden_dim", None)
    q_projection = getattr(getattr(block, "self_attn", None), "q", None)
    q_in_features = getattr(q_projection, "in_features", None)
    if hidden_dim is None:
        hidden_dim = q_in_features
    if hidden_dim is None or int(hidden_dim) <= 0:
        raise ValueError(
            "Cannot resolve adapter residual hidden dimension from block.hidden_dim "
            "or block.self_attn.q.in_features."
        )
    if q_in_features is not None and int(q_in_features) != int(hidden_dim):
        raise ValueError(
            "Block hidden dimension disagrees with self-attention input dimension: "
            f"hidden_dim={hidden_dim}, q.in_features={q_in_features}."
        )
    return int(hidden_dim)


def _normalize_injection_contract(
    *,
    experts: Sequence[str],
    sites: Sequence[str],
    bottleneck_dims: Mapping[str, int] | None,
    reduction_factor: int | None,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, int] | None, int | None]:
    experts = tuple(str(name) for name in experts)
    if not experts:
        raise ValueError("At least one bottleneck adapter expert is required.")
    if len(set(experts)) != len(experts):
        raise ValueError(
            f"Duplicate bottleneck adapter experts are not allowed: {list(experts)!r}."
        )
    sites = tuple(str(name) for name in sites)
    if not sites:
        raise ValueError("At least one bottleneck adapter site is required.")
    if len(set(sites)) != len(sites):
        raise ValueError(
            f"Duplicate bottleneck adapter sites are not allowed: {list(sites)!r}."
        )
    unknown_sites = sorted(set(sites) - set(_SITE_ATTRIBUTES))
    if unknown_sites:
        raise ValueError(
            f"Unknown bottleneck adapter sites: {unknown_sites!r}; "
            f"allowed={list(_SITE_ATTRIBUTES)!r}."
        )

    resolved_dims = None
    if bottleneck_dims is not None:
        if reduction_factor is not None:
            raise ValueError(
                "Specify bottleneck_dims or reduction_factor, not both."
            )
        resolved_dims = {
            str(name): int(value) for name, value in dict(bottleneck_dims).items()
        }
        missing = sorted(set(experts) - set(resolved_dims))
        extra = sorted(set(resolved_dims) - set(experts))
        if missing or extra:
            raise ValueError(
                "bottleneck_dims keys must exactly match configured experts: "
                f"missing={missing!r}, extra={extra!r}."
            )
        invalid = {name: value for name, value in resolved_dims.items() if value <= 0}
        if invalid:
            raise ValueError(f"Bottleneck dimensions must be positive: {invalid!r}.")
    else:
        reduction_factor = 32 if reduction_factor is None else int(reduction_factor)
        if reduction_factor <= 0:
            raise ValueError(
                f"Bottleneck reduction_factor must be positive, got {reduction_factor}."
            )
    return experts, sites, resolved_dims, reduction_factor


def inject_bottleneck_adapters(
    mot: nn.Module,
    *,
    experts: Sequence[str] = DEFAULT_BOTTLENECK_ADAPTER_EXPERTS,
    sites: Sequence[str] = DEFAULT_BOTTLENECK_ADAPTER_SITES,
    bottleneck_dims: Mapping[str, int] | None = None,
    reduction_factor: int | None = None,
    activation: str = "gelu_tanh",
    dropout: float = 0.0,
    residual_scale: float = 1.0,
    init_seed: int = 42,
) -> BottleneckAdapterInjectionReport:
    """Attach independent adapters to selected sites of every selected expert."""

    experts, sites, resolved_dims, reduction_factor = _normalize_injection_contract(
        experts=experts,
        sites=sites,
        bottleneck_dims=bottleneck_dims,
        reduction_factor=reduction_factor,
    )
    mixtures = getattr(mot, "mixtures", None)
    if not isinstance(mixtures, nn.ModuleDict):
        raise ValueError(
            "Bottleneck adapter injection expects MoT.mixtures to be an nn.ModuleDict."
        )

    injected: list[str] = []
    trainable_parameters = 0
    parameter_tensor_count = 0
    final_dims: dict[str, int] = {}
    for expert_name in experts:
        if expert_name not in mixtures:
            raise ValueError(
                f"Unknown bottleneck adapter expert {expert_name!r}; "
                f"available={list(mixtures.keys())!r}."
            )
        expert = mixtures[expert_name]
        blocks = getattr(expert, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
            raise ValueError(
                f"MoT expert {expert_name!r} has no non-empty ModuleList blocks."
            )
        expert_hidden_dim = _block_hidden_dim(blocks[0])
        for block in blocks[1:]:
            hidden_dim = _block_hidden_dim(block)
            if hidden_dim != expert_hidden_dim:
                raise ValueError(
                    f"Expert {expert_name!r} changes residual hidden dimension across "
                    f"blocks: {expert_hidden_dim} vs {hidden_dim}."
                )
        if resolved_dims is not None:
            bottleneck_dim = int(resolved_dims[expert_name])
        else:
            assert reduction_factor is not None
            if expert_hidden_dim % reduction_factor != 0:
                raise ValueError(
                    f"Expert {expert_name!r} hidden_dim={expert_hidden_dim} is not "
                    f"divisible by reduction_factor={reduction_factor}; configure an "
                    "explicit bottleneck_dims mapping."
                )
            bottleneck_dim = expert_hidden_dim // reduction_factor
        final_dims[expert_name] = bottleneck_dim

        for block_index, block in enumerate(blocks):
            q_projection = getattr(getattr(block, "self_attn", None), "q", None)
            weight = getattr(q_projection, "weight", None)
            if not isinstance(weight, nn.Parameter):
                raise ValueError(
                    f"Expert {expert_name!r} block {block_index} has no "
                    "self_attn.q.weight parameter for adapter device/dtype placement."
                )
            for site in sites:
                attribute = _SITE_ATTRIBUTES[site]
                existing = getattr(block, attribute, None)
                if existing is not None:
                    raise ValueError(
                        "Bottleneck adapter site was already injected: "
                        f"mixtures.{expert_name}.blocks.{block_index}.{attribute}."
                    )
                parameter_seed = _derived_parameter_seed(
                    int(init_seed), expert_name, block_index, site
                )
                adapter = BottleneckAdapter(
                    hidden_dim=expert_hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    dropout=dropout,
                    residual_scale=residual_scale,
                    eps=float(getattr(getattr(block, "norm1", None), "eps", 1e-6)),
                    init_seed=int(init_seed),
                    parameter_seed=parameter_seed,
                    expert_name=expert_name,
                    site=site,
                    device=weight.device,
                    dtype=weight.dtype,
                )
                adapter.train(block.training)
                setattr(block, attribute, adapter)
                full_name = (
                    f"mixtures.{expert_name}.blocks.{block_index}.{attribute}"
                )
                injected.append(full_name)
                parameters = tuple(adapter.parameters())
                trainable_parameters += sum(parameter.numel() for parameter in parameters)
                parameter_tensor_count += len(parameters)

    return BottleneckAdapterInjectionReport(
        module_names=tuple(injected),
        bottleneck_dims=final_dims,
        trainable_parameters=int(trainable_parameters),
        parameter_tensor_count=int(parameter_tensor_count),
    )


def iter_bottleneck_adapters(
    module: nn.Module,
) -> Iterator[tuple[str, BottleneckAdapter]]:
    for name, child in module.named_modules():
        if isinstance(child, BottleneckAdapter):
            yield name, child


def set_bottleneck_adapter_trainable(
    model: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
) -> tuple[str, ...]:
    """Freeze the model, then enable adapter tensors and an exact allowlist."""

    model.requires_grad_(False)
    trainable_names: set[str] = set()
    adapted = list(iter_bottleneck_adapters(model))
    if not adapted:
        raise ValueError(
            "Bottleneck adapter train mode requested, but the model has no adapters."
        )
    for module_name, adapter in adapted:
        adapter.train()
        adapter.requires_grad_(True)
        for parameter_name, parameter in adapter.named_parameters():
            if parameter.requires_grad:
                trainable_names.add(_join_module_key(module_name, parameter_name))

    trainable_modules = tuple(str(path) for path in trainable_modules)
    if len(set(trainable_modules)) != len(trainable_modules):
        raise ValueError(
            f"Duplicate trainable module paths are not allowed: {list(trainable_modules)!r}."
        )
    if any(not path.strip() for path in trainable_modules):
        raise ValueError("Trainable module paths must be non-empty strings.")
    for module_path in trainable_modules:
        allowed = _get_child(model, module_path)
        allowed.train()
        allowed.requires_grad_(True)
        for parameter_name, parameter in allowed.named_parameters():
            if parameter.requires_grad:
                trainable_names.add(_join_module_key(module_path, parameter_name))
    return tuple(sorted(trainable_names))


def _adapter_parameter_keys(module: nn.Module) -> set[str]:
    return {
        _join_module_key(module_name, parameter_name)
        for module_name, adapter in iter_bottleneck_adapters(module)
        for parameter_name, _ in adapter.named_parameters()
    }


def bottleneck_adapter_state_dict(
    module: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
    cpu: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Collect only bottleneck adapter tensors plus explicitly allowed modules."""

    trainable_modules = tuple(str(path) for path in trainable_modules)
    if any(not path.strip() for path in trainable_modules):
        raise ValueError("Trainable module paths must be non-empty strings.")
    adapter_keys = _adapter_parameter_keys(module)
    if not adapter_keys:
        raise ValueError("Cannot collect state from a model with no bottleneck adapters.")
    prefixes = tuple(f"{path}." for path in trainable_modules)
    full_state = module.state_dict()
    selected: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in full_state.items():
        if name not in adapter_keys and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        value = tensor.detach().clone()
        if cpu:
            value = value.cpu()
        selected[name] = value
    missing = sorted(adapter_keys.difference(selected))
    if missing:
        raise RuntimeError(
            f"Failed to collect bottleneck adapter state keys: {missing[:10]!r}."
        )
    return selected


def _uniform_adapter_value(adapted, attribute: str):
    values = {getattr(child, attribute) for _, child in adapted}
    if len(values) != 1:
        raise ValueError(
            f"Bottleneck adapters have heterogeneous {attribute}: {sorted(values)!r}."
        )
    return next(iter(values))


def _current_adapter_contract(module: nn.Module) -> dict[str, object]:
    adapted = list(iter_bottleneck_adapters(module))
    if not adapted:
        raise ValueError("Model has no bottleneck adapters.")
    experts = _ordered_unique([child.expert_name for _, child in adapted])
    sites = _ordered_unique([child.site for _, child in adapted])
    if any(not value for value in (*experts, *sites)):
        raise ValueError(
            "Bottleneck adapters are missing injection topology metadata."
        )
    dims: dict[str, int] = {}
    for _, child in adapted:
        previous = dims.setdefault(child.expert_name, child.bottleneck_dim)
        if previous != child.bottleneck_dim:
            raise ValueError(
                f"Expert {child.expert_name!r} has heterogeneous bottleneck dims."
            )
    return {
        "experts": list(experts),
        "sites": list(sites),
        "bottleneck_dims": dims,
        "activation": _uniform_adapter_value(adapted, "activation_name"),
        "dropout": float(_uniform_adapter_value(adapted, "dropout_probability")),
        "residual_scale": float(
            _uniform_adapter_value(adapted, "residual_scale")
        ),
        "init_seed": int(_uniform_adapter_value(adapted, "init_seed")),
        "module_count": len(adapted),
        "parameter_tensor_count": sum(
            len(tuple(child.parameters())) for _, child in adapted
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for _, child in adapted
            for parameter in child.parameters()
        ),
    }


def bottleneck_adapter_payload(
    module: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract = _current_adapter_contract(module)
    contract["trainable_modules"] = [str(path) for path in trainable_modules]
    supplied = dict(metadata or {})
    for name, expected in contract.items():
        if name in supplied and supplied[name] != expected:
            raise ValueError(
                f"Bottleneck adapter metadata conflicts with live topology for {name}: "
                f"metadata={supplied[name]!r}, model={expected!r}."
            )
    contract.update(supplied)
    return {
        "format": BOTTLENECK_ADAPTER_FORMAT,
        "state_dict": bottleneck_adapter_state_dict(
            module,
            trainable_modules=trainable_modules,
            cpu=True,
        ),
        "metadata": contract,
    }


def save_bottleneck_adapter(
    module: nn.Module,
    path: str | Path,
    *,
    trainable_modules: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> None:
    output = Path(path)
    if not str(output).endswith(BOTTLENECK_ADAPTER_SUFFIX):
        raise ValueError(
            f"Bottleneck adapter checkpoint must end with "
            f"{BOTTLENECK_ADAPTER_SUFFIX!r}: {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(
            bottleneck_adapter_payload(
                module,
                trainable_modules=trainable_modules,
                metadata=metadata,
            ),
            temporary,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_bottleneck_adapter_payload(path: str | Path) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Bottleneck adapter checkpoint must contain a mapping, "
            f"got {type(payload).__name__}."
        )
    payload = dict(payload)
    if payload.get("format") != BOTTLENECK_ADAPTER_FORMAT:
        raise ValueError(
            f"Unsupported bottleneck adapter format: {payload.get('format')!r}."
        )
    if not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError(
            "Bottleneck adapter payload is missing a mapping `state_dict`."
        )
    if not isinstance(payload.get("metadata"), Mapping):
        raise ValueError(
            "Bottleneck adapter payload is missing a mapping `metadata`."
        )
    return payload


def is_bottleneck_adapter_checkpoint(path: str | Path) -> bool:
    return str(path).endswith(BOTTLENECK_ADAPTER_SUFFIX)


def _validate_payload_state_contract(payload: Mapping[str, object]) -> None:
    metadata = payload.get("metadata")
    state = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Bottleneck adapter payload is missing metadata/state_dict.")
    adapter_keys = {
        str(name): tensor
        for name, tensor in state.items()
        if str(name).endswith(tuple(f".{suffix}" for suffix in _ADAPTER_PARAMETER_SUFFIXES))
        or str(name) in _ADAPTER_PARAMETER_SUFFIXES
    }
    groups: dict[str, set[str]] = {}
    for name, tensor in adapter_keys.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Adapter state value for {name!r} is not a tensor.")
        suffix = next(
            suffix
            for suffix in _ADAPTER_PARAMETER_SUFFIXES
            if name == suffix or name.endswith(f".{suffix}")
        )
        prefix = name[: -(len(suffix) + 1)] if name != suffix else ""
        groups.setdefault(prefix, set()).add(suffix)
    incomplete = {
        prefix: sorted(set(_ADAPTER_PARAMETER_SUFFIXES) - suffixes)
        for prefix, suffixes in groups.items()
        if suffixes != set(_ADAPTER_PARAMETER_SUFFIXES)
    }
    if incomplete:
        raise ValueError(
            f"Bottleneck adapter payload has incomplete modules: {incomplete!r}."
        )
    declared_modules = int(metadata.get("module_count", -1))
    declared_tensors = int(metadata.get("parameter_tensor_count", -1))
    declared_parameters = int(metadata.get("trainable_parameters", -1))
    if declared_modules != len(groups):
        raise ValueError(
            "Bottleneck adapter module count mismatch: "
            f"metadata={declared_modules}, state={len(groups)}."
        )
    if declared_tensors != len(adapter_keys):
        raise ValueError(
            "Bottleneck adapter parameter tensor count mismatch: "
            f"metadata={declared_tensors}, state={len(adapter_keys)}."
        )
    observed_parameters = sum(tensor.numel() for tensor in adapter_keys.values())
    if declared_parameters != observed_parameters:
        raise ValueError(
            "Bottleneck adapter trainable parameter count mismatch: "
            f"metadata={declared_parameters}, state={observed_parameters}."
        )


def load_bottleneck_adapter(
    module: nn.Module,
    adapter: str | Path | Mapping[str, object],
) -> object:
    payload = (
        load_bottleneck_adapter_payload(adapter)
        if isinstance(adapter, (str, Path))
        else dict(adapter)
    )
    if payload.get("format") != BOTTLENECK_ADAPTER_FORMAT:
        raise ValueError(
            f"Unsupported bottleneck adapter format: {payload.get('format')!r}."
        )
    state = payload.get("state_dict")
    metadata = payload.get("metadata")
    if not isinstance(state, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Bottleneck adapter payload is missing metadata/state_dict.")
    _validate_payload_state_contract(payload)

    current = _current_adapter_contract(module)
    for name, expected in current.items():
        if name not in metadata:
            raise ValueError(
                f"Bottleneck adapter metadata is missing topology field {name!r}."
            )
        if metadata[name] != expected:
            raise ValueError(
                f"Bottleneck adapter topology mismatch for {name}: "
                f"adapter={metadata[name]!r}, model={expected!r}."
            )
    trainable_modules = tuple(str(path) for path in metadata.get("trainable_modules", ()))
    expected = set(
        bottleneck_adapter_state_dict(
            module,
            trainable_modules=trainable_modules,
            cpu=False,
        )
    )
    provided = {str(name) for name in state}
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing:
        raise ValueError(
            f"Bottleneck adapter is incomplete; missing keys: {missing[:10]!r}."
        )
    if unexpected:
        raise ValueError(
            f"Bottleneck adapter contains unexpected keys: {unexpected[:10]!r}."
        )
    incompatible = module.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "Bottleneck adapter contains unexpected model keys: "
            f"{incompatible.unexpected_keys[:10]!r}."
        )
    return incompatible


def _bottleneck_base_binding(
    metadata: Mapping[str, object],
) -> tuple[str, str | None]:
    binding = metadata.get("base_checkpoint")
    if not isinstance(binding, Mapping):
        raise ValueError(
            "Bottleneck adapter metadata is missing `base_checkpoint` binding."
        )
    path = str(binding.get("path", "")).strip()
    if not path:
        raise ValueError("Bottleneck adapter base checkpoint path is empty.")
    sha256 = str(binding.get("sha256", "")).strip().lower() or None
    if sha256 is not None and (
        len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(
            f"Invalid bottleneck adapter base checkpoint SHA-256: {sha256!r}."
        )
    return path, sha256


def validate_bottleneck_training_contract(
    payload: Mapping[str, object],
    *,
    experts: Sequence[str],
    sites: Sequence[str],
    bottleneck_dims: Mapping[str, int],
    activation: str,
    dropout: float,
    residual_scale: float,
    init_seed: int,
    trainable_modules: Sequence[str],
) -> Mapping[str, object]:
    """Fail closed when a checkpoint and configured adapter topology differ."""

    if payload.get("format") != BOTTLENECK_ADAPTER_FORMAT:
        raise ValueError(
            f"Unsupported bottleneck adapter format: {payload.get('format')!r}."
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "Bottleneck adapter payload is missing a mapping `metadata`."
        )
    _bottleneck_base_binding(metadata)
    _validate_payload_state_contract(payload)
    expected = {
        "experts": [str(name) for name in experts],
        "sites": [str(name) for name in sites],
        "bottleneck_dims": {
            str(name): int(value) for name, value in dict(bottleneck_dims).items()
        },
        "activation": str(activation).strip().lower(),
        "dropout": float(dropout),
        "residual_scale": float(residual_scale),
        "init_seed": int(init_seed),
        "trainable_modules": [str(path) for path in trainable_modules],
    }
    missing = [name for name in expected if name not in metadata]
    if missing:
        raise ValueError(
            "Bottleneck adapter metadata predates the production contract; "
            f"missing fields: {missing!r}."
        )
    observed = {
        "experts": [str(name) for name in metadata["experts"]],
        "sites": [str(name) for name in metadata["sites"]],
        "bottleneck_dims": {
            str(name): int(value)
            for name, value in dict(metadata["bottleneck_dims"]).items()
        },
        "activation": str(metadata["activation"]),
        "dropout": float(metadata["dropout"]),
        "residual_scale": float(metadata["residual_scale"]),
        "init_seed": int(metadata["init_seed"]),
        "trainable_modules": [str(path) for path in metadata["trainable_modules"]],
    }
    mismatches = {
        name: {"adapter": observed[name], "configured": value}
        for name, value in expected.items()
        if observed[name] != value
    }
    if mismatches:
        raise ValueError(
            f"Bottleneck adapter training contract mismatch: {mismatches!r}."
        )
    return metadata


def resolve_bottleneck_base_checkpoint(
    payload: Mapping[str, object],
    *,
    override: str | Path | None = None,
    allow_path_override: bool = False,
    verify_sha256: bool = True,
) -> Path:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "Bottleneck adapter payload is missing a mapping `metadata`."
        )
    recorded_path, recorded_sha256 = _bottleneck_base_binding(metadata)
    recorded = Path(recorded_path).expanduser()
    chosen = Path(override).expanduser() if override is not None else recorded
    if override is not None:
        same_path = chosen.resolve(strict=False) == recorded.resolve(strict=False)
        if not same_path and not allow_path_override:
            raise ValueError(
                "Explicit base checkpoint does not match the bottleneck adapter "
                f"binding: adapter={recorded}, override={chosen}."
            )
    if not chosen.is_file():
        raise FileNotFoundError(
            f"Bottleneck adapter base checkpoint is missing: {chosen}."
        )
    if verify_sha256 and recorded_sha256 is not None:
        actual_sha256 = checkpoint_sha256(chosen)
        if actual_sha256 != recorded_sha256:
            raise ValueError(
                "Bottleneck adapter base checkpoint SHA-256 mismatch: "
                f"expected={recorded_sha256}, actual={actual_sha256}, path={chosen}."
            )
    return chosen


__all__ = [
    "BOTTLENECK_ADAPTER_FORMAT",
    "BOTTLENECK_ADAPTER_SUFFIX",
    "DEFAULT_BOTTLENECK_ADAPTER_EXPERTS",
    "DEFAULT_BOTTLENECK_ADAPTER_SITES",
    "BottleneckAdapter",
    "BottleneckAdapterInjectionReport",
    "bottleneck_adapter_payload",
    "bottleneck_adapter_state_dict",
    "inject_bottleneck_adapters",
    "is_bottleneck_adapter_checkpoint",
    "iter_bottleneck_adapters",
    "load_bottleneck_adapter",
    "load_bottleneck_adapter_payload",
    "resolve_bottleneck_base_checkpoint",
    "save_bottleneck_adapter",
    "set_bottleneck_adapter_trainable",
    "validate_bottleneck_training_contract",
]
