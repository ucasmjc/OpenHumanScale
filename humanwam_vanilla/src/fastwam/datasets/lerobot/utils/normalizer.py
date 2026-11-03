from typing import Literal, Dict, Annotated, Union, Any, List, Tuple, Optional
import torch
import json
import os
import time
from collections import defaultdict
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hashlib
from pathlib import Path
from git import Repo
from fastwam.utils.logging_config import get_logger

from fastwam.utils.pytorch_utils import dict_apply

logger = get_logger(__name__)

ConstConstStr = Annotated[str, "format: 'const_min/const_max', where const_min and const_max give the constant range"]
QuantileModeStr = Annotated[str, "format: 'quantile' or 'quantile:qLO/qHI' (e.g. 'quantile:q005/q995'); reads the true sampled quantiles stored under stats['global_quantiles'/'stepwise_quantiles'] and maps [qLO, qHI] -> [-1, 1]"]
NormMode = Union[Literal["min/max", "q01/q99", "z-score", "quantile"], ConstConstStr, QuantileModeStr]

DEFAULT_QUANTILE_MODE_BOUNDS = ("q005", "q995")


def parse_quantile_mode(mode: str) -> Optional[Tuple[str, str]]:
    """Return (lo_key, hi_key) for a 'quantile[:qLO/qHI]' mode string, else None."""
    if not isinstance(mode, str) or not mode.startswith("quantile"):
        return None
    if mode == "quantile":
        return DEFAULT_QUANTILE_MODE_BOUNDS
    if not mode.startswith("quantile:"):
        raise ValueError(f"invalid quantile normalization mode {mode!r}")
    bounds = mode[len("quantile:"):].split("/")
    if len(bounds) != 2 or not all(b.startswith("q") for b in bounds):
        raise ValueError(f"invalid quantile normalization mode {mode!r}; expected 'quantile:qLO/qHI'")
    return bounds[0], bounds[1]

class LinearNormalizer:
    def __init__(
            self,
            shape_meta,
            use_stepwise_action_norm,
            default_mode: NormMode,
            exception_mode: Dict[str, Dict[str, NormMode]],
            stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
            post_normalization_clips: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
        ):
        super().__init__()
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats

        for meta in shape_meta["action"]:
            key = meta["key"]

            if use_stepwise_action_norm:
                cur_stats = {k.removeprefix("stepwise_"): v for k, v in stats["action"][key].items() if k.startswith("stepwise_")}
            else:
                cur_stats = {k.removeprefix("global_"): v for k, v in stats["action"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "action" in exception_mode and key in exception_mode["action"]:
                cur_mode = exception_mode["action"][key]
            else:
                cur_mode = default_mode

            self.normalizers["action"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
                output_clips=(
                    post_normalization_clips.get("action", {}).get(key)
                    if post_normalization_clips is not None
                    else None
                ),
            )

        for meta in shape_meta["state"]:
            key = meta["key"]
            cur_stats = {k.removeprefix("global_"): v for k, v in stats["state"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "state" in exception_mode and key in exception_mode["state"]:
                cur_mode = exception_mode["state"][key]
            else:
                cur_mode = default_mode

            self.normalizers["state"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
                output_clips=(
                    post_normalization_clips.get("state", {}).get(key)
                    if post_normalization_clips is not None
                    else None
                ),
            )

    def get_stats(self):
        stats = {
            "action": {key: norm.get_stats() for key, norm in self.normalizers["action"].items()},
            "state": {key: norm.get_stats() for key, norm in self.normalizers["state"].items()}
        }
        return stats

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "action" in batch:
            for key, norm in self.normalizers["action"].items():
                batch["action"][key] = norm.forward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.forward(batch["state"][key])

        return batch

    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        for key, norm in self.normalizers["action"].items():
            batch["action"][key] = norm.backward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.backward(batch["state"][key])

        return batch


class SingleFieldLinearNormalizer:
    std_reg = 1e-8
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0
    def __init__(
        self,
        stats,
        mode: NormMode = "min/max",
        output_clips: Optional[List[Dict[str, Any]]] = None,
    ):
        self.stats = stats
        self.mode = mode

        if mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            scale = 1.0 / (input_std + self.std_reg)
            offset = - input_mean / (input_std + self.std_reg)
        else:
            if mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif mode == "q01/q99":
                input_min, input_max = stats["q01"], stats["q99"]
            elif parse_quantile_mode(mode) is not None:
                lo_key, hi_key = parse_quantile_mode(mode)
                quantiles = stats.get("quantiles")
                if not isinstance(quantiles, dict) or lo_key not in quantiles or hi_key not in quantiles:
                    raise KeyError(
                        f"normalization mode {mode!r} needs stats['global_quantiles'/'stepwise_quantiles']"
                        f"[{lo_key!r}/{hi_key!r}]; regenerate stats with quantile_reservoir_size > 0"
                    )
                input_min = torch.as_tensor(quantiles[lo_key]).clone()
                input_max = torch.as_tensor(quantiles[hi_key]).clone()
            else:
                # parse const_min/const_max
                input_min, input_max = map(float, mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]

        self.scale = scale
        self.offset = offset
        self.output_clips = []
        seen_indices = set()
        for clip in output_clips or []:
            index = int(clip["index"])
            lower = float(clip["min"])
            upper = float(clip["max"])
            if index < 0 or index >= self.scale.shape[-1]:
                raise ValueError(
                    f"post-normalization clip index {index} is outside the last "
                    f"dimension of shape {tuple(self.scale.shape)}"
                )
            if index in seen_indices:
                raise ValueError(f"duplicate post-normalization clip index: {index}")
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError(
                    f"invalid post-normalization clip bounds for index {index}: "
                    f"[{lower}, {upper}]"
                )
            seen_indices.add(index)
            self.output_clips.append((index, lower, upper))

    def _clip_selected_outputs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.output_clips:
            return x
        x = x.clone()
        for index, lower, upper in self.output_clips:
            x[..., index] = x[..., index].clamp(min=lower, max=upper)
        return x

    def get_stats(self):
        return self.stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.offset
        x = torch.clamp(x, -5.0, 5.0)
        return self._clip_selected_outputs(x)
    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._clip_selected_outputs(x)
        x = (x - self.offset) / self.scale
        return x

def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):

    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable_stats = convert_tensor(dataset_stats)

    # Atomic write: serialize to a temp file in the same dir, then os.replace().
    # The rename is atomic on POSIX, so a concurrent reader (e.g. the val dataset
    # on a non-main rank, built right after the train dataset writes this) never
    # observes a partially written file -> kills the JSONDecodeError race.
    tmp_path = f"{file_path}.tmp.{os.getpid()}"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)

def load_dataset_stats_from_json(file_path: str,
                                 try_convert_tensor: bool = True) -> Dict[str, Any]:

    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True
            first = obj[0]
            if isinstance(first, (int, float)):
                return all(isinstance(x, (int, float)) for x in obj)
            elif isinstance(first, list):
                return all(is_numeric_list(item) for item in obj)
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj

    # Resilient read: under multi-rank launch the train dataset's main rank writes
    # this file while other ranks build the val dataset and read it. The atomic
    # write above prevents partial content, but a reader can still arrive before
    # the file exists at all (main rank still computing stats). Retry briefly
    # rather than crashing with FileNotFoundError/JSONDecodeError.
    deadline = time.time() + 120.0
    while True:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            break
        except (FileNotFoundError, json.JSONDecodeError) as e:
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for a valid dataset stats file at {file_path}"
                ) from e
            time.sleep(1.0)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    data = dict_apply(
        data,
        lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) else x,
    )

    return data


def search_dataset_stats_cache_json(cache_dir: str | Path, data_config: DictConfig) -> Tuple[bool, str | None]:
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get_git_hash() -> Optional[str]:
        repo = Repo(__file__, search_parent_directories=True)
        return repo.head.commit.hexsha

    def to_plain(value: Any) -> Any:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def normalize_str_list(value: Any) -> List[str]:
        value = to_plain(value)
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(item) for item in value if item is not None]

    def normalize_transforms(value: Any) -> Any:
        value = to_plain(value)
        if isinstance(value, dict):
            return [value]
        return value

    def normalize_dataset_dirs(cfg: DictConfig) -> Any:
        dataset_cfg = cfg.get("dataset")
        if dataset_cfg is None:
            return None
        embodiment_datasets = dataset_cfg.get("embodiment_datasets")
        if embodiment_datasets is not None:
            emb_dirs: Dict[str, List[str]] = {}
            for emb, emb_cfg in embodiment_datasets.items():
                dataset_groups = emb_cfg.get("dataset_groups")
                if dataset_groups is None:
                    emb_dirs[emb] = []
                    continue
                dirs: List[str] = []
                for group in dataset_groups:
                    group_dirs = group.get("dataset_dirs")
                    if group_dirs is None:
                        continue
                    dirs.extend(normalize_str_list(group_dirs))
                emb_dirs[emb] = sorted(dirs)
            return emb_dirs

        dataset_dirs = dataset_cfg.get("dataset_dirs")
        return sorted(normalize_str_list(dataset_dirs))

    def normalize_action_state_transforms(cfg: DictConfig) -> Any:
        processor_cfg = cfg.get("processor")
        if processor_cfg is None:
            return None
        embodiment_processors = processor_cfg.get("embodiment_processors")
        if embodiment_processors is not None:
            emb_transforms: Dict[str, Any] = {}
            for emb, emb_cfg in embodiment_processors.items():
                transforms = emb_cfg.get("action_state_transforms")
                emb_transforms[emb] = normalize_transforms(transforms)
            return emb_transforms

        transforms = processor_cfg.get("action_state_transforms")
        return normalize_transforms(transforms)

    signature = {
        "action_size": data_config.dataset.action_size,
        "dataset_dirs": normalize_dataset_dirs(data_config),
        "action_state_transforms": normalize_action_state_transforms(data_config),
        # Prevent loss-aligned frame/anchor stats from reusing an episode-equal
        # legacy cache (or vice versa). Missing means the backward-compatible mode.
        "norm_stats_weighting": str(
            data_config.dataset.get("norm_stats_weighting", "legacy")
        ).lower(),
    }
    signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    dataset_hash = hashlib.sha256(signature_json.encode("utf-8"), usedforsecurity=False).hexdigest()

    git_hash = get_git_hash()
    precise_name = f"dataset_stats_{dataset_hash}_{git_hash}.json"
    precise = cache_dir / precise_name
    if precise.exists():
        logger.info(f"Found dataset stats cache with precisely matching dataset and git hash: {precise_name}.")
        return True, str(precise)

    candidates = sorted(cache_dir.glob(f"dataset_stats_{dataset_hash}_*.json"))
    if not candidates:
        logger.info(f"No dataset stats cache found for dataset hash {dataset_hash}")
        return False, str(precise) # return precise cache path for saving cache

    picked = candidates[0]
    prefix = f"dataset_stats_{dataset_hash}_"
    picked_git_hash = picked.name[len(prefix):-5]
    assert picked_git_hash != git_hash
    logger.warning(f"Found substitute dataset stats cache {picked.name} which mismatch current git hash {git_hash}.")
    return True, str(picked)
