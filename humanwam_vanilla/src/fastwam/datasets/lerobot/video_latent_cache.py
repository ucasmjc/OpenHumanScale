"""Read-only BF16 mmap cache for deterministic frozen video-VAE latents."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


SCHEMA = "fastwam_video_latent_cache_v1"
STORAGE_DTYPE = np.dtype("<u2")


def sha256_file(path: Path, chunk_bytes: int = 64 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def bf16_tensor_to_storage(value: torch.Tensor) -> np.ndarray:
    """Return a NumPy uint16 view preserving every BF16 payload bit."""
    return (
        value.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype(STORAGE_DTYPE, copy=False)
    )


def storage_to_bf16_tensor(value: np.ndarray) -> torch.Tensor:
    """Copy a mmap slice and reinterpret its uint16 payload as BF16."""
    storage = np.array(value, dtype=STORAGE_DTYPE, copy=True)
    return torch.from_numpy(storage).view(torch.bfloat16)


class VideoLatentCache:
    """Lazily mmap contiguous sample-index shards produced by the frozen VAE."""

    def __init__(
        self,
        cache_dir: str | os.PathLike,
        *,
        expected_samples: int | None = None,
        expected_video_frames: int | None = None,
        expected_segment_manifest_sha256: str | None = None,
    ) -> None:
        self.root = Path(cache_dir)
        manifest_path = self.root / "CACHE_MANIFEST.json"
        complete_path = self.root / "COMPLETE.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"latent cache manifest is absent: {manifest_path}")
        if not complete_path.is_file():
            raise FileNotFoundError(f"latent cache completion marker is absent: {complete_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA:
            raise RuntimeError(
                f"unsupported latent cache schema: {self.manifest.get('schema')!r}"
            )
        if complete.get("status") != "PASS" or complete.get("complete") is not True:
            raise RuntimeError(f"latent cache is not complete: {complete_path}")
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if complete.get("manifest_sha256") != manifest_digest:
            raise RuntimeError("latent cache manifest checksum mismatch")
        if self.manifest.get("dtype") != "bfloat16":
            raise RuntimeError(
                f"latent cache must contain bfloat16, got {self.manifest.get('dtype')}"
            )
        if self.manifest.get("storage_dtype") != STORAGE_DTYPE.str:
            raise RuntimeError(
                "latent cache storage dtype mismatch: "
                f"{self.manifest.get('storage_dtype')} != {STORAGE_DTYPE.str}"
            )
        self.samples = int(self.manifest["samples"])
        self.video_frames = int(self.manifest["video_frames"])
        self.latent_shape = tuple(int(value) for value in self.manifest["latent_shape"])
        if len(self.latent_shape) != 4:
            raise RuntimeError(
                f"latent_shape must be [C,T,H,W], got {self.latent_shape}"
            )
        if expected_samples is not None and self.samples != int(expected_samples):
            raise RuntimeError(
                f"latent sample count mismatch: cache={self.samples} "
                f"dataset={expected_samples}"
            )
        if (
            expected_video_frames is not None
            and self.video_frames != int(expected_video_frames)
        ):
            raise RuntimeError(
                f"latent video-frame mismatch: cache={self.video_frames} "
                f"dataset={expected_video_frames}"
            )
        if expected_segment_manifest_sha256 is not None:
            observed = self.manifest.get("segment_manifest_sha256")
            if observed != expected_segment_manifest_sha256:
                raise RuntimeError(
                    "latent segment-manifest checksum mismatch: "
                    f"cache={observed!r} dataset={expected_segment_manifest_sha256!r}"
                )

        self.shards = tuple(self.manifest["shards"])
        if not self.shards:
            raise RuntimeError("latent cache has no shards")
        expected_start = 0
        self._stops = []
        for shard in self.shards:
            start, stop = int(shard["start"]), int(shard["stop"])
            if start != expected_start or stop <= start:
                raise RuntimeError(
                    f"latent shard ranges are not contiguous at {shard!r}"
                )
            path = self.root / shard["file"]
            expected_bytes = (
                (stop - start)
                * int(np.prod(self.latent_shape))
                * STORAGE_DTYPE.itemsize
            )
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"latent shard is absent/truncated: {path} "
                    f"expected_bytes={expected_bytes}"
                )
            sidecar = self.root / shard["checksum_file"]
            fields = (
                sidecar.read_text(encoding="ascii").strip().split()
                if sidecar.is_file()
                else []
            )
            if (
                len(fields) != 2
                or fields[0] != shard["sha256"]
                or fields[1] != path.name
            ):
                raise RuntimeError(f"latent shard checksum sidecar mismatch: {sidecar}")
            expected_start = stop
            self._stops.append(stop)
        if expected_start != self.samples:
            raise RuntimeError(
                f"latent shard coverage ends at {expected_start}, expected {self.samples}"
            )
        self._buffers: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.samples

    def _buffer(self, shard_index: int) -> np.memmap:
        buffer = self._buffers.get(shard_index)
        if buffer is None:
            shard = self.shards[shard_index]
            start, stop = int(shard["start"]), int(shard["stop"])
            buffer = np.memmap(
                self.root / shard["file"],
                dtype=STORAGE_DTYPE,
                mode="r",
                shape=(stop - start, *self.latent_shape),
            )
            self._buffers[shard_index] = buffer
        return buffer

    def lookup(self, sample_index: int) -> torch.Tensor:
        index = int(sample_index)
        if not 0 <= index < self.samples:
            raise IndexError(
                f"latent sample index {index} is outside [0, {self.samples})"
            )
        shard_index = bisect.bisect_right(self._stops, index)
        shard = self.shards[shard_index]
        local_index = index - int(shard["start"])
        return storage_to_bf16_tensor(self._buffer(shard_index)[local_index])
