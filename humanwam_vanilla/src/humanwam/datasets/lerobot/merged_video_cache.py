"""Pre-merged decoded-frame cache for RobotVideoDataset.

Moves all video decode + per-camera resize + multi-camera merge off the training
hot path into a one-time, parallelizable preprocessing pass. For each episode we
store the *already merged* frame stack -- exactly the float tensor the model
consumes -- as a per-episode float32 memmap:

    <cache_dir>/<dataset_name>_<sig>/episode_<ep:06d>.f32   # [ep_len, 3, Ho, Wo]

For valid source clips, the stored value is bit-for-bit
`RobotVideoDataset._assemble_frames` applied to the per-camera frames after the
processor's image `train_transforms`. Because _assemble_frames is per-frame
independent (resize/crop/normalize/concat have no temporal coupling),
precomputing per-frame and stacking a training window is identical to the
on-the-fly path. Stored as float32 (no uint8 quantization) so training is
numerically identical to the non-cached pipeline. A small set of pinned
RoboTwin AV1 files is shorter than its original episode metadata; the local
metadata/parquet corpus is trimmed to the shortest camera before cache build,
so no synthetic frames are introduced.

Per-episode files make the build embarrassingly parallel across nodes/shards and
idempotent (atomic rename + size check), and let lookup mmap only the episode it
needs (page cache keeps hot frames resident).  The training hot path assumes a
POSIX filesystem with efficient random mmap reads (tmpfs/local disk/GPFS).  Object
storage exposed through FUSE must be staged locally first: concurrent mmap page
faults across all DataLoader workers otherwise turn into rank stragglers and can
surface later as an apparently unrelated NCCL collective timeout.
"""
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from .lerobot.datasets.video_utils import decode_video_frames

logger = logging.getLogger(__name__)


def image_transforms_signature(image_transforms, camera_keys) -> str:
    """Return a stable signature for the selected cameras' image transforms."""
    entries = []
    for camera_key in camera_keys:
        camera_name = camera_key.removeprefix("observation.images.")
        transforms = (
            image_transforms.get(camera_name)
            if isinstance(image_transforms, dict)
            else image_transforms
        )
        if transforms is None:
            raise KeyError(f"no image transforms configured for {camera_key}")
        entries.append(
            {
                "camera_key": camera_key,
                "transforms": [
                    f"{type(transform).__module__}.{type(transform).__qualname__}:"
                    f"{transform!r}"
                    for transform in transforms
                ],
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_pyav_invalid_data_error(error: Exception) -> bool:
    error_type = type(error)
    return (
        error_type.__name__ == "InvalidDataError"
        and error_type.__module__.startswith("av")
    )


def _decode_video_frames_sequential_ffmpeg(
    path: Path,
    *,
    length: int,
    fps: float,
    timestamp_offset: float,
) -> torch.Tensor:
    """Decode a whole clip through an isolated ffmpeg process.

    Some RoboTwin AV1 clips deterministically raise PyAV
    ``avcodec_receive_frame`` InvalidDataError under concurrent in-process
    decoding even though their pinned Git-LFS SHA is correct and ffmpeg validates
    them sequentially. This fallback is intentionally restricted to offline cache
    construction; training never reaches it once the merged cache is complete.
    """
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    width_text, height_text = probe.stdout.strip().split("x", 1)
    width, height = int(width_text), int(height_text)
    frame_bytes = height * width * 3
    start = int(round(float(timestamp_offset) * float(fps)))
    stop = start + int(length)
    observed_usable: list[int] = []
    last_stderr = ""
    for attempt in range(1, 4):
        temp_dir = os.environ.get("TMPDIR")
        if temp_dir and not os.path.isdir(temp_dir):
            temp_dir = None
        with tempfile.TemporaryFile(prefix="fastwam-ffmpeg-", dir=temp_dir) as raw:
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-vsync",
                    "0",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ],
                check=False,
                stdout=raw,
                stderr=subprocess.PIPE,
            )
            last_stderr = decoded.stderr.decode("utf-8", errors="replace").strip()
            raw.flush()
            raw.seek(0, os.SEEK_END)
            raw_bytes = raw.tell()
            if decoded.returncode != 0:
                observed_usable.append(0)
                continue
            if raw_bytes % frame_bytes != 0:
                last_stderr = (
                    f"partial RGB frame: bytes={raw_bytes}, "
                    f"frame_bytes={frame_bytes}; {last_stderr}"
                )
                observed_usable.append(0)
                continue
            total_frames = raw_bytes // frame_bytes
            usable = max(0, min(stop, total_frames) - start)
            observed_usable.append(usable)
            if start < 0 or start >= total_frames or usable != length:
                continue
            raw.seek(start * frame_bytes)
            payload = raw.read(length * frame_bytes)
        array = np.frombuffer(payload, dtype=np.uint8).reshape(
            length, height, width, 3
        ).copy()
        if attempt > 1:
            logger.warning(
                "[video-cache] sequential single-thread ffmpeg recovered %s "
                "on attempt %s/3",
                path,
                attempt,
            )
        return torch.from_numpy(array).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)

    raise RuntimeError(
        f"{path} contains at most {max(observed_usable, default=0)} usable frames "
        f"from offset across 3 single-thread ffmpeg attempts but metadata requires "
        f"{length}. Run scripts/align_lerobot_video_tails.py; synthetic "
        f"terminal-frame padding is intentionally disabled. ffmpeg={last_stderr[-500:]}"
    )


def _unescape_mountinfo_path(value: str) -> str:
    """Decode the octal escapes used for mount points in /proc/self/mountinfo."""
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def filesystem_type_for_path(path) -> str | None:
    """Return the filesystem type for ``path`` using the current mount namespace.

    Reading mountinfo avoids shelling out from every distributed rank and follows
    symlinks before selecting the longest matching mount point.
    """
    resolved = os.path.realpath(os.fspath(path))
    best_mount = ""
    best_fstype = None
    try:
        with open("/proc/self/mountinfo", "r", encoding="utf-8") as f:
            for line in f:
                left, right = line.rstrip("\n").split(" - ", 1)
                fields = left.split()
                if len(fields) < 5:
                    continue
                mount_point = _unescape_mountinfo_path(fields[4])
                if resolved != mount_point and not resolved.startswith(mount_point.rstrip("/") + "/"):
                    continue
                if len(mount_point) > len(best_mount):
                    best_mount = mount_point
                    best_fstype = right.split()[0]
    except (OSError, ValueError):
        return None
    return best_fstype


def is_fuse_filesystem(filesystem_type: str | None) -> bool:
    return bool(filesystem_type) and (
        filesystem_type == "fuse" or filesystem_type.startswith("fuse.")
    )


class MergedFrameCache:
    VERSION = 2
    DTYPE = np.dtype("<f4")
    LAYOUT = "TCHW"
    LOGICAL_DTYPE = None
    PAYLOAD_SUFFIX = ".f32"

    def __init__(
        self,
        inner_dataset,
        out_hw,
        cache_dir,
        *,
        camera_keys=None,
        transform_signature=None,
    ):
        """`inner_dataset` is the underlying LeRobotDataset (has .root, .meta,
        .episode_data_index, .episodes, .tolerance_s, .video_backend).
        `out_hw` = (Ho, Wo) merged output resolution (e.g. (384, 320))."""
        self.inner = inner_dataset
        self.meta = inner_dataset.meta
        self.fps = inner_dataset.meta.fps
        self.tolerance_s = inner_dataset.tolerance_s
        self.video_backend = inner_dataset.video_backend
        # Camera order MUST match shape_meta / _assemble_frames expectations.
        self.camera_keys = list(
            inner_dataset.meta.video_keys if camera_keys is None else camera_keys
        )
        if not self.camera_keys:
            raise ValueError("MergedFrameCache requires at least one camera")
        unknown_cameras = sorted(set(self.camera_keys) - set(inner_dataset.meta.video_keys))
        if unknown_cameras:
            raise ValueError(
                f"cache cameras are absent from source metadata: {unknown_cameras}"
            )
        self.Ho, self.Wo = int(out_hw[0]), int(out_hw[1])
        self.transform_signature = str(transform_signature or "unspecified")

        episodes = (
            inner_dataset.episodes
            if inner_dataset.episodes is not None
            else list(range(inner_dataset.meta.total_episodes))
        )
        self.episodes = [int(e) for e in episodes]
        from_idx = inner_dataset.episode_data_index["from"]
        to_idx = inner_dataset.episode_data_index["to"]
        self.ep_lengths = {}
        for local_pos, ep in enumerate(self.episodes):
            self.ep_lengths[int(ep)] = int(to_idx[local_pos].item() - from_idx[local_pos].item())

        self.manifest = {
            "schema": "fastwam_merged_frame_cache",
            "version": self.VERSION,
            "source_root": str(Path(inner_dataset.root).resolve()),
            "video_backend": str(self.video_backend),
            "fps": float(self.fps),
            "camera_keys": self.camera_keys,
            "out_hw": [self.Ho, self.Wo],
            "dtype": self.DTYPE.str,
            "layout": self.LAYOUT,
            "transform_signature": self.transform_signature,
            "episodes": self.episodes,
            "episode_lengths": {
                str(ep): self.ep_lengths[ep] for ep in self.episodes
            },
        }
        if self.LOGICAL_DTYPE is not None:
            self.manifest["dtype"] = self.LOGICAL_DTYPE
            self.manifest["storage_dtype"] = self.DTYPE.str
        manifest_bytes = json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        sig = hashlib.sha256(manifest_bytes).hexdigest()[:16]
        self.dir = (
            Path(cache_dir)
            / f"{Path(inner_dataset.root).name}_merged_v{self.VERSION}_{sig}"
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        self._publish_or_validate_manifest()
        self._buffers = {}  # ep_idx -> read-only memmap
        # Staging is an optional local read path; source identity and all model
        # transforms stay unchanged. Capacity is fixed, with independent slots.
        self._local_stage = None
        stage_root = os.environ.get("FASTWAM_MERGED_CACHE_STAGE_DIR", "").strip()
        if stage_root:
            from .local_frame_stage import LocalFrameStage
            mode = os.environ.get("FASTWAM_MERGED_CACHE_STAGE_MODE", "slots")
            if mode != "slots":
                raise ValueError("legacy chunks/full staging is unsafe; use stage mode slots")
            budget = int(os.environ.get("FASTWAM_MERGED_CACHE_STAGE_MAX_BYTES", "0"))
            block = int(os.environ.get("FASTWAM_MERGED_CACHE_STAGE_CHUNK_FRAMES", "8"))
            stride = int(os.environ.get("FASTWAM_MERGED_CACHE_STAGE_STRIDE", "4"))
            identity = {
                "source_cache": str(self.dir.resolve()),
                "source_manifest_sha256": hashlib.sha256(self._manifest_path().read_bytes()).hexdigest(),
                "schema": "fixed_slots_v2", "budget": budget, "block": block, "stride": stride,
            }
            signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            stage_dir = Path(stage_root) / f"{self.dir.name}_slots_{signature}"
            self._local_stage = LocalFrameStage(
                stage_dir, max_bytes=budget,
                frame_bytes=3 * self.Ho * self.Wo * self.DTYPE.itemsize,
                block_frames=block, stride=stride,
            )
            manifest = stage_dir / "STAGE_MANIFEST.json"
            # Identical publishers can race safely; no hot-path global lock.
            tmp = manifest.with_name(f".{manifest.name}.{os.getpid()}")
            tmp.write_text(json.dumps(identity, sort_keys=True) + "\n")
            os.replace(tmp, manifest)

    # ---- paths / validity ----
    def _manifest_path(self):
        return self.dir / "CACHE_MANIFEST.json"

    def _publish_or_validate_manifest(self):
        path = self._manifest_path()
        if path.exists():
            observed = json.loads(path.read_text(encoding="utf-8"))
            if observed != self.manifest:
                raise RuntimeError(
                    f"merged cache manifest mismatch under {self.dir}"
                )
            return
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def _ep_path(self, ep):
        return self.dir / f"episode_{ep:06d}{self.PAYLOAD_SUFFIX}"

    def _ep_checksum_path(self, ep):
        payload = self._ep_path(ep)
        return payload.with_suffix(payload.suffix + ".sha256")

    def _ep_lock_path(self, ep):
        return self.dir / f"episode_{ep:06d}.lock"

    def _ep_nbytes(self, ep):
        return self.ep_lengths[ep] * self.Ho * self.Wo * 3 * self.DTYPE.itemsize

    def ep_done(self, ep):
        p = self._ep_path(ep)
        checksum_path = self._ep_checksum_path(ep)
        if not p.exists() or p.stat().st_size != self._ep_nbytes(ep):
            return False
        if not checksum_path.is_file():
            return False
        fields = checksum_path.read_text(encoding="ascii").strip().split()
        return (
            len(fields) == 2
            and len(fields[0]) == 64
            and all(char in "0123456789abcdef" for char in fields[0])
            and fields[1] == p.name
        )

    def missing_episodes(self, episodes=None):
        eps = self.episodes if episodes is None else episodes
        return [e for e in eps if not self.ep_done(e)]

    def is_complete(self):
        return len(self.missing_episodes()) == 0

    def expected_nbytes(self):
        return sum(self._ep_nbytes(ep) for ep in self.episodes)

    def validate(self, *, verify_checksums=True):
        """Validate shape/size sidecars and optionally re-hash every payload."""
        failures = []
        checked_bytes = 0
        for ep in self.episodes:
            path = self._ep_path(ep)
            if not self.ep_done(ep):
                failures.append(f"episode {ep}: missing/invalid size or checksum sidecar")
                continue
            checked_bytes += path.stat().st_size
            if not verify_checksums:
                continue
            expected = self._ep_checksum_path(ep).read_text(
                encoding="ascii"
            ).split()[0]
            digest = hashlib.sha256()
            with path.open("rb") as payload:
                while chunk := payload.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                failures.append(f"episode {ep}: sha256 mismatch")
        if failures:
            preview = "; ".join(failures[:5])
            raise RuntimeError(
                f"merged cache validation failed ({len(failures)} episodes): {preview}"
            )
        return {
            "episodes": len(self.episodes),
            "bytes": checked_bytes,
            "checksums_verified": bool(verify_checksums),
        }

    def validate_training_storage(self):
        """Reject FUSE-backed object storage for the mmap training hot path.

        A deliberate diagnostic override remains available, but it is not a
        supported production configuration.
        """
        filesystem_type = filesystem_type_for_path(self.dir)
        if not is_fuse_filesystem(filesystem_type):
            return
        message = (
            f"[merged-cache] {self.dir} is on {filesystem_type}, but training reads "
            "hundreds of large per-episode .f32 files through concurrent np.memmap "
            "random access. FUSE/object-storage mounts can stall one DataLoader rank "
            "and make the other ranks fail later with an NCCL collective timeout. "
            "Stage the cache to /dev/shm, local NVMe, or GPFS and point "
            "merged_video_cache_dir there. scripts/train_5task.sh supports "
            "STAGE_VIDEO_CACHE=1 for this purpose."
        )
        if os.environ.get("FASTWAM_MERGED_ALLOW_REMOTE_FUSE", "0") == "1":
            logger.warning("%s Override FASTWAM_MERGED_ALLOW_REMOTE_FUSE=1 is active.", message)
            return
        raise RuntimeError(message)

    # ---- build ----
    def build(
        self,
        assemble_fn,
        image_transforms,
        shard=(0, 1),
        workers=64,
        log_every=100,
        progress_callback=None,
    ):
        """Decode + transform + merge every (missing) episode in this shard and
        write its per-episode float32 memmap.

        assemble_fn:     RobotVideoDataset._assemble_frames (bound method).
        image_transforms: the processor's image transform list (ToTensor, Resize)
                          applied per camera, exactly as training does.
        shard=(i, n):    build episodes where (position % n == i) for parallelism.
        """
        i, n = shard
        my_eps = [e for pos, e in enumerate(self.episodes) if pos % n == i]
        todo = self.missing_episodes(my_eps)
        logger.info(
            f"[merged-cache] shard {i}/{n}: {len(todo)}/{len(my_eps)} episodes to build "
            f"-> {self.dir} (out {self.Ho}x{self.Wo}, ~{self._ep_nbytes(self.episodes[0])/1e6:.0f} MB/ep)"
        )

        def _build_one(ep):
            lock_path = self._ep_lock_path(ep)
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if self.ep_done(ep):
                    return False, self.ep_lengths[ep]

                length = self.ep_lengths[ep]
                per_cam = []
                for cam in self.camera_keys:
                    ts_off = self.meta.get_video_timestamp_offset(ep, cam)
                    ts = [ts_off + k / self.fps for k in range(length)]
                    path = self.inner.root / self.meta.get_video_file_path(ep, cam)
                    try:
                        frames = decode_video_frames(
                            path,
                            ts,
                            self.tolerance_s,
                            self.video_backend,
                        )  # [T,C,Hraw,Wraw]
                    except Exception as error:
                        if not _is_pyav_invalid_data_error(error):
                            raise
                        logger.warning(
                            "[merged-cache] PyAV rejected %s episode=%s camera=%s; "
                            "retrying once through sequential ffmpeg",
                            path,
                            ep,
                            cam,
                        )
                        frames = _decode_video_frames_sequential_ffmpeg(
                            path,
                            length=length,
                            fps=self.fps,
                            timestamp_offset=ts_off,
                        )
                    camera_name = cam.removeprefix("observation.images.")
                    transforms = (
                        image_transforms.get(camera_name)
                        if isinstance(image_transforms, dict)
                        else image_transforms
                    )
                    if transforms is None:
                        raise KeyError(f"no image transforms configured for {cam}")
                    for transform in transforms:
                        if hasattr(transform, "apply_with_context"):
                            tasks = self.meta.episodes[ep]["tasks"]
                            if len(tasks) != 1:
                                raise ValueError(f"Context transform requires one task per episode: {ep}")
                            frames = transform.apply_with_context(frames, task=tasks[0])
                        else:
                            frames = transform(frames)
                    per_cam.append(frames)
                multicam = torch.stack(per_cam, dim=0)
                merged = assemble_fn(multicam)
                arr = self._encode_storage(merged)
                assert arr.shape == (
                    length,
                    3,
                    self.Ho,
                    self.Wo,
                ), f"ep {ep}: got {arr.shape}"
                digest = hashlib.sha256(memoryview(arr).cast("B")).hexdigest()
                final_path = self._ep_path(ep)
                tmp = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
                checksum_path = self._ep_checksum_path(ep)
                checksum_tmp = checksum_path.with_name(
                    f"{checksum_path.name}.tmp.{os.getpid()}"
                )
                try:
                    mm = np.memmap(
                        tmp,
                        dtype=self.DTYPE,
                        mode="w+",
                        shape=(length, 3, self.Ho, self.Wo),
                    )
                    mm[:] = arr
                    mm.flush()
                    del mm
                    with tmp.open("rb") as payload:
                        os.fsync(payload.fileno())
                    os.replace(tmp, final_path)
                    checksum_tmp.write_text(
                        f"{digest}  {final_path.name}\n", encoding="ascii"
                    )
                    os.replace(checksum_tmp, checksum_path)
                finally:
                    tmp.unlink(missing_ok=True)
                    checksum_tmp.unlink(missing_ok=True)
                return True, length

        done = 0
        processed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_build_one, ep): ep for ep in todo}
            for k, fut in enumerate(as_completed(futs)):
                try:
                    built, _ = fut.result()
                except Exception as e:  # noqa
                    logger.error(f"[merged-cache] episode {futs[fut]} FAILED: {e!r}")
                    raise
                done += int(built)
                processed += 1
                if processed % log_every == 0 or processed == len(todo):
                    logger.info(
                        f"[merged-cache] shard {i}/{n}: processed "
                        f"{processed}/{len(todo)}, built {done}"
                    )
                    if progress_callback is not None:
                        progress_callback()
        return done

    # ---- lookup (training hot path) ----
    def _encode_storage(self, merged: torch.Tensor) -> np.ndarray:
        return (
            merged.permute(1, 0, 2, 3)
            .contiguous()
            .to(torch.float32)
            .numpy()
            .astype(self.DTYPE, copy=False)
        )

    def _decode_storage(self, value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(value, copy=True))

    def _buf(self, ep_idx):
        ep_idx = int(ep_idx)
        buf = self._buffers.get(ep_idx)
        if buf is None:
            if not self.ep_done(ep_idx):
                raise FileNotFoundError(f"[merged-cache] missing episode {ep_idx}: {self._ep_path(ep_idx)}")
            storage_path = self._ep_path(ep_idx)
            buf = np.memmap(
                storage_path, dtype=self.DTYPE, mode="r",
                shape=(self.ep_lengths[ep_idx], 3, self.Ho, self.Wo),
            )
            self._buffers[ep_idx] = buf
        return buf

    def _fd(self, ep_idx):
        # Keyed by pid so forked DataLoader workers never share descriptors;
        # bounded to stay far below the per-process open-file limit.
        fds = self.__dict__.setdefault("_fds", {})
        key = (os.getpid(), ep_idx)
        fd = fds.pop(key, None)
        if fd is None:
            if not self.ep_done(ep_idx):
                raise FileNotFoundError(f"[merged-cache] missing episode {ep_idx}: {self._ep_path(ep_idx)}")
            fd = os.open(self._ep_path(ep_idx), os.O_RDONLY)
            while len(fds) >= 256:
                old_key = next(iter(fds))
                old_fd = fds.pop(old_key)
                if old_key[0] == key[0]:
                    os.close(old_fd)
        fds[key] = fd
        return fd

    def _pread_frames(self, ep_idx: int, indices: np.ndarray) -> np.ndarray:
        """Read selected frames with one pread per contiguous run.

        Avoids GPFS mmap page-fault serialization across many DataLoader
        workers; returned bytes are identical to the memmap path.
        """
        indices = np.asarray(indices, dtype=np.int64)
        frame_shape = (3, self.Ho, self.Wo)
        frame_bytes = 3 * self.Ho * self.Wo * self.DTYPE.itemsize
        out = np.empty((len(indices),) + frame_shape, dtype=self.DTYPE)
        if not len(indices):
            return out
        if indices.min() < 0 or indices.max() >= self.ep_lengths[int(ep_idx)]:
            raise IndexError(f"frame indices out of range for episode {ep_idx}")
        fd = self._fd(int(ep_idx))
        flat = out.reshape(len(indices), frame_bytes // self.DTYPE.itemsize)
        pos = 0
        while pos < len(indices):
            end = pos + 1
            while end < len(indices) and indices[end] == indices[end - 1] + 1:
                end += 1
            view = memoryview(flat[pos:end]).cast("B")
            offset = int(indices[pos]) * frame_bytes
            while view:
                n = os.preadv(fd, [view], offset)
                if n <= 0:
                    raise OSError(f"[merged-cache] short read in episode {ep_idx}")
                view, offset = view[n:], offset + n
            pos = end
        return out

    def _lookup_storage(self, ep_idx: int, indices: np.ndarray) -> np.ndarray:
        if self._local_stage is None:
            if os.environ.get("FASTWAM_MERGED_CACHE_READ", "mmap") == "pread":
                return self._pread_frames(ep_idx, indices)
            return self._buf(ep_idx)[indices]
        return self._local_stage.lookup(
            self._ep_path(ep_idx), ep_idx, self.ep_lengths[ep_idx], indices,
            dtype=self.DTYPE, frame_shape=(3, self.Ho, self.Wo),
        )

    def _ts_to_indices(self, query_ts, ep_len):
        # Same mapping as _DiskVideoFrameCache.lookup: round(ts * fps), clip.
        idx = np.asarray(query_ts, dtype=np.float64) * self.fps
        return np.clip(np.round(idx), 0, ep_len - 1).astype(np.int64)

    def lookup_window(
        self,
        query_timestamps: dict,
        ep_idx: int,
        frame_positions=None,
    ) -> torch.Tensor:
        """Return merged window frames [T, 3, Ho, Wo] in [-1, 1] for the given
        per-camera query timestamps (all cameras share frame indices).

        ``frame_positions`` optionally selects positions inside the query window
        before touching the memmap.  RoboTwin asks LeRobot for 33 trajectory
        frames but the model consumes only positions [0,4,...,32]; selecting here
        avoids reading and copying the other 24 full-resolution float frames.
        """
        ep_idx = int(ep_idx)
        ep_len = self.ep_lengths[ep_idx]
        any_ts = next(iter(query_timestamps.values()))
        indices = self._ts_to_indices(any_ts, ep_len)
        if frame_positions is not None:
            positions = np.asarray(frame_positions, dtype=np.int64)
            if positions.ndim != 1:
                raise ValueError(f"frame_positions must be 1D, got shape {positions.shape}")
            if positions.size and (positions.min() < 0 or positions.max() >= len(indices)):
                raise IndexError(
                    f"frame_positions {positions.tolist()} out of range for query window of length {len(indices)}"
                )
            indices = indices[positions]
        return self._decode_storage(self._lookup_storage(ep_idx, indices))

    def lookup_frame(self, ep_idx: int, frame_index: int) -> torch.Tensor:
        """Return a single merged frame [3, Ho, Wo] in [-1, 1]."""
        ep_idx = int(ep_idx)
        ep_len = self.ep_lengths[ep_idx]
        fi = int(np.clip(frame_index, 0, ep_len - 1))
        return self._decode_storage(self._lookup_storage(ep_idx, np.asarray([fi], dtype=np.int64))[0])


class BFloat16MergedFrameCache(MergedFrameCache):
    """Model-input-equivalent pixel cache stored as raw BF16 payload bits."""

    VERSION = 3
    DTYPE = np.dtype("<u2")
    LOGICAL_DTYPE = "bfloat16"
    PAYLOAD_SUFFIX = ".bf16"

    def _encode_storage(self, merged: torch.Tensor) -> np.ndarray:
        return (
            merged.permute(1, 0, 2, 3)
            .contiguous()
            .to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
            .astype(self.DTYPE, copy=False)
        )

    def _decode_storage(self, value: np.ndarray) -> torch.Tensor:
        # pread already returns a private writable array; memmap slices are copied.
        if isinstance(value, np.memmap) or not value.flags.writeable or value.dtype != self.DTYPE:
            value = np.array(value, dtype=self.DTYPE, copy=True)
        return torch.from_numpy(value).view(torch.bfloat16)


class TShapeMp4FrameCache:
    """Pre-merged RoboTwin T canvas stored as one compressed MP4 per episode.

    Each frame is head [256,352] over left|right wrist [128,176] (= [384,352]),
    built offline from the BF16 separate-view cache with the model's own wrist
    resize (bilinear, antialias). Training decodes only the selected frames of a
    ~1 MB file instead of reading ~15 MB of raw BF16 per sample. Frame i has
    timestamp i/fps. Returns [T,3,384,352] bfloat16 in [-1,1], fed to
    the model with ``latent_tshape_premerged_input=True``.
    """

    SCHEMA = "fastwam_tshape_mp4_v1"
    PREMERGED_T = True

    def __init__(self, inner_dataset, out_hw, cache_dir, *, camera_keys=None, transform_signature=None):
        self.inner = inner_dataset
        self.fps = inner_dataset.meta.fps
        self.Ho, self.Wo = int(out_hw[0]), int(out_hw[1])
        root = Path(inner_dataset.root).resolve()
        self.dir = Path(cache_dir) / root.name
        manifest = json.loads((self.dir / "manifest.json").read_text())
        if manifest.get("schema") != self.SCHEMA:
            raise ValueError(f"{self.dir}: not a {self.SCHEMA} cache")
        if manifest["source_root"] != str(root):
            raise ValueError(f"{self.dir}: built for {manifest['source_root']}, not {root}")
        if list(manifest["out_hw"]) != [self.Ho, self.Wo]:
            raise ValueError(f"{self.dir}: out_hw {manifest['out_hw']} != {[self.Ho, self.Wo]}")
        episodes = inner_dataset.episodes if inner_dataset.episodes is not None else list(range(inner_dataset.meta.total_episodes))
        self.episodes = [int(e) for e in episodes]
        from_idx, to_idx = inner_dataset.episode_data_index["from"], inner_dataset.episode_data_index["to"]
        self.ep_lengths = {ep: int(to_idx[i].item() - from_idx[i].item()) for i, ep in enumerate(self.episodes)}
        for ep, n in self.ep_lengths.items():
            if int(manifest["episode_lengths"].get(str(ep), -1)) != n:
                raise ValueError(f"{self.dir}: episode {ep} length mismatch")

    def _ep_path(self, ep):
        return self.dir / f"episode_{int(ep):06d}.mp4"

    def ep_done(self, ep):
        return self._ep_path(ep).is_file() and self._ep_path(ep).with_suffix(".ok").is_file()

    def missing_episodes(self, episodes=None):
        return [e for e in (self.episodes if episodes is None else episodes) if not self.ep_done(e)]

    def is_complete(self):
        return not self.missing_episodes()

    def validate_training_storage(self):
        return None

    def _ts_to_indices(self, query_ts, ep_len):
        idx = np.asarray(query_ts, dtype=np.float64) * self.fps
        return np.clip(np.round(idx), 0, ep_len - 1).astype(np.int64)

    def _decode(self, ep_idx, indices):
        import av
        wanted = sorted(set(int(i) for i in indices))
        frames = {}
        with av.open(str(self._ep_path(ep_idx))) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = stream.time_base
            container.seek(int(wanted[0] / self.fps / tb), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                i = int(round(float(frame.pts * tb) * self.fps))
                if i in frames or i not in wanted:
                    if i > wanted[-1]:
                        break
                    continue
                frames[i] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break
        missing = [i for i in wanted if i not in frames]
        if missing:
            raise RuntimeError(f"[tshape-mp4] episode {ep_idx}: frames {missing[:4]} not decoded from {self._ep_path(ep_idx)}")
        out = np.stack([frames[int(i)] for i in indices])  # [T,H,W,3] uint8
        if out.shape[1:3] != (self.Ho, self.Wo):
            raise RuntimeError(f"[tshape-mp4] decoded {out.shape[1:3]}, expected {(self.Ho, self.Wo)}")
        return torch.from_numpy(out).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).to(torch.bfloat16)

    def lookup_window(self, query_timestamps, ep_idx, frame_positions=None):
        ep_idx = int(ep_idx)
        indices = self._ts_to_indices(next(iter(query_timestamps.values())), self.ep_lengths[ep_idx])
        if frame_positions is not None:
            indices = indices[np.asarray(frame_positions, dtype=np.int64)]
        return self._decode(ep_idx, indices)

    def lookup_frame(self, ep_idx, frame_index):
        ep_idx = int(ep_idx)
        fi = int(np.clip(frame_index, 0, self.ep_lengths[ep_idx] - 1))
        return self._decode(ep_idx, [fi])[0]
