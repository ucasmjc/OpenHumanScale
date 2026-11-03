"""Checkpoint retention and verified synchronous/asynchronous publishing.

Weights are first written to a normal filesystem path by the trainer.  They are
then uploaded with a bounded ``ossutil`` subprocess, verified by object size,
and only then become eligible for local deletion.  Training-state directories
are pruned independently and are never uploaded by this module.

``AsyncCheckpointMirror`` adds an ordered, rank-agnostic background publisher.
The trainer is responsible for constructing and enqueueing it on global rank
zero only.  The mirror deliberately accepts only inference-weight artifacts;
optimizer/RNG state directories are outside its contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Iterable
import uuid


_STEP_RE = re.compile(
    r"^step_(\d+)(?:\.(?:adapter|bottleneck))?(?:\.pt)?$"
)
_WEIGHT_RE = re.compile(r"^step_(\d+)(?:\.(?:adapter|bottleneck))?\.pt$")
_MIRROR_FORMAT = "fastwam_checkpoint_mirror_v1"
_COPY_BUFFER_BYTES = 16 * 1024 * 1024

logger = logging.getLogger(__name__)


def checkpoint_step(path: os.PathLike[str] | str) -> int | None:
    match = _STEP_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def _sorted_steps(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        (path for path in paths if checkpoint_step(path) is not None),
        key=lambda path: checkpoint_step(path) or -1,
    )


def prune_training_states(state_dir: os.PathLike[str] | str, keep_last: int) -> list[Path]:
    """Delete complete trainer states except the newest ``keep_last`` entries."""
    if keep_last < 0:
        raise ValueError(f"keep_last must be >= 0, got {keep_last}")
    root = Path(state_dir)
    if not root.is_dir():
        return []
    complete = _sorted_steps(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "trainer_state.json").is_file()
    )
    victims = complete[:-keep_last] if keep_last else complete
    for path in victims:
        shutil.rmtree(path)
    return victims


@dataclass(frozen=True)
class UploadResult:
    local_path: Path
    oss_uri: str
    size: int
    etag: str | None
    skipped: bool = False


class OssCheckpointStore:
    def __init__(
        self,
        *,
        root_uri: str,
        output_dir: os.PathLike[str] | str,
        ossutil_bin: os.PathLike[str] | str,
        config_file: os.PathLike[str] | str,
        region: str = "cn-shanghai",
        endpoint: str = "oss-cn-shanghai.aliyuncs.com",
        timeout_seconds: int = 900,
        retry_times: int = 3,
        parallel: int = 16,
        force_overwrite: bool = True,
    ) -> None:
        if not root_uri.startswith("oss://"):
            raise ValueError(f"root_uri must start with oss://, got {root_uri!r}")
        self.root_uri = root_uri.rstrip("/")
        self.output_dir = Path(output_dir)
        self.ossutil_bin = str(ossutil_bin)
        self.config_file = str(config_file)
        self.region = region
        self.endpoint = endpoint
        self.timeout_seconds = int(timeout_seconds)
        self.retry_times = int(retry_times)
        self.parallel = int(parallel)
        self.force_overwrite = bool(force_overwrite)

    def oss_uri(self, local_path: os.PathLike[str] | str) -> str:
        local = Path(local_path)
        task_name = self.output_dir.parent.name
        run_name = self.output_dir.name
        return (
            f"{self.root_uri}/{task_name}/{run_name}/checkpoints/weights/{local.name}"
        )

    @property
    def backend_name(self) -> str:
        return "oss"

    def destination(self, local_path: os.PathLike[str] | str) -> str:
        """Return the remote location without mutating remote state."""
        return self.oss_uri(local_path)

    def _common_args(self) -> list[str]:
        return [
            "-c",
            self.config_file,
            "--region",
            self.region,
            "-e",
            self.endpoint,
            "--retry-times",
            str(self.retry_times),
        ]

    def stat(self, oss_uri: str) -> dict:
        command = [
            self.ossutil_bin,
            "stat",
            oss_uri,
            *self._common_args(),
            "--output-format",
            "json",
        ]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        # ossutil v2 appends a human-readable ``0.123(s) elapsed`` line even
        # when JSON output is requested.  Decode only the first JSON value.
        metadata, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
        return metadata

    def try_stat(self, oss_uri: str) -> dict | None:
        """Return object metadata, distinguishing a real 404 from I/O errors."""
        try:
            return self.stat(oss_uri)
        except subprocess.CalledProcessError as exc:
            detail = f"{exc.stdout or ''}\n{exc.stderr or ''}"
            if "NoSuchKey" in detail or "Http Status Code: 404" in detail:
                return None
            raise

    def probe(self) -> None:
        """Fail fast if the configured OSS root cannot be listed.

        This is intentionally a read-only, one-result listing.  Calling it at
        trainer startup avoids discovering invalid credentials only when the
        first (potentially hours-later) checkpoint is published.
        """
        command = [
            self.ossutil_bin,
            "ls",
            self.root_uri,
            "--limited-num",
            "1",
            *self._common_args(),
        ]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )

    def upload(self, local_path: os.PathLike[str] | str) -> UploadResult:
        local = Path(local_path)
        if not local.is_file():
            raise FileNotFoundError(local)
        local_size = local.stat().st_size
        oss_uri = self.oss_uri(local)
        if not self.force_overwrite:
            existing = self.try_stat(oss_uri)
            if existing is not None:
                manifest_path = self.manifest_path(local)
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if (
                        manifest.get("oss_uri") == oss_uri
                        and int(manifest.get("size", -1)) == local_size
                        and int(existing["Content-Length"]) == local_size
                    ):
                        return UploadResult(
                            local_path=local,
                            oss_uri=oss_uri,
                            size=local_size,
                            etag=existing.get("Etag"),
                            skipped=True,
                        )
                raise FileExistsError(
                    "Refusing to overwrite an existing checkpoint without a "
                    f"matching verified local manifest: {oss_uri}"
                )
        command = [
            self.ossutil_bin,
            "cp",
            str(local),
            oss_uri,
        ]
        if self.force_overwrite:
            command.append("-f")
        command.extend(
            [*self._common_args(), "--parallel", str(self.parallel)]
        )
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # A multipart completion can reach OSS just before the local
            # process times out.  Only recover this case when a preflight stat
            # proved the destination absent and the completed size matches.
            if self.force_overwrite:
                raise
            recovered = self.try_stat(oss_uri)
            if recovered is None or int(recovered["Content-Length"]) != local_size:
                raise
        metadata = self.stat(oss_uri)
        remote_size = int(metadata["Content-Length"])
        if remote_size != local_size:
            raise RuntimeError(
                f"OSS size mismatch for {oss_uri}: local={local_size} remote={remote_size}"
            )
        result = UploadResult(
            local_path=local,
            oss_uri=oss_uri,
            size=local_size,
            etag=metadata.get("Etag"),
        )
        self._write_manifest(result)
        return result

    def publish(self, local_path: os.PathLike[str] | str) -> UploadResult:
        """Backend-neutral alias used by :class:`AsyncCheckpointMirror`."""
        return self.upload(local_path)

    @staticmethod
    def manifest_path(local_path: os.PathLike[str] | str) -> Path:
        return Path(f"{local_path}.oss.json")

    def _write_manifest(self, result: UploadResult) -> None:
        manifest = self.manifest_path(result.local_path)
        temporary = manifest.with_suffix(manifest.suffix + ".tmp")
        payload = {
            "local_path": str(result.local_path),
            "oss_uri": result.oss_uri,
            "size": result.size,
            "etag": result.etag,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, manifest)


@dataclass(frozen=True)
class FilesystemCopyResult:
    local_path: Path
    destination_path: Path
    size: int
    skipped: bool = False


class FilesystemCheckpointStore:
    """Atomically copy checkpoint weights into a mounted filesystem root.

    ``root_dir`` itself must already exist.  This prevents a missing BOS mount
    from silently turning into an ordinary directory on ephemeral node storage.
    Directories below the verified root are created on demand.
    """

    def __init__(
        self,
        *,
        root_dir: os.PathLike[str] | str,
        output_dir: os.PathLike[str] | str,
    ) -> None:
        root = Path(root_dir).expanduser()
        if not root.exists():
            raise FileNotFoundError(
                f"Checkpoint mirror root does not exist (is BOS mounted?): {root}"
            )
        if not root.is_dir():
            raise NotADirectoryError(root)
        if not os.access(root, os.W_OK):
            raise PermissionError(f"Checkpoint mirror root is not writable: {root}")
        write_probe = root / f".fastwam-mirror-write-probe-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            with write_probe.open("xb") as stream:
                stream.write(b"fastwam-checkpoint-mirror-probe\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            raise PermissionError(
                f"Checkpoint mirror root failed a write probe: {root}"
            ) from exc
        finally:
            write_probe.unlink(missing_ok=True)
        self.root_dir = root
        self.output_dir = Path(output_dir)
        if not self.output_dir.name or not self.output_dir.parent.name:
            raise ValueError(
                f"output_dir must include task/run components, got {self.output_dir}"
            )

    @property
    def backend_name(self) -> str:
        return "filesystem"

    def destination_path(self, local_path: os.PathLike[str] | str) -> Path:
        local = Path(local_path)
        task_name = self.output_dir.parent.name
        run_name = self.output_dir.name
        return (
            self.root_dir
            / task_name
            / run_name
            / "checkpoints"
            / "weights"
            / local.name
        )

    def destination(self, local_path: os.PathLike[str] | str) -> str:
        return str(self.destination_path(local_path))

    def publish(
        self, local_path: os.PathLike[str] | str
    ) -> FilesystemCopyResult:
        local = Path(local_path)
        if not local.is_file():
            raise FileNotFoundError(local)
        destination = self.destination_path(local)
        if os.path.abspath(local) == os.path.abspath(destination):
            raise ValueError(
                f"Checkpoint mirror source and destination are identical: {local}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing mirrored checkpoint: "
                f"{destination}"
            )
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            with local.open("rb") as source, temporary.open("xb") as target:
                source_before = os.fstat(source.fileno())
                shutil.copyfileobj(source, target, length=_COPY_BUFFER_BYTES)
                target.flush()
                os.fsync(target.fileno())
                source_after = os.fstat(source.fileno())
            if (
                source_before.st_size != source_after.st_size
                or source_before.st_mtime_ns != source_after.st_mtime_ns
            ):
                raise RuntimeError(
                    "Checkpoint source changed during mirror copy: "
                    f"{local} before=(size={source_before.st_size}, "
                    f"mtime_ns={source_before.st_mtime_ns}) "
                    f"after=(size={source_after.st_size}, "
                    f"mtime_ns={source_after.st_mtime_ns})"
                )
            temporary_size = temporary.stat().st_size
            if temporary_size != source_before.st_size:
                raise RuntimeError(
                    f"Checkpoint mirror size mismatch before publish: source={local} "
                    f"bytes={source_before.st_size} temporary={temporary} "
                    f"bytes={temporary_size}"
                )
            os.replace(temporary, destination)
            published_size = destination.stat().st_size
            if published_size != source_before.st_size:
                raise RuntimeError(
                    f"Checkpoint mirror size mismatch after publish: source={local} "
                    f"bytes={source_before.st_size} destination={destination} "
                    f"bytes={published_size}"
                )
            return FilesystemCopyResult(
                local_path=local,
                destination_path=destination,
                size=int(published_size),
            )
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class MirrorResult:
    local_path: Path
    destination: str
    size: int
    attempts: int
    backend: str
    completed_at: str
    skipped: bool = False


@dataclass(frozen=True)
class MirrorFailure:
    local_path: Path
    destination: str
    attempts: int
    backend: str
    errors: tuple[str, ...]
    failed_at: str


class CheckpointMirrorError(RuntimeError):
    def __init__(self, failures: Iterable[MirrorFailure]) -> None:
        self.failures = tuple(failures)
        details = ", ".join(
            f"{failure.local_path.name} -> {failure.destination}"
            for failure in self.failures
        )
        super().__init__(
            f"{len(self.failures)} asynchronous checkpoint mirror task(s) failed: "
            f"{details}"
        )


class CheckpointMirrorTimeoutError(TimeoutError):
    def __init__(self, pending_paths: Iterable[Path], timeout_seconds: float) -> None:
        self.pending_paths = tuple(pending_paths)
        self.timeout_seconds = float(timeout_seconds)
        names = ", ".join(path.name for path in self.pending_paths)
        super().__init__(
            "Timed out draining asynchronous checkpoint mirror after "
            f"{self.timeout_seconds:.3f}s; pending={names or 'unknown'}"
        )


@dataclass(frozen=True)
class _MirrorTask:
    local_path: Path
    destination: str
    size: int
    enqueued_at: str


_QUEUE_SENTINEL = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AsyncCheckpointMirror:
    """Publish checkpoint weight files through one ordered background worker.

    The queue is FIFO and the worker count is fixed at one.  ``enqueue`` does
    only local validation/marker I/O before returning the destination string;
    all large-file or network I/O happens in the worker.  Construct and call it
    on global rank zero only.
    """

    def __init__(
        self,
        backend,
        *,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError(f"max_attempts must be > 0, got {max_attempts}")
        if retry_backoff_seconds < 0:
            raise ValueError(
                "retry_backoff_seconds must be >= 0, got "
                f"{retry_backoff_seconds}"
            )
        for attribute in ("backend_name", "destination", "publish"):
            if not hasattr(backend, attribute):
                raise TypeError(
                    f"Checkpoint mirror backend lacks required attribute {attribute!r}"
                )
        self.backend = backend
        self.max_attempts = int(max_attempts)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._queue: queue.Queue = queue.Queue()
        self._condition = threading.Condition()
        self._closed = False
        self._sentinel_enqueued = False
        self._unfinished = 0
        self._active: dict[Path, _MirrorTask] = {}
        self._results: list[MirrorResult] = []
        self._failures: dict[Path, MirrorFailure] = {}
        # Start lazily on the first completed checkpoint. Trainer construction
        # happens before DataLoader workers are spawned; forking those workers
        # from a process that already owns a live Python thread is unsafe.
        self._worker: threading.Thread | None = None

    def _start_worker_locked(self) -> None:
        if self._worker is not None:
            return
        # Explicit close/drain is the normal lifecycle. Daemon mode is a final
        # safeguard so a wedged FUSE copy cannot keep a process alive forever
        # after a bounded close() has reported a timeout.
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="fastwam-checkpoint-mirror",
            daemon=True,
        )
        self._worker.start()

    @classmethod
    def from_filesystem(
        cls,
        root_dir: os.PathLike[str] | str,
        output_dir: os.PathLike[str] | str,
        *,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> "AsyncCheckpointMirror":
        return cls(
            FilesystemCheckpointStore(root_dir=root_dir, output_dir=output_dir),
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    @classmethod
    def from_oss(
        cls,
        store: OssCheckpointStore,
        *,
        probe: bool = True,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> "AsyncCheckpointMirror":
        if not isinstance(store, OssCheckpointStore):
            raise TypeError(
                f"store must be OssCheckpointStore, got {type(store).__name__}"
            )
        if probe:
            store.probe()
        return cls(
            store,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    @staticmethod
    def pending_marker(local_path: os.PathLike[str] | str) -> Path:
        return Path(f"{local_path}.mirror_pending.json")

    @staticmethod
    def success_marker(local_path: os.PathLike[str] | str) -> Path:
        return Path(f"{local_path}.mirror.json")

    @staticmethod
    def failed_marker(local_path: os.PathLike[str] | str) -> Path:
        return Path(f"{local_path}.mirror_failed.json")

    def enqueue(self, local_path: os.PathLike[str] | str) -> str:
        local = Path(local_path)
        if not _WEIGHT_RE.match(local.name):
            raise ValueError(
                "Checkpoint mirror accepts only step_*.pt, step_*.adapter.pt, "
                f"or step_*.bottleneck.pt files, got {local.name!r}"
            )
        if not local.is_file():
            raise FileNotFoundError(local)
        size = int(local.stat().st_size)
        destination = str(self.backend.destination(local))
        task = _MirrorTask(
            local_path=local,
            destination=destination,
            size=size,
            enqueued_at=_utc_now(),
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot enqueue after checkpoint mirror close()")
            if local in self._active:
                raise RuntimeError(f"Checkpoint is already queued for mirroring: {local}")
            self.success_marker(local).unlink(missing_ok=True)
            self.failed_marker(local).unlink(missing_ok=True)
            _atomic_json(
                self.pending_marker(local),
                {
                    "format": _MIRROR_FORMAT,
                    "status": "pending",
                    "backend": str(self.backend.backend_name),
                    "local_path": str(local),
                    "destination": destination,
                    "size": size,
                    "max_attempts": self.max_attempts,
                    "enqueued_at": task.enqueued_at,
                },
            )
            self._active[local] = task
            self._unfinished += 1
            self._queue.put_nowait(task)
            self._start_worker_locked()
        return destination

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _QUEUE_SENTINEL:
                    return
                assert isinstance(item, _MirrorTask)
                self._process_task(item)
            finally:
                if isinstance(item, _MirrorTask):
                    with self._condition:
                        self._active.pop(item.local_path, None)
                        self._unfinished -= 1
                        self._condition.notify_all()
                self._queue.task_done()

    def _process_task(self, task: _MirrorTask) -> None:
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                published = self.backend.publish(task.local_path)
                published_size = int(getattr(published, "size"))
                if published_size != task.size:
                    raise RuntimeError(
                        "Checkpoint backend returned an unexpected published size: "
                        f"source={task.local_path} expected={task.size} "
                        f"published={published_size}"
                    )
                completed_at = _utc_now()
                result = MirrorResult(
                    local_path=task.local_path,
                    destination=task.destination,
                    size=published_size,
                    attempts=attempt,
                    backend=str(self.backend.backend_name),
                    completed_at=completed_at,
                    skipped=bool(getattr(published, "skipped", False)),
                )
                _atomic_json(
                    self.success_marker(task.local_path),
                    {
                        "format": _MIRROR_FORMAT,
                        "status": "success",
                        "backend": result.backend,
                        "local_path": str(result.local_path),
                        "destination": result.destination,
                        "size": result.size,
                        "attempts": result.attempts,
                        "enqueued_at": task.enqueued_at,
                        "completed_at": result.completed_at,
                        "skipped": result.skipped,
                    },
                )
                self.pending_marker(task.local_path).unlink(missing_ok=True)
                self.failed_marker(task.local_path).unlink(missing_ok=True)
                with self._condition:
                    self._results.append(result)
                    self._failures.pop(task.local_path, None)
                logger.info(
                    "[ckpt-mirror] bytes=%d attempts=%d destination=%s",
                    result.size,
                    result.attempts,
                    result.destination,
                )
                return
            except Exception as exc:
                errors.append(repr(exc))
                if attempt < self.max_attempts:
                    delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "[ckpt-mirror] attempt=%d/%d failed source=%s "
                        "destination=%s retry_in=%.3fs error=%r",
                        attempt,
                        self.max_attempts,
                        task.local_path,
                        task.destination,
                        delay,
                        exc,
                    )
                    if delay > 0:
                        time.sleep(delay)

        failure = MirrorFailure(
            local_path=task.local_path,
            destination=task.destination,
            attempts=self.max_attempts,
            backend=str(self.backend.backend_name),
            errors=tuple(errors),
            failed_at=_utc_now(),
        )
        try:
            _atomic_json(
                self.failed_marker(task.local_path),
                {
                    "format": _MIRROR_FORMAT,
                    "status": "failed",
                    "backend": failure.backend,
                    "local_path": str(failure.local_path),
                    "destination": failure.destination,
                    "size": task.size,
                    "attempts": failure.attempts,
                    "enqueued_at": task.enqueued_at,
                    "failed_at": failure.failed_at,
                    "errors": list(failure.errors),
                },
            )
            self.pending_marker(task.local_path).unlink(missing_ok=True)
        except Exception:
            logger.exception(
                "[ckpt-mirror] could not write failure marker for %s",
                task.local_path,
            )
        with self._condition:
            self._failures[task.local_path] = failure
        logger.error(
            "[ckpt-mirror] exhausted %d attempts source=%s destination=%s marker=%s",
            failure.attempts,
            failure.local_path,
            failure.destination,
            self.failed_marker(failure.local_path),
        )

    @staticmethod
    def _deadline(timeout_seconds: float | None) -> tuple[float | None, float | None]:
        if timeout_seconds is None:
            return None, None
        timeout = float(timeout_seconds)
        if timeout < 0:
            raise ValueError(f"timeout_seconds must be >= 0, got {timeout_seconds}")
        return time.monotonic() + timeout, timeout

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _completed(self, *, strict: bool) -> list[MirrorResult]:
        with self._condition:
            failures = tuple(self._failures.values())
            results = list(self._results)
        if strict and failures:
            raise CheckpointMirrorError(failures)
        return results

    def drain(
        self,
        *,
        strict: bool = True,
        timeout_seconds: float | None = None,
    ) -> list[MirrorResult]:
        deadline, timeout = self._deadline(timeout_seconds)
        with self._condition:
            while self._unfinished:
                remaining = self._remaining(deadline)
                if remaining is not None and remaining <= 0:
                    raise CheckpointMirrorTimeoutError(
                        self._active.keys(), timeout if timeout is not None else 0.0
                    )
                self._condition.wait(timeout=remaining)
        return self._completed(strict=strict)

    def _write_timeout_markers(self, timeout_seconds: float) -> tuple[Path, ...]:
        with self._condition:
            tasks = tuple(self._active.values())
        failed_at = _utc_now()
        for task in tasks:
            try:
                _atomic_json(
                    self.failed_marker(task.local_path),
                    {
                        "format": _MIRROR_FORMAT,
                        "status": "failed",
                        "reason": "close_timeout",
                        "backend": str(self.backend.backend_name),
                        "local_path": str(task.local_path),
                        "destination": task.destination,
                        "size": task.size,
                        "enqueued_at": task.enqueued_at,
                        "failed_at": failed_at,
                        "timeout_seconds": timeout_seconds,
                    },
                )
            except Exception:
                logger.exception(
                    "[ckpt-mirror] could not write timeout marker for %s",
                    task.local_path,
                )
        return tuple(task.local_path for task in tasks)

    def close(
        self,
        *,
        strict: bool = True,
        timeout_seconds: float | None = None,
    ) -> list[MirrorResult]:
        deadline, timeout = self._deadline(timeout_seconds)
        with self._condition:
            self._closed = True
            worker = self._worker
            if worker is not None and not self._sentinel_enqueued:
                self._queue.put_nowait(_QUEUE_SENTINEL)
                self._sentinel_enqueued = True
        if worker is None:
            return self._completed(strict=strict)
        worker.join(timeout=self._remaining(deadline))
        if worker.is_alive():
            paths = self._write_timeout_markers(timeout if timeout is not None else 0.0)
            raise CheckpointMirrorTimeoutError(
                paths, timeout if timeout is not None else 0.0
            )
        return self._completed(strict=strict)


def prune_uploaded_weights(
    weights_dir: os.PathLike[str] | str, keep_last: int
) -> list[Path]:
    """Delete verified local weights, retaining the newest requested files."""
    if keep_last < 0:
        raise ValueError(f"keep_last must be >= 0, got {keep_last}")
    root = Path(weights_dir)
    if not root.is_dir():
        return []
    uploaded = _sorted_steps(
        path
        for path in root.glob("step_*.pt")
        if OssCheckpointStore.manifest_path(path).is_file()
    )
    victims = uploaded[:-keep_last] if keep_last else uploaded
    for path in victims:
        path.unlink()
    return victims
