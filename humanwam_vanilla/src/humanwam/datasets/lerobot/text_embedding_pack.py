# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
"""Read-only sharded SQLite packs for FastWAM text embeddings.

The legacy cache stores one torch payload per prompt.  On object-storage mounts
that turns every sample into metadata lookup + open + read operations.  A pack
keeps the exact serialized payload bytes but indexes them in a small number of
SQLite files.  Connections are persistent and process/thread local so DataLoader
fork/spawn workers never share SQLite state.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch


TEXT_EMBEDDING_PACK_SCHEMA = "humanwam.text-embedding-pack.v1"
TEXT_EMBEDDING_PACK_USER_VERSION = 1


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while True:
            payload = stream.read(chunk_size)
            if not payload:
                return digest.hexdigest()
            digest.update(payload)


class TextEmbeddingPack:
    """Validated, read-only lookup over prefix-sharded SQLite files."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        context_len: int,
        verify_payload_sha256: bool = False,
        verify_shard_sha256: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.context_len = int(context_len)
        self.verify_payload_sha256 = bool(verify_payload_sha256)
        self.verify_shard_sha256 = bool(verify_shard_sha256)
        if self.context_len <= 0:
            raise ValueError(f"context_len must be positive, got {context_len}")
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"text embedding pack manifest is missing: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid text embedding pack manifest: {manifest_path}"
            ) from exc
        if manifest.get("schema") != TEXT_EMBEDDING_PACK_SCHEMA:
            raise ValueError(
                f"unsupported text embedding pack schema "
                f"{manifest.get('schema')!r} in {manifest_path}"
            )
        if int(manifest.get("context_len", -1)) != self.context_len:
            raise ValueError(
                f"text embedding pack context_len mismatch: "
                f"pack={manifest.get('context_len')} configured={self.context_len}"
            )
        if manifest.get("key") != "legacy_cache_filename":
            raise ValueError(
                "text embedding pack must be keyed by legacy_cache_filename"
            )
        if int(manifest.get("prefix_chars", -1)) != 1:
            raise ValueError("text embedding pack prefix_chars must be 1")
        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, dict) or not raw_shards:
            raise ValueError("text embedding pack manifest contains no shards")

        self._shards: dict[str, Path] = {}
        self._shard_records: dict[str, dict[str, Any]] = {}
        for prefix, record in raw_shards.items():
            if (
                not isinstance(prefix, str)
                or len(prefix) != 1
                or prefix not in "0123456789abcdef"
                or not isinstance(record, dict)
            ):
                raise ValueError(f"invalid text embedding shard record: {prefix!r}")
            filename = record.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(
                    f"shard {prefix} file must be a basename, got {filename!r}"
                )
            shard_path = self.root / filename
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"text embedding shard {prefix} is missing: {shard_path}"
                )
            expected_bytes = int(record.get("bytes", -1))
            if expected_bytes <= 0 or shard_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"text embedding shard {prefix} size mismatch: "
                    f"manifest={expected_bytes}, file={shard_path.stat().st_size}"
                )
            expected_rows = int(record.get("rows", -1))
            if expected_rows <= 0:
                raise ValueError(
                    f"text embedding shard {prefix} has invalid row count "
                    f"{expected_rows}"
                )
            expected_sha = record.get("sha256")
            if (
                not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha)
            ):
                raise ValueError(
                    f"text embedding shard {prefix} has invalid sha256"
                )
            if self.verify_shard_sha256:
                observed_sha = _sha256_file(shard_path)
                if observed_sha != expected_sha:
                    raise ValueError(
                        f"text embedding shard {prefix} checksum mismatch: "
                        f"expected={expected_sha}, observed={observed_sha}"
                    )
            self._shards[prefix] = shard_path
            self._shard_records[prefix] = record

        self.manifest = manifest
        self._reset_connections()
        # Open each shard once during construction and validate the on-disk schema.
        # This is metadata-only and catches an accidental/non-pack SQLite file before
        # training workers are launched.
        for prefix in self._shards:
            connection = self._connection(prefix)
            user_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if user_version != TEXT_EMBEDDING_PACK_USER_VERSION:
                raise ValueError(
                    f"text embedding shard {prefix} user_version mismatch: "
                    f"{user_version} != {TEXT_EMBEDDING_PACK_USER_VERSION}"
                )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(embeddings)"
                ).fetchall()
            }
            expected_columns = {
                "key",
                "payload",
                "payload_bytes",
                "payload_sha256",
            }
            if columns != expected_columns:
                raise ValueError(
                    f"text embedding shard {prefix} schema mismatch: "
                    f"{sorted(columns)} != {sorted(expected_columns)}"
                )

    def _reset_connections(self) -> None:
        self._connections: dict[tuple[int, int, str], sqlite3.Connection] = {}
        self._connections_pid = os.getpid()
        self._connections_lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_connections"] = {}
        state["_connections_pid"] = None
        state["_connections_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._reset_connections()

    def close(self) -> None:
        connections = list(getattr(self, "_connections", {}).values())
        self._connections = {}
        for connection in connections:
            connection.close()

    def _connection(self, prefix: str) -> sqlite3.Connection:
        if getattr(self, "_connections_pid", None) != os.getpid():
            self.close()
            self._reset_connections()
        key = (os.getpid(), threading.get_ident(), prefix)
        connection = self._connections.get(key)
        if connection is not None:
            return connection
        with self._connections_lock:
            connection = self._connections.get(key)
            if connection is not None:
                return connection
            shard_path = self._shards[prefix]
            uri = f"file:{quote(str(shard_path), safe='/')}?mode=ro"
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,
            )
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA temp_store=MEMORY")
            self._connections[key] = connection
            return connection

    def load(self, legacy_cache_filename: str) -> dict[str, Any]:
        """Load one exact legacy torch payload from the indexed pack."""

        filename = str(legacy_cache_filename)
        if Path(filename).name != filename:
            raise ValueError(
                f"legacy text cache key must be a basename, got {filename!r}"
            )
        prefix = filename[:1].lower()
        if prefix not in self._shards:
            raise FileNotFoundError(
                f"text embedding pack {self.root} has no shard for key {filename}"
            )
        row = self._connection(prefix).execute(
            "SELECT payload, payload_bytes, payload_sha256 "
            "FROM embeddings WHERE key = ?",
            (filename,),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(
                f"text embedding {filename} is absent from pack {self.root}"
            )
        payload, expected_bytes, expected_sha = row
        payload = bytes(payload)
        if len(payload) != int(expected_bytes):
            raise ValueError(
                f"packed text embedding {filename} length mismatch: "
                f"expected={expected_bytes}, observed={len(payload)}"
            )
        if self.verify_payload_sha256:
            observed_sha = hashlib.sha256(payload).hexdigest()
            if observed_sha != expected_sha:
                raise ValueError(
                    f"packed text embedding {filename} checksum mismatch: "
                    f"expected={expected_sha}, observed={observed_sha}"
                )
        loaded = torch.load(io.BytesIO(payload), map_location="cpu")
        if not isinstance(loaded, dict):
            raise ValueError(
                f"packed text embedding {filename} must contain a mapping"
            )
        return loaded
