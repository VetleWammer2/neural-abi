"""Stable hashes for graphs, plans, schemas, and generated files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from neuralabi.util.canonical_json import canonical_dump_bytes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_canonical(value: Any) -> str:
    return sha256_bytes(canonical_dump_bytes(value))


def hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
