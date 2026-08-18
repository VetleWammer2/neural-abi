"""Filesystem validation helpers for untrusted paths."""

from __future__ import annotations

import os
from pathlib import Path

from neuralabi.status import NeuralABIError


def ensure_distinct_paths(source: Path, output: Path) -> None:
    source_resolved = source.resolve(strict=True)
    output_resolved = output.resolve(strict=False)
    if source_resolved == output_resolved:
        raise NeuralABIError("output path must not overwrite the input checkpoint")
    if source_resolved.is_dir() and output_resolved.is_relative_to(source_resolved):
        raise NeuralABIError("output path must not be inside the input checkpoint directory")


def ensure_disjoint_paths(first: Path, second: Path) -> None:
    """Reject equal paths and either direction of path nesting."""

    first_resolved = first.resolve(strict=False)
    second_resolved = second.resolve(strict=False)
    if (
        first_resolved == second_resolved
        or first_resolved.is_relative_to(second_resolved)
        or second_resolved.is_relative_to(first_resolved)
    ):
        raise NeuralABIError(
            f"paths must be distinct and non-nested: {first_resolved} and {second_resolved}"
        )


def atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def safe_index_filename(name: str) -> str:
    candidate = Path(name)
    if candidate.is_absolute() or len(candidate.parts) != 1 or name in {"", ".", ".."}:
        raise NeuralABIError(f"unsafe shard filename {name!r}")
    if any(separator in name for separator in ("/", "\\", "\x00")):
        raise NeuralABIError(f"unsafe shard filename {name!r}")
    return name
