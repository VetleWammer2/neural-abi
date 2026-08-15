"""Deterministic, bounded-shard, atomic SafeTensors output."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import save_file

from neuralabi.status import CheckpointError
from neuralabi.util.canonical_json import pretty_dumps
from neuralabi.util.paths import atomic_replace

DEFAULT_MAX_SHARD_SIZE = 2 * 1024**3


def parse_size(value: str) -> int:
    text = value.strip().upper().replace("IB", "B")
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suffix in ("GB", "MB", "KB", "B"):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            try:
                result = int(float(number) * units[suffix])
            except ValueError as exc:
                raise CheckpointError(f"invalid shard size {value!r}") from exc
            if result <= 0:
                raise CheckpointError("shard size must be positive")
            return result
    raise CheckpointError(f"shard size requires B, KB, MB, or GB suffix: {value!r}")


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _write_shard(path: Path, tensors: dict[str, torch.Tensor]) -> int:
    ordered = {key: tensors[key].detach().cpu().contiguous().clone() for key in sorted(tensors)}
    # SafeTensors stores metadata in a hash map whose multi-key iteration order is not stable.
    # A single fixed entry keeps byte output deterministic across repeated conversions.
    save_file(ordered, path, metadata={"neuralabi": "0.1"})
    return sum(_tensor_bytes(tensor) for tensor in ordered.values())


def write_checkpoint(
    tensors: Iterable[tuple[str, torch.Tensor]],
    output: Path,
    *,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> tuple[Path, tuple[Path, ...], int]:
    """Write a stream of sorted unique tensors and return index/main path, files, total bytes."""

    output = output.resolve(strict=False)
    if output.exists() and (output.is_file() or any(output.iterdir())):
        raise CheckpointError(f"refusing to replace non-empty output path {output}")
    output_is_file = output.suffix == ".safetensors"
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.stem}.neuralabi-", dir=parent))
    shards: list[tuple[Path, tuple[str, ...], int]] = []
    staged: dict[str, torch.Tensor] = {}
    staged_bytes = 0
    previous_key: str | None = None
    total_bytes = 0
    try:
        for key, tensor in tensors:
            if not key or "\x00" in key:
                raise CheckpointError("output tensor key is invalid")
            if previous_key is not None and key <= previous_key:
                raise CheckpointError("output tensors must be unique and sorted")
            previous_key = key
            nbytes = _tensor_bytes(tensor)
            if staged and staged_bytes + nbytes > max_shard_size:
                provisional = temporary_root / f"shard-{len(shards) + 1:05d}.safetensors"
                shard_bytes = _write_shard(provisional, staged)
                shards.append((provisional, tuple(sorted(staged)), shard_bytes))
                staged = {}
                staged_bytes = 0
            staged[key] = tensor
            staged_bytes += nbytes
            total_bytes += nbytes
        if not staged and not shards:
            raise CheckpointError("cannot write an empty checkpoint")
        if staged:
            provisional = temporary_root / f"shard-{len(shards) + 1:05d}.safetensors"
            shard_bytes = _write_shard(provisional, staged)
            shards.append((provisional, tuple(sorted(staged)), shard_bytes))
        if output_is_file:
            if len(shards) != 1:
                raise CheckpointError(
                    "single-file output exceeds max shard size; select a directory"
                )
            atomic_replace(shards[0][0], output)
            shutil.rmtree(temporary_root, ignore_errors=True)
            return output, (output,), total_bytes
        final_temp = temporary_root / "bundle"
        final_temp.mkdir()
        files: list[Path] = []
        if len(shards) == 1:
            destination = final_temp / "model.safetensors"
            os.replace(shards[0][0], destination)
            files.append(destination)
            main = destination
        else:
            count = len(shards)
            weight_map: dict[str, str] = {}
            for index, (provisional, keys, _) in enumerate(shards, 1):
                filename = f"model-{index:05d}-of-{count:05d}.safetensors"
                destination = final_temp / filename
                os.replace(provisional, destination)
                files.append(destination)
                for key in keys:
                    weight_map[key] = filename
            main = final_temp / "model.safetensors.index.json"
            main.write_text(
                pretty_dumps(
                    {
                        "metadata": {"total_size": total_bytes},
                        "weight_map": dict(sorted(weight_map.items())),
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            files.append(main)
        if output.exists():
            output.rmdir()
        os.replace(final_temp, output)
        resolved_files = tuple(output / path.name for path in files)
        resolved_main = output / main.name
        shutil.rmtree(temporary_root, ignore_errors=True)
        return resolved_main, resolved_files, total_bytes
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
