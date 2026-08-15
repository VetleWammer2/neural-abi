"""Read-only, SafeTensors-only tensor-store abstraction."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from neuralabi.status import CheckpointError


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    numel: int
    nbytes: int


class TensorStore(Protocol):
    def __iter__(self) -> Iterator[str]: ...

    def keys(self) -> Sequence[str]: ...

    def metadata(self, key: str) -> TensorMetadata: ...

    def read(self, key: str) -> torch.Tensor: ...

    def fingerprint(self) -> str: ...


def update_tensor_fingerprint(
    digest: hashlib._Hash, key: str, metadata: TensorMetadata, tensor: torch.Tensor
) -> None:
    key_bytes = key.encode("utf-8")
    digest.update(len(key_bytes).to_bytes(8, "big"))
    digest.update(key_bytes)
    shape_text = ",".join(str(size) for size in metadata.shape).encode("ascii")
    dtype_text = metadata.dtype.encode("ascii")
    digest.update(len(shape_text).to_bytes(8, "big"))
    digest.update(shape_text)
    digest.update(len(dtype_text).to_bytes(8, "big"))
    digest.update(dtype_text)
    contiguous = tensor.detach().cpu().contiguous()
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())


def logical_fingerprint(store: TensorStore) -> str:
    digest = hashlib.sha256(b"neuralabi-logical-checkpoint-v1\0")
    for key in sorted(store.keys()):
        metadata = store.metadata(key)
        update_tensor_fingerprint(digest, key, metadata, store.read(key))
    return digest.hexdigest()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def metadata_from_tensor(tensor: torch.Tensor) -> TensorMetadata:
    return TensorMetadata(
        tuple(int(size) for size in tensor.shape),
        dtype_name(tensor.dtype),
        tensor.numel(),
        tensor.numel() * tensor.element_size(),
    )


def require_key(keys: Sequence[str], key: str) -> None:
    if key not in keys:
        raise CheckpointError(f"checkpoint does not contain tensor {key!r}")


def open_tensor_store(path: Path) -> TensorStore:
    from neuralabi.checkpoints.sharded import ShardedSafeTensorStore
    from neuralabi.checkpoints.single import SingleSafeTensorStore

    path = path.resolve(strict=True)
    if path.is_file() and path.name.endswith(".safetensors.index.json"):
        return ShardedSafeTensorStore(path)
    if path.is_file() and path.suffix == ".safetensors":
        return SingleSafeTensorStore(path)
    if path.is_dir():
        indexes = sorted(path.glob("*.safetensors.index.json"))
        singles = sorted(path.glob("*.safetensors"))
        if len(indexes) == 1:
            return ShardedSafeTensorStore(indexes[0])
        if len(indexes) > 1:
            raise CheckpointError(f"multiple SafeTensors index files in {path}")
        if len(singles) == 1:
            return SingleSafeTensorStore(singles[0])
        raise CheckpointError(f"expected one SafeTensors file or index in {path}")
    raise CheckpointError(f"unsupported checkpoint path {path}; v0.1 accepts SafeTensors only")
