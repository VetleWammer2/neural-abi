"""Single-file SafeTensors reader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open

from neuralabi.checkpoints.store import (
    TensorMetadata,
    logical_fingerprint,
    metadata_from_tensor,
    require_key,
)
from neuralabi.status import CheckpointError


class SingleSafeTensorStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=True)
        if self.path.suffix != ".safetensors":
            raise CheckpointError(f"not a SafeTensors file: {self.path}")
        try:
            with safe_open(  # type: ignore[no-untyped-call]
                self.path, framework="pt", device="cpu"
            ) as handle:
                self._keys = tuple(sorted(handle.keys()))
                self._metadata = {
                    key: metadata_from_tensor(handle.get_tensor(key)) for key in self._keys
                }
        except Exception as exc:
            raise CheckpointError(f"cannot inspect SafeTensors file {self.path}: {exc}") from exc
        if len(set(self._keys)) != len(self._keys):
            raise CheckpointError(f"duplicate tensor keys in {self.path}")
        self._fingerprint: str | None = None

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def metadata(self, key: str) -> TensorMetadata:
        require_key(self._keys, key)
        return self._metadata[key]

    def read(self, key: str) -> torch.Tensor:
        require_key(self._keys, key)
        try:
            with safe_open(  # type: ignore[no-untyped-call]
                self.path, framework="pt", device="cpu"
            ) as handle:
                return cast(torch.Tensor, handle.get_tensor(key)).clone()
        except Exception as exc:
            raise CheckpointError(f"cannot read tensor {key!r} from {self.path}: {exc}") from exc

    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = logical_fingerprint(self)
        return self._fingerprint
