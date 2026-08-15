"""Hugging Face-style sharded SafeTensors reader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from neuralabi.checkpoints.single import SingleSafeTensorStore
from neuralabi.checkpoints.store import TensorMetadata, logical_fingerprint, require_key
from neuralabi.status import CheckpointError, PlanValidationError
from neuralabi.util.canonical_json import load_json
from neuralabi.util.paths import safe_index_filename


class ShardedSafeTensorStore:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path.resolve(strict=True)
        try:
            raw = load_json(self.index_path)
        except PlanValidationError as exc:
            raise CheckpointError(f"invalid checkpoint index {self.index_path}: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) - {"metadata", "weight_map"}:
            raise CheckpointError("checkpoint index must contain only metadata and weight_map")
        weight_map = raw.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError("checkpoint index weight_map must be a non-empty object")
        self._weight_map: dict[str, str] = {}
        for key, filename in weight_map.items():
            if (
                not isinstance(key, str)
                or not key
                or "\x00" in key
                or not isinstance(filename, str)
            ):
                raise CheckpointError("checkpoint index contains an invalid key or shard name")
            try:
                self._weight_map[key] = safe_index_filename(filename)
            except Exception as exc:
                raise CheckpointError(
                    f"checkpoint index has unsafe shard name {filename!r}"
                ) from exc
        self._keys = tuple(sorted(self._weight_map))
        self._stores: dict[str, SingleSafeTensorStore] = {}
        actual_owners: dict[str, str] = {}
        for filename in sorted(set(self._weight_map.values())):
            shard_path = self.index_path.parent / filename
            if not shard_path.is_file():
                raise CheckpointError(f"checkpoint shard is missing: {shard_path}")
            store = SingleSafeTensorStore(shard_path)
            self._stores[filename] = store
            for key in store:
                if key in actual_owners:
                    raise CheckpointError(
                        f"tensor {key!r} appears in both {actual_owners[key]!r} and {filename!r}"
                    )
                actual_owners[key] = filename
        expected = self._weight_map
        if set(actual_owners) != set(expected):
            missing = sorted(set(expected) - set(actual_owners))
            extra = sorted(set(actual_owners) - set(expected))
            raise CheckpointError(
                f"checkpoint index/shard keys disagree: missing={missing}, extra={extra}"
            )
        wrong = [key for key in self._keys if actual_owners[key] != expected[key]]
        if wrong:
            raise CheckpointError(f"checkpoint index points to the wrong shard for keys: {wrong}")
        self._fingerprint: str | None = None

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def metadata(self, key: str) -> TensorMetadata:
        require_key(self._keys, key)
        return self._stores[self._weight_map[key]].metadata(key)

    def read(self, key: str) -> torch.Tensor:
        require_key(self._keys, key)
        return self._stores[self._weight_map[key]].read(key)

    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = logical_fingerprint(self)
        return self._fingerprint
