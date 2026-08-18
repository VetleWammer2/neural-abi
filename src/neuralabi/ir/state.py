"""Complete physical persistent-state schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from neuralabi.ir.serialization import to_data
from neuralabi.status import CheckpointError
from neuralabi.util.hashing import hash_canonical

StateKind = Literal["parameter", "buffer"]


@dataclass(frozen=True)
class StateTensor:
    key: str
    kind: StateKind
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool
    persistent: bool
    alias_group: str | None
    numel: int


@dataclass(frozen=True)
class StateSchema:
    tensors: tuple[StateTensor, ...]
    schema_hash: str

    def by_key(self) -> dict[str, StateTensor]:
        return {tensor.key: tensor for tensor in self.tensors}

    def to_dict(self) -> dict[str, object]:
        return {"tensors": to_data(self.tensors), "schema_hash": self.schema_hash}


def parameter_identity_groups(schema: StateSchema) -> tuple[tuple[str, ...], ...]:
    """Return deterministic physical parameter identities.

    ``StateSchema`` records aliases by storage identity.  Optimizers attach state to
    parameter *objects*, so every trainable parameter binding in one alias group must
    be represented by one optimizer identity rather than by one state entry per key.
    Singleton parameters form singleton identities.  Frozen parameters and buffer
    bindings are deliberately excluded.
    """

    groups: dict[tuple[str, str], list[str]] = {}
    for tensor in schema.tensors:
        if tensor.kind != "parameter" or not tensor.requires_grad:
            continue
        identity = (
            ("alias", tensor.alias_group)
            if tensor.alias_group is not None
            else ("parameter", tensor.key)
        )
        groups.setdefault(identity, []).append(tensor.key)
    identities = [tuple(sorted(keys)) for keys in groups.values()]
    return tuple(sorted(identities))


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _storage_identity(tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    storage = tensor.untyped_storage()
    return (
        storage.data_ptr(),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def extract_state_schema(model: nn.Module) -> StateSchema:
    state = model.state_dict(keep_vars=True)
    parameter_keys = dict(model.named_parameters(remove_duplicate=False))
    buffer_keys = dict(model.named_buffers(remove_duplicate=False))
    identities: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], list[str]] = {}
    for key, tensor in state.items():
        identities.setdefault(_storage_identity(tensor), []).append(key)
    alias_for: dict[str, str] = {}
    alias_index = 0
    for keys in sorted(
        (sorted(group) for group in identities.values()), key=lambda value: value[0]
    ):
        if len(keys) > 1:
            group_id = f"alias[{alias_index}]"
            alias_index += 1
            alias_for.update({key: group_id for key in keys})
    entries: list[StateTensor] = []
    for key in sorted(state):
        tensor = state[key]
        if key in parameter_keys:
            kind: StateKind = "parameter"
        elif key in buffer_keys:
            kind = "buffer"
        else:
            raise CheckpointError(f"persistent state {key!r} is neither a parameter nor a buffer")
        entries.append(
            StateTensor(
                key=key,
                kind=kind,
                shape=tuple(int(size) for size in tensor.shape),
                dtype=dtype_name(tensor.dtype),
                requires_grad=bool(tensor.requires_grad),
                persistent=True,
                alias_group=alias_for.get(key),
                numel=tensor.numel(),
            )
        )
    payload = {"tensors": to_data(tuple(entries))}
    return StateSchema(tuple(entries), hash_canonical(payload))


def validate_state_metadata(
    schema: StateSchema, metadata: dict[str, tuple[tuple[int, ...], str]], *, label: str
) -> None:
    expected = schema.by_key()
    missing = sorted(set(expected) - set(metadata))
    unexpected = sorted(set(metadata) - set(expected))
    problems: list[str] = []
    if missing:
        problems.append(f"missing keys: {missing}")
    if unexpected:
        problems.append(f"unexpected keys: {unexpected}")
    for key in sorted(set(expected) & set(metadata)):
        shape, dtype = metadata[key]
        item = expected[key]
        if shape != item.shape:
            problems.append(f"{key}: shape {shape} != {item.shape}")
        if dtype != item.dtype:
            problems.append(f"{key}: dtype {dtype} != {item.dtype}")
    if problems:
        raise CheckpointError(
            f"{label} state does not match implementation: " + "; ".join(problems)
        )
