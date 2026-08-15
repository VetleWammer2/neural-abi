"""Portable normalized tensor-program representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from neuralabi.ir.serialization import to_data


@dataclass(frozen=True)
class TensorValue:
    shape: tuple[int | str, ...]
    dtype: str


@dataclass(frozen=True)
class TensorNode:
    node_id: str
    kind: str
    target: str
    canonical_op: str
    args: Any
    kwargs: Any
    outputs: tuple[TensorValue, ...]
    users: tuple[str, ...]


@dataclass(frozen=True)
class TensorProgram:
    nodes: tuple[TensorNode, ...]
    state_bindings: tuple[tuple[str, str], ...]
    user_inputs: tuple[str, ...]
    user_outputs: tuple[str, ...]
    rewrite_rules_used: tuple[str, ...]

    def by_id(self) -> dict[str, TensorNode]:
        return {node.node_id: node for node in self.nodes}

    def state_by_node(self) -> dict[str, str]:
        return dict(self.state_bindings)

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], to_data(self))


def node_ref(argument: Any) -> str | None:
    if (
        isinstance(argument, dict)
        and set(argument) == {"node"}
        and isinstance(argument["node"], str)
    ):
        return argument["node"]
    return None


def node_refs(argument: Any) -> tuple[str, ...]:
    result: list[str] = []
    if reference := node_ref(argument):
        result.append(reference)
    elif isinstance(argument, list):
        for item in argument:
            result.extend(node_refs(item))
    elif isinstance(argument, dict):
        for key in sorted(argument):
            result.extend(node_refs(argument[key]))
    return tuple(result)
