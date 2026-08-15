"""Dataflow helpers shared by semantic recognizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuralabi.ir.graph import TensorNode, TensorProgram, node_ref, node_refs
from neuralabi.status import UnsupportedGraphError


@dataclass(frozen=True)
class GraphIndex:
    program: TensorProgram

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", self.program.by_id())
        object.__setattr__(self, "state", self.program.state_by_node())

    nodes: dict[str, TensorNode] = None  # type: ignore[assignment]
    state: dict[str, str] = None  # type: ignore[assignment]

    def node(self, node_id: str) -> TensorNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise UnsupportedGraphError(f"graph references missing node {node_id!r}") from exc

    def input_node(self, node: TensorNode, index: int) -> TensorNode:
        if not isinstance(node.args, list) or index >= len(node.args):
            raise UnsupportedGraphError(f"node {node.node_id} has no input {index}")
        reference = node_ref(node.args[index])
        if reference is None:
            raise UnsupportedGraphError(f"node {node.node_id} input {index} is not a tensor node")
        return self.node(reference)

    def descendants(self, start: str) -> set[str]:
        result: set[str] = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for user in self.node(current).users:
                if user not in result:
                    result.add(user)
                    frontier.append(user)
        return result

    def ancestors(self, start: str) -> set[str]:
        result: set[str] = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for parent in node_refs(self.node(current).args):
                if parent not in result:
                    result.add(parent)
                    frontier.append(parent)
        return result

    def single(self, canonical_op: str) -> TensorNode:
        matches = [node for node in self.program.nodes if node.canonical_op == canonical_op]
        if len(matches) != 1:
            raise UnsupportedGraphError(
                f"expected exactly one {canonical_op} node, found {len(matches)}"
            )
        return matches[0]


def literal_arg(node: TensorNode, index: int) -> Any:
    if not isinstance(node.args, list) or index >= len(node.args):
        raise UnsupportedGraphError(f"node {node.node_id} is missing argument {index}")
    return node.args[index]
