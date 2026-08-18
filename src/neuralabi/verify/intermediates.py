"""Runtime capture of graph nodes aligned to semantic anchors."""

from __future__ import annotations

from typing import Any

import torch
from torch import fx

from neuralabi.ir.semantic import CanonicalModel
from neuralabi.status import UnsupportedGraphError
from neuralabi.verify.forward import TensorComparison, compare_tensor


class _RecordingInterpreter(fx.Interpreter):
    def __init__(self, module: fx.GraphModule) -> None:
        super().__init__(module)
        self.values: dict[str, Any] = {}

    def run_node(self, node: fx.Node) -> Any:
        value = super().run_node(node)
        self.values[node.name] = value
        return value


def record_graph_values(
    module: fx.GraphModule, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    if kwargs:
        # Exported GraphModule flattens keyword arguments into its positional signature. The
        # adapter contract remains general, but anchor capture currently requires positional probes.
        raise UnsupportedGraphError("semantic anchor capture does not support keyword-only probes")
    interpreter = _RecordingInterpreter(module)
    interpreter.run(*args)
    return interpreter.values


def compare_anchors(
    source_values: dict[str, Any],
    target_values: dict[str, Any],
    source_model: CanonicalModel,
    target_model: CanonicalModel,
) -> tuple[TensorComparison, ...]:
    source_anchors = {anchor.anchor_id: anchor for anchor in source_model.anchors}
    target_anchors = {anchor.anchor_id: anchor for anchor in target_model.anchors}
    if set(source_anchors) != set(target_anchors):
        raise UnsupportedGraphError("source and target semantic anchors are not aligned")
    results: list[TensorComparison] = []
    positions = {node.node_id: position for position, node in enumerate(source_model.graph.nodes)}
    ordered_anchor_ids = sorted(
        source_anchors,
        key=lambda anchor_id: (positions.get(source_anchors[anchor_id].node_id, 10**12), anchor_id),
    )
    for anchor_id in ordered_anchor_ids:
        source_value = source_values.get(source_anchors[anchor_id].node_id)
        target_value = target_values.get(target_anchors[anchor_id].node_id)
        if not isinstance(source_value, torch.Tensor) or not isinstance(target_value, torch.Tensor):
            raise UnsupportedGraphError(f"semantic anchor {anchor_id} is not a tensor")
        results.append(compare_tensor(source_value, target_value, path=anchor_id))
    return tuple(results)
