"""Exact reverse-mode adjoints for layout-only transformations."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from neuralabi.status import PlanValidationError
from neuralabi.transforms.ast import (
    Alias,
    Concat,
    Deinterleave,
    Identity,
    Interleave,
    Permute,
    Reshape,
    Slice,
    Source,
    Squeeze,
    Stack,
    TransformExpr,
    Transpose,
    Unsqueeze,
)
from neuralabi.transforms.shapes import TensorSpec, normalize_axis


def _add(result: dict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
    result[key] = result[key] + value if key in result else value


def expression_adjoint(
    expression: TransformExpr,
    output_gradient: torch.Tensor,
    source_specs: Mapping[str, TensorSpec],
) -> dict[str, torch.Tensor]:
    """Map one target-physical gradient back to every physical source dependency."""

    result: dict[str, torch.Tensor] = {}

    def visit(node: TransformExpr, gradient: torch.Tensor) -> None:
        if isinstance(node, Source):
            _add(result, node.key, gradient)
            return
        if isinstance(node, Identity | Alias):
            visit(node.input, gradient)
            return
        input_spec = node.input.infer_spec(source_specs) if hasattr(node, "input") else None
        if isinstance(node, Reshape):
            assert input_spec is not None
            visit(node.input, gradient.reshape(input_spec.shape))
        elif isinstance(node, Permute):
            assert input_spec is not None
            axes = node._normalized(len(input_spec.shape))
            inverse = [0] * len(axes)
            for output_axis, input_axis in enumerate(axes):
                inverse[input_axis] = output_axis
            visit(node.input, gradient.permute(inverse))
        elif isinstance(node, Transpose):
            visit(node.input, gradient.transpose(*node._normalized(gradient.ndim)))
        elif isinstance(node, Slice):
            assert input_spec is not None
            axis, item, _ = node._parameters(input_spec.shape)
            expanded = torch.zeros(input_spec.shape, dtype=gradient.dtype, device=gradient.device)
            index: list[slice] = [slice(None)] * len(input_spec.shape)
            index[axis] = item
            expanded[tuple(index)] = gradient
            visit(node.input, expanded)
        elif isinstance(node, Concat):
            specs = [child.infer_spec(source_specs) for child in node.inputs]
            axis = normalize_axis(node.axis, gradient.ndim)
            parts = torch.split(gradient, [spec.shape[axis] for spec in specs], dim=axis)
            for child, part in zip(node.inputs, parts, strict=True):
                visit(child, part)
        elif isinstance(node, Stack):
            axis = normalize_axis(node.axis, gradient.ndim - 1, insertion=True)
            for child, part in zip(node.inputs, torch.unbind(gradient, dim=axis), strict=True):
                visit(child, part)
        elif isinstance(node, Squeeze):
            assert input_spec is not None
            visit(node.input, gradient.unsqueeze(normalize_axis(node.axis, len(input_spec.shape))))
        elif isinstance(node, Unsqueeze):
            assert input_spec is not None
            axis = normalize_axis(node.axis, len(input_spec.shape), insertion=True)
            visit(node.input, gradient.squeeze(axis))
        elif isinstance(node, Interleave):
            visit(
                node.input,
                Deinterleave(Source("_gradient"), node.axis, node.groups, node.segment_sizes).apply(
                    {"_gradient": gradient}
                ),
            )
        elif isinstance(node, Deinterleave):
            visit(
                node.input,
                Interleave(Source("_gradient"), node.axis, node.groups, node.segment_sizes).apply(
                    {"_gradient": gradient}
                ),
            )
        else:
            raise PlanValidationError(f"adjoint is unavailable for {node.op}")

    expected = expression.infer_spec(source_specs)
    actual = TensorSpec(
        tuple(output_gradient.shape), str(output_gradient.dtype).removeprefix("torch.")
    )
    if expected != actual:
        raise PlanValidationError(
            f"gradient spec {actual} does not match expression output {expected}"
        )
    visit(expression, output_gradient)
    return result
