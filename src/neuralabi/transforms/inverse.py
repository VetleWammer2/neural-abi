"""Exact inverses for bijective unary layout transformations."""

from __future__ import annotations

from collections.abc import Mapping

from neuralabi.status import PlanValidationError
from neuralabi.transforms.ast import (
    Alias,
    Deinterleave,
    Identity,
    Interleave,
    Permute,
    Reshape,
    Source,
    Squeeze,
    TransformExpr,
    Transpose,
    Unsqueeze,
)
from neuralabi.transforms.shapes import TensorSpec, normalize_axis


def inverse_expression(
    expression: TransformExpr,
    source_specs: Mapping[str, TensorSpec],
    *,
    output_key: str,
) -> TransformExpr:
    """Build an inverse applied to ``Source(output_key)`` for a unary bijection."""

    operations: list[tuple[TransformExpr, TensorSpec]] = []

    def descend(node: TransformExpr) -> None:
        if isinstance(node, Source):
            return
        child = getattr(node, "input", None)
        if not isinstance(child, TransformExpr):
            raise PlanValidationError(f"{node.op} is not independently invertible")
        operations.append((node, child.infer_spec(source_specs)))
        descend(child)

    descend(expression)
    current: TransformExpr = Source(output_key)
    for operation, input_spec in operations:
        if isinstance(operation, Identity | Alias):
            continue
        if isinstance(operation, Reshape):
            current = Reshape(current, input_spec.shape)
        elif isinstance(operation, Permute):
            axes = operation._normalized(len(input_spec.shape))
            inverse = [0] * len(axes)
            for output_axis, input_axis in enumerate(axes):
                inverse[input_axis] = output_axis
            current = Permute(current, tuple(inverse))
        elif isinstance(operation, Transpose):
            current = Transpose(current, operation.axes)
        elif isinstance(operation, Squeeze):
            axis = normalize_axis(operation.axis, len(input_spec.shape))
            current = Unsqueeze(current, axis)
        elif isinstance(operation, Unsqueeze):
            axis = normalize_axis(operation.axis, len(input_spec.shape), insertion=True)
            current = Squeeze(current, axis)
        elif isinstance(operation, Interleave):
            current = Deinterleave(
                current, operation.axis, operation.groups, operation.segment_sizes
            )
        elif isinstance(operation, Deinterleave):
            current = Interleave(current, operation.axis, operation.groups, operation.segment_sizes)
        else:
            raise PlanValidationError(f"{operation.op} is not independently invertible")
    return current
