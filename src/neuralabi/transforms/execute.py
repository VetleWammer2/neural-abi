"""Checked execution of declarative transform expressions."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from neuralabi.status import PlanValidationError
from neuralabi.transforms.ast import TransformExpr
from neuralabi.transforms.shapes import TensorSpec


def tensor_spec(tensor: torch.Tensor) -> TensorSpec:
    return TensorSpec(
        tuple(int(size) for size in tensor.shape), str(tensor.dtype).removeprefix("torch.")
    )


def execute_checked(
    expression: TransformExpr,
    sources: Mapping[str, torch.Tensor],
    *,
    expected: TensorSpec | None = None,
) -> torch.Tensor:
    actual_sources = {key: tensor_spec(tensor) for key, tensor in sources.items()}
    inferred = expression.infer_spec(actual_sources)
    if expected is not None and inferred != expected:
        raise PlanValidationError(
            f"expression output {inferred} does not match expected {expected}"
        )
    result = expression.apply(sources)
    actual = tensor_spec(result)
    if actual != inferred:
        raise PlanValidationError(
            f"runtime transform output {actual} disagrees with inferred {inferred}"
        )
    return result
