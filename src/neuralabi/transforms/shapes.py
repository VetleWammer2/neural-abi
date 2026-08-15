"""Shape and dtype rules for the NeuralABI transform grammar."""

from __future__ import annotations

import math
from dataclasses import dataclass

from neuralabi.status import PlanValidationError

MAX_RANK = 16
MAX_DIMENSION = 2**40
MAX_NUMEL = 2**63 - 1


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        validate_shape(self.shape)
        if not self.dtype or len(self.dtype) > 32:
            raise PlanValidationError(f"invalid dtype name {self.dtype!r}")

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def validate_shape(shape: tuple[int, ...]) -> None:
    if len(shape) > MAX_RANK:
        raise PlanValidationError(f"tensor rank exceeds {MAX_RANK}: {shape}")
    numel = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise PlanValidationError(f"tensor dimension is not an integer: {dimension!r}")
        if dimension < 0 or dimension > MAX_DIMENSION:
            raise PlanValidationError(f"invalid tensor dimension {dimension}")
        numel *= dimension
        if numel > MAX_NUMEL:
            raise PlanValidationError(f"tensor numel exceeds {MAX_NUMEL}: {shape}")


def normalize_axis(axis: int, rank: int, *, insertion: bool = False) -> int:
    lower = -(rank + 1) if insertion else -rank
    upper = rank if insertion else rank - 1
    if rank == 0 and not insertion:
        raise PlanValidationError("scalar tensor has no axes")
    if axis < lower or axis > upper:
        raise PlanValidationError(f"axis {axis} is out of range for rank {rank}")
    return axis + rank + (1 if insertion else 0) if axis < 0 else axis


def reshape_shape(input_shape: tuple[int, ...], requested: tuple[int, ...]) -> tuple[int, ...]:
    if len(requested) > MAX_RANK:
        raise PlanValidationError(f"reshape rank exceeds {MAX_RANK}")
    inferred_positions = [index for index, size in enumerate(requested) if size == -1]
    if len(inferred_positions) > 1:
        raise PlanValidationError("reshape may contain at most one inferred dimension")
    if any(size < -1 for size in requested):
        raise PlanValidationError(f"invalid reshape dimensions {requested}")
    input_numel = math.prod(input_shape)
    known_numel = math.prod(size for size in requested if size != -1)
    output = list(requested)
    if inferred_positions:
        if known_numel == 0 or input_numel % known_numel != 0:
            raise PlanValidationError(
                f"cannot infer reshape {requested} for input shape {input_shape}"
            )
        output[inferred_positions[0]] = input_numel // known_numel
    elif known_numel != input_numel:
        raise PlanValidationError(
            f"reshape changes numel from {input_numel} to {known_numel}: {input_shape} -> {requested}"
        )
    result = tuple(output)
    validate_shape(result)
    return result
