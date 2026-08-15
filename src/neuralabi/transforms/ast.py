"""Typed, declarative, non-executable checkpoint transformation language."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from neuralabi.status import PlanValidationError
from neuralabi.transforms.shapes import TensorSpec, normalize_axis, reshape_shape
from neuralabi.util.hashing import hash_canonical

MAX_EXPRESSION_DEPTH = 64
MAX_EXPRESSION_NODES = 100_000
MAX_INPUTS = 16_384


class TransformExpr(ABC):
    op: ClassVar[str]

    @abstractmethod
    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec: ...

    @abstractmethod
    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor: ...

    @abstractmethod
    def cost(self) -> int: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def source_keys(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class Source(TransformExpr):
    op: ClassVar[str] = "source"
    key: str

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        try:
            return sources[self.key]
        except KeyError as exc:
            raise PlanValidationError(
                f"expression references unknown source key {self.key!r}"
            ) from exc

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        try:
            return sources[self.key]
        except KeyError as exc:
            raise PlanValidationError(f"source tensor {self.key!r} was not supplied") from exc

    def cost(self) -> int:
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "key": self.key}

    def source_keys(self) -> tuple[str, ...]:
        return (self.key,)


@dataclass(frozen=True)
class UnaryExpr(TransformExpr):
    input: TransformExpr

    def source_keys(self) -> tuple[str, ...]:
        return self.input.source_keys()


@dataclass(frozen=True)
class Identity(UnaryExpr):
    op: ClassVar[str] = "identity"

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        return self.input.infer_spec(sources)

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.input.apply(sources)

    def cost(self) -> int:
        return self.input.cost()

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "input": self.input.to_dict()}


@dataclass(frozen=True)
class Alias(Identity):
    op: ClassVar[str] = "alias"


@dataclass(frozen=True)
class Reshape(UnaryExpr):
    op: ClassVar[str] = "reshape"
    shape: tuple[int, ...]

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        spec = self.input.infer_spec(sources)
        return TensorSpec(reshape_shape(spec.shape, self.shape), spec.dtype)

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor = self.input.apply(sources)
        expected = reshape_shape(tuple(tensor.shape), self.shape)
        return tensor.reshape(expected)

    def cost(self) -> int:
        return 1 + self.input.cost()

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "shape": list(self.shape), "input": self.input.to_dict()}


@dataclass(frozen=True)
class Permute(UnaryExpr):
    op: ClassVar[str] = "permute"
    axes: tuple[int, ...]

    def _normalized(self, rank: int) -> tuple[int, ...]:
        axes = tuple(normalize_axis(axis, rank) for axis in self.axes)
        if len(axes) != rank or sorted(axes) != list(range(rank)):
            raise PlanValidationError(f"axes {self.axes} are not a permutation of rank {rank}")
        return axes

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        spec = self.input.infer_spec(sources)
        axes = self._normalized(len(spec.shape))
        return TensorSpec(tuple(spec.shape[index] for index in axes), spec.dtype)

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor = self.input.apply(sources)
        return tensor.permute(self._normalized(tensor.ndim))

    def cost(self) -> int:
        return 2 + self.input.cost()

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "axes": list(self.axes), "input": self.input.to_dict()}


@dataclass(frozen=True)
class Transpose(UnaryExpr):
    op: ClassVar[str] = "transpose"
    axes: tuple[int, int]

    def _normalized(self, rank: int) -> tuple[int, int]:
        first, second = (normalize_axis(axis, rank) for axis in self.axes)
        if first == second:
            raise PlanValidationError("transpose axes must be distinct")
        return first, second

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        spec = self.input.infer_spec(sources)
        first, second = self._normalized(len(spec.shape))
        shape = list(spec.shape)
        shape[first], shape[second] = shape[second], shape[first]
        return TensorSpec(tuple(shape), spec.dtype)

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor = self.input.apply(sources)
        return tensor.transpose(*self._normalized(tensor.ndim))

    def cost(self) -> int:
        return 2 + self.input.cost()

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "axes": list(self.axes), "input": self.input.to_dict()}


@dataclass(frozen=True)
class Slice(UnaryExpr):
    op: ClassVar[str] = "slice"
    axis: int
    start: int
    stop: int
    step: int = 1

    def _parameters(self, shape: tuple[int, ...]) -> tuple[int, slice, int]:
        axis = normalize_axis(self.axis, len(shape))
        if self.step <= 0:
            raise PlanValidationError("slice step must be positive")
        normalized = slice(self.start, self.stop, self.step)
        start, stop, step = normalized.indices(shape[axis])
        length = max(0, (stop - start + step - 1) // step)
        return axis, slice(start, stop, step), length

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        spec = self.input.infer_spec(sources)
        axis, _, length = self._parameters(spec.shape)
        shape = list(spec.shape)
        shape[axis] = length
        return TensorSpec(tuple(shape), spec.dtype)

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor = self.input.apply(sources)
        axis, item, _ = self._parameters(tuple(tensor.shape))
        index: list[slice] = [slice(None)] * tensor.ndim
        index[axis] = item
        return tensor[tuple(index)]

    def cost(self) -> int:
        return 2 + self.input.cost()

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "axis": self.axis,
            "start": self.start,
            "stop": self.stop,
            "step": self.step,
            "input": self.input.to_dict(),
        }


@dataclass(frozen=True)
class Squeeze(UnaryExpr):
    op: ClassVar[str] = "squeeze"
    axis: int

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        spec = self.input.infer_spec(sources)
        axis = normalize_axis(self.axis, len(spec.shape))
        if spec.shape[axis] != 1:
            raise PlanValidationError(
                f"cannot squeeze non-singleton axis {self.axis}: {spec.shape}"
            )
        return TensorSpec(spec.shape[:axis] + spec.shape[axis + 1 :], spec.dtype)

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor = self.input.apply(sources)
        axis = normalize_axis(self.axis, tensor.ndim)
        if tensor.shape[axis] != 1:
            raise PlanValidationError(f"cannot squeeze non-singleton axis {self.axis}")
        return tensor.squeeze(axis)

    def cost(self) -> int:
        return 1 + self.input.cost()

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "axis": self.axis, "input": self.input.to_dict()}


@dataclass(frozen=True)
class Unsqueeze(UnaryExpr):
    op: ClassVar[str] = "unsqueeze"
    axis: int

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        spec = self.input.infer_spec(sources)
        axis = normalize_axis(self.axis, len(spec.shape), insertion=True)
        return TensorSpec((*spec.shape[:axis], 1, *spec.shape[axis:]), spec.dtype)

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor = self.input.apply(sources)
        return tensor.unsqueeze(normalize_axis(self.axis, tensor.ndim, insertion=True))

    def cost(self) -> int:
        return 1 + self.input.cost()

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "axis": self.axis, "input": self.input.to_dict()}


@dataclass(frozen=True)
class MultiExpr(TransformExpr):
    inputs: tuple[TransformExpr, ...]

    def _validated_inputs(self) -> tuple[TransformExpr, ...]:
        if not self.inputs or len(self.inputs) > MAX_INPUTS:
            raise PlanValidationError(f"operation requires between 1 and {MAX_INPUTS} inputs")
        return self.inputs

    def source_keys(self) -> tuple[str, ...]:
        return tuple(key for expression in self.inputs for key in expression.source_keys())


@dataclass(frozen=True)
class Concat(MultiExpr):
    op: ClassVar[str] = "concat"
    axis: int

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        specs = [expression.infer_spec(sources) for expression in self._validated_inputs()]
        rank = len(specs[0].shape)
        axis = normalize_axis(self.axis, rank)
        dtype = specs[0].dtype
        shape = list(specs[0].shape)
        shape[axis] = 0
        for spec in specs:
            if len(spec.shape) != rank or spec.dtype != dtype:
                raise PlanValidationError("concat inputs must have equal rank and dtype")
            for index, (actual, expected) in enumerate(
                zip(spec.shape, specs[0].shape, strict=True)
            ):
                if index != axis and actual != expected:
                    raise PlanValidationError(
                        "concat input shapes differ outside the concatenation axis"
                    )
            shape[axis] += spec.shape[axis]
        return TensorSpec(tuple(shape), dtype)

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensors = [expression.apply(sources) for expression in self._validated_inputs()]
        self.infer_spec(
            {
                key: TensorSpec(tuple(value.shape), str(value.dtype).removeprefix("torch."))
                for key, value in sources.items()
            }
        )
        return torch.cat(tensors, dim=normalize_axis(self.axis, tensors[0].ndim))

    def cost(self) -> int:
        return (
            3
            + sum(expression.cost() for expression in self.inputs)
            + len(set(self.source_keys()))
            - 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "axis": self.axis,
            "inputs": [item.to_dict() for item in self.inputs],
        }


@dataclass(frozen=True)
class Stack(MultiExpr):
    op: ClassVar[str] = "stack"
    axis: int

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        specs = [expression.infer_spec(sources) for expression in self._validated_inputs()]
        if any(spec != specs[0] for spec in specs[1:]):
            raise PlanValidationError("stack inputs must have identical shape and dtype")
        axis = normalize_axis(self.axis, len(specs[0].shape), insertion=True)
        shape = (*specs[0].shape[:axis], len(specs), *specs[0].shape[axis:])
        return TensorSpec(shape, specs[0].dtype)

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensors = [expression.apply(sources) for expression in self._validated_inputs()]
        return torch.stack(tensors, dim=normalize_axis(self.axis, tensors[0].ndim, insertion=True))

    def cost(self) -> int:
        return (
            3
            + sum(expression.cost() for expression in self.inputs)
            + len(set(self.source_keys()))
            - 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "axis": self.axis,
            "inputs": [item.to_dict() for item in self.inputs],
        }


@dataclass(frozen=True)
class InterleaveBase(UnaryExpr):
    axis: int
    groups: int
    segment_sizes: tuple[int, ...]

    def _validate(self, shape: tuple[int, ...]) -> int:
        axis = normalize_axis(self.axis, len(shape))
        if self.groups <= 0:
            raise PlanValidationError("interleave groups must be positive")
        if len(self.segment_sizes) < 2 or any(size < 0 for size in self.segment_sizes):
            raise PlanValidationError("interleave requires at least two non-negative segment sizes")
        if sum(self.segment_sizes) != shape[axis]:
            raise PlanValidationError(
                f"interleave segment sizes {self.segment_sizes} do not cover axis of size {shape[axis]}"
            )
        if any(size % self.groups for size in self.segment_sizes):
            raise PlanValidationError("every interleave segment must divide evenly across groups")
        return axis

    def infer_spec(self, sources: Mapping[str, TensorSpec]) -> TensorSpec:
        spec = self.input.infer_spec(sources)
        self._validate(spec.shape)
        return spec

    def cost(self) -> int:
        return 4 + self.input.cost()

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "axis": self.axis,
            "groups": self.groups,
            "segment_sizes": list(self.segment_sizes),
            "input": self.input.to_dict(),
        }


@dataclass(frozen=True)
class Interleave(InterleaveBase):
    op: ClassVar[str] = "interleave"

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor = self.input.apply(sources)
        axis = self._validate(tuple(tensor.shape))
        segments = torch.split(tensor, list(self.segment_sizes), dim=axis)
        chunks = [torch.chunk(segment, self.groups, dim=axis) for segment in segments]
        ordered = [
            chunks[segment][group] for group in range(self.groups) for segment in range(len(chunks))
        ]
        return torch.cat(ordered, dim=axis)


@dataclass(frozen=True)
class Deinterleave(InterleaveBase):
    op: ClassVar[str] = "deinterleave"

    def apply(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor = self.input.apply(sources)
        axis = self._validate(tuple(tensor.shape))
        per_group = [size // self.groups for size in self.segment_sizes]
        physical_sizes = per_group * self.groups
        chunks = torch.split(tensor, physical_sizes, dim=axis)
        ordered = [
            chunks[group * len(per_group) + segment]
            for segment in range(len(per_group))
            for group in range(self.groups)
        ]
        return torch.cat(ordered, dim=axis)


ExprType = (
    Source
    | Identity
    | Alias
    | Reshape
    | Permute
    | Transpose
    | Slice
    | Concat
    | Stack
    | Squeeze
    | Unsqueeze
    | Interleave
    | Deinterleave
)


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanValidationError(f"{label} must be an integer")
    return int(value)


def _as_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) > MAX_INPUTS:
        raise PlanValidationError(f"{label} must be a bounded JSON array")
    return tuple(_as_int(item, label) for item in value)


def expr_from_dict(
    value: Any, *, _depth: int = 0, _counter: list[int] | None = None
) -> TransformExpr:
    counter = [0] if _counter is None else _counter
    counter[0] += 1
    if counter[0] > MAX_EXPRESSION_NODES or _depth > MAX_EXPRESSION_DEPTH:
        raise PlanValidationError("transform expression exceeds safety limits")
    if not isinstance(value, dict) or not isinstance(value.get("op"), str):
        raise PlanValidationError("transform expression must be an object with a string op")
    op = value["op"]
    allowed_fields: dict[str, set[str]] = {
        "source": {"op", "key"},
        "identity": {"op", "input"},
        "alias": {"op", "input"},
        "reshape": {"op", "input", "shape"},
        "permute": {"op", "input", "axes"},
        "transpose": {"op", "input", "axes"},
        "slice": {"op", "input", "axis", "start", "stop", "step"},
        "concat": {"op", "inputs", "axis"},
        "stack": {"op", "inputs", "axis"},
        "squeeze": {"op", "input", "axis"},
        "unsqueeze": {"op", "input", "axis"},
        "interleave": {"op", "input", "axis", "groups", "segment_sizes"},
        "deinterleave": {"op", "input", "axis", "groups", "segment_sizes"},
    }
    if op not in allowed_fields:
        raise PlanValidationError(f"unknown transform operation {op!r}")
    unknown = set(value) - allowed_fields[op]
    if unknown:
        raise PlanValidationError(f"unknown fields for {op}: {sorted(unknown)}")
    if op == "source":
        key = value.get("key")
        if not isinstance(key, str) or not key or len(key) > 4096 or "\x00" in key:
            raise PlanValidationError("source key must be a non-empty bounded string")
        return Source(key)
    if op in {"concat", "stack"}:
        raw_inputs = value.get("inputs")
        if not isinstance(raw_inputs, list) or not raw_inputs or len(raw_inputs) > MAX_INPUTS:
            raise PlanValidationError(f"{op} inputs must be a non-empty bounded array")
        inputs = tuple(
            expr_from_dict(item, _depth=_depth + 1, _counter=counter) for item in raw_inputs
        )
        axis = _as_int(value.get("axis"), "axis")
        return Concat(inputs, axis) if op == "concat" else Stack(inputs, axis)
    raw_input = value.get("input")
    child = expr_from_dict(raw_input, _depth=_depth + 1, _counter=counter)
    if op == "identity":
        return Identity(child)
    if op == "alias":
        return Alias(child)
    if op == "reshape":
        return Reshape(child, _as_int_tuple(value.get("shape"), "shape"))
    if op == "permute":
        return Permute(child, _as_int_tuple(value.get("axes"), "axes"))
    if op == "transpose":
        axes = _as_int_tuple(value.get("axes"), "axes")
        if len(axes) != 2:
            raise PlanValidationError("transpose requires exactly two axes")
        return Transpose(child, (axes[0], axes[1]))
    if op == "slice":
        return Slice(
            child,
            _as_int(value.get("axis"), "axis"),
            _as_int(value.get("start"), "start"),
            _as_int(value.get("stop"), "stop"),
            _as_int(value.get("step", 1), "step"),
        )
    if op == "squeeze":
        return Squeeze(child, _as_int(value.get("axis"), "axis"))
    if op == "unsqueeze":
        return Unsqueeze(child, _as_int(value.get("axis"), "axis"))
    if op in {"interleave", "deinterleave"}:
        arguments = (
            child,
            _as_int(value.get("axis"), "axis"),
            _as_int(value.get("groups"), "groups"),
            _as_int_tuple(value.get("segment_sizes"), "segment_sizes"),
        )
        return Interleave(*arguments) if op == "interleave" else Deinterleave(*arguments)
    raise AssertionError(f"unreachable op {op}")


def expression_hash(expression: TransformExpr) -> str:
    return hash_canonical(expression.to_dict())


def collect_source_keys(expressions: Sequence[TransformExpr]) -> tuple[str, ...]:
    return tuple(sorted({key for expression in expressions for key in expression.source_keys()}))
