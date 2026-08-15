"""Canonical semantic slots and physical layout views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from neuralabi.export.signature import ArchitectureSignature
from neuralabi.ir.graph import TensorProgram
from neuralabi.ir.serialization import to_data
from neuralabi.ir.state import StateSchema
from neuralabi.status import Evidence, PlanValidationError
from neuralabi.transforms import (
    Concat,
    Deinterleave,
    Interleave,
    Reshape,
    Slice,
    Source,
    TensorSpec,
    TransformExpr,
    Transpose,
)

Orientation = Literal["out_in", "in_out", "vector"]


@dataclass(frozen=True)
class AxisLineage:
    operation: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    detail: str


@dataclass(frozen=True)
class SemanticSlot:
    semantic_id: str
    canonical_shape: tuple[int, ...]
    flat_shape: tuple[int, ...]
    dtype: str
    supporting_nodes: tuple[str, ...]
    evidence: Evidence
    assumptions: tuple[str, ...] = ()

    @property
    def spec(self) -> TensorSpec:
        return TensorSpec(self.canonical_shape, self.dtype)


@dataclass(frozen=True)
class PhysicalLayout:
    physical_key: str
    components: tuple[str, ...]
    component_flat_shapes: tuple[tuple[int, ...], ...]
    orientation: Orientation
    physical_shape: tuple[int, ...]
    dtype: str
    interleave_groups: int | None
    supporting_nodes: tuple[str, ...]
    axis_lineage: tuple[AxisLineage, ...] = ()

    def _logical_shape(self) -> tuple[int, ...]:
        if not self.components or len(self.components) != len(self.component_flat_shapes):
            raise PlanValidationError(f"invalid component layout for {self.physical_key}")
        if self.orientation == "vector":
            if any(len(shape) != 1 for shape in self.component_flat_shapes):
                raise PlanValidationError("vector layout components must be rank one")
            return (sum(shape[0] for shape in self.component_flat_shapes),)
        if any(len(shape) != 2 for shape in self.component_flat_shapes):
            raise PlanValidationError("matrix layout components must be rank two")
        input_sizes = {shape[1] for shape in self.component_flat_shapes}
        if len(input_sizes) != 1:
            raise PlanValidationError("fused matrix components must share their input dimension")
        out_size = sum(shape[0] for shape in self.component_flat_shapes)
        in_size = next(iter(input_sizes))
        return (out_size, in_size) if self.orientation == "out_in" else (in_size, out_size)

    def physical_spec(self) -> TensorSpec:
        logical = self._logical_shape()
        if _numel(logical) != _numel(self.physical_shape):
            raise PlanValidationError(
                f"layout {self.physical_key} logical shape {logical} does not fit physical shape {self.physical_shape}"
            )
        return TensorSpec(self.physical_shape, self.dtype)

    def decode_views(self, slots: Mapping[str, SemanticSlot]) -> tuple[PhysicalView, ...]:
        stored: TransformExpr = Source(self.physical_key)
        logical_shape = self._logical_shape()
        if self.physical_shape != logical_shape:
            stored = Reshape(stored, logical_shape)
        canonical_matrix: TransformExpr = stored
        if self.orientation == "in_out":
            canonical_matrix = Transpose(canonical_matrix, (0, 1))
        segment_sizes = tuple(shape[0] for shape in self.component_flat_shapes)
        if self.interleave_groups is not None:
            canonical_matrix = Deinterleave(
                canonical_matrix, 0, self.interleave_groups, segment_sizes
            )
        views: list[PhysicalView] = []
        offset = 0
        for semantic_id, flat_shape in zip(
            self.components, self.component_flat_shapes, strict=True
        ):
            slot = slots[semantic_id]
            size = flat_shape[0]
            piece: TransformExpr = canonical_matrix
            if len(self.components) > 1:
                piece = Slice(piece, 0, offset, offset + size)
            if flat_shape != slot.canonical_shape:
                piece = Reshape(piece, slot.canonical_shape)
            if piece.infer_spec({self.physical_key: self.physical_spec()}) != slot.spec:
                raise PlanValidationError(f"physical view for {semantic_id} has the wrong spec")
            views.append(
                PhysicalView(
                    semantic_id=semantic_id,
                    physical_key=self.physical_key,
                    decode=piece,
                    canonical_spec=slot.spec,
                    supporting_nodes=self.supporting_nodes,
                    axis_lineage=self.axis_lineage,
                )
            )
            offset += size
        return tuple(views)

    def encode(self, semantic_expressions: Mapping[str, TransformExpr]) -> TransformExpr:
        pieces: list[TransformExpr] = []
        for semantic_id, flat_shape in zip(
            self.components, self.component_flat_shapes, strict=True
        ):
            try:
                expression = semantic_expressions[semantic_id]
            except KeyError as exc:
                raise PlanValidationError(f"missing semantic expression {semantic_id}") from exc
            pieces.append(Reshape(expression, flat_shape))
        combined: TransformExpr = pieces[0] if len(pieces) == 1 else Concat(tuple(pieces), 0)
        segment_sizes = tuple(shape[0] for shape in self.component_flat_shapes)
        if self.interleave_groups is not None:
            combined = Interleave(combined, 0, self.interleave_groups, segment_sizes)
        if self.orientation == "in_out":
            combined = Transpose(combined, (0, 1))
        logical_shape = self._logical_shape()
        if logical_shape != self.physical_shape:
            combined = Reshape(combined, self.physical_shape)
        return combined


@dataclass(frozen=True)
class PhysicalView:
    semantic_id: str
    physical_key: str
    decode: TransformExpr
    canonical_spec: TensorSpec
    supporting_nodes: tuple[str, ...]
    axis_lineage: tuple[AxisLineage, ...]


@dataclass(frozen=True)
class SemanticAnchor:
    anchor_id: str
    node_id: str
    canonical_shape: tuple[int, ...]
    supporting_nodes: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalModel:
    adapter_id: str
    graph: TensorProgram
    state_schema: StateSchema
    architecture: ArchitectureSignature
    slots: tuple[SemanticSlot, ...]
    layouts: tuple[PhysicalLayout, ...]
    anchors: tuple[SemanticAnchor, ...]
    assumptions: tuple[str, ...]
    unsupported_regions: tuple[str, ...]

    def slot_map(self) -> dict[str, SemanticSlot]:
        return {slot.semantic_id: slot for slot in self.slots}

    def views(self) -> tuple[PhysicalView, ...]:
        slots = self.slot_map()
        return tuple(view for layout in self.layouts for view in layout.decode_views(slots))

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], to_data(self))


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dimension in shape:
        value *= dimension
    return value
