"""Frozen export metadata consumed outside the capture boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from neuralabi.ir.graph import TensorProgram
from neuralabi.ir.serialization import to_data
from neuralabi.ir.state import StateSchema


@dataclass(frozen=True)
class ExportOptions:
    strict: bool
    canonicalization_version: int
    functional_aten: bool
    dynamic_shapes_supplied: bool


@dataclass(frozen=True)
class InputSignature:
    positional_count: int
    keyword_names: tuple[str, ...]
    tensor_specs: tuple[tuple[str, tuple[int | str, ...], str], ...]


@dataclass(frozen=True)
class ExportArtifact:
    adapter_id: str
    torch_version: str
    graph: TensorProgram
    state_schema: StateSchema
    graph_signature: dict[str, Any]
    range_constraints: tuple[str, ...]
    example_input_signature: InputSignature
    export_options: ExportOptions
    graph_hash: str

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], to_data(self))
