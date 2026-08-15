"""Architecture signature models and exact compatibility diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from neuralabi.ir.serialization import to_data
from neuralabi.status import ArchitectureMismatchError


@dataclass(frozen=True)
class TwinMLPSignature:
    kind: Literal["twin_mlp"]
    input_size: int
    intermediate_size: int
    output_size: int
    bias: bool
    activation: Literal["silu"] = "silu"

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], to_data(self))


@dataclass(frozen=True)
class DecoderSignature:
    kind: Literal["decoder"]
    vocabulary_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    normalization: str
    activation: str
    position_encoding: str
    attention_bias: bool
    mlp_bias: bool
    tied_embeddings: bool

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], to_data(self))


ArchitectureSignature = TwinMLPSignature | DecoderSignature


def compare_signatures(source: ArchitectureSignature, target: ArchitectureSignature) -> None:
    source_data = source.to_dict()
    target_data = target.to_dict()
    fields = sorted(set(source_data) | set(target_data))
    differences = [
        f"{field}: source={source_data.get(field)!r}, target={target_data.get(field)!r}"
        for field in fields
        if source_data.get(field) != target_data.get(field)
    ]
    if differences:
        raise ArchitectureMismatchError("ARCHITECTURE_MISMATCH: " + "; ".join(differences))
