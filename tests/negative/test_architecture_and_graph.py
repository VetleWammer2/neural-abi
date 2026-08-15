from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from examples.generated_transformer.models import GeneratedAdapter, LogicalConfig
from neuralabi.export.capture import capture_adapter
from neuralabi.export.signature import compare_signatures
from neuralabi.recognize.decoder import recognize_decoder
from neuralabi.status import ArchitectureMismatchError, UnsupportedGraphError


def _model(config: LogicalConfig, side: str = "source") -> object:
    return recognize_decoder(capture_adapter(GeneratedAdapter(config, side)).artifact)


@pytest.mark.parametrize(
    ("source_config", "target_config", "field"),
    [
        (
            LogicalConfig(hidden_size=32, attention_heads=4, key_value_heads=2),
            LogicalConfig(hidden_size=48, attention_heads=6, key_value_heads=2),
            "hidden_size",
        ),
        (LogicalConfig(layers=1), LogicalConfig(layers=2), "num_layers"),
        (
            LogicalConfig(attention_heads=4, key_value_heads=2),
            LogicalConfig(attention_heads=4, key_value_heads=4),
            "num_key_value_heads",
        ),
    ],
)
def test_architecture_mismatch_has_field_diff(
    source_config: LogicalConfig, target_config: LogicalConfig, field: str
) -> None:
    source = _model(source_config)
    target = _model(target_config, "target")
    with pytest.raises(ArchitectureMismatchError, match=field):
        compare_signatures(source.architecture, target.architecture)  # type: ignore[attr-defined]


def test_gelu_target_is_rejected_as_unsupported() -> None:
    target = capture_adapter(GeneratedAdapter(LogicalConfig(activation="gelu"), "target"))
    with pytest.raises(UnsupportedGraphError, match="attention/SwiGLU"):
        recognize_decoder(target.artifact)


class _ParameterizedWrapper(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.opaque_base = base
        self.extra = nn.Parameter(torch.ones(()))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.opaque_base(tokens) * self.extra


class _WrappedAdapter:
    adapter_id = "unsupported-parameterized-wrapper"

    def __init__(self, *, use_extra: bool) -> None:
        self.base = GeneratedAdapter(LogicalConfig(layers=1), "target")
        self.use_extra = use_extra

    def build(self, *, device: torch.device | str, dtype: torch.dtype) -> nn.Module:
        base = self.base.build(device=device, dtype=dtype)
        if self.use_extra:
            return _ParameterizedWrapper(base).to(device=device, dtype=dtype)
        base.register_parameter(
            "unused_opaque", nn.Parameter(torch.ones((), device=device, dtype=dtype))
        )
        return base

    def prepare(self, model: nn.Module) -> None:
        model.eval()

    def example_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return self.base.example_inputs(seed=seed, device=device)

    def select_outputs(self, output: Any) -> Any:
        return output

    def dynamic_shapes(self) -> Any | None:
        return None

    def differentiable_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        return None


@pytest.mark.parametrize("use_extra", [False, True])
def test_uncovered_or_unsupported_parameterized_state_is_rejected(use_extra: bool) -> None:
    capture = capture_adapter(_WrappedAdapter(use_extra=use_extra))
    with pytest.raises(UnsupportedGraphError, match="unrecognized persistent decoder state"):
        recognize_decoder(capture.artifact)
