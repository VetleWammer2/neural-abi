"""The narrow, trusted adapter boundary."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from neuralabi.status import NeuralABIError


@runtime_checkable
class ModelAdapter(Protocol):
    """Trusted local code which constructs and drives one model implementation."""

    adapter_id: str

    def build(self, *, device: torch.device | str, dtype: torch.dtype) -> nn.Module: ...

    def prepare(self, model: nn.Module) -> None: ...

    def example_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]]: ...

    def select_outputs(self, output: Any) -> Any: ...

    def dynamic_shapes(self) -> Any | None: ...

    def differentiable_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None: ...


def load_adapter(spec: str) -> ModelAdapter:
    """Load ``module:attribute`` from trusted local Python code."""

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise NeuralABIError(f"adapter must be MODULE:ATTRIBUTE, got {spec!r}")
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise NeuralABIError(f"cannot load adapter {spec!r}: {exc}") from exc
    adapter = value() if isinstance(value, type) else value
    if not isinstance(adapter, ModelAdapter):
        raise NeuralABIError(f"{spec!r} does not implement the ModelAdapter protocol")
    if not adapter.adapter_id or not isinstance(adapter.adapter_id, str):
        raise NeuralABIError(f"{spec!r} has an invalid adapter_id")
    return adapter
