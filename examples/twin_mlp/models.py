"""Opaque equivalent MLPs; no fixture mapping metadata is exposed."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class _SourceVault(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(1907)
        self.n7c2 = nn.Parameter(torch.randn(12, 8, generator=generator, dtype=dtype) / 8)
        self.p4f9 = nn.Parameter(torch.randn(12, 8, generator=generator, dtype=dtype) / 8)
        self.r1a6 = nn.Parameter(torch.randn(8, 12, generator=generator, dtype=dtype) / 12)


class OpaqueSource(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.x3d8 = _SourceVault(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = F.linear(x, self.x3d8.n7c2)
        right = F.linear(x, self.x3d8.p4f9)
        return F.linear(F.silu(left) * right, self.x3d8.r1a6)


class _TargetVault(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(4001)
        self.b8e0 = nn.Parameter(torch.randn(8, 24, generator=generator, dtype=dtype) / 8)
        self.c2a5 = nn.Parameter(torch.randn(12, 8, generator=generator, dtype=dtype) / 12)


class _TargetNest(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.j6f3 = _TargetVault(dtype)


class OpaqueTarget(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.w9b1 = _TargetNest(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = x @ self.w9b1.j6f3.b8e0
        first, second = torch.split(packed, [12, 12], dim=-1)
        return (F.silu(second) * first) @ self.w9b1.j6f3.c2a5


class _Adapter:
    def __init__(self, adapter_id: str, model_type: type[nn.Module]) -> None:
        self.adapter_id = adapter_id
        self._model_type = model_type

    def build(self, *, device: torch.device | str, dtype: torch.dtype) -> nn.Module:
        return self._model_type(dtype=dtype).to(device=device)

    def prepare(self, model: nn.Module) -> None:
        model.eval()

    def example_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        generator = torch.Generator(device="cpu").manual_seed(seed + 17)
        value = torch.randn(2, 5, 8, generator=generator).to(device)
        return (value,), {}

    def select_outputs(self, output: Any) -> Any:
        return output

    def dynamic_shapes(self) -> Any | None:
        return None

    def differentiable_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        args, kwargs = self.example_inputs(seed=seed, device=device)
        return (args[0].detach().requires_grad_(True),), kwargs


source_adapter = _Adapter("opaque-twin-source", OpaqueSource)
target_adapter = _Adapter("opaque-twin-target", OpaqueTarget)
