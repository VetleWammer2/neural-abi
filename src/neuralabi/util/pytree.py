"""Pytree comparison helpers which retain meaningful tensor paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TensorAtPath:
    path: str
    tensor: torch.Tensor


def flatten_tensors(value: Any, path: str = "output") -> list[TensorAtPath]:
    if isinstance(value, torch.Tensor):
        return [TensorAtPath(path, value)]
    if isinstance(value, Mapping):
        result: list[TensorAtPath] = []
        for key in sorted(value, key=str):
            result.extend(flatten_tensors(value[key], f"{path}.{key}"))
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        result = []
        for index, child in enumerate(value):
            result.extend(flatten_tensors(child, f"{path}[{index}]"))
        return result
    raise TypeError(f"selected output at {path} is not a tensor pytree: {type(value).__name__}")
