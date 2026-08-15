"""Recursive deterministic forward verification with explicit numeric evidence."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import torch

from neuralabi.util.pytree import flatten_tensors


@dataclass(frozen=True)
class NumericTolerance:
    absolute: float
    relative: float


@dataclass(frozen=True)
class TensorComparison:
    path: str
    shape: tuple[int, ...]
    dtype: str
    max_absolute_error: float
    max_relative_error: float
    mismatch_count: int
    nan_count_source: int
    nan_count_target: int
    inf_count_source: int
    inf_count_target: int
    tolerance: NumericTolerance
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "mismatch_count": self.mismatch_count,
            "nan_count_source": self.nan_count_source,
            "nan_count_target": self.nan_count_target,
            "inf_count_source": self.inf_count_source,
            "inf_count_target": self.inf_count_target,
            "tolerance": {
                "absolute": self.tolerance.absolute,
                "relative": self.tolerance.relative,
            },
            "passed": self.passed,
        }


def default_tolerance(dtype: torch.dtype) -> NumericTolerance:
    if dtype == torch.float64:
        return NumericTolerance(1e-8, 1e-7)
    if dtype == torch.float32:
        return NumericTolerance(2e-5, 2e-4)
    if dtype == torch.float16:
        return NumericTolerance(3e-3, 3e-2)
    if dtype == torch.bfloat16:
        return NumericTolerance(2e-2, 5e-2)
    return NumericTolerance(0.0, 0.0)


def compare_tensor(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    path: str,
    tolerance: NumericTolerance | None = None,
) -> TensorComparison:
    dtype_name = str(source.dtype).removeprefix("torch.")
    selected = tolerance or default_tolerance(source.dtype)
    if source.shape != target.shape or source.dtype != target.dtype:
        return TensorComparison(
            path,
            tuple(source.shape),
            dtype_name,
            sys.float_info.max,
            sys.float_info.max,
            max(source.numel(), target.numel()),
            int(torch.isnan(source).sum()) if source.is_floating_point() else 0,
            int(torch.isnan(target).sum()) if target.is_floating_point() else 0,
            int(torch.isinf(source).sum()) if source.is_floating_point() else 0,
            int(torch.isinf(target).sum()) if target.is_floating_point() else 0,
            selected,
            False,
        )
    if not source.is_floating_point() and not source.is_complex():
        mismatch = int(torch.count_nonzero(source != target))
        return TensorComparison(
            path,
            tuple(source.shape),
            dtype_name,
            float(mismatch > 0),
            float(mismatch > 0),
            mismatch,
            0,
            0,
            0,
            0,
            selected,
            mismatch == 0,
        )
    source_detached = source.detach()
    target_detached = target.detach()
    source_nan = torch.isnan(source_detached)
    target_nan = torch.isnan(target_detached)
    source_inf = torch.isinf(source_detached)
    target_inf = torch.isinf(target_detached)
    special_equal = (
        torch.equal(source_nan, target_nan)
        and torch.equal(source_detached[source_inf], target_detached[target_inf])
        and torch.equal(source_inf, target_inf)
    )
    finite = torch.isfinite(source_detached) & torch.isfinite(target_detached)
    if finite.any():
        source_finite = source_detached[finite].to(torch.float64)
        target_finite = target_detached[finite].to(torch.float64)
        absolute = torch.abs(source_finite - target_finite)
        denominator = torch.maximum(
            torch.maximum(torch.abs(source_finite), torch.abs(target_finite)),
            torch.tensor(torch.finfo(torch.float64).tiny),
        )
        relative = absolute / denominator
        allowed = selected.absolute + selected.relative * torch.abs(source_finite)
        mismatch = int(torch.count_nonzero(absolute > allowed))
        max_absolute = float(absolute.max()) if absolute.numel() else 0.0
        max_relative = float(relative.max()) if relative.numel() else 0.0
    else:
        mismatch = 0
        max_absolute = 0.0
        max_relative = 0.0
    special_mismatch = int(torch.count_nonzero(source_nan != target_nan)) + int(
        torch.count_nonzero(source_inf != target_inf)
    )
    mismatch += special_mismatch
    return TensorComparison(
        path,
        tuple(source.shape),
        dtype_name,
        max_absolute,
        max_relative,
        mismatch,
        int(source_nan.sum()),
        int(target_nan.sum()),
        int(source_inf.sum()),
        int(target_inf.sum()),
        selected,
        mismatch == 0 and special_equal,
    )


def compare_pytrees(source: Any, target: Any) -> tuple[TensorComparison, ...]:
    source_items = flatten_tensors(source)
    target_items = flatten_tensors(target)
    if [item.path for item in source_items] != [item.path for item in target_items]:
        return (
            TensorComparison(
                "output",
                (),
                "structure",
                sys.float_info.max,
                sys.float_info.max,
                1,
                0,
                0,
                0,
                0,
                NumericTolerance(0.0, 0.0),
                False,
            ),
        )
    return tuple(
        compare_tensor(source_item.tensor, target_item.tensor, path=source_item.path)
        for source_item, target_item in zip(source_items, target_items, strict=True)
    )
