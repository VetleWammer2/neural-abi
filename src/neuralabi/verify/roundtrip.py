"""Bitwise inverse round-trip verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from neuralabi.formats.plan import ConversionPlan
from neuralabi.status import PlanValidationError
from neuralabi.transforms.execute import execute_checked


@dataclass(frozen=True)
class RoundTripTensor:
    key: str
    exact: bool


def verify_roundtrip(
    plan: ConversionPlan,
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> tuple[RoundTripTensor, ...]:
    if plan.inverse_targets is None:
        raise PlanValidationError("plan has no exact inverse")
    results: list[RoundTripTensor] = []
    for key, inverse in sorted(plan.inverse_targets.items()):
        dependencies = {
            source_key: target_state[source_key]
            for source_key in set(inverse.expression.source_keys())
        }
        recovered = execute_checked(inverse.expression, dependencies, expected=inverse.spec)
        results.append(RoundTripTensor(key, torch.equal(recovered, source_state[key])))
    return tuple(results)
