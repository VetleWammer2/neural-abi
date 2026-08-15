"""Physical parameter-gradient verification through inverse-plan adjoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from neuralabi.adapters import ModelAdapter
from neuralabi.formats.plan import ConversionPlan
from neuralabi.status import PlanValidationError
from neuralabi.transforms.adjoint import expression_adjoint
from neuralabi.transforms.shapes import TensorSpec
from neuralabi.util.pytree import flatten_tensors
from neuralabi.verify.forward import TensorComparison, compare_tensor


@dataclass(frozen=True)
class GradientVerification:
    comparisons: tuple[TensorComparison, ...]
    objective_value_source: float
    objective_value_target: float


@dataclass(frozen=True)
class InputGradientVerification:
    comparisons: tuple[TensorComparison, ...]
    objective_value_source: float
    objective_value_target: float


def _physical_gradients(model: nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    seen: set[int] = set()
    for key, tensor in model.state_dict(keep_vars=True).items():
        if isinstance(tensor, nn.Parameter) and tensor.grad is not None:
            identity = id(tensor)
            if identity in seen:
                continue
            seen.add(identity)
            result[key] = tensor.grad.detach().clone()
    return result


def _objective(selected: Any, seed: int) -> torch.Tensor:
    tensors = flatten_tensors(selected)
    if not tensors:
        raise PlanValidationError("selected output contains no tensors")
    result = torch.zeros((), dtype=torch.float32, device=tensors[0].tensor.device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 9701)
    for item in tensors:
        probe = torch.randn(item.tensor.shape, generator=generator, dtype=torch.float32).to(
            item.tensor.device
        )
        result = result + (item.tensor.float() * probe).sum()
    return result


def verify_parameter_gradients(
    plan: ConversionPlan,
    source_model: nn.Module,
    target_model: nn.Module,
    source_adapter: ModelAdapter,
    target_adapter: ModelAdapter,
    source_args: tuple[Any, ...],
    source_kwargs: dict[str, Any],
    target_args: tuple[Any, ...],
    target_kwargs: dict[str, Any],
    *,
    seed: int,
) -> GradientVerification:
    if plan.inverse_targets is None:
        raise PlanValidationError("gradient verification requires an inverse plan")
    source_model.zero_grad(set_to_none=True)
    target_model.zero_grad(set_to_none=True)
    source_objective = _objective(
        source_adapter.select_outputs(source_model(*source_args, **source_kwargs)), seed
    )
    target_objective = _objective(
        target_adapter.select_outputs(target_model(*target_args, **target_kwargs)), seed
    )
    source_objective.backward()  # type: ignore[no-untyped-call]
    target_objective.backward()  # type: ignore[no-untyped-call]
    source_gradients = _physical_gradients(source_model)
    target_gradients = _physical_gradients(target_model)
    target_specs = {
        key: TensorSpec(target.shape, target.dtype) for key, target in plan.targets.items()
    }
    expected_target: dict[str, torch.Tensor] = {}
    for source_key, inverse in sorted(plan.inverse_targets.items()):
        if source_key not in source_gradients:
            continue
        contributions = expression_adjoint(
            inverse.expression, source_gradients[source_key], target_specs
        )
        for key, contribution in contributions.items():
            expected_target[key] = (
                expected_target[key] + contribution if key in expected_target else contribution
            )
    comparisons: list[TensorComparison] = []
    for key in sorted(set(expected_target) | set(target_gradients)):
        if key not in expected_target or key not in target_gradients:
            missing = expected_target.get(key, torch.empty(0))
            actual = target_gradients.get(key, torch.empty(1))
            comparisons.append(compare_tensor(missing, actual, path=f"parameter_gradient.{key}"))
        else:
            comparisons.append(
                compare_tensor(
                    expected_target[key], target_gradients[key], path=f"parameter_gradient.{key}"
                )
            )
    return GradientVerification(
        tuple(comparisons), float(source_objective.detach()), float(target_objective.detach())
    )


def verify_input_gradients(
    source_model: nn.Module,
    target_model: nn.Module,
    source_adapter: ModelAdapter,
    target_adapter: ModelAdapter,
    source_args: tuple[Any, ...],
    source_kwargs: dict[str, Any],
    target_args: tuple[Any, ...],
    target_kwargs: dict[str, Any],
    *,
    seed: int,
) -> InputGradientVerification:
    source_model.zero_grad(set_to_none=True)
    target_model.zero_grad(set_to_none=True)
    source_objective = _objective(
        source_adapter.select_outputs(source_model(*source_args, **source_kwargs)), seed
    )
    target_objective = _objective(
        target_adapter.select_outputs(target_model(*target_args, **target_kwargs)), seed
    )
    source_objective.backward()  # type: ignore[no-untyped-call]
    target_objective.backward()  # type: ignore[no-untyped-call]
    source_inputs = flatten_tensors((source_args, source_kwargs), path="input")
    target_inputs = flatten_tensors((target_args, target_kwargs), path="input")
    if [item.path for item in source_inputs] != [item.path for item in target_inputs]:
        raise PlanValidationError("differentiable input pytrees are not aligned")
    comparisons: list[TensorComparison] = []
    for source_item, target_item in zip(source_inputs, target_inputs, strict=True):
        source_gradient = source_item.tensor.grad
        target_gradient = target_item.tensor.grad
        if source_gradient is None or target_gradient is None:
            raise PlanValidationError(
                f"differentiable input {source_item.path} did not receive a gradient"
            )
        comparisons.append(
            compare_tensor(
                source_gradient,
                target_gradient,
                path=f"input_gradient.{source_item.path}",
            )
        )
    return InputGradientVerification(
        tuple(comparisons), float(source_objective.detach()), float(target_objective.detach())
    )
