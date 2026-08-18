"""Numerical evidence for resumed training after model and optimizer conversion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neuralabi.apply.engine import apply_to_state
from neuralabi.export.capture import CapturedModel
from neuralabi.formats.optimizer import tensors_bitwise_equal
from neuralabi.formats.plan import ConversionPlan
from neuralabi.optimizers.torch import load_optimizer_state
from neuralabi.status import PlanValidationError
from neuralabi.util.pytree import TensorAtPath, flatten_tensors
from neuralabi.verify.forward import NumericTolerance, TensorComparison, compare_tensor

FLOAT32_RESUME_TOLERANCE = NumericTolerance(1e-7, 1e-5)
FLOAT64_RESUME_TOLERANCE = NumericTolerance(1e-12, 1e-10)


@dataclass(frozen=True)
class ResumeTensorEvidence:
    """One tolerance comparison plus an observational bitwise-equality flag."""

    comparison: TensorComparison
    exact: bool

    @property
    def passed(self) -> bool:
        return self.comparison.passed

    def to_dict(self) -> dict[str, Any]:
        return {**self.comparison.to_dict(), "exact": self.exact}


@dataclass(frozen=True)
class ResumeStepEvidence:
    """Loss, post-update output, and physical model-state evidence for one step."""

    step: int
    seed: int
    loss: ResumeTensorEvidence
    outputs: tuple[ResumeTensorEvidence, ...]
    model_state: tuple[ResumeTensorEvidence, ...]

    @property
    def passed(self) -> bool:
        return (
            self.loss.passed
            and all(item.passed for item in self.outputs)
            and all(item.passed for item in self.model_state)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "seed": self.seed,
            "loss": self.loss.to_dict(),
            "outputs": [item.to_dict() for item in self.outputs],
            "model_state": [item.to_dict() for item in self.model_state],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ResumedTrainingEvidence:
    """Evidence under the documented numerical contract, not a bitwise-resume claim."""

    optimizer_algorithm: str
    steps: tuple[ResumeStepEvidence, ...]
    source_device: str
    target_device: str
    source_backend: str
    target_backend: str
    seed_base: int
    deterministic_algorithms_enabled: bool

    @property
    def float32_tolerance(self) -> NumericTolerance:
        return FLOAT32_RESUME_TOLERANCE

    @property
    def float64_tolerance(self) -> NumericTolerance:
        return FLOAT64_RESUME_TOLERANCE

    @property
    def passed(self) -> bool:
        return bool(self.steps) and all(item.passed for item in self.steps)

    def numerical_contract(self) -> dict[str, Any]:
        return {
            "version": "neuralabi.resumed-training.v1",
            "execution": {
                "model_mode": "eval",
                "source_device": self.source_device,
                "target_device": self.target_device,
                "source_backend": self.source_backend,
                "target_backend": self.target_backend,
                "pytorch_version": str(torch.__version__),
                "deterministic_algorithms_enabled": self.deterministic_algorithms_enabled,
            },
            "objective": {
                "id": "selected-output-seeded-linear-probe-v1",
                "inputs": "adapter.example_inputs",
                "values": "adapter.select_outputs floating tensors",
            },
            "seed_policy": {
                "seed_base": self.seed_base,
                "input_seed": "seed_base + step_index",
                "objective_probe_seed": "seed_base + 10000 + step_index",
                "post_update_forward_seed": "seed_base + 20000 + step_index",
            },
            "comparison_timing": {
                "loss": "pre_update",
                "outputs": "post_update",
                "model_state": "post_update",
                "model_state_reference": "plan(updated_source_state)",
            },
            "tolerances": {
                "float32": {
                    "absolute": self.float32_tolerance.absolute,
                    "relative": self.float32_tolerance.relative,
                },
                "float64": {
                    "absolute": self.float64_tolerance.absolute,
                    "relative": self.float64_tolerance.relative,
                },
            },
            "steps_requested": len(self.steps),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_algorithm": self.optimizer_algorithm,
            "numerical_contract": self.numerical_contract(),
            "steps": [item.to_dict() for item in self.steps],
            "passed": self.passed,
        }


def _tolerance(dtype: torch.dtype) -> NumericTolerance:
    if dtype == torch.float32:
        return FLOAT32_RESUME_TOLERANCE
    if dtype == torch.float64:
        return FLOAT64_RESUME_TOLERANCE
    if dtype.is_floating_point or dtype.is_complex:
        raise PlanValidationError(
            f"resumed-training verification has no numerical contract for {dtype}"
        )
    return NumericTolerance(0.0, 0.0)


def _compare(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    path: str,
) -> ResumeTensorEvidence:
    source_cpu = source.detach().cpu()
    target_cpu = target.detach().cpu()
    exact = tensors_bitwise_equal(source_cpu, target_cpu)
    comparison = compare_tensor(
        source_cpu,
        target_cpu,
        path=path,
        tolerance=_tolerance(source_cpu.dtype),
    )
    return ResumeTensorEvidence(comparison, exact)


def _model_device(model: torch.nn.Module) -> torch.device:
    first = next(model.parameters(), None)
    if first is None:
        raise PlanValidationError("resumed-training verification requires model parameters")
    return first.device


def _selected_output(
    capture: CapturedModel,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    return capture.adapter.select_outputs(capture.model(*args, **kwargs))


def _aligned_outputs(source: Any, target: Any) -> tuple[list[TensorAtPath], list[TensorAtPath]]:
    source_items = flatten_tensors(source)
    target_items = flatten_tensors(target)
    if [item.path for item in source_items] != [item.path for item in target_items]:
        raise PlanValidationError("source and target selected-output structures differ")
    for source_item, target_item in zip(source_items, target_items, strict=True):
        if source_item.tensor.shape != target_item.tensor.shape:
            raise PlanValidationError(
                f"selected output {source_item.path} has mismatched shapes: "
                f"{tuple(source_item.tensor.shape)} != {tuple(target_item.tensor.shape)}"
            )
        if source_item.tensor.dtype != target_item.tensor.dtype:
            raise PlanValidationError(
                f"selected output {source_item.path} has mismatched dtypes: "
                f"{source_item.tensor.dtype} != {target_item.tensor.dtype}"
            )
        if not source_item.tensor.is_floating_point():
            raise PlanValidationError(f"selected output {source_item.path} is not floating point")
        _tolerance(source_item.tensor.dtype)
    return source_items, target_items


def _shared_probe_objectives(
    source: Any,
    target: Any,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_items, target_items = _aligned_outputs(source, target)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source_objective: torch.Tensor | None = None
    target_objective: torch.Tensor | None = None
    for source_item, target_item in zip(source_items, target_items, strict=True):
        probe_cpu = torch.randn(
            source_item.tensor.shape,
            generator=generator,
            dtype=source_item.tensor.dtype,
            device="cpu",
        )
        source_term = (source_item.tensor * probe_cpu.to(device=source_item.tensor.device)).sum()
        target_term = (target_item.tensor * probe_cpu.to(device=target_item.tensor.device)).sum()
        source_objective = (
            source_term if source_objective is None else source_objective + source_term
        )
        target_objective = (
            target_term if target_objective is None else target_objective + target_term
        )
    if source_objective is None or target_objective is None:
        raise PlanValidationError("selected outputs contain no tensors")
    return source_objective, target_objective


def _optimizer_algorithm(optimizer: torch.optim.Optimizer) -> str:
    if type(optimizer) is torch.optim.Adam:
        return "adam"
    if type(optimizer) is torch.optim.AdamW:
        return "adamw"
    raise PlanValidationError(
        f"resumed-training verification does not support {type(optimizer).__name__}"
    )


def verify_resumed_training(
    plan: ConversionPlan,
    source_capture: CapturedModel,
    target_capture: CapturedModel,
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    source_optimizer_path: Path,
    target_optimizer_path: Path,
    *,
    steps: int,
    seed_base: int,
) -> ResumedTrainingEvidence:
    """Resume both implementations and compare them after every synchronized seeded update."""

    if steps < 1:
        raise PlanValidationError("resumed-training verification requires at least one step")
    source_capture.model.load_state_dict(source_state, strict=True)
    target_capture.model.load_state_dict(target_state, strict=True)
    source_capture.model.eval()
    target_capture.model.eval()
    source_capture.model.zero_grad(set_to_none=True)
    target_capture.model.zero_grad(set_to_none=True)
    source_optimizer = load_optimizer_state(source_capture.model, source_optimizer_path)
    target_optimizer = load_optimizer_state(target_capture.model, target_optimizer_path)
    source_algorithm = _optimizer_algorithm(source_optimizer)
    target_algorithm = _optimizer_algorithm(target_optimizer)
    if source_algorithm != target_algorithm:
        raise PlanValidationError(
            f"source and target optimizers differ: {source_algorithm} != {target_algorithm}"
        )
    source_device = _model_device(source_capture.model)
    target_device = _model_device(target_capture.model)
    evidence: list[ResumeStepEvidence] = []
    for step in range(steps):
        seed = seed_base + step
        source_args, source_kwargs = source_capture.adapter.example_inputs(
            seed=seed, device=source_device
        )
        target_args, target_kwargs = target_capture.adapter.example_inputs(
            seed=seed, device=target_device
        )
        source_optimizer.zero_grad(set_to_none=True)
        target_optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        source_output = _selected_output(source_capture, source_args, source_kwargs)
        torch.manual_seed(seed)
        target_output = _selected_output(target_capture, target_args, target_kwargs)
        source_loss, target_loss = _shared_probe_objectives(
            source_output,
            target_output,
            seed=seed_base + 10_000 + step,
        )
        loss_evidence = _compare(
            source_loss,
            target_loss,
            path=f"resume.step[{step}].loss",
        )
        source_loss.backward()  # type: ignore[no-untyped-call]
        target_loss.backward()  # type: ignore[no-untyped-call]
        source_optimizer.step()
        target_optimizer.step()
        with torch.no_grad():
            torch.manual_seed(seed_base + 20_000 + step)
            source_after = _selected_output(source_capture, source_args, source_kwargs)
            torch.manual_seed(seed_base + 20_000 + step)
            target_after = _selected_output(target_capture, target_args, target_kwargs)
        source_items, target_items = _aligned_outputs(source_after, target_after)
        output_evidence = tuple(
            _compare(
                source_item.tensor,
                target_item.tensor,
                path=f"resume.step[{step}].{source_item.path}",
            )
            for source_item, target_item in zip(source_items, target_items, strict=True)
        )
        updated_source_state = {
            key: value.detach() for key, value in source_capture.model.state_dict().items()
        }
        expected_target_state = apply_to_state(plan, updated_source_state)
        actual_target_state = target_capture.model.state_dict()
        state_evidence = tuple(
            _compare(
                expected_target_state[key],
                actual_target_state[key],
                path=f"resume.step[{step}].model_state.{key}",
            )
            for key in sorted(expected_target_state)
        )
        evidence.append(
            ResumeStepEvidence(
                step=step,
                seed=seed,
                loss=loss_evidence,
                outputs=output_evidence,
                model_state=state_evidence,
            )
        )
    return ResumedTrainingEvidence(
        optimizer_algorithm=source_algorithm,
        steps=tuple(evidence),
        source_device=str(source_device),
        target_device=str(target_device),
        source_backend=source_device.type,
        target_backend=target_device.type,
        seed_base=seed_base,
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
    )
