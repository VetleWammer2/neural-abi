"""End-to-end verification orchestration shared by ``verify`` and ``link``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from neuralabi.apply.engine import apply_to_state
from neuralabi.export.capture import CapturedModel
from neuralabi.formats.certificate import (
    VerificationCertificate,
    VerificationClaim,
    certificate_timestamp,
)
from neuralabi.formats.plan import ConversionPlan, validate_plan
from neuralabi.ir.semantic import CanonicalModel
from neuralabi.status import ClaimStatus, PlanValidationError
from neuralabi.verify.forward import compare_pytrees
from neuralabi.verify.gradients import verify_input_gradients, verify_parameter_gradients
from neuralabi.verify.intermediates import compare_anchors, record_graph_values
from neuralabi.verify.optimizer import verify_optimizer_state
from neuralabi.verify.resume import verify_resumed_training
from neuralabi.verify.roundtrip import verify_roundtrip


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def _load_runtime_state(module: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected = set(module.state_dict())
    selected = {key: value for key, value in state.items() if key in expected}
    module.load_state_dict(selected, strict=True)


def verify_conversion(
    plan: ConversionPlan,
    source_capture: CapturedModel,
    target_capture: CapturedModel,
    source_canonical: CanonicalModel,
    target_canonical: CanonicalModel,
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor] | None = None,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3),
    source_checkpoint_fingerprint: str,
    target_checkpoint_fingerprint: str,
    generated_file_hashes: dict[str, str] | None = None,
    source_optimizer_state: Path | None = None,
    target_optimizer_state: Path | None = None,
    resume_steps: int = 0,
    resume_seed_base: int = 30_000,
) -> VerificationCertificate:
    validate_plan(plan)
    if (source_optimizer_state is None) != (target_optimizer_state is None):
        raise PlanValidationError("source and target optimizer states must be supplied together")
    if resume_steps < 0 or resume_steps > 128:
        raise PlanValidationError("resume_steps must be between zero and 128")
    if resume_steps and source_optimizer_state is None:
        raise PlanValidationError("resumed training requires source and target optimizer states")
    target_values = (
        apply_to_state(plan, source_state) if target_state is None else _clone_state(target_state)
    )
    source_capture.model.load_state_dict(source_state, strict=True)
    target_capture.model.load_state_dict(target_values, strict=True)
    _load_runtime_state(source_capture.runtime_graph, source_state)
    _load_runtime_state(target_capture.runtime_graph, target_values)
    source_capture.model.eval()
    target_capture.model.eval()
    forward_results: list[dict[str, Any]] = []
    intermediate_results: list[dict[str, Any]] = []
    all_forward = True
    all_intermediate = True
    first_divergence: str | None = None
    with torch.no_grad():
        for seed in seeds:
            source_args, source_kwargs = source_capture.adapter.example_inputs(
                seed=seed, device="cpu"
            )
            target_args, target_kwargs = target_capture.adapter.example_inputs(
                seed=seed, device="cpu"
            )
            source_output = source_capture.adapter.select_outputs(
                source_capture.model(*source_args, **source_kwargs)
            )
            target_output = target_capture.adapter.select_outputs(
                target_capture.model(*target_args, **target_kwargs)
            )
            comparisons = compare_pytrees(source_output, target_output)
            forward_results.append(
                {"seed": seed, "comparisons": [item.to_dict() for item in comparisons]}
            )
            for comparison in comparisons:
                if not comparison.passed:
                    all_forward = False
                    first_divergence = first_divergence or comparison.path
            source_nodes = record_graph_values(
                source_capture.runtime_graph, source_args, source_kwargs
            )
            target_nodes = record_graph_values(
                target_capture.runtime_graph, target_args, target_kwargs
            )
            anchors = compare_anchors(
                source_nodes, target_nodes, source_canonical, target_canonical
            )
            intermediate_results.append(
                {"seed": seed, "comparisons": [item.to_dict() for item in anchors]}
            )
            for comparison in anchors:
                if not comparison.passed:
                    all_intermediate = False
                    if first_divergence is None or first_divergence.startswith("output"):
                        first_divergence = comparison.path
    gradient_args_source, gradient_kwargs_source = source_capture.adapter.example_inputs(
        seed=seeds[0], device="cpu"
    )
    gradient_args_target, gradient_kwargs_target = target_capture.adapter.example_inputs(
        seed=seeds[0], device="cpu"
    )
    gradients = verify_parameter_gradients(
        plan,
        source_capture.model,
        target_capture.model,
        source_capture.adapter,
        target_capture.adapter,
        gradient_args_source,
        gradient_kwargs_source,
        gradient_args_target,
        gradient_kwargs_target,
        seed=seeds[0],
    )
    gradient_results = tuple(item.to_dict() for item in gradients.comparisons)
    all_gradients = all(item.passed for item in gradients.comparisons)
    if not all_gradients and first_divergence is None:
        first_divergence = next(item.path for item in gradients.comparisons if not item.passed)
    source_differentiable = source_capture.adapter.differentiable_inputs(
        seed=seeds[0], device="cpu"
    )
    target_differentiable = target_capture.adapter.differentiable_inputs(
        seed=seeds[0], device="cpu"
    )
    input_gradient_status = ClaimStatus.NOT_APPLICABLE
    input_gradient_detail = "both adapters reported no equivalent differentiable inputs"
    input_gradient_results: tuple[dict[str, Any], ...] = ()
    if (source_differentiable is None) != (target_differentiable is None):
        input_gradient_status = ClaimStatus.INCONCLUSIVE
        input_gradient_detail = "only one adapter supplied differentiable inputs"
    elif source_differentiable is not None and target_differentiable is not None:
        input_gradients = verify_input_gradients(
            source_capture.model,
            target_capture.model,
            source_capture.adapter,
            target_capture.adapter,
            source_differentiable[0],
            source_differentiable[1],
            target_differentiable[0],
            target_differentiable[1],
            seed=seeds[0],
        )
        input_gradient_results = tuple(item.to_dict() for item in input_gradients.comparisons)
        input_gradient_status = (
            ClaimStatus.VERIFIED
            if all(item.passed for item in input_gradients.comparisons)
            else ClaimStatus.FAILED
        )
        input_gradient_detail = f"{len(input_gradients.comparisons)} differentiable input tensors"
        if input_gradient_status == ClaimStatus.FAILED and first_divergence is None:
            first_divergence = next(
                item.path for item in input_gradients.comparisons if not item.passed
            )
    roundtrip = verify_roundtrip(plan, source_state, target_values)
    roundtrip_results = tuple({"key": item.key, "exact": item.exact} for item in roundtrip)
    all_roundtrip = all(item.exact for item in roundtrip)
    if not all_roundtrip and first_divergence is None:
        first_divergence = next(item.key for item in roundtrip if not item.exact)

    optimizer_supplied = source_optimizer_state is not None
    optimizer_coverage: dict[str, Any] = {
        "complete": False,
        "reason": "no optimizer state supplied",
    }
    optimizer_associations: tuple[dict[str, Any], ...] = ()
    optimizer_state_results: tuple[dict[str, Any], ...] = ()
    optimizer_converted_status = ClaimStatus.NOT_APPLICABLE
    optimizer_coverage_status = ClaimStatus.NOT_APPLICABLE
    optimizer_verification_status = ClaimStatus.NOT_APPLICABLE
    optimizer_detail = "no optimizer state supplied"
    optimizer_pass = True
    if source_optimizer_state is not None and target_optimizer_state is not None:
        optimizer = verify_optimizer_state(
            plan,
            source_optimizer_state,
            target_optimizer_state,
            target_checkpoint_fingerprint=target_checkpoint_fingerprint,
        )
        optimizer_coverage = optimizer.coverage_dict()
        optimizer_associations = optimizer.associations
        optimizer_state_results = tuple(item.to_dict() for item in optimizer.comparisons)
        optimizer_converted_status = ClaimStatus.VERIFIED
        optimizer_coverage_status = (
            ClaimStatus.VERIFIED if optimizer.coverage_complete else ClaimStatus.FAILED
        )
        optimizer_verification_status = (
            ClaimStatus.VERIFIED if optimizer.passed else ClaimStatus.FAILED
        )
        optimizer_detail = (
            f"{optimizer.algorithm} state for "
            f"{optimizer.target_parameter_count} unique target parameters"
        )
        optimizer_pass = optimizer.passed
        if not optimizer.passed and first_divergence is None:
            first_divergence = next(
                item.comparison.path for item in optimizer.comparisons if not item.passed
            )

    resume_results: tuple[dict[str, Any], ...] = ()
    resume_status = ClaimStatus.NOT_APPLICABLE
    resume_detail = "no optimizer state supplied"
    resume_pass = True
    resume_contract: dict[str, Any] = {}
    if optimizer_supplied and resume_steps == 0:
        resume_detail = "no resumed optimizer steps requested"
    elif (
        source_optimizer_state is not None
        and target_optimizer_state is not None
        and resume_steps > 0
    ):
        resumed = verify_resumed_training(
            plan,
            source_capture,
            target_capture,
            source_state,
            target_values,
            source_optimizer_state,
            target_optimizer_state,
            steps=resume_steps,
            seed_base=resume_seed_base,
        )
        resumed_data = resumed.to_dict()
        resume_results = tuple(item.to_dict() for item in resumed.steps)
        resume_contract = resumed_data["numerical_contract"]
        resume_status = ClaimStatus.VERIFIED if resumed.passed else ClaimStatus.FAILED
        resume_detail = f"{resume_steps} synchronized seeded {resumed.optimizer_algorithm} steps"
        resume_pass = resumed.passed
        if not resumed.passed and first_divergence is None:
            for step in resumed.steps:
                candidates = (step.loss, *step.outputs, *step.model_state)
                failed = next((item for item in candidates if not item.passed), None)
                if failed is not None:
                    first_divergence = failed.comparison.path
                    break
    claims = (
        VerificationClaim(
            "PARAMETER_STATE_CONVERTED",
            ClaimStatus.VERIFIED,
            f"{len(plan.targets)} physical target tensors generated",
        ),
        VerificationClaim(
            "PARAMETER_STATE_COVERAGE_COMPLETE",
            ClaimStatus.VERIFIED,
            f"{len(plan.targets)} target tensors covered",
        ),
        VerificationClaim(
            "STATE_COVERAGE_COMPLETE",
            ClaimStatus.VERIFIED,
            f"{len(plan.targets)} target tensors covered",
        ),
        VerificationClaim(
            "STRUCTURAL_ALIGNMENT",
            ClaimStatus.VERIFIED,
            f"{len(source_canonical.slots)} semantic slots aligned",
        ),
        VerificationClaim(
            "FORWARD_VERIFIED",
            ClaimStatus.VERIFIED if all_forward else ClaimStatus.FAILED,
            f"{len(seeds)} deterministic probe seeds",
        ),
        VerificationClaim(
            "INTERMEDIATE_VERIFIED",
            ClaimStatus.VERIFIED if all_intermediate else ClaimStatus.FAILED,
            f"{len(source_canonical.anchors)} aligned semantic anchors",
        ),
        VerificationClaim(
            "PARAMETER_GRADIENT_VERIFIED",
            ClaimStatus.VERIFIED if all_gradients else ClaimStatus.FAILED,
            f"{len(gradients.comparisons)} physical gradients",
        ),
        VerificationClaim("INPUT_GRADIENT_VERIFIED", input_gradient_status, input_gradient_detail),
        VerificationClaim(
            "ROUNDTRIP_EXACT",
            ClaimStatus.VERIFIED if all_roundtrip else ClaimStatus.FAILED,
            f"{len(roundtrip)} source tensors",
        ),
        VerificationClaim(
            "BITWISE_FORWARD_EQUIVALENT",
            ClaimStatus.VERIFIED
            if all(
                comparison["max_absolute_error"] == 0.0 and comparison["mismatch_count"] == 0
                for item in forward_results
                for comparison in item["comparisons"]
            )
            else ClaimStatus.NOT_APPLICABLE,
            "reported only when observed",
        ),
        VerificationClaim(
            "OPTIMIZER_STATE_CONVERTED",
            optimizer_converted_status,
            optimizer_detail,
        ),
        VerificationClaim(
            "OPTIMIZER_COVERAGE_COMPLETE",
            optimizer_coverage_status,
            optimizer_detail,
        ),
        VerificationClaim(
            "OPTIMIZER_STATE_VERIFIED",
            optimizer_verification_status,
            "bitwise step/exp_avg/exp_avg_sq comparison"
            if optimizer_supplied
            else optimizer_detail,
        ),
        VerificationClaim(
            "RESUMED_TRAINING_EQUIVALENT",
            resume_status,
            resume_detail,
        ),
    )
    required_pass = (
        all_forward
        and all_intermediate
        and all_gradients
        and all_roundtrip
        and optimizer_pass
        and resume_pass
    )
    if input_gradient_status == ClaimStatus.FAILED:
        required_pass = False
    outcome = ClaimStatus.VERIFIED if required_pass else ClaimStatus.FAILED
    rewrite_rules = tuple(
        sorted(
            set(
                source_capture.artifact.graph.rewrite_rules_used
                + target_capture.artifact.graph.rewrite_rules_used
            )
        )
    )
    return VerificationCertificate(
        source_capture.adapter.adapter_id,
        target_capture.adapter.adapter_id,
        source_capture.artifact.graph_hash,
        target_capture.artifact.graph_hash,
        source_capture.artifact.state_schema.schema_hash,
        target_capture.artifact.state_schema.schema_hash,
        source_checkpoint_fingerprint,
        target_checkpoint_fingerprint,
        plan.plan_hash,
        plan.architecture_signature,
        plan.name_hint_mode,
        rewrite_rules,
        plan.assumptions,
        tuple(
            sorted(set(source_canonical.unsupported_regions + target_canonical.unsupported_regions))
        ),
        tuple(seeds),
        {
            "source": source_capture.artifact.example_input_signature.__dict__,
            "target": target_capture.artifact.example_input_signature.__dict__,
        },
        {"policy": "dtype-aware", "float32": {"absolute": 2e-5, "relative": 2e-4}},
        {"target_tensor_count": len(plan.targets), "covered": sorted(plan.targets)},
        tuple(forward_results),
        tuple(intermediate_results),
        tuple(
            [{"kind": "parameter", **item} for item in gradient_results]
            + [{"kind": "input", **item} for item in input_gradient_results]
        ),
        roundtrip_results,
        first_divergence,
        claims,
        outcome,
        generated_file_hashes or {},
        certificate_timestamp(),
        conversion_scope={
            "parameter_state": True,
            "optimizer_state": optimizer_supplied,
        },
        parameter_coverage={
            "complete": True,
            "target_tensor_count": len(plan.targets),
            "covered": sorted(plan.targets),
        },
        optimizer_coverage=optimizer_coverage,
        optimizer_associations=optimizer_associations,
        optimizer_state_results=optimizer_state_results,
        resume_results=resume_results,
        numeric_contract={
            "optimizer_tensor_conversion": {
                "applies": optimizer_supplied,
                "equality": "bitwise" if optimizer_supplied else None,
            },
            "resumed_training": resume_contract,
        },
    )
