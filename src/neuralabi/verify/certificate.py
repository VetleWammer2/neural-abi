"""End-to-end verification orchestration shared by ``verify`` and ``link``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from neuralabi.status import ClaimStatus
from neuralabi.verify.forward import compare_pytrees
from neuralabi.verify.gradients import verify_input_gradients, verify_parameter_gradients
from neuralabi.verify.intermediates import compare_anchors, record_graph_values
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
) -> VerificationCertificate:
    validate_plan(plan)
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
    claims = (
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
    )
    required_pass = all_forward and all_intermediate and all_gradients and all_roundtrip
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
    )
