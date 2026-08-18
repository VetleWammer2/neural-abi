"""Static optimizer-state coverage, association, and tensor verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuralabi.apply.optimizer import convert_optimizer_bundle
from neuralabi.formats.optimizer import load_optimizer_bundle, tensors_bitwise_equal
from neuralabi.formats.plan import ConversionPlan
from neuralabi.status import CheckpointError
from neuralabi.verify.forward import NumericTolerance, TensorComparison, compare_tensor


@dataclass(frozen=True)
class OptimizerTensorComparison:
    comparison: TensorComparison
    exact: bool

    @property
    def passed(self) -> bool:
        return self.comparison.passed and self.exact

    def to_dict(self) -> dict[str, Any]:
        return {**self.comparison.to_dict(), "exact": self.exact}


@dataclass(frozen=True)
class OptimizerStateVerification:
    algorithm: str
    source_bundle_hash: str
    target_bundle_hash: str
    source_parameter_count: int
    target_parameter_count: int
    expected_source_parameter_count: int
    expected_target_parameter_count: int
    associations: tuple[dict[str, Any], ...]
    comparisons: tuple[OptimizerTensorComparison, ...]

    @property
    def coverage_complete(self) -> bool:
        return (
            self.expected_source_parameter_count > 0
            and self.expected_target_parameter_count > 0
            and self.source_parameter_count == self.expected_source_parameter_count
            and self.target_parameter_count == self.expected_target_parameter_count
        )

    @property
    def passed(self) -> bool:
        return self.coverage_complete and all(item.passed for item in self.comparisons)

    def coverage_dict(self) -> dict[str, Any]:
        return {
            "complete": self.coverage_complete,
            "algorithm": self.algorithm,
            "source_bundle_hash": self.source_bundle_hash,
            "target_bundle_hash": self.target_bundle_hash,
            "source_unique_parameter_count": self.source_parameter_count,
            "target_unique_parameter_count": self.target_parameter_count,
            "expected_source_unique_parameter_count": self.expected_source_parameter_count,
            "expected_target_unique_parameter_count": self.expected_target_parameter_count,
            "state_fields": ["step", "exp_avg", "exp_avg_sq"],
        }


def verify_optimizer_state(
    plan: ConversionPlan,
    source_path: Path,
    target_path: Path,
    *,
    target_checkpoint_fingerprint: str,
) -> OptimizerStateVerification:
    """Recompute target Adam state and compare every field with exact tolerance."""

    source = load_optimizer_bundle(source_path)
    target = load_optimizer_bundle(target_path)
    expected_bundle, expected_tensors = convert_optimizer_bundle(
        plan,
        source,
        target_checkpoint_fingerprint=target_checkpoint_fingerprint,
    )
    actual_bundle = target.bundle
    if actual_bundle.algorithm != expected_bundle.algorithm:
        raise CheckpointError(
            f"target optimizer algorithm {actual_bundle.algorithm!r} != "
            f"{expected_bundle.algorithm!r}"
        )
    if actual_bundle.model_binding != expected_bundle.model_binding:
        raise CheckpointError("target optimizer model binding is incorrect")
    if actual_bundle.parameter_groups != expected_bundle.parameter_groups:
        raise CheckpointError("target optimizer parameter groups or options are incorrect")
    if actual_bundle.parameters != expected_bundle.parameters:
        raise CheckpointError("target optimizer semantic parameter associations are incorrect")

    exact_tolerance = NumericTolerance(0.0, 0.0)
    comparisons: list[OptimizerTensorComparison] = []
    for parameter in actual_bundle.parameters:
        actual = target.read_state(parameter.parameter_id)
        for field, tensor_key in (
            ("step", parameter.step),
            ("exp_avg", parameter.exp_avg),
            ("exp_avg_sq", parameter.exp_avg_sq),
        ):
            expected = expected_tensors[tensor_key]
            observed = actual[field]
            comparison = compare_tensor(
                expected,
                observed,
                path=f"optimizer_state.{parameter.parameter_id}.{field}",
                tolerance=exact_tolerance,
            )
            comparisons.append(
                OptimizerTensorComparison(
                    comparison,
                    tensors_bitwise_equal(expected, observed),
                )
            )

    mapping = plan.optimizer_mapping
    if mapping is None:  # convert_optimizer_bundle already rejects this; keep typing total.
        raise CheckpointError("optimizer verification requires a plan optimizer mapping")
    associations = tuple(
        {
            "target_parameter_id": item.identity,
            "target_model_keys": list(item.model_keys),
            "expression_target": item.expression_target,
            "source_parameter_ids": list(item.source_identities),
            "semantic_slots": list(item.semantic_slots),
        }
        for item in mapping.target_parameter_identities
    )
    return OptimizerStateVerification(
        algorithm=actual_bundle.algorithm,
        source_bundle_hash=source.bundle.bundle_hash,
        target_bundle_hash=actual_bundle.bundle_hash,
        source_parameter_count=len(source.bundle.parameters),
        target_parameter_count=len(actual_bundle.parameters),
        expected_source_parameter_count=len(mapping.source_parameter_identities),
        expected_target_parameter_count=len(mapping.target_parameter_identities),
        associations=associations,
        comparisons=tuple(comparisons),
    )
