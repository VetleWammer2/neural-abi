from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from safetensors.torch import save_file

from examples.twin_mlp.models import source_adapter, target_adapter
from neuralabi.apply.engine import apply_to_state
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import capture_adapter
from neuralabi.formats.plan import (
    AliasRecord,
    ConversionPlan,
    plan_from_dict,
    validate_plan,
)
from neuralabi.ir.state import StateSchema, StateTensor, parameter_identity_groups
from neuralabi.recognize.twin_mlp import recognize_twin_mlp
from neuralabi.status import PlanValidationError
from neuralabi.synth.solver import synthesize_plan
from neuralabi.transforms import expr_from_dict
from neuralabi.util.hashing import hash_canonical


def _plan(tmp_path: Path) -> ConversionPlan:
    source = capture_adapter(source_adapter)
    target = capture_adapter(target_adapter)
    checkpoint = tmp_path / "source.safetensors"
    save_file(
        {key: value.detach().clone() for key, value in source.model.state_dict().items()},
        checkpoint,
    )
    return synthesize_plan(
        recognize_twin_mlp(source.artifact),
        recognize_twin_mlp(target.artifact),
        source.artifact,
        target.artifact,
        checkpoint_fingerprint=open_tensor_store(checkpoint).fingerprint(),
    )


def test_unknown_transform_is_rejected() -> None:
    with pytest.raises(PlanValidationError, match="unknown transform"):
        expr_from_dict({"op": "load_python", "module": "os"})


def test_alias_dependency_cycle_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    first, second = sorted(plan.targets)
    cyclic = replace(
        plan,
        aliases=(AliasRecord(first, second), AliasRecord(second, first)),
        plan_hash="",
    ).with_hash()
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_plan(cyclic)


def test_uncovered_target_tensor_is_rejected_without_adapters(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    targets = dict(plan.targets)
    targets.pop(sorted(targets)[0])
    uncovered = replace(plan, targets=targets, plan_hash="").with_hash()
    with pytest.raises(PlanValidationError, match="coverage"):
        validate_plan(uncovered)


def test_schema_v2_optimizer_mapping_references_model_expressions(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan = plan_from_dict(plan.to_dict())
    mapping = plan.optimizer_mapping
    assert plan.schema_version == 2
    assert mapping is not None
    assert (
        tuple(
            sorted(
                key for record in mapping.source_parameter_identities for key in record.model_keys
            )
        )
        == mapping.source_parameter_keys
    )
    assert (
        tuple(
            sorted(
                key for record in mapping.target_parameter_identities for key in record.model_keys
            )
        )
        == mapping.target_parameter_keys
    )
    assert all(
        record.expression_target in plan.targets for record in mapping.target_parameter_identities
    )
    serialized = mapping.to_dict()
    assert "expression" not in serialized
    assert serialized["tensor_fields"] == ["exp_avg", "exp_avg_sq"]


def test_schema_v1_plan_hash_load_and_parameter_execution_remain_compatible(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    raw = plan.to_dict(include_hash=False)
    raw["schema_version"] = 1
    raw.pop("optimizer_mapping")
    raw["plan_hash"] = hash_canonical(raw)
    loaded = plan_from_dict(raw)
    assert loaded.schema_version == 1
    assert loaded.optimizer_mapping is None
    assert loaded.to_dict() == raw
    model = source_adapter.build(device="cpu", dtype=__import__("torch").float32)
    converted = apply_to_state(loaded, model.state_dict())
    assert set(converted) == set(loaded.target_tensors)


def test_optimizer_mapping_requires_inverse_and_complete_source_partition(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    without_inverse = replace(plan, inverse_targets=None, plan_hash="").with_hash()
    with pytest.raises(PlanValidationError, match="requires an exact inverse"):
        validate_plan(without_inverse)

    mapping = plan.optimizer_mapping
    assert mapping is not None
    incomplete_mapping = replace(
        mapping, source_parameter_identities=mapping.source_parameter_identities[:-1]
    )
    incomplete = replace(plan, optimizer_mapping=incomplete_mapping, plan_hash="").with_hash()
    with pytest.raises(PlanValidationError, match="partition"):
        validate_plan(incomplete)


def test_optimizer_mapping_rejects_dependency_and_alias_mutations(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    mapping = plan.optimizer_mapping
    assert mapping is not None
    first_record = mapping.target_parameter_identities[0]
    mutated_record = replace(first_record, source_identities=())
    mutated_mapping = replace(
        mapping,
        target_parameter_identities=(
            mutated_record,
            *mapping.target_parameter_identities[1:],
        ),
    )
    dependency_mutation = replace(plan, optimizer_mapping=mutated_mapping, plan_hash="").with_hash()
    with pytest.raises(PlanValidationError, match="dependencies disagree"):
        validate_plan(dependency_mutation)

    first, second = sorted(plan.targets)
    alias_mutation = replace(
        plan,
        aliases=(AliasRecord(first, second),),
        plan_hash="",
    ).with_hash()
    with pytest.raises(PlanValidationError, match="aliases"):
        validate_plan(alias_mutation)


def test_parameter_identities_exclude_frozen_parameters_and_buffers() -> None:
    tensors = (
        StateTensor("alias.train", "parameter", (2,), "float32", True, True, "a", 2),
        StateTensor("train", "parameter", (2,), "float32", True, True, "a", 2),
        StateTensor("frozen", "parameter", (2,), "float32", False, True, None, 2),
        StateTensor("buffer", "buffer", (2,), "float32", False, True, "a", 2),
    )
    assert parameter_identity_groups(StateSchema(tensors, "schema")) == (("alias.train", "train"),)
