"""Apply a model conversion plan to safe Adam/AdamW optimizer state."""

from __future__ import annotations

from pathlib import Path

import torch

from neuralabi.checkpoints.store import TensorStore, open_tensor_store
from neuralabi.checkpoints.writer import DEFAULT_MAX_SHARD_SIZE
from neuralabi.formats.optimizer import (
    OPTIMIZER_FORMAT,
    OPTIMIZER_SCHEMA_VERSION,
    LoadedOptimizerBundle,
    OptimizerBundle,
    OptimizerModelBinding,
    OptimizerParameterGroup,
    OptimizerParameterState,
    OptimizerStateResult,
    load_optimizer_bundle,
    tensors_bitwise_equal,
    write_optimizer_bundle,
)
from neuralabi.formats.plan import ConversionPlan, OptimizerMapping, validate_plan
from neuralabi.status import CheckpointError, PlanValidationError
from neuralabi.transforms.execute import execute_checked
from neuralabi.util.paths import ensure_distinct_paths


def _validate_target_checkpoint(plan: ConversionPlan, store: TensorStore) -> str:
    actual = set(store.keys())
    expected = set(plan.target_tensors)
    if actual != expected:
        raise CheckpointError(
            "target model checkpoint keys disagree with plan: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    for key, spec in sorted(plan.target_tensors.items()):
        metadata = store.metadata(key)
        if metadata.shape != spec.shape or metadata.dtype != spec.dtype:
            raise CheckpointError(
                f"target model tensor {key!r} metadata "
                f"{(metadata.shape, metadata.dtype)} != {(spec.shape, spec.dtype)}"
            )
    return store.fingerprint()


def _required_mapping(plan: ConversionPlan) -> OptimizerMapping:
    validate_plan(plan)
    if plan.schema_version != 2 or plan.optimizer_mapping is None:
        raise PlanValidationError("optimizer conversion requires a schema-version-2 plan")
    if plan.inverse_targets is None:
        raise PlanValidationError("optimizer conversion requires the plan's exact inverse")
    return plan.optimizer_mapping


def _validate_source_binding(
    plan: ConversionPlan,
    mapping: OptimizerMapping,
    source: LoadedOptimizerBundle,
) -> None:
    bundle = source.bundle
    if bundle.model_binding.state_schema_hash != plan.source.state_schema_hash:
        raise CheckpointError("optimizer state does not match the plan's source state schema")
    if (
        plan.source.checkpoint_fingerprint is None
        or bundle.model_binding.checkpoint_fingerprint != plan.source.checkpoint_fingerprint
    ):
        raise CheckpointError("optimizer state does not match the plan's source checkpoint")
    expected_identities = {
        record.identity: record.model_keys for record in mapping.source_parameter_identities
    }
    actual_identities = {record.parameter_id: record.model_keys for record in bundle.parameters}
    if actual_identities != expected_identities:
        raise CheckpointError(
            "optimizer parameter identities do not match the plan's source identities"
        )
    for item in bundle.parameters:
        specs = {plan.source_tensors[key] for key in item.model_keys}
        if specs != {item.spec}:
            raise CheckpointError(
                f"optimizer parameter {item.parameter_id!r} metadata disagrees with the plan"
            )


def _group_by_parameter(bundle: OptimizerBundle) -> dict[str, int]:
    result: dict[str, int] = {}
    for group_index, group in enumerate(bundle.parameter_groups):
        for parameter_id in group.parameters:
            result[parameter_id] = group_index
    return result


def _steps_equal(values: tuple[torch.Tensor, ...]) -> bool:
    if not values:
        return False
    first = values[0]
    return all(tensors_bitwise_equal(value, first) for value in values[1:])


def convert_optimizer_bundle(
    plan: ConversionPlan,
    source: LoadedOptimizerBundle,
    *,
    target_checkpoint_fingerprint: str,
) -> tuple[OptimizerBundle, dict[str, torch.Tensor]]:
    """Convert a validated bundle in memory using the parameter expressions in ``plan``."""

    mapping = _required_mapping(plan)
    _validate_source_binding(plan, mapping, source)
    source_by_id = source.bundle.by_id()
    source_id_by_key = {
        key: record.identity
        for record in mapping.source_parameter_identities
        for key in record.model_keys
    }
    source_group = _group_by_parameter(source.bundle)
    state_cache: dict[str, dict[str, torch.Tensor]] = {}

    def state(parameter_id: str) -> dict[str, torch.Tensor]:
        if parameter_id not in state_cache:
            state_cache[parameter_id] = source.read_state(parameter_id)
        return state_cache[parameter_id]

    tensors: dict[str, torch.Tensor] = {}
    parameters: list[OptimizerParameterState] = []
    target_group: dict[str, int] = {}
    for index, target_identity in enumerate(mapping.target_parameter_identities):
        dependencies = target_identity.source_identities
        if not dependencies or any(item not in source_by_id for item in dependencies):
            raise CheckpointError(
                f"target optimizer parameter {target_identity.identity!r} has unknown dependencies"
            )
        dependency_groups = {source_group[item] for item in dependencies}
        if len(dependency_groups) != 1:
            raise CheckpointError(
                f"cannot fuse {target_identity.identity!r} across optimizer parameter groups"
            )
        target_group[target_identity.identity] = next(iter(dependency_groups))
        steps = tuple(state(item)["step"] for item in dependencies)
        if not _steps_equal(steps):
            raise CheckpointError(
                f"cannot fuse {target_identity.identity!r} with unequal Adam steps"
            )
        try:
            planned = plan.targets[target_identity.expression_target]
        except KeyError as exc:
            raise PlanValidationError(
                f"optimizer mapping references missing target expression "
                f"{target_identity.expression_target!r}"
            ) from exc
        expression_source_keys = set(planned.expression.source_keys())
        expression_identities = {source_id_by_key[key] for key in expression_source_keys}
        if expression_identities != set(dependencies):
            raise PlanValidationError(
                f"optimizer dependencies for {target_identity.identity!r} disagree with its expression"
            )
        target_spec = plan.target_tensors[target_identity.expression_target]
        stem = f"state.{index:06d}"
        step_key = f"{stem}.step"
        exp_avg_key = f"{stem}.exp_avg"
        exp_avg_sq_key = f"{stem}.exp_avg_sq"
        tensors[step_key] = steps[0].detach().cpu().clone()
        for field, output_key in (
            ("exp_avg", exp_avg_key),
            ("exp_avg_sq", exp_avg_sq_key),
        ):
            sources = {key: state(source_id_by_key[key])[field] for key in expression_source_keys}
            tensors[output_key] = (
                execute_checked(planned.expression, sources, expected=target_spec)
                .detach()
                .cpu()
                .contiguous()
                .clone()
            )
        parameters.append(
            OptimizerParameterState(
                target_identity.identity,
                target_identity.model_keys,
                target_spec.shape,
                target_spec.dtype,
                True,
                step_key,
                exp_avg_key,
                exp_avg_sq_key,
            )
        )
    groups: list[OptimizerParameterGroup] = []
    for index, source_group_record in enumerate(source.bundle.parameter_groups):
        target_parameters = tuple(
            item.parameter_id for item in parameters if target_group[item.parameter_id] == index
        )
        groups.append(
            OptimizerParameterGroup(target_parameters, source_group_record.hyperparameters)
        )
    result = OptimizerBundle(
        schema_version=OPTIMIZER_SCHEMA_VERSION,
        format=OPTIMIZER_FORMAT,
        algorithm=source.bundle.algorithm,
        model_binding=OptimizerModelBinding(
            plan.target.state_schema_hash, target_checkpoint_fingerprint
        ),
        parameter_groups=tuple(groups),
        parameters=tuple(parameters),
        tensor_fingerprint="pending",
        bundle_hash="pending",
    )
    return result, tensors


def apply_optimizer_checkpoint(
    plan: ConversionPlan,
    source: Path,
    target_model_checkpoint: Path,
    output: Path,
    *,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> OptimizerStateResult:
    """Convert a safe optimizer bundle and bind it to a converted model checkpoint."""

    ensure_distinct_paths(source, output)
    ensure_distinct_paths(target_model_checkpoint, output)
    loaded = load_optimizer_bundle(source)
    target_store = open_tensor_store(target_model_checkpoint)
    target_fingerprint = _validate_target_checkpoint(plan, target_store)
    bundle, tensors = convert_optimizer_bundle(
        plan, loaded, target_checkpoint_fingerprint=target_fingerprint
    )
    return write_optimizer_bundle(bundle, tensors, output, max_shard_size=max_shard_size)
