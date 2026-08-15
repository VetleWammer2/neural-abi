"""Deterministic composition of source decoders and target encoders."""

from __future__ import annotations

from collections import defaultdict

from neuralabi import __version__
from neuralabi.export.artifact import ExportArtifact
from neuralabi.export.signature import compare_signatures
from neuralabi.formats.plan import (
    PLAN_SCHEMA_VERSION,
    AliasRecord,
    ConversionPlan,
    PlanEndpoint,
    PlannedTensor,
)
from neuralabi.ir.semantic import CanonicalModel, PhysicalView
from neuralabi.status import LinkStatus, PlanValidationError
from neuralabi.transforms import Alias, TensorSpec
from neuralabi.util.hashing import hash_canonical


def _view_map(model: CanonicalModel) -> dict[str, PhysicalView]:
    result: dict[str, PhysicalView] = {}
    for view in model.views():
        if view.semantic_id in result:
            raise PlanValidationError(
                f"semantic slot {view.semantic_id} has multiple physical views"
            )
        result[view.semantic_id] = view
    return result


def _state_specs(model: CanonicalModel) -> dict[str, TensorSpec]:
    return {
        tensor.key: TensorSpec(tensor.shape, tensor.dtype) for tensor in model.state_schema.tensors
    }


def _alias_peers(model: CanonicalModel) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in model.state_schema.tensors:
        if item.alias_group is not None:
            groups[item.alias_group].append(item.key)
    result: dict[str, tuple[str, ...]] = {}
    for keys in groups.values():
        peers = tuple(sorted(keys))
        for key in keys:
            result[key] = peers
    return result


def _compose_targets(
    producer: CanonicalModel,
    consumer: CanonicalModel,
) -> tuple[dict[str, PlannedTensor], tuple[AliasRecord, ...]]:
    producer_views = _view_map(producer)
    consumer_slots = consumer.slot_map()
    if set(producer_views) != set(consumer_slots):
        missing = sorted(set(consumer_slots) - set(producer_views))
        extra = sorted(set(producer_views) - set(consumer_slots))
        raise PlanValidationError(f"semantic slot mismatch: missing={missing}, extra={extra}")
    expressions = {semantic_id: view.decode for semantic_id, view in producer_views.items()}
    targets: dict[str, PlannedTensor] = {}
    for layout in sorted(consumer.layouts, key=lambda item: item.physical_key):
        expression = layout.encode(expressions)
        spec = layout.physical_spec()
        planned = PlannedTensor(expression, spec.shape, spec.dtype, layout.components)
        previous = targets.get(layout.physical_key)
        if previous is not None:
            if previous.expression.to_dict() != expression.to_dict() or previous.spec != spec:
                raise PlanValidationError(
                    f"tied physical target {layout.physical_key!r} has inconsistent semantic encodings"
                )
            planned = PlannedTensor(
                expression,
                spec.shape,
                spec.dtype,
                tuple(sorted(set(previous.semantic_slots + layout.components))),
            )
        targets[layout.physical_key] = planned
    aliases: list[AliasRecord] = []
    peers = _alias_peers(consumer)
    state = consumer.state_schema.by_key()
    for key in sorted(state):
        if key in targets:
            continue
        candidates = [peer for peer in peers.get(key, ()) if peer in targets]
        if not candidates:
            raise PlanValidationError(f"persistent target tensor {key!r} is unexplained")
        primary = candidates[0]
        primary_plan = targets[primary]
        item = state[key]
        if item.shape != primary_plan.shape or item.dtype != primary_plan.dtype:
            raise PlanValidationError(f"alias {key!r} metadata conflicts with {primary!r}")
        targets[key] = PlannedTensor(
            Alias(primary_plan.expression), item.shape, item.dtype, primary_plan.semantic_slots
        )
        aliases.append(AliasRecord(key, primary))
    return targets, tuple(aliases)


def synthesize_plan(
    source: CanonicalModel,
    target: CanonicalModel,
    source_artifact: ExportArtifact,
    target_artifact: ExportArtifact,
    *,
    checkpoint_fingerprint: str,
    name_hint_mode: str = "off",
    max_candidates: int = 128,
) -> ConversionPlan:
    if name_hint_mode not in {"off", "weak"}:
        raise PlanValidationError(f"unsupported name-hint mode {name_hint_mode!r}")
    if max_candidates < 1:
        raise PlanValidationError("max_candidates must be at least one")
    compare_signatures(source.architecture, target.architecture)
    targets, aliases = _compose_targets(source, target)
    inverse_targets, _ = _compose_targets(target, source)
    source_specs = _state_specs(source)
    target_specs = _state_specs(target)
    for key, planned in targets.items():
        inferred = planned.expression.infer_spec(source_specs)
        if inferred != planned.spec or inferred != target_specs[key]:
            raise PlanValidationError(f"target expression for {key!r} has inconsistent metadata")
    for key, planned in inverse_targets.items():
        if planned.expression.infer_spec(target_specs) != planned.spec:
            raise PlanValidationError(f"inverse expression for {key!r} has inconsistent metadata")
    consumed = {
        key for target_plan in targets.values() for key in target_plan.expression.source_keys()
    }
    source_peers = _alias_peers(source)
    unexplained_source = [
        key
        for key in source_specs
        if key not in consumed and not any(peer in consumed for peer in source_peers.get(key, ()))
    ]
    if unexplained_source:
        raise PlanValidationError(f"unconsumed source state: {sorted(unexplained_source)}")
    plan = ConversionPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        tool_version=__version__,
        source=PlanEndpoint(
            source.adapter_id,
            source_artifact.graph_hash,
            source.state_schema.schema_hash,
            checkpoint_fingerprint,
        ),
        target=PlanEndpoint(
            target.adapter_id,
            target_artifact.graph_hash,
            target.state_schema.schema_hash,
        ),
        architecture_signature=source.architecture.to_dict(),
        source_tensors=source_specs,
        target_tensors=target_specs,
        targets=targets,
        aliases=aliases,
        derived_state=(),
        assumptions=tuple(sorted(set(source.assumptions + target.assumptions))),
        name_hint_mode=name_hint_mode,
        name_hints_used=False,
        mapping_status=LinkStatus.UNIQUE,
        plan_complexity=sum(item.expression.cost() for item in targets.values()),
        synthesis_statistics={
            "candidate_count": 1,
            "candidate_budget": max_candidates,
            "semantic_slot_count": len(source.slots),
            "source_physical_tensor_count": len(source.layouts),
            "target_physical_tensor_count": len(target.layouts),
        },
        candidate_hashes=(),
        ambiguity=(),
        inverse_targets=inverse_targets,
        plan_hash="",
    ).with_hash()
    candidate_hash = hash_canonical(
        {key: target_tensor.expression.to_dict() for key, target_tensor in sorted(targets.items())}
    )
    return ConversionPlan(
        **{**plan.__dict__, "candidate_hashes": (candidate_hash,), "plan_hash": ""}
    ).with_hash()
