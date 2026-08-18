"""Versioned, validated conversion-plan format."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from neuralabi import __version__
from neuralabi.status import LinkStatus, PlanValidationError
from neuralabi.transforms import TensorSpec, TransformExpr, expr_from_dict
from neuralabi.util.canonical_json import load_json, pretty_dumps
from neuralabi.util.hashing import hash_canonical

PLAN_SCHEMA_VERSION = 2
SUPPORTED_PLAN_SCHEMA_VERSIONS = frozenset({1, PLAN_SCHEMA_VERSION})
MAX_TARGETS = 1_000_000

OPTIMIZER_MAPPING_SEMANTICS = "coordinate-reindex-v1"
OPTIMIZER_TENSOR_FIELDS = ("exp_avg", "exp_avg_sq")
OPTIMIZER_SCALAR_FIELDS = ("step",)
OPTIMIZER_FUSION_STEP_POLICY = "require_equal"
OPTIMIZER_PARAMETER_GROUP_POLICY = "require_same_group"
COORDINATE_REINDEX_OPS = frozenset(
    {
        "source",
        "identity",
        "alias",
        "reshape",
        "permute",
        "transpose",
        "slice",
        "concat",
        "stack",
        "squeeze",
        "unsqueeze",
        "interleave",
        "deinterleave",
    }
)


@dataclass(frozen=True)
class PlanEndpoint:
    adapter_id: str
    graph_hash: str
    state_schema_hash: str
    checkpoint_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "adapter_id": self.adapter_id,
            "graph_hash": self.graph_hash,
            "state_schema_hash": self.state_schema_hash,
        }
        if self.checkpoint_fingerprint is not None:
            result["checkpoint_fingerprint"] = self.checkpoint_fingerprint
        return result


@dataclass(frozen=True)
class PlannedTensor:
    expression: TransformExpr
    shape: tuple[int, ...]
    dtype: str
    semantic_slots: tuple[str, ...]

    @property
    def spec(self) -> TensorSpec:
        return TensorSpec(self.shape, self.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression.to_dict(),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "semantic_slots": list(self.semantic_slots),
        }


@dataclass(frozen=True)
class AliasRecord:
    key: str
    target_of: str

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "target_of": self.target_of}


@dataclass(frozen=True)
class ParameterIdentityRecord:
    identity: str
    model_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.identity, "model_keys": list(self.model_keys)}


@dataclass(frozen=True)
class TargetParameterIdentityRecord:
    identity: str
    model_keys: tuple[str, ...]
    expression_target: str
    source_identities: tuple[str, ...]
    semantic_slots: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identity,
            "model_keys": list(self.model_keys),
            "expression_target": self.expression_target,
            "source_identities": list(self.source_identities),
            "semantic_slots": list(self.semantic_slots),
        }


@dataclass(frozen=True)
class OptimizerMapping:
    semantics: str
    tensor_fields: tuple[str, ...]
    scalar_fields: tuple[str, ...]
    fusion_step_policy: str
    parameter_group_policy: str
    source_parameter_keys: tuple[str, ...]
    target_parameter_keys: tuple[str, ...]
    source_parameter_identities: tuple[ParameterIdentityRecord, ...]
    target_parameter_identities: tuple[TargetParameterIdentityRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantics": self.semantics,
            "tensor_fields": list(self.tensor_fields),
            "scalar_fields": list(self.scalar_fields),
            "fusion_step_policy": self.fusion_step_policy,
            "parameter_group_policy": self.parameter_group_policy,
            "source_parameter_keys": list(self.source_parameter_keys),
            "target_parameter_keys": list(self.target_parameter_keys),
            "source_parameter_identities": [
                record.to_dict() for record in self.source_parameter_identities
            ],
            "target_parameter_identities": [
                record.to_dict() for record in self.target_parameter_identities
            ],
        }


@dataclass(frozen=True)
class ConversionPlan:
    schema_version: int
    tool_version: str
    source: PlanEndpoint
    target: PlanEndpoint
    architecture_signature: dict[str, Any]
    source_tensors: dict[str, TensorSpec]
    target_tensors: dict[str, TensorSpec]
    targets: dict[str, PlannedTensor]
    aliases: tuple[AliasRecord, ...]
    derived_state: tuple[str, ...]
    assumptions: tuple[str, ...]
    name_hint_mode: str
    name_hints_used: bool
    mapping_status: LinkStatus
    plan_complexity: int
    synthesis_statistics: dict[str, int | float | str]
    candidate_hashes: tuple[str, ...]
    ambiguity: tuple[str, ...]
    inverse_targets: dict[str, PlannedTensor] | None
    plan_hash: str
    optimizer_mapping: OptimizerMapping | None = None

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "tool_version": self.tool_version,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "architecture_signature": self.architecture_signature,
            "source_tensors": {
                key: {"shape": list(spec.shape), "dtype": spec.dtype}
                for key, spec in sorted(self.source_tensors.items())
            },
            "target_tensors": {
                key: {"shape": list(spec.shape), "dtype": spec.dtype}
                for key, spec in sorted(self.target_tensors.items())
            },
            "targets": {key: value.to_dict() for key, value in sorted(self.targets.items())},
            "aliases": [alias.to_dict() for alias in self.aliases],
            "derived_state": list(self.derived_state),
            "assumptions": list(self.assumptions),
            "name_hint_mode": self.name_hint_mode,
            "name_hints_used": self.name_hints_used,
            "mapping_status": self.mapping_status.value,
            "plan_complexity": self.plan_complexity,
            "synthesis_statistics": dict(sorted(self.synthesis_statistics.items())),
            "candidate_hashes": list(self.candidate_hashes),
            "ambiguity": list(self.ambiguity),
            "inverse_targets": None
            if self.inverse_targets is None
            else {key: value.to_dict() for key, value in sorted(self.inverse_targets.items())},
        }
        if self.schema_version == 2:
            result["optimizer_mapping"] = (
                None if self.optimizer_mapping is None else self.optimizer_mapping.to_dict()
            )
        if include_hash:
            result["plan_hash"] = self.plan_hash
        return result

    def with_hash(self) -> ConversionPlan:
        return replace(self, plan_hash=hash_canonical(self.to_dict(include_hash=False)))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pretty_dumps(self.to_dict()), encoding="utf-8", newline="\n")


def _required_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanValidationError(f"{label} must be an object")
    return value


def _endpoint(value: Any, label: str) -> PlanEndpoint:
    data = _required_dict(value, label)
    allowed = {"adapter_id", "graph_hash", "state_schema_hash", "checkpoint_fingerprint"}
    if set(data) - allowed:
        raise PlanValidationError(f"unknown {label} endpoint fields: {sorted(set(data) - allowed)}")
    required = ("adapter_id", "graph_hash", "state_schema_hash")
    if any(not isinstance(data.get(field), str) or not data[field] for field in required):
        raise PlanValidationError(f"{label} endpoint has missing string fields")
    fingerprint = data.get("checkpoint_fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise PlanValidationError(f"{label} fingerprint must be a string")
    return PlanEndpoint(
        data["adapter_id"], data["graph_hash"], data["state_schema_hash"], fingerprint
    )


def _tensor_spec(value: Any, label: str) -> TensorSpec:
    data = _required_dict(value, label)
    if (
        set(data) != {"shape", "dtype"}
        or not isinstance(data["shape"], list)
        or not isinstance(data["dtype"], str)
    ):
        raise PlanValidationError(f"{label} must contain only shape and dtype")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in data["shape"]):
        raise PlanValidationError(f"{label} shape must contain integers")
    return TensorSpec(tuple(data["shape"]), data["dtype"])


def _planned_tensor(value: Any, label: str) -> PlannedTensor:
    data = _required_dict(value, label)
    if set(data) != {"expression", "shape", "dtype", "semantic_slots"}:
        raise PlanValidationError(f"{label} has unknown or missing fields")
    spec = _tensor_spec({"shape": data["shape"], "dtype": data["dtype"]}, label)
    slots = data["semantic_slots"]
    if not isinstance(slots, list) or any(not isinstance(item, str) for item in slots):
        raise PlanValidationError(f"{label} semantic_slots must be a string array")
    return PlannedTensor(expr_from_dict(data["expression"]), spec.shape, spec.dtype, tuple(slots))


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PlanValidationError(f"{label} must be a string array")
    return tuple(value)


def _sorted_unique_string_tuple(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    result = _string_tuple(value, label)
    if (not result and not allow_empty) or any(not item or "\x00" in item for item in result):
        raise PlanValidationError(f"{label} must contain non-empty valid strings")
    if len(set(result)) != len(result) or result != tuple(sorted(result)):
        raise PlanValidationError(f"{label} must be sorted and unique")
    return result


def _parameter_identity(value: Any, label: str) -> ParameterIdentityRecord:
    data = _required_dict(value, label)
    if set(data) != {"id", "model_keys"}:
        raise PlanValidationError(f"{label} has unknown or missing fields")
    identity = data["id"]
    if not isinstance(identity, str) or not identity or "\x00" in identity:
        raise PlanValidationError(f"{label}.id must be a non-empty valid string")
    return ParameterIdentityRecord(
        identity,
        _sorted_unique_string_tuple(data["model_keys"], f"{label}.model_keys"),
    )


def _target_parameter_identity(value: Any, label: str) -> TargetParameterIdentityRecord:
    data = _required_dict(value, label)
    required = {
        "id",
        "model_keys",
        "expression_target",
        "source_identities",
        "semantic_slots",
    }
    if set(data) != required:
        raise PlanValidationError(f"{label} has unknown or missing fields")
    identity = data["id"]
    expression_target = data["expression_target"]
    if not isinstance(identity, str) or not identity or "\x00" in identity:
        raise PlanValidationError(f"{label}.id must be a non-empty valid string")
    if (
        not isinstance(expression_target, str)
        or not expression_target
        or "\x00" in expression_target
    ):
        raise PlanValidationError(f"{label}.expression_target must be a non-empty valid string")
    return TargetParameterIdentityRecord(
        identity,
        _sorted_unique_string_tuple(data["model_keys"], f"{label}.model_keys"),
        expression_target,
        _sorted_unique_string_tuple(data["source_identities"], f"{label}.source_identities"),
        _sorted_unique_string_tuple(data["semantic_slots"], f"{label}.semantic_slots"),
    )


def _optimizer_mapping(value: Any) -> OptimizerMapping:
    data = _required_dict(value, "optimizer_mapping")
    required = {
        "semantics",
        "tensor_fields",
        "scalar_fields",
        "fusion_step_policy",
        "parameter_group_policy",
        "source_parameter_keys",
        "target_parameter_keys",
        "source_parameter_identities",
        "target_parameter_identities",
    }
    if set(data) != required:
        raise PlanValidationError(
            "optimizer_mapping fields differ from schema: "
            f"missing={sorted(required - set(data))}, "
            f"unknown={sorted(set(data) - required)}"
        )
    scalar_values = ("semantics", "fusion_step_policy", "parameter_group_policy")
    if any(not isinstance(data[field], str) for field in scalar_values):
        raise PlanValidationError("optimizer_mapping policies must be strings")
    source_raw = data["source_parameter_identities"]
    target_raw = data["target_parameter_identities"]
    if (
        not isinstance(source_raw, list)
        or not isinstance(target_raw, list)
        or len(source_raw) > MAX_TARGETS
        or len(target_raw) > MAX_TARGETS
    ):
        raise PlanValidationError("optimizer parameter identities must be bounded arrays")
    source = tuple(
        _parameter_identity(item, f"optimizer_mapping.source_parameter_identities[{index}]")
        for index, item in enumerate(source_raw)
    )
    target = tuple(
        _target_parameter_identity(item, f"optimizer_mapping.target_parameter_identities[{index}]")
        for index, item in enumerate(target_raw)
    )
    return OptimizerMapping(
        data["semantics"],
        _sorted_unique_string_tuple(data["tensor_fields"], "optimizer_mapping.tensor_fields"),
        _sorted_unique_string_tuple(data["scalar_fields"], "optimizer_mapping.scalar_fields"),
        data["fusion_step_policy"],
        data["parameter_group_policy"],
        _sorted_unique_string_tuple(
            data["source_parameter_keys"], "optimizer_mapping.source_parameter_keys"
        ),
        _sorted_unique_string_tuple(
            data["target_parameter_keys"], "optimizer_mapping.target_parameter_keys"
        ),
        source,
        target,
    )


def plan_from_dict(value: Any) -> ConversionPlan:
    data = _required_dict(value, "plan")
    schema_version = data.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in SUPPORTED_PLAN_SCHEMA_VERSIONS
    ):
        raise PlanValidationError(f"unsupported plan schema version {schema_version!r}")
    required = {
        "schema_version",
        "tool_version",
        "source",
        "target",
        "architecture_signature",
        "source_tensors",
        "target_tensors",
        "targets",
        "aliases",
        "derived_state",
        "assumptions",
        "name_hint_mode",
        "name_hints_used",
        "mapping_status",
        "plan_complexity",
        "synthesis_statistics",
        "candidate_hashes",
        "ambiguity",
        "inverse_targets",
        "plan_hash",
    }
    if schema_version == 2:
        required.add("optimizer_mapping")
    if set(data) != required:
        raise PlanValidationError(
            f"plan fields differ from schema: missing={sorted(required - set(data))}, unknown={sorted(set(data) - required)}"
        )
    if not isinstance(data["tool_version"], str):
        raise PlanValidationError("tool_version must be a string")
    source_raw = _required_dict(data["source_tensors"], "source_tensors")
    target_specs_raw = _required_dict(data["target_tensors"], "target_tensors")
    target_raw = _required_dict(data["targets"], "targets")
    if (
        len(source_raw) > MAX_TARGETS
        or len(target_specs_raw) > MAX_TARGETS
        or len(target_raw) > MAX_TARGETS
    ):
        raise PlanValidationError("plan tensor count exceeds safety limit")
    source_tensors = {
        str(key): _tensor_spec(item, f"source_tensors.{key}") for key, item in source_raw.items()
    }
    target_tensors = {
        str(key): _tensor_spec(item, f"target_tensors.{key}")
        for key, item in target_specs_raw.items()
    }
    targets = {
        str(key): _planned_tensor(item, f"targets.{key}") for key, item in target_raw.items()
    }
    aliases_raw = data["aliases"]
    if not isinstance(aliases_raw, list):
        raise PlanValidationError("aliases must be an array")
    aliases: list[AliasRecord] = []
    for item in aliases_raw:
        record = _required_dict(item, "alias")
        if set(record) != {"key", "target_of"} or not all(
            isinstance(record[field], str) for field in record
        ):
            raise PlanValidationError("alias must contain string key and target_of")
        aliases.append(AliasRecord(record["key"], record["target_of"]))
    inverse_raw = data["inverse_targets"]
    inverse = (
        None
        if inverse_raw is None
        else {
            str(key): _planned_tensor(item, f"inverse_targets.{key}")
            for key, item in _required_dict(inverse_raw, "inverse_targets").items()
        }
    )
    try:
        status = LinkStatus(data["mapping_status"])
    except (TypeError, ValueError) as exc:
        raise PlanValidationError("invalid mapping_status") from exc
    if data["name_hint_mode"] not in {"off", "weak"} or not isinstance(
        data["name_hints_used"], bool
    ):
        raise PlanValidationError("invalid name-hint fields")
    if isinstance(data["plan_complexity"], bool) or not isinstance(data["plan_complexity"], int):
        raise PlanValidationError("plan_complexity must be an integer")
    plan = ConversionPlan(
        schema_version,
        data["tool_version"],
        _endpoint(data["source"], "source"),
        _endpoint(data["target"], "target"),
        _required_dict(data["architecture_signature"], "architecture_signature"),
        source_tensors,
        target_tensors,
        targets,
        tuple(aliases),
        _string_tuple(data["derived_state"], "derived_state"),
        _string_tuple(data["assumptions"], "assumptions"),
        data["name_hint_mode"],
        data["name_hints_used"],
        status,
        data["plan_complexity"],
        _required_dict(data["synthesis_statistics"], "synthesis_statistics"),
        _string_tuple(data["candidate_hashes"], "candidate_hashes"),
        _string_tuple(data["ambiguity"], "ambiguity"),
        inverse,
        data["plan_hash"] if isinstance(data["plan_hash"], str) else "",
        None if schema_version == 1 else _optimizer_mapping(data["optimizer_mapping"]),
    )
    validate_plan(plan)
    return plan


def _validate_coordinate_expression(expression: TransformExpr, label: str) -> None:
    def visit(node: dict[str, Any]) -> None:
        op = node.get("op")
        if op not in COORDINATE_REINDEX_OPS:
            raise PlanValidationError(
                f"{label} uses {op!r}, which is not a coordinate-reindex operation"
            )
        child = node.get("input")
        if isinstance(child, dict):
            visit(child)
        children = node.get("inputs")
        if isinstance(children, list):
            for item in children:
                if isinstance(item, dict):
                    visit(item)

    visit(expression.to_dict())


def _validate_identity_partition(
    records: tuple[ParameterIdentityRecord, ...] | tuple[TargetParameterIdentityRecord, ...],
    parameter_keys: tuple[str, ...],
    tensor_specs: dict[str, TensorSpec],
    *,
    label: str,
) -> tuple[dict[str, str], dict[str, ParameterIdentityRecord | TargetParameterIdentityRecord]]:
    identities = tuple(record.identity for record in records)
    if (
        not identities
        or identities != tuple(sorted(identities))
        or len(set(identities)) != len(identities)
    ):
        raise PlanValidationError(f"{label} identity ids must be non-empty, sorted, and unique")
    by_key: dict[str, str] = {}
    by_identity: dict[str, ParameterIdentityRecord | TargetParameterIdentityRecord] = {}
    for record in records:
        by_identity[record.identity] = record
        specs: set[TensorSpec] = set()
        for key in record.model_keys:
            if key in by_key:
                raise PlanValidationError(f"{label} parameter key {key!r} occurs in two identities")
            if key not in tensor_specs:
                raise PlanValidationError(f"{label} parameter key {key!r} is absent from state")
            by_key[key] = record.identity
            specs.add(tensor_specs[key])
        if len(specs) != 1:
            raise PlanValidationError(
                f"{label} identity {record.identity!r} has conflicting alias tensor specs"
            )
    if tuple(sorted(by_key)) != parameter_keys:
        raise PlanValidationError(
            f"{label} identities do not partition the declared parameter binding keys"
        )
    return by_key, by_identity


def _alias_root(alias_map: dict[str, str], key: str) -> str:
    while key in alias_map:
        key = alias_map[key]
    return key


def _validate_optimizer_mapping(plan: ConversionPlan, alias_map: dict[str, str]) -> None:
    mapping = plan.optimizer_mapping
    if mapping is None:
        raise PlanValidationError("plan schema version 2 requires optimizer_mapping")
    expected_policy = (
        mapping.semantics == OPTIMIZER_MAPPING_SEMANTICS
        and mapping.tensor_fields == OPTIMIZER_TENSOR_FIELDS
        and mapping.scalar_fields == OPTIMIZER_SCALAR_FIELDS
        and mapping.fusion_step_policy == OPTIMIZER_FUSION_STEP_POLICY
        and mapping.parameter_group_policy == OPTIMIZER_PARAMETER_GROUP_POLICY
    )
    if not expected_policy:
        raise PlanValidationError(
            "optimizer_mapping has unsupported semantics, fields, or policies"
        )
    source_by_key, source_by_id = _validate_identity_partition(
        mapping.source_parameter_identities,
        mapping.source_parameter_keys,
        plan.source_tensors,
        label="source optimizer mapping",
    )
    target_by_key, _ = _validate_identity_partition(
        mapping.target_parameter_identities,
        mapping.target_parameter_keys,
        plan.target_tensors,
        label="target optimizer mapping",
    )
    source_parameter_key_set = set(source_by_key)
    target_parameter_key_set = set(target_by_key)
    if set(mapping.target_parameter_keys) & set(plan.derived_state):
        raise PlanValidationError("derived target state cannot be represented as optimizer state")

    actual_target_aliases: dict[str, list[str]] = {}
    for key in mapping.target_parameter_keys:
        root = _alias_root(alias_map, key)
        if root not in target_by_key:
            raise PlanValidationError(
                f"target optimizer parameter alias {key!r} resolves outside parameter state"
            )
        actual_target_aliases.setdefault(root, []).append(key)
    declared_target_aliases = {
        tuple(record.model_keys) for record in mapping.target_parameter_identities
    }
    if declared_target_aliases != {tuple(sorted(keys)) for keys in actual_target_aliases.values()}:
        raise PlanValidationError(
            "target optimizer identities disagree with the plan's parameter aliases"
        )

    forward_edges: set[tuple[str, str]] = set()
    used_source_identities: set[str] = set()
    for record in mapping.target_parameter_identities:
        if record.expression_target not in record.model_keys:
            raise PlanValidationError(
                f"optimizer target {record.identity!r} references an expression outside its aliases"
            )
        if _alias_root(alias_map, record.expression_target) != record.expression_target:
            raise PlanValidationError(
                f"optimizer target {record.identity!r} must reference its primary expression"
            )
        try:
            planned = plan.targets[record.expression_target]
        except KeyError as exc:
            raise PlanValidationError(
                f"optimizer target {record.identity!r} references a missing plan target"
            ) from exc
        slots = tuple(
            sorted({slot for key in record.model_keys for slot in plan.targets[key].semantic_slots})
        )
        if record.semantic_slots != slots:
            raise PlanValidationError(
                f"optimizer target {record.identity!r} has inconsistent semantic slots"
            )
        _validate_coordinate_expression(planned.expression, f"optimizer target {record.identity!r}")
        dependency_keys = set(planned.expression.source_keys())
        unexpected = dependency_keys - source_parameter_key_set
        if unexpected:
            raise PlanValidationError(
                f"optimizer target {record.identity!r} depends on non-parameter state: "
                f"{sorted(unexpected)}"
            )
        dependencies = tuple(sorted({source_by_key[key] for key in dependency_keys}))
        if record.source_identities != dependencies:
            raise PlanValidationError(
                f"optimizer target {record.identity!r} source identity dependencies disagree "
                "with its plan expression"
            )
        used_source_identities.update(dependencies)
        forward_edges.update((source_id, record.identity) for source_id in dependencies)
    if used_source_identities != set(source_by_id):
        raise PlanValidationError(
            "optimizer mapping does not consume every source parameter identity"
        )

    if plan.inverse_targets is None:
        raise PlanValidationError("optimizer mapping requires an exact inverse plan")
    missing_inverse = set(mapping.source_parameter_keys) - set(plan.inverse_targets)
    if missing_inverse:
        raise PlanValidationError(
            f"optimizer mapping inverse coverage is incomplete: {sorted(missing_inverse)}"
        )
    inverse_edges: set[tuple[str, str]] = set()
    for source_record in mapping.source_parameter_identities:
        for key in source_record.model_keys:
            inverse = plan.inverse_targets[key]
            inferred = inverse.expression.infer_spec(plan.target_tensors)
            if inferred != inverse.spec or inverse.spec != plan.source_tensors[key]:
                raise PlanValidationError(
                    f"optimizer inverse expression for {key!r} has inconsistent metadata"
                )
            _validate_coordinate_expression(
                inverse.expression, f"optimizer inverse source {source_record.identity!r}"
            )
            dependency_keys = set(inverse.expression.source_keys())
            unexpected = dependency_keys - target_parameter_key_set
            if unexpected:
                raise PlanValidationError(
                    f"optimizer inverse source {source_record.identity!r} depends on "
                    f"non-parameter target state: {sorted(unexpected)}"
                )
            inverse_edges.update(
                (source_record.identity, target_by_key[target_key])
                for target_key in dependency_keys
            )
    if inverse_edges != forward_edges:
        raise PlanValidationError(
            "optimizer forward and inverse parameter identity dependencies disagree"
        )


def validate_plan(plan: ConversionPlan) -> None:
    if plan.schema_version not in SUPPORTED_PLAN_SCHEMA_VERSIONS:
        raise PlanValidationError(f"unsupported plan schema version {plan.schema_version!r}")
    if plan.schema_version == 1 and plan.optimizer_mapping is not None:
        raise PlanValidationError("plan schema version 1 cannot contain optimizer_mapping")
    if plan.mapping_status != LinkStatus.UNIQUE:
        raise PlanValidationError(
            f"only UNIQUE plans are executable, got {plan.mapping_status.value}"
        )
    if not plan.plan_hash or plan.with_hash().plan_hash != plan.plan_hash:
        raise PlanValidationError("plan hash does not match canonical plan content")
    if not plan.targets:
        raise PlanValidationError("plan has no target tensors")
    if set(plan.targets) != set(plan.target_tensors):
        raise PlanValidationError(
            "target state coverage is incomplete: "
            f"missing={sorted(set(plan.target_tensors) - set(plan.targets))}, "
            f"unexpected={sorted(set(plan.targets) - set(plan.target_tensors))}"
        )
    for key, target in sorted(plan.targets.items()):
        if not key or "\x00" in key:
            raise PlanValidationError("target key is invalid")
        inferred = target.expression.infer_spec(plan.source_tensors)
        if inferred != target.spec:
            raise PlanValidationError(
                f"target {key!r} expression yields {inferred}, expected {target.spec}"
            )
        if target.spec != plan.target_tensors[key]:
            raise PlanValidationError(
                f"target {key!r} metadata does not match the embedded target state schema"
            )
    alias_map = {record.key: record.target_of for record in plan.aliases}
    if len(alias_map) != len(plan.aliases):
        raise PlanValidationError("duplicate alias keys")
    for start in sorted(alias_map):
        seen: set[str] = set()
        current = start
        while current in alias_map:
            if current in seen:
                raise PlanValidationError(f"alias dependency cycle includes {current!r}")
            seen.add(current)
            current = alias_map[current]
        if current not in plan.targets:
            raise PlanValidationError(f"alias {start!r} resolves to missing target {current!r}")
    if set(alias_map) - set(plan.targets):
        raise PlanValidationError("every alias must also have a target expression")
    if plan.schema_version == 2:
        _validate_optimizer_mapping(plan, alias_map)


def load_plan(path: Path) -> ConversionPlan:
    return plan_from_dict(load_json(path))


def empty_plan(
    *, source: PlanEndpoint, target: PlanEndpoint, architecture_signature: dict[str, Any]
) -> ConversionPlan:
    return ConversionPlan(
        PLAN_SCHEMA_VERSION,
        __version__,
        source,
        target,
        architecture_signature,
        {},
        {},
        {},
        (),
        (),
        (),
        "off",
        False,
        LinkStatus.UNSAT,
        0,
        {},
        (),
        (),
        None,
        "",
    ).with_hash()
