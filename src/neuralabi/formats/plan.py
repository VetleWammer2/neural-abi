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

PLAN_SCHEMA_VERSION = 1
MAX_TARGETS = 1_000_000


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


def plan_from_dict(value: Any) -> ConversionPlan:
    data = _required_dict(value, "plan")
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
    if set(data) != required:
        raise PlanValidationError(
            f"plan fields differ from schema: missing={sorted(required - set(data))}, unknown={sorted(set(data) - required)}"
        )
    if data["schema_version"] != PLAN_SCHEMA_VERSION:
        raise PlanValidationError(f"unsupported plan schema version {data['schema_version']!r}")
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
        PLAN_SCHEMA_VERSION,
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
    )
    validate_plan(plan)
    return plan


def validate_plan(plan: ConversionPlan) -> None:
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
