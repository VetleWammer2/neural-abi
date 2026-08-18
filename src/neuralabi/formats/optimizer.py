"""Safe Adam/AdamW optimizer-state bundles: bounded JSON manifest plus SafeTensors."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from neuralabi.checkpoints.store import TensorStore, open_tensor_store
from neuralabi.checkpoints.writer import DEFAULT_MAX_SHARD_SIZE, write_checkpoint
from neuralabi.status import CheckpointError, PlanValidationError
from neuralabi.transforms import TensorSpec
from neuralabi.util.canonical_json import load_json, pretty_dumps
from neuralabi.util.hashing import hash_canonical, hash_file

OPTIMIZER_SCHEMA_VERSION = 1
OPTIMIZER_FORMAT = "neuralabi.optimizer-state"
OPTIMIZER_MANIFEST = "optimizer.neuralabi.json"
OPTIMIZER_TENSORS = "state"
MAX_OPTIMIZER_PARAMETERS = 1_000_000
MAX_OPTIMIZER_GROUPS = 65_536
MAX_MODEL_KEYS = 64
MAX_STRING = 4096


@dataclass(frozen=True)
class OptimizerModelBinding:
    state_schema_hash: str
    checkpoint_fingerprint: str

    def to_dict(self) -> dict[str, str]:
        return {
            "state_schema_hash": self.state_schema_hash,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
        }


@dataclass(frozen=True)
class AdamHyperparameters:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    maximize: bool
    amsgrad: bool
    foreach: bool | None
    capturable: bool
    differentiable: bool
    fused: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "lr": self.lr,
            "betas": list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "maximize": self.maximize,
            "amsgrad": self.amsgrad,
            "foreach": self.foreach,
            "capturable": self.capturable,
            "differentiable": self.differentiable,
            "fused": self.fused,
        }


@dataclass(frozen=True)
class OptimizerParameterGroup:
    parameters: tuple[str, ...]
    hyperparameters: AdamHyperparameters

    def to_dict(self) -> dict[str, object]:
        return {
            "parameters": list(self.parameters),
            "hyperparameters": self.hyperparameters.to_dict(),
        }


@dataclass(frozen=True)
class OptimizerParameterState:
    parameter_id: str
    model_keys: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool
    step: str
    exp_avg: str
    exp_avg_sq: str

    @property
    def spec(self) -> TensorSpec:
        return TensorSpec(self.shape, self.dtype)

    def tensor_keys(self) -> tuple[str, str, str]:
        return self.step, self.exp_avg, self.exp_avg_sq

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "model_keys": list(self.model_keys),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
            "state": {
                "step": self.step,
                "exp_avg": self.exp_avg,
                "exp_avg_sq": self.exp_avg_sq,
            },
        }


@dataclass(frozen=True)
class OptimizerBundle:
    schema_version: int
    format: str
    algorithm: str
    model_binding: OptimizerModelBinding
    parameter_groups: tuple[OptimizerParameterGroup, ...]
    parameters: tuple[OptimizerParameterState, ...]
    tensor_fingerprint: str
    bundle_hash: str

    def by_id(self) -> dict[str, OptimizerParameterState]:
        return {item.parameter_id: item for item in self.parameters}

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "format": self.format,
            "algorithm": self.algorithm,
            "model_binding": self.model_binding.to_dict(),
            "parameter_groups": [group.to_dict() for group in self.parameter_groups],
            "parameters": [item.to_dict() for item in self.parameters],
            "tensor_fingerprint": self.tensor_fingerprint,
        }
        if include_hash:
            result["bundle_hash"] = self.bundle_hash
        return result

    def with_hash(self) -> OptimizerBundle:
        return replace(self, bundle_hash=hash_canonical(self.to_dict(include_hash=False)))


@dataclass(frozen=True)
class LoadedOptimizerBundle:
    checkpoint_path: Path
    bundle: OptimizerBundle
    store: TensorStore

    def read_state(self, parameter_id: str) -> dict[str, torch.Tensor]:
        try:
            item = self.bundle.by_id()[parameter_id]
        except KeyError as exc:
            raise CheckpointError(f"optimizer state has no parameter {parameter_id!r}") from exc
        return {
            "step": self.store.read(item.step),
            "exp_avg": self.store.read(item.exp_avg),
            "exp_avg_sq": self.store.read(item.exp_avg_sq),
        }


@dataclass(frozen=True)
class OptimizerStateResult:
    checkpoint_path: Path
    tensor_checkpoint_path: Path
    tensor_fingerprint: str
    bundle_hash: str
    file_hashes: dict[str, str]
    tensor_count: int
    logical_bytes: int


def optimizer_tensor_fingerprint(store: TensorStore) -> str:
    """Fingerprint optimizer tensors, including zero-dimensional step tensors."""

    digest = hashlib.sha256(b"neuralabi-logical-optimizer-state-v1\0")
    for key in sorted(store.keys()):
        metadata = store.metadata(key)
        key_bytes = key.encode("utf-8")
        digest.update(len(key_bytes).to_bytes(8, "big"))
        digest.update(key_bytes)
        shape_text = ",".join(str(size) for size in metadata.shape).encode("ascii")
        dtype_text = metadata.dtype.encode("ascii")
        digest.update(len(shape_text).to_bytes(8, "big"))
        digest.update(shape_text)
        digest.update(len(dtype_text).to_bytes(8, "big"))
        digest.update(dtype_text)
        contiguous = store.read(key).detach().cpu().contiguous().reshape(-1)
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensors_bitwise_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Compare tensor metadata and raw contiguous CPU bytes."""

    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    first_bytes = first.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    second_bytes = second.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(first_bytes, second_bytes)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{label} must be an object")
    return value


def _exact_fields(data: dict[str, Any], expected: set[str], label: str) -> None:
    if set(data) != expected:
        raise CheckpointError(
            f"{label} fields differ from schema: "
            f"missing={sorted(expected - set(data))}, unknown={sorted(set(data) - expected)}"
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_STRING or "\x00" in value:
        raise CheckpointError(f"{label} must be a non-empty bounded string")
    return value


def _number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CheckpointError(f"{label} must be a numeric scalar")
    try:
        result = float(value)
    except OverflowError as exc:
        raise CheckpointError(f"{label} must be a finite numeric scalar") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise CheckpointError(f"{label} must be finite and non-negative")
    return result


def _optional_bool(value: Any, label: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise CheckpointError(f"{label} must be false or null")
    if value is True:
        raise CheckpointError(f"{label}=true is unsupported")
    return value


def _required_bool(value: Any, label: str, *, false_only: bool = False) -> bool:
    if not isinstance(value, bool):
        raise CheckpointError(f"{label} must be a boolean")
    if false_only and value:
        raise CheckpointError(f"{label}=true is unsupported")
    return value


def _hyperparameters(value: Any, label: str) -> AdamHyperparameters:
    data = _object(value, label)
    expected = {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "maximize",
        "amsgrad",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    }
    _exact_fields(data, expected, label)
    betas = data["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise CheckpointError(f"{label}.betas must contain two numeric scalars")
    beta_values = (
        _number(betas[0], f"{label}.betas[0]", nonnegative=True),
        _number(betas[1], f"{label}.betas[1]", nonnegative=True),
    )
    if any(beta >= 1 for beta in beta_values):
        raise CheckpointError(f"{label}.betas must be less than one")
    eps = _number(data["eps"], f"{label}.eps", nonnegative=True)
    if eps == 0:
        raise CheckpointError(f"{label}.eps must be positive")
    return AdamHyperparameters(
        lr=_number(data["lr"], f"{label}.lr", nonnegative=True),
        betas=beta_values,
        eps=eps,
        weight_decay=_number(data["weight_decay"], f"{label}.weight_decay", nonnegative=True),
        maximize=_required_bool(data["maximize"], f"{label}.maximize"),
        amsgrad=_required_bool(data["amsgrad"], f"{label}.amsgrad", false_only=True),
        foreach=_optional_bool(data["foreach"], f"{label}.foreach"),
        capturable=_required_bool(data["capturable"], f"{label}.capturable", false_only=True),
        differentiable=_required_bool(
            data["differentiable"], f"{label}.differentiable", false_only=True
        ),
        fused=_optional_bool(data["fused"], f"{label}.fused"),
    )


def optimizer_bundle_from_dict(value: Any) -> OptimizerBundle:
    data = _object(value, "optimizer manifest")
    expected = {
        "schema_version",
        "format",
        "algorithm",
        "model_binding",
        "parameter_groups",
        "parameters",
        "tensor_fingerprint",
        "bundle_hash",
    }
    _exact_fields(data, expected, "optimizer manifest")
    if (
        isinstance(data["schema_version"], bool)
        or not isinstance(data["schema_version"], int)
        or data["schema_version"] != OPTIMIZER_SCHEMA_VERSION
    ):
        raise CheckpointError(f"unsupported optimizer schema version {data['schema_version']!r}")
    if data["format"] != OPTIMIZER_FORMAT:
        raise CheckpointError(f"unsupported optimizer format {data['format']!r}")
    if not isinstance(data["algorithm"], str) or data["algorithm"] not in {"adam", "adamw"}:
        raise CheckpointError(f"unsupported optimizer algorithm {data['algorithm']!r}")
    model_data = _object(data["model_binding"], "model_binding")
    _exact_fields(model_data, {"state_schema_hash", "checkpoint_fingerprint"}, "model_binding")
    model_binding = OptimizerModelBinding(
        _string(model_data["state_schema_hash"], "model_binding.state_schema_hash"),
        _string(model_data["checkpoint_fingerprint"], "model_binding.checkpoint_fingerprint"),
    )
    groups_raw = data["parameter_groups"]
    if not isinstance(groups_raw, list) or not groups_raw or len(groups_raw) > MAX_OPTIMIZER_GROUPS:
        raise CheckpointError("parameter_groups must be a non-empty bounded array")
    groups: list[OptimizerParameterGroup] = []
    for index, raw in enumerate(groups_raw):
        group_data = _object(raw, f"parameter_groups[{index}]")
        _exact_fields(group_data, {"parameters", "hyperparameters"}, f"parameter_groups[{index}]")
        raw_parameters = group_data["parameters"]
        if not isinstance(raw_parameters, list) or len(raw_parameters) > MAX_OPTIMIZER_PARAMETERS:
            raise CheckpointError(f"parameter_groups[{index}].parameters must be a bounded array")
        group_parameter_ids = tuple(
            _string(item, f"parameter_groups[{index}].parameters") for item in raw_parameters
        )
        if len(set(group_parameter_ids)) != len(group_parameter_ids):
            raise CheckpointError(f"parameter_groups[{index}] contains duplicate parameters")
        groups.append(
            OptimizerParameterGroup(
                group_parameter_ids,
                _hyperparameters(
                    group_data["hyperparameters"], f"parameter_groups[{index}].hyperparameters"
                ),
            )
        )
    parameters_raw = data["parameters"]
    if (
        not isinstance(parameters_raw, list)
        or not parameters_raw
        or len(parameters_raw) > MAX_OPTIMIZER_PARAMETERS
    ):
        raise CheckpointError("parameters must be a non-empty bounded array")
    parameters: list[OptimizerParameterState] = []
    for index, raw in enumerate(parameters_raw):
        item = _object(raw, f"parameters[{index}]")
        _exact_fields(
            item,
            {
                "parameter_id",
                "model_keys",
                "shape",
                "dtype",
                "requires_grad",
                "state",
            },
            f"parameters[{index}]",
        )
        model_keys_raw = item["model_keys"]
        if (
            not isinstance(model_keys_raw, list)
            or not model_keys_raw
            or len(model_keys_raw) > MAX_MODEL_KEYS
        ):
            raise CheckpointError(f"parameters[{index}].model_keys must be a bounded array")
        model_keys = tuple(
            _string(key, f"parameters[{index}].model_keys") for key in model_keys_raw
        )
        if model_keys != tuple(sorted(set(model_keys))):
            raise CheckpointError(f"parameters[{index}].model_keys must be sorted and unique")
        shape_raw = item["shape"]
        if not isinstance(shape_raw, list) or any(
            isinstance(size, bool) or not isinstance(size, int) for size in shape_raw
        ):
            raise CheckpointError(f"parameters[{index}].shape must be an integer array")
        try:
            spec = TensorSpec(tuple(shape_raw), _string(item["dtype"], "parameter dtype"))
        except PlanValidationError as exc:
            raise CheckpointError(
                f"parameters[{index}] has invalid tensor metadata: {exc}"
            ) from exc
        requires_grad = _required_bool(item["requires_grad"], f"parameters[{index}].requires_grad")
        if not requires_grad:
            raise CheckpointError("optimizer bundles may contain only trainable parameters")
        state = _object(item["state"], f"parameters[{index}].state")
        _exact_fields(state, {"step", "exp_avg", "exp_avg_sq"}, f"parameters[{index}].state")
        parameters.append(
            OptimizerParameterState(
                _string(item["parameter_id"], f"parameters[{index}].parameter_id"),
                model_keys,
                spec.shape,
                spec.dtype,
                requires_grad,
                _string(state["step"], f"parameters[{index}].state.step"),
                _string(state["exp_avg"], f"parameters[{index}].state.exp_avg"),
                _string(state["exp_avg_sq"], f"parameters[{index}].state.exp_avg_sq"),
            )
        )
    bundle = OptimizerBundle(
        OPTIMIZER_SCHEMA_VERSION,
        OPTIMIZER_FORMAT,
        data["algorithm"],
        model_binding,
        tuple(groups),
        tuple(parameters),
        _string(data["tensor_fingerprint"], "tensor_fingerprint"),
        _string(data["bundle_hash"], "bundle_hash"),
    )
    validate_optimizer_bundle(bundle)
    return bundle


def validate_optimizer_bundle(bundle: OptimizerBundle, store: TensorStore | None = None) -> None:
    if (
        isinstance(bundle.schema_version, bool)
        or bundle.schema_version != OPTIMIZER_SCHEMA_VERSION
        or bundle.format != OPTIMIZER_FORMAT
    ):
        raise CheckpointError("optimizer bundle has an unsupported schema")
    if not isinstance(bundle.algorithm, str) or bundle.algorithm not in {"adam", "adamw"}:
        raise CheckpointError(f"unsupported optimizer algorithm {bundle.algorithm!r}")
    if not bundle.bundle_hash or bundle.with_hash().bundle_hash != bundle.bundle_hash:
        raise CheckpointError("optimizer bundle hash does not match canonical manifest content")
    by_id = bundle.by_id()
    if len(by_id) != len(bundle.parameters):
        raise CheckpointError("optimizer bundle contains duplicate parameter IDs")
    parameter_ids = tuple(item.parameter_id for item in bundle.parameters)
    if parameter_ids != tuple(sorted(parameter_ids)):
        raise CheckpointError("optimizer parameter IDs must be sorted")
    all_group_ids = tuple(
        parameter for group in bundle.parameter_groups for parameter in group.parameters
    )
    if len(set(all_group_ids)) != len(all_group_ids):
        raise CheckpointError("an optimizer parameter appears in more than one parameter group")
    missing = sorted(set(by_id) - set(all_group_ids))
    unknown = sorted(set(all_group_ids) - set(by_id))
    if missing or unknown:
        raise CheckpointError(
            f"optimizer parameter coverage is incomplete: missing={missing}, unknown={unknown}"
        )
    model_key_owner: dict[str, str] = {}
    tensor_keys: list[str] = []
    for item in bundle.parameters:
        for key in item.model_keys:
            if key in model_key_owner:
                raise CheckpointError(
                    f"model key {key!r} belongs to both {model_key_owner[key]!r} and {item.parameter_id!r}"
                )
            model_key_owner[key] = item.parameter_id
        tensor_keys.extend(item.tensor_keys())
    if len(set(tensor_keys)) != len(tensor_keys):
        raise CheckpointError("optimizer tensor references must be unique")
    if store is None:
        return
    actual_keys = set(store.keys())
    expected_keys = set(tensor_keys)
    if actual_keys != expected_keys:
        raise CheckpointError(
            "optimizer tensor coverage is incomplete: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unknown={sorted(actual_keys - expected_keys)}"
        )
    if optimizer_tensor_fingerprint(store) != bundle.tensor_fingerprint:
        raise CheckpointError("optimizer tensor fingerprint does not match the manifest")
    for item in bundle.parameters:
        for field, key in (("exp_avg", item.exp_avg), ("exp_avg_sq", item.exp_avg_sq)):
            metadata = store.metadata(key)
            if metadata.shape != item.shape or metadata.dtype != item.dtype:
                raise CheckpointError(
                    f"{item.parameter_id}.{field} metadata "
                    f"{(metadata.shape, metadata.dtype)} != {(item.shape, item.dtype)}"
                )
        step_metadata = store.metadata(item.step)
        if step_metadata.shape != () or step_metadata.dtype not in {"float32", "float64"}:
            raise CheckpointError(f"{item.parameter_id}.step must be a float32/float64 scalar")
        step = store.read(item.step)
        value = float(step.item())
        if not math.isfinite(value) or value < 0 or value != math.floor(value):
            raise CheckpointError(
                f"{item.parameter_id}.step must be finite, non-negative, and integer-valued"
            )


def load_optimizer_bundle(path: Path) -> LoadedOptimizerBundle:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise CheckpointError(
            "optimizer state must be a NeuralABI JSON+SafeTensors bundle directory; pickle is unsupported"
        )
    entries = {item.name for item in root.iterdir()}
    expected = {OPTIMIZER_MANIFEST, OPTIMIZER_TENSORS}
    if entries != expected:
        raise CheckpointError(
            f"optimizer bundle entries differ from schema: missing={sorted(expected - entries)}, "
            f"unknown={sorted(entries - expected)}"
        )
    manifest_path = root / OPTIMIZER_MANIFEST
    if not manifest_path.is_file() or not (root / OPTIMIZER_TENSORS).is_dir():
        raise CheckpointError("optimizer bundle manifest/state paths have the wrong type")
    try:
        raw = load_json(manifest_path)
    except PlanValidationError as exc:
        raise CheckpointError(f"invalid optimizer manifest: {exc}") from exc
    bundle = optimizer_bundle_from_dict(raw)
    store = open_tensor_store(root / OPTIMIZER_TENSORS)
    validate_optimizer_bundle(bundle, store)
    return LoadedOptimizerBundle(root, bundle, store)


def write_optimizer_bundle(
    bundle: OptimizerBundle,
    tensors: dict[str, torch.Tensor],
    output: Path,
    *,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> OptimizerStateResult:
    output = output.resolve(strict=False)
    if output.exists() and (output.is_file() or any(output.iterdir())):
        raise CheckpointError(f"refusing to replace non-empty output path {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.optimizer-", dir=output.parent))
    try:
        tensor_main, tensor_files, logical_bytes = write_checkpoint(
            iter(sorted(tensors.items())),
            temporary / OPTIMIZER_TENSORS,
            max_shard_size=max_shard_size,
        )
        store = open_tensor_store(temporary / OPTIMIZER_TENSORS)
        complete = replace(
            bundle,
            tensor_fingerprint=optimizer_tensor_fingerprint(store),
            bundle_hash="",
        ).with_hash()
        complete = optimizer_bundle_from_dict(complete.to_dict())
        validate_optimizer_bundle(complete, store)
        manifest = temporary / OPTIMIZER_MANIFEST
        manifest.write_text(pretty_dumps(complete.to_dict()), encoding="utf-8", newline="\n")
        staged_files = (manifest, *tensor_files)
        hashes = {
            str(path.relative_to(temporary)).replace("\\", "/"): hash_file(path)
            for path in staged_files
        }
        if output.exists():
            output.rmdir()
        os.replace(temporary, output)
        return OptimizerStateResult(
            checkpoint_path=output,
            tensor_checkpoint_path=output
            / OPTIMIZER_TENSORS
            / tensor_main.relative_to(temporary / OPTIMIZER_TENSORS),
            tensor_fingerprint=complete.tensor_fingerprint,
            bundle_hash=complete.bundle_hash,
            file_hashes=dict(sorted(hashes.items())),
            tensor_count=len(tensors),
            logical_bytes=logical_bytes,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
