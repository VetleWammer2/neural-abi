"""Trusted bridges between live PyTorch Adam optimizers and safe bundles."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from neuralabi.checkpoints.store import metadata_from_tensor, update_tensor_fingerprint
from neuralabi.checkpoints.writer import DEFAULT_MAX_SHARD_SIZE
from neuralabi.formats.optimizer import (
    OPTIMIZER_FORMAT,
    OPTIMIZER_SCHEMA_VERSION,
    AdamHyperparameters,
    LoadedOptimizerBundle,
    OptimizerBundle,
    OptimizerModelBinding,
    OptimizerParameterGroup,
    OptimizerParameterState,
    OptimizerStateResult,
    load_optimizer_bundle,
    write_optimizer_bundle,
)
from neuralabi.ir.state import dtype_name, extract_state_schema, parameter_identity_groups
from neuralabi.status import CheckpointError


@dataclass(frozen=True)
class LiveParameterIdentity:
    parameter_id: str
    model_keys: tuple[str, ...]
    parameter: nn.Parameter


def model_checkpoint_fingerprint(model: nn.Module) -> str:
    """Compute the same logical fingerprint used by SafeTensors checkpoint stores."""

    digest = hashlib.sha256(b"neuralabi-logical-checkpoint-v1\0")
    for key, tensor in sorted(model.state_dict().items()):
        update_tensor_fingerprint(digest, key, metadata_from_tensor(tensor), tensor)
    return digest.hexdigest()


def live_parameter_identities(
    model: nn.Module, *, prefix: str = "source.parameter"
) -> tuple[LiveParameterIdentity, ...]:
    """Return unique trainable parameters with every exact state-dict alias."""

    groups: dict[int, tuple[nn.Parameter, list[str]]] = {}
    for key, parameter in model.named_parameters(remove_duplicate=False):
        existing = groups.get(id(parameter))
        if existing is None:
            groups[id(parameter)] = (parameter, [key])
        else:
            existing[1].append(key)
    ordered = [
        item
        for item in sorted(groups.values(), key=lambda item: tuple(sorted(item[1])))
        if item[0].requires_grad
    ]
    result: list[LiveParameterIdentity] = []
    for index, (parameter, keys) in enumerate(ordered):
        result.append(
            LiveParameterIdentity(f"{prefix}[{index:06d}]", tuple(sorted(keys)), parameter)
        )
    return tuple(result)


def _numeric(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or torch.is_tensor(value):
        raise CheckpointError(f"{label} must be a numeric scalar")
    try:
        result = float(value)
    except OverflowError as exc:
        raise CheckpointError(f"{label} must be a finite numeric scalar") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise CheckpointError(f"{label} must be finite and non-negative")
    return result


def _option_false_or_none(value: Any, label: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise CheckpointError(f"{label} must be false or null")
    if value is True:
        raise CheckpointError(f"{label}=true is unsupported")
    return cast(bool | None, value)


def _group_hyperparameters(
    group: dict[str, Any], *, algorithm: str, index: int
) -> AdamHyperparameters:
    label = f"optimizer parameter group {index}"
    allowed = {
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
    }
    unknown = set(group) - allowed
    if unknown:
        raise CheckpointError(f"{label} has unsupported fields: {sorted(unknown)}")
    required = allowed - {"decoupled_weight_decay"}
    missing = required - set(group)
    if missing:
        raise CheckpointError(f"{label} is missing fields: {sorted(missing)}")
    decoupled = group.get("decoupled_weight_decay", algorithm == "adamw")
    if not isinstance(decoupled, bool) or decoupled != (algorithm == "adamw"):
        raise CheckpointError(f"{label} weight-decay semantics disagree with {algorithm}")
    betas = group["betas"]
    if not isinstance(betas, tuple | list) or len(betas) != 2:
        raise CheckpointError(f"{label} betas must contain two numeric scalars")
    beta_values = (
        _numeric(betas[0], f"{label} beta1", nonnegative=True),
        _numeric(betas[1], f"{label} beta2", nonnegative=True),
    )
    if any(value >= 1 for value in beta_values):
        raise CheckpointError(f"{label} betas must be less than one")
    for field in ("amsgrad", "capturable", "differentiable"):
        if not isinstance(group[field], bool):
            raise CheckpointError(f"{label} {field} must be a boolean")
        if group[field]:
            raise CheckpointError(f"{label} {field}=true is unsupported")
    if not isinstance(group["maximize"], bool):
        raise CheckpointError(f"{label} maximize must be a boolean")
    eps = _numeric(group["eps"], f"{label} eps", nonnegative=True)
    if eps == 0:
        raise CheckpointError(f"{label} eps must be positive")
    return AdamHyperparameters(
        lr=_numeric(group["lr"], f"{label} lr", nonnegative=True),
        betas=beta_values,
        eps=eps,
        weight_decay=_numeric(group["weight_decay"], f"{label} weight_decay", nonnegative=True),
        maximize=group["maximize"],
        amsgrad=False,
        foreach=_option_false_or_none(group["foreach"], f"{label} foreach"),
        capturable=False,
        differentiable=False,
        fused=_option_false_or_none(group["fused"], f"{label} fused"),
    )


def _algorithm(optimizer: torch.optim.Optimizer) -> str:
    if type(optimizer) is torch.optim.Adam:
        return "adam"
    if type(optimizer) is torch.optim.AdamW:
        return "adamw"
    raise CheckpointError(
        f"unsupported optimizer {type(optimizer).__module__}.{type(optimizer).__qualname__}; "
        "only exact torch.optim.Adam and AdamW are supported"
    )


def export_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    output: Path,
    *,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> OptimizerStateResult:
    """Export a fully initialized live Adam/AdamW without pickle."""

    algorithm = _algorithm(optimizer)
    identities = live_parameter_identities(model)
    schema = extract_state_schema(model)
    if {item.model_keys for item in identities} != set(parameter_identity_groups(schema)):
        raise CheckpointError(
            "model storage aliases do not correspond one-to-one with Parameter objects"
        )
    by_object = {id(item.parameter): item for item in identities}
    covered: set[str] = set()
    groups: list[OptimizerParameterGroup] = []
    for index, raw_group in enumerate(optimizer.param_groups):
        group = raw_group
        hyperparameters = _group_hyperparameters(group, algorithm=algorithm, index=index)
        parameter_ids: list[str] = []
        for parameter in group["params"]:
            if not isinstance(parameter, nn.Parameter):
                raise CheckpointError("optimizer contains a non-Parameter tensor")
            try:
                identity = by_object[id(parameter)]
            except KeyError as exc:
                raise CheckpointError(
                    "optimizer contains a parameter that is not a trainable model parameter"
                ) from exc
            if identity.parameter_id in covered:
                raise CheckpointError(
                    f"optimizer parameter {identity.parameter_id!r} appears more than once"
                )
            covered.add(identity.parameter_id)
            parameter_ids.append(identity.parameter_id)
        groups.append(OptimizerParameterGroup(tuple(parameter_ids), hyperparameters))
    expected = set(by_object_item.parameter_id for by_object_item in identities)
    if covered != expected:
        raise CheckpointError(
            "optimizer does not cover every trainable model parameter: "
            f"missing={sorted(expected - covered)}, unknown={sorted(covered - expected)}"
        )
    optimizer_state_objects = {id(parameter) for parameter in optimizer.state}
    expected_state_objects = {id(item.parameter) for item in identities}
    if optimizer_state_objects != expected_state_objects:
        raise CheckpointError(
            "optimizer must have fully initialized state with no orphan parameter entries"
        )
    tensors: dict[str, torch.Tensor] = {}
    parameters: list[OptimizerParameterState] = []
    for index, identity in enumerate(identities):
        state = optimizer.state.get(identity.parameter)
        if state is None or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            fields = [] if state is None else sorted(str(key) for key in state)
            raise CheckpointError(
                f"{identity.parameter_id} must have fully initialized Adam state; got {fields}"
            )
        if any(not isinstance(state[field], torch.Tensor) for field in state):
            raise CheckpointError(f"{identity.parameter_id} state fields must all be tensors")
        step = cast(torch.Tensor, state["step"])
        exp_avg = cast(torch.Tensor, state["exp_avg"])
        exp_avg_sq = cast(torch.Tensor, state["exp_avg_sq"])
        shape = tuple(int(size) for size in identity.parameter.shape)
        dtype = dtype_name(identity.parameter.dtype)
        if tuple(step.shape) != () or step.dtype not in {torch.float32, torch.float64}:
            raise CheckpointError(f"{identity.parameter_id}.step must be a float scalar")
        value = float(step.detach().cpu().item())
        if not math.isfinite(value) or value < 0 or value != math.floor(value):
            raise CheckpointError(
                f"{identity.parameter_id}.step must be finite, non-negative, and integer-valued"
            )
        for field, tensor in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
            if tuple(tensor.shape) != shape or dtype_name(tensor.dtype) != dtype:
                raise CheckpointError(
                    f"{identity.parameter_id}.{field} metadata does not match its parameter"
                )
        stem = f"state.{index:06d}"
        step_key = f"{stem}.step"
        exp_avg_key = f"{stem}.exp_avg"
        exp_avg_sq_key = f"{stem}.exp_avg_sq"
        tensors[step_key] = step.detach().cpu().clone()
        tensors[exp_avg_key] = exp_avg.detach().cpu().contiguous().clone()
        tensors[exp_avg_sq_key] = exp_avg_sq.detach().cpu().contiguous().clone()
        parameters.append(
            OptimizerParameterState(
                identity.parameter_id,
                identity.model_keys,
                shape,
                dtype,
                True,
                step_key,
                exp_avg_key,
                exp_avg_sq_key,
            )
        )
    bundle = OptimizerBundle(
        schema_version=OPTIMIZER_SCHEMA_VERSION,
        format=OPTIMIZER_FORMAT,
        algorithm=algorithm,
        model_binding=OptimizerModelBinding(
            schema.schema_hash, model_checkpoint_fingerprint(model)
        ),
        parameter_groups=tuple(groups),
        parameters=tuple(parameters),
        tensor_fingerprint="pending",
        bundle_hash="pending",
    )
    return write_optimizer_bundle(bundle, tensors, output, max_shard_size=max_shard_size)


def _constructor_group(
    group: OptimizerParameterGroup,
    parameters: dict[str, nn.Parameter],
) -> dict[str, Any]:
    options = group.hyperparameters
    return {
        "params": [parameters[item] for item in group.parameters],
        "lr": options.lr,
        "betas": options.betas,
        "eps": options.eps,
        "weight_decay": options.weight_decay,
        "amsgrad": False,
        "maximize": options.maximize,
        "foreach": options.foreach,
        "capturable": False,
        "differentiable": False,
        "fused": options.fused,
    }


def load_optimizer_from_bundle(
    model: nn.Module, loaded: LoadedOptimizerBundle
) -> torch.optim.Optimizer:
    """Construct and restore an optimizer after all model/identity checks pass."""

    bundle = loaded.bundle
    schema_hash = extract_state_schema(model).schema_hash
    if schema_hash != bundle.model_binding.state_schema_hash:
        raise CheckpointError("optimizer state does not match the model state schema")
    fingerprint = model_checkpoint_fingerprint(model)
    if fingerprint != bundle.model_binding.checkpoint_fingerprint:
        raise CheckpointError("optimizer state does not match the model checkpoint fingerprint")
    live = live_parameter_identities(model, prefix="live.parameter")
    live_by_keys = {item.model_keys: item.parameter for item in live}
    bundle_keys = {item.model_keys for item in bundle.parameters}
    if set(live_by_keys) != bundle_keys:
        raise CheckpointError(
            "optimizer parameter identities do not match the model: "
            f"missing={sorted(bundle_keys - set(live_by_keys))}, "
            f"unknown={sorted(set(live_by_keys) - bundle_keys)}"
        )
    parameters: dict[str, nn.Parameter] = {}
    by_id = bundle.by_id()
    for parameter_id, item in by_id.items():
        parameter = live_by_keys[item.model_keys]
        if tuple(parameter.shape) != item.shape or dtype_name(parameter.dtype) != item.dtype:
            raise CheckpointError(f"{parameter_id} metadata does not match the live parameter")
        parameters[parameter_id] = parameter
    groups = [_constructor_group(group, parameters) for group in bundle.parameter_groups]
    if bundle.algorithm == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(groups)
    elif bundle.algorithm == "adamw":
        optimizer = torch.optim.AdamW(groups)
    else:  # The bundle parser already rejects this; retain a local totality check.
        raise CheckpointError(f"unsupported optimizer algorithm {bundle.algorithm!r}")
    for parameter_id in by_id:
        parameter = parameters[parameter_id]
        state = loaded.read_state(parameter_id)
        optimizer.state[parameter] = {
            "step": state["step"].detach().cpu().clone(),
            "exp_avg": state["exp_avg"].to(parameter.device).clone(),
            "exp_avg_sq": state["exp_avg_sq"].to(parameter.device).clone(),
        }
    return optimizer


def load_optimizer_state(model: nn.Module, path: Path) -> torch.optim.Optimizer:
    """Load a safe bundle and build the Adam/AdamW it names."""

    return load_optimizer_from_bundle(model, load_optimizer_bundle(path))
