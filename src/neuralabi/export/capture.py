"""Mandatory ``torch.export`` capture and normalization boundary."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch import fx, nn

from neuralabi.adapters import ModelAdapter
from neuralabi.canonical.pipeline import canonicalize_program
from neuralabi.canonical.rewrites import CANONICALIZATION_VERSION
from neuralabi.export.artifact import ExportArtifact, ExportOptions, InputSignature
from neuralabi.ir.graph import TensorNode, TensorProgram, TensorValue
from neuralabi.ir.state import extract_state_schema
from neuralabi.status import UnsupportedGraphError
from neuralabi.util.hashing import hash_canonical


@dataclass
class CapturedModel:
    """Runtime model plus portable artifact; raw ExportedProgram stays inside capture."""

    adapter: ModelAdapter
    model: nn.Module
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    artifact: ExportArtifact
    runtime_graph: fx.GraphModule


def _target_name(target: Any) -> str:
    if target is operator.getitem:
        return "operator.getitem"
    return str(target)


def _argument(value: Any) -> Any:
    if isinstance(value, fx.Node):
        return {"node": value.name}
    if isinstance(value, tuple | list):
        return [_argument(child) for child in value]
    if isinstance(value, dict):
        return {
            str(key): _argument(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.memory_format):
        return str(value)
    return repr(value)


def _tensor_values(value: Any) -> tuple[TensorValue, ...]:
    values = value if isinstance(value, tuple | list) else (value,)
    result: list[TensorValue] = []
    for item in values:
        if isinstance(item, torch.Tensor):
            shape = tuple(int(size) if isinstance(size, int) else str(size) for size in item.shape)
            result.append(TensorValue(shape, str(item.dtype).removeprefix("torch.")))
    return tuple(result)


def _normalize_graph(exported: torch.export.ExportedProgram) -> TensorProgram:
    graph_signature = exported.graph_signature
    parameter_bindings = dict(graph_signature.inputs_to_parameters)
    buffer_bindings = dict(graph_signature.inputs_to_buffers)
    state_bindings = {**parameter_bindings, **buffer_bindings}
    nodes: list[TensorNode] = []
    user_inputs: list[str] = []
    user_outputs: list[str] = []
    for node in exported.graph_module.graph.nodes:
        if node.op == "placeholder" and node.name not in state_bindings:
            user_inputs.append(node.name)
        if node.op == "output":

            def collect(value: Any) -> None:
                if isinstance(value, fx.Node):
                    user_outputs.append(value.name)
                elif isinstance(value, tuple | list):
                    for child in value:
                        collect(child)

            collect(node.args)
        target = _target_name(node.target)
        nodes.append(
            TensorNode(
                node_id=node.name,
                kind=node.op,
                target=target,
                canonical_op=target,
                args=_argument(node.args),
                kwargs=_argument(node.kwargs),
                outputs=_tensor_values(node.meta.get("val")),
                users=tuple(sorted(user.name for user in node.users)),
            )
        )
    program = TensorProgram(
        nodes=tuple(nodes),
        state_bindings=tuple(sorted(state_bindings.items())),
        user_inputs=tuple(user_inputs),
        user_outputs=tuple(user_outputs),
        rewrite_rules_used=(),
    )
    return canonicalize_program(program)


def _input_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> InputSignature:
    specs: list[tuple[str, tuple[int | str, ...], str]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            specs.append(
                (
                    path,
                    tuple(int(size) for size in value.shape),
                    str(value.dtype).removeprefix("torch."),
                )
            )
        elif isinstance(value, tuple | list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}.{key}")

    visit(args, "args")
    visit(kwargs, "kwargs")
    return InputSignature(len(args), tuple(sorted(kwargs)), tuple(specs))


def capture_adapter(
    adapter: ModelAdapter,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> CapturedModel:
    torch.manual_seed(seed)
    model = adapter.build(device=device, dtype=dtype)
    adapter.prepare(model)
    model.eval()
    args, kwargs = adapter.example_inputs(seed=seed, device=device)
    dynamic_shapes = adapter.dynamic_shapes()
    try:
        exported = torch.export.export(
            model,
            args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    except Exception as exc:
        raise UnsupportedGraphError(
            f"torch.export failed for adapter {adapter.adapter_id!r}: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        # An explicit empty decomposition table retains recognizable higher-level ATen operators
        # while applying ExportedProgram's functionalization pass to local tensor mutation.
        exported = exported.run_decompositions({})
    except Exception as exc:
        raise UnsupportedGraphError(
            f"export functionalization failed for adapter {adapter.adapter_id!r}: {type(exc).__name__}: {exc}"
        ) from exc
    program = _normalize_graph(exported)
    signature = exported.graph_signature
    mutated_bindings = {
        "parameters": dict(signature.parameters_to_mutate),
        "buffers": dict(signature.buffers_to_mutate),
        "user_inputs": dict(signature.user_inputs_to_mutate),
    }
    if any(mutated_bindings.values()):
        raise UnsupportedGraphError(
            f"exported graph mutates persistent or user inputs: {mutated_bindings}"
        )
    state_schema = extract_state_schema(model)
    graph_payload = {
        "canonicalization_version": CANONICALIZATION_VERSION,
        "program": program.to_dict(),
        "state_schema_hash": state_schema.schema_hash,
    }
    signature_data = {
        "inputs_to_parameters": dict(sorted(exported.graph_signature.inputs_to_parameters.items())),
        "inputs_to_buffers": dict(sorted(exported.graph_signature.inputs_to_buffers.items())),
        "user_inputs": list(program.user_inputs),
        "user_outputs": list(program.user_outputs),
    }
    artifact = ExportArtifact(
        adapter_id=adapter.adapter_id,
        torch_version=torch.__version__,
        graph=program,
        state_schema=state_schema,
        graph_signature=signature_data,
        range_constraints=tuple(sorted(str(item) for item in exported.range_constraints.items())),
        example_input_signature=_input_signature(args, kwargs),
        export_options=ExportOptions(
            strict=True,
            canonicalization_version=CANONICALIZATION_VERSION,
            functional_aten=True,
            dynamic_shapes_supplied=dynamic_shapes is not None,
        ),
        graph_hash=hash_canonical(graph_payload),
    )
    runtime_graph = exported.module()
    return CapturedModel(adapter, model, args, kwargs, artifact, runtime_graph)
