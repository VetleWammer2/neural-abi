"""Structural recognizer for the opaque TwinMLP milestone."""

from __future__ import annotations

from dataclasses import dataclass

from neuralabi.export.artifact import ExportArtifact
from neuralabi.export.signature import TwinMLPSignature
from neuralabi.ir.graph import TensorNode, node_ref
from neuralabi.ir.semantic import (
    AxisLineage,
    CanonicalModel,
    PhysicalLayout,
    SemanticAnchor,
    SemanticSlot,
)
from neuralabi.ir.state import StateTensor
from neuralabi.recognize.common import GraphIndex, literal_arg
from neuralabi.status import Evidence, UnsupportedGraphError


@dataclass(frozen=True)
class Projection:
    node_id: str
    branch_node_id: str
    input_node_id: str
    weight_key: str
    weight_shape: tuple[int, ...]
    weight_dtype: str
    orientation: str
    output_size: int
    split_index: int | None
    split_sizes: tuple[int, ...] | None
    split_offset: int
    bias_key: str | None
    bias_shape: tuple[int, ...] | None
    bias_dtype: str | None


def _state_item(artifact: ExportArtifact, key: str) -> StateTensor:
    try:
        return artifact.state_schema.by_key()[key]
    except KeyError as exc:
        raise UnsupportedGraphError(
            f"graph-bound state {key!r} is absent from state schema"
        ) from exc


def _resolve_state_operand(
    index: GraphIndex, node: TensorNode
) -> tuple[str, bool, tuple[str, ...]]:
    direct = index.state.get(node.node_id)
    if direct is not None:
        return direct, False, (node.node_id,)
    if node.canonical_op in {
        "transpose",
        "permute",
        "contiguous",
        "reshape",
        "squeeze",
        "unsqueeze",
    }:
        parent = index.input_node(node, 0)
        key, transposed, support = _resolve_state_operand(index, parent)
        if node.canonical_op == "transpose":
            transposed = not transposed
        elif node.canonical_op == "permute":
            axes = literal_arg(node, 1)
            if axes not in ([1, 0], (1, 0)):
                raise UnsupportedGraphError("only rank-two weight permutation is supported")
            transposed = not transposed
        return key, transposed, (*support, node.node_id)
    raise UnsupportedGraphError(f"node {node.node_id} is not a graph-bound state tensor")


def _branch_projection(
    index: GraphIndex, artifact: ExportArtifact, branch: TensorNode
) -> Projection:
    split_index: int | None = None
    split_sizes: tuple[int, ...] | None = None
    split_offset = 0
    projection_node = branch
    if branch.canonical_op == "getitem":
        split = index.input_node(branch, 0)
        if split.canonical_op != "split":
            raise UnsupportedGraphError("branch getitem does not read an explicit split")
        raw_index = literal_arg(branch, 1)
        raw_sizes = literal_arg(split, 1)
        raw_axis = literal_arg(split, 2)
        if not isinstance(raw_index, int) or not isinstance(raw_sizes, list):
            raise UnsupportedGraphError("split index and sizes must be statically exported")
        if raw_axis not in (-1, len(split.outputs[0].shape) - 1 if split.outputs else -1):
            raise UnsupportedGraphError("fused projection must split its last output axis")
        split_sizes = tuple(int(size) for size in raw_sizes)
        if raw_index < 0 or raw_index >= len(split_sizes):
            raise UnsupportedGraphError("split branch index is out of range")
        split_index = raw_index
        split_offset = sum(split_sizes[:raw_index])
        projection_node = index.input_node(split, 0)
    if projection_node.canonical_op not in {"linear", "matmul", "addmm"}:
        raise UnsupportedGraphError(
            f"semantic branch is produced by unsupported op {projection_node.target}"
        )
    bias_key: str | None = None
    input_node: TensorNode
    if projection_node.canonical_op == "linear":
        input_node = index.input_node(projection_node, 0)
        weight_operand = index.input_node(projection_node, 1)
        weight_key, operand_transposed, _ = _resolve_state_operand(index, weight_operand)
        orientation = "in_out" if operand_transposed else "out_in"
        if isinstance(projection_node.args, list) and len(projection_node.args) > 2:
            bias_ref = node_ref(projection_node.args[2])
            if bias_ref is not None:
                bias_key, bias_transposed, _ = _resolve_state_operand(index, index.node(bias_ref))
                if bias_transposed:
                    raise UnsupportedGraphError("linear bias may not be transposed")
    elif projection_node.canonical_op == "matmul":
        input_node = index.input_node(projection_node, 0)
        weight_operand = index.input_node(projection_node, 1)
        weight_key, operand_transposed, _ = _resolve_state_operand(index, weight_operand)
        orientation = "out_in" if operand_transposed else "in_out"
    else:
        bias_operand = index.input_node(projection_node, 0)
        input_node = index.input_node(projection_node, 1)
        weight_operand = index.input_node(projection_node, 2)
        bias_key, bias_transposed, _ = _resolve_state_operand(index, bias_operand)
        weight_key, operand_transposed, _ = _resolve_state_operand(index, weight_operand)
        if bias_transposed:
            raise UnsupportedGraphError("addmm bias may not be transposed")
        orientation = "out_in" if operand_transposed else "in_out"
    weight = _state_item(artifact, weight_key)
    if len(weight.shape) != 2:
        raise UnsupportedGraphError(f"projection weight {weight_key!r} is not rank two")
    total_output = weight.shape[0] if orientation == "out_in" else weight.shape[1]
    output_size = (
        split_sizes[split_index]
        if split_sizes is not None and split_index is not None
        else total_output
    )
    bias = _state_item(artifact, bias_key) if bias_key is not None else None
    if bias is not None and bias.shape != (total_output,):
        raise UnsupportedGraphError(
            f"projection bias {bias.key!r} has incompatible shape {bias.shape}"
        )
    return Projection(
        projection_node.node_id,
        branch.node_id,
        input_node.node_id,
        weight_key,
        weight.shape,
        weight.dtype,
        orientation,
        output_size,
        split_index,
        split_sizes,
        split_offset,
        bias_key,
        bias.shape if bias else None,
        bias.dtype if bias else None,
    )


def _layout_for_branches(
    artifact: ExportArtifact,
    semantic: tuple[tuple[str, Projection], ...],
) -> tuple[PhysicalLayout, ...]:
    groups: dict[str, list[tuple[str, Projection]]] = {}
    for semantic_id, projection in semantic:
        groups.setdefault(projection.weight_key, []).append((semantic_id, projection))
    layouts: list[PhysicalLayout] = []
    for key in sorted(groups):
        pieces = groups[key]
        first = pieces[0][1]
        if any(piece.orientation != first.orientation for _, piece in pieces):
            raise UnsupportedGraphError("one physical weight has inconsistent orientations")
        if first.split_sizes is None:
            if len(pieces) != 1:
                raise UnsupportedGraphError(
                    "one unsplit projection cannot serve multiple semantic branches"
                )
            ordered = pieces
        else:
            if any(piece.split_sizes != first.split_sizes for _, piece in pieces):
                raise UnsupportedGraphError("fused projection branches disagree on split sizes")
            ordered = sorted(pieces, key=lambda item: item[1].split_offset)
            covered = sum(piece.output_size for _, piece in ordered)
            total = sum(first.split_sizes)
            if covered != total:
                raise UnsupportedGraphError("fused physical weight has an uncovered output segment")
        input_size = (
            first.weight_shape[1] if first.orientation == "out_in" else first.weight_shape[0]
        )
        layouts.append(
            PhysicalLayout(
                physical_key=key,
                components=tuple(semantic_id for semantic_id, _ in ordered),
                component_flat_shapes=tuple(
                    (projection.output_size, input_size) for _, projection in ordered
                ),
                orientation=first.orientation,  # type: ignore[arg-type]
                physical_shape=first.weight_shape,
                dtype=first.weight_dtype,
                interleave_groups=None,
                supporting_nodes=tuple(sorted({projection.node_id for _, projection in ordered})),
                axis_lineage=(
                    AxisLineage(
                        "linear-use",
                        first.weight_shape,
                        (sum(projection.output_size for _, projection in ordered), input_size),
                        f"storage orientation inferred as {first.orientation} from graph use",
                    ),
                ),
            )
        )
        bias_groups = {projection.bias_key for _, projection in ordered}
        if bias_groups != {None}:
            if None in bias_groups or len(bias_groups) != 1:
                raise UnsupportedGraphError("fused branches have inconsistent bias state")
            bias_key = next(iter(bias_groups))
            assert bias_key is not None
            bias_item = _state_item(artifact, bias_key)
            layouts.append(
                PhysicalLayout(
                    physical_key=bias_key,
                    components=tuple(f"{semantic_id[:-6]}bias" for semantic_id, _ in ordered),
                    component_flat_shapes=tuple(
                        (projection.output_size,) for _, projection in ordered
                    ),
                    orientation="vector",
                    physical_shape=bias_item.shape,
                    dtype=bias_item.dtype,
                    interleave_groups=None,
                    supporting_nodes=tuple(
                        sorted({projection.node_id for _, projection in ordered})
                    ),
                )
            )
    return tuple(layouts)


def recognize_twin_mlp(artifact: ExportArtifact) -> CanonicalModel:
    index = GraphIndex(artifact.graph)
    silu = index.single("silu")
    gate_branch = index.input_node(silu, 0)
    mul_users = [index.node(user) for user in silu.users if index.node(user).canonical_op == "mul"]
    if len(mul_users) != 1:
        raise UnsupportedGraphError("SiLU must feed exactly one gate/value multiplication")
    multiply = mul_users[0]
    if not isinstance(multiply.args, list) or len(multiply.args) < 2:
        raise UnsupportedGraphError("gate/value multiplication has malformed operands")
    operands = [node_ref(multiply.args[0]), node_ref(multiply.args[1])]
    if silu.node_id not in operands:
        raise UnsupportedGraphError("SiLU result is absent from the gate/value multiplication")
    up_id = operands[1] if operands[0] == silu.node_id else operands[0]
    if up_id is None:
        raise UnsupportedGraphError("up branch is not a tensor node")
    up_branch = index.node(up_id)
    down_candidates = [
        node
        for node in artifact.graph.nodes
        if node.canonical_op in {"linear", "matmul", "addmm"}
        and multiply.node_id in [node_ref(arg) for arg in node.args]
        if isinstance(node.args, list)
    ]
    if len(down_candidates) != 1:
        raise UnsupportedGraphError("gate/value product must feed exactly one output projection")
    down_branch = down_candidates[0]
    gate = _branch_projection(index, artifact, gate_branch)
    up = _branch_projection(index, artifact, up_branch)
    down = _branch_projection(index, artifact, down_branch)
    if gate.input_node_id != up.input_node_id:
        raise UnsupportedGraphError("gate and value projections do not share an input")
    if gate.output_size != up.output_size:
        raise UnsupportedGraphError("gate and value branches have different intermediate sizes")
    gate_input = index.node(gate.input_node_id)
    if not gate_input.outputs or not down_branch.outputs:
        raise UnsupportedGraphError("export lacks static tensor metadata for TwinMLP")
    input_size_raw = gate_input.outputs[0].shape[-1]
    output_size_raw = down_branch.outputs[0].shape[-1]
    if not isinstance(input_size_raw, int) or not isinstance(output_size_raw, int):
        raise UnsupportedGraphError("TwinMLP feature dimensions must be static")
    down_input_size = down.weight_shape[1] if down.orientation == "out_in" else down.weight_shape[0]
    if down_input_size != gate.output_size:
        raise UnsupportedGraphError("output projection input does not match intermediate size")
    bias_flags = [gate.bias_key is not None, up.bias_key is not None, down.bias_key is not None]
    if len(set(bias_flags)) != 1:
        raise UnsupportedGraphError("TwinMLP requires a consistent optional bias configuration")
    slots = [
        SemanticSlot(
            "model.mlp.gate.weight",
            (gate.output_size, input_size_raw),
            (gate.output_size, input_size_raw),
            gate.weight_dtype,
            (gate.node_id, gate.branch_node_id, silu.node_id),
            Evidence.PROVEN_BY_STRUCTURE,
        ),
        SemanticSlot(
            "model.mlp.up.weight",
            (up.output_size, input_size_raw),
            (up.output_size, input_size_raw),
            up.weight_dtype,
            (up.node_id, up.branch_node_id, multiply.node_id),
            Evidence.PROVEN_BY_STRUCTURE,
        ),
        SemanticSlot(
            "model.mlp.down.weight",
            (output_size_raw, gate.output_size),
            (output_size_raw, gate.output_size),
            down.weight_dtype,
            (down.node_id, multiply.node_id),
            Evidence.PROVEN_BY_STRUCTURE,
        ),
    ]
    semantic_projections = (
        ("model.mlp.gate.weight", gate),
        ("model.mlp.up.weight", up),
        ("model.mlp.down.weight", down),
    )
    if all(bias_flags):
        for semantic_id, projection in semantic_projections:
            assert projection.bias_dtype is not None
            slots.append(
                SemanticSlot(
                    f"{semantic_id[:-6]}bias",
                    (projection.output_size,),
                    (projection.output_size,),
                    projection.bias_dtype,
                    (projection.node_id,),
                    Evidence.PROVEN_BY_STRUCTURE,
                )
            )
    layouts = _layout_for_branches(artifact, semantic_projections)
    recognized = {layout.physical_key for layout in layouts}
    schema = artifact.state_schema
    alias_groups: dict[str, list[str]] = {
        item.alias_group: [] for item in schema.tensors if item.alias_group is not None
    }
    for item in schema.tensors:
        if item.alias_group is not None:
            alias_groups[item.alias_group].append(item.key)
    unexplained = []
    for item in schema.tensors:
        if item.key in recognized:
            continue
        peers = alias_groups.get(item.alias_group or "", [])
        if not any(peer in recognized for peer in peers):
            unexplained.append(item.key)
    if unexplained:
        raise UnsupportedGraphError(f"unrecognized persistent TwinMLP state: {unexplained}")
    architecture = TwinMLPSignature(
        "twin_mlp", input_size_raw, gate.output_size, output_size_raw, all(bias_flags)
    )
    return CanonicalModel(
        artifact.adapter_id,
        artifact.graph,
        schema,
        architecture,
        tuple(sorted(slots, key=lambda slot: slot.semantic_id)),
        tuple(sorted(layouts, key=lambda layout: layout.physical_key)),
        (
            SemanticAnchor(
                "model.mlp.gate.pre_activation",
                gate.branch_node_id,
                (2, 5, gate.output_size),
                (gate.node_id, gate.branch_node_id, silu.node_id),
            ),
            SemanticAnchor(
                "model.mlp.up.branch",
                up.branch_node_id,
                (2, 5, up.output_size),
                (up.node_id, up.branch_node_id, multiply.node_id),
            ),
            SemanticAnchor(
                "model.mlp.output",
                down.node_id,
                (2, 5, output_size_raw),
                (multiply.node_id, down.node_id),
            ),
        ),
        (),
        (),
    )
