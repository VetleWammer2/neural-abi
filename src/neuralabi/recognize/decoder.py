"""Structural decoder-only transformer semantic recognizer.

Roles are recovered from attention and SwiGLU dataflow. Physical keys are consulted only after a
graph use has established the role and storage orientation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from neuralabi.export.artifact import ExportArtifact
from neuralabi.export.signature import DecoderSignature
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
from neuralabi.recognize.twin_mlp import _resolve_state_operand
from neuralabi.status import Evidence, UnsupportedGraphError


@dataclass(frozen=True)
class ProjectionUse:
    node_id: str
    input_node_id: str
    weight_key: str
    weight_shape: tuple[int, ...]
    weight_dtype: str
    orientation: Literal["out_in", "in_out"]
    bias_key: str | None
    bias_shape: tuple[int, ...] | None
    bias_dtype: str | None

    @property
    def input_size(self) -> int:
        matrix = tuple(size for size in self.weight_shape if size != 1)
        return matrix[1] if self.orientation == "out_in" else matrix[0]

    @property
    def output_size(self) -> int:
        matrix = tuple(size for size in self.weight_shape if size != 1)
        return matrix[0] if self.orientation == "out_in" else matrix[1]


@dataclass(frozen=True)
class BranchUse:
    projection: ProjectionUse
    branch_node_id: str
    split_index: int | None
    split_sizes: tuple[int, ...] | None
    interleave_groups: int | None
    supporting_nodes: tuple[str, ...]


def _state(artifact: ExportArtifact, key: str) -> StateTensor:
    try:
        return artifact.state_schema.by_key()[key]
    except KeyError as exc:
        raise UnsupportedGraphError(
            f"graph-bound state {key!r} is absent from persistent schema"
        ) from exc


def _projection(index: GraphIndex, artifact: ExportArtifact, node: TensorNode) -> ProjectionUse:
    if node.canonical_op not in {"linear", "matmul", "addmm"}:
        raise UnsupportedGraphError(f"node {node.node_id} is not a supported projection")
    bias_key: str | None = None
    if node.canonical_op == "linear":
        input_node = index.input_node(node, 0)
        operand = index.input_node(node, 1)
        weight_key, transposed, _ = _resolve_state_operand(index, operand)
        orientation: Literal["out_in", "in_out"] = "in_out" if transposed else "out_in"
        if (
            isinstance(node.args, list)
            and len(node.args) > 2
            and (bias_ref := node_ref(node.args[2]))
        ):
            bias_key, bias_transposed, _ = _resolve_state_operand(index, index.node(bias_ref))
            if bias_transposed:
                raise UnsupportedGraphError("projection bias may not be transposed")
    elif node.canonical_op == "matmul":
        input_node = index.input_node(node, 0)
        operand = index.input_node(node, 1)
        weight_key, transposed, _ = _resolve_state_operand(index, operand)
        orientation = "out_in" if transposed else "in_out"
    else:
        bias_operand = index.input_node(node, 0)
        input_node = index.input_node(node, 1)
        operand = index.input_node(node, 2)
        bias_key, bias_transposed, _ = _resolve_state_operand(index, bias_operand)
        weight_key, transposed, _ = _resolve_state_operand(index, operand)
        if bias_transposed:
            raise UnsupportedGraphError("projection bias may not be transposed")
        orientation = "out_in" if transposed else "in_out"
    weight = _state(artifact, weight_key)
    if len(weight.shape) < 2:
        raise UnsupportedGraphError(f"projection weight {weight_key!r} has rank below two")
    squeezed = tuple(size for size in weight.shape if size != 1)
    if len(squeezed) != 2:
        raise UnsupportedGraphError(
            f"projection weight {weight_key!r} has unsupported non-singleton shape {weight.shape}"
        )
    bias = _state(artifact, bias_key) if bias_key is not None else None
    if bias is not None and bias.numel != (squeezed[0] if orientation == "out_in" else squeezed[1]):
        raise UnsupportedGraphError(f"projection bias {bias.key!r} has the wrong size")
    return ProjectionUse(
        node.node_id,
        input_node.node_id,
        weight_key,
        weight.shape,
        weight.dtype,
        orientation,
        bias_key,
        bias.shape if bias else None,
        bias.dtype if bias else None,
    )


def _try_projection(
    index: GraphIndex, artifact: ExportArtifact, node: TensorNode
) -> ProjectionUse | None:
    if node.canonical_op not in {"linear", "matmul", "addmm"}:
        return None
    try:
        return _projection(index, artifact, node)
    except UnsupportedGraphError:
        return None


def _nearest_projection_ancestor(
    index: GraphIndex, artifact: ExportArtifact, start: str
) -> ProjectionUse:
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    matches: list[tuple[int, ProjectionUse]] = []
    best_depth: int | None = None
    while queue:
        node_id, depth = queue.popleft()
        if best_depth is not None and depth > best_depth:
            break
        node = index.node(node_id)
        if projection := _try_projection(index, artifact, node):
            best_depth = depth
            matches.append((depth, projection))
            continue
        if isinstance(node.args, list):
            for argument in node.args:
                if (parent := node_ref(argument)) and parent not in seen:
                    seen.add(parent)
                    queue.append((parent, depth + 1))
    unique = {item.node_id: item for _, item in matches}
    if len(unique) != 1:
        raise UnsupportedGraphError(
            f"expected one nearest state projection upstream of {start}, found {sorted(unique)}"
        )
    return next(iter(unique.values()))


def _nearest_operation_ancestor(index: GraphIndex, start: str, operation: str) -> TensorNode:
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    matches: list[TensorNode] = []
    best_depth: int | None = None
    while queue:
        node_id, depth = queue.popleft()
        if best_depth is not None and depth > best_depth:
            break
        node = index.node(node_id)
        if node.canonical_op == operation:
            best_depth = depth
            matches.append(node)
            continue
        if isinstance(node.args, list):
            for argument in node.args:
                if (parent := node_ref(argument)) and parent not in seen:
                    seen.add(parent)
                    queue.append((parent, depth + 1))
    unique = {node.node_id: node for node in matches}
    if len(unique) != 1:
        raise UnsupportedGraphError(
            f"expected one nearest {operation} upstream of {start}, found {sorted(unique)}"
        )
    return next(iter(unique.values()))


def _nearest_operation_descendant(index: GraphIndex, start: str, operation: str) -> TensorNode:
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    matches: list[TensorNode] = []
    best_depth: int | None = None
    while queue:
        node_id, depth = queue.popleft()
        if best_depth is not None and depth >= best_depth:
            continue
        for user_id in index.node(node_id).users:
            if user_id in seen:
                continue
            seen.add(user_id)
            user = index.node(user_id)
            if user.canonical_op == operation:
                best_depth = depth + 1
                matches.append(user)
            else:
                queue.append((user_id, depth + 1))
    unique = {node.node_id: node for node in matches}
    if len(unique) != 1:
        raise UnsupportedGraphError(
            f"expected one nearest {operation} downstream of {start}, found {sorted(unique)}"
        )
    return next(iter(unique.values()))


def _nearest_projection_descendant(
    index: GraphIndex,
    artifact: ExportArtifact,
    start: str,
    *,
    before_index: int | None = None,
) -> ProjectionUse:
    positions = {node.node_id: position for position, node in enumerate(index.program.nodes)}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    matches: list[tuple[int, ProjectionUse]] = []
    best_depth: int | None = None
    while queue:
        node_id, depth = queue.popleft()
        if best_depth is not None and depth > best_depth:
            break
        for user_id in index.node(node_id).users:
            if user_id in seen or (before_index is not None and positions[user_id] >= before_index):
                continue
            seen.add(user_id)
            user = index.node(user_id)
            if projection := _try_projection(index, artifact, user):
                best_depth = depth + 1
                matches.append((depth + 1, projection))
            else:
                queue.append((user_id, depth + 1))
    unique = {item.node_id: item for _, item in matches}
    if len(unique) != 1:
        raise UnsupportedGraphError(
            f"expected one nearest state projection downstream of {start}, found {sorted(unique)}"
        )
    return next(iter(unique.values()))


def _branch(
    index: GraphIndex,
    projection: ProjectionUse,
    role_root: str,
) -> BranchUse:
    ancestors = index.ancestors(role_root) | {role_root}
    projection_descendants = index.descendants(projection.node_id)
    candidates: list[tuple[int, TensorNode, TensorNode]] = []
    positions = {node.node_id: position for position, node in enumerate(index.program.nodes)}
    for node_id in ancestors & projection_descendants:
        node = index.node(node_id)
        if node.canonical_op != "getitem":
            continue
        split = index.input_node(node, 0)
        if split.canonical_op != "split" or projection.node_id not in index.ancestors(
            split.node_id
        ):
            continue
        candidates.append((positions[node_id], node, split))
    if not candidates:
        return BranchUse(projection, projection.node_id, None, None, None, (projection.node_id,))
    _, getitem, split = min(candidates)
    raw_index = literal_arg(getitem, 1)
    raw_sizes = literal_arg(split, 1)
    raw_axis = literal_arg(split, 2)
    if (
        not isinstance(raw_index, int)
        or not isinstance(raw_sizes, list)
        or not all(isinstance(size, int) for size in raw_sizes)
    ):
        raise UnsupportedGraphError("fused projection split is not statically known")
    split_sizes = tuple(raw_sizes)
    split_input = index.input_node(split, 0)
    interleave_groups: int | None = None
    if split_input.node_id != projection.node_id:
        if split_input.canonical_op != "reshape" or not split_input.outputs:
            raise UnsupportedGraphError("unsupported operation between fused projection and split")
        shape = split_input.outputs[0].shape
        normalized_axis = raw_axis + len(shape) if raw_axis < 0 else raw_axis
        if len(shape) != 5 or normalized_axis != 3 or not isinstance(shape[2], int):
            raise UnsupportedGraphError(
                "head interleaving requires [batch, sequence, groups, pieces, head_dim]"
            )
        interleave_groups = shape[2]
    return BranchUse(
        projection,
        getitem.node_id,
        raw_index,
        split_sizes,
        interleave_groups,
        (projection.node_id, split_input.node_id, split.node_id, getitem.node_id),
    )


def _canonical_head_node(
    index: GraphIndex,
    projection: ProjectionUse,
    role_root: str,
    *,
    batch: int,
    heads: int,
    sequence: int,
    head_dim: int,
) -> str:
    positions = {node.node_id: position for position, node in enumerate(index.program.nodes)}
    candidates: list[tuple[int, str]] = []
    path = (index.ancestors(role_root) | {role_root}) & index.descendants(projection.node_id)
    for node_id in path:
        node = index.node(node_id)
        if not node.outputs:
            continue
        shape = node.outputs[0].shape
        if shape == (batch, heads, sequence, head_dim):
            candidates.append((positions[node_id], node_id))
    if not candidates:
        raise UnsupportedGraphError(f"cannot locate canonical head layout upstream of {role_root}")
    return min(candidates)[1]


def _norm_scale(
    index: GraphIndex,
    artifact: ExportArtifact,
    normalized_node_id: str,
) -> tuple[str, tuple[str, ...]]:
    queue: deque[tuple[str, int]] = deque([(normalized_node_id, 0)])
    seen = {normalized_node_id}
    candidates: list[tuple[int, TensorNode]] = []
    best_depth: int | None = None
    while queue:
        node_id, depth = queue.popleft()
        if best_depth is not None and depth > best_depth:
            break
        candidate = index.node(node_id)
        if candidate.canonical_op == "mul" and isinstance(candidate.args, list):
            state_count = sum(
                1
                for argument in candidate.args
                if (reference := node_ref(argument)) is not None
                and reference in index.state
                and len(_state(artifact, index.state[reference]).shape) == 1
            )
            if state_count == 1:
                best_depth = depth
                candidates.append((depth, candidate))
                continue
        if isinstance(candidate.args, list):
            for argument in candidate.args:
                if (parent := node_ref(argument)) and parent not in seen:
                    seen.add(parent)
                    queue.append((parent, depth + 1))
    unique = {candidate.node_id: candidate for _, candidate in candidates}
    if len(unique) != 1:
        raise UnsupportedGraphError(
            f"projection input {normalized_node_id} has no unique RMSNorm scale multiplication"
        )
    node = next(iter(unique.values()))
    state_operands = [
        (reference, index.state[reference])
        for argument in node.args
        if (reference := node_ref(argument)) is not None and reference in index.state
    ]
    if len(state_operands) != 1:
        raise UnsupportedGraphError("RMSNorm output must multiply exactly one graph-bound scale")
    state_node, key = state_operands[0]
    item = _state(artifact, key)
    if len(item.shape) != 1:
        raise UnsupportedGraphError("RMSNorm scale must be rank one")
    non_state_parents = [
        reference
        for argument in node.args
        if (reference := node_ref(argument)) is not None and reference != state_node
    ]
    if len(non_state_parents) != 1 or not any(
        index.node(ancestor).canonical_op == "rsqrt"
        for ancestor in index.ancestors(non_state_parents[0])
    ):
        raise UnsupportedGraphError("RMSNorm scale lacks reciprocal-root variance evidence")
    if not any(
        index.node(ancestor).canonical_op == "mean"
        for ancestor in index.ancestors(non_state_parents[0])
    ):
        raise UnsupportedGraphError("RMSNorm scale lacks mean-square evidence")
    return key, (normalized_node_id, node.node_id, state_node)


def _residual_user(index: GraphIndex, projection_id: str) -> TensorNode:
    users = [index.node(user) for user in index.node(projection_id).users]
    additions = [user for user in users if user.canonical_op == "add"]
    if len(additions) != 1:
        raise UnsupportedGraphError(f"projection {projection_id} must feed one residual addition")
    return additions[0]


def _layout_group(
    artifact: ExportArtifact,
    entries: tuple[tuple[str, tuple[int, ...], BranchUse], ...],
) -> tuple[PhysicalLayout, ...]:
    grouped: dict[str, list[tuple[str, tuple[int, ...], BranchUse]]] = {}
    for entry in entries:
        grouped.setdefault(entry[2].projection.weight_key, []).append(entry)
    layouts: list[PhysicalLayout] = []
    for key in sorted(grouped):
        pieces = grouped[key]
        first = pieces[0][2]
        if first.split_index is None:
            if len(pieces) != 1:
                raise UnsupportedGraphError("unfused projection serves multiple semantic roles")
            ordered = pieces
        else:
            if any(
                branch.split_sizes != first.split_sizes
                or branch.interleave_groups != first.interleave_groups
                for _, _, branch in pieces
            ):
                raise UnsupportedGraphError("fused projection branches disagree on packing")
            ordered = sorted(pieces, key=lambda item: item[2].split_index or 0)
            if len(ordered) != len(first.split_sizes or ()):
                raise UnsupportedGraphError("fused projection contains an unexplained component")
        projection = first.projection
        layouts.append(
            PhysicalLayout(
                key,
                tuple(semantic_id for semantic_id, _, _ in ordered),
                tuple(flat_shape for _, flat_shape, _ in ordered),
                projection.orientation,
                projection.weight_shape,
                projection.weight_dtype,
                first.interleave_groups,
                tuple(
                    sorted({node for _, _, branch in ordered for node in branch.supporting_nodes})
                ),
                (
                    AxisLineage(
                        "projection-use",
                        projection.weight_shape,
                        (
                            sum(shape[0] for _, shape, _ in ordered),
                            projection.input_size,
                        ),
                        f"{projection.orientation}; fused order derived from graph split use",
                    ),
                ),
            )
        )
        bias_keys = {branch.projection.bias_key for _, _, branch in ordered}
        if bias_keys != {None}:
            if None in bias_keys or len(bias_keys) != 1:
                raise UnsupportedGraphError("projection branches disagree on optional bias")
            bias_key = next(iter(bias_keys))
            assert bias_key is not None
            bias = _state(artifact, bias_key)
            layouts.append(
                PhysicalLayout(
                    bias_key,
                    tuple(f"{semantic_id[:-6]}bias" for semantic_id, _, _ in ordered),
                    tuple((shape[0],) for _, shape, _ in ordered),
                    "vector",
                    bias.shape,
                    bias.dtype,
                    first.interleave_groups,
                    tuple(
                        sorted(
                            {node for _, _, branch in ordered for node in branch.supporting_nodes}
                        )
                    ),
                )
            )
    return tuple(layouts)


def _slot(
    semantic_id: str,
    canonical_shape: tuple[int, ...],
    flat_shape: tuple[int, ...],
    dtype: str,
    nodes: tuple[str, ...],
) -> SemanticSlot:
    return SemanticSlot(
        semantic_id,
        canonical_shape,
        flat_shape,
        dtype,
        tuple(sorted(set(nodes))),
        Evidence.PROVEN_BY_STRUCTURE,
    )


def _has_causal_mask(index: GraphIndex, ancestors: set[str], sequence: int) -> bool:
    if any(index.node(node_id).canonical_op == "masked_fill" for node_id in ancestors):
        return True
    for node_id in ancestors:
        node = index.node(node_id)
        if (
            not node.target.startswith("aten.where.")
            or not node.outputs
            or not isinstance(node.args, list)
        ):
            continue
        shape = node.outputs[0].shape
        fill = node.args[2] if len(node.args) > 2 else None
        condition = index.input_node(node, 0)
        if (
            len(shape) == 4
            and shape[-2:] == (sequence, sequence)
            and isinstance(fill, float)
            and fill < -1e20
            and condition.outputs
            and condition.outputs[0].dtype == "bool"
        ):
            return True
    return False


def recognize_decoder(artifact: ExportArtifact) -> CanonicalModel:
    index = GraphIndex(artifact.graph)
    positions = {node.node_id: position for position, node in enumerate(artifact.graph.nodes)}
    softmaxes = [node for node in artifact.graph.nodes if node.canonical_op == "softmax"]
    embeddings = [node for node in artifact.graph.nodes if node.canonical_op == "embedding"]
    silus = [node for node in artifact.graph.nodes if node.canonical_op == "silu"]
    if len(embeddings) != 1 or not softmaxes or len(silus) != len(softmaxes):
        raise UnsupportedGraphError(
            "decoder requires one embedding and equal non-zero attention/SwiGLU block counts"
        )
    embedding = embeddings[0]
    embedding_weight_node = index.input_node(embedding, 0)
    embedding_key, embedding_transposed, _ = _resolve_state_operand(index, embedding_weight_node)
    if embedding_transposed:
        raise UnsupportedGraphError("token embedding table may not be transposed")
    embedding_state = _state(artifact, embedding_key)
    if len(embedding_state.shape) != 2:
        raise UnsupportedGraphError("token embedding table must be rank two")
    vocabulary_size, hidden_size = embedding_state.shape
    if not embedding.outputs or len(embedding.outputs[0].shape) != 3:
        raise UnsupportedGraphError("embedding output must be [batch, sequence, hidden]")
    raw_batch, raw_sequence, raw_embedded_hidden = embedding.outputs[0].shape
    if (
        not isinstance(raw_batch, int)
        or not isinstance(raw_sequence, int)
        or not isinstance(raw_embedded_hidden, int)
        or raw_batch <= 0
        or raw_embedded_hidden != hidden_size
    ):
        raise UnsupportedGraphError(
            "decoder example dimensions must be static and internally consistent"
        )
    batch, sequence = raw_batch, raw_sequence

    slots: list[SemanticSlot] = [
        _slot(
            "model.token_embedding.weight",
            (vocabulary_size, hidden_size),
            (vocabulary_size, hidden_size),
            embedding_state.dtype,
            (embedding.node_id,),
        )
    ]
    layouts: list[PhysicalLayout] = [
        PhysicalLayout(
            embedding_key,
            ("model.token_embedding.weight",),
            ((vocabulary_size, hidden_size),),
            "out_in",
            embedding_state.shape,
            embedding_state.dtype,
            None,
            (embedding.node_id,),
        )
    ]
    anchors: list[SemanticAnchor] = [
        SemanticAnchor(
            "model.embedding.output",
            embedding.node_id,
            (batch, sequence, hidden_size),
            (embedding.node_id,),
        )
    ]
    q_head_count: int | None = None
    kv_head_count: int | None = None
    head_dimension: int | None = None
    intermediate_size: int | None = None
    attention_bias_values: set[bool] = set()
    mlp_bias_values: set[bool] = set()
    layer_outputs: list[str] = []

    for layer, softmax in enumerate(softmaxes):
        ancestors = index.ancestors(softmax.node_id)
        previous_softmax_position = positions[softmaxes[layer - 1].node_id] if layer else -1
        score = _nearest_operation_ancestor(index, softmax.node_id, "matmul")
        if (
            positions[score.node_id] <= previous_softmax_position
            or not score.outputs
            or len(score.outputs[0].shape) != 4
            or score.outputs[0].shape[-1] != sequence
            or score.outputs[0].shape[-2] != sequence
        ):
            raise UnsupportedGraphError(f"layer {layer} has no unique attention score matmul")
        if not _has_causal_mask(index, ancestors, sequence):
            raise UnsupportedGraphError(f"layer {layer} has no structurally visible causal mask")
        if not any(
            index.node(node_id).canonical_op == "sin" for node_id in index.ancestors(score.node_id)
        ) or not any(
            index.node(node_id).canonical_op == "cos" for node_id in index.ancestors(score.node_id)
        ):
            raise UnsupportedGraphError(f"layer {layer} lacks rotary sine/cosine application")
        aggregation = _nearest_operation_descendant(index, softmax.node_id, "matmul")
        if softmax.node_id not in index.ancestors(aggregation.node_id):
            raise UnsupportedGraphError(f"layer {layer} has no unique attention value aggregation")
        q_root = index.input_node(score, 0).node_id
        k_root = index.input_node(score, 1).node_id
        v_root = index.input_node(aggregation, 1).node_id
        q_projection = _nearest_projection_ancestor(index, artifact, q_root)
        k_projection = _nearest_projection_ancestor(index, artifact, k_root)
        v_projection = _nearest_projection_ancestor(index, artifact, v_root)
        if (
            len(
                {q_projection.input_node_id, k_projection.input_node_id, v_projection.input_node_id}
            )
            != 1
        ):
            raise UnsupportedGraphError(
                f"layer {layer} Q/K/V projections do not share normalized input"
            )
        score_shape = score.outputs[0].shape
        heads = score_shape[1]
        q_shape = index.node(q_root).outputs[0].shape
        dimension = q_shape[-1]
        if (
            not isinstance(heads, int)
            or not isinstance(dimension, int)
            or heads * dimension != hidden_size
        ):
            raise UnsupportedGraphError(f"layer {layer} attention head dimensions are inconsistent")
        q_branch = _branch(index, q_projection, q_root)
        k_branch = _branch(index, k_projection, k_root)
        v_branch = _branch(index, v_projection, v_root)
        # The nearest canonical rank-four tensor before GQA repetition reveals the logical KV count.
        kv_candidates: list[int] = []
        for root, projection in ((k_root, k_projection), (v_root, v_projection)):
            path = (index.ancestors(root) | {root}) & index.descendants(projection.node_id)
            logical = [
                node.outputs[0].shape[1]
                for node_id in path
                for node in (index.node(node_id),)
                if node.outputs
                and node.outputs[0].shape[:1] == (batch,)
                and len(node.outputs[0].shape) == 4
                and node.outputs[0].shape[2:] == (sequence, dimension)
                and isinstance(node.outputs[0].shape[1], int)
            ]
            if not logical:
                raise UnsupportedGraphError(f"layer {layer} cannot recover logical KV head count")
            kv_candidates.append(min(logical))
        if len(set(kv_candidates)) != 1:
            raise UnsupportedGraphError(f"layer {layer} K and V head counts disagree")
        kv_heads = kv_candidates[0]
        if heads % kv_heads:
            raise UnsupportedGraphError(f"layer {layer} GQA expansion is not integral")
        q_canonical_node = _canonical_head_node(
            index,
            q_projection,
            q_root,
            batch=batch,
            heads=heads,
            sequence=sequence,
            head_dim=dimension,
        )
        k_canonical_node = _canonical_head_node(
            index,
            k_projection,
            k_root,
            batch=batch,
            heads=kv_heads,
            sequence=sequence,
            head_dim=dimension,
        )
        v_canonical_node = _canonical_head_node(
            index,
            v_projection,
            v_root,
            batch=batch,
            heads=kv_heads,
            sequence=sequence,
            head_dim=dimension,
        )
        if q_head_count is None:
            q_head_count, kv_head_count, head_dimension = heads, kv_heads, dimension
        elif (q_head_count, kv_head_count, head_dimension) != (heads, kv_heads, dimension):
            raise UnsupportedGraphError("attention dimensions change between decoder layers")
        prefix = f"model.layer[{layer}]"
        q_id = f"{prefix}.attention.query.weight"
        k_id = f"{prefix}.attention.key.weight"
        v_id = f"{prefix}.attention.value.weight"
        q_flat, k_flat, v_flat = (
            (heads * dimension, hidden_size),
            (kv_heads * dimension, hidden_size),
            (kv_heads * dimension, hidden_size),
        )
        slots.extend(
            (
                _slot(
                    q_id,
                    (heads, dimension, hidden_size),
                    q_flat,
                    q_projection.weight_dtype,
                    q_branch.supporting_nodes,
                ),
                _slot(
                    k_id,
                    (kv_heads, dimension, hidden_size),
                    k_flat,
                    k_projection.weight_dtype,
                    k_branch.supporting_nodes,
                ),
                _slot(
                    v_id,
                    (kv_heads, dimension, hidden_size),
                    v_flat,
                    v_projection.weight_dtype,
                    v_branch.supporting_nodes,
                ),
            )
        )
        layouts.extend(
            _layout_group(
                artifact,
                ((q_id, q_flat, q_branch), (k_id, k_flat, k_branch), (v_id, v_flat, v_branch)),
            )
        )
        attention_bias_values.update(
            projection.bias_key is not None
            for projection in (q_projection, k_projection, v_projection)
        )
        if q_projection.bias_key is not None:
            slots.extend(
                (
                    _slot(
                        f"{q_id[:-6]}bias",
                        (q_flat[0],),
                        (q_flat[0],),
                        _state(artifact, q_projection.bias_key).dtype,
                        q_branch.supporting_nodes,
                    ),
                    _slot(
                        f"{k_id[:-6]}bias",
                        (k_flat[0],),
                        (k_flat[0],),
                        _state(artifact, k_projection.bias_key or "").dtype,
                        k_branch.supporting_nodes,
                    ),
                    _slot(
                        f"{v_id[:-6]}bias",
                        (v_flat[0],),
                        (v_flat[0],),
                        _state(artifact, v_projection.bias_key or "").dtype,
                        v_branch.supporting_nodes,
                    ),
                )
            )
        anchors.extend(
            (
                SemanticAnchor(
                    f"{prefix}.pre_attention_norm",
                    q_projection.input_node_id,
                    (batch, sequence, hidden_size),
                    (q_projection.input_node_id,),
                ),
                SemanticAnchor(
                    f"{prefix}.attention.query",
                    q_canonical_node,
                    (batch, heads, sequence, dimension),
                    (q_canonical_node,),
                ),
                SemanticAnchor(
                    f"{prefix}.attention.key",
                    k_canonical_node,
                    (batch, kv_heads, sequence, dimension),
                    (k_canonical_node,),
                ),
                SemanticAnchor(
                    f"{prefix}.attention.value",
                    v_canonical_node,
                    (batch, kv_heads, sequence, dimension),
                    (v_canonical_node,),
                ),
                SemanticAnchor(
                    f"{prefix}.attention.scores",
                    score.node_id,
                    tuple(int(x) for x in score_shape),
                    (score.node_id, softmax.node_id),
                ),
            )
        )
        norm_key, norm_nodes = _norm_scale(index, artifact, q_projection.input_node_id)
        norm_state = _state(artifact, norm_key)
        norm_id = f"{prefix}.pre_attention_norm.scale"
        slots.append(_slot(norm_id, (hidden_size,), (hidden_size,), norm_state.dtype, norm_nodes))
        layouts.append(
            PhysicalLayout(
                norm_key,
                (norm_id,),
                ((hidden_size,),),
                "vector",
                norm_state.shape,
                norm_state.dtype,
                None,
                norm_nodes,
            )
        )
        next_softmax_position = (
            positions[softmaxes[layer + 1].node_id] if layer + 1 < len(softmaxes) else None
        )
        output_projection = _nearest_projection_descendant(
            index, artifact, aggregation.node_id, before_index=next_softmax_position
        )
        output_residual = _residual_user(index, output_projection.node_id)
        output_id = f"{prefix}.attention.output.weight"
        output_flat = (hidden_size, heads * dimension)
        output_branch = BranchUse(
            output_projection,
            output_projection.node_id,
            None,
            None,
            None,
            (aggregation.node_id, output_projection.node_id),
        )
        slots.append(
            _slot(
                output_id,
                (hidden_size, heads, dimension),
                output_flat,
                output_projection.weight_dtype,
                output_branch.supporting_nodes,
            )
        )
        layouts.extend(_layout_group(artifact, ((output_id, output_flat, output_branch),)))
        attention_bias_values.add(output_projection.bias_key is not None)
        if output_projection.bias_key is not None:
            bias = _state(artifact, output_projection.bias_key)
            slots.append(
                _slot(
                    f"{output_id[:-6]}bias",
                    (hidden_size,),
                    (hidden_size,),
                    bias.dtype,
                    output_branch.supporting_nodes,
                )
            )
        anchors.extend(
            (
                SemanticAnchor(
                    f"{prefix}.attention.output",
                    output_projection.node_id,
                    (batch, sequence, hidden_size),
                    (aggregation.node_id, output_projection.node_id),
                ),
                SemanticAnchor(
                    f"{prefix}.post_attention_residual",
                    output_residual.node_id,
                    (batch, sequence, hidden_size),
                    (output_residual.node_id,),
                ),
            )
        )
        layer_silus = [
            node
            for node in silus
            if positions[node.node_id] > positions[softmax.node_id]
            and (next_softmax_position is None or positions[node.node_id] < next_softmax_position)
        ]
        if len(layer_silus) != 1:
            raise UnsupportedGraphError(f"layer {layer} has no unique SwiGLU activation")
        silu = layer_silus[0]
        gate_root = index.input_node(silu, 0).node_id
        multiply_users = [
            index.node(user) for user in silu.users if index.node(user).canonical_op == "mul"
        ]
        if len(multiply_users) != 1:
            raise UnsupportedGraphError(
                f"layer {layer} SwiGLU gate does not feed one multiplication"
            )
        swiglu_mul = multiply_users[0]
        if not isinstance(swiglu_mul.args, list):
            raise UnsupportedGraphError("SwiGLU multiplication operands are malformed")
        operand_refs = [node_ref(item) for item in swiglu_mul.args]
        up_root = operand_refs[1] if operand_refs[0] == silu.node_id else operand_refs[0]
        if up_root is None:
            raise UnsupportedGraphError("SwiGLU up branch is not a tensor")
        gate_projection = _nearest_projection_ancestor(index, artifact, gate_root)
        up_projection = _nearest_projection_ancestor(index, artifact, up_root)
        if gate_projection.input_node_id != up_projection.input_node_id:
            raise UnsupportedGraphError(
                f"layer {layer} SwiGLU branches do not share normalized input"
            )
        gate_branch = _branch(index, gate_projection, gate_root)
        up_branch = _branch(index, up_projection, up_root)
        gate_node = index.node(gate_branch.branch_node_id)
        up_node = index.node(up_branch.branch_node_id)
        if not gate_node.outputs or not up_node.outputs:
            raise UnsupportedGraphError("SwiGLU branches lack static output metadata")
        layer_intermediate = gate_node.outputs[0].shape[-1]
        if (
            not isinstance(layer_intermediate, int)
            or up_node.outputs[0].shape[-1] != layer_intermediate
        ):
            raise UnsupportedGraphError("SwiGLU branch sizes disagree")
        if intermediate_size is None:
            intermediate_size = layer_intermediate
        elif intermediate_size != layer_intermediate:
            raise UnsupportedGraphError("intermediate size changes between decoder layers")
        gate_id = f"{prefix}.mlp.gate.weight"
        up_id = f"{prefix}.mlp.up.weight"
        mlp_flat = (layer_intermediate, hidden_size)
        slots.extend(
            (
                _slot(
                    gate_id,
                    mlp_flat,
                    mlp_flat,
                    gate_projection.weight_dtype,
                    gate_branch.supporting_nodes,
                ),
                _slot(
                    up_id,
                    mlp_flat,
                    mlp_flat,
                    up_projection.weight_dtype,
                    up_branch.supporting_nodes,
                ),
            )
        )
        layouts.extend(
            _layout_group(
                artifact,
                ((gate_id, mlp_flat, gate_branch), (up_id, mlp_flat, up_branch)),
            )
        )
        mlp_bias_values.update(
            projection.bias_key is not None for projection in (gate_projection, up_projection)
        )
        if gate_projection.bias_key is not None:
            slots.extend(
                (
                    _slot(
                        f"{gate_id[:-6]}bias",
                        (layer_intermediate,),
                        (layer_intermediate,),
                        _state(artifact, gate_projection.bias_key).dtype,
                        gate_branch.supporting_nodes,
                    ),
                    _slot(
                        f"{up_id[:-6]}bias",
                        (layer_intermediate,),
                        (layer_intermediate,),
                        _state(artifact, up_projection.bias_key or "").dtype,
                        up_branch.supporting_nodes,
                    ),
                )
            )
        mlp_norm_key, mlp_norm_nodes = _norm_scale(index, artifact, gate_projection.input_node_id)
        mlp_norm_state = _state(artifact, mlp_norm_key)
        mlp_norm_id = f"{prefix}.pre_mlp_norm.scale"
        slots.append(
            _slot(mlp_norm_id, (hidden_size,), (hidden_size,), mlp_norm_state.dtype, mlp_norm_nodes)
        )
        layouts.append(
            PhysicalLayout(
                mlp_norm_key,
                (mlp_norm_id,),
                ((hidden_size,),),
                "vector",
                mlp_norm_state.shape,
                mlp_norm_state.dtype,
                None,
                mlp_norm_nodes,
            )
        )
        down_projection = _nearest_projection_descendant(
            index, artifact, swiglu_mul.node_id, before_index=next_softmax_position
        )
        down_residual = _residual_user(index, down_projection.node_id)
        down_id = f"{prefix}.mlp.down.weight"
        down_flat = (hidden_size, layer_intermediate)
        down_branch = BranchUse(
            down_projection,
            down_projection.node_id,
            None,
            None,
            None,
            (swiglu_mul.node_id, down_projection.node_id),
        )
        slots.append(
            _slot(
                down_id,
                down_flat,
                down_flat,
                down_projection.weight_dtype,
                down_branch.supporting_nodes,
            )
        )
        layouts.extend(_layout_group(artifact, ((down_id, down_flat, down_branch),)))
        mlp_bias_values.add(down_projection.bias_key is not None)
        if down_projection.bias_key is not None:
            bias = _state(artifact, down_projection.bias_key)
            slots.append(
                _slot(
                    f"{down_id[:-6]}bias",
                    (hidden_size,),
                    (hidden_size,),
                    bias.dtype,
                    down_branch.supporting_nodes,
                )
            )
        anchors.extend(
            (
                SemanticAnchor(
                    f"{prefix}.pre_mlp_norm",
                    gate_projection.input_node_id,
                    (batch, sequence, hidden_size),
                    (gate_projection.input_node_id,),
                ),
                SemanticAnchor(
                    f"{prefix}.mlp.gate_input",
                    gate_branch.branch_node_id,
                    (batch, sequence, layer_intermediate),
                    gate_branch.supporting_nodes,
                ),
                SemanticAnchor(
                    f"{prefix}.mlp.up_branch",
                    up_branch.branch_node_id,
                    (batch, sequence, layer_intermediate),
                    up_branch.supporting_nodes,
                ),
                SemanticAnchor(
                    f"{prefix}.mlp.output",
                    down_projection.node_id,
                    (batch, sequence, hidden_size),
                    down_branch.supporting_nodes,
                ),
                SemanticAnchor(
                    f"{prefix}.output",
                    down_residual.node_id,
                    (batch, sequence, hidden_size),
                    (down_residual.node_id,),
                ),
            )
        )
        layer_outputs.append(down_residual.node_id)

    assert q_head_count is not None and kv_head_count is not None and head_dimension is not None
    assert intermediate_size is not None
    if len(attention_bias_values) != 1 or len(mlp_bias_values) != 1:
        raise UnsupportedGraphError("decoder has inconsistent optional bias configuration")
    # The final RMSNorm feeds the last vocabulary projection.
    output_projections: list[ProjectionUse] = []
    for node in artifact.graph.nodes:
        if not node.outputs or node.outputs[0].shape[-1:] != (vocabulary_size,):
            continue
        candidate_projection = _try_projection(index, artifact, node)
        if candidate_projection is not None:
            output_projections.append(candidate_projection)
    if len(output_projections) != 1:
        raise UnsupportedGraphError("decoder has no unique language-model output projection")
    output_projection = output_projections[0]
    final_norm_key, final_norm_nodes = _norm_scale(index, artifact, output_projection.input_node_id)
    final_norm_state = _state(artifact, final_norm_key)
    final_norm_id = "model.final_norm.scale"
    slots.append(
        _slot(
            final_norm_id, (hidden_size,), (hidden_size,), final_norm_state.dtype, final_norm_nodes
        )
    )
    layouts.append(
        PhysicalLayout(
            final_norm_key,
            (final_norm_id,),
            ((hidden_size,),),
            "vector",
            final_norm_state.shape,
            final_norm_state.dtype,
            None,
            final_norm_nodes,
        )
    )
    output_id = "model.output_head.weight"
    output_flat = (vocabulary_size, hidden_size)
    output_branch = BranchUse(
        output_projection, output_projection.node_id, None, None, None, (output_projection.node_id,)
    )
    slots.append(
        _slot(
            output_id,
            output_flat,
            output_flat,
            output_projection.weight_dtype,
            output_branch.supporting_nodes,
        )
    )
    layouts.extend(_layout_group(artifact, ((output_id, output_flat, output_branch),)))
    anchors.extend(
        (
            SemanticAnchor(
                "model.final_norm.output",
                output_projection.input_node_id,
                (batch, sequence, hidden_size),
                final_norm_nodes,
            ),
            SemanticAnchor(
                "model.logits",
                output_projection.node_id,
                (batch, sequence, vocabulary_size),
                (output_projection.node_id,),
            ),
        )
    )
    alias_by_key = {item.key: item.alias_group for item in artifact.state_schema.tensors}
    tied_embeddings = output_projection.weight_key == embedding_key or (
        alias_by_key.get(output_projection.weight_key) is not None
        and alias_by_key.get(output_projection.weight_key) == alias_by_key.get(embedding_key)
    )
    architecture = DecoderSignature(
        "decoder",
        vocabulary_size,
        hidden_size,
        len(softmaxes),
        q_head_count,
        kv_head_count,
        head_dimension,
        intermediate_size,
        "rmsnorm",
        "swiglu",
        "rotary",
        next(iter(attention_bias_values)),
        next(iter(mlp_bias_values)),
        tied_embeddings,
    )
    recognized = {layout.physical_key for layout in layouts}
    alias_groups: dict[str, list[str]] = {}
    for item in artifact.state_schema.tensors:
        if item.alias_group is not None:
            alias_groups.setdefault(item.alias_group, []).append(item.key)
    unexplained = [
        item.key
        for item in artifact.state_schema.tensors
        if item.key not in recognized
        and not any(peer in recognized for peer in alias_groups.get(item.alias_group or "", ()))
    ]
    if unexplained:
        raise UnsupportedGraphError(f"unrecognized persistent decoder state: {sorted(unexplained)}")
    return CanonicalModel(
        artifact.adapter_id,
        artifact.graph,
        artifact.state_schema,
        architecture,
        tuple(sorted(slots, key=lambda item: item.semantic_id)),
        tuple(sorted(layouts, key=lambda item: (item.physical_key, item.components))),
        tuple(sorted(anchors, key=lambda item: item.anchor_id)),
        (),
        (),
    )
