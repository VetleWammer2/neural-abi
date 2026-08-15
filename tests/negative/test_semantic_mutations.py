from __future__ import annotations

from pathlib import Path

from safetensors.torch import save_file

from examples.generated_transformer.models import GeneratedAdapter, LogicalConfig
from neuralabi.apply.engine import apply_to_state
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import CapturedModel, capture_adapter
from neuralabi.formats.plan import ConversionPlan
from neuralabi.ir.semantic import CanonicalModel
from neuralabi.recognize.decoder import recognize_decoder
from neuralabi.status import ClaimStatus
from neuralabi.synth.solver import synthesize_plan
from neuralabi.verify.certificate import verify_conversion
from neuralabi.verify.mutations import (
    mutate_interleave_groups,
    mutate_semantic_bindings,
    mutate_square_transpose,
)


def _fixture(
    tmp_path: Path, *, gqa: bool = False
) -> tuple[
    ConversionPlan,
    CapturedModel,
    CapturedModel,
    CanonicalModel,
    CanonicalModel,
    dict[str, object],
    str,
]:
    config = LogicalConfig(
        seed=811,
        layout_seed=0,
        layers=2,
        hidden_size=32,
        attention_heads=4,
        key_value_heads=2 if gqa else 4,
        intermediate_size=48,
        sequence_length=7,
    )
    source = capture_adapter(GeneratedAdapter(config, "source"))
    target = capture_adapter(GeneratedAdapter(config, "target"))
    source_model = recognize_decoder(source.artifact)
    target_model = recognize_decoder(target.artifact)
    state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
    path = tmp_path / ("gqa.safetensors" if gqa else "mha.safetensors")
    save_file(state, path)
    fingerprint = open_tensor_store(path).fingerprint()
    plan = synthesize_plan(
        source_model,
        target_model,
        source.artifact,
        target.artifact,
        checkpoint_fingerprint=fingerprint,
    )
    return plan, source, target, source_model, target_model, state, fingerprint


def _target_for(model: CanonicalModel, semantic_fragment: str, *, fused: bool = False) -> str:
    matches = [
        layout.physical_key
        for layout in model.layouts
        if any(semantic_fragment in component for component in layout.components)
        and (not fused or len(layout.components) > 1)
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_rejected(
    plan: ConversionPlan,
    source: CapturedModel,
    target: CapturedModel,
    source_model: CanonicalModel,
    target_model: CanonicalModel,
    state: dict[str, object],
    fingerprint: str,
) -> str:
    converted = apply_to_state(plan, state)  # type: ignore[arg-type]
    certificate = verify_conversion(
        plan,
        source,
        target,
        source_model,
        target_model,
        state,  # type: ignore[arg-type]
        converted,
        seeds=(0,),
        source_checkpoint_fingerprint=fingerprint,
        target_checkpoint_fingerprint="mutated-in-memory",
    )
    assert certificate.verification_outcome == ClaimStatus.FAILED
    assert certificate.first_divergence is not None
    return certificate.first_divergence


def test_shape_compatible_semantic_mutations_are_detected(tmp_path: Path) -> None:
    plan, source, target, source_model, target_model, state, fingerprint = _fixture(tmp_path)
    qkv_key = _target_for(target_model, "layer[0].attention.query.weight", fused=True)
    q = "model.layer[0].attention.query.weight"
    k = "model.layer[0].attention.key.weight"
    v = "model.layer[0].attention.value.weight"
    gate = "model.layer[0].mlp.gate.weight"
    up = "model.layer[0].mlp.up.weight"
    mutations = {
        "q_k_order": mutate_semantic_bindings(
            plan,
            source_model,
            target_model,
            target_key=qkv_key,
            replacements={q: k, k: q},
        ),
        "k_v_order": mutate_semantic_bindings(
            plan,
            source_model,
            target_model,
            target_key=qkv_key,
            replacements={k: v, v: k},
        ),
        "gate_up_order": mutate_semantic_bindings(
            plan,
            source_model,
            target_model,
            target_key=_target_for(target_model, "layer[0].mlp.gate.weight", fused=True),
            replacements={gate: up, up: gate},
        ),
        "transpose_axes": mutate_square_transpose(
            plan,
            target_key=_target_for(target_model, "layer[0].attention.output.weight"),
        ),
        "layer_index": mutate_semantic_bindings(
            plan,
            source_model,
            target_model,
            target_key=qkv_key,
            replacements={
                q: "model.layer[1].attention.query.weight",
                k: "model.layer[1].attention.key.weight",
                v: "model.layer[1].attention.value.weight",
            },
        ),
        "target_dependency": mutate_semantic_bindings(
            plan,
            source_model,
            target_model,
            target_key=_target_for(target_model, "layer[0].pre_attention_norm.scale"),
            replacements={
                "model.layer[0].pre_attention_norm.scale": "model.layer[0].pre_mlp_norm.scale"
            },
        ),
    }
    first_divergences = {
        name: _assert_rejected(
            candidate,
            source,
            target,
            source_model,
            target_model,
            state,
            fingerprint,
        )
        for name, candidate in mutations.items()
    }
    assert all(first_divergences.values())


def test_wrong_gqa_interleave_stride_is_detected(tmp_path: Path) -> None:
    plan, source, target, source_model, target_model, state, fingerprint = _fixture(
        tmp_path, gqa=True
    )
    qkv_key = _target_for(target_model, "layer[0].attention.query.weight", fused=True)
    mutated = mutate_interleave_groups(plan, target_key=qkv_key, groups=4)
    divergence = _assert_rejected(
        mutated,
        source,
        target,
        source_model,
        target_model,
        state,
        fingerprint,
    )
    assert ".attention." in divergence
