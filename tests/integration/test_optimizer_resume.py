from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from examples.generated_transformer.models import (
    GeneratedAdapter,
    LogicalConfig,
    _layer_layout,
)
from examples.twin_mlp.models import source_adapter, target_adapter
from neuralabi.apply.engine import apply_to_state
from neuralabi.apply.optimizer import apply_optimizer_checkpoint
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import CapturedModel, capture_adapter
from neuralabi.formats.optimizer import load_optimizer_bundle, write_optimizer_bundle
from neuralabi.formats.plan import ConversionPlan
from neuralabi.optimizers.torch import export_optimizer_state, load_optimizer_state
from neuralabi.recognize.decoder import recognize_decoder
from neuralabi.recognize.twin_mlp import recognize_twin_mlp
from neuralabi.status import ClaimStatus
from neuralabi.synth.solver import synthesize_plan
from neuralabi.util.pytree import flatten_tensors
from neuralabi.verify.certificate import verify_conversion
from neuralabi.verify.optimizer import verify_optimizer_state
from neuralabi.verify.resume import verify_resumed_training

OptimizerType = type[torch.optim.Adam] | type[torch.optim.AdamW]


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: tensor.detach().cpu().contiguous().clone()
        for key, tensor in model.state_dict().items()
    }


def _write_model_state(state: Mapping[str, torch.Tensor], path: Path) -> None:
    save_file(
        {key: tensor.detach().cpu().contiguous().clone() for key, tensor in state.items()},
        path,
    )


def _device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters())
    return parameter.device


def _warm_up(
    capture: CapturedModel,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    seed_base: int,
) -> None:
    capture.model.eval()
    for step in range(steps):
        args, kwargs = capture.adapter.example_inputs(
            seed=seed_base + step,
            device=_device(capture.model),
        )
        optimizer.zero_grad(set_to_none=True)
        selected = capture.adapter.select_outputs(capture.model(*args, **kwargs))
        tensors = flatten_tensors(selected)
        loss = sum((item.tensor.square().mean() for item in tensors), torch.zeros(()))
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    assert len(optimizer.state) == len(tuple(capture.model.parameters()))
    assert all(float(state["step"]) == float(steps) for state in optimizer.state.values())
    assert all(torch.count_nonzero(state["exp_avg"]) for state in optimizer.state.values())
    assert all(torch.count_nonzero(state["exp_avg_sq"]) for state in optimizer.state.values())


def _optimizer(
    optimizer_type: OptimizerType,
    model: nn.Module,
) -> torch.optim.Optimizer:
    return optimizer_type(
        model.parameters(),
        lr=3e-4,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )


def _prepare_case(
    tmp_path: Path,
    source_capture: CapturedModel,
    target_capture: CapturedModel,
    recognizer: Callable[[Any], Any],
    optimizer_type: OptimizerType,
) -> tuple[
    ConversionPlan,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    Path,
    Path,
]:
    source_optimizer = _optimizer(optimizer_type, source_capture.model)
    _warm_up(source_capture, source_optimizer, steps=3, seed_base=400)
    source_state = _model_state(source_capture.model)
    source_model_path = tmp_path / "source-model.safetensors"
    _write_model_state(source_state, source_model_path)
    source_store = open_tensor_store(source_model_path)
    plan = synthesize_plan(
        recognizer(source_capture.artifact),
        recognizer(target_capture.artifact),
        source_capture.artifact,
        target_capture.artifact,
        checkpoint_fingerprint=source_store.fingerprint(),
        name_hint_mode="off",
    )
    target_state = apply_to_state(plan, source_state)
    target_model_path = tmp_path / "target-model.safetensors"
    _write_model_state(target_state, target_model_path)
    source_result = export_optimizer_state(
        source_capture.model,
        source_optimizer,
        tmp_path / "source-optimizer",
    )
    target_result = apply_optimizer_checkpoint(
        plan,
        source_result.checkpoint_path,
        target_model_path,
        tmp_path / "target-optimizer",
    )
    return (
        plan,
        source_state,
        target_state,
        source_result.checkpoint_path,
        target_result.checkpoint_path,
    )


def _parameters_by_key(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        key: tensor
        for key, tensor in model.state_dict(keep_vars=True).items()
        if isinstance(tensor, nn.Parameter)
    }


def _assert_initial_optimizer_conversion_exact(
    plan: ConversionPlan,
    source_capture: CapturedModel,
    target_capture: CapturedModel,
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    source_optimizer_path: Path,
    target_optimizer_path: Path,
) -> None:
    source_capture.model.load_state_dict(source_state, strict=True)
    target_capture.model.load_state_dict(target_state, strict=True)
    source_optimizer = load_optimizer_state(source_capture.model, source_optimizer_path)
    target_optimizer = load_optimizer_state(target_capture.model, target_optimizer_path)
    source_parameters = _parameters_by_key(source_capture.model)
    target_parameters = _parameters_by_key(target_capture.model)
    mapping = plan.optimizer_mapping
    assert mapping is not None
    assert len(target_optimizer.state) == len(mapping.target_parameter_identities)
    for target_identity in mapping.target_parameter_identities:
        target_key = target_identity.expression_target
        expression = plan.targets[target_key].expression
        dependencies = set(expression.source_keys())
        target_optimizer_state = target_optimizer.state[target_parameters[target_key]]
        source_optimizer_states = [
            source_optimizer.state[source_parameters[key]] for key in dependencies
        ]
        assert source_optimizer_states
        assert all(
            torch.equal(source_optimizer_states[0]["step"], state["step"])
            for state in source_optimizer_states[1:]
        )
        assert torch.equal(target_optimizer_state["step"], source_optimizer_states[0]["step"])
        for field in ("exp_avg", "exp_avg_sq"):
            expected = expression.apply(
                {key: source_optimizer.state[source_parameters[key]][field] for key in dependencies}
            )
            assert torch.equal(expected, target_optimizer_state[field])


@pytest.mark.parametrize(
    "optimizer_type",
    (torch.optim.Adam, torch.optim.AdamW),
    ids=("adam", "adamw"),
)
def test_twin_mlp_optimizer_resume(
    tmp_path: Path,
    optimizer_type: OptimizerType,
) -> None:
    source_capture = capture_adapter(source_adapter)
    target_capture = capture_adapter(target_adapter)
    plan, source_state, target_state, source_optimizer_path, target_optimizer_path = _prepare_case(
        tmp_path,
        source_capture,
        target_capture,
        recognize_twin_mlp,
        optimizer_type,
    )
    expression_ops = [str(item.expression.to_dict()) for item in plan.targets.values()]
    assert any("concat" in item and "transpose" in item for item in expression_ops)
    _assert_initial_optimizer_conversion_exact(
        plan,
        source_capture,
        target_capture,
        source_state,
        target_state,
        source_optimizer_path,
        target_optimizer_path,
    )
    evidence = verify_resumed_training(
        plan,
        source_capture,
        target_capture,
        source_state,
        target_state,
        source_optimizer_path,
        target_optimizer_path,
        steps=2,
        seed_base=900,
    )
    assert evidence.optimizer_algorithm == (
        "adam" if optimizer_type is torch.optim.Adam else "adamw"
    )
    assert evidence.passed
    assert len(evidence.steps) == 2
    assert all(step.loss.passed and step.outputs and step.model_state for step in evidence.steps)
    if optimizer_type is torch.optim.AdamW:
        target_fingerprint = load_optimizer_bundle(
            target_optimizer_path
        ).bundle.model_binding.checkpoint_fingerprint
        certificate = verify_conversion(
            plan,
            source_capture,
            target_capture,
            recognize_twin_mlp(source_capture.artifact),
            recognize_twin_mlp(target_capture.artifact),
            source_state,
            target_state,
            seeds=(0,),
            source_checkpoint_fingerprint=plan.source.checkpoint_fingerprint or "",
            target_checkpoint_fingerprint=target_fingerprint,
            source_optimizer_state=source_optimizer_path,
            target_optimizer_state=target_optimizer_path,
            resume_steps=1,
        )
        claims = {claim.claim: claim.status for claim in certificate.claims}
        assert certificate.verification_outcome == ClaimStatus.VERIFIED
        assert claims["OPTIMIZER_STATE_CONVERTED"] == ClaimStatus.VERIFIED
        assert claims["OPTIMIZER_COVERAGE_COMPLETE"] == ClaimStatus.VERIFIED
        assert claims["OPTIMIZER_STATE_VERIFIED"] == ClaimStatus.VERIFIED
        assert claims["RESUMED_TRAINING_EQUIVALENT"] == ClaimStatus.VERIFIED
        assert certificate.conversion_scope["optimizer_state"]
        assert certificate.optimizer_state_results
        assert all(item["exact"] for item in certificate.optimizer_state_results)
        assert len(certificate.resume_results) == 1


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("float32", "float64"))
def test_generated_transformer_optimizer_resume(tmp_path: Path, dtype: torch.dtype) -> None:
    config = LogicalConfig(
        seed=101,
        layout_seed=0,
        layers=1,
        vocabulary_size=31,
        hidden_size=16,
        attention_heads=2,
        key_value_heads=1,
        intermediate_size=24,
        sequence_length=3,
        bias=False,
    )
    layout = _layer_layout(config.layout_seed, 0)
    assert layout.qkv_order == ("b", "c", "a")
    assert layout.interleaved
    assert not layout.gate_first
    assert layout.qkv_singleton and layout.mlp_singleton
    assert config.attention_heads != config.key_value_heads
    source_capture = capture_adapter(GeneratedAdapter(config, "source"), dtype=dtype)
    target_capture = capture_adapter(GeneratedAdapter(config, "target"), dtype=dtype)
    plan, source_state, target_state, source_optimizer_path, target_optimizer_path = _prepare_case(
        tmp_path,
        source_capture,
        target_capture,
        recognize_decoder,
        torch.optim.AdamW,
    )
    assert plan.aliases
    qkv_targets = [
        target
        for target in plan.targets.values()
        if len(target.semantic_slots) == 3 and ".attention." in target.semantic_slots[0]
    ]
    assert len(qkv_targets) == 1
    expression_text = str(qkv_targets[0].expression.to_dict())
    assert "concat" in expression_text
    assert "interleave" in expression_text
    assert "transpose" in expression_text
    assert "reshape" in expression_text
    _assert_initial_optimizer_conversion_exact(
        plan,
        source_capture,
        target_capture,
        source_state,
        target_state,
        source_optimizer_path,
        target_optimizer_path,
    )
    evidence = verify_resumed_training(
        plan,
        source_capture,
        target_capture,
        source_state,
        target_state,
        source_optimizer_path,
        target_optimizer_path,
        steps=2,
        seed_base=1200,
    )
    assert evidence.optimizer_algorithm == "adamw"
    assert evidence.passed
    assert len(evidence.steps) == 2
    serialized = evidence.to_dict()
    contract = serialized["numerical_contract"]
    assert contract["version"] == "neuralabi.resumed-training.v1"
    assert contract["execution"]["model_mode"] == "eval"
    assert contract["execution"]["source_backend"] == "cpu"
    assert contract["execution"]["target_backend"] == "cpu"
    assert contract["objective"]["id"] == "selected-output-seeded-linear-probe-v1"
    assert contract["comparison_timing"] == {
        "loss": "pre_update",
        "outputs": "post_update",
        "model_state": "post_update",
        "model_state_reference": "plan(updated_source_state)",
    }
    assert contract["tolerances"] == {
        "float32": {"absolute": 1e-7, "relative": 1e-5},
        "float64": {"absolute": 1e-12, "relative": 1e-10},
    }
    assert contract["steps_requested"] == 2
    assert all("exact" in result for step in serialized["steps"] for result in step["outputs"])


def test_optimizer_tensor_mutation_fails_static_verification(tmp_path: Path) -> None:
    source_capture = capture_adapter(source_adapter)
    target_capture = capture_adapter(target_adapter)
    plan, _, _, source_optimizer_path, target_optimizer_path = _prepare_case(
        tmp_path,
        source_capture,
        target_capture,
        recognize_twin_mlp,
        torch.optim.AdamW,
    )
    loaded = load_optimizer_bundle(target_optimizer_path)
    tensors = {key: loaded.store.read(key) for key in loaded.store}
    first_parameter = loaded.bundle.parameters[0]
    tensors[first_parameter.exp_avg] = tensors[first_parameter.exp_avg].neg()
    mutated = tmp_path / "mutated-target-optimizer"
    write_optimizer_bundle(
        replace(loaded.bundle, tensor_fingerprint="pending", bundle_hash="pending"),
        tensors,
        mutated,
    )
    verification = verify_optimizer_state(
        plan,
        source_optimizer_path,
        mutated,
        target_checkpoint_fingerprint=loaded.bundle.model_binding.checkpoint_fingerprint,
    )
    assert not verification.passed
    failed = [item for item in verification.comparisons if not item.passed]
    assert len(failed) == 1
    assert failed[0].comparison.path.endswith(".exp_avg")
