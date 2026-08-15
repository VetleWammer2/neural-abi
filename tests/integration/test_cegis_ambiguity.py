from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from examples.twin_mlp.models import source_adapter, target_adapter
from neuralabi.apply.engine import apply_to_state
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import capture_adapter
from neuralabi.formats.plan import PlannedTensor
from neuralabi.recognize.twin_mlp import recognize_twin_mlp
from neuralabi.status import LinkStatus
from neuralabi.synth.cegis import CandidateCache, CandidateEvidence, refine_candidates
from neuralabi.synth.solver import synthesize_plan


def _plans(tmp_path: Path) -> tuple[object, object, dict[str, torch.Tensor], object, object]:
    source = capture_adapter(source_adapter)
    target = capture_adapter(target_adapter)
    source_state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
    checkpoint = tmp_path / "source.safetensors"
    save_file(source_state, checkpoint)
    store = open_tensor_store(checkpoint)
    base = synthesize_plan(
        recognize_twin_mlp(source.artifact),
        recognize_twin_mlp(target.artifact),
        source.artifact,
        target.artifact,
        checkpoint_fingerprint=store.fingerprint(),
    )
    fused_key = next(key for key, item in base.targets.items() if len(item.semantic_slots) == 2)
    source_views = {
        view.semantic_id: view.decode for view in recognize_twin_mlp(source.artifact).views()
    }
    gate = source_views["model.mlp.gate.weight"]
    up = source_views["model.mlp.up.weight"]
    target_layout = next(
        layout
        for layout in recognize_twin_mlp(target.artifact).layouts
        if layout.physical_key == fused_key
    )
    swapped_expression = target_layout.encode(
        {
            **source_views,
            "model.mlp.gate.weight": up,
            "model.mlp.up.weight": gate,
        }
    )
    targets = dict(base.targets)
    old = targets[fused_key]
    targets[fused_key] = PlannedTensor(swapped_expression, old.shape, old.dtype, old.semantic_slots)
    swapped = replace(base, targets=targets, plan_hash="").with_hash()
    return base, swapped, source_state, source, target


def test_execution_probe_resolves_structural_candidates_and_cache(tmp_path: Path) -> None:
    base, swapped, source_state, source, target = _plans(tmp_path)

    def evaluate(plan: object) -> CandidateEvidence:
        converted = apply_to_state(plan, source_state)  # type: ignore[arg-type]
        target.model.load_state_dict(converted, strict=True)
        source_output = source.model(*source.args)
        target_output = target.model(*target.args)
        passed = torch.allclose(source_output, target_output, rtol=2e-4, atol=2e-5)
        return CandidateEvidence(
            plan.plan_hash,
            passed,
            None if passed else "model.mlp.gate.pre_activation",
            "deterministic seed 0",
        )  # type: ignore[attr-defined]

    cache = CandidateCache()
    result = refine_candidates([swapped, base], evaluate, probe_id="seed=0", cache=cache)  # type: ignore[list-item]
    assert result.status == LinkStatus.UNIQUE
    assert result.survivors[0].plan_hash == base.plan_hash  # type: ignore[attr-defined]
    cached = refine_candidates([base, swapped], evaluate, probe_id="seed=0", cache=cache)  # type: ignore[list-item]
    assert cached.cache_hits == 2


def test_indistinguishable_probe_returns_ambiguous(tmp_path: Path) -> None:
    base, swapped, source_state, source, target = _plans(tmp_path)
    keys = sorted(source_state)
    source_state[keys[1]] = source_state[keys[0]].clone()

    def evaluate(plan: object) -> CandidateEvidence:
        converted = apply_to_state(plan, source_state)  # type: ignore[arg-type]
        target.model.load_state_dict(converted, strict=True)
        source.model.load_state_dict(source_state, strict=True)
        passed = torch.allclose(
            source.model(*source.args), target.model(*target.args), rtol=2e-4, atol=2e-5
        )
        return CandidateEvidence(plan.plan_hash, passed, None, "symmetric checkpoint probe")  # type: ignore[attr-defined]

    result = refine_candidates([base, swapped], evaluate, probe_id="symmetric-seed=0")  # type: ignore[list-item]
    assert result.status == LinkStatus.AMBIGUOUS
    assert len(result.survivors) == 2
