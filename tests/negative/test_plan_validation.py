from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from safetensors.torch import save_file

from examples.twin_mlp.models import source_adapter, target_adapter
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import capture_adapter
from neuralabi.formats.plan import AliasRecord, validate_plan
from neuralabi.recognize.twin_mlp import recognize_twin_mlp
from neuralabi.status import PlanValidationError
from neuralabi.synth.solver import synthesize_plan
from neuralabi.transforms import expr_from_dict


def _plan(tmp_path: Path) -> object:
    source = capture_adapter(source_adapter)
    target = capture_adapter(target_adapter)
    checkpoint = tmp_path / "source.safetensors"
    save_file(
        {key: value.detach().clone() for key, value in source.model.state_dict().items()},
        checkpoint,
    )
    return synthesize_plan(
        recognize_twin_mlp(source.artifact),
        recognize_twin_mlp(target.artifact),
        source.artifact,
        target.artifact,
        checkpoint_fingerprint=open_tensor_store(checkpoint).fingerprint(),
    )


def test_unknown_transform_is_rejected() -> None:
    with pytest.raises(PlanValidationError, match="unknown transform"):
        expr_from_dict({"op": "load_python", "module": "os"})


def test_alias_dependency_cycle_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    first, second = sorted(plan.targets)  # type: ignore[attr-defined]
    cyclic = replace(
        plan,
        aliases=(AliasRecord(first, second), AliasRecord(second, first)),
        plan_hash="",
    ).with_hash()
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_plan(cyclic)


def test_uncovered_target_tensor_is_rejected_without_adapters(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    targets = dict(plan.targets)  # type: ignore[attr-defined]
    targets.pop(sorted(targets)[0])
    uncovered = replace(plan, targets=targets, plan_hash="").with_hash()
    with pytest.raises(PlanValidationError, match="coverage"):
        validate_plan(uncovered)
