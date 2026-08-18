from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from examples.twin_mlp.models import source_adapter, target_adapter
from neuralabi.apply.engine import apply_to_state
from neuralabi.apply.optimizer import apply_optimizer_checkpoint
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import capture_adapter
from neuralabi.formats.optimizer import (
    OPTIMIZER_MANIFEST,
    load_optimizer_bundle,
    write_optimizer_bundle,
)
from neuralabi.formats.plan import ConversionPlan
from neuralabi.optimizers.torch import export_optimizer_state, load_optimizer_state
from neuralabi.recognize.twin_mlp import recognize_twin_mlp
from neuralabi.status import CheckpointError
from neuralabi.synth.solver import synthesize_plan
from neuralabi.transforms.execute import execute_checked


class TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(3, 3, bias=False)
        self.right = nn.Linear(3, 3, bias=False)
        self.right.weight = self.left.weight
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.left(value) + self.right(value) + self.bias


def _trained(
    algorithm: type[torch.optim.Adam] | type[torch.optim.AdamW] = torch.optim.AdamW,
    **options: object,
) -> tuple[TiedModel, torch.optim.Optimizer]:
    torch.manual_seed(11)
    model = TiedModel()
    options.setdefault("foreach", False)
    optimizer = algorithm(model.parameters(), lr=2e-3, **options)
    for seed in range(2):
        generator = torch.Generator().manual_seed(seed + 30)
        value = torch.randn(4, 3, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        model(value).square().mean().backward()
        optimizer.step()
    return model, optimizer


def _tensor_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()
    }


def _twin_case(
    tmp_path: Path, *, reverse: bool = False, cross_groups: bool = False
) -> tuple[
    ConversionPlan,
    nn.Module,
    torch.optim.Optimizer,
    Path,
]:
    source_side = target_adapter if reverse else source_adapter
    target_side = source_adapter if reverse else target_adapter
    source_capture = capture_adapter(source_side)
    target_capture = capture_adapter(target_side)
    source_canonical = recognize_twin_mlp(source_capture.artifact)
    target_canonical = recognize_twin_mlp(target_capture.artifact)
    parameters = list(source_capture.model.parameters())
    if cross_groups:
        preliminary = synthesize_plan(
            source_canonical,
            target_canonical,
            source_capture.artifact,
            target_capture.artifact,
            checkpoint_fingerprint="preliminary",
        )
        assert preliminary.optimizer_mapping is not None
        fused = next(
            item
            for item in preliminary.optimizer_mapping.target_parameter_identities
            if len(item.source_identities) > 1
        )
        source_records = {
            item.identity: item
            for item in preliminary.optimizer_mapping.source_parameter_identities
        }
        named = dict(source_capture.model.named_parameters(remove_duplicate=False))
        isolated = named[source_records[fused.source_identities[0]].model_keys[0]]
        groups: list[dict[str, object]] = [
            {"params": [isolated]},
            {"params": [item for item in parameters if item is not isolated]},
        ]
        optimizer = torch.optim.Adam(groups, lr=1e-3, foreach=False)
    else:
        optimizer = torch.optim.Adam(parameters, lr=1e-3, foreach=False)
    for seed in range(2):
        args, kwargs = source_side.example_inputs(seed=seed + 80, device="cpu")
        optimizer.zero_grad(set_to_none=True)
        source_side.select_outputs(source_capture.model(*args, **kwargs)).square().mean().backward()
        optimizer.step()
    source_checkpoint = tmp_path / "source.safetensors"
    save_file(_tensor_state(source_capture.model), source_checkpoint)
    plan = synthesize_plan(
        source_canonical,
        target_canonical,
        source_capture.artifact,
        target_capture.artifact,
        checkpoint_fingerprint=open_tensor_store(source_checkpoint).fingerprint(),
    )
    converted = apply_to_state(plan, _tensor_state(source_capture.model))
    target_checkpoint = tmp_path / "target.safetensors"
    save_file(
        {key: value.detach().cpu().contiguous().clone() for key, value in converted.items()},
        target_checkpoint,
    )
    return plan, source_capture.model, optimizer, target_checkpoint


def _expression_ops(value: dict[str, object]) -> set[str]:
    result = {str(value["op"])}
    child = value.get("input")
    if isinstance(child, dict):
        result.update(_expression_ops(child))
    children = value.get("inputs")
    if isinstance(children, list):
        for item in children:
            if isinstance(item, dict):
                result.update(_expression_ops(item))
    return result


@pytest.mark.parametrize("algorithm", [torch.optim.Adam, torch.optim.AdamW])
def test_safe_optimizer_bundle_roundtrip_preserves_alias_identity(
    tmp_path: Path,
    algorithm: type[torch.optim.Adam] | type[torch.optim.AdamW],
) -> None:
    model, optimizer = _trained(algorithm)
    output = tmp_path / "optimizer"
    result = export_optimizer_state(model, optimizer, output, max_shard_size=32)

    assert result.checkpoint_path == output.resolve()
    loaded = load_optimizer_bundle(output)
    assert loaded.bundle.algorithm == ("adam" if algorithm is torch.optim.Adam else "adamw")
    assert any(len(item.model_keys) == 2 for item in loaded.bundle.parameters)
    assert len(loaded.store.keys()) == 3 * len(loaded.bundle.parameters)

    restored_model = TiedModel()
    restored_model.load_state_dict(model.state_dict(), strict=True)
    restored = load_optimizer_state(restored_model, output)
    assert type(restored) is algorithm
    assert len(restored.state) == len(tuple(restored_model.parameters()))
    source_states = sorted(
        optimizer.state.values(), key=lambda state: tuple(state["exp_avg"].shape)
    )
    target_states = sorted(restored.state.values(), key=lambda state: tuple(state["exp_avg"].shape))
    for source, target in zip(source_states, target_states, strict=True):
        assert torch.equal(source["step"], target["step"])
        assert torch.equal(source["exp_avg"], target["exp_avg"])
        assert torch.equal(source["exp_avg_sq"], target["exp_avg_sq"])


def test_export_rejects_missing_and_unknown_state(tmp_path: Path) -> None:
    model = TiedModel()
    optimizer = torch.optim.Adam(model.parameters(), foreach=False)
    with pytest.raises(CheckpointError, match="fully initialized"):
        export_optimizer_state(model, optimizer, tmp_path / "missing")

    model, optimizer = _trained(torch.optim.Adam)
    parameter = next(iter(model.parameters()))
    optimizer.state[parameter]["unknown"] = torch.zeros_like(parameter)
    with pytest.raises(CheckpointError, match="fully initialized"):
        export_optimizer_state(model, optimizer, tmp_path / "unknown")

    model, optimizer = _trained(torch.optim.Adam)
    optimizer.state[nn.Parameter(torch.ones(1))] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros(1),
        "exp_avg_sq": torch.zeros(1),
    }
    with pytest.raises(CheckpointError, match="orphan"):
        export_optimizer_state(model, optimizer, tmp_path / "orphan")


@pytest.mark.parametrize(
    "options,match",
    [
        ({"amsgrad": True}, "amsgrad=true"),
        ({"foreach": True}, "foreach=true"),
    ],
)
def test_export_rejects_unsupported_adam_modes(
    tmp_path: Path, options: dict[str, object], match: str
) -> None:
    model, optimizer = _trained(torch.optim.Adam, **options)
    with pytest.raises(CheckpointError, match=match):
        export_optimizer_state(model, optimizer, tmp_path / "unsupported")


def test_loader_rejects_pickle_and_unknown_manifest_fields(tmp_path: Path) -> None:
    pickle_path = tmp_path / "optimizer.pt"
    pickle_path.write_bytes(b"not loaded")
    with pytest.raises(CheckpointError, match="pickle is unsupported"):
        load_optimizer_bundle(pickle_path)

    model, optimizer = _trained()
    output = tmp_path / "safe"
    export_optimizer_state(model, optimizer, output)
    manifest = output / OPTIMIZER_MANIFEST
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["unknown"] = "rejected"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CheckpointError, match="unknown=.*unknown"):
        load_optimizer_bundle(output)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("schema_version", True, "unsupported optimizer schema version"),
        ("algorithm", [], "unsupported optimizer algorithm"),
    ),
)
def test_loader_rejects_wrong_manifest_scalar_types(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    model, optimizer = _trained()
    output = tmp_path / "safe"
    export_optimizer_state(model, optimizer, output)
    manifest = output / OPTIMIZER_MANIFEST
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data[field] = value
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CheckpointError, match=match):
        load_optimizer_bundle(output)


def test_writer_rejects_invalid_in_memory_hyperparameters(tmp_path: Path) -> None:
    model, optimizer = _trained()
    source = tmp_path / "source"
    export_optimizer_state(model, optimizer, source)
    loaded = load_optimizer_bundle(source)
    first_group = loaded.bundle.parameter_groups[0]
    invalid_group = replace(
        first_group,
        hyperparameters=replace(first_group.hyperparameters, amsgrad=True),
    )
    invalid_bundle = replace(
        loaded.bundle,
        parameter_groups=(invalid_group, *loaded.bundle.parameter_groups[1:]),
        tensor_fingerprint="pending",
        bundle_hash="pending",
    )
    tensors = {key: loaded.store.read(key) for key in loaded.store}
    with pytest.raises(CheckpointError, match="amsgrad=true"):
        write_optimizer_bundle(invalid_bundle, tensors, tmp_path / "rejected")


def test_loader_rejects_orphan_tensor_and_wrong_model_binding(tmp_path: Path) -> None:
    model, optimizer = _trained()
    output = tmp_path / "safe"
    export_optimizer_state(model, optimizer, output)

    changed_model = TiedModel()
    changed_model.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        changed_model.bias.add_(1)
    with pytest.raises(CheckpointError, match="checkpoint fingerprint"):
        load_optimizer_state(changed_model, output)

    state_file = next((output / "state").glob("*.safetensors"))
    from safetensors.torch import load_file, save_file

    tensors = {key: value.clone() for key, value in load_file(state_file).items()}
    tensors["orphan"] = torch.ones(1)
    replacement = state_file.with_name("replacement.safetensors")
    save_file(tensors, replacement)
    replacement.replace(state_file)
    with pytest.raises(CheckpointError, match="unknown=.*orphan"):
        load_optimizer_bundle(output)


def test_export_rejects_incompatible_state_shape(tmp_path: Path) -> None:
    model, optimizer = _trained()
    parameter = next(iter(model.parameters()))
    optimizer.state[parameter]["exp_avg"] = torch.zeros(1)
    with pytest.raises(CheckpointError, match="metadata does not match"):
        export_optimizer_state(model, optimizer, tmp_path / "wrong-shape")


def test_apply_fusion_uses_parameter_expression_exactly(tmp_path: Path) -> None:
    plan, model, optimizer, target_checkpoint = _twin_case(tmp_path)
    source_path = tmp_path / "source-optimizer"
    target_path = tmp_path / "target-optimizer"
    export_optimizer_state(model, optimizer, source_path)
    apply_optimizer_checkpoint(plan, source_path, target_checkpoint, target_path)
    source = load_optimizer_bundle(source_path)
    target = load_optimizer_bundle(target_path)
    assert plan.optimizer_mapping is not None
    fused = next(
        item
        for item in plan.optimizer_mapping.target_parameter_identities
        if len(item.source_identities) > 1
    )
    planned = plan.targets[fused.expression_target]
    assert {"concat", "transpose"} <= _expression_ops(planned.expression.to_dict())
    source_id_by_key = {
        key: item.identity
        for item in plan.optimizer_mapping.source_parameter_identities
        for key in item.model_keys
    }
    for field in ("exp_avg", "exp_avg_sq"):
        sources = {
            key: source.read_state(source_id_by_key[key])[field]
            for key in set(planned.expression.source_keys())
        }
        expected = execute_checked(planned.expression, sources, expected=planned.spec)
        assert torch.equal(expected, target.read_state(fused.identity)[field])


def test_apply_rejects_bitwise_distinct_steps_across_fusion(tmp_path: Path) -> None:
    plan, model, optimizer, target_checkpoint = _twin_case(tmp_path)
    assert plan.optimizer_mapping is not None
    fused = next(
        item
        for item in plan.optimizer_mapping.target_parameter_identities
        if len(item.source_identities) > 1
    )
    source_records = {
        item.identity: item for item in plan.optimizer_mapping.source_parameter_identities
    }
    named = dict(model.named_parameters(remove_duplicate=False))
    dependency_parameters = [
        named[source_records[identity].model_keys[0]] for identity in fused.source_identities
    ]
    for parameter in dependency_parameters:
        optimizer.state[parameter]["step"].fill_(0.0)
    optimizer.state[dependency_parameters[-1]]["step"].fill_(-0.0)
    assert torch.equal(
        optimizer.state[dependency_parameters[0]]["step"],
        optimizer.state[dependency_parameters[-1]]["step"],
    )
    source_path = tmp_path / "source-optimizer"
    export_optimizer_state(model, optimizer, source_path)
    with pytest.raises(CheckpointError, match="unequal Adam steps"):
        apply_optimizer_checkpoint(plan, source_path, target_checkpoint, tmp_path / "rejected")


def test_apply_rejects_fusion_across_parameter_groups(tmp_path: Path) -> None:
    plan, model, optimizer, target_checkpoint = _twin_case(tmp_path, cross_groups=True)
    source_path = tmp_path / "source-optimizer"
    export_optimizer_state(model, optimizer, source_path)
    with pytest.raises(CheckpointError, match="across optimizer parameter groups"):
        apply_optimizer_checkpoint(plan, source_path, target_checkpoint, tmp_path / "rejected")


def test_apply_split_copies_scalar_step_exactly(tmp_path: Path) -> None:
    plan, model, optimizer, target_checkpoint = _twin_case(tmp_path, reverse=True)
    source_path = tmp_path / "source-optimizer"
    target_path = tmp_path / "target-optimizer"
    export_optimizer_state(model, optimizer, source_path)
    apply_optimizer_checkpoint(plan, source_path, target_checkpoint, target_path)
    source = load_optimizer_bundle(source_path)
    target = load_optimizer_bundle(target_path)
    assert plan.optimizer_mapping is not None
    dependency_counts: dict[str, int] = {}
    for item in plan.optimizer_mapping.target_parameter_identities:
        for dependency in item.source_identities:
            dependency_counts[dependency] = dependency_counts.get(dependency, 0) + 1
    split_source = next(identity for identity, count in dependency_counts.items() if count > 1)
    source_step = source.read_state(split_source)["step"]
    split_targets = [
        item
        for item in plan.optimizer_mapping.target_parameter_identities
        if split_source in item.source_identities
    ]
    assert len(split_targets) > 1
    for item in split_targets:
        target_step = target.read_state(item.identity)["step"]
        assert target_step.dtype == source_step.dtype
        assert torch.equal(target_step, source_step)
