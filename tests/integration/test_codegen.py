from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from examples.twin_mlp.models import source_adapter, target_adapter
from neuralabi.apply.engine import apply_checkpoint
from neuralabi.apply.optimizer import apply_optimizer_checkpoint
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.codegen.converter import emit_converter
from neuralabi.export.capture import capture_adapter
from neuralabi.formats.optimizer import load_optimizer_bundle
from neuralabi.formats.plan import plan_from_dict
from neuralabi.optimizers.torch import export_optimizer_state
from neuralabi.recognize.twin_mlp import recognize_twin_mlp
from neuralabi.synth.solver import synthesize_plan
from neuralabi.util.hashing import hash_canonical


def test_standalone_converter_matches_apply(tmp_path: Path) -> None:
    source_capture = capture_adapter(source_adapter)
    target_capture = capture_adapter(target_adapter)
    source_path = tmp_path / "source.safetensors"
    save_file(
        {key: value.detach().clone() for key, value in source_capture.model.state_dict().items()},
        source_path,
    )
    source_store = open_tensor_store(source_path)
    plan = synthesize_plan(
        recognize_twin_mlp(source_capture.artifact),
        recognize_twin_mlp(target_capture.artifact),
        source_capture.artifact,
        target_capture.artifact,
        checkpoint_fingerprint=source_store.fingerprint(),
    )
    converter = tmp_path / "converter.py"
    emit_converter(plan, converter)
    assert "import neuralabi" not in converter.read_text(encoding="utf-8")
    library_output = tmp_path / "library"
    standalone_output = tmp_path / "standalone"
    library = apply_checkpoint(plan, source_path, library_output, max_shard_size=384)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(converter),
            "--source",
            str(source_path),
            "--output",
            str(standalone_output),
            "--max-shard-size",
            "384B",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    standalone = open_tensor_store(standalone_output)
    assert standalone.fingerprint() == library.checkpoint_fingerprint


def test_standalone_converter_rejects_wrong_fingerprint(tmp_path: Path) -> None:
    source_capture = capture_adapter(source_adapter)
    target_capture = capture_adapter(target_adapter)
    source_path = tmp_path / "source.safetensors"
    state = {
        key: value.detach().clone() for key, value in source_capture.model.state_dict().items()
    }
    save_file(state, source_path)
    store = open_tensor_store(source_path)
    plan = synthesize_plan(
        recognize_twin_mlp(source_capture.artifact),
        recognize_twin_mlp(target_capture.artifact),
        source_capture.artifact,
        target_capture.artifact,
        checkpoint_fingerprint=store.fingerprint(),
    )
    converter = tmp_path / "converter.py"
    emit_converter(plan, converter)
    first_key = sorted(state)[0]
    state[first_key] = state[first_key] + 1
    wrong = tmp_path / "wrong.safetensors"
    save_file(state, wrong)
    completed = subprocess.run(
        [sys.executable, str(converter), "--source", str(wrong), "--output", str(tmp_path / "out")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "fingerprint" in completed.stderr


def test_standalone_converter_matches_library_optimizer_conversion(tmp_path: Path) -> None:
    source_capture = capture_adapter(source_adapter)
    target_capture = capture_adapter(target_adapter)
    optimizer = torch.optim.AdamW(
        source_capture.model.parameters(), lr=3e-4, foreach=False, fused=False
    )
    for seed in (71, 72, 73):
        optimizer.zero_grad(set_to_none=True)
        args, kwargs = source_capture.adapter.example_inputs(seed=seed, device="cpu")
        selected = source_capture.adapter.select_outputs(source_capture.model(*args, **kwargs))
        selected.float().square().mean().backward()
        optimizer.step()

    source_path = tmp_path / "source-with-optimizer.safetensors"
    save_file(
        {key: value.detach().clone() for key, value in source_capture.model.state_dict().items()},
        source_path,
    )
    source_store = open_tensor_store(source_path)
    plan = synthesize_plan(
        recognize_twin_mlp(source_capture.artifact),
        recognize_twin_mlp(target_capture.artifact),
        source_capture.artifact,
        target_capture.artifact,
        checkpoint_fingerprint=source_store.fingerprint(),
    )
    source_optimizer = tmp_path / "source-optimizer"
    export_optimizer_state(source_capture.model, optimizer, source_optimizer, max_shard_size=384)

    library_model = tmp_path / "library-model"
    library_optimizer = tmp_path / "library-optimizer"
    apply_checkpoint(plan, source_path, library_model, max_shard_size=384)
    apply_optimizer_checkpoint(
        plan,
        source_optimizer,
        library_model,
        library_optimizer,
        max_shard_size=384,
    )

    converter = tmp_path / "optimizer-converter.py"
    emit_converter(plan, converter)
    standalone_model = tmp_path / "standalone-model"
    standalone_optimizer = tmp_path / "standalone-optimizer"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(converter),
            "--source",
            str(source_path),
            "--output",
            str(standalone_model),
            "--source-optimizer",
            str(source_optimizer),
            "--optimizer-output",
            str(standalone_optimizer),
            "--max-shard-size",
            "384B",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (
        open_tensor_store(standalone_model).fingerprint()
        == open_tensor_store(library_model).fingerprint()
    )
    library = load_optimizer_bundle(library_optimizer)
    standalone = load_optimizer_bundle(standalone_optimizer)
    assert standalone.bundle.tensor_fingerprint == library.bundle.tensor_fingerprint
    assert standalone.bundle.bundle_hash == library.bundle.bundle_hash

    unpaired = subprocess.run(
        [
            sys.executable,
            str(converter),
            "--source",
            str(source_path),
            "--output",
            str(tmp_path / "unpaired-model"),
            "--source-optimizer",
            str(source_optimizer),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert unpaired.returncode != 0
    assert "supplied together" in unpaired.stderr

    raw_v1 = plan.to_dict(include_hash=False)
    raw_v1["schema_version"] = 1
    raw_v1.pop("optimizer_mapping")
    raw_v1["plan_hash"] = hash_canonical(raw_v1)
    v1_converter = tmp_path / "v1-converter.py"
    emit_converter(plan_from_dict(raw_v1), v1_converter)
    v1 = subprocess.run(
        [
            sys.executable,
            str(v1_converter),
            "--source",
            str(source_path),
            "--output",
            str(tmp_path / "v1-model"),
            "--source-optimizer",
            str(source_optimizer),
            "--optimizer-output",
            str(tmp_path / "v1-optimizer"),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert v1.returncode != 0
    assert "schema-version-2" in v1.stderr
