from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from safetensors.torch import save_file

from examples.twin_mlp.models import source_adapter, target_adapter
from neuralabi.apply.engine import apply_checkpoint
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.codegen.converter import emit_converter
from neuralabi.export.capture import capture_adapter
from neuralabi.recognize.twin_mlp import recognize_twin_mlp
from neuralabi.synth.solver import synthesize_plan


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
