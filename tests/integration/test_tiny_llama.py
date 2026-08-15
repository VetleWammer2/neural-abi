from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from safetensors.torch import save_file

pytest.importorskip("transformers")

from examples.tiny_llama.models import source_adapter, target_adapter
from neuralabi.apply.engine import apply_checkpoint
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.codegen.converter import emit_converter
from neuralabi.export.capture import capture_adapter
from neuralabi.recognize.decoder import recognize_decoder
from neuralabi.status import ClaimStatus
from neuralabi.synth.solver import synthesize_plan
from neuralabi.verify.certificate import verify_conversion


@pytest.mark.integration
def test_offline_tiny_llama_to_fused_runtime(tmp_path: Path) -> None:
    source = capture_adapter(source_adapter)
    target = capture_adapter(target_adapter)
    source_model = recognize_decoder(source.artifact)
    target_model = recognize_decoder(target.artifact)
    source_state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
    source_path = tmp_path / "source.safetensors"
    save_file(source_state, source_path)
    source_store = open_tensor_store(source_path)
    plan = synthesize_plan(
        source_model,
        target_model,
        source.artifact,
        target.artifact,
        checkpoint_fingerprint=source_store.fingerprint(),
        name_hint_mode="off",
    )
    converted_path = tmp_path / "library"
    result = apply_checkpoint(plan, source_path, converted_path, max_shard_size=2048)
    target_store = open_tensor_store(result.checkpoint_path)
    target_state = {key: target_store.read(key) for key in target_store}
    certificate = verify_conversion(
        plan,
        source,
        target,
        source_model,
        target_model,
        source_state,
        target_state,
        seeds=(0, 1),
        source_checkpoint_fingerprint=source_store.fingerprint(),
        target_checkpoint_fingerprint=result.checkpoint_fingerprint,
    )
    assert certificate.verification_outcome == ClaimStatus.VERIFIED
    converter = tmp_path / "converter.py"
    emit_converter(plan, converter)
    standalone_path = tmp_path / "standalone"
    completed = subprocess.run(
        [
            sys.executable,
            str(converter),
            "--source",
            str(source_path),
            "--output",
            str(standalone_path),
            "--max-shard-size",
            "2KB",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    standalone = open_tensor_store(standalone_path)
    assert standalone.fingerprint() == result.checkpoint_fingerprint
    loaded = target_adapter.build(device="cpu", dtype=__import__("torch").float32)
    loaded.load_state_dict({key: standalone.read(key) for key in standalone}, strict=True)
