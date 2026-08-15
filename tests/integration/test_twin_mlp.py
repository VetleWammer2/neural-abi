from __future__ import annotations

from pathlib import Path

from safetensors.torch import save_file

from examples.twin_mlp.models import source_adapter, target_adapter
from neuralabi.apply.engine import apply_checkpoint
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import capture_adapter
from neuralabi.recognize.twin_mlp import recognize_twin_mlp
from neuralabi.status import ClaimStatus
from neuralabi.synth.solver import synthesize_plan
from neuralabi.verify.certificate import verify_conversion


def test_opaque_twin_mlp_end_to_end(tmp_path: Path) -> None:
    source_capture = capture_adapter(source_adapter)
    target_capture = capture_adapter(target_adapter)
    source_canonical = recognize_twin_mlp(source_capture.artifact)
    target_canonical = recognize_twin_mlp(target_capture.artifact)
    source_path = tmp_path / "source.safetensors"
    source_state = {
        key: tensor.detach().clone() for key, tensor in source_capture.model.state_dict().items()
    }
    save_file(source_state, source_path)
    source_store = open_tensor_store(source_path)
    plan = synthesize_plan(
        source_canonical,
        target_canonical,
        source_capture.artifact,
        target_capture.artifact,
        checkpoint_fingerprint=source_store.fingerprint(),
        name_hint_mode="off",
    )
    assert not plan.name_hints_used
    assert all("gate" not in key and "up" not in key for key in source_state)
    output = tmp_path / "target"
    result = apply_checkpoint(plan, source_path, output, max_shard_size=1024)
    target_store = open_tensor_store(result.checkpoint_path)
    target_state = {key: target_store.read(key) for key in target_store}
    certificate = verify_conversion(
        plan,
        source_capture,
        target_capture,
        source_canonical,
        target_canonical,
        source_state,
        target_state,
        seeds=(0, 1, 2, 3),
        source_checkpoint_fingerprint=source_store.fingerprint(),
        target_checkpoint_fingerprint=result.checkpoint_fingerprint,
        generated_file_hashes=result.file_hashes,
    )
    assert certificate.verification_outcome == ClaimStatus.VERIFIED
    claims = {claim.claim: claim.status for claim in certificate.claims}
    assert claims["FORWARD_VERIFIED"] == ClaimStatus.VERIFIED
    assert claims["INTERMEDIATE_VERIFIED"] == ClaimStatus.VERIFIED
    assert claims["PARAMETER_GRADIENT_VERIFIED"] == ClaimStatus.VERIFIED
    assert claims["ROUNDTRIP_EXACT"] == ClaimStatus.VERIFIED
    assert all(item["exact"] for item in certificate.roundtrip_results)
