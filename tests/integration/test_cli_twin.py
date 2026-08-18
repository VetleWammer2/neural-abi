from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from examples.twin_mlp.models import source_adapter
from neuralabi.optimizers.torch import export_optimizer_state


def test_all_cli_commands_and_link_bundle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.safetensors"
    model = source_adapter.build(device="cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False)
    for seed in (70, 71):
        args, kwargs = source_adapter.example_inputs(seed=seed, device="cpu")
        optimizer.zero_grad(set_to_none=True)
        source_adapter.select_outputs(model(*args, **kwargs)).square().mean().backward()
        optimizer.step()
    save_file(
        {key: value.detach().clone() for key, value in model.state_dict().items()}, checkpoint
    )
    source_optimizer = tmp_path / "source-optimizer"
    export_optimizer_state(model, optimizer, source_optimizer)
    pair = [
        "--source",
        "examples.twin_mlp.models:source_adapter",
        "--target",
        "examples.twin_mlp.models:target_adapter",
        "--name-hints",
        "off",
    ]
    nested_link = source_optimizer / "nested-link-output"
    unsafe_link = subprocess.run(
        [
            sys.executable,
            "-m",
            "neuralabi",
            "link",
            *pair,
            "--source-checkpoint",
            str(checkpoint),
            "--source-optimizer-state",
            str(source_optimizer),
            "--output",
            str(nested_link),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert unsafe_link.returncode != 0
    assert not nested_link.exists()
    scan = subprocess.run(
        [sys.executable, "-m", "neuralabi", "scan", *pair, "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert scan.returncode == 0, scan.stderr
    assert json.loads(scan.stdout)["linking_appears_possible"] is True
    plan = tmp_path / "plan.json"
    infer = subprocess.run(
        [
            sys.executable,
            "-m",
            "neuralabi",
            "infer",
            *pair,
            "--source-checkpoint",
            str(checkpoint),
            "--output",
            str(plan),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert infer.returncode == 0, infer.stderr
    optimizer_manifest = source_optimizer / "optimizer.neuralabi.json"
    original_optimizer_manifest = optimizer_manifest.read_bytes()
    unsafe_target = tmp_path / "unsafe-target"
    unsafe_apply = subprocess.run(
        [
            sys.executable,
            "-m",
            "neuralabi",
            "apply",
            str(plan),
            str(checkpoint),
            "--output",
            str(unsafe_target),
            "--manifest",
            str(optimizer_manifest),
            "--source-optimizer-state",
            str(source_optimizer),
            "--optimizer-output",
            str(tmp_path / "unsafe-optimizer-output"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert unsafe_apply.returncode != 0
    assert optimizer_manifest.read_bytes() == original_optimizer_manifest
    assert not unsafe_target.exists()
    target = tmp_path / "applied"
    target_optimizer = tmp_path / "applied-optimizer"
    apply = subprocess.run(
        [
            sys.executable,
            "-m",
            "neuralabi",
            "apply",
            str(plan),
            str(checkpoint),
            "--output",
            str(target),
            "--max-shard-size",
            "384B",
            "--source-optimizer-state",
            str(source_optimizer),
            "--optimizer-output",
            str(target_optimizer),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert apply.returncode == 0, apply.stderr
    unsafe_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "neuralabi",
            "verify",
            str(plan),
            *pair,
            "--source-checkpoint",
            str(checkpoint),
            "--target-checkpoint",
            str(target),
            "--source-optimizer-state",
            str(source_optimizer),
            "--target-optimizer-state",
            str(target_optimizer),
            "--output",
            str(optimizer_manifest),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert unsafe_verify.returncode != 0
    assert optimizer_manifest.read_bytes() == original_optimizer_manifest
    explain = subprocess.run(
        [sys.executable, "-m", "neuralabi", "explain", str(plan), "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert explain.returncode == 0, explain.stderr
    assert json.loads(explain.stdout)["inverse_available"] is True
    emitted = tmp_path / "emitted.py"
    emit = subprocess.run(
        [sys.executable, "-m", "neuralabi", "emit", str(plan), "--output", str(emitted)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert emit.returncode == 0, emit.stderr
    certificate = tmp_path / "verified.json"
    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "neuralabi",
            "verify",
            str(plan),
            *pair,
            "--source-checkpoint",
            str(checkpoint),
            "--target-checkpoint",
            str(target),
            "--source-optimizer-state",
            str(source_optimizer),
            "--target-optimizer-state",
            str(target_optimizer),
            "--resume-steps",
            "1",
            "--seeds",
            "0,1",
            "--output",
            str(certificate),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stderr
    certificate_data = json.loads(certificate.read_text(encoding="utf-8"))
    assert certificate_data["verification_outcome"] == "VERIFIED"
    claims = {item["claim"]: item["status"] for item in certificate_data["claims"]}
    assert claims["OPTIMIZER_STATE_VERIFIED"] == "VERIFIED"
    assert claims["RESUMED_TRAINING_EQUIVALENT"] == "VERIFIED"
    bundle = tmp_path / "bundle"
    link = subprocess.run(
        [
            sys.executable,
            "-m",
            "neuralabi",
            "link",
            *pair,
            "--source-checkpoint",
            str(checkpoint),
            "--output",
            str(bundle),
            "--seeds",
            "0,1",
            "--source-optimizer-state",
            str(source_optimizer),
            "--resume-steps",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert link.returncode == 0, link.stderr
    expected = {
        "analysis.json",
        "plan.neuralabi.json",
        "converter.py",
        "conversion-manifest.json",
        "certificate.json",
        "report.md",
        "target",
        "target-optimizer",
    }
    assert {path.name for path in bundle.iterdir()} == expected
