from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from safetensors.torch import save_file

from examples.twin_mlp.models import source_adapter


def test_all_cli_commands_and_link_bundle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.safetensors"
    model = source_adapter.build(device="cpu", dtype=__import__("torch").float32)
    save_file(
        {key: value.detach().clone() for key, value in model.state_dict().items()}, checkpoint
    )
    pair = [
        "--source",
        "examples.twin_mlp.models:source_adapter",
        "--target",
        "examples.twin_mlp.models:target_adapter",
        "--name-hints",
        "off",
    ]
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
    target = tmp_path / "applied"
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
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert apply.returncode == 0, apply.stderr
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
    assert json.loads(certificate.read_text(encoding="utf-8"))["verification_outcome"] == "VERIFIED"
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
    }
    assert {path.name for path in bundle.iterdir()} == expected
