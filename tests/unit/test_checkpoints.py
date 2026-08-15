from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.checkpoints.writer import write_checkpoint
from neuralabi.status import CheckpointError
from neuralabi.util.hashing import hash_file


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "opaque.a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "opaque.b": torch.arange(7, dtype=torch.int64),
        "opaque.c": torch.randn(5, 5, generator=torch.Generator().manual_seed(4)),
    }


def test_single_and_sharded_logical_fingerprints_match(tmp_path: Path) -> None:
    tensors = _tensors()
    single = tmp_path / "single.safetensors"
    save_file(tensors, single)
    sharded = tmp_path / "sharded"
    main, files, _ = write_checkpoint(iter(sorted(tensors.items())), sharded, max_shard_size=64)
    assert main.name == "model.safetensors.index.json"
    assert len(files) >= 3
    assert open_tensor_store(single).fingerprint() == open_tensor_store(sharded).fingerprint()


def test_sharded_output_is_deterministic(tmp_path: Path) -> None:
    tensors = _tensors()
    outputs = []
    for name in ("one", "two"):
        destination = tmp_path / name
        _, files, _ = write_checkpoint(
            iter(sorted(tensors.items())), destination, max_shard_size=64
        )
        outputs.append({path.name: hash_file(path) for path in files})
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "index_text",
    [
        '{"weight_map":{"a":"../escape.safetensors"}}',
        '{"weight_map":{"a":"missing.safetensors"}}',
        '{"weight_map":{"a":"x.safetensors","a":"y.safetensors"}}',
        '{"weight_map":[]}',
    ],
)
def test_malformed_indexes_are_rejected(tmp_path: Path, index_text: str) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(index_text, encoding="utf-8")
    with pytest.raises(CheckpointError):
        open_tensor_store(index)


def test_atomic_cleanup_after_generation_failure(tmp_path: Path) -> None:
    output = tmp_path / "failed"

    def broken() -> object:
        yield "a", torch.ones(2)
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected"):
        write_checkpoint(broken(), output, max_shard_size=1024)  # type: ignore[arg-type]
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.neuralabi-*"))


def test_writer_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(CheckpointError, match="refusing"):
        write_checkpoint(iter(sorted(_tensors().items())), output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"
