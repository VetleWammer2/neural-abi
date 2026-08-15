"""Measure bounded-shard conversion using synthetic SafeTensors data."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch
from safetensors.torch import save_file

from neuralabi import __version__
from neuralabi.apply.engine import apply_checkpoint
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.formats.plan import (
    PLAN_SCHEMA_VERSION,
    ConversionPlan,
    PlanEndpoint,
    PlannedTensor,
)
from neuralabi.status import LinkStatus
from neuralabi.transforms import Source, TensorSpec, Transpose


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        try:
            import psutil
        except ImportError:
            return None
        memory = psutil.Process().memory_info()
        return int(getattr(memory, "peak_wset", memory.rss))
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--max-shard-size", type=int, default=16 * 1024**2)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(901)
    source_state = {
        f"opaque.{index:04d}": torch.randn(args.dimension, args.dimension, generator=generator)
        for index in range(args.tensors)
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_path = root / "source.safetensors"
        save_file(source_state, source_path)
        del source_state
        source = open_tensor_store(source_path)
        source_specs = {
            key: TensorSpec(source.metadata(key).shape, source.metadata(key).dtype)
            for key in source
        }
        targets = {
            f"linked.{index:04d}": PlannedTensor(
                Transpose(Source(key), (0, 1)),
                (args.dimension, args.dimension),
                "float32",
                (f"synthetic.slot[{index}]",),
            )
            for index, key in enumerate(sorted(source_specs))
        }
        target_specs = {key: target.spec for key, target in targets.items()}
        plan = ConversionPlan(
            PLAN_SCHEMA_VERSION,
            __version__,
            PlanEndpoint("synthetic-source", "0" * 64, "1" * 64, source.fingerprint()),
            PlanEndpoint("synthetic-target", "2" * 64, "3" * 64),
            {"kind": "synthetic-layout-benchmark"},
            source_specs,
            target_specs,
            targets,
            (),
            (),
            (),
            "off",
            False,
            LinkStatus.UNIQUE,
            sum(target.expression.cost() for target in targets.values()),
            {"candidate_count": 1},
            (),
            (),
            None,
            "",
        ).with_hash()
        output = root / "target"
        started = perf_counter()
        result = apply_checkpoint(
            plan,
            source_path,
            output,
            max_shard_size=args.max_shard_size,
        )
        duration = perf_counter() - started
        source_bytes = source_path.stat().st_size
        target_files = [path for path in output.iterdir() if path.suffix == ".safetensors"]
        temporary_bytes = sum(path.stat().st_size for path in output.iterdir())
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint_file_bytes": source_bytes,
                    "logical_tensor_bytes": result.logical_bytes,
                    "source_shard_count": 1,
                    "target_shard_count": len(target_files),
                    "tensor_transformations": args.tensors,
                    "wall_clock_seconds": duration,
                    "throughput_logical_bytes_per_second": result.logical_bytes / duration,
                    "peak_rss_bytes": _peak_rss_bytes(),
                    "temporary_and_output_disk_bytes": temporary_bytes,
                    "target_checkpoint_fingerprint": result.checkpoint_fingerprint,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
