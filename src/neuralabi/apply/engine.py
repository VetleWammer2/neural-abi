"""Plan execution in memory or against bounded SafeTensors stores."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch

from neuralabi.checkpoints.store import TensorStore, open_tensor_store
from neuralabi.checkpoints.writer import DEFAULT_MAX_SHARD_SIZE, write_checkpoint
from neuralabi.formats.optimizer import OptimizerStateResult
from neuralabi.formats.plan import ConversionPlan, validate_plan
from neuralabi.status import CheckpointError
from neuralabi.transforms.execute import execute_checked
from neuralabi.util.canonical_json import pretty_dumps
from neuralabi.util.hashing import hash_file
from neuralabi.util.paths import ensure_disjoint_paths


@dataclass(frozen=True)
class ConversionResult:
    checkpoint_path: Path
    checkpoint_fingerprint: str
    file_hashes: dict[str, str]
    tensor_count: int
    logical_bytes: int
    duration_seconds: float
    optimizer_state: OptimizerStateResult | None = None

    def manifest(self, plan: ConversionPlan) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 2,
            "tool_version": plan.tool_version,
            "plan_hash": plan.plan_hash,
            "conversion_scope": {
                "parameter_state": True,
                "optimizer_state": self.optimizer_state is not None,
            },
            "source_checkpoint_fingerprint": plan.source.checkpoint_fingerprint,
            "target_checkpoint_fingerprint": self.checkpoint_fingerprint,
            "target_tensor_count": self.tensor_count,
            "target_logical_bytes": self.logical_bytes,
            "duration_seconds": self.duration_seconds,
            "file_hashes": dict(sorted(self.file_hashes.items())),
        }
        if self.optimizer_state is not None:
            result["optimizer_state"] = {
                "bundle_hash": self.optimizer_state.bundle_hash,
                "tensor_fingerprint": self.optimizer_state.tensor_fingerprint,
                "tensor_count": self.optimizer_state.tensor_count,
                "logical_bytes": self.optimizer_state.logical_bytes,
                "file_hashes": dict(sorted(self.optimizer_state.file_hashes.items())),
            }
        return result


def _validate_store(plan: ConversionPlan, store: TensorStore) -> None:
    expected = plan.source_tensors
    actual_keys = set(store.keys())
    if actual_keys != set(expected):
        raise CheckpointError(
            f"source checkpoint keys disagree with plan: missing={sorted(set(expected) - actual_keys)}, unexpected={sorted(actual_keys - set(expected))}"
        )
    for key in sorted(expected):
        metadata = store.metadata(key)
        spec = expected[key]
        if metadata.shape != spec.shape or metadata.dtype != spec.dtype:
            raise CheckpointError(
                f"source tensor {key!r} metadata {(metadata.shape, metadata.dtype)} != {(spec.shape, spec.dtype)}"
            )
    expected_fingerprint = plan.source.checkpoint_fingerprint
    if expected_fingerprint is None or store.fingerprint() != expected_fingerprint:
        raise CheckpointError("source checkpoint fingerprint does not match the plan")


def apply_to_state(
    plan: ConversionPlan,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    validate_plan(plan)
    result: dict[str, torch.Tensor] = {}
    for key, target in sorted(plan.targets.items()):
        dependencies = {name: source_state[name] for name in set(target.expression.source_keys())}
        result[key] = execute_checked(target.expression, dependencies, expected=target.spec).clone()
    return result


def apply_store(
    plan: ConversionPlan,
    source: TensorStore,
    output: Path,
    *,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> ConversionResult:
    validate_plan(plan)
    _validate_store(plan, source)
    start = perf_counter()

    def generated() -> Iterator[tuple[str, torch.Tensor]]:
        for key, target in sorted(plan.targets.items()):
            dependencies = {
                source_key: source.read(source_key)
                for source_key in sorted(set(target.expression.source_keys()))
            }
            yield key, execute_checked(target.expression, dependencies, expected=target.spec)

    checkpoint_path, files, logical_bytes = write_checkpoint(
        generated(),
        output,
        max_shard_size=max_shard_size,
    )
    target_store = open_tensor_store(checkpoint_path)
    file_hashes = {path.name: hash_file(path) for path in sorted(files)}
    return ConversionResult(
        checkpoint_path,
        target_store.fingerprint(),
        file_hashes,
        len(plan.targets),
        logical_bytes,
        perf_counter() - start,
    )


def apply_checkpoint(
    plan: ConversionPlan,
    source_path: Path,
    output: Path,
    *,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    manifest_path: Path | None = None,
    source_optimizer_state: Path | None = None,
    optimizer_output: Path | None = None,
) -> ConversionResult:
    if (source_optimizer_state is None) != (optimizer_output is None):
        raise CheckpointError(
            "source_optimizer_state and optimizer_output must be supplied together"
        )
    default_destination = (
        output.with_suffix(".conversion-manifest.json")
        if output.suffix
        else output / "conversion-manifest.json"
    )
    destination = manifest_path or default_destination
    ensure_disjoint_paths(source_path, output)
    ensure_disjoint_paths(source_path, destination)
    if source_optimizer_state is not None and optimizer_output is not None:
        ensure_disjoint_paths(source_optimizer_state, output)
        ensure_disjoint_paths(source_optimizer_state, optimizer_output)
        ensure_disjoint_paths(source_optimizer_state, destination)
        ensure_disjoint_paths(source_path, optimizer_output)
        ensure_disjoint_paths(output, optimizer_output)
        ensure_disjoint_paths(optimizer_output, destination)
    if destination.resolve(strict=False) != default_destination.resolve(strict=False):
        ensure_disjoint_paths(output, destination)
    source = open_tensor_store(source_path)
    result = apply_store(plan, source, output, max_shard_size=max_shard_size)
    if source_optimizer_state is not None and optimizer_output is not None:
        from neuralabi.apply.optimizer import apply_optimizer_checkpoint

        optimizer_result = apply_optimizer_checkpoint(
            plan,
            source_optimizer_state,
            result.checkpoint_path,
            optimizer_output,
            max_shard_size=max_shard_size,
        )
        result = replace(result, optimizer_state=optimizer_result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            handle.write(pretty_dumps(result.manifest(plan)))
            temporary = Path(handle.name)
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
