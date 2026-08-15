"""NeuralABI command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neuralabi.adapters import ModelAdapter, load_adapter
from neuralabi.apply.engine import ConversionResult, apply_checkpoint
from neuralabi.checkpoints.store import TensorStore, open_tensor_store
from neuralabi.checkpoints.writer import parse_size
from neuralabi.codegen.converter import emit_converter
from neuralabi.export.capture import CapturedModel, capture_adapter
from neuralabi.export.signature import compare_signatures
from neuralabi.formats.plan import ConversionPlan, load_plan
from neuralabi.ir.semantic import CanonicalModel
from neuralabi.ir.state import validate_state_metadata
from neuralabi.recognize.twin_mlp import recognize_twin_mlp
from neuralabi.status import (
    ArchitectureMismatchError,
    CheckpointError,
    ClaimStatus,
    LinkStatus,
    NeuralABIError,
    PlanValidationError,
    UnsupportedGraphError,
)
from neuralabi.synth.solver import synthesize_plan
from neuralabi.util.canonical_json import pretty_dumps
from neuralabi.util.hashing import hash_file
from neuralabi.verify.certificate import verify_conversion


@dataclass(frozen=True)
class AnalyzedPair:
    source_capture: CapturedModel
    target_capture: CapturedModel
    source_model: CanonicalModel
    target_model: CanonicalModel

    def analysis(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": LinkStatus.UNIQUE.value,
            "source": {
                "adapter_id": self.source_capture.adapter.adapter_id,
                "graph_hash": self.source_capture.artifact.graph_hash,
                "state_schema_hash": self.source_capture.artifact.state_schema.schema_hash,
                "exported_node_count": len(self.source_capture.artifact.graph.nodes),
                "semantic_slot_count": len(self.source_model.slots),
                "physical_layout_count": len(self.source_model.layouts),
                "unsupported_regions": list(self.source_model.unsupported_regions),
            },
            "target": {
                "adapter_id": self.target_capture.adapter.adapter_id,
                "graph_hash": self.target_capture.artifact.graph_hash,
                "state_schema_hash": self.target_capture.artifact.state_schema.schema_hash,
                "exported_node_count": len(self.target_capture.artifact.graph.nodes),
                "semantic_slot_count": len(self.target_model.slots),
                "physical_layout_count": len(self.target_model.layouts),
                "unsupported_regions": list(self.target_model.unsupported_regions),
            },
            "architecture_signature": self.source_model.architecture.to_dict(),
            "signature_compatible": True,
            "linking_appears_possible": True,
            "rewrite_rules_used": sorted(
                set(
                    self.source_capture.artifact.graph.rewrite_rules_used
                    + self.target_capture.artifact.graph.rewrite_rules_used
                )
            ),
        }


def _recognize(artifact: Any) -> CanonicalModel:
    operations = {node.canonical_op for node in artifact.graph.nodes}
    if "softmax" in operations or "embedding" in operations:
        try:
            from neuralabi.recognize.decoder import recognize_decoder
        except ImportError as exc:
            raise UnsupportedGraphError("decoder recognizer is unavailable") from exc
        return recognize_decoder(artifact)
    return recognize_twin_mlp(artifact)


def _dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise NeuralABIError(f"unsupported dtype {name!r}") from exc


def _analyze(
    source_adapter: ModelAdapter,
    target_adapter: ModelAdapter,
    *,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> AnalyzedPair:
    source_capture = capture_adapter(source_adapter, device=device, dtype=dtype, seed=seed)
    target_capture = capture_adapter(target_adapter, device=device, dtype=dtype, seed=seed)
    source_model = _recognize(source_capture.artifact)
    target_model = _recognize(target_capture.artifact)
    compare_signatures(source_model.architecture, target_model.architecture)
    return AnalyzedPair(source_capture, target_capture, source_model, target_model)


def _store_metadata(store: TensorStore) -> dict[str, tuple[tuple[int, ...], str]]:
    return {key: (store.metadata(key).shape, store.metadata(key).dtype) for key in store}


def _state_from_store(store: TensorStore) -> dict[str, torch.Tensor]:
    return {key: store.read(key) for key in store}


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not result or len(result) > 128:
        raise argparse.ArgumentTypeError("provide between 1 and 128 probe seeds")
    return result


def _check_plan_endpoints(plan: ConversionPlan, pair: AnalyzedPair) -> None:
    checks = {
        "source graph": (plan.source.graph_hash, pair.source_capture.artifact.graph_hash),
        "target graph": (plan.target.graph_hash, pair.target_capture.artifact.graph_hash),
        "source state schema": (
            plan.source.state_schema_hash,
            pair.source_capture.artifact.state_schema.schema_hash,
        ),
        "target state schema": (
            plan.target.state_schema_hash,
            pair.target_capture.artifact.state_schema.schema_hash,
        ),
    }
    differences = [label for label, (expected, actual) in checks.items() if expected != actual]
    if differences:
        raise PlanValidationError("plan does not match adapters: " + ", ".join(differences))


def _add_pair_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="trusted local MODULE:ATTRIBUTE adapter")
    parser.add_argument("--target", required=True, help="trusted local MODULE:ATTRIBUTE adapter")
    parser.add_argument("--name-hints", choices=("off", "weak"), default="off")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float64", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--export-seed", type=int, default=0)


def _pair_from_args(args: argparse.Namespace) -> AnalyzedPair:
    return _analyze(
        load_adapter(args.source),
        load_adapter(args.target),
        device=args.device,
        dtype=_dtype(args.dtype),
        seed=args.export_seed,
    )


def _cmd_scan(args: argparse.Namespace) -> int:
    pair = _pair_from_args(args)
    analysis = pair.analysis()
    if args.json:
        print(pretty_dumps(analysis), end="")
    else:
        signature = analysis["architecture_signature"]
        print(
            f"Source graph: {analysis['source']['adapter_id']} ({analysis['source']['exported_node_count']} nodes)"
        )
        print(
            f"Target graph: {analysis['target']['adapter_id']} ({analysis['target']['exported_node_count']} nodes)"
        )
        print(f"Architecture: {json.dumps(signature, sort_keys=True)}")
        print("Mapping status: LINKING_APPEARS_POSSIBLE")
    return 0


def _infer(
    pair: AnalyzedPair,
    checkpoint: Path,
    name_hint_mode: str,
    max_candidates: int = 128,
) -> ConversionPlan:
    store = open_tensor_store(checkpoint)
    validate_state_metadata(
        pair.source_capture.artifact.state_schema,
        _store_metadata(store),
        label="source checkpoint",
    )
    return synthesize_plan(
        pair.source_model,
        pair.target_model,
        pair.source_capture.artifact,
        pair.target_capture.artifact,
        checkpoint_fingerprint=store.fingerprint(),
        name_hint_mode=name_hint_mode,
        max_candidates=max_candidates,
    )


def _cmd_infer(args: argparse.Namespace) -> int:
    pair = _pair_from_args(args)
    plan = _infer(pair, args.source_checkpoint, args.name_hints, args.max_candidates)
    plan.write(args.output)
    print(f"Mapping status: {plan.mapping_status.value}")
    print(f"Plan hash: {plan.plan_hash}")
    print(f"Generated: {args.output}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    result = apply_checkpoint(
        plan,
        args.source_checkpoint,
        args.output,
        max_shard_size=parse_size(args.max_shard_size),
        manifest_path=args.manifest,
    )
    print(f"Target checkpoint fingerprint: {result.checkpoint_fingerprint}")
    print(f"Generated: {result.checkpoint_path}")
    return 0


def _verify(
    plan: ConversionPlan,
    pair: AnalyzedPair,
    source_checkpoint: Path,
    target_checkpoint: Path,
    *,
    seeds: Sequence[int],
    file_hashes: dict[str, str] | None = None,
) -> Any:
    _check_plan_endpoints(plan, pair)
    source_store = open_tensor_store(source_checkpoint)
    target_store = open_tensor_store(target_checkpoint)
    validate_state_metadata(
        pair.source_capture.artifact.state_schema,
        _store_metadata(source_store),
        label="source checkpoint",
    )
    validate_state_metadata(
        pair.target_capture.artifact.state_schema,
        _store_metadata(target_store),
        label="target checkpoint",
    )
    if source_store.fingerprint() != plan.source.checkpoint_fingerprint:
        raise CheckpointError("source checkpoint fingerprint does not match plan")
    return verify_conversion(
        plan,
        pair.source_capture,
        pair.target_capture,
        pair.source_model,
        pair.target_model,
        _state_from_store(source_store),
        _state_from_store(target_store),
        seeds=seeds,
        source_checkpoint_fingerprint=source_store.fingerprint(),
        target_checkpoint_fingerprint=target_store.fingerprint(),
        generated_file_hashes=file_hashes,
    )


def _cmd_verify(args: argparse.Namespace) -> int:
    pair = _pair_from_args(args)
    plan = load_plan(args.plan)
    certificate = _verify(
        plan,
        pair,
        args.source_checkpoint,
        args.target_checkpoint,
        seeds=args.seeds,
    )
    certificate.write(args.output)
    print(f"Verification: {certificate.verification_outcome.value}")
    print(f"Generated: {args.output}")
    return 0 if certificate.verification_outcome == ClaimStatus.VERIFIED else 3


def _cmd_explain(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    explanation = {
        "plan_hash": plan.plan_hash,
        "source": plan.source.to_dict(),
        "target": plan.target.to_dict(),
        "architecture_signature": plan.architecture_signature,
        "mapping_status": plan.mapping_status.value,
        "plan_complexity": plan.plan_complexity,
        "targets": {
            key: {
                "semantic_slots": list(target.semantic_slots),
                "physical_dependencies": sorted(set(target.expression.source_keys())),
                "expression": target.expression.to_dict(),
            }
            for key, target in sorted(plan.targets.items())
        },
        "aliases": [alias.to_dict() for alias in plan.aliases],
        "assumptions": list(plan.assumptions),
        "ambiguity": list(plan.ambiguity),
        "inverse_available": plan.inverse_targets is not None,
    }
    if args.json:
        print(pretty_dumps(explanation), end="")
    else:
        print(
            f"Plan {plan.plan_hash} ({plan.mapping_status.value}, complexity {plan.plan_complexity})"
        )
        print(f"Source graph: {plan.source.graph_hash}")
        print(f"Target graph: {plan.target.graph_hash}")
        for key, target in sorted(plan.targets.items()):
            slots = ", ".join(target.semantic_slots)
            dependencies = ", ".join(sorted(set(target.expression.source_keys())))
            print(f"  {key} <- [{dependencies}] via {target.expression.op} ({slots})")
        print(
            f"Exact inverse: {'available' if plan.inverse_targets is not None else 'not available'}"
        )
    return 0


def _cmd_emit(args: argparse.Namespace) -> int:
    if args.format != "python":
        raise NeuralABIError(f"unsupported converter format {args.format!r}")
    plan = load_plan(args.plan)
    emit_converter(plan, args.output)
    print(f"Generated: {args.output}")
    return 0


def _report(plan: ConversionPlan, certificate: Any, result: ConversionResult) -> str:
    claims = "\n".join(f"- {claim.claim}: {claim.status.value}" for claim in certificate.claims)
    return f"""# NeuralABI link report

- Mapping status: {plan.mapping_status.value}
- Plan hash: `{plan.plan_hash}`
- Source adapter: `{plan.source.adapter_id}`
- Target adapter: `{plan.target.adapter_id}`
- Target checkpoint fingerprint: `{result.checkpoint_fingerprint}`
- Verification outcome: {certificate.verification_outcome.value}

## Claims

{claims}

This report records structural compatibility under NeuralABI's documented canonicalization rules
plus empirical verification over the certificate's deterministic probe suite. It is not a formal
proof for every possible input.
"""


def _cmd_link(args: argparse.Namespace) -> int:
    output: Path = args.output
    if output.exists() and (output.is_file() or any(output.iterdir())):
        raise NeuralABIError(f"link output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pair = _pair_from_args(args)
    analysis_path = output / "analysis.json"
    analysis_path.write_text(pretty_dumps(pair.analysis()), encoding="utf-8", newline="\n")
    plan = _infer(pair, args.source_checkpoint, args.name_hints, args.max_candidates)
    plan_path = output / "plan.neuralabi.json"
    plan.write(plan_path)
    target_path = output / "target"
    manifest_path = output / "conversion-manifest.json"
    conversion = apply_checkpoint(
        plan,
        args.source_checkpoint,
        target_path,
        max_shard_size=parse_size(args.max_shard_size),
        manifest_path=manifest_path,
    )
    converter_path = output / "converter.py"
    emit_converter(plan, converter_path)
    generated_hashes = {
        path.name: hash_file(path)
        for path in (analysis_path, plan_path, manifest_path, converter_path)
    }
    generated_hashes.update(conversion.file_hashes)
    certificate = _verify(
        plan,
        pair,
        args.source_checkpoint,
        conversion.checkpoint_path,
        seeds=args.seeds,
        file_hashes=generated_hashes,
    )
    certificate_path = output / "certificate.json"
    certificate.write(certificate_path)
    report_path = output / "report.md"
    report_path.write_text(_report(plan, certificate, conversion), encoding="utf-8", newline="\n")
    print(f"Source graph: {pair.source_capture.adapter.adapter_id}")
    print(f"Target graph: {pair.target_capture.adapter.adapter_id}")
    print(f"Mapping status: {plan.mapping_status.value}")
    print(f"Plan complexity: {plan.plan_complexity}")
    print(f"Generated: {output}")
    for claim in certificate.claims:
        print(f"  {claim.claim:<31} {claim.status.value}")
    return 0 if certificate.verification_outcome == ClaimStatus.VERIFIED else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralabi", description="A linker and ABI verifier for neural checkpoints"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="capture and compare two model implementations")
    _add_pair_arguments(scan)
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(handler=_cmd_scan)

    infer = subparsers.add_parser("infer", help="synthesize a declarative conversion plan")
    _add_pair_arguments(infer)
    infer.add_argument("--source-checkpoint", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--max-candidates", type=int, default=128)
    infer.set_defaults(handler=_cmd_infer)

    apply = subparsers.add_parser("apply", help="execute a plan without loading adapters")
    apply.add_argument("plan", type=Path)
    apply.add_argument("source_checkpoint", type=Path)
    apply.add_argument("--output", required=True, type=Path)
    apply.add_argument("--manifest", type=Path)
    apply.add_argument("--max-shard-size", default="2GB")
    apply.set_defaults(handler=_cmd_apply)

    verify = subparsers.add_parser("verify", help="verify a converted checkpoint")
    verify.add_argument("plan", type=Path)
    _add_pair_arguments(verify)
    verify.add_argument("--source-checkpoint", required=True, type=Path)
    verify.add_argument("--target-checkpoint", required=True, type=Path)
    verify.add_argument("--seeds", type=_seeds, default=(0, 1, 2, 3))
    verify.add_argument("--output", type=Path, default=Path("certificate.json"))
    verify.set_defaults(handler=_cmd_verify)

    link = subparsers.add_parser("link", help="scan, infer, apply, verify, and emit")
    _add_pair_arguments(link)
    link.add_argument("--source-checkpoint", required=True, type=Path)
    link.add_argument("--output", required=True, type=Path)
    link.add_argument("--max-candidates", type=int, default=128)
    link.add_argument("--max-shard-size", default="2GB")
    link.add_argument("--seeds", type=_seeds, default=(0, 1, 2, 3))
    link.set_defaults(handler=_cmd_link)

    explain = subparsers.add_parser("explain", help="explain a conversion plan")
    explain.add_argument("plan", type=Path)
    explain.add_argument("--json", action="store_true")
    explain.set_defaults(handler=_cmd_explain)

    emit = subparsers.add_parser("emit", help="emit a standalone converter")
    emit.add_argument("plan", type=Path)
    emit.add_argument("--format", choices=("python",), default="python")
    emit.add_argument("--output", required=True, type=Path)
    emit.set_defaults(handler=_cmd_emit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ArchitectureMismatchError as exc:
        parser.exit(4, f"neuralabi: {exc}\n")
    except UnsupportedGraphError as exc:
        parser.exit(5, f"neuralabi: UNSUPPORTED: {exc}\n")
    except (NeuralABIError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"neuralabi: error: {exc}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
