"""Measure graph capture, semantic recognition, and plan synthesis."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

from safetensors.torch import save_file

from examples.generated_transformer.models import GeneratedAdapter, LogicalConfig
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import capture_adapter
from neuralabi.recognize.decoder import recognize_decoder
from neuralabi.synth.solver import synthesize_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--intermediate", type=int, default=48)
    parser.add_argument("--layout-seed", type=int, default=0)
    args = parser.parse_args()
    config = LogicalConfig(
        seed=7001,
        layout_seed=args.layout_seed,
        layers=args.layers,
        hidden_size=args.hidden,
        attention_heads=args.heads,
        key_value_heads=args.kv_heads,
        intermediate_size=args.intermediate,
    )
    source_adapter = GeneratedAdapter(config, "source")
    target_adapter = GeneratedAdapter(config, "target")
    started = perf_counter()
    source = capture_adapter(source_adapter)
    target = capture_adapter(target_adapter)
    captured = perf_counter()
    source_model = recognize_decoder(source.artifact)
    target_model = recognize_decoder(target.artifact)
    recognized = perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "source.safetensors"
        save_file(
            {key: value.detach().clone() for key, value in source.model.state_dict().items()},
            checkpoint,
        )
        store = open_tensor_store(checkpoint)
        plan = synthesize_plan(
            source_model,
            target_model,
            source.artifact,
            target.artifact,
            checkpoint_fingerprint=store.fingerprint(),
        )
    finished = perf_counter()
    print(
        json.dumps(
            {
                "schema_version": 1,
                "decoder_layer_count": config.layers,
                "source_exported_node_count": len(source.artifact.graph.nodes),
                "target_exported_node_count": len(target.artifact.graph.nodes),
                "semantic_slot_count": len(source_model.slots),
                "candidate_count": plan.synthesis_statistics["candidate_count"],
                "graph_capture_and_canonicalization_seconds": captured - started,
                "semantic_recognition_seconds": recognized - captured,
                "constraint_solving_seconds": finished - recognized,
                "numerical_refinement_seconds": 0.0,
                "total_inference_seconds": finished - started,
                "plan_hash": plan.plan_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
