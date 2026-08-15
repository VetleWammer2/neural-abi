from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from examples.generated_transformer.models import GeneratedAdapter, LogicalConfig
from neuralabi.apply.engine import apply_to_state
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import capture_adapter
from neuralabi.recognize.decoder import recognize_decoder
from neuralabi.status import LinkStatus
from neuralabi.synth.solver import synthesize_plan


def _config(case: int) -> LogicalConfig:
    heads = (2, 4, 6, 8)[case % 4]
    head_dim = (8, 10, 12)[(case // 4) % 3]
    divisors = [value for value in range(1, heads + 1) if heads % value == 0]
    kv_heads = divisors[(case // 3) % len(divisors)]
    hidden = heads * head_dim
    return LogicalConfig(
        seed=1000 + case,
        layout_seed=case,
        layers=1 + case % 3,
        vocabulary_size=67 + case,
        hidden_size=hidden,
        attention_heads=heads,
        key_value_heads=kv_heads,
        intermediate_size=((hidden * 3 // 2 + 7) // 8) * 8,
        sequence_length=2 + case % 8,
        bias=case % 8 == 0,
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", range(32))
def test_generated_layout_matrix_is_unique_and_name_independent(case: int, tmp_path: Path) -> None:
    config = _config(case)
    source = capture_adapter(GeneratedAdapter(config, "source"))
    target = capture_adapter(GeneratedAdapter(config, "target"))
    source_model = recognize_decoder(source.artifact)
    target_model = recognize_decoder(target.artifact)
    source_state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
    checkpoint = tmp_path / f"case-{case}.safetensors"
    save_file(source_state, checkpoint)
    store = open_tensor_store(checkpoint)
    plan = synthesize_plan(
        source_model,
        target_model,
        source.artifact,
        target.artifact,
        checkpoint_fingerprint=store.fingerprint(),
        name_hint_mode="off",
    )
    assert plan.mapping_status == LinkStatus.UNIQUE
    assert plan.name_hints_used is False
    converted = apply_to_state(plan, source_state)
    target.model.load_state_dict(converted, strict=True)
    source_output = source.model(*source.args, **source.kwargs)
    target_output = target.model(*target.args, **target.kwargs)
    torch.testing.assert_close(source_output, target_output, rtol=2e-4, atol=2e-5)
    assert plan.with_hash().plan_hash == plan.plan_hash


def test_matrix_covers_mha_gqa_all_orders_and_interleaving() -> None:
    orders: set[tuple[str, str, str]] = set()
    saw_interleaved = False
    saw_contiguous = False
    saw_mha = False
    saw_gqa = False
    for case in range(32):
        config = _config(case)
        target = capture_adapter(GeneratedAdapter(config, "target"))
        model = recognize_decoder(target.artifact)
        saw_mha |= config.attention_heads == config.key_value_heads
        saw_gqa |= config.attention_heads != config.key_value_heads
        qkv_layouts = [
            layout
            for layout in model.layouts
            if len(layout.components) == 3 and ".attention." in layout.components[0]
        ]
        for layout in qkv_layouts:
            order = tuple(component.rsplit(".", 2)[-2] for component in layout.components)
            orders.add(order)  # type: ignore[arg-type]
            saw_interleaved |= layout.interleave_groups is not None
            saw_contiguous |= layout.interleave_groups is None
    assert len(orders) == 6
    assert saw_interleaved and saw_contiguous and saw_mha and saw_gqa
