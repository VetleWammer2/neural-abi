from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from examples.generated_transformer.models import GeneratedAdapter, LogicalConfig, _layer_layout
from neuralabi.apply.engine import apply_to_state
from neuralabi.apply.optimizer import apply_optimizer_checkpoint
from neuralabi.checkpoints.store import open_tensor_store
from neuralabi.export.capture import capture_adapter
from neuralabi.optimizers.torch import export_optimizer_state
from neuralabi.recognize.decoder import recognize_decoder
from neuralabi.status import LinkStatus
from neuralabi.synth.solver import synthesize_plan
from neuralabi.verify.optimizer import verify_optimizer_state


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
    target_checkpoint = tmp_path / f"case-{case}-target.safetensors"
    save_file(
        {key: value.detach().contiguous().clone() for key, value in converted.items()},
        target_checkpoint,
    )
    optimizer = torch.optim.Adam(source.model.parameters(), foreach=False, fused=False)
    for index, parameter in enumerate(source.model.parameters()):
        values = torch.arange(
            parameter.numel(), dtype=parameter.dtype, device=parameter.device
        ).reshape(parameter.shape)
        optimizer.state[parameter] = {
            "step": torch.tensor(7.0),
            "exp_avg": (values + 1000 * (index + 1)).clone(),
            "exp_avg_sq": (values.flip(tuple(range(values.ndim))) + 2000 * (index + 1)).clone(),
        }
    source_optimizer = export_optimizer_state(
        source.model, optimizer, tmp_path / f"case-{case}-source-optimizer"
    )
    target_optimizer = apply_optimizer_checkpoint(
        plan,
        source_optimizer.checkpoint_path,
        target_checkpoint,
        tmp_path / f"case-{case}-target-optimizer",
    )
    optimizer_verification = verify_optimizer_state(
        plan,
        source_optimizer.checkpoint_path,
        target_optimizer.checkpoint_path,
        target_checkpoint_fingerprint=open_tensor_store(target_checkpoint).fingerprint(),
    )
    assert optimizer_verification.coverage_complete
    assert optimizer_verification.passed
    assert all(item.exact for item in optimizer_verification.comparisons)
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
    qkv_singletons: set[bool] = set()
    mlp_singletons: set[bool] = set()
    gate_first_orders: set[bool] = set()
    for case in range(32):
        config = _config(case)
        target = capture_adapter(GeneratedAdapter(config, "target"))
        model = recognize_decoder(target.artifact)
        saw_mha |= config.attention_heads == config.key_value_heads
        saw_gqa |= config.attention_heads != config.key_value_heads
        for layer in range(config.layers):
            layout = _layer_layout(config.layout_seed, layer)
            qkv_singletons.add(layout.qkv_singleton)
            mlp_singletons.add(layout.mlp_singleton)
            gate_first_orders.add(layout.gate_first)
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
    assert qkv_singletons == {False, True}
    assert mlp_singletons == {False, True}
    assert gate_first_orders == {False, True}
