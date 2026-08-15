"""Offline tiny Hugging Face Llama versus the opaque fused runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from examples.generated_transformer.models import LogicalConfig, _FusedDecoder

_CONFIG = LogicalConfig(
    seed=311,
    layout_seed=0,
    layers=1,
    vocabulary_size=97,
    hidden_size=32,
    attention_heads=4,
    key_value_heads=2,
    intermediate_size=48,
    sequence_length=7,
    bias=False,
)


class HuggingFaceLlamaAdapter:
    adapter_id = "offline-huggingface-tiny-llama"

    def build(self, *, device: torch.device | str, dtype: torch.dtype) -> nn.Module:
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError as exc:
            raise RuntimeError("install neuralabi[llama] for the tiny Llama integration") from exc
        torch.manual_seed(_CONFIG.seed)
        config = LlamaConfig(
            vocab_size=_CONFIG.vocabulary_size,
            hidden_size=_CONFIG.hidden_size,
            intermediate_size=_CONFIG.intermediate_size,
            num_hidden_layers=_CONFIG.layers,
            num_attention_heads=_CONFIG.attention_heads,
            num_key_value_heads=_CONFIG.key_value_heads,
            head_dim=_CONFIG.head_dim,
            max_position_embeddings=32,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            attention_dropout=0.0,
            tie_word_embeddings=True,
            use_cache=False,
            return_dict=True,
        )
        config._attn_implementation = "eager"
        return LlamaForCausalLM(config).to(device=device, dtype=dtype)

    def prepare(self, model: nn.Module) -> None:
        model.eval()
        model.config.use_cache = False  # type: ignore[attr-defined]

    def example_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        generator = torch.Generator().manual_seed(seed + 364)
        tokens = torch.randint(
            0,
            _CONFIG.vocabulary_size,
            (2, _CONFIG.sequence_length),
            generator=generator,
        ).to(device)
        return (tokens,), {}

    def select_outputs(self, output: Any) -> Any:
        return output.logits if hasattr(output, "logits") else output[0]

    def dynamic_shapes(self) -> Any | None:
        return None

    def differentiable_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        return None


class FusedLlamaAdapter:
    adapter_id = "opaque-fused-tiny-llama"

    def build(self, *, device: torch.device | str, dtype: torch.dtype) -> nn.Module:
        return _FusedDecoder(_CONFIG, dtype).to(device=device)

    def prepare(self, model: nn.Module) -> None:
        model.eval()

    def example_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        generator = torch.Generator().manual_seed(seed + 364)
        tokens = torch.randint(
            0,
            _CONFIG.vocabulary_size,
            (2, _CONFIG.sequence_length),
            generator=generator,
        ).to(device)
        return (tokens,), {}

    def select_outputs(self, output: Any) -> Any:
        return output

    def dynamic_shapes(self) -> Any | None:
        return None

    def differentiable_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        return None


source_adapter = HuggingFaceLlamaAdapter()
target_adapter = FusedLlamaAdapter()
