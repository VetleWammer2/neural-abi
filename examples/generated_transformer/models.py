"""Seeded opaque decoder implementations used for layout-matrix testing.

The target implementation necessarily knows how it consumes its own physical tensors. The linker
does not import, inspect, or receive that layout metadata; it sees only the exported tensor program
and persistent-state schema.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LogicalConfig:
    seed: int = 11
    layout_seed: int = 29
    layers: int = 2
    vocabulary_size: int = 97
    hidden_size: int = 32
    attention_heads: int = 4
    key_value_heads: int = 2
    intermediate_size: int = 48
    sequence_length: int = 7
    bias: bool = False
    activation: Literal["silu", "gelu"] = "silu"

    def __post_init__(self) -> None:
        if not 1 <= self.layers <= 3:
            raise ValueError("generated decoder supports 1-3 layers")
        if not 16 <= self.hidden_size <= 96:
            raise ValueError("hidden size must be between 16 and 96")
        if not 2 <= self.attention_heads <= 8:
            raise ValueError("attention head count must be between 2 and 8")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden size must divide evenly across attention heads")
        if self.attention_heads % self.key_value_heads:
            raise ValueError("attention heads must divide evenly across KV heads")
        if self.head_dim % 2:
            raise ValueError("rotary head dimension must be even")
        if not 2 <= self.sequence_length <= 16:
            raise ValueError("sequence length must be between 2 and 16")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.attention_heads


def _opaque(seed: int, index: int, prefix: str = "n") -> str:
    digest = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:10]
    return f"{prefix}{digest}"


def _parameter(
    shape: tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype, scale: int
) -> nn.Parameter:
    return nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype) / scale)


class _Shelf(nn.Module):
    def __init__(
        self,
        specs: tuple[tuple[int, ...], ...],
        *,
        seed: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self._names: list[str] = []
        for index, shape in enumerate(specs):
            name = _opaque(seed, index)
            self._names.append(name)
            setattr(
                self,
                name,
                _parameter(shape, generator=generator, dtype=dtype, scale=max(4, shape[-1])),
            )

    def at(self, index: int) -> nn.Parameter:
        return getattr(self, self._names[index])


def _rms(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + 1e-5)).to(x.dtype) * scale


def _rotary(x: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    sequence = x.shape[-2]
    positions = torch.arange(sequence, device=x.device, dtype=torch.float32)
    frequencies = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim
    angles = positions[:, None] / torch.exp(math.log(10_000.0) * frequencies[None, :])
    doubled = torch.cat((angles, angles), dim=-1)
    cos = torch.cos(doubled).to(x.dtype)
    sin = torch.sin(doubled).to(x.dtype)
    first, second = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos[None, None, :, :] + rotated * sin[None, None, :, :]


def _attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attention_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    repeat = attention_heads // key_value_heads
    q = _rotary(q, head_dim=head_dim)
    k = _rotary(k, head_dim=head_dim)
    if repeat != 1:
        k = torch.repeat_interleave(k, repeat, dim=1)
        v = torch.repeat_interleave(v, repeat, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    sequence = q.shape[-2]
    causal = torch.ones((sequence, sequence), device=q.device, dtype=torch.bool).triu(1)
    probabilities = torch.softmax(scores.masked_fill(causal, torch.finfo(scores.dtype).min), dim=-1)
    return torch.matmul(probabilities, v)


class _SeparateBlock(nn.Module):
    def __init__(self, config: LogicalConfig, *, layer: int, dtype: torch.dtype) -> None:
        super().__init__()
        h = config.hidden_size
        q = config.attention_heads * config.head_dim
        kv = config.key_value_heads * config.head_dim
        m = config.intermediate_size
        weight_shapes = ((q, h), (kv, h), (kv, h), (h, q), (m, h), (m, h), (h, m), (h,), (h,))
        bias_shapes = ((q,), (kv,), (kv,), (h,), (m,), (m,), (h,)) if config.bias else ()
        self._s = _Shelf(weight_shapes + bias_shapes, seed=config.seed + layer * 101, dtype=dtype)
        self._config = config

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        c = self._config
        offset = 9
        pre = _rms(hidden, self._s.at(7))
        q = F.linear(pre, self._s.at(0), self._s.at(offset) if c.bias else None)
        k = F.linear(pre, self._s.at(1), self._s.at(offset + 1) if c.bias else None)
        v = F.linear(pre, self._s.at(2), self._s.at(offset + 2) if c.bias else None)
        q = q.reshape(q.shape[0], q.shape[1], c.attention_heads, c.head_dim).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], c.key_value_heads, c.head_dim).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], c.key_value_heads, c.head_dim).transpose(1, 2)
        context = _attention_core(
            q,
            k,
            v,
            attention_heads=c.attention_heads,
            key_value_heads=c.key_value_heads,
            head_dim=c.head_dim,
        )
        context = context.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], c.hidden_size)
        projected = F.linear(context, self._s.at(3), self._s.at(offset + 3) if c.bias else None)
        after_attention = hidden + projected
        pre_mlp = _rms(after_attention, self._s.at(8))
        gate = F.linear(pre_mlp, self._s.at(4), self._s.at(offset + 4) if c.bias else None)
        up = F.linear(pre_mlp, self._s.at(5), self._s.at(offset + 5) if c.bias else None)
        activated = F.silu(gate) if c.activation == "silu" else F.gelu(gate)
        down = F.linear(activated * up, self._s.at(6), self._s.at(offset + 6) if c.bias else None)
        return after_attention + down


class _SeparateDecoder(nn.Module):
    def __init__(self, config: LogicalConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self._config = config
        names = [_opaque(config.seed + 8000, index, "m") for index in range(config.layers)]
        self._order = names
        self._blocks = nn.ModuleDict(
            {
                name: _SeparateBlock(config, layer=index, dtype=dtype)
                for index, name in enumerate(names)
            }
        )
        generator = torch.Generator().manual_seed(config.seed + 9001)
        embedding = _parameter(
            (config.vocabulary_size, config.hidden_size),
            generator=generator,
            dtype=dtype,
            scale=config.hidden_size,
        )
        self._entry = nn.Module()
        setattr(self._entry, _opaque(config.seed, 700), embedding)
        self._entry_name = _opaque(config.seed, 700)
        self._exit = nn.Module()
        setattr(self._exit, _opaque(config.seed, 701), embedding)
        self._exit_name = _opaque(config.seed, 701)
        self._final = _Shelf(((config.hidden_size,),), seed=config.seed + 9021, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedding = getattr(self._entry, self._entry_name)
        hidden = F.embedding(token_ids, embedding)
        for name in self._order:
            hidden = self._blocks[name](hidden)
        hidden = _rms(hidden, self._final.at(0))
        output_weight = getattr(self._exit, self._exit_name)
        return F.linear(hidden, output_weight)


@dataclass(frozen=True)
class _LayerLayout:
    qkv_order: tuple[str, str, str]
    interleaved: bool
    gate_first: bool
    qkv_singleton: bool
    mlp_singleton: bool


def _layer_layout(seed: int, layer: int) -> _LayerLayout:
    rng = random.Random(seed + layer * 313)
    orders = tuple(itertools.permutations(("a", "b", "c")))
    return _LayerLayout(
        orders[rng.randrange(len(orders))],
        bool(rng.randrange(2)),
        bool(rng.randrange(2)),
        bool(rng.randrange(2)),
        bool(rng.randrange(2)),
    )


class _FusedBlock(nn.Module):
    def __init__(self, config: LogicalConfig, *, layer: int, dtype: torch.dtype) -> None:
        super().__init__()
        self._config = config
        self._layout = _layer_layout(config.layout_seed, layer)
        h = config.hidden_size
        q = config.attention_heads * config.head_dim
        kv = config.key_value_heads * config.head_dim
        total = q + 2 * kv
        m = config.intermediate_size
        qkv_shape = (1, h, total) if self._layout.qkv_singleton else (h, total)
        mlp_shape = (h, 2 * m, 1) if self._layout.mlp_singleton else (h, 2 * m)
        weight_shapes = (qkv_shape, (q, h), mlp_shape, (m, h), (h,), (h,))
        bias_shapes = ((total,), (h,), (2 * m,), (h,)) if config.bias else ()
        self._s = _Shelf(
            weight_shapes + bias_shapes,
            seed=config.seed + 20_000 + layer * 109,
            dtype=dtype,
        )

    def _qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self._config
        layout = self._layout
        q_heads_per_group = c.attention_heads // c.key_value_heads
        counts = {"a": q_heads_per_group, "b": 1, "c": 1}
        flat_sizes = {
            "a": c.attention_heads * c.head_dim,
            "b": c.key_value_heads * c.head_dim,
            "c": c.key_value_heads * c.head_dim,
        }
        matrix = self._s.at(0).reshape(c.hidden_size, sum(flat_sizes.values()))
        bias = self._s.at(6).reshape(-1) if c.bias else None
        packed = F.linear(hidden, matrix.transpose(0, 1), bias)
        if layout.interleaved:
            grouped = packed.reshape(
                hidden.shape[0],
                hidden.shape[1],
                c.key_value_heads,
                q_heads_per_group + 2,
                c.head_dim,
            )
            pieces = torch.split(grouped, [counts[item] for item in layout.qkv_order], dim=-2)
            selected = {item: pieces[index] for index, item in enumerate(layout.qkv_order)}
            q = selected["a"].reshape(
                hidden.shape[0], hidden.shape[1], c.attention_heads, c.head_dim
            )
            k = selected["b"].reshape(
                hidden.shape[0], hidden.shape[1], c.key_value_heads, c.head_dim
            )
            v = selected["c"].reshape(
                hidden.shape[0], hidden.shape[1], c.key_value_heads, c.head_dim
            )
        else:
            pieces = torch.split(packed, [flat_sizes[item] for item in layout.qkv_order], dim=-1)
            selected = {item: pieces[index] for index, item in enumerate(layout.qkv_order)}
            q = selected["a"].reshape(
                hidden.shape[0], hidden.shape[1], c.attention_heads, c.head_dim
            )
            k = selected["b"].reshape(
                hidden.shape[0], hidden.shape[1], c.key_value_heads, c.head_dim
            )
            v = selected["c"].reshape(
                hidden.shape[0], hidden.shape[1], c.key_value_heads, c.head_dim
            )
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        c = self._config
        pre = _rms(hidden, self._s.at(4))
        q, k, v = self._qkv(pre)
        context = _attention_core(
            q,
            k,
            v,
            attention_heads=c.attention_heads,
            key_value_heads=c.key_value_heads,
            head_dim=c.head_dim,
        )
        context = context.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], c.hidden_size)
        output_matrix = self._s.at(1)
        output_bias = self._s.at(7) if c.bias else None
        after_attention = hidden + F.linear(context, output_matrix.transpose(0, 1), output_bias)
        pre_mlp = _rms(after_attention, self._s.at(5))
        packed_matrix = self._s.at(2).reshape(c.hidden_size, 2 * c.intermediate_size)
        packed_bias = self._s.at(8).reshape(-1) if c.bias else None
        packed = F.linear(pre_mlp, packed_matrix.transpose(0, 1), packed_bias)
        pieces = torch.split(packed, [c.intermediate_size, c.intermediate_size], dim=-1)
        gate = pieces[0] if self._layout.gate_first else pieces[1]
        up = pieces[1] if self._layout.gate_first else pieces[0]
        down_matrix = self._s.at(3)
        down_bias = self._s.at(9) if c.bias else None
        activated = F.silu(gate) if c.activation == "silu" else F.gelu(gate)
        return after_attention + F.linear(activated * up, down_matrix.transpose(0, 1), down_bias)


class _FusedDecoder(nn.Module):
    def __init__(self, config: LogicalConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self._config = config
        names = [_opaque(config.layout_seed + 40_000, index, "z") for index in range(config.layers)]
        self._order = names
        self._cavern = nn.Module()
        self._cavern.rooms = nn.ModuleDict(
            {
                name: _FusedBlock(config, layer=index, dtype=dtype)
                for index, name in enumerate(names)
            }
        )
        generator = torch.Generator().manual_seed(config.seed + 30_001)
        embedding = _parameter(
            (config.vocabulary_size, config.hidden_size),
            generator=generator,
            dtype=dtype,
            scale=config.hidden_size,
        )
        self._mouth = nn.Module()
        setattr(self._mouth, _opaque(config.layout_seed, 800), embedding)
        self._mouth_name = _opaque(config.layout_seed, 800)
        self._sky = nn.Module()
        setattr(self._sky, _opaque(config.layout_seed, 801), embedding)
        self._sky_name = _opaque(config.layout_seed, 801)
        self._last = _Shelf(((config.hidden_size,),), seed=config.seed + 30_021, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.embedding(token_ids, getattr(self._mouth, self._mouth_name))
        for name in self._order:
            hidden = self._cavern.rooms[name](hidden)
        hidden = _rms(hidden, self._last.at(0))
        return torch.matmul(hidden, getattr(self._sky, self._sky_name).transpose(0, 1))


class GeneratedAdapter:
    def __init__(self, config: LogicalConfig, side: Literal["source", "target"]) -> None:
        self.config = config
        self.side = side
        self.adapter_id = f"generated-{side}-{config.seed}-{config.layout_seed}"

    def build(self, *, device: torch.device | str, dtype: torch.dtype) -> nn.Module:
        model: nn.Module = (
            _SeparateDecoder(self.config, dtype)
            if self.side == "source"
            else _FusedDecoder(self.config, dtype)
        )
        return model.to(device=device)

    def prepare(self, model: nn.Module) -> None:
        model.eval()

    def example_inputs(
        self, *, seed: int, device: torch.device | str
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        generator = torch.Generator().manual_seed(seed + self.config.seed + 53)
        tokens = torch.randint(
            0,
            self.config.vocabulary_size,
            (2, self.config.sequence_length),
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


_DEFAULT = LogicalConfig()
default_source_adapter = GeneratedAdapter(_DEFAULT, "source")
default_target_adapter = GeneratedAdapter(_DEFAULT, "target")
