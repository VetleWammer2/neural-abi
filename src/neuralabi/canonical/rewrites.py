"""Versioned, conservative operator normalization rules."""

from __future__ import annotations

from dataclasses import dataclass

CANONICALIZATION_VERSION = 1


@dataclass(frozen=True)
class RewriteRule:
    rule_id: str
    targets: tuple[str, ...]
    canonical_op: str
    precondition: str


RULES: tuple[RewriteRule, ...] = (
    RewriteRule("linear.aten-linear.v1", ("aten.linear.default",), "linear", "exact ATen linear"),
    RewriteRule(
        "linear.matmul.v1",
        ("aten.matmul.default", "aten.mm.default"),
        "matmul",
        "exact matrix product",
    ),
    RewriteRule("linear.addmm.v1", ("aten.addmm.default",), "addmm", "exact ATen addmm"),
    RewriteRule(
        "shape.reshape-view.v1",
        ("aten.view.default", "aten.reshape.default", "aten._unsafe_view.default"),
        "reshape",
        "shape-only view with exported static rank",
    ),
    RewriteRule(
        "shape.permute.v1", ("aten.permute.default",), "permute", "explicit axis permutation"
    ),
    RewriteRule(
        "shape.transpose.v1",
        ("aten.transpose.int", "aten.t.default"),
        "transpose",
        "explicit transpose",
    ),
    RewriteRule(
        "shape.split.v1",
        ("aten.split_with_sizes.default", "aten.split.Tensor"),
        "split",
        "explicit split sizes",
    ),
    RewriteRule("shape.slice.v1", ("aten.slice.Tensor",), "slice", "explicit tensor slice"),
    RewriteRule(
        "shape.squeeze.v1",
        ("aten.squeeze.dim", "aten.squeeze.default"),
        "squeeze",
        "singleton removal",
    ),
    RewriteRule(
        "shape.unsqueeze.v1", ("aten.unsqueeze.default",), "unsqueeze", "singleton insertion"
    ),
    RewriteRule(
        "shape.contiguous.v1",
        ("aten.contiguous.default", "aten.clone.default"),
        "contiguous",
        "layout materialization only",
    ),
    RewriteRule("activation.silu.v1", ("aten.silu.default",), "silu", "exact SiLU"),
    RewriteRule(
        "activation.softmax.v1",
        ("aten.softmax.int", "aten._softmax.default"),
        "softmax",
        "exact softmax",
    ),
    RewriteRule(
        "elementwise.mul.v1", ("aten.mul.Tensor", "aten.mul.Scalar"), "mul", "exact multiplication"
    ),
    RewriteRule(
        "elementwise.add.v1", ("aten.add.Tensor", "aten.add.Scalar"), "add", "exact addition"
    ),
    RewriteRule("elementwise.pow.v1", ("aten.pow.Tensor_Scalar",), "pow", "exact scalar power"),
    RewriteRule("reduction.mean.v1", ("aten.mean.dim",), "mean", "exact dimension reduction"),
    RewriteRule(
        "normalization.rsqrt.v1", ("aten.rsqrt.default",), "rsqrt", "exact reciprocal square root"
    ),
    RewriteRule(
        "embedding.aten.v1", ("aten.embedding.default",), "embedding", "exact embedding lookup"
    ),
    RewriteRule("shape.expand.v1", ("aten.expand.default",), "expand", "broadcast view"),
    RewriteRule(
        "shape.repeat.v1",
        ("aten.repeat.default", "aten.repeat_interleave.self_int"),
        "repeat",
        "explicit repeat",
    ),
    RewriteRule("shape.cat.v1", ("aten.cat.default",), "concat", "explicit concatenation"),
    RewriteRule("shape.select.v1", ("aten.select.int",), "select", "explicit axis selection"),
    RewriteRule(
        "attention.mask.v1",
        ("aten.masked_fill.Scalar",),
        "masked_fill",
        "explicit scalar mask fill",
    ),
    RewriteRule("rotary.sin.v1", ("aten.sin.default",), "sin", "exact sine"),
    RewriteRule("rotary.cos.v1", ("aten.cos.default",), "cos", "exact cosine"),
)

_BY_TARGET = {target: rule for rule in RULES for target in rule.targets}


def canonicalize_target(target: str) -> tuple[str, str | None]:
    if target == "operator.getitem":
        return "getitem", "container.getitem.v1"
    rule = _BY_TARGET.get(target)
    return (rule.canonical_op, rule.rule_id) if rule else (target, None)
