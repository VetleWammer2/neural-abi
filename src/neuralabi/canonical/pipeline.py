"""Central entry point for versioned graph canonicalization."""

from __future__ import annotations

from dataclasses import replace

from neuralabi.canonical.rewrites import canonicalize_target
from neuralabi.ir.graph import TensorNode, TensorProgram


def canonicalize_program(program: TensorProgram) -> TensorProgram:
    rules: set[str] = set(program.rewrite_rules_used)
    nodes: list[TensorNode] = []
    for node in program.nodes:
        canonical_op, rule = canonicalize_target(node.target)
        if rule:
            rules.add(rule)
        nodes.append(replace(node, canonical_op=canonical_op))
    return replace(program, nodes=tuple(nodes), rewrite_rules_used=tuple(sorted(rules)))
