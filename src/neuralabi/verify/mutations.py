"""Generic semantic plan mutations used to demonstrate verification sensitivity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from neuralabi.formats.plan import ConversionPlan, PlannedTensor
from neuralabi.ir.semantic import CanonicalModel
from neuralabi.status import PlanValidationError
from neuralabi.transforms import TransformExpr, Transpose, expr_from_dict


def mutate_semantic_bindings(
    plan: ConversionPlan,
    source_model: CanonicalModel,
    target_model: CanonicalModel,
    *,
    target_key: str,
    replacements: Mapping[str, str],
) -> ConversionPlan:
    source_views = {view.semantic_id: view.decode for view in source_model.views()}
    layouts = [layout for layout in target_model.layouts if layout.physical_key == target_key]
    if len(layouts) != 1:
        raise PlanValidationError(
            f"target mutation requires one physical layout for {target_key!r}"
        )
    semantic_expressions = dict(source_views)
    for destination, source in replacements.items():
        if destination not in semantic_expressions or source not in source_views:
            raise PlanValidationError("mutation references an unknown semantic slot")
        semantic_expressions[destination] = source_views[source]
    layout = layouts[0]
    expression = layout.encode(semantic_expressions)
    previous = plan.targets[target_key]
    targets = dict(plan.targets)
    targets[target_key] = PlannedTensor(
        expression, previous.shape, previous.dtype, previous.semantic_slots
    )
    return replace(plan, targets=targets, plan_hash="").with_hash()


def mutate_square_transpose(plan: ConversionPlan, *, target_key: str) -> ConversionPlan:
    previous = plan.targets[target_key]
    if len(previous.shape) != 2 or previous.shape[0] != previous.shape[1]:
        raise PlanValidationError("transpose mutation requires a square rank-two target")
    targets = dict(plan.targets)
    targets[target_key] = replace(previous, expression=Transpose(previous.expression, (0, 1)))
    return replace(plan, targets=targets, plan_hash="").with_hash()


def mutate_interleave_groups(
    plan: ConversionPlan, *, target_key: str, groups: int
) -> ConversionPlan:
    previous = plan.targets[target_key]
    data = previous.expression.to_dict()
    changed = False

    def visit(value: Any) -> None:
        nonlocal changed
        if not isinstance(value, dict):
            return
        if value.get("op") == "interleave" and not changed:
            value["groups"] = groups
            changed = True
            return
        if "input" in value:
            visit(value["input"])
        for child in value.get("inputs", []):
            visit(child)

    visit(data)
    if not changed:
        raise PlanValidationError(f"target {target_key!r} has no interleave operation")
    expression: TransformExpr = expr_from_dict(data)
    expression.infer_spec(plan.source_tensors)
    targets = dict(plan.targets)
    targets[target_key] = replace(previous, expression=expression)
    return replace(plan, targets=targets, plan_hash="").with_hash()
