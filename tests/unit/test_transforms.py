from __future__ import annotations

import pytest
import torch

from neuralabi.status import PlanValidationError
from neuralabi.transforms import (
    Concat,
    Deinterleave,
    Interleave,
    Permute,
    Reshape,
    Slice,
    Source,
    Squeeze,
    Stack,
    TensorSpec,
    Transpose,
    Unsqueeze,
    expr_from_dict,
)
from neuralabi.transforms.adjoint import expression_adjoint
from neuralabi.transforms.execute import execute_checked
from neuralabi.transforms.inverse import inverse_expression


def test_all_primitives_serialize_and_execute() -> None:
    tensors = {
        "a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "b": torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
    }
    expressions = [
        Source("a"),
        Reshape(Source("a"), (2, 6)),
        Permute(Source("a"), (1, 0)),
        Transpose(Source("a"), (0, 1)),
        Slice(Source("a"), 1, 1, 4, 2),
        Concat((Source("a"), Source("b")), 0),
        Stack((Source("a"), Source("b")), 1),
        Squeeze(Unsqueeze(Source("a"), 0), 0),
        Interleave(Concat((Source("a"), Source("b")), 1), 1, 2, (4, 4)),
        Deinterleave(
            Interleave(Concat((Source("a"), Source("b")), 1), 1, 2, (4, 4)),
            1,
            2,
            (4, 4),
        ),
    ]
    for expression in expressions:
        round_tripped = expr_from_dict(expression.to_dict())
        assert round_tripped == expression
        assert torch.equal(round_tripped.apply(tensors), expression.apply(tensors))


def test_shape_errors_are_rejected_before_execution() -> None:
    specs = {"a": TensorSpec((3, 4), "float32")}
    invalid = [
        Reshape(Source("a"), (5, 5)),
        Permute(Source("a"), (0, 0)),
        Squeeze(Source("a"), 0),
        Interleave(Source("a"), 1, 3, (2, 2)),
    ]
    for expression in invalid:
        with pytest.raises(PlanValidationError):
            expression.infer_spec(specs)


def test_unary_inverse_is_exact_with_singletons() -> None:
    source = torch.randn(1, 2, 3, dtype=torch.float64)
    specs = {"x": TensorSpec(tuple(source.shape), "float64")}
    expression = Permute(Reshape(Squeeze(Source("x"), 0), (3, 2)), (1, 0))
    output = execute_checked(expression, {"x": source})
    inverse = inverse_expression(expression, specs, output_key="y")
    recovered = execute_checked(inverse, {"y": output})
    assert torch.equal(recovered, source)


def test_interleave_deinterleave_are_bitwise_inverses() -> None:
    source = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    expression = Interleave(Source("x"), 1, 2, (4, 2, 2))
    physical = expression.apply({"x": source})
    recovered = Deinterleave(Source("y"), 1, 2, (4, 2, 2)).apply({"y": physical})
    assert torch.equal(source, recovered)


def test_concat_slice_adjoint_matches_autograd() -> None:
    a = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(2, 2, requires_grad=True)
    expression = Transpose(Concat((Slice(Source("a"), 1, 0, 3), Source("b")), 1), (0, 1))
    output = expression.apply({"a": a, "b": b})
    probe = torch.randn_like(output)
    (output * probe).sum().backward()
    specs = {"a": TensorSpec((2, 3), "float32"), "b": TensorSpec((2, 2), "float32")}
    adjoints = expression_adjoint(expression, probe, specs)
    assert torch.equal(adjoints["a"], a.grad)
    assert torch.equal(adjoints["b"], b.grad)


def test_zero_length_shapes_where_torch_permits_them() -> None:
    tensor = torch.empty(2, 0, 3)
    expression = Reshape(Source("x"), (0, 6))
    result = execute_checked(expression, {"x": tensor})
    assert result.shape == (0, 6)


def test_unknown_operation_is_rejected() -> None:
    with pytest.raises(PlanValidationError, match="unknown transform"):
        expr_from_dict({"op": "python_eval", "code": "1 + 1"})
