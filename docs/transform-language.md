# Transform language

Plans contain a bounded JSON AST. Operations are `source`, `identity`, `alias`, `reshape`,
`permute`, `transpose`, `slice`, `concat`, `stack`, `squeeze`, `unsqueeze`, `interleave`, and
`deinterleave`.

`interleave` takes an axis, group count, and logical segment sizes. It splits each logical segment
evenly across groups and emits each group's components together. `deinterleave` is its exact
inverse. This expresses per-KV-group QKV packing without arbitrary code.

Every node performs shape inference before tensor data is read, preserves dtype, has a deterministic
cost, and serializes canonically. Unary bijections have symbolic inverses. Reverse-mode adjoints
cover all operations, including zero-padding a sliced gradient and splitting concatenated
gradients. There is no `eval`, callable, cast, or user-defined operation.

The grammar carries `coordinate-reindex-v1` optimizer semantics: operations move whole tensor
coordinates and never scale or mix their values. So Adam and AdamW `exp_avg` and `exp_avg_sq`
follow the same forward expression as their parameter. Adjoints stay the gradient-verification
path, not a general second-moment rule. A future non-reindex transform has to define its own
optimizer semantics before it can carry optimizer state.

Complexity weights are identity 0; reshape/singleton 1; transpose/permute/slice 2; concat/stack 3;
interleave/deinterleave 4; and one point per additional physical source dependency.
