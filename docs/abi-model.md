# Neural ABI model

Checkpoint keys locate physical tensors; exported tensor-program use sites establish semantic
roles. NeuralABI therefore links two implementations through canonical slots instead of matching
keys directly.

```mermaid
flowchart LR
  SP[Source physical state] --> SV[Source physical views]
  SV --> CS[Canonical semantic slots]
  CS --> TV[Inverse target views]
  TV --> TP[Target physical state]
```

Canonical decoder weights use `[vocab, hidden]` embeddings, `[heads, head_dim, hidden]` Q/K/V,
`[hidden, heads, head_dim]` attention output, `[intermediate, hidden]` gate/up,
`[hidden, intermediate]` down, and `[hidden]` RMSNorm scales. A physical layout records matrix
orientation, fused component order, optional per-KV-group interleaving, singleton storage axes,
state aliases, and the supporting exported nodes.

The linker composes each source physical decoder with the inverse target physical view. Every
persistent target key must occur in the plan's embedded target schema, as a generated expression or
an explicit alias. Source state must be consumed or accounted for through a physical alias.

Optimizer state binds to unique parameter identities, not to checkpoint-key spelling and not to
PyTorch optimizer integer IDs. A tied identity may expose several model-state keys and still own one
Adam state record. For each target identity the plan reuses its parameter expression for both
moments. Scalar steps and group membership follow the fusion and splitting rules in the
[optimizer-state format](optimizer-state-format.md).
