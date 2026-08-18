# v0.2 limitations

The implemented recognizer is intentionally narrow. It supports statically exportable,
decoder-only eager-attention graphs with embedding, explicit RMSNorm, rotary sine/cosine, causal
mask plus softmax, MHA/GQA, linear or explicit matmul projections, flat or per-KV-group fused QKV,
SwiGLU, residuals, final norm, and output projection. Projection storage may be rank two with
harmless singleton dimensions. Tied or untied embeddings are represented, provided both sides have
the same semantics.

Currently unsupported include fused SDPA/custom attention operators, non-positional semantic-anchor
probes, data-dependent control flow, non-static feature/head dimensions, LayerNorm, GELU decoder
blocks, learned absolute positions, MoE, quantization, tensor/pipeline/expert parallel shards,
pickle checkpoints, and architecture resizing. CUDA verification is implemented through the device
option but was not tested for this release; no GPU claim is made.

Optimizer-state linking is limited to fully initialized Adam and AdamW `step`, `exp_avg`, and
`exp_avg_sq` state in the safe NeuralABI optimizer format. AMSGrad, sparse/partial state, scheduler
state, tensor-valued hyperparameters, capturable/differentiable modes, explicit `fused=True` or
`foreach=True`, cross-group fusion, other optimizer algorithms, and distributed optimizer
partitions are unsupported and rejected. Generated converters can link optimizer tensors but do
not establish resumed-training
equivalence; that claim requires adapter-backed verification over at least one recorded optimizer
step. Optimizer conversion materializes the complete source and target moment state in memory even
when the emitted tensor store is sharded.

Recognition rejects missing evidence rather than falling back to names or shapes. Numerically
indistinguishable candidate plans remain `AMBIGUOUS` after the bounded probe suite.
