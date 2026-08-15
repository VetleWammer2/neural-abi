# v0.1 limitations

The implemented recognizer is intentionally narrow. It supports statically exportable,
decoder-only eager-attention graphs with embedding, explicit RMSNorm, rotary sine/cosine, causal
mask plus softmax, MHA/GQA, linear or explicit matmul projections, flat or per-KV-group fused QKV,
SwiGLU, residuals, final norm, and output projection. Projection storage may be rank two with
harmless singleton dimensions. Tied or untied embeddings are represented, provided both sides have
the same semantics.

Currently unsupported include fused SDPA/custom attention operators, non-positional semantic-anchor
probes, data-dependent control flow, non-static feature/head dimensions, LayerNorm, GELU decoder
blocks, learned absolute positions, MoE, quantization, tensor/pipeline/expert parallel shards,
optimizer state, pickle checkpoints, and architecture resizing. CUDA verification is implemented
through the device option but was not tested for this release; no GPU claim is made.

Recognition rejects missing evidence rather than falling back to names or shapes. Numerically
indistinguishable candidate plans remain `AMBIGUOUS` after the bounded probe suite.
