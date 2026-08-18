# Changelog

## 0.2.0 — unreleased

- Link fully initialized Adam and AdamW state through plan-v2 parameter identities and the existing
  coordinate-reindex transform expressions.
- Add a SafeTensors-only optimizer bundle, strict coverage/schema validation, and explicit rejection
  of pickle and unsupported optimizer fields.
- Verify optimizer tensors and semantic associations separately from deterministic resumed-training
  equivalence under a recorded numerical contract.
- Keep plan schema v1 loadable for parameter-only conversion.

## 0.1.0 — unreleased

- Initial NeuralABI linker, converter, and compatibility verifier.
