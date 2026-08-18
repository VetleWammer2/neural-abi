# Changelog

## 0.2.0 — unreleased

- Link fully initialized Adam and AdamW state through plan-v2 parameter identities, reusing the
  coordinate-reindex expressions the plan already carries.
- Add a SafeTensors-only optimizer bundle with strict coverage and schema validation. It rejects
  pickle and unsupported optimizer fields.
- Verify optimizer tensors and semantic associations separately from resumed training. Resume runs
  deterministic seeded steps and is reported against a recorded numerical contract.
- Keep plan schema v1 loadable for parameter-only conversion.

## 0.1.0 — unreleased

- Initial NeuralABI linker, converter, and compatibility verifier.
