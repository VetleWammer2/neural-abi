# Contributing

NeuralABI accepts changes that preserve deterministic inference, safe declarative plans, and the
distinction between physical keys and semantic graph roles. Run formatting, linting, type checks,
the complete test suite, and package build before submitting a change. New canonicalization rules
need positive and negative tests. New transforms need shape, execution, serialization, inverse,
and adjoint tests where those operations are defined.

Do not commit downloaded checkpoints, pickle model files, fixture-specific key mappings, or
network-dependent tests.

