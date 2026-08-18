# Compatibility certificate

Certificate schema version 2 records endpoint identities and hashes, architecture, rules and
assumptions, complete target coverage, probe seeds and input signatures, numeric tolerances,
per-tensor forward and intermediate errors, parameter and optional input gradients, exact
round-trip results, the first topological semantic divergence, generated-file hashes, and separate
claim statuses.

It also records conversion scope, unique-parameter optimizer coverage and semantic associations,
per-field optimizer tensor comparisons, and synchronized resumed-training evidence. Separate claims
state whether parameter state and optimizer state were converted, whether optimizer coverage was
complete, whether the optimizer tensors verified, and whether every requested resumed step
satisfied the recorded numerical contract. The contract identifies the seeded verifier objective,
device/backend, eval mode, seed policy, comparison timing, and dtype tolerances. When no optimizer
bundle is supplied, optimizer and resume claims are `NOT_APPLICABLE`; they are never inferred from
parameter shapes.

Claim statuses are `VERIFIED`, `FAILED`, `INCONCLUSIVE`, and `NOT_APPLICABLE`. A normal successful
run requires structural coverage, forward probes, aligned semantic intermediates, physical
parameter gradients, and an exact round trip when the plan is bijective. Integer token inputs make
input gradients not applicable. Bitwise forward equivalence is reported only when observed.

The certificate is not a mathematical proof. It means structural compatibility under NeuralABI's
documented canonicalization rules, plus empirical verification over the recorded probe suite.
