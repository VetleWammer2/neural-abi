# Compatibility certificate

The versioned certificate records endpoint identities and hashes, architecture, rules and
assumptions, complete target coverage, probe seeds and input signatures, numeric tolerances,
per-tensor forward and intermediate errors, parameter and optional input gradients, exact
round-trip results, the first topological semantic divergence, generated-file hashes, and separate
claim statuses.

Claim statuses are `VERIFIED`, `FAILED`, `INCONCLUSIVE`, and `NOT_APPLICABLE`. A normal successful
run requires structural coverage, forward probes, aligned semantic intermediates, physical
parameter gradients, and an exact round trip when the plan is bijective. Integer token inputs make
input gradients not applicable. Bitwise forward equivalence is reported only when observed.

The certificate is not a mathematical proof. It means structural compatibility under NeuralABI's
documented canonicalization rules, plus empirical verification over the recorded probe suite.
