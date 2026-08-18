# Conversion plan format

Schema version 2 contains:

- endpoint adapter IDs, graph and state-schema hashes
- the logical source checkpoint fingerprint and architecture signature
- complete source and target tensor schemas
- target-centric expressions, semantic-slot dependencies, aliases, derived state
- assumptions, name-hint mode, status, complexity
- synthesis statistics, candidate hashes, ambiguity details
- an optional exact inverse

It also records source and target trainable-parameter identities with every alias key. Each target
optimizer identity points at one parameter expression the plan already has, plus its source
identity dependencies. No optimizer transform is duplicated.

The plan hash is SHA-256 over canonical JSON, excluding only the hash field itself. Parsing is
bounded by byte size, nesting, expression depth, node count, rank, dimensions and tensor count.
Rejected before execution: unknown fields and operations, duplicate JSON keys, invalid shapes,
alias cycles, incomplete target coverage, a mismatched plan hash.

`neuralabi apply` validates source keys, shapes, dtypes and the logical checkpoint fingerprint
without importing adapters. Execution requires a `UNIQUE` status.

Schema version 1 stays loadable for parameter-only conversion. It does not separate parameter
identities from buffers and does not embed source aliases, so optimizer application rejects it.
Schema v2 optimizer mappings are held to the coordinate-reindex transform grammar. They require an
exact inverse, complete unique-parameter coverage, and consistent forward/inverse identity
dependencies.
