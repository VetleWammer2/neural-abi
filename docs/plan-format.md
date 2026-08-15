# Conversion plan format

Schema version 1 contains endpoint adapter IDs, graph and state-schema hashes, the logical source
checkpoint fingerprint, architecture signature, complete source and target tensor schemas,
target-centric expressions, semantic-slot dependencies, aliases, derived state, assumptions,
name-hint mode, status, complexity, synthesis statistics, candidate hashes, ambiguity details, and
an optional exact inverse.

The plan hash is SHA-256 over canonical JSON excluding only the hash field itself. Parsing is
bounded by byte size, nesting, expression depth, node count, rank, dimensions, and tensor count.
Unknown fields and operations, duplicate JSON keys, invalid shapes, alias cycles, incomplete target
coverage, and a mismatched plan hash are rejected before execution.

`neuralabi apply` validates source keys, shapes, dtypes, and logical checkpoint fingerprint without
importing adapters. A `UNIQUE` status is required for execution.
