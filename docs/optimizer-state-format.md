# Optimizer-state format

NeuralABI 0.2 introduces a safe optimizer bundle for fully initialized Adam and AdamW state. The
bundle is a directory containing a bounded, strictly validated `optimizer.neuralabi.json` manifest
and a SafeTensors state store. NeuralABI never calls `torch.load` and does not accept `.pt`, `.pth`,
pickle, or an arbitrary PyTorch optimizer `state_dict` as untrusted input.

The trusted Python bridge exports a live model and optimizer after the caller has constructed or
loaded them:

```python
from pathlib import Path

from neuralabi.optimizers.torch import export_optimizer_state

export_optimizer_state(model, optimizer, Path("source-optimizer"))
```

Loading a legacy PyTorch optimizer checkpoint, if necessary, stays outside NeuralABI's untrusted
data path. The exporter associates state with parameters by Python object identity and records every
exact `state_dict` alias. Numeric parameter IDs from `Optimizer.state_dict()` and parameter names are
not used as matching heuristics.

## Bundle layout

```text
source-optimizer/
├── optimizer.neuralabi.json
└── state/
    └── model.safetensors             # single-file form
```

The sharded form replaces that file with numbered `model-*.safetensors` shards and
`model.safetensors.index.json`.

The manifest contains only bounded JSON data: schema/format versions, `adam` or `adamw`, complete
parameter groups and scalar options, the model schema and checkpoint fingerprints, unique parameter
records with all alias keys, opaque SafeTensor keys for each state field, the logical tensor
fingerprint, and the canonical bundle hash. State references such as `state.000000.exp_avg` are
SafeTensor keys, not filesystem paths. The actual tensor-key set must exactly equal the referenced
set; unknown manifest fields, directory entries, tensors, and group/state fields are rejected.

## Supported state

Format version 1 supports `torch.optim.Adam` and `torch.optim.AdamW` with:

- one state record for every unique trainable parameter identity;
- scalar `step` plus `exp_avg` and `exp_avg_sq` SafeTensors;
- explicit parameter groups and JSON scalar hyperparameters;
- `amsgrad=False`, `capturable=False`, `differentiable=False`, with `fused` and `foreach` either
  `False` or `None` (`True` is rejected);
- a model state-schema hash and exact model-checkpoint fingerprint.

All trainable parameters must have fully initialized state. Missing parameters or fields, unknown
state keys, orphan payload tensors, wrong shapes or dtypes, non-scalar/invalid steps, duplicate
membership, and unsupported group options are errors. Schedulers and their state are not included.

## Mapping semantics

Plan schema v2 collapses tied model keys into unique source and target parameter identities. Each
target identity points to an existing model-parameter expression in the conversion plan. NeuralABI
evaluates that same expression with its leaves rebound to `exp_avg`, then to `exp_avg_sq`; it does
not use parameter names and does not define a second optimizer transform language.

This rule is valid for the current transform grammar because every operation is a coordinate
reindexing: reshape, singleton changes, permutation/transpose, slice plus concat/stack, and
interleave/deinterleave. Such operations move individual coordinates without scaling or mixing
them. A future transform that scales or linearly combines values would require a new, explicit
second-moment rule and is outside optimizer-state format v1.

Splitting one source parameter copies its step and parameter-group identity to each target piece.
Fusing multiple parameters is accepted only when every dependency belongs to the same source group
and has a bitwise-equal step. Cross-group fusion, unequal steps, partial initialization, or mixed
optimized/unoptimized dependencies cannot be represented by stock Adam and are rejected.

Tied aliases implemented by one shared `Parameter` object have one optimizer record and one runtime
optimizer slot. Equal-valued but distinct parameters are never treated as aliases. Distinct
`Parameter` wrappers that merely share storage are rejected because PyTorch optimizers attach state
to object identity.

## Verification contract

Optimizer conversion first checks schema and complete unique-parameter coverage, then recomputes
the expected target state through the plan and compares `step`, `exp_avg`, and `exp_avg_sq`.
Coordinate conversion and its inverse are expected to be bitwise exact.

Resumed-training verification is a separate empirical claim. In evaluation mode it generates
inputs through each adapter with the same recorded seed and builds a seeded random linear probe over
the adapter-selected output tensors. The probe loss is compared before each synchronized optimizer
update; outputs and the target physical model are compared after the update against a freshly
converted updated source model. Every requested step must pass. The v1 CPU contract is:

| dtype | absolute tolerance | relative tolerance |
|---|---:|---:|
| float32 | `1e-7` | `1e-5` |
| float64 | `1e-12` | `1e-10` |

The certificate records the PyTorch build, device/backend identifiers, eval mode, objective version,
seed derivation, comparison timing, dtype tolerances, and whether deterministic algorithms were
enabled. The contract is numerical, not a promise of bitwise-identical future training.
Certificates report optimizer conversion, coverage, tensor verification, and resumed-training
equivalence independently; shape validation alone never establishes optimizer continuity.

## Version boundary and exclusions

Plan schema v1 remains executable for parameter-only conversion but contains no optimizer identity
mapping and is rejected by optimizer commands. Plan schema v2, optimizer-state schema v1, and
certificate schema v2 form the NeuralABI 0.2 boundary.

Unsupported optimizer state includes SGD and other algorithms, AMSGrad, sparse or partially
initialized state, tensor-valued hyperparameters, capturable/differentiable modes, explicit
`fused=True` or `foreach=True`, scheduler state, distributed optimizer partitions, and unsafe
serialized Python objects.

Optimizer conversion currently materializes all source and converted moment tensors in memory;
only output shard size is bounded. Distributed and out-of-core optimizer conversion are deferred.
