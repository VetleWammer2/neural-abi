# Optimizer-state format

NeuralABI 0.2 adds a safe optimizer bundle for fully initialized Adam and AdamW state. The bundle is
a directory: a bounded, schema-checked `optimizer.neuralabi.json` manifest and a SafeTensors state
store. NeuralABI never calls `torch.load`. It does not accept `.pt`, `.pth`, pickle, or an arbitrary
PyTorch optimizer `state_dict` as untrusted input.

The trusted Python bridge exports a live model and optimizer once the caller has built or loaded
them:

```python
from pathlib import Path

from neuralabi.optimizers.torch import export_optimizer_state

export_optimizer_state(model, optimizer, Path("source-optimizer"))
```

Loading a legacy PyTorch optimizer checkpoint, where that is needed, stays outside NeuralABI's
untrusted data path. The exporter associates state with parameters by Python object identity and
records every exact `state_dict` alias. Numeric parameter IDs from `Optimizer.state_dict()` and
parameter names are not used as matching heuristics.

## Bundle layout

```text
source-optimizer/
├── optimizer.neuralabi.json
└── state/
    └── model.safetensors             # single-file form
```

The sharded form replaces that file with numbered `model-*.safetensors` shards and
`model.safetensors.index.json`.

The manifest holds bounded JSON only:

- schema and format versions, `adam` or `adamw`
- complete parameter groups and their scalar options
- the model schema and checkpoint fingerprints
- one record per unique parameter, with all alias keys
- opaque SafeTensor keys for each state field
- the logical tensor fingerprint and the canonical bundle hash

State references such as `state.000000.exp_avg` are SafeTensor keys, not filesystem paths. The
tensor-key set must equal the referenced set. Unknown manifest fields, directory entries, tensors
and group/state fields are rejected.

## Supported state

Format version 1 supports `torch.optim.Adam` and `torch.optim.AdamW` with:

- one state record for every unique trainable parameter identity;
- scalar `step` plus `exp_avg` and `exp_avg_sq` SafeTensors;
- parameter groups written out, with JSON scalar hyperparameters;
- `amsgrad=False`, `capturable=False`, `differentiable=False`, with `fused` and `foreach` either
  `False` or `None` (`True` is rejected);
- a model state-schema hash and exact model-checkpoint fingerprint.

Every trainable parameter needs fully initialized state. Errors: missing parameters or fields,
unknown state keys, orphan payload tensors, wrong shapes or dtypes, non-scalar or invalid steps,
duplicate membership, unsupported group options. Schedulers and their state are not included.

## Mapping semantics

Plan schema v2 collapses tied model keys into unique source and target parameter identities. Each
target identity points at a model-parameter expression the conversion plan already has. NeuralABI
evaluates that same expression with its leaves rebound to `exp_avg`, then to `exp_avg_sq`. No
parameter names. No second transform language for optimizers.

The rule holds for the current grammar because every operation is a coordinate reindexing: reshape,
singleton changes, permutation/transpose, slice plus concat/stack, and interleave/deinterleave.
Those move individual coordinates without scaling or mixing them. A future transform that scales or
linearly combines values would need its own second-moment rule, and is outside optimizer-state
format v1.

Splitting one source parameter copies its step and parameter-group identity to each target piece.
Fusing several parameters is accepted only when every dependency sits in the same source group and
has a bitwise-equal step. Cross-group fusion, unequal steps, partial initialization and mixed
optimized/unoptimized dependencies cannot be represented by stock Adam, so they are rejected.

Tied aliases implemented by one shared `Parameter` object have one optimizer record and one runtime
optimizer slot. Equal-valued but distinct parameters are never aliases. Distinct `Parameter`
wrappers that merely share storage are rejected.[^identity]

## Verification contract

Optimizer conversion checks schema and complete unique-parameter coverage first. It then recomputes
the expected target state through the plan and compares `step`, `exp_avg` and `exp_avg_sq`.
Coordinate conversion and its inverse are expected to be bitwise exact.

Resumed training is a separate empirical claim. Verification runs in evaluation mode. It generates
inputs through each adapter with the same recorded seed and builds a seeded random linear probe
over the adapter-selected output tensors. The probe loss is compared before each synchronized
optimizer update. Outputs and the target physical model are compared after the update, against a
freshly converted updated source model. Every requested step must pass. The v1 CPU contract:

| dtype | absolute tolerance | relative tolerance |
|---|---:|---:|
| float32 | `1e-7` | `1e-5` |
| float64 | `1e-12` | `1e-10` |

The certificate records the PyTorch build, device/backend identifiers, eval mode, objective version,
seed derivation, comparison timing, dtype tolerances, and whether deterministic algorithms were
enabled. The contract is numerical. It is not a promise of bitwise-identical future training.
Optimizer conversion, coverage, tensor verification and resumed-training equivalence are reported
independently. Shape validation alone never establishes optimizer continuity.

## Version boundary

Plan schema v1 stays executable for parameter-only conversion. It carries no optimizer identity
mapping, so optimizer commands reject it. Plan schema v2, optimizer-state schema v1 and certificate
schema v2 form the NeuralABI 0.2 boundary.

Unsupported optimizer state:

- SGD and other algorithms
- AMSGrad
- sparse or partially initialized state
- tensor-valued hyperparameters
- capturable/differentiable modes
- explicit `fused=True` or `foreach=True`
- scheduler state
- distributed optimizer partitions
- unsafe serialized Python objects

Optimizer conversion materializes all source and converted moment tensors in memory. Only output
shard size is bounded. Distributed and out-of-core optimizer conversion are deferred.

[^identity]: Not a policy choice. PyTorch optimizers attach state to the Parameter object.
