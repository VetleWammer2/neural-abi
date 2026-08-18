# NeuralABI

A linker and ABI verifier for neural checkpoints.

Two implementations can compute the same transformer while storing its weights in incompatible
physical layouts. NeuralABI reads both exported tensor programs and infers the mapping. It emits a
standalone converter and verifies the converted model. Plan v2 carries fully initialized Adam and
AdamW state through those same parameter expressions, and can check synchronized seeded training
against a recorded numerical contract.

Output from the checked-in generated transformer. CPU, PyTorch 2.13.0, opaque parameter and module
names, name hints off:

```console
$ neuralabi link \
    --source examples.generated_transformer.models:default_source_adapter \
    --target examples.generated_transformer.models:default_target_adapter \
    --source-checkpoint artifacts/readme-run/source.safetensors \
    --output artifacts/readme-run/output \
    --name-hints off \
    --seeds 0,1,2,3
Source graph: generated-source-11-29
Target graph: generated-target-11-29
Mapping status: UNIQUE
Plan complexity: 64
Generated: artifacts\readme-run\output
  PARAMETER_STATE_CONVERTED       VERIFIED
  PARAMETER_STATE_COVERAGE_COMPLETE VERIFIED
  STATE_COVERAGE_COMPLETE         VERIFIED
  STRUCTURAL_ALIGNMENT            VERIFIED
  FORWARD_VERIFIED                VERIFIED
  INTERMEDIATE_VERIFIED           VERIFIED
  PARAMETER_GRADIENT_VERIFIED     VERIFIED
  INPUT_GRADIENT_VERIFIED         NOT_APPLICABLE
  ROUNDTRIP_EXACT                 VERIFIED
  BITWISE_FORWARD_EQUIVALENT      VERIFIED
  OPTIMIZER_STATE_CONVERTED       NOT_APPLICABLE
  OPTIMIZER_COVERAGE_COMPLETE     NOT_APPLICABLE
  OPTIMIZER_STATE_VERIFIED        NOT_APPLICABLE
  RESUMED_TRAINING_EQUIVALENT     NOT_APPLICABLE
```

For its first decoder layer the inferred target expression took three unrelated source keys in
semantic V/Q/K order, reshaped canonical head axes, concatenated them, transposed storage and added
a singleton physical axis:

```text
3 separate [out, in] source tensors
    ↓ semantic Q / K / V recognition
reshape heads → V,Q,K concat → transpose → singleton reshape
    ↓
1 fused [1, in, qkv] target tensor
```

No fixture key map exists in the inference core. Q, K and V come from score/value dataflow; gate
and up come from the SiLU/multiply branches; physical order comes from the exported split use.

## Names and graphs

Names say where a tensor is stored. Graphs say what it means. A regex renamer cannot explain
transposed linear storage, fused projections, QKV permutations, GQA interleaving, singleton axes or
fused SwiGLU. Shape-only filling is worse: a wrong K/V permutation is often shape-compatible.

NeuralABI captures both implementations with strict `torch.export` and recognizes a canonical
decoder semantic graph. It derives physical views on both sides, then composes

```text
source physical → source semantic → inverse target view → target physical
```

The result is a deterministic, inspectable JSON program with no executable code.

## Example

From a checkout, create the deterministic example checkpoint and link it:

```console
$ python -m pip install -e ".[dev]"
$ python - <<'PY'
from pathlib import Path
import torch
from safetensors.torch import save_file
from examples.generated_transformer.models import default_source_adapter

model = default_source_adapter.build(device="cpu", dtype=torch.float32)
Path("artifacts/demo").mkdir(parents=True, exist_ok=True)
save_file({k: v.detach().clone() for k, v in model.state_dict().items()},
          "artifacts/demo/source.safetensors")
PY
$ neuralabi link \
    --source examples.generated_transformer.models:default_source_adapter \
    --target examples.generated_transformer.models:default_target_adapter \
    --source-checkpoint artifacts/demo/source.safetensors \
    --output artifacts/demo/linked \
    --name-hints off
```

The output directory holds the analysis, plan, standalone converter, conversion manifest,
certificate, human report, and the target SafeTensors as a single file or shards.

Each stage is also its own command, on the same code path:

```console
neuralabi scan --source MODULE:ADAPTER --target MODULE:ADAPTER --name-hints off
neuralabi infer --source MODULE:ADAPTER --target MODULE:ADAPTER \
  --source-checkpoint source.safetensors --output plan.neuralabi.json --name-hints off
neuralabi apply plan.neuralabi.json source.safetensors --output target/
neuralabi verify plan.neuralabi.json --source MODULE:ADAPTER --target MODULE:ADAPTER \
  --source-checkpoint source.safetensors --target-checkpoint target/ --seeds 0,1,2,3
neuralabi explain plan.neuralabi.json
neuralabi emit plan.neuralabi.json --format python --output converter.py
```

## Install

Requirements:

- Linux, or another PyTorch-supported development platform
- Python 3.11+
- PyTorch `>=2.6,<3`, CPU
- SafeTensors

The v0.2 CPU validation recorded here used PyTorch 2.13.0+cpu. The offline Hugging Face Llama
integration lives in the optional `llama` dependency group:

```console
python -m pip install ".[llama]"
```

CUDA can be selected for capture and probes. This release did not test CUDA and makes no GPU
compatibility claim.

## Adam and AdamW state

Optimizer input is NeuralABI's bounded JSON plus SafeTensors bundle. Never pickle. Export a trusted
live optimizer only after its model has been loaded and stepped, and only if that model exactly
matches the source checkpoint the plan was inferred from:

```python
from pathlib import Path
from neuralabi.optimizers.torch import export_optimizer_state

export_optimizer_state(model, optimizer, Path("artifacts/demo/source-optimizer"))
```

Load the converted weights into the target implementation, then build its optimizer from the safe
bundle:

```python
from neuralabi.optimizers.torch import load_optimizer_state

target_optimizer = load_optimizer_state(target_model, Path("target-optimizer"))
```

Convert and verify both states together:

```console
neuralabi apply plan.neuralabi.json source.safetensors --output target/ \
  --source-optimizer-state source-optimizer/ --optimizer-output target-optimizer/
neuralabi verify plan.neuralabi.json --source MODULE:SOURCE --target MODULE:TARGET \
  --source-checkpoint source.safetensors --target-checkpoint target/ \
  --source-optimizer-state source-optimizer/ --target-optimizer-state target-optimizer/ \
  --resume-steps 2
```

`neuralabi link` accepts `--source-optimizer-state` and defaults to two resumed steps when it is
present. The generated standalone converter takes the paired `--source-optimizer` and
`--optimizer-output` arguments. Its conversion is deterministic. Resumed-training evidence comes
only from adapter-backed `verify` or `link`.

Format v1 converts `step`, `exp_avg` and `exp_avg_sq` exactly, for fully initialized
`torch.optim.Adam` and `torch.optim.AdamW`. Tied aliases get one optimizer slot. Splitting copies
the step; fusion needs equal steps and one source parameter group.[^fuse] Unknown or missing state
and unsupported options fail closed. Details in the
[optimizer-state format](docs/optimizer-state-format.md).

## Adapters

Adapters are narrow. They build a model, prepare deterministic eval behavior, produce reproducible
inputs and select comparable outputs. They may offer differentiable input probes. They cannot
supply parameter mappings or layout instructions.

```python
class MyAdapter:
    adapter_id = "my-local-decoder"

    def build(self, *, device, dtype): ...
    def prepare(self, model):
        model.eval()
        model.config.use_cache = False
    def example_inputs(self, *, seed, device): ...
    def select_outputs(self, output): return output.logits
    def dynamic_shapes(self): return None
    def differentiable_inputs(self, *, seed, device): return None
```

Model adapters execute arbitrary user-provided Python code. Only run adapters you trust.

## Supported model family

The implemented recognizer covers statically exportable decoder-only eager-attention graphs with:

- token embedding and language-model output projection, including tied aliases;
- explicit RMSNorm and rotary sine/cosine application;
- causal self-attention, softmax, MHA, and GQA expansion;
- separate or fused Q/K/V with all six orders;
- flat or per-KV-group interleaved QKV packing;
- `[out, in]` or `[in, out]` projection storage and singleton dimensions;
- separate or fused SwiGLU gate/up in either order;
- attention/MLP output projections, residuals, final RMSNorm, and optional projection biases.

The test suite carries 32 deterministic generated layouts: one to three layers, MHA and GQA, all
QKV orders, both head layouts, both gate/up orders, singleton axes, biases, opaque names. The
offline tiny Llama test builds a random local config and downloads nothing.

## Capture and synthesis

`torch.export.export(..., strict=True)` is mandatory. NeuralABI functionalizes the exported program
with an empty decomposition table, preserves graph-signature state bindings and freezes a portable
typed tensor program. The canonicalizer applies a versioned registry of conservative operator
rules. Recognizers reject missing evidence rather than labeling arbitrary linears as attention.

Physical layouts decode into stable semantic IDs such as `model.layer[0].attention.query.weight`.
The deterministic solver aligns equal slots and composes target encoders. It verifies global state
coverage and aliases, minimizes the documented transform cost, and records the bounded candidate
budget. When several structural candidates survive, execution probes and earliest semantic
divergence refine them; more than one survivor is `AMBIGUOUS`.

Names have modes `off` and `weak`. Weak names may only break a tie after semantic, shape, topology,
and physical-view constraints. They never override a contradiction.

## Plan and standalone converter

A plan embeds:

- complete source and target schemas, endpoint hashes, source checkpoint fingerprint
- target expressions, semantic dependencies, aliases
- assumptions, cost, statistics, ambiguity
- an exact inverse when one exists

Schema v2 adds unique trainable-parameter identities with all their alias keys, and points
optimizer state at target expressions that already exist. Canonical JSON gives the plan its stable
SHA-256 hash. Schema-v1 plans stay parameter-only.

The generated converter embeds that data and a small checked interpreter:

```console
$ python converter.py --source source/model.safetensors --output converted/ \
    --source-optimizer source-optimizer/ --optimizer-output converted-optimizer/ \
    --max-shard-size 2GB
```

It imports the standard library, PyTorch and SafeTensors. Nothing else: not NeuralABI, not either
model, not either adapter. It reads and writes single or sharded checkpoints, validates
fingerprints, keeps deterministic key order, checks paths and installs output atomically. Model
conversion holds one bounded output shard. Optimizer conversion holds the whole source and
converted moment state in memory before writing bounded output shards.

## Verification levels

A certificate reports each of these separately:

- parameter and optimizer conversion scope and coverage
- structural alignment
- forward pytree comparison
- aligned semantic intermediates
- physical parameter gradients through inverse-plan adjoints
- optional input gradients
- exact inverse round-trip
- optimizer tensor equality
- synchronized resumed steps
- observed bitwise forward equivalence

NaN/Inf counts and dtype-aware tolerances stay explicit. Mutation tests swap Q/K, K/V, gate/up,
transpose axes, interleave stride, layer source and one dependency. Each is rejected and localized
to the first topological semantic divergence.

A certificate is not a formal proof. It records structural compatibility under NeuralABI's
documented canonicalization rules plus empirical verification over the recorded probes.

## Trust model

Adapters and the live-optimizer exporter are trusted code. Plans, certificates, optimizer
manifests, indexes, metadata, keys and output paths are untrusted data. No `eval`, no pickle, no
shell strings, no remote code, no network calls. JSON and expression sizes are bounded. Rejected:
duplicate keys, unknown operations or state, cycles, path traversal, fingerprint mismatch,
uncovered target state, and any attempt to overwrite an input.

## Benchmarks

Benchmark commands emit machine-readable JSON and make no unmeasured claims:

```console
python -m benchmarks.synthesis --layers 2 --hidden 32 --heads 4 --kv-heads 2
python -m benchmarks.conversion --tensors 8 --dimension 1024 --max-shard-size 16777216
```

They record graph nodes, semantic slots, candidates and phase durations, plus checkpoint/shard
sizes, throughput and output disk use. Peak RSS too, when the platform exposes it.

## Limitations and roadmap

v0.2 does not handle:

- architecture changes
- fused SDPA and custom kernels
- LayerNorm/GELU decoders
- unrelated position families
- MoE, quantization
- distributed checkpoint or optimizer partitions
- optimizer algorithms outside the documented Adam/AdamW subset
- pickle checkpoints
- unrestricted control flow

See [limitations](docs/limitations.md) for the precise boundary.

Next milestone: more decoder graph families. Tensor parallelism, quantization, MoE, broader
optimizer variants and cross-framework adapters stay deferred until this narrow
model-plus-optimizer loop stays reliable.

## Docs

- [TECHNICAL_NOTE.md](TECHNICAL_NOTE.md)
- [ABI model](docs/abi-model.md)
- [adapter API](docs/adapter-api.md)
- [canonicalization](docs/canonicalization.md)
- [transform language](docs/transform-language.md)
- [plan format](docs/plan-format.md)
- [optimizer-state format](docs/optimizer-state-format.md)
- [certificate](docs/certificate.md)
- [trust boundary](docs/trust-model.md)

## Errata

- v0.1 planned optimizer linking on the transform adjoints. Moments use the forward expression.
- "Bounded shard memory" covered model conversion only. Optimizer conversion holds every moment in
  memory.
- LICENSE held a 17-line notice, not the license terms. The full text is there now.

[^fuse]: Not chosen. Stock Adam keeps one step per parameter.
