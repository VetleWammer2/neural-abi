# NeuralABI technical note

## Neural state has an ABI

Two programs may implement the same mathematical decoder but disagree about the physical contract
of its persistent state. Keys and module paths locate objects; graph uses establish roles, axes,
packing, and alias semantics. This resembles a linker reconciling two binary interfaces more than
a key-renaming utility.

Let $G_s$ and $G_t$ be source and target tensor programs, $\theta_s$ and $\theta_t$ their physical
states, and $D_s$ and $D_t$ decoders into canonical semantic tensors. NeuralABI seeks

$$P = D_t^{-1} \circ D_s$$

such that

$$\theta_t = P(\theta_s)$$

and, for valid inputs,

$$F_s(\theta_s,x) \approx F_t(P(\theta_s),x).$$

The least-complex valid plan is

$$P^*=\arg\min_{P\in\mathcal G} C(P)$$

subject to compatible logical signatures, complete target state, accounted source state, exact
shape and dtype rules, aligned slots, consistent aliases, and configured verification checks.
$\mathcal G$ is deliberately small and typed. Numerical probes are validation evidence, not a
proof for every possible input.

## Physical and semantic state

Canonical decoder weights use these implemented shapes:

```text
embedding / output head     [vocabulary, hidden]
query                       [attention_heads, head_dim, hidden]
key / value                 [kv_heads, head_dim, hidden]
attention output            [hidden, attention_heads, head_dim]
SwiGLU gate / up            [intermediate, hidden]
SwiGLU down                 [hidden, intermediate]
RMSNorm scale               [hidden]
projection bias             [logical output]
```

A `PhysicalLayout` describes one stored tensor as an ordered collection of flattened semantic
components, an `out_in`, `in_out`, or vector orientation, optional group interleaving, singleton
storage shape, dtype, graph evidence, and axis lineage. Its decoder produces one `PhysicalView` per
slot. Its encoder performs the inverse composition for a target tensor.

For GQA interleaving, the logical contiguous vector `[Q_all,K_all,V_all]` is divided across KV
groups. Each group emits its Q heads followed by that group's K and V in the physical component
order. `Interleave` and `Deinterleave` encode this as a checked permutation with group and segment
divisibility preconditions. Equal overall tensor shapes therefore do not erase layout identity.

Actual object/storage identity establishes state aliases. Equal values never do. A tied embedding
can support both embedding and output semantic views; duplicate physical encodings must be
identical, and every alias key remains explicit in the plan.

## Export and canonical graph

Strict `torch.export` is the only complete frontend in v0.2. Capture builds and prepares the model,
creates deterministic inputs, exports, runs an empty decomposition table to functionalize local
mutation without destroying high-level ATen patterns, and rejects graph-signature mutation of
parameters, buffers, or user inputs.

The frozen artifact contains a normalized node program, parameter/buffer bindings, complete
persistent state schema, aliases, ranges, input signature, options, PyTorch version, and graph hash.
Raw `ExportedProgram` does not flow through recognition and synthesis. A bound exported GraphModule
is retained only for recording verification anchors.

Canonicalization rules have stable IDs and exact operator preconditions. They normalize linear,
matmul/addmm, shape views, explicit axis changes and splits, embedding, RMS primitives, softmax,
and safe elementwise forms. Floating reordering is not applied merely because an identity is
usually algebraically true.

## Semantic recognition

The decoder recognizer starts from invariant computation boundaries:

1. one embedding establishes vocabulary and hidden axes;
2. topological softmax nodes delimit decoder attention blocks;
3. the nearest upstream score matmul establishes Q and K roles;
4. the nearest downstream matmul establishes the V aggregation role;
5. pre-expansion rank-four axes recover logical KV heads;
6. sine/cosine ancestry proves rotary application;
7. explicit fill or finite-minimum additive masks prove causal masking;
8. the output projection and residual addition close attention;
9. SiLU feeding multiplication establishes gate and up branches;
10. the following projection and residual close SwiGLU;
11. mean-square/rsqrt ancestry and a rank-one state use prove RMSNorm;
12. the final norm and vocabulary projection close the decoder.

Repeated blocks are ordered by softmax topology, not module names. A projection operand's graph use
establishes matrix orientation. A branch's getitem/split path establishes fused order. A rank-five
group split establishes interleaving and group count. Missing evidence produces `UNSUPPORTED`.

The logical signature contains vocabulary, hidden size, layer count, attention and KV head counts,
head dimension, intermediate size, normalization, activation, position family, projection bias
configuration, and tying semantics. A field-level mismatch stops before synthesis.

## Constraints, cost, and candidates

Source physical views decode every slot. Target layouts state how those slots encode one physical
tensor. Composition is target-centric and deterministic. Global checks require exactly one view per
semantic slot, complete embedded target schema, accounted source state, valid component dimensions,
monotonic layer order, invertible target views, consistent aliases, and exact inverse metadata.

The implemented cost is identity 0; reshape and singleton 1; transpose, permutation, and slice 2;
concat and stack 3; interleave/deinterleave 4; plus one per additional distinct source dependency.
Candidate hashes are canonical JSON hashes, enumeration order is lexical by hash, and the default
budget is 128. Name hints are off or a weak late tie-breaker; current structural examples need none.

CEGIS refinement applies each structural candidate in memory, executes deterministic probes, and
records the earliest topological semantic divergence. A cache keys evidence by plan and probe hash.
One survivor is unique; none is unsatisfiable; multiple indistinguishable survivors are ambiguous.
The ambiguity fixture includes a symmetric checkpoint for which two physical candidates really do
survive the configured probe. NeuralABI does not guess.

## Transform inverses and adjoints

Every transform has static shape inference, checked execution, stable serialization, and cost.
Unary bijections construct exact symbolic inverses. A complete plan also stores a target-to-source
composition when the collection of split/fuse operations is bijective.

For an expression $y=P(x)$ and scalar loss $L$, reverse mode uses

$$\nabla_x L = P^T \nabla_y L.$$

Reshape and singleton operations invert shape; permutations use inverse axes; slice adjoints place
the gradient into zeros; concat adjoints split; stack adjoints unbind; interleave adjoints
deinterleave. To compare target physical gradients with source gradients, NeuralABI applies the
adjoint of the stored inverse plan. Tied parameters are deduplicated so accumulated logical
gradients are not counted twice.

## Optimizer state as an ABI

Plan schema v2 records every unique trainable source and target parameter identity, with all exact
`state_dict` aliases. A target identity references one of the plan's existing parameter expressions
and the source identities on which it depends. This makes optimizer association structural and
explicit; PyTorch optimizer integer IDs and parameter-name similarity do not participate.

For the current transform grammar, let $P$ be the global physical coordinate reindexing. Adam's
first and diagonal second moments convert as

$$m_t=P(m_s), \qquad v_t=P(v_s).$$

The same forward expression is evaluated twice with source leaves rebound to the matching moment.
This is not an application of the gradient adjoint and is not a generic rule for arbitrary linear
transforms: it is valid because reshape, permutation, singleton, split/concat, and interleave
operations move coordinates without scaling or mixing them. Plan validation constrains optimizer
targets to this grammar and requires exact inverse coverage.

One source parameter may split into several targets, which copy its scalar step and group. Several
source parameters may fuse only if their steps are bitwise equal and they belong to one source
parameter group; otherwise stock Adam cannot represent their continuation in one target parameter.
Tied aliases own one state record. Coverage is measured over these unique identities.

The safe optimizer format is a strict bounded JSON manifest plus a single or sharded SafeTensors
store. It binds algorithm, groups, identity aliases, tensor metadata, logical tensor fingerprint,
model schema, and exact model-checkpoint fingerprint. A trusted bridge exports or restores live
`torch.optim.Adam`/`AdamW` objects by Python parameter identity. Untrusted `.pt`/`.pth` and pickle
are never loaded.

## Checkpoints and generated converters

The logical checkpoint fingerprint streams sorted key, shape, dtype, and raw contiguous bytes, so
it is independent of physical sharding. Readers accept one SafeTensors file or a validated
Hugging Face-style index. Index keys, shard basenames, actual shard contents, duplicate ownership,
and missing files are checked.

Application schedules one target at a time, reading only its source dependencies. The writer holds
at most one bounded output shard, writes deterministic key order and index JSON to a sibling
temporary directory, and atomically installs the completed output. One fixed SafeTensors metadata
entry avoids unordered metadata bytes. Inputs cannot be selected as outputs.

Optimizer application follows the same target expression order for `exp_avg` and `exp_avg_sq`,
validates complete state and scalar-step/group rules, and emits a separately atomic safe bundle.
Unlike streamed model conversion, optimizer format v1 materializes the complete source and
converted moment state before writing bounded output shards. The generated converter embeds the
same strict optimizer interpreter without importing NeuralABI or model code.

Code generation embeds the same plan and a compact interpreter. It needs only Python, PyTorch, and
SafeTensors; it does not load adapters, model implementations, NeuralABI, pickle, or arbitrary code.

## Verification semantics

Structural verification checks signatures, slots, full state coverage, source consumption, aliases,
schemas, and supported graph alignment. Forward verification recursively compares selected output
pytrees across recorded seeds with dtype-aware tolerances and explicit NaN/Inf counts.

Semantic anchors include embedding output; per-layer normalized states, canonical Q/K/V, scores,
attention output and residual, gate/up branches, MLP output and layer result; final normalized state;
and logits. Results retain all comparisons while the human diagnosis points to the first
topological divergence.

Parameter gradients use a deterministic scalar probe and the inverse-plan adjoint. Input gradients
run only when both adapters supply aligned differentiable inputs; integer token IDs are not
differentiated. Bijective state plans require bitwise source recovery. Bitwise forward equivalence
is a separate observed claim, not a success prerequisite.

Optimizer verification recomputes every target moment through the plan and requires bitwise tensor
and step equality. Resumed-training verification is deliberately separate: it reloads the converted
model and optimizer states and executes synchronized seeded updates in evaluation mode. Its loss is
a seeded linear probe over adapter-selected output tensors and is compared before each update;
outputs and target physical state are compared after each update, with state checked against a
freshly converted updated source state.
Float32 uses `atol=1e-7, rtol=1e-5`; float64 uses `atol=1e-12, rtol=1e-10`. Certificates record
observed errors and exactness but do not claim bitwise-identical continued training.

A certificate is therefore not formal equivalence for unrestricted networks. It states structural
compatibility under these documented recognition and rewrite rules plus empirical verification on
the recorded probes. Failures can arise from unsupported export, missing structural evidence,
architecture mismatch, unresolved symmetry, floating tolerance, environment execution, malformed
untrusted data, or a real semantic difference.
