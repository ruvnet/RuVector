# RuGraph Runtime

**Native graph and model execution for the RuGraph programming language.**

[Language guide](../ruvector-language/README.md) · [Specification](../../docs/research/graph-language/SPEC.md)

This standalone crate connects the RuGraph language to the real `ruvector-graph`
and `ruvector-kge` crates through path dependencies. It provides `rvg`, the native
language CLI, and `rvg-model`, an offline model lifecycle CLI. Rust 1.90 or newer
is required. Run these commands from the RuVector repository root:

```sh
cargo build --release --locked --manifest-path crates/ruvector-language-native/Cargo.toml
cargo test --locked --manifest-path crates/ruvector-language-native/Cargo.toml
crates/ruvector-language-native/target/release/rvg --help
```

`rvg` supports the same `check`, `compile`, `run`, `replay` and `verify` commands
as `rvg-core`, adding actual native model execution when `--models` is supplied.
The language runtime itself has no filesystem access or ambient clock. Host CLI
code reads the explicitly supplied files.

## Capture a real graph

```rust
use ruvector_graph::{GraphDB, NodeBuilder};
use ruvector_language_native::capture_quiescent;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = GraphDB::new();
    graph.create_node(NodeBuilder::new().id("api").label("Service").build())?;
    // The application must exclude EVERY graph writer for this whole call.
    let snapshot = capture_quiescent(&graph, "revision_1", vec![], vec![])?;
    println!("{}", serde_json::to_string(&snapshot)?);
    Ok(())
}
```

Capture requires an application owned writer gate. A borrowed `&GraphDB` is not
a transaction because GraphDB permits writes through shared references. Count
checks catch some races but cannot establish isolation. Never describe this API
as an atomic database snapshot without the caller's exclusion guarantee.

The supported profile has one label per node, binary edges, and no edge properties
or hyperedges. Unsupported shapes fail explicitly. Node properties are converted
to JSON; a finite `FloatArray` property named `embedding` also becomes the node
vector. Observations and rules are supplied explicitly, never manufactured from
arbitrary properties. Capture checks the 100,000 record limit before graph scans
and validates the resulting snapshot with the core runtime. Borrowed graph visitors
check identities, nesting and a 100,000 unit property/identity budget before copying
payloads. One UTF8 byte or primitive property consumes at least one unit. Visitors
hold read guards, so their callbacks must not mutate or reenter the same GraphDB.

## Train, evaluate and promote

The checked in example is a three entity toy graph with one held out triple. It
exercises the full process but establishes no real world prediction accuracy.
The training request supplies disjoint `train` and `evaluation` triples, seed,
dimensions, epochs, learning rate and work budget. Native `Trainer::fit` updates
the actual KGE tables. Both HolE and RotatE are supported. Vocabulary is learned
from training; evaluation with unseen entities or relations is rejected.

```sh
crates/ruvector-language-native/target/release/rvg-model train \
  crates/ruvector-language-native/examples/training.json candidate.json

crates/ruvector-language-native/target/release/rvg-model promote \
  candidate.json 0.0 models.json

crates/ruvector-language-native/target/release/rvg-model evaluate \
  models.json incident_v1 \
  crates/ruvector-language-native/examples/evaluation.json evaluation_report.json
```

Every output must be a new path. Existing files are never overwritten. Promotion
produces a new bundle; it does not replace a running model. The explicit MRR floor
`0.0` above is suitable only for a wiring demonstration. Select a meaningful floor
using independent application validation before deployment. The CLI reports the
initial and trained metrics but does not promise improvement over initialization
or an incumbent. Repeatedly tuning against this evaluation set makes it validation
data; reserve separate final confirmation data for a quality claim.

Evaluation uses the native filtered rank evaluator on both heads and tails with
seeded RANDOM tie breaking. Training sees only training triples. Evaluation truth
is used only to exclude other known positives while ranking. Reports include MR,
MRR, Hits@1/3/10, query counts, epoch losses, seed, and hashes of the request,
training split, evaluation split and model artifact. Hashes provide integrity,
not producer authentication. Only promote artifacts from a trusted execution or
independently rerun their evaluation. This pipeline tests transductive link
prediction, not transfer to unseen graphs.

The native bundle format is `{ "version": 1, "models": [...] }`. Each model
includes a name, revision, scorer, power of two dimension, ordered entity and
relation vocabularies, flat weight tables and explicit imported or trained origin.
Input JSON rejects duplicate keys and unknown fields. The loader caps input at
16 MiB, bundles at 16 models and all tables at 1,000,000 weights. Model dimensions
range from 2 through 1024; training dimensions from 2 through 128. Offline training
and evaluation are capped at one billion deterministic work units.

## Predict and replay

Create `predict.rvg`:

```text
program rank {
  effect graph.read;
  effect model.predict;
  effect decision.emit;
  budget 100000;
  let subject = select ["api"];
  let candidates = predict subject "Cause" model "incident_v1" top 3;
  emit candidates;
}
```

Supply a valid snapshot containing `api` and context granting the three declared
capabilities, with explicit `now`, `max_steps` and `max_results`:

```sh
crates/ruvector-language-native/target/release/rvg run predict.rvg \
  --snapshot snapshot.json --context context.json --models models.json
```

For replay, store the canonical program, snapshot and context in a `RunRequest`
JSON object and retain the receipt and exact model bundle:

```sh
crates/ruvector-language-native/target/release/rvg replay \
  --request request.json --receipt receipt.json --models models.json
```

Ranking enumerates every entity in the selected model vocabulary and calls the
actual native scorer. Ranking buffers retain score and entity index pairs; only
the selected rows copy identity strings. A conservative projection checks the
16 MiB output bound before scoring or allocating the candidate array. It includes
escaped string sizes and owned row storage, using the longest vocabulary identity
as the object bound. Unknown subject or relation identities fail. Work is
preflighted before scoring; exhaustion produces an error, never a partial ranking.
Returned scores express plausibility, not probabilities. `exact: true` means
exhaustive ranking using the native f32 scorer, not exact real arithmetic or
guaranteed truth. Receipts retain the model revision and SHA256 of its complete
artifact. Identical artifacts and inputs replay deterministically within the same
execution environment; cross platform FFT bit equality is not guaranteed.

## Verification and measurement

Integration tests compare both scorers and trained weights directly with native
KGE calls, capture a real GraphDB, exercise expiry, reject training leakage and
malformed artifacts, and run the CLI train, promote, evaluate, predict and replay
workflow in separate processes.

```sh
cargo run --release --locked \
  --manifest-path crates/ruvector-language-native/Cargo.toml --example native_bench
```

The benchmark checks equal candidate scores and ranking before measuring direct
native scoring, the adapter, and the full language runtime. It uses synthetic
seeded weights solely to measure execution. The direct baseline omits language
validation, permissions and receipt hashing. Results are workload and machine
specific; rerun on the intended deployment target.
