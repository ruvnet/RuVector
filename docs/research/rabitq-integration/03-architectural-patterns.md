# 03 — Architectural Patterns

Three sane shapes to add a new RaBitQ consumer. Each preserves the
ADR-154 / ADR-155 / ADR-157 invariants and matches a different
consumer profile. The choice is consequential because every shape
implies a different contract about who owns the index, who owns the
witness, and who picks the kernel.

The goal of this section is to make the choice explicit at integration
time, so we don't accidentally fragment what is currently one
deterministic compression substrate.

---

## Pattern 1 — Direct embed

The consumer crate adds `ruvector-rabitq` to its `Cargo.toml` and uses
`RabitqPlusIndex` (or any of the four indexes) as a private field of
its own type. The consumer owns the index lifecycle: build, add,
search, persist.

**Sketch.**

```toml
# Cargo.toml of consumer
[dependencies]
ruvector-rabitq = { path = "../ruvector-rabitq", version = "2.2" }
```

```rust
// inside consumer
use ruvector_rabitq::{AnnIndex, RabitqPlusIndex};

pub struct VectorPropertyIndex {
    by_property: HashMap<PropertyKey, RabitqPlusIndex>,
    seed: u64,
    rerank_factor: usize,
}

impl VectorPropertyIndex {
    pub fn add_node(&mut self, node_id: NodeId, property: PropertyKey, vector: Vec<f32>) {
        self.by_property
            .entry(property)
            .or_insert_with(|| RabitqPlusIndex::new(self.dim, self.seed, self.rerank_factor))
            .add(node_id.0, vector)
            .unwrap();
    }

    pub fn knn(&self, property: &PropertyKey, q: &[f32], k: usize) -> Vec<NodeId> {
        self.by_property
            .get(property)
            .map(|idx| idx.search(q, k).unwrap())
            .unwrap_or_default()
            .into_iter()
            .map(|r| NodeId(r.id))
            .collect()
    }
}
```

**Best when.** The consumer owns its index lifecycle, doesn't need
witness chaining, and doesn't need to share the index across processes
or backends. Examples in §02:

- **A1 — `ruvector-diskann`:** the index *is* the consumer's product;
  it manages its own SSD-backed storage and its own rebuild policy.
  RaBitQ is a backend choice, not a foreign service.
- **A2 — `ruvector-graph`:** the property index is a sub-component of
  a graph database that already owns its lifecycle.
- **A3 — `ruvector-gnn`:** the candidate set passed to
  `differentiable_search` is owned by the GNN forward pass; building a
  fresh `RabitqPlusIndex` per layer is fine for inference and the
  index is short-lived.
- **B4 — `ruvector-domain-expansion`:** the embedding store is
  internal state, no cross-crate sharing required.

**What this pattern doesn't give you.** The witness chain. Cross-
process cache sharing. Pluggable kernels (you get whatever ships in
`ruvector-rabitq` proper, which today means `CpuKernel`).

---

## Pattern 2 — Behind the `VectorKernel` trait (ADR-157)

The consumer registers a `VectorKernel` implementation — typically the
default `CpuKernel`, optionally an SIMD or GPU one — and dispatches
queries through it. The trait shape is at
`crates/ruvector-rabitq/src/kernel.rs:78-126`:

```rust
pub trait VectorKernel: Send + Sync {
    fn id(&self) -> &str;
    fn caps(&self) -> KernelCaps;
    fn scan(&self, req: ScanRequest<'_>) -> Result<ScanResponse, RabitqError>;
}
```

`ScanRequest` carries a borrowed `&RabitqPlusIndex` plus a query
batch; the consumer (or a coordinator) picks the kernel based on
batch size + dim + determinism requirement.

**Sketch.** A consumer that wants pluggable backends keeps an
`Arc<dyn VectorKernel>` field and calls `.scan(...)` in the hot
path:

```rust
use ruvector_rabitq::{CpuKernel, ScanRequest, VectorKernel};

pub struct AcceleratedSearcher {
    kernel: Arc<dyn VectorKernel>,
    // …
}

impl AcceleratedSearcher {
    pub fn new() -> Self {
        Self { kernel: Arc::new(CpuKernel::new()) }
    }
    pub fn register_kernel(&mut self, k: Arc<dyn VectorKernel>) {
        // ranked dispatch by caps()
        if self.should_prefer(&*k) { self.kernel = k; }
    }
    pub fn search(&self, idx: &RabitqPlusIndex, queries: &[Vec<f32>], k: usize)
        -> Result<ScanResponse, RabitqError>
    {
        self.kernel.scan(ScanRequest { index: idx, queries, k, rerank_factor: None })
    }
}
```

**Best when.** The consumer wants pluggable acceleration but doesn't
need cross-process witness/cache. Examples:

- **B1 — `ruvector-attention` KV cache:** wants SIMD on server, WASM
  SIMD in browser, GPU on a Cognitum box. Same source, different
  kernels. The trait was literally designed for this in ADR-157.
- **B2 — `ruvllm`:** if RaBitQ becomes the K-cache compression,
  ruvllm picks Metal or CUDA per platform.
- **C3 — `ruvector-fpga-transformer`:** an `RabitqFpgaKernel`
  registered at startup, with `caps().min_batch ≥ 1024` so it only
  fires on bulk inference.

**Critical caveat.** The trait is shipped (`src/kernel.rs`) but
**no caller wires it up today** — `ruvector-rulake` references it
only in a doc comment at `lake.rs:595`. The first consumer that
uses Pattern 2 must also write the dispatch policy (ADR-157
§"Dispatch policy normative") in its own crate; this is *not* free.
Roadmap Phase 2 (§05) is exactly this work.

---

## Pattern 3 — Through `ruLake`

The consumer doesn't manage a RaBitQ index at all. It delegates to a
`RuLake` instance with a `LocalBackend` (or a remote one) holding the
vectors, and calls `lake.search_one(backend, collection, query, k)`.

**Sketch.**

```rust
use ruvector_rulake::{LocalBackend, RuLake};

let backend = LocalBackend::with_vectors("agent-mem", "episodic", dim, vecs);
let lake = RuLake::builder()
    .register_backend(Arc::new(backend))
    .with_seed(42)
    .with_rerank_factor(20)
    .build()?;

let hits = lake.search_one("agent-mem", "episodic", &q, 10)?;
```

The consumer gets:

- 1.02× tax on the cache-hit path (measured —
  `crates/ruvector-rulake/BENCHMARK.md`).
- A SHAKE-256 witness chain via `RuLakeBundle`
  (`crates/ruvector-rulake/src/bundle.rs`).
- Cross-process cache sharing: two ruLake instances reading the same
  bundle reuse one compressed copy
  (`crates/ruvector-rulake/src/cache.rs`,
  test `two_backends_share_cache_when_witness_matches`).
- `Consistency::{Fresh, Eventual, Frozen}` knob for staleness SLA
  (ADR-156).
- Witness-by-reference for cross-agent handoff via
  `ArtifactKind::RuLakeWitness`
  (`crates/rvAgent/rvagent-a2a/src/artifact_types.rs:64`).

**Best when.** The consumer wants witness-sealed memory, cross-process
sharing, freshness modes, or zero-copy handoff to other agents.

- **B2 — `ruvllm` RAG:** any retrieval ruvllm does should sit on a
  `RuLake`, not on its own `RabitqPlusIndex` — gets the witness +
  freshness modes for free.
- **Any rvAgent subagent:** the agent memory hierarchy from ADR-156 is
  literally this pattern. Direct embed would re-implement bundle +
  witness; through-ruLake is "the brain on the substrate".
- **A future `ruvector-postgres` extension:** a Postgres function that
  returns top-k from a managed lake of vectors — ruLake is the right
  shape because the function may run in many backend processes
  sharing one cache.

**What this pattern doesn't give you.** Bare-metal min latency. The
1.02× tax is measured on `LocalBackend`; on a Parquet-on-GCS backend
the cold path is network-bound. Direct embed wins for in-process
single-user workloads where the consumer already has the vectors
materialised.

---

## Mapping §02 candidates to patterns

| Candidate | Pattern | Why |
|-----------|---------|-----|
| A1 `ruvector-diskann` | **1** direct embed | Owns its index lifecycle; SSD/PQ already custom |
| A2 `ruvector-graph` | **1** direct embed | Sub-index of an existing graph store |
| A3 `ruvector-gnn` | **1** direct embed | Short-lived per-layer index in forward pass |
| B1 `ruvector-attention` | **2** VectorKernel | Needs SIMD/GPU/WASM kernel choice per target |
| B2 `ruvllm` | **3** through ruLake (RAG) + **2** kernel (KV cache) | Two integrations, two patterns |
| B3 `ruvector-temporal-tensor` | **1** direct embed | New temperature tier inside the existing crate |
| B4 `ruvector-domain-expansion` | **3** through ruLake | Already produces witness-shaped outputs |
| C1, C4 (mincut, sparsifier) | downstream of A2 | n/a until A2 lands |
| C2 `ruvector-cnn` | none (user code) | Producers, not indexers |
| C3 `ruvector-fpga-transformer` | **2** VectorKernel | The kernel pattern's poster child |
| C5 `rvagent-a2a` | **3** (already, via witness) | Done |

Note the split for B2 — ruvllm probably wants both. That's fine; the
patterns compose.

---

## Anti-patterns to refuse

The following shapes look reasonable in a PR review but each one
breaks an existing ADR invariant or fragments the substrate. None of
them should pass review.

### Anti-pattern A — re-implementing rotation

A consumer crate copy-pastes the rotation code from
`crates/ruvector-rabitq/src/rotation.rs` into its own module to "avoid
the dependency". Breaks ADR-154's determinism guarantee — a divergent
copy means `(seed, dim, vectors) → bit-identical codes` no longer holds
across crates. **Always import from `ruvector-rabitq`.**

### Anti-pattern B — ad-hoc 1-bit compression

A consumer crate ships its own `pack_bits` function and its own
distance estimator because "we just need a quick binary code". This
re-creates the original `BinaryQuantized` problem ADR-154 §"Measured
gap" was written to fix: ~15–20% recall vs RaBitQ's 40.8%/98.9%. **If
you're doing 1-bit compression of vectors in this workspace, it's
RaBitQ.**

### Anti-pattern C — exposing `originals_flat`

A consumer crate's PR widens `RabitqPlusIndex` to expose its private
`originals_flat: Vec<f32>` field (`src/index.rs:546`) "for
zero-copy". Breaks the encapsulation that the persist format relies
on (`src/persist.rs:1-18`) — and the persist format is the contract
that lets two processes warm-load each other's bundles. The Python
SDK explicitly works around this at `src/rabitq.rs:35-43` by calling
`export_items()` instead. **Use `export_items()` or extend
`AnnIndex`; do not widen the struct.**

### Anti-pattern D — fragmenting the witness

A consumer crate runs RaBitQ to compress vectors and ships them under
a *different* witness scheme (e.g. its own SHA-3 over a private
serialization format). Breaks ADR-155 cross-backend cache sharing and
ADR-159 by-reference artifact handoff. **All compressed-vector
artifacts that traverse process boundaries use ruLake's
`RuLakeBundle` witness or none at all.**

### Anti-pattern E — RaBitQ everywhere by default

The mirror of D — adding `ruvector-rabitq` as a default dep on every
crate "because it's available". Adds ~50 KB compiled size and the
rotation tables to every WASM bundle and embedded build. The §04
performance budget is explicit: only candidate consumers with
demonstrated benefit (Tier A) get the dep on the default build path.
WASM consumers must feature-gate.

### Anti-pattern F — ignoring the kernel determinism gate

A consumer crate registers a non-deterministic GPU kernel and
serves Fresh/Frozen consistency from it. Breaks ADR-157 §"Determinism
as a hard gate". **Caps are advisory at compile time but enforced at
dispatch.** The consumer must implement the dispatch filter from
ADR-157, not just `kernels.iter().next()`.
