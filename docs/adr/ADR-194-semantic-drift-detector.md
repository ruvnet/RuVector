---
adr: 194
title: "Semantic Drift Detection and Spectral Memory Eviction for Agent Memory"
status: accepted
date: 2026-05-29
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193, ADR-189]
tags: [agent-memory, drift-detection, spectral, eviction, mmd, frechet, fiedler, nightly-research]
---

# ADR-194 — Semantic Drift Detection and Spectral Memory Eviction

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-29-semantic-drift-detector` as
`crates/ruvector-drift`.  All 20 unit and integration tests pass; build is green with
`cargo build --release -p ruvector-drift`.

---

## Context

RuVector's HNSW and DiskANN indexes assume a relatively stable vector distribution.
When agents run ruFlo workflow loops over hours or days, the working topic shifts —
a customer-support agent moves from billing questions to API questions; a coding agent
moves from Python to Rust.  Without detecting this shift:

1. Old embeddings remain in the index and degrade retrieval quality.
2. The HNSW graph accumulates low-utility nodes, increasing traversal cost.
3. DiskANN's page cache is polluted with stale entries.

Prior ruvector nightlies addressed index algorithms (RaBitQ compression,
ACORN filtered search, RAIRS IVF recall recovery) but none addressed **memory
lifecycle**: how to know *when* an index needs compaction and *which* vectors to remove.

Two research gaps exist:

**Gap A — Drift detection**: no Rust-native in-process drift detector exists for
vector index monitoring.  Python libraries (Alibi-Detect, EvidentlyAI) require an
out-of-process monitoring service.

**Gap B — Graph-aware eviction**: standard eviction policies (LRU, TTL) are unaware
of graph topology.  Evicting a structurally central node (high betweenness, many
k-NN connections) damages recall disproportionately compared to evicting a peripheral
node.  The Fiedler vector of the similarity graph provides a principled way to
identify peripheral nodes.

---

## Decision

We introduce `crates/ruvector-drift` implementing three drift detectors and three
eviction policies behind common traits:

### Drift detectors

| Detector | Algorithm | Cost per observation | Sensitivity |
|----------|-----------|---------------------|-------------|
| `CentroidDrift` | L2 centroid shift | O(D) | mean shift only |
| `MmdDrift` | MMD² with RBF kernel | O(S²·D) | full distributional shift |
| `FrechetDrift` | Diagonal Fréchet distance | O(W·D) | mean + variance shift |

All three satisfy `DriftDetector`:

```rust
pub trait DriftDetector {
    fn observe(&mut self, vector: &[f32]) -> DriftObservation;
    fn score(&self) -> f64;
    fn is_drifted(&self) -> bool;
    fn name(&self) -> &str;
    fn observations(&self) -> usize;
}
```

### Eviction policies

| Policy | Algorithm | Topologically aware |
|--------|-----------|-------------------|
| `RandomEviction` | Uniform random subset | No |
| `LruEviction` | Sort by last_access ascending | No |
| `SpectralEviction` | Fiedler vector of k-NN graph, sweep cut | **Yes** |

All three satisfy `EvictionPolicy`:

```rust
pub trait EvictionPolicy {
    fn plan_eviction(&mut self, entries: &[MemoryEntry], target_size: usize) -> EvictionPlan;
    fn name(&self) -> &str;
}
```

`EvictionPlan` returns the list of IDs to evict and the conductance of the cut —
a quality signal for logging and proof-gated eviction.

### SpectralEviction algorithm

1. Build a k-NN cosine-similarity graph G on all `MemoryEntry` vectors (k=5 default).
2. Compute the random-walk matrix P = D⁻¹A.
3. Estimate the Fiedler vector v₂ via power iteration (30 steps), deflating the
   leading constant eigenvector at each step.
4. Sort nodes by v₂[i]; evict the `n - target_size` most negative.
5. Return conductance of the partition as a quality metric.

The Cheeger inequality bounds the partition quality:
`φ(cut) ≤ 2√λ₂(L̃)`, where λ₂ is the algebraic connectivity of the normalised
Laplacian L̃.  Low conductance ⇒ the two sides are semantically distinct clusters,
making the evicted side genuinely peripheral.

---

## Consequences

### Positive

- Fills a genuine gap: first Rust-native in-process drift detector for vector indexes.
- FrechetDrift achieves 23-query detection latency with zero false positives at Δ=4
  (D=64, W=500).  CentroidDrift achieves 150-query latency at lower cost.
- SpectralEviction produces conductance 0.100 on a 5-cluster dataset vs no topology
  guarantee from LRU — the post-compaction index has provably clean cluster structure.
- All three detectors are WASM-deployable (deps: `rand`, `rand_distr` only).
- Trait-based design allows future implementations (e.g., HNSW-backed SpectralEviction).

### Negative / risks

- `MmdDrift` is O(S²·D) per check and takes ~19s for 2000 observations at S=167,
  D=64.  **Must not be used in the per-observation hot path.**  Intended for
  batch/async use.
- `SpectralEviction` k-NN build is O(N²·D) — too slow for N > 5K without HNSW.
- Drift thresholds are hand-tuned; a self-calibrating threshold is future work.
- The Fiedler partition's conductance bound does not directly bound recall@k.

---

## Alternatives Considered

### A. TTL-only eviction with no drift detection

Simple to implement; already achievable without this crate.  **Rejected**: TTL
is unaware of topic shift (fresh memories can be irrelevant; old memories can
be structurally essential) and unaware of graph topology.

### B. LLM-based memory summarisation (MemGPT-style)

Summarises old memories into compressed form before eviction.  **Rejected for this
crate**: requires an LLM inference call, adding latency, cost, and an external
dependency.  Complementary rather than competing — a future crate could use
ruvector-drift drift signals to *trigger* LLM summarisation.

### C. Streaming HNSW with soft deletes (tombstones)

`ruvector-delta-index` already handles incremental HNSW updates and repair.
Soft delete is a related but distinct problem: it marks nodes for future removal
without deciding *which* nodes to mark.  **Rejected as primary topic**: already
partially implemented; less research novelty.

### D. Hybrid sparse-dense drift detection (SPLADE-style)

Monitor sparse token distributions alongside dense embeddings.  **Rejected**:
requires sparse vector support not yet in `ruvector-drift`; more complex
without proportionally higher novelty for this crate's scope.

---

## Implementation Plan

| Phase | Task | Status |
|-------|------|--------|
| PoC | CentroidDrift, MmdDrift, FrechetDrift | ✅ Done |
| PoC | RandomEviction, LruEviction, SpectralEviction | ✅ Done |
| PoC | Benchmark binary + acceptance tests | ✅ Done |
| Next | HNSW-backed k-NN in SpectralEviction | Planned |
| Next | Self-calibrating thresholds | Planned |
| Next | Async compaction via tokio | Planned |
| Later | ruFlo hook integration | Planned |
| Later | MCP tool surface | Planned |
| Later | Proof-gated EvictionPlan via ruvector-verified | Planned |

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-drift` on Intel Xeon 2.80 GHz,
rustc 1.94.1, Linux 6.18.5.

**Drift detection (N=4000, D=64, W=500, Δ=4.0):**

| Detector | Detect latency | FP count | Time |
|----------|---------------|----------|------|
| CentroidDrift | 150 | 0 | 84.8 ms |
| MmdDrift | 27 | 0 | 19 245 ms |
| FrechetDrift | 23 | 0 | 191.5 ms |

**Eviction quality (N=1000, D=64, K=5 clusters, 30% eviction, recall@10):**

| Policy | Recall ratio | Conductance | Time |
|--------|-------------|-------------|------|
| RandomEviction | 1.000 | — | <1 ms |
| LruEviction | 1.000 | — | <1 ms |
| SpectralEviction | 1.000 | **0.100** | 178 ms |

Acceptance: all tests PASS.

---

## Failure Modes

| Failure | Trigger | Detection | Mitigation |
|---------|---------|-----------|------------|
| MmdDrift blocks hot path | Per-observation call | Latency spike | Run async; check docs |
| SpectralEviction OOM on N=100K | k-NN graph alloc | OOM error | Cap N or use HNSW k-NN |
| CentroidDrift misses bimodal split | Same mean, different variance | Score stays low | Switch to FrechetDrift |
| Fiedler vector doesn't converge | Near-disconnected graph | Oscillating conductance | Increase k or iters |
| Threshold fires on benign burst | Short-term query spike | High FP rate | Add hysteresis or min-duration rule |

---

## Security Considerations

1. **Drift poisoning**: injecting adversarial vectors to manipulate drift scores can
   trigger premature compaction (memory loss attack) or suppress compaction (memory
   bloat attack).  Mitigation: authenticate insertions via ruvector-verified before
   they enter the drift window.

2. **Eviction manipulation**: an adversary who can observe conductance scores could
   learn which memories are "peripheral" and craft queries that shift them toward
   the boundary.  Mitigation: add ε-DP noise to conductance reports.

3. **Side-channel**: drift scores encode information about the query distribution.
   Log drift scores with access controls; do not expose them via unauthenticated
   MCP endpoints.

---

## Migration Path

- The `DriftDetector` and `EvictionPolicy` traits are stable; implementors can
  swap backends without breaking callers.
- `MemoryEntry` uses `pub` fields — add accessors before stabilising if field
  layout needs to change.
- `SpectralEviction::knn` and `iters` are `pub` for PoC tuning; hide behind
  builder API before 1.0.

---

## Open Questions

1. Should `FrechetDrift` be the default detector, replacing `CentroidDrift`?
   The added variance-detection ability is worth the 2× cost in most cases.

2. What is the recall lower bound as a function of conductance for an HNSW graph?
   This would make SpectralEviction's quality guarantee formal.

3. Should the reference window be reset on a schedule (e.g., weekly), or only
   on operator request?  Automatic reset risks masking genuine long-term drift.

4. Should `ruvector-drift` depend on `ruvector-coherence`'s `SpectralTracker`
   for Fiedler estimation (reuse), or remain standalone (minimal deps)?
