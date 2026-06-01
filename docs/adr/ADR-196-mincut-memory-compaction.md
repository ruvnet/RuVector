# ADR-196: MinCut-Guided Agent Memory Compaction

**Status:** Proposed  
**Date:** 2026-06-01  
**Deciders:** RuVector maintainers  
**Branch:** research/nightly/2026-06-01-mincut-memory-compaction  
**Crate:** crates/ruvector-memory-compaction

---

## Context

RuVector is a Rust-native cognition substrate for AI agents. Agents write
vectors continuously — conversation turns, retrieved document chunks, tool
results, sensor observations. Without active memory management, the vector
index grows without bound, causing:

1. HNSW search latency to increase as the graph grows.
2. Retrieval quality to degrade as stale or redundant entries dominate results.
3. Memory budgets to be exceeded on edge deployments (Cognitum Seed, ESP32).

Existing compaction approaches in production vector databases (Qdrant, Milvus,
LanceDB) are storage-oriented: they merge small segments for I/O efficiency but
do not consider the *semantic structure* of the retained set. This means the
compacted index may be smaller but not semantically optimal.

The prior nightlies addressed search-time algorithms (RaBitQ quantization,
ACORN filtered HNSW, RAIRS IVF). This nightly addresses the pre-index data
management layer: which vectors to keep before indexing.

---

## Decision

Introduce `crates/ruvector-memory-compaction` as a standalone Rust crate
providing a `MemoryCompactor` trait with three implementations:

1. **GreedyAgeCompactor**: FIFO baseline, O(n log n). Evicts oldest entries.

2. **DecayScoreCompactor**: Exponential temporal decay combined with greedy
   diversity selection. O(n²) greedy. Maintains semantic spread of retained set.

3. **MinCutCompactor**: Builds a k-NN cosine-similarity graph, computes
   isolation scores (1 − mean edge weight), and evicts the most-isolated nodes.
   Preserves dense semantic cluster cores. O(n² × D) build.

The `MinCutCompactor` is the primary contribution. Its isolation scoring
approximates the minimum-cut criterion: nodes with high isolation scores lie
near graph cut boundaries and contribute least to the global connectivity of
the semantic memory graph.

---

## Consequences

### Positive

- **Quality improvement on clustered agent memory**: +0.11–0.12 centroid
  cosine similarity vs. GreedyAge on 8-cluster Gaussian data (N=1000–3000,
  D=64–128, 50% retention). Measured. See benchmark results.

- **No external dependencies**: self-contained crate using only `rand`,
  `rand_distr`, `serde`, `serde_json`, `thiserror`, `rayon` — all in workspace.

- **Trait-based API**: callers can inject any `impl MemoryCompactor`, enabling
  future variants (HNSW-accelerated, streaming, spectral) without API breaks.

- **Composable with existing ecosystem**:
  - `ruvector-verified`: `CompactionResult.evicted_ids` → witness log
  - `mcp-gate`: can expose as `memory/compact` MCP tool
  - `ruFlo`: compaction trigger as a workflow action
  - `ruvector-coherence`: `SpectralCoherenceScore` as quality gate

- **All 13 unit tests pass. All benchmark acceptance checks pass.**

### Negative / Constraints

- **MinCutCompactor is O(n² × D)**: at n=5000, D=128 takes 3.6 s on x86_64.
  Not production-ready as written. Needs HNSW-accelerated graph build.

- **Centroid cosine similarity as quality metric** is a proxy. It does not
  directly measure held-out query recall. On isotropic zero-mean data it is
  numerically unstable.

- **No streaming/incremental support**: full batch compaction only. Each run
  rebuilds the similarity graph from scratch.

---

## Alternatives Considered

### A. Pure recency decay (no graph structure)
Simpler but ignores topology. The DecayScoreCompactor provides this with
the addition of greedy diversity. Rejected as the primary approach because
it does not exploit cluster structure. Kept as a variant.

### B. K-means clustering + centroid retention
Cluster the memory into k groups, retain the centroid of each cluster and
its nearest actual entry. O(n × k × D × iterations). Advantage: explicit
cluster awareness. Disadvantage: requires choosing k, and k-means can be
unstable on high-D data. May be worth implementing as variant D.

### C. PageRank over similarity graph
Use PageRank centrality instead of isolation score. High-PageRank nodes are
the most "referenced" by their neighbours — the inverse of isolation.
Advantage: well-studied. Disadvantage: PageRank is sensitive to dangling
nodes; requires normalisation. Kept as a research direction for ADR-197.

### D. Reservoir sampling with recency weighting
Keep a fixed-size reservoir, weighting by recency. O(n), very fast.
Advantage: suitable for streaming inserts. Disadvantage: no topology awareness.
Useful for edge deployments where O(n²) is unacceptable.

---

## Implementation Plan

### Phase 1 (this PR — complete)
- [x] Implement `GreedyAgeCompactor`, `DecayScoreCompactor`, `MinCutCompactor`
- [x] 13 unit tests passing
- [x] Benchmark binary with acceptance checks
- [x] Feature-gated rayon dependency (WASM-safe)

### Phase 2 (follow-on)
- [ ] Replace O(n²) graph build with HNSW approximate k-NN from `ruvector-core`
- [ ] Add `SpectralCoherenceScore` quality gate from `ruvector-coherence`
- [ ] Add `CompactionResult` → witness log via `ruvector-verified`

### Phase 3 (production hardening)
- [ ] MCP tool surface via `mcp-gate` (`memory/compact`)
- [ ] ruFlo workflow trigger (`post-write` hook on store size)
- [ ] Streaming/incremental isolation score updates
- [ ] Benchmark against held-out query recall@10

---

## Benchmark Evidence

All numbers measured on this machine. `cargo run --release -p ruvector-memory-compaction`.

**Clustered data (8 Gaussian clusters, σ=0.5):**

| N     | Dim | Variant         | Quality | Duration (µs) |
|-------|-----|-----------------|---------|---------------|
| 1,000 | 64  | GreedyAge       | 0.7118  | 32            |
| 1,000 | 64  | DecayScore      | 0.7178  | 22,924        |
| 1,000 | 64  | MinCutGraph     | **0.8331** | 82,986   |
| 3,000 | 128 | GreedyAge       | 0.7263  | 103           |
| 3,000 | 128 | DecayScore      | 0.7281  | 377,013       |
| 3,000 | 128 | MinCutGraph     | **0.8328** | 1,269,918 |

**MinCutGraph leads GreedyAge by +0.1065–0.1213 on clustered data.**

**Isotropic data (N(0,1)):**

| N     | Dim | Variant     | Quality | Duration (µs) |
|-------|-----|-------------|---------|---------------|
| 5,000 | 128 | GreedyAge   | 0.6950  | 117           |
| 5,000 | 128 | DecayScore  | 0.7305  | 1,102,642     |
| 5,000 | 128 | MinCutGraph | 0.7392  | 3,631,342     |

On isotropic data (no cluster structure) all variants perform similarly.
This is the expected and correct behaviour — topology provides no signal
when all entries are equally isolated.

---

## Failure Modes

1. **Isotropic memory**: quality improvement disappears. Use `DecayScoreCompactor`.
2. **Adversarial flood**: recent low-quality entries displace important older ones.
   Mitigation: access-count bonus already implemented.
3. **Zero-vector centroid**: cosine similarity is unstable. Use spectral quality gate.
4. **n > 10,000**: O(n²) build is too slow. Must use HNSW approximate k-NN.
5. **Mixed-modal embeddings**: cosine similarity is poorly calibrated across modalities.

---

## Security Considerations

- **Eviction audit trail**: `CompactionResult.evicted_ids` should be written to a
  witness log via `ruvector-verified` in regulated deployments.
- **Input validation**: `MemoryEntry.id` values should be validated as unique before
  insertion. Duplicate IDs in compaction results would corrupt the store.
- **Access-controlled memories**: if entries carry ACL tags, compaction must
  not evict entries that would leave a user without accessible memories.
- **No secrets in vectors**: this crate does not inspect vector contents, but
  callers must ensure sensitive data is not stored as raw embeddings without
  access controls.

---

## Migration Path

This is a new, standalone crate. There is no existing code to migrate.

Adoption path for existing `ruvector-core` users:
1. Add `ruvector-memory-compaction` dependency.
2. Wrap existing `Vec<(Vec<f32>, metadata)>` store with `MemoryStore`.
3. Run `MinCutCompactor::compact()` when `store.len() > budget`.
4. Apply result to reduce entries before re-inserting into `ruvector-core` HNSW.

---

## Open Questions

1. Should `MinCutCompactor` use `ruvector-mincut`'s dynamic graph directly,
   or maintain the current standalone graph implementation?
   (Tradeoff: richer algorithms vs. added compile-time complexity.)

2. What is the right quality metric for production use — centroid sim, recall@10,
   or `SpectralCoherenceScore`?

3. Should compaction be integrated into the write path of `ruvector-core`
   (automatic background compaction) or kept as an explicit caller operation?

4. What is the memory budget policy for the Cognitum Seed edge appliance?
   (This determines whether `GreedyAgeCompactor` or a future micro-variant is used.)
