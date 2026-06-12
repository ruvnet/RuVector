---
adr: 199
title: "Multi-Subspace HNSW with Coherence-Weighted Fusion"
status: proposed
date: 2026-06-12
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-196, ADR-197, ADR-198]
tags: [ann, hnsw, subspace, coherence, vector-search, agent-memory, ruFlo, mcp, edge-ai]
---

# ADR-199 — Multi-Subspace HNSW with Coherence-Weighted Fusion

## Status

**Proposed.** Proof of concept implemented and benchmarked in
`crates/ruvector-subspace-hnsw`. Further work needed before production merge.

---

## Context

RuVector's HNSW implementation (`ruvector-core`) builds a single navigable small-world
graph across all D embedding dimensions. This is optimal when all dimensions carry
equal signal, but practical embeddings — agent memories, multi-facet documents,
multi-modal representations — have unequal per-dimension information content.

Three converging pressures drive the need for subspace-aware retrieval:

1. **Agent memory diversity**: a single agent memory embedding encodes episodic,
   semantic, and procedural facets in different regions of the embedding space.
   A query about a past action and a query about a learned fact should use
   different subspace weights.

2. **Embedding dimension growth**: modern LLMs emit 768–4096-dim embeddings;
   near-future models may use 8K–16K dims. Monolithic HNSW at these dimensions
   faces distance concentration — all pairs become equidistant. Subspace
   decomposition delays this effect.

3. **Prior art gap**: subspace retrieval systems (Subspace Collision [^1],
   TaCo [^2]) use clustering-based indexes and static collision-count fusion.
   No published work uses HNSW per subspace with runtime variance-based
   coherence weighting.

---

## Decision

We introduce `SubspaceHnsw` and `CoherenceHnsw` as optional components
in the RuVector retrieval stack. The design:

**Build:**
- Partition the D embedding dimensions into K equal subspaces of D/K dimensions.
- Build one independent HNSW graph per subspace.
- Store full-dimensional vectors for final re-ranking.

**Query:**
- Project query into K subspace vectors.
- Search each subspace HNSW independently with beam width ef.
- Compute per-subspace coherence weight: `w_s = 1 / (1 + CV_s)` where
  `CV_s = std(distances) / mean(distances)` of the subspace's top-ef results.
- Compute weighted distance score for each candidate across all subspaces.
- Return top-k by weighted score.

**API surface (production candidate):**
```rust
pub struct SubspaceConfig {
    pub num_subspaces: usize,   // K — number of equal-width dimension partitions
    pub m: usize,               // HNSW M parameter per subspace
    pub ef_construction: usize,
}

pub struct CoherenceHnsw {
    pub fn build(vectors: &[Vec<f32>], config: &SubspaceConfig) -> Self;
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)>;
    pub fn coherence_scores(&self, query: &[f32], ef: usize) -> Vec<f32>;
    pub fn memory_bytes(&self) -> usize;
}
```

**What belongs behind a feature flag:**
- `feature = "subspace-hnsw"` — the entire subsystem, as it increases binary size
  and memory usage by ~3× for the same N

**What should remain in the PoC only:**
- The minimal 2-layer NSW used in this PoC; production requires full HNSW from
  `ruvector-core`
- Naive equal-width subspace partitioning; production should use entropy-balanced
  assignment

---

## Consequences

**Benefits:**

- Query-adaptive recall: coherence weighting naturally up-weights informative subspaces
  without any training or per-dataset tuning
- Measured +21pp recall improvement at N=2K, D=64 vs. single-space HNSW
- Coherence scores as observable signals for ruFlo memory management workflows
- Partial indexing: add a new semantic subspace without rebuilding all K graphs
- Composable with RaBitQ quantization for subgraph memory reduction

**Costs:**

- ~3× memory overhead (K subgraphs + full vectors for re-ranking)
- ~4× build time (K independent HNSW builds)
- ~5× query latency at N=10K (874 µs vs. 184 µs baseline)
- At N=10K, D=128 the subspace variants underperform the baseline (0.443 vs. 0.543
  recall) — subspace decomposition hurts when noise dimensions dominate

**Scale characteristics (measured):**

| Dataset | Baseline recall | Coherence recall | Coherence vs. baseline |
|---------|----------------|-----------------|----------------------|
| N=500, D=32 | 1.000 | 0.980 | –2pp (small overhead) |
| N=2K, D=64 | 0.630 | 0.840 | **+21pp** (clear benefit) |
| N=10K, D=128 | 0.543 | 0.443 | –10pp (use baseline at this scale) |

---

## Alternatives Considered

**1. Single HNSW with dimensionality reduction (PCA pre-processing)**
Reduces D before indexing. Simpler, but loses fine-grained structure and requires
a pre-processing step. Cannot do per-query coherence weighting.

**2. Subspace Collision (arXiv:2411.14754) with clustering indexes**
SOTA at SIGMOD 2025. Better QPS via clustering, but: no HNSW (lower quality
at same search budget), no runtime variance-based coherence, static fusion weights.
Could be a complementary approach in a `SubspaceCollision` crate.

**3. IVF-based subspace quantization (FAISS IVF-PQ style)**
Lower memory, but: no adaptive coherence, requires training centroids, less
flexible for streaming inserts.

**4. Anisotropic quantization (ScaNN/Google)**
Query-direction-sensitive quantization. Requires training; not zero-shot.
Does not preserve graph structure.

---

## Implementation Plan

### Phase 1 (now — this PoC)
- [x] Minimal NSW with subspace projection and coherence fusion
- [x] Three variants: Baseline, SubspaceUnion, CoherenceHnsw
- [x] Real benchmark with recall@10, latency, memory
- [x] ADR and research documentation

### Phase 2 (production hardening)
- [ ] Replace minimal NSW with `ruvector-core` HNSW via trait abstraction
- [ ] Entropy-balanced dimension assignment (sort dims by variance pre-build)
- [ ] Parallel subspace construction via Rayon
- [ ] RaBitQ quantization for subgraph memory reduction
- [ ] Coherence score threaded into `ruvector-server` query response

### Phase 3 (research direction)
- [ ] Learned subspace boundaries via mincut over embedding graph
- [ ] Coherence → ruFlo observable signal integration
- [ ] MCP `ruvector_search_subspace` tool surface
- [ ] Temporal coherence decay for agent memory tiers

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-subspace-hnsw --bin benchmark`.

```
 OS:   linux / Arch: x86_64
 N=10,000, D=128, clusters=20, signal_dims=96, queries=200
 M=16, ef_construction=100, ef_search=80, K_subspaces=4

 Variant              Build(ms)  Recall@10  Mean(µs)  p50(µs)  p95(µs)   QPS  Mem(MB)
 Baseline-HNSW          1,464      0.543      184       179      237     5,422   6.59
 SubspaceUnion-HNSW     5,890      0.443      874       868    1,001     1,144  16.53
 CoherenceHnsw          5,817      0.443      880       872    1,031     1,136  16.53
```

Unit test (N=2K, D=64):
```
 [small]  baseline=1.000, coherence=0.980
 [medium] baseline=0.630, coherence=0.840  (+21pp)
```

Acceptance: baseline recall@10 ≥ 0.50 ✓; coherence delta vs. union ≥ –0.05 ✓

---

## Failure Modes

| Mode | Trigger | Impact | Detection |
|------|---------|--------|-----------|
| Subspace underperforms baseline | N > 5K with high noise/signal ratio | –10pp recall | Measure recall before serving traffic |
| Memory blowup | K=8 on large N | 8× memory overhead | Pre-check `memory_bytes()` |
| Build OOM | Large N + large K | Process kill | Add N×K pre-flight check |
| All coherence weights equal | Homogeneous data | No improvement; graceful degradation | Log coherence variance; warn |
| Degenerate subspace | Correlated dims in one subspace | Poor per-subspace ANN quality | Validate subspace variance > threshold |

---

## Security Considerations

- No external service; all computation local — safe for air-gapped edge deployments
- Subspace scores are internal to the index; not exposed to callers by default
- No secret vectors: the subspace graphs do not expose individual stored vectors
- Full-vector storage (for re-ranking) carries same sensitivity as primary index

---

## Migration Path

- **From `ruvector-core` HNSW**: `CoherenceHnsw::build()` accepts `Vec<Vec<f32>>`
  — same input as the existing index builders. No format change needed.
- **From RaBitQ**: keep existing quantized index; add CoherenceHnsw as a parallel
  candidate generation stage, with RaBitQ for fast re-ranking.
- **Rollout**: feature-flagged behind `subspace-hnsw`; enable per-namespace in
  `ruvector-server` config.

---

## Open Questions

1. What is the optimal K for a given embedding dimensionality D?
2. Should subspace boundaries be fixed (equal-width) or learned (via mincut/PCA)?
3. Does entropy-balanced assignment (TaCo) close the gap at N=10K?
4. Can coherence scores serve as a reliable confidence signal for ruFlo workflows?
5. What is the minimum N where CoherenceHnsw's recall improvement justifies
   the 3× memory cost?

---

## Footnotes

[^1]: Wei, Zewei, et al. "Subspace Collision." SIGMOD 2025. arXiv:2411.14754.
[^2]: "TaCo: Data-adaptive and Query-aware Subspace Collision." arXiv:2603.24919, 2026.
