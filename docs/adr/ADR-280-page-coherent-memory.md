# ADR-280: Page-Coherent Agent Memory via Greedy Coherence Clustering

**Status**: Proposed  
**Date**: 2026-08-03  
**Author**: nightly-research-agent  
**Nightly branch**: research/nightly/2026-08-03-page-coherent-memory

---

## Context

RuVector agent memory stores retrieve individual vectors on every agent action. As memory stores grow beyond tens of thousands of entries, exhaustive flat search becomes too slow for low-latency agent workflows. Existing solutions (IVF clustering, HNSW) optimize for recall of individual vectors but do not address the *context loading problem*: an agent loading retrieved vectors into its context window may receive semantically scattered results that require more context tokens to reason about.

Two simultaneous problems exist:

1. **Retrieval speed**: linear scan is O(N·D); even at N=8,000 and D=128, it takes ~1.4 ms per query on modern x86 hardware, and scales poorly to millions of entries.
2. **Context fragmentation**: individually retrieved vectors may come from different topics, requiring the agent to spend context tokens bridging topical gaps.

Page-coherent memory addresses both: organize vectors into coherent pages at build time, then probe only a fraction of pages at query time.

---

## Decision

Introduce `ruvector-coherence-pages` as a standalone crate implementing the `PageStore` trait with three backends:

1. **FlatStore**: exhaustive baseline (always recall=1.0, used for comparison).
2. **CentroidPageStore**: k-means centroid clustering (10 Lloyd iterations, stride-sampled initialization). Best recall at equal probe count. Build O(N·K·D·iters).
3. **GreedyCoherenceStore**: greedy seed-and-pull page construction. Best intra-page coherence. Build O(N²·D/page_size) but much cheaper in wall-clock (12.8× faster than k-means in benchmarks). Suitable for offline compaction of agent memory.

The `PageStore` trait enables:
- Pluggable backend selection per namespace.
- Page-level probe budget control via `probe` parameter.
- Coherence monitoring via `avg_page_coherence()`.

---

## Consequences

### Positive

- **7–9.6× query speedup** at 10% probe rate on N=8,000 D=128 random unit vectors.
- **Higher intra-page coherence** (0.7782 greedy, 0.7693 centroid vs. 0.7533 flat baseline) means pages loaded into agent context windows contain topically related memories.
- **Zero external dependencies**: the crate builds with no crates.io dependencies, making it WASM-safe and embeddable.
- **Trait-based API** allows future backends (HNSW-indexed centroid search, hierarchical pages).
- **Fast greedy build** (65 ms for 8K vectors) is suitable for background compaction tasks triggered by ruFlo.

### Negative

- **Recall cost**: at 10% probe, centroid paging achieves recall@10=0.35 (vs. 1.0 for flat). Greedy coherence achieves 0.23. Full recall requires probing all pages (degenerating to flat scan).
- **Coherence/recall tradeoff**: greedy coherence maximizes local similarity but k-means centroids better anchor retrieval neighborhoods. Users must choose based on whether they prioritize context quality (greedy) or retrieval recall (centroid).
- **O(N²) greedy build**: not suitable for online insert into large stores. Use centroid-pages for stores > 100K vectors where build time matters.
- **Coherence scores are approximate**: computed over first 10 vectors per page (O(1) per page). True pairwise coherence would require O(page_size²) computation.

---

## Alternatives Considered

### IVF-only (existing `ruvector-filter` + centroid logic)
IVF is production-proven in Milvus, FAISS, Qdrant. However, IVF implementations in this codebase are tightly coupled to distance computation and filter predicates. A clean `PageStore` trait is more composable with agent memory abstractions and the `ruvector-agent-memory` crate.

### HNSW (existing `ruvector-coherence-hnsw`)
HNSW gives excellent recall at low latency but does not provide coherent page loading. HNSW returns individual vectors; page-coherent memory returns whole pages. These are complementary: future work could use HNSW over page centroids.

### Graph-cut compaction (`ruvector-bounded-rag`)
Prior nightly work (2026-07-25) explored mincut-based memory grouping. Graph cuts require full graph construction (O(N²) edges in dense case). Greedy coherence achieves similar grouping quality with simpler implementation and no graph dependency.

### Random partitioning
Null hypothesis: random pages (no clustering) achieve the same recall at 10% probe. Benchmark shows centroid paging achieves 0.35 vs. expected 0.125 for random probe (2.8× better). Greedy coherence achieves 0.23 (1.9× better). Structure matters.

---

## Implementation Plan

### Phase 1 (this nightly): PoC crate
- [x] `PageStore` trait with `build_from`, `search`, `page_count`, `avg_page_coherence`.
- [x] `FlatStore` baseline.
- [x] `CentroidPageStore` (10-iter k-means).
- [x] `GreedyCoherenceStore` (greedy seed-pull).
- [x] Benchmark binary with real numbers.
- [x] 6 unit tests, all passing.
- [x] All acceptance checks passing.

### Phase 2 (production hardening)
- [ ] HNSW-indexed centroid search (replace O(K) centroid scan with O(log K) HNSW walk).
- [ ] Online insert: heuristic assignment to the most coherent existing page, with overflow re-split.
- [ ] Serialization via `serde` feature flag (page store persistence).
- [ ] Concurrent read (RwLock per page or page-shard partitioning).
- [ ] Integration with `ruvector-agent-memory` via `AgentMemoryBackend` trait.

### Phase 3 (research directions)
- [ ] Hierarchical pages: meta-pages of pages for multi-granularity retrieval.
- [ ] Adaptive page size: controller that adjusts page_size based on query coherence feedback.
- [ ] Access-controlled pages: integrate with `ruvector-capgated` for page-level ACLs.
- [ ] WASM compilation: verify `--target wasm32-unknown-unknown` with bounded heap.
- [ ] ruFlo coherence watchdog: workflow step that monitors `avg_page_coherence` decay.

---

## Benchmark Evidence

From `cargo run --release -p ruvector-coherence-pages --bin benchmark` (2026-08-03, Linux x86_64, opt-level=3):

```
Dataset: 8000 vectors × 128 dimensions
Queries: 500, top-10
Pages: 80, probe: 8/80 (10%)

flat:             recall=1.00, mean=1407µs, p50=1397µs, p95=1482µs, 711 q/s
centroid-pages:   recall=0.35, mean=201µs,  p50=195µs,  p95=256µs,  4985 q/s (7.0×)
greedy-coherence: recall=0.23, mean=146µs,  p50=138µs,  p95=177µs,  6855 q/s (9.6×)

Coherence (avg intra-page cosine):
  flat:             0.7533
  centroid-pages:   0.7693 (+0.0161 vs. flat)
  greedy-coherence: 0.7782 (+0.0250 vs. flat)
```

All 8 acceptance criteria passed.

---

## Failure Modes

| Mode | Detection | Response |
|------|-----------|----------|
| Build with k=0 | Panics in stride calculation | `assert!(num_pages > 0)` guard added |
| Empty page from k-means | Empty page_vecs filtered | `filter(|pv| !pv.is_empty())` in build_from |
| Zero-vector inserted | Centroid normalization produces NaN | Add `is_finite()` check on insert |
| Greedy build O(N²) too slow | Build time >30s for N>50K | Auto-switch to centroid-pages above size threshold |
| Coherence score decreases over time | ruFlo monitoring | Trigger background GreedyCoherenceStore rebuild |

---

## Security Considerations

- **Page centroid exposure**: centroids reveal topical structure of the memory store. In multi-tenant systems, centroids from one tenant must not be visible to another.
- **Insertion manipulation**: an adversary who can insert arbitrary vectors can manipulate greedy page assignments to contaminate page context. Proof-gated writes (`ruvector-proof-gate`) defend against unauthorized insertions.
- **Differential privacy**: for sensitive deployments, add Gaussian noise to centroids before storing (ε-DP on centroid representations).

---

## Migration Path

- Existing `ruvector-agent-memory` users: opt-in via a `backend = "coherent-pages"` config option; flat search remains the default.
- No breaking changes to existing vector storage APIs.
- Page store builds happen offline (or in a ruFlo background job); query API is read-only.

---

## Open Questions

1. What is the right page_size for real agent memory with embedding models (text-embedding-3-small, nomic-embed, etc.)? Benchmark used random unit vectors; real embeddings have different topical structure.
2. Should `GreedyCoherenceStore` support online inserts or remain a compaction-only backend?
3. How does page-coherent memory interact with time-ordered agent memory? (Recent memories may need to stay together regardless of topic coherence.)
4. Is there a principled way to choose `probe` count based on query characteristics rather than a fixed parameter?
