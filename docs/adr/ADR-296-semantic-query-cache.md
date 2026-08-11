# ADR-296: Semantic Query Cache for ANN Search

**Status:** Proposed
**Date:** 2026-08-05
**Author:** Nightly Research Agent

---

## Context

RuVector serves AI agents, ruFlo workflow loops, and MCP memory tools that produce workloads with high query locality: the same or semantically near-identical queries repeat within seconds or minutes. Every repeat query currently pays the full corpus ANN scan cost (7–30 ms for 50 K–1 M in-memory vectors).

Published work on semantic caching has focused on caching LLM responses (GPTCache, vCache) rather than caching ANN result sets. QVCache (arXiv 2602.02057, Feb 2026) is the first paper to cache ANN result sets at the retrieval-middleware layer; it claims 40–1000× speedup for disk-based systems. No Rust-native implementation exists.

RuVector needs a zero-dependency, edge-deployable, WASM-compatible semantic cache that:
1. Caches (query_vector, result_set) pairs indexed by cosine similarity.
2. Short-circuits corpus scans for near-duplicate queries.
3. Invalidates on corpus mutation.
4. Operates with configurable recall/hit-rate tradeoffs via the similarity threshold.

---

## Decision

Introduce `ruvector-semantic-cache` as a standalone, zero-dependency crate implementing:

- **`SemanticCacheLayer` trait** — the stable API for all cache variants.
- **`NoCache`** — pass-through baseline.
- **`FlatSemanticCache`** — flat cosine scan over ≤ 1 000 entries, configurable threshold, LRU eviction.
- **`coarse(n)`** constructor — threshold = 0.90, maximises hit rate.
- **`fine(n)`** constructor — threshold = 0.97, maximises result fidelity.
- **`invalidate_all()`** — mandatory call on any corpus mutation.

The crate is registered in the workspace and integrated as an optional read-through layer for `ruvector-agent-memory`.

---

## Consequences

### Positive

- 3.5× end-to-end latency reduction at 72.8% hit rate and 94.7% hit recall (measured).
- 86× speedup per cache hit (92 µs vs 7 900 µs corpus scan, measured).
- < 300 KB memory overhead for 500-entry cache at 128 dims.
- Zero external dependencies → WASM-safe, edge-safe.
- Clear invalidation contract prevents stale results.

### Negative

- `invalidate_all()` is coarse; high-write corpora will see poor effective hit rates until selective invalidation is implemented.
- Flat-scan cost grows linearly with cache size; degrades at > 5 000 entries without an HNSW cache index.
- Cache recall is probabilistic; adversarial inputs could construct queries that collide with cached entries but return wrong results (see arXiv:2601.23088 on semantic cache key collision attacks).
- Cold start after invalidation produces a full-miss window; warm-up from query logs is not yet automated.

---

## Alternatives Considered

### 1. Exact-match LRU cache (hash by quantised vector fingerprint)

A simple LRU map keyed by a coarse quantisation of the query vector (e.g. per-dimension sign bits as a u128 hash). Hit only when two queries produce identical fingerprints.

**Rejected because:** Hit rate is near-zero for real paraphrase workloads. The semantic relationship between near-duplicate queries is lost entirely.

### 2. HNSW-indexed cache (O(log n) lookup)

Build a small HNSW over cached query vectors; for new queries, find the HNSW approximate nearest cached query.

**Deferred:** For cache sizes ≤ 1 000, flat scan is cheaper (92 µs) than HNSW construction overhead. At > 10 K entries the HNSW becomes necessary. Tracked as `ruvector-hnsw-cache` future crate.

### 3. Per-region adaptive threshold (QVCache-style)

Maintain a per-region threshold that adapts online based on hit recall feedback.

**Deferred:** Adds significant complexity (clustering, online learning, EMA) that is not justified for the PoC. The coarse/fine binary provides a useful first approximation. Tracked as future enhancement.

### 4. Integrate cache directly into corpus search

Intercept queries at the corpus scan level; if a nearly-identical recent query exists, skip the HNSW graph walk.

**Rejected for PoC:** Requires modifying the corpus search internals. The middleware approach (cache as a separate layer) is more composable and works with any corpus backend.

---

## Implementation Plan

### Phase 1 (this ADR, nightly 2026-08-05)
- [x] `ruvector-semantic-cache` crate: `SemanticCacheLayer` trait, `FlatSemanticCache`, `NoCache`
- [x] `overlap_recall()` helper for recall measurement
- [x] Three-variant benchmark binary with acceptance tests
- [x] 12 unit tests passing
- [x] Workspace registration

### Phase 2 (production hardening)
- [ ] HNSW-indexed cache for > 1 000 entries (`ruvector-hnsw-cache`)
- [ ] Selective invalidation: track corpus ID sets per cache entry
- [ ] Per-user namespace partitioning (multi-tenant)
- [ ] Proof-gated cache writes (ruvector-proof-gate integration)
- [ ] ruFlo node for cache warm-up and invalidation scheduling

### Phase 3 (10–20 year research)
- [ ] Adaptive per-region thresholds
- [ ] Agent cognitive working memory (cache as first-class short-term memory)
- [ ] Distributed multi-node cache with CRDT reconciliation
- [ ] Formal correctness guarantees (vCache-style error bounds)

---

## Benchmark Evidence

All numbers are from `cargo run --release -p ruvector-semantic-cache --bin benchmark` on 2026-08-05, x86\_64 Linux.

Dataset: 50 000 × 128-dim f32 corpus, 2 400 benchmark queries (35% exact, 40% near-dup σ=0.02, 25% diverse σ=0.10), k=10, cache capacity=500.

| Variant | Hit Rate | Mean Lat µs | p50 µs | Hit Recall@10 |
|---------|----------|-------------|--------|---------------|
| NoCache | 0.0% | 7 925.8 | 7 885.4 | — |
| CacheCoarse (t=0.90) | 72.8% | 2 263.0 | 97.8 | 0.947 |
| CacheFine (t=0.97) | 52.3% | 3 921.8 | 157.6 | 0.958 |

Acceptance: all 5 tests PASS.

---

## Failure Modes

| Failure | Trigger | Mitigation |
|---------|---------|------------|
| Stale results | Corpus mutated; `invalidate_all()` not called | Require invalidation on every corpus write; version tracking |
| Cache poisoning | Adversary crafts query matching cached entries | Proof-gate inserts; validate source agent identity |
| Multi-tenant leakage | Users A and B share cache | Per-user namespace partitioning |
| Perpetual cold cache | Write rate > invalidation budget | Selective invalidation or TTL-bounded cache entries |
| Low recall at t=0.90 | Query distribution includes many border-region pairs | Raise threshold or use adaptive per-region threshold |

---

## Security Considerations

1. **Key collision attack (arXiv:2601.23088):** Adversarially crafted queries can achieve high cosine similarity with cached entries while semantically differing. This could cause an agent to receive wrong retrieval results with high confidence. Mitigation: proof-gate cache inserts; flag as security-relevant.
2. **Side-channel:** Cache hit/miss pattern reveals recent query topics. Mask cache timing in public APIs.
3. **Data retention:** Cached query vectors represent user intent; treat as PII in multi-tenant deployments. Apply differential privacy perturbation to stored query vectors.

---

## Migration Path

- `ruvector-agent-memory` can adopt `SemanticCacheLayer` as an optional read-through with no breaking API changes: add a `cache: Option<Box<dyn SemanticCacheLayer>>` field and check it before corpus search.
- Existing callers not using the cache are unaffected (`NoCache` is the zero-cost default).
- The `invalidate_all()` contract must be explicitly documented at every corpus mutation site.

---

## Open Questions

1. What is the right threshold for production agent workloads? Is 0.90 too aggressive?
2. Should cache eviction be strictly LRU or also consider coherence score (prefer to evict low-coherence queries)?
3. Should cached result sets be compressed (delta-encoded IDs, quantised distances)?
4. Can `ruvector-mincut` be used to cluster the cache query space and assign per-cluster thresholds?
5. What is the correctness guarantee needed for safety-critical agent systems?
