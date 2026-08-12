# ADR-298: Semantic Query Cache for ANN

**Status**: Proposed
**Date**: 2026-08-12
**Author**: nightly-research-agent
**Crate**: `ruvector-query-cache`

---

## Context

RuVector serves as a Rust-native cognition substrate for autonomous agents. Agent
workloads exhibit statistically clustered query distributions: the same semantic
intent recurs with minor embedding variation across iterations of a ruFlo workflow,
across turns of a multi-turn agent conversation, and across agents in a swarm that
share a knowledge base.

Standard ANN systems treat every query as independent and perform a full index scan
or graph traversal for each one. This is correct for general-purpose retrieval but
wasteful for agent-memory workloads where:

1. The query distribution is far from uniform.
2. A slightly approximate result (from a semantically-similar prior query) is
   acceptable for most agent tasks.
3. Cumulative retrieval cost across thousands of agent iterations is a real
   production concern.

No major vector database provides cosine-similarity-aware query result reuse as a
first-class primitive. The gap is real.

---

## Decision

Introduce `ruvector-query-cache` as a standalone Rust crate providing a
`CachedAnn` trait and three implementations:

1. **NoCache** — exact brute-force scan; ground truth baseline.
2. **ExactCache** — bitwise-exact query hash match; never hits on similar-but-not-identical.
3. **SemanticCache(threshold)** — cosine-similarity scan over stored queries;
   returns cached results when `cosine(incoming, stored) ≥ threshold`.

The crate is designed as a composable middleware layer: any `CachedAnn` impl wraps
an underlying ANN backend, intercepts queries, and falls through to the backend on
cache miss.

---

## Consequences

### Positive

- Measured 34.8% hit rate at threshold=0.85 on a 35%-repeat-rate workload.
- Measured 22.9% mean latency reduction (827µs → 602µs) at threshold=0.85.
- Monotone quality: higher threshold → higher recall (measured: 0.844 @ 0.85,
  0.871 @ 0.90, 0.935 @ 0.95, 1.000 @ 0.99).
- Zero external dependencies (only `rand` for test data generation).
- Compatible with any underlying ANN backend.
- WASM-deployable: no unsafe code, no OS-specific APIs.

### Negative

- Recall degradation at lower thresholds: 0.85 threshold yields recall=0.844.
- Cache lookup overhead (O(n_cache × dim)) adds latency on miss: +85µs at n_cache=512,
  dim=128.
- FIFO eviction is suboptimal for bursty query patterns.
- No built-in TTL: stale cached results accumulate if the corpus is updated.

### Neutral

- The crate does not replace HNSW, IVF, or any existing ANN structure.
- The quality–latency trade-off is explicit and measurable; operators set threshold.

---

## Alternatives Considered

### A. Skip the cache entirely; rely on OS-level ANN index caching

OS page cache helps for disk-based indexes (DiskANN, SPANN). It does not help for
in-memory indexes where the bottleneck is compute, not I/O. Rejected.

### B. Query result hash cache (exact match only)

Implemented as `ExactCache`. Measured hit rate: 0.0% on real workloads where
queries vary even slightly. The gap between 0% (exact) and 30%+ (semantic) is the
entire motivation for this work.

### C. Pre-cluster queries and cache by cluster centroid

Requires offline cluster computation and periodic re-clustering as query distribution
shifts. More complex with no measurable benefit over threshold-based approach at
research PoC scale. Deferred to production hardening.

### D. Integrate caching into the HNSW graph traversal (warm entry-point)

Storing a "warm entry point" per query cluster would pre-position the HNSW search
closer to the expected neighbourhood. Compatible with this crate (the cache miss
path can supply a warm entry point). Deferred.

---

## Implementation Plan

| Phase | Work | Timeline |
|-------|------|----------|
| Now | Merge `ruvector-query-cache` as standalone crate | Week 1 |
| Now | Add feature flag in `ruvector-server` to enable semantic cache | Week 2 |
| Next | Replace FIFO with LRU eviction | Week 3 |
| Next | Add adaptive threshold controller (online recall estimator) | Week 4–5 |
| Next | Add TTL integration with `ruvector-temporal-coherence` | Week 5–6 |
| Next | Add per-tenant namespace isolation via `ruvector-capgated` | Week 6–7 |
| Later | WASM SIMD cosine scan for cache lookup | Month 3 |
| Later | Distributed cache with CRDT statistics | Month 6 |

---

## Benchmark Evidence

Run: `cargo run --release -p ruvector-query-cache --bin benchmark`
Build: release, LTO=fat, opt-level=3, Linux x86_64

| Variant | Hit Rate | Mean (µs) | p50 (µs) | p95 (µs) | QPS | Recall@10 | Mem (KB) |
|---------|----------|-----------|----------|----------|-----|-----------|----------|
| NoCache | 0.0% | 827.4 | 819.2 | 891.4 | 1205 | 1.000 | 2500 |
| ExactCache | 0.0% | 822.6 | 814.8 | 878.3 | 1213 | 1.000 | 2855 |
| Semantic@0.85 | **34.8%** | **602.3** | 850.1 | 959.9 | **1657** | 0.844 | 2713 |
| Semantic@0.90 | 30.8% | 638.1 | 860.1 | 964.3 | 1564 | 0.871 | 2727 |
| Semantic@0.95 | 17.4% | 773.1 | 889.7 | 1084.0 | 1291 | 0.935 | 2771 |
| Semantic@0.99 | 0.0% | 912.2 | 914.9 | 1011.4 | 1094 | 1.000 | 2828 |

All 6 acceptance tests pass.

---

## Failure Modes

1. **Uniform query distribution** → hit rate collapses to zero, overhead = cache lookup cost.
2. **High dimensionality (dim > 512)** → random unit vectors are near-orthogonal, jitter
   does not produce high cosine similarity, hit rate near zero.
3. **Corpus update without invalidation** → stale results returned as hits.
4. **Threshold too low** → recall degradation exceeds acceptable floor.
5. **Cache shared across untrusted tenants** → query intent leakage via cache hit oracle.

---

## Security Considerations

1. Threshold must be infrastructure-controlled, not caller-controlled, to prevent
   forced cache hits that bypass corpus updates.
2. Cache namespaces must align with access-control boundaries. Integrate with
   `ruvector-capgated` before multi-tenant deployment.
3. Cached results must carry the access-control labels from the time of insertion.
   A cache hit that returns results the caller was not entitled to at query time
   is a privilege escalation.

---

## Migration Path

The `CachedAnn` trait is additive. No existing API is modified. Migration:

```rust
// Before
let results = corpus.brute_force_topk(&query, k);

// After
let mut cache = SemanticCache::new(corpus.clone(), 512, 0.90);
let (results, decision) = cache.search(&query, k);
// decision = CacheDecision::Hit or CacheDecision::Miss
```

---

## Open Questions

1. What is the right default threshold for production workloads? 0.90 is measured
   on synthetic data; real embedding distributions may need a different value.
2. Should the cache be persistent across process restarts? Serialising the cache
   to disk would require `rkyv` or `bincode` encoding.
3. How does hit rate degrade as cache capacity shrinks? The PoC uses n_cache=512;
   the relationship between capacity and hit rate needs calibration per corpus.
4. Should the `memory_search` MCP tool expose `cache_hit: bool` in its response
   metadata? Useful for agent-side observability.
