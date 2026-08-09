# ADR-298: Semantic Query Cache for RuVector Agent Memory

**Status:** Proposed  
**Date:** 2026-08-09  
**Deciders:** ruvnet engineering  
**Tags:** agent-memory, vector-search, caching, performance

---

## Context

AI agents powered by RuVector issue repeated semantic queries. A coding agent that asks "what functions handle authentication?" may issue dozens of nearly identical vector searches across a long session. Each search costs O(N·D) time scanning the base vector set, wasting compute on semantically duplicate work.

Semantic caching — caching (query_vector → result_ids) pairs and detecting cache hits using cosine similarity — eliminates this waste. Unlike exact-match caching (which requires byte-identical queries), semantic caching tolerates natural language variation: the same intent expressed with slightly different embeddings still hits the cache.

This is distinct from all prior nightly work:
- Not a new ANN index variant (cf. coherence-hnsw, speculative-ann, diverse-beam-ann)
- Not a quantization scheme (cf. rabitq, pq-adc-search, matryoshka)
- Not a graph repair or merge operation (cf. hnsw-delete-repair)
- A first-class caching layer above the retrieval engine

---

## Decision

Add `crates/ruvector-semantic-cache` with two production-relevant backends:

1. **LinearScanCache**: O(C·D) exhaustive cosine scan. Optimal for C < 1 000 entries. Zero setup cost.

2. **ShardedCache**: LSH-bucketed cache with 6-bit random projection (64 buckets) and 1-hamming-distance multi-probe. Lookup scans ~7 × C/64 entries. Suitable for C up to ~50 000 entries.

Both implement the `QueryCache` trait, allowing callers to swap backends without changing call sites.

The cache uses pre-normalized (unit) query vectors so cosine similarity reduces to a dot product — no sqrt required per comparison.

---

## Consequences

### Positive

- **9.5x mean latency reduction** on clustered workloads (LinearCache, measured).
- **6.7x mean latency reduction** (ShardedCache, measured).
- **90% hit rate** on agent-style workloads with 50 clusters and 500 queries.
- Zero external dependencies; pure Rust stdlib.
- TTL eviction prevents stale results after collection mutations.
- Composable with any `QueryCache`-compatible backend.

### Negative / Tradeoffs

- **Recall loss on hits**: cached results come from a similar prior query, not the current query. Mean recall ≈ 0.74 in benchmarks (vs 1.0 for exact search). This is inherent and documented.
- **Cache invalidation**: mutations to the base vector collection make cached results stale. A generation counter or TTL must be used. Not yet implemented as a first-class API.
- **Memory overhead**: each cached entry consumes `dims × 4 + k × 8 + overhead` bytes. At D=128, k=10: ~592 bytes per entry. 500 entries = ~290 KB.

---

## Alternatives Considered

### 1. Exact-match (hash) cache
Cache by exact query byte-hash. Zero recall loss. But agents rephrase constantly — the exact-match cache hit rate approaches zero in practice.

### 2. Cluster-based cache (pre-clustered centroids)
Pre-cluster the query space into K centroids at startup, then route queries to centroids. High hit rate but requires offline training on query distribution, which is unknown for new agents.

### 3. Full HNSW cache index
Build an HNSW index over cached query vectors for O(log C) lookup. Highest scalability but adds significant code complexity. Appropriate for C > 100 000; overkill for typical agent sessions.

### 4. No change
Accept repeated full scans. Correct but slow; unacceptable for long-running agents with recurring queries.

---

## Implementation Plan

1. [x] `crates/ruvector-semantic-cache/src/lib.rs` — `QueryCache` trait, `NoCache`, utilities.
2. [x] `crates/ruvector-semantic-cache/src/linear.rs` — `LinearScanCache`.
3. [x] `crates/ruvector-semantic-cache/src/sharded.rs` — `ShardedCache` with multi-probe LSH.
4. [x] `crates/ruvector-semantic-cache/src/dataset.rs` — deterministic clustered dataset generator.
5. [x] `crates/ruvector-semantic-cache/src/bin/benchmark.rs` — benchmark binary with three variants.
6. [ ] Production: integrate with `ruvector-server` HTTP API as an optional middleware layer.
7. [ ] Production: generation counter for cache invalidation on collection writes.
8. [ ] Production: MCP tool surface exposing cache hit/miss metrics.

---

## Benchmark Evidence

Measured on x86_64 Linux, Rust stable, release build. N=10 000 base vectors, D=128, 50 clusters, 500 queries, k=10, threshold=0.92, noise_std=0.02.

| Variant     | Hit Rate | Mean µs | p50 µs | p95 µs | QPS  | Mem KB | Recall | Accept |
|-------------|----------|---------|--------|--------|------|--------|--------|--------|
| NoCache     | 0.0%     | 1345.5  | 1333   | 1462   | 743  | 0.0    | 1.000  | PASS   |
| LinearCache | 90.0%    | 141.6   | 7      | 1342   | 7064 | 29.7   | 0.744  | PASS   |
| ShardedCache| 85.6%    | 202.3   | 2      | 1391   | 4944 | 45.8   | 0.757  | PASS   |

p50 latency of LinearCache (7 µs) vs NoCache (1333 µs): 190x reduction on cache hits.

---

## Failure Modes

1. **Low hit rate on diverse query workloads**: if each agent query is semantically distinct, the cache never warms up. Hit rate approaches 0%. The cache adds lookup overhead with no benefit. Mitigation: monitor hit rate and disable cache when hit_rate < 5%.

2. **Stale results after index mutation**: inserting or deleting base vectors changes which IDs are top-k for a given query. Cached results become incorrect. Mitigation: generation counter incremented on mutations; lookups reject entries from prior generations.

3. **Memory unbounded growth**: without capacity limits or TTL, the cache grows indefinitely. Mitigation: capacity limit (implemented) and TTL eviction (implemented). Default capacity = 2× expected cluster count.

4. **Bucket boundary misses (ShardedCache)**: two very similar queries may fall into different LSH buckets. Multi-probe (implemented) recovers ~89% of near-boundary cases. Residual ~11% become false misses.

---

## Security Considerations

- Cache entries contain result IDs from previous queries. If the caller does not filter results by access policy after retrieval, a cache hit could return IDs the current user is not authorized to see. **The cache must not be used as a substitute for post-retrieval access control.**
- No cross-user cache sharing should occur without explicit consent. Each agent session should maintain a private cache, or cache entries should be tagged with access context and compared before returning.

---

## Migration Path

No migration required. The cache is opt-in: callers that do not instantiate a `QueryCache` are unaffected. To enable:

```rust
use ruvector_semantic_cache::linear::LinearScanCache;
use ruvector_semantic_cache::QueryCache;

let mut cache = LinearScanCache::new(200);
// Before DB search:
if let Some(results) = cache.lookup(&query, 0.92, now_tick, ttl) {
    return results;
}
// After DB search:
cache.insert(query, results.clone(), now_tick);
```

---

## Open Questions

1. Should the cache be integrated into `ruvector-server` as HTTP middleware, or as a library the caller manages?
2. What is the right TTL default? Depends on collection write frequency, which varies by workload.
3. Should the ShardedCache auto-tune B (number of bits) based on observed similarity distributions?
4. Should cache miss/hit statistics be exposed as a Prometheus metric or MCP tool resource?
