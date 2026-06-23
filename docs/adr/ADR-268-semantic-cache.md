# ADR-268: Semantic Vector Cache for RAG and Agent Memory

**Status:** Proposed
**Date:** 2026-06-23
**Author:** Nightly research agent
**Crate:** `crates/ruvector-semantic-cache`
**Branch:** `research/nightly/2026-06-23-semantic-cache`

---

## Context

RuVector serves as a cognition substrate for AI agents. Every query to an agent memory store, RAG pipeline, or semantic search endpoint triggers a full approximate nearest-neighbor (ANN) search across the vector corpus. For agents with repetitive or semantically similar query patterns — the dominant workload in production RAG systems — this full-corpus search is paid every time.

Semantic caching addresses this: store (query_vector → result_ids) pairs, and on a new query, first check whether a sufficiently similar query was recently answered. If so, return the cached result immediately, skipping the corpus search.

The existing RuVector crates (`ruvector-agent-memory`, `ruvector-lsm-ann`, `ruvector-coherence-hnsw`) have no caching layer. Every query is cold. Production systems built on these primitives bear the full corpus-search cost regardless of query repetition.

The SOTA in 2026 includes several semantic cache implementations (GPTCache, QVCache, vCache, LiteLLM), but all are Python-first and lack:
- Native integration with a Rust vector store's HNSW index
- Cache invalidation hooks tied to the vector store WAL
- A trait-based API that other RuVector crates can implement against
- WASM-compatible, dependency-free implementation

---

## Decision

**Adopt** `ruvector-semantic-cache` as a first-class RuVector capability.

Specifically:
1. The `ruvector-semantic-cache` crate enters the workspace as a standalone PoC.
2. The `SemanticCache` trait defines the public API: `get`, `put`, `invalidate`, `stats`.
3. Three variants are implemented and benchmarked: `NoCache`, `FixedSemanticCache`, `AdaptiveSemanticCache`.
4. The internal key index is a lightweight HNSW over L2-normalized query vectors.
5. Production integration into `ruvector-agent-memory` and `ruvector-server` is gated on a production corpus evaluation (query log replay from a real agent workload).

---

## Consequences

**Positive:**
- `FixedSemanticCache` achieves **13.49× mean latency reduction** on near-duplicate query workloads (66.6 µs vs. 899.2 µs for NoCache).
- 100% hit rate on near-duplicate queries (cosine similarity ≥ 0.985 with noise_scale=0.02, threshold=0.92).
- **Zero false positives** on random queries (threshold 0.92 correctly rejects all uncached queries in the random workload).
- Memory footprint: **103 KB for 200 cached entries** (128 dims). Fits in WASM, Cognitum Seed, edge devices.
- No external dependencies beyond `rand`. Zero crates outside the workspace.
- Cache warmup: 200 entries indexed in 23.4 ms.
- Breakeven: cache provides net latency benefit at **≥23% hit rate** (proven by 50% hit rate mixed workload at 1.55× speedup).

**Negative:**
- Miss-path overhead: a cache miss takes ~1,147 µs vs. 899 µs for no-cache (27.6% overhead on misses due to HNSW key lookup before corpus search).
- `AdaptiveSemanticCache` over-tightens for uniform near-dup workloads (threshold rises to ~0.99, reducing hit rate to 13.2%). Correct use case: heterogeneous workloads with precision-recall tradeoffs.
- No concurrent access: current `CacheStore` requires external synchronization for multi-threaded use.
- No corpus-update invalidation: `invalidate()` flushes the entire cache. Partial invalidation is future work.

---

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Exact-match hash cache (L1 only) | Misses semantically equivalent rephrased queries. Hash collision resistant but recall-incomplete. |
| LRU cache over the corpus HNSW entry point | Not query-level caching; doesn't skip the corpus search. |
| Use external Redis/Memcached | External service dependency; not Rust-native; no vector-similarity matching. |
| vCache per-entry adaptive threshold | More complex to implement correctly; requires feedback signal (LLM judge) for proper threshold convergence. Research direction, not today's implementation. |
| Two-level (hash + semantic) | Additional complexity; exact-hash hit rate is negligible for real queries with whitespace/punctuation variation. Addressed in research notes. |

---

## Implementation Plan

**Phase 1 (this branch):** PoC crate with three variants, tests, benchmark, research doc, ADR. ✓

**Phase 2 (production hardening):**
1. Thread-safe wrapper: `Arc<RwLock<CacheStore>>` in a new `ConcurrentSemanticCache`.
2. Persistence: `serialize() → Vec<u8>` / `deserialize(bytes) → Self` using `bincode` or `rkyv`.
3. TTL: add `expires_at: Option<Instant>` to `CacheEntry`; sweep on `get` and `put`.
4. Hook into `ruvector-agent-memory`: wrap `AgentMemory::search` with a `FixedSemanticCache` that gates on same namespace.

**Phase 3 (integration):**
1. Corpus-update invalidation: subscribe to `ruvector-lsm-ann` compaction events via channel; on compaction, `invalidate()` affected entries by namespace.
2. MCP tool surface: expose `semantic_cache_get/put/invalidate` tools via `ruvector-server`.
3. RVF packaging: serialize cache state into an RVF bundle for portable edge deployment.

---

## Benchmark Evidence

From `cargo run --release -p ruvector-semantic-cache --bin benchmark` on 2026-06-23:

- **13.49× speedup**: 66.6 µs vs. 899.2 µs mean latency on near_dup workload
- **1.55× speedup**: 572.4 µs vs. 886.6 µs mean latency on mixed (50% hit rate) workload
- **100% hit rate** on near-duplicate queries (cosine similarity ≥ 0.985, threshold 0.92)
- **0% false positive rate** on random queries (no wrong cached results)
- **103.1 KB** cache memory for 200 entries at 128 dims
- **23.4 ms** warmup time for 200-entry cache from query history
- Dataset: 5,000-vector corpus, 200 history queries, 500 test queries/class

---

## Failure Modes

| Mode | Trigger | Risk | Mitigation |
|------|---------|------|-----------|
| Stale results | Corpus updated without invalidation | Medium: wrong results served | Hook invalidate() to corpus mutation log |
| False positive | Threshold too low | High: wrong search results | Keep threshold ≥ 0.90 for 128-dim MRL embeddings |
| Cold cache | No warmup, first queries all miss | Low: just slower | Warm from query logs pre-deployment |
| Cache thrash | max_entries too small | Medium: high eviction rate | Tune max_entries to P90 unique query count |
| Adaptive overfitting | Uniform workload causes threshold instability | Low: just lower hit rate | Use FixedSemanticCache for uniform workloads |

---

## Security Considerations

1. **Cache poisoning:** A malicious actor can poison the cache by inserting a high-similarity but incorrect (query, result_ids) pair. Mitigation: proof-gate `put()` with `ruvector-proof-gate` (merkle hash chain over inserted pairs); only trusted writers can mutate the cache.
2. **Query pattern leakage:** Cache hit/miss timing reveals whether a similar query was previously made. In multi-tenant deployments, this is a side-channel. Mitigation: constant-time response (always wait for a fixed time before responding) or per-tenant isolated cache instances.
3. **Index poisoning via crafted embeddings:** Adversarially crafted embeddings could disrupt the HNSW key graph. Mitigation: validate embedding norms on `put()` (already done via `l2_normalize`).

---

## Migration Path

`ruvector-semantic-cache` is a new standalone crate. There is no migration from existing code.

Integration into existing crates is additive:
- Wrap `AgentMemory::search` in a `FixedSemanticCache::get` check.
- If hit: return immediately. If miss: call `search`, then `put`.
- The trait-based design (`SemanticCache`) means the wrapper is a two-line addition in the search path.

---

## Open Questions

1. **Recall guarantee:** Is 0.823 recall (near-dup queries vs historical ground truth) acceptable for production? Depends on the application. Research gap: measure on real corpus with real embedding model.
2. **Threshold tuning guidance:** How should users pick the cosine threshold? Need a calibration procedure using a sample of the actual query workload.
3. **Invalidation granularity:** When a single vector is updated in the corpus, which cache entries are affected? Need a mapping from result_ids back to cache entries (inverted index on result_ids).
4. **Embedding model version pinning:** Should cache entries store the embedding model version hash to auto-invalidate on model upgrade?
5. **Distributed cache:** For a replicated RuVector cluster, should the semantic cache be per-node or shared? Shared reduces total misses but adds consistency complexity.
