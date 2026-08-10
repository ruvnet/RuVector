# ADR-298: Semantic Query Cache for ANN Search

- **Status**: Accepted
- **Date**: 2026-08-10
- **Crate**: `ruvector-semantic-cache`
- **Related**: ADR-297 (Adaptive Compression & Retrieval Plane), ADR-272 (Speculative ANN)

## Context

Agentic workloads generate semantically similar — but not bit-identical — queries
in rapid succession.  An agent retrieving "past planning context" and then
"recent memory about task planning" may produce query embeddings with cosine
similarity > 0.97.  Every such pair currently triggers two full ANN passes over
the vector index, wasting compute on results that differ by at most one or two
neighbours.

The existing speculative-ANN crate (ADR-272) reduces the *cost per query* by
drafting on quantized vectors then verifying on float32.  That is a per-query
optimization.  What is still missing is a *cross-query* optimization: recognising
that adjacent queries are semantically redundant and returning the cached result
instead of issuing a second ANN call.

This problem becomes acute on edge devices and in ruFlo workflow loops, where:

- Compute budgets are tight (edge appliance, WASM runtime).
- Agents issue hundreds of retrieval calls per session.
- Many calls are follow-ups on the same topic, differing only in phrasing.
- ANN index round-trips dominate latency.

A semantic query cache intercepts near-duplicate queries before they reach the
index.  Unlike an exact (hash-keyed) cache, it matches on approximate cosine
similarity so that rephrased queries benefit from prior results.

## Decision

Introduce `crates/ruvector-semantic-cache`, a standalone zero-dependency Rust
library providing three cache variants under a common `SemanticCache` trait:

| Variant | Match strategy | Use case |
|---------|---------------|----------|
| `ExactCache` | Bit-identical hash key | Baseline; zero false positives |
| `LinearCache` | Cosine scan, fixed threshold | Small caches (≤ 256 entries) |
| `AdaptiveCache` | Cosine scan, self-tuning threshold | Production; handles distribution shift |

### API shape

```rust
pub trait SemanticCache {
    fn query(&mut self, q: &[f32]) -> Option<Vec<SearchResult>>;
    fn insert(&mut self, q: Vec<f32>, results: Vec<SearchResult>);
    fn record_ann_latency(&mut self, ann_latency_ns: u64);
    fn stats(&self) -> &CacheStats;
    fn capacity(&self) -> usize;
    fn len(&self) -> usize;
}
```

Callers follow a simple protocol:
1. Call `query()` — `Some(results)` means a cache hit; skip ANN.
2. On `None` (miss), run ANN, then call `insert()` + `record_ann_latency()`.

### LinearCache

Ring-buffer of at most `capacity` (query, result) pairs.  Each incoming query
is normalised to unit length and compared via dot product against all stored
unit-normalised queries.  If `max_cosine_sim ≥ threshold`, the stored results
are returned.  Ring-buffer eviction replaces the oldest entry when full.

**Time complexity per query**: O(N × D) where N = cache size, D = dimensions.
With N ≤ 256 and D = 128, this is 32 768 multiplications — negligible versus
a brute-force ANN scan over 10 000 vectors (1.28 M multiplications).

### AdaptiveCache

Same as `LinearCache` but the threshold is self-tuned every `tune_interval`
queries.  A rolling precision check compares returned top-1 IDs against the
stored ground-truth ID; if the false-positive rate exceeds
`max_false_positive_rate`, the threshold is raised.  If the hit rate is below
`target_hit_rate` with no false positives, the threshold is lowered.  Bounds
`[min_threshold, max_threshold]` prevent runaway drift.

### Memory model

Per-entry cost (D = 128 dimensions, k = 10 results):
- Query vector: 128 × 4 = 512 bytes
- Result list: 10 × 8 = 80 bytes (u32 id + f32 distance)
- Overhead: ~48 bytes (Vec metadata)
- **Total per entry: ~640 bytes**

At N = 64: **~40 KB**  
At N = 256: **~163 KB**

Both fit comfortably inside edge L2 caches.

## Consequences

**Positive**:
- Measurable latency reduction on repeated/rephrased queries (see benchmark results).
- Pure Rust, zero dependencies — compiles to WASM unchanged.
- Pluggable via trait: any ANN backend benefits without modification.
- `AdaptiveCache` self-calibrates without human-in-the-loop tuning.
- Linear scan over ≤ 256 entries is faster than L3 cache miss for ANN index.

**Negative**:
- Cache is per-session in-memory; it does not persist across restarts without
  an RVF snapshot layer (future work — see §Open questions).
- False positives are possible when two semantically similar queries have
  genuinely different correct answers (distinct nearest neighbours).
  The `AdaptiveCache` threshold tuner detects and corrects for this.
- Linear scan cost is O(N × D); for N > 512 an approximate structure
  (e.g. a mini HNSW over the cached queries) would be preferable.

## Alternatives Considered

### A. Exact hash cache only
Rejected: bit-identical hits are too rare in embedding-based systems.  Agents
rephrase; model temperature introduces non-determinism; batching reorders.

### B. LRU cache with approximate deduplication on insert
Rejected: deduplication on insert doesn't help if two *incoming* queries are
near-duplicates but neither is in the cache yet.  The lookup-side similarity
check is the essential operation.

### C. Mini-HNSW over cached queries
Suitable for N > 512.  At N ≤ 256 the build cost and pointer overhead outweigh
the log-N search benefit.  Recommended as a follow-on crate for higher-capacity
use cases (see §Open questions).

### D. Embedding model memoisation at call site
Rejected: memoising at the embedding model level requires access to the raw text,
which is often unavailable to the retrieval layer.  The cache operates on
float32 vectors and is model-agnostic.

## Implementation Plan

1. `crates/ruvector-semantic-cache` — new crate (this ADR).
2. Feature flag `semantic-cache` in `ruvector-server` wires the cache in front of
   the HNSW search handler (future ADR).
3. MCP tool `vector/cache/stats` exposes hit rate and threshold over the model
   context protocol (future ADR).
4. ruFlo hook `on_cache_cold` triggers cache warm-up using the agent's recent
   query log (future ADR).
5. RVF snapshot extension to serialise/restore the cache across sessions
   (future ADR).

## Benchmark Evidence

See `docs/research/nightly/2026-08-10-semantic-query-cache/README.md` §Benchmark
Results for measured numbers.

Results from `cargo run --release -p ruvector-semantic-cache --bin benchmark`
on 10 000 × 128-dim dataset, 600 unique + 400 near-duplicate (ε = 0.04) queries
with topic-local ordering, cache capacity = 64, k = 10, x86_64 Linux release build:

| Variant | Hit rate | Recall@1 | Mean µs | QPS | Accept |
|---------|----------|----------|---------|-----|--------|
| ExactCache (baseline) | 0.0% | 1.000 | 1321.3 | 757 | PASS |
| LinearCache (0.97) | **40.0%** | 0.973 | **802.8** | **1246** | PASS |
| AdaptiveCache (0.95) | **40.0%** | 0.973 | **799.4** | **1251** | PASS |

Key: 39% mean latency reduction, 65% throughput gain at 40% near-dup workload,
recall@1 = 0.973 on cache hits (2.7% false-positive rate on top-1 result).

## Failure Modes

| Mode | Probability | Mitigation |
|------|------------|------------|
| False positive hit (wrong results returned) | Low at threshold ≥ 0.97 | AdaptiveCache FP counter; raise threshold automatically |
| Cache poisoning (stale results after index update) | Medium | TTL per entry (future work); or flush on index write |
| Ring-buffer stale eviction | Low | Swap for LRU eviction in production build |
| WASM size growth | Negligible | No unsafe, no std dependencies beyond collections |

## Security Considerations

The cache stores previous query vectors in memory.  In a multi-tenant
deployment (multiple agents sharing one cache), a query can infer approximate
content of another agent's recent queries by observing cache hits.  Mitigations:

1. **Per-tenant cache instance**: isolate by agent ID (recommended default).
2. **Differential privacy noise**: add small noise to stored query vectors to
   prevent exact inference (future ADR).
3. **Hit indicator suppression**: do not expose hit/miss in API responses where
   timing side-channels exist.

## Migration Path

- This crate is additive; no existing code is modified.
- Integration into `ruvector-server` is behind a compile-time feature flag.
- No migration is required for existing deployments.

## Open Questions

1. **Persistence**: should the cache survive process restarts via RVF snapshot?
2. **Mini-HNSW backend**: when should the linear scan graduate to an
   approximate structure?  The crossover point depends on D and cache capacity.
3. **Multi-tenant isolation**: should the trait include an `agent_id` parameter?
4. **Integration with `ruvector-agent-memory`**: agent memory is already a
   semantic store; should the cache be a thin layer over it rather than
   standalone?
5. **Threshold initialisation heuristic**: can we use the dataset intrinsic
   dimensionality to pick a good initial threshold automatically?
