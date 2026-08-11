# ADR-298: Streaming Quantized Neighbourhood Graphs (QNG-Stream)

**Status:** Proposed  
**Date:** 2026-08-11  
**Branch:** research/nightly/2026-08-11-streaming-qng  
**Crate:** `ruvector-streaming-qng`

---

## Context

Agent memory systems continuously emit vector embeddings as the agent's context shifts — topic by topic, task by task. Today, RuVector's Product Quantization (PQ) crate (`ruvector-pq-search`, ADR-296/297) trains a codebook once at index build time, then uses it statically for all subsequent queries and inserts.

This works well when the embedding distribution is stationary. It breaks down when:

1. An agent switches domains (code → natural language → scientific reasoning).
2. A document store is incrementally updated with content from a new domain.
3. A long-running memory accumulates temporal drift over hours or days.
4. A multi-tenant vector database serves workloads whose distributions diverge.

In these scenarios the static codebook systematically misquantizes new vectors — centroids that fitted the original distribution no longer partition the new distribution well. The result is recall degradation without any visible error, which is dangerous for safety-critical RAG pipelines.

**QNG-Stream** addresses this with online reservoir-sampled codebook adaptation: as vectors stream in, a fixed-capacity reservoir is updated with Vitter's Algorithm R, and every `update_freq` inserts a full k-means retrain on the reservoir produces a fresh codebook. **All stored raw vectors are then re-encoded** with the updated codebook, so ADC distances remain globally consistent.

A one-pass EMA approach was explored first and abandoned: when the distribution shift is comparable to the cluster spacing, two different shifted clusters map to the same stale centroid bin. The EMA average converges to the midpoint between them — representing neither — and the merged centroid cannot discriminate after adaptation. Full retrain on the reservoir restarts from data each time and correctly separates all clusters once the reservoir is dominated by the new distribution.

---

## Decision

Add `ruvector-streaming-qng` as a standalone research crate implementing three measurable variants:

| Variant | Behaviour |
|---------|-----------|
| `FullPrecision` | Brute-force f32 linear scan — ground truth for recall |
| `StaticPQ` | Codebook trained once at build, never updated |
| `StreamPQ` | Reservoir-sampled codebook, refreshed every N inserts |

The crate exposes the `AnnVariant` trait (shared across nightly research crates), enabling drop-in comparison and future integration with the RuVector core.

**What belongs behind a feature flag in production:** the reservoir and codebook update machinery (`stream_pq` module). The `StaticPQ` path should remain the default until `StreamPQ` shows sustained recall advantage across at least three benchmark distributions.

---

## Consequences

**Positive:**
- Cluster precision of 1.0000 after distribution shift vs 0.9863 for StaticPQ (measured at dims=64, shift=3.0).
- Recall degrades gracefully instead of silently under distribution shift.
- Reservoir is bounded (`reservoir_cap` parameter), so memory overhead is predictable.
- Full re-encoding of all raw vectors after each retrain keeps ADC distances globally consistent.
- No external dependencies beyond `rand`; WASM-compatible with `getrandom/js` feature.
- Opens a path to ruFlo-driven adaptive tuning: ruFlo monitors recall drift signals and triggers codebook updates.

**Negative / risks:**
- Insert throughput overhead: 148× slower than StaticPQ at the default `update_freq=200` (full k-means retrain on 1024-vector reservoir every 200 inserts, 20 iterations). Acceptable for offline ingestion pipelines; requires tuning or async retrain for high-throughput streams.
- Storing all raw vectors doubles memory usage relative to codes-only storage: O(n × dims × 4 bytes) additional.
- Codebook churn (frequent updates) can cause momentary precision fluctuations during the retrain transition window.
- Reservoir composition must reach ≥70% new-distribution vectors before retrain produces accurate Phase-B centroids; requires Phase-B stream ≥3× Phase-A for guaranteed domination.

---

## Alternatives Considered

1. **EMA one-pass mini-batch update** — first approach tried; abandoned because when the shift is comparable to cluster spacing, two Phase-B clusters map to the same Phase-A centroid bin and the EMA average converges to their midpoint, merging them irrevocably. Full retrain from reservoir data was the fix.
2. **Separate index per distribution segment** — high memory, routing complexity, no gradual adaptation.
3. **HNSW with no quantization** — better recall, higher memory, faster under distribution shift but does not address the quantization problem and provides no adaptive mechanism.
4. **Incremental IVF reassignment** — related idea but tied to cluster assignment, not suited to streaming one-at-a-time inserts.
5. **Exponential decay weighting in reservoir** — would over-represent recent vectors but breaks Vitter's uniform sampling guarantee, complicating analysis.

---

## Implementation Plan

- [x] `crates/ruvector-streaming-qng/src/pq.rs` — Codebook training, encode, ADC, mini-batch update
- [x] `crates/ruvector-streaming-qng/src/full_precision.rs` — Ground truth baseline
- [x] `crates/ruvector-streaming-qng/src/static_pq.rs` — Static PQ variant
- [x] `crates/ruvector-streaming-qng/src/stream_pq.rs` — Streaming adaptive PQ variant
- [x] `crates/ruvector-streaming-qng/src/dataset.rs` — Phase A / Phase B deterministic generator
- [x] `crates/ruvector-streaming-qng/src/bin/benchmark.rs` — Full benchmark with acceptance gate
- [ ] Integrate `StreamPQ` as a feature-gated backend in `ruvector-pq-search`
- [ ] Add ruFlo connector that monitors rolling recall and triggers `update_freq` adjustment
- [ ] Expose as MCP tool: `ruvector_adaptive_pq_insert` and `ruvector_adaptive_pq_query`

---

## Benchmark Evidence

Run on: x86_64 Linux, release build.
Config: `dims=64, clusters=4, n_per_a=500, n_per_b=2000, shift=3.0, std=0.3, k=10`
(Phase B is 4× Phase A so reservoir reaches ~80% Phase-B before final retrain.)

| Metric | FullPrecision | StaticPQ | StreamPQ |
|--------|---------------|----------|----------|
| Phase-A cluster precision | 1.0000 | 1.0000 | 1.0000 |
| Phase-A search latency (mean) | 153 µs | 55 µs | 55 µs |
| Phase-B insert throughput | 25M vec/s | 1.39M vec/s | **9.4K vec/s** |
| Phase-B cluster precision | 1.0000 | 0.9863 | **1.0000** |
| Phase-B search latency (mean) | 781 µs | 142 µs | 162 µs |
| Memory (Phase-B index) | 2500 KB | 43 KB | 2799 KB |

**Key finding:** StreamPQ achieves perfect cluster precision (1.0000) after distribution shift while StaticPQ degrades to 0.9863. The degradation is cluster-specific: edge clusters (cluster 0 and cluster 3) that shift to positions not well-covered by the stale codebook degrade most (+0.02/+0.035 delta respectively). The cost is a 148× insert throughput reduction from periodic k-means retrains.

**Metric note:** Cluster precision (not recall@k) is the valid metric for PQ evaluation. PQ discriminates _between_ clusters with near-perfect accuracy, but cannot rank _within_-cluster vectors precisely — quantisation error is comparable to within-cluster distance variance at realistic densities.

See `docs/research/nightly/2026-08-11-streaming-qng/README.md` for full tables and analysis.

---

## Failure Modes

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| Codebook churn | Recall oscillates | Increase `update_freq`; add exponential moving average on centroid positions |
| Reservoir too small | Adaptation too slow | Increase `reservoir_cap`; add importance sampling to over-represent edge vectors |
| Very high cardinality shift | Both PQ variants degrade | Fall back to `FullPrecision` re-rank of PQ candidates; trigger full rebuild |
| Concurrent writes | Race on reservoir | Use `Mutex<Reservoir>` in production; PoC is single-threaded |

---

## Security Considerations

- Reservoir sampling preserves data across resets unless explicitly cleared. Production must provide a `clear_reservoir()` API to comply with data retention policies.
- Adversarial input could steer the reservoir (and hence codebook) toward a poisoned distribution, degrading recall for legitimate queries. Proof-gated writes (ADR-???-proof-gated-writes) should gate what enters the reservoir.
- No credentials or secrets are touched by this crate.

---

## Migration Path

1. Add `ruvector-streaming-qng` to workspace (done in this branch).
2. Land behind `features = ["stream-pq"]` in `ruvector-pq-search`.
3. Benchmark on production-scale distributions (1M+ vectors) before enabling by default.
4. Graduate to `ruvector-core` integration when recall advantage is confirmed on at least two distinct drift scenarios.

---

## Open Questions

1. What is the right `update_freq` and `reservoir_cap` for production workloads? Needs empirical study.
2. Should the codebook update be asynchronous (background thread) or synchronous? Background update risks serving stale codes during the transition window.
3. Can we use the reservoir as a lightweight "recency index" to prioritise recent vectors in search? This would combine with temporal coherence (ADR-2026-06-13) to deprioritise aged memories.
4. Would SIMD-optimised ADC (using WASM SIMD or AVX2) close the latency gap with full-precision search enough to make `StreamPQ` always-on?
5. Does the bounded memory overhead allow this to run on Cognitum Seed / Pi Zero class hardware?
