# ADR-272: Scalar-Quantized HNSW with Online Calibration and Approximate-then-Rerank Search

**Date:** 2026-07-07  
**Status:** Accepted  
**Deciders:** nightly research agent  
**Supersedes:** —  
**Superseded by:** —

---

## Status

Accepted — proof-of-concept implemented, benchmarked, and merged as `crates/ruvector-sq-hnsw`.

---

## Context

RuVector's existing ANN search infrastructure stores vectors as f32 (4 bytes per dimension).  At 128 dimensions with 10M vectors, that is ~5 GB of raw vector data, excluding the HNSW graph topology.  Two forces push toward reducing per-vector footprint:

1. **Memory pressure at scale** — cache locality dominates HNSW traversal latency.  Smaller codes mean more nodes fit in L2/L3, directly reducing random-access penalties during beam search.
2. **Throughput goals** — RuVector targets sub-millisecond p99 at moderate ef_search values.  Integer arithmetic on 8-bit codes is 2–4× faster than f32 SIMD on most x86-64 microarchitectures when the compiler cannot auto-vectorise the f32 loop.

Scalar quantization (SQ8) is the simplest lossy compression scheme that preserves L2 distance ordering: map each dimension linearly to `[-127, 127]` using per-dimension min/max statistics collected over a calibration corpus.  The scheme is:

```
q[d] = clamp(round((x[d] − min[d]) / (max[d] − min[d]) * 254 − 127), −128, 127)
```

The critical design question for a production system is **where** the quantization boundary sits relative to the HNSW graph:

| Strategy | Graph traversal | Final score | Memory |
|----------|-----------------|-------------|--------|
| A: f32 graph, i8 storage | f32 | f32 | 5× f32 (codes + originals + graph) |
| B: i8 graph (this ADR) | i8 | i8 | 1× f32 equivalent |
| C: i8 graph + f32 rerank | i8 | f32 | ~1.25× f32 equivalent |

Strategy B maximises speed; strategy C recovers precision lost during quantisation without re-traversing the graph.  Strategy A is not studied here — it is trivially correct but offers no traversal speedup.

---

## Decision

Implement three composable index variants in a new crate `ruvector-sq-hnsw`:

1. **`F32Index`** — standard HNSW, f32 storage and traversal.  Baseline.
2. **`Sq8Index`** — HNSW with i8 codes for both storage and graph traversal.  Maximum speed.
3. **`Sq8RerankIndex`** — i8 graph traversal, f32 exact rerank of the top `overquery_factor × k` candidates.  Balanced.

All three expose the same `AnnIndex` trait so callers are not coupled to a specific variant.

**Calibration strategy:** Collect the first `N` vectors into a buffer, compute per-dimension min/max, freeze calibration, then flush the buffer into the graph.  This is "offline" calibration — the quantizer is determined entirely from the data seen before the first graph edge is written.  Online extension (incremental recalibration with graph rebuild) is deferred to a future ADR.

**Distance function design:** `HnswGraph::insert_node` and `::search` accept a two-argument closure `dist_fn(i, j) -> f32` rather than a single-argument `dist_to_query(j)`.  The two-argument form is mandatory for correct neighbor pruning: when an existing node `nb` overflows its neighbor list, candidates must be ranked by distance *from `nb`*, not from the inserting node.  A single-argument closure can only express distance from the current inserting node, which silently breaks pruning and causes severe recall degradation (observed: recall drops from 0.77 to 0.03 on the same data).

---

## Consequences

### Positive

- **4× memory reduction** for Sq8Index: 128 B/vector vs 512 B/vector (f32, 128 dims).
- **35% latency reduction**: Sq8Index mean 257 μs vs F32Index 397 μs (−35%).
- **55% QPS improvement**: Sq8Index 3,897 QPS vs F32Index 2,521 QPS.
- **Negligible recall impact**: Sq8Index recall@10 = 0.7682 vs F32Index 0.7704 (−0.3%).
- Rerank variant recovers to 0.7690 with only +5% latency overhead over Sq8Index.
- Build time 33% faster for quantized variants (5.6 s vs 8.3 s, n=10k).
- Trait-based design allows switching between variants without caller changes.

### Negative / Tradeoffs

- Calibration requires holding a buffer of `calibration_budget` raw f32 vectors before inserting any graph edges.  Memory spike = `calibration_budget × dims × 4` bytes.
- Calibration is frozen at construction.  Vectors with values far outside the calibration range are clamped and lose precision.  Streaming or adversarial distributions (e.g., values grow monotonically) will degrade over time.
- `Sq8RerankIndex` stores both i8 codes and f32 originals: 640 B/vector (128 dims), 25% more than F32Index.  It is memory-expensive if the use case is purely storage-limited.
- The `dist_fn(i, j)` closure captures a shared reference to the code/data slice.  Parallel insertion (multiple threads sharing the graph) is not safe without additional synchronisation — deferred to a future ADR.

---

## Alternatives Considered

### 1. Product Quantization (PQ) with ADC

**PQ** splits each vector into `M` sub-vectors and quantizes each independently via a learned codebook, achieving 16–32× compression at similar recall.  The asymmetric distance computation (ADC) runs query against all `M` codebook tables and sums look-ups.

**Why deferred:** PQ requires k-means training (typically offline, O(N·M·iter) time), codebook storage, and more complex distance kernels.  SQ8 achieves the primary engineering goal (4× memory reduction, sub-millisecond search) with far simpler machinery.  PQ is a natural follow-on once this calibration infrastructure is proven in production.

### 2. Binary quantization (RaBITq / 1-bit)

**RaBITq** encodes each dimension as a single bit, achieving 32× compression with a fast popcount kernel.  Recall drop is more severe (typically 10–20% at comparable ef) and requires higher overquery ratios.

**Why deferred:** 32× compression is compelling for extreme scale.  However, the recall-vs-speed operating point for 1-bit encoding is significantly different from SQ8, and the popcount kernel requires SIMD intrinsics for best performance.  A dedicated nightly is planned.

### 3. fp16 (half-precision float) storage

**fp16** reduces storage to 2 B/dim (2× reduction) with negligible rounding error.  x86-64 f16 arithmetic requires AVX-512 FP16 (Sapphire Rapids+).

**Why deferred:** Storage benefit is 2× vs 4× for SQ8.  On pre-Sapphire Rapids hardware, f16 must be widened to f32 for arithmetic, negating speed gains.  SQ8 achieves a better memory-speed tradeoff on common server hardware.

### 4. IVF (Inverted File Index) + SQ8

**IVF** partitions the corpus into `nlist` Voronoi cells and searches only `nprobe` cells at query time.  Combined with SQ8, it can achieve 10–100× throughput improvements for large N.

**Why deferred:** IVF requires a training phase (k-means clustering) and introduces a different recall-latency trade-off surface.  The HNSW-only baseline is simpler to reason about and is the right starting point for the calibration infrastructure.

---

## Implementation Plan

The implementation is complete in `crates/ruvector-sq-hnsw/src/`:

| File | Role | Lines |
|------|------|-------|
| `lib.rs` | Public API, trait definitions, exact_knn | 102 |
| `quantizer.rs` | ScalarQuantizer: calibrate, encode, decode, sq8_l2_sq | 205 |
| `hnsw.rs` | HnswGraph: insert_node, search, search_layer, random_level | ~260 |
| `index.rs` | F32Index, Sq8Index, Sq8RerankIndex | ~370 |
| `main.rs` | Benchmark binary with acceptance tests | 345 |

All files are under 500 lines.

Future integration milestones:

1. **M1 (next sprint):** Wire `Sq8Index` as an optional storage backend in `ruvector-core` behind a feature flag `sq8`.
2. **M2:** Add SIMD distance kernel for `sq8_l2_sq` using `std::simd` (portable SIMD, nightly gated, then stable when stabilised).
3. **M3:** Online calibration with periodic histogram merging — no full rebuild, approximate recalibration via exponential moving average.
4. **M4:** Parallel safe insertion via epoch-based concurrency or sharded subgraphs.

---

## Benchmark Evidence

All numbers from a single `cargo run --release -p ruvector-sq-hnsw` on:
- n=10,000, dims=128, queries=200, k=10
- M=16, ef_construction=200, ef_search=64
- OS: linux, Arch: x86_64, rustc 1.94.1

```
┌─────────────────┬──────────┬────────────┬────────────┬────────────┬─────────────┬──────────────┬──────────────┐
│ Variant         │ Recall@10│ Mean(μs)   │ p50(μs)    │ p95(μs)    │ QPS         │ Mem/vec(B)   │ Build(ms)    │
├─────────────────┼──────────┼────────────┼────────────┼────────────┼─────────────┼──────────────┼──────────────┤
│ F32 (baseline)  │ 0.7704   │      396.7 │      386.6 │      464.0 │        2521 │          512 │         8333 │
│ SQ8 (no-rerank) │ 0.7682   │      256.6 │      244.2 │      302.3 │        3897 │          128 │         5556 │
│ SQ8 + Rerank    │ 0.7690   │      270.6 │      259.6 │      315.9 │        3696 │          640 │         5595 │
└─────────────────┴──────────┴────────────┴────────────┴────────────┴─────────────┴──────────────┴──────────────┘
```

Acceptance tests (all PASS):
- F32 recall@10 ≥ 0.70: PASS (0.7704)
- SQ8 recall@10 ≥ 0.55: PASS (0.7682)
- Rerank recall@10 ≥ 0.70: PASS (0.7690)
- SQ8 mean latency ≤ 1.5× F32: PASS (256.6 μs vs 595.1 μs threshold)
- SQ8 mem ratio ∈ [0.20, 0.30]: PASS (0.250)

---

## Failure Modes

| Failure | Trigger | Symptom | Mitigation |
|---------|---------|---------|------------|
| Recall collapse | calibration_budget too small (<100 vectors) | per-dim statistics unreliable, codes collide | enforce `calib_budget >= 256` in constructor |
| Clamping at insert | test distribution has wider range than calibration | silent precision loss, no crash | log a warning when > 1% of encoded values hit ±127 clamp |
| Recall collapse (pruning bug) | single-arg dist_fn passed to insert_node | recall@10 ≈ 0.03 | unit test in `index.rs` verifies recall@10 ≥ 0.60 for sq8 |
| OOM at calibration | large calibration_budget, high dims | Vec allocation fails | stream calibration using reservoir sampling |
| Integer overflow in sq8_l2_sq | diff per dim up to 255, 128 dims, 255²×128 = 8.3M | fits in i32; safe with i64 | i64 accumulator used; panic impossible |

---

## Security Considerations

- No network I/O; no user-supplied format parsing in this crate.
- `calibrate` asserts all calibration vectors have equal dimensionality — panics (not UB) on violation.
- All array accesses use safe Rust indexing (bounds-checked); no `unsafe` blocks.
- Distance functions accept closures over borrowed data — no raw pointers.
- The crate does not persist or transmit any data.

---

## Migration Path

For existing callers using `ruvector-core` vector search:

1. No API change required — the new `AnnIndex` trait is additive.
2. Drop-in replacement: swap `F32Index::new(config, dims)` for `Sq8Index::new(config, dims, corpus_size)` and call `add` identically.
3. Callers that require exact distances in search results should use `Sq8RerankIndex`; `SearchResult.distance` is then a true f32 L2².
4. Existing indices are not serializable in this ADR — persistence format is deferred to a future ADR.

---

## Open Questions

1. **Incremental calibration:** How to update per-dimension statistics as the distribution shifts without a full rebuild?  Candidate approach: maintain running min/max with exponential decay; rebuild quantizer and re-encode every `N` insertions.

2. **SIMD acceleration of `sq8_l2_sq`:** The current loop compiles to scalar i64 arithmetic.  With `std::simd` (portable SIMD), the inner loop could process 16 dimensions per cycle on x86_64 with AVX2.  Estimated 4–8× speedup for the distance kernel alone.

3. **Asymmetric quantization:** The current scheme maps uniformly to `[-127, 127]`.  For distributions with outliers (e.g., learned embeddings from large language models), a percentile-clipping strategy (e.g., use p1–p99 range instead of min–max) could reduce quantization error for typical values at the cost of clamping outliers harder.

4. **Graph compression:** The HNSW graph topology (neighbor lists) currently uses `Vec<Vec<usize>>` — each neighbor ID is 8 bytes.  At M=16, layer-0 stores 32 neighbors × 8 B = 256 B/node.  Delta-coding or 4-byte IDs (for N < 4B) would reduce graph overhead.

5. **Thread safety:** The current `insert_node` takes `&mut self`.  A read-write lock around the graph enables concurrent reads; a segment-locking scheme could enable parallel batch insertions.
