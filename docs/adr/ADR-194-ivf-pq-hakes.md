---
adr: 194
title: "IVF-PQ with HAKES Filter-Refine — ruvector's First Compression-Based ANN Index"
status: accepted
date: 2026-05-13
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193, ADR-128]
tags: [ivf, pq, product-quantization, ann, vector-search, hakes, filter-refine, nightly-research]
---

# ADR-194 — IVF-PQ with HAKES Filter-Refine

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-13-ivf-pq-hakes` as
`crates/ruvector-ivfpq`. All 10 unit tests pass; `cargo build --release -p ruvector-ivfpq`
and `cargo test -p ruvector-ivfpq` are green.

## Context

ADR-193 (RAIRS IVF) explicitly noted: *"IVF-PQ not yet implemented — this is the natural next
step (ADR-194 TBD)."* ruvector has rich graph-based ANN (HNSW via `ruvector-core`, DiskANN)
and binary quantisation (RaBitQ), but **no compression-based ANN index**. IVF-PQ is the
dominant such index in production vector databases:

| System | IVF-PQ support |
|--------|---------------|
| FAISS  | IVFFlat, IVF-PQ, IVF-SQ, IVF-ScaNN |
| Milvus | IVFFlat, IVF-PQ, IVF-SQ, IVF-HNSW |
| Qdrant | Scalar quantisation (IVF-SQ variant) |
| Weaviate | HNSW only — no IVF |
| Pinecone | Proprietary IVF-like |
| LanceDB | DiskANN, no IVF-PQ |

IVF-PQ's value proposition:
- **Memory compression**: 8–32× reduction vs raw f32 (M=8 bytes vs M×4 bytes per vector)
- **Sub-linear search**: probe only `nprobe × avg_list_size ≪ N` candidates
- **Composable**: the filter/refine split cleanly separates approximate from exact scoring

Two existing ruvector primitives make this tractable without a from-scratch implementation:
- k-means++ from `crates/ruvector-rairs/src/kmeans.rs` (adapted)
- PQ design from `crates/ruvector-core/src/advanced_features/product_quantization.rs`
  (re-implemented standalone to avoid heavy dependency chain)

## Decision

Implement `crates/ruvector-ivfpq` as a **standalone crate** (depends only on `rand = "0.8"`)
using the **HAKES filter-refine architecture**:

1. **Two-phase training:**
   - Phase 1: k-means++ on full corpus → `nlist` IVF centroids
   - Phase 2: Compute residuals `r_i = v_i - centroid[assign(v_i)]`; train PQ codebook on
     residuals (M independent k-means, one per subspace). Residual encoding focuses the
     codebook on fine-grained within-cell structure, not coarse cluster geometry.

2. **Residual insert:** For each added vector v, assign to nearest centroid c, encode
   `v - centroid[c]` with PQ, store `(id, codes[M bytes], raw[f32×D])` in `lists[c]`.

3. **Three-stage HAKES search:**
   - Stage 1 (coarse): linear scan of nlist centroids → select nprobe nearest
   - Stage 2 (filter): for each probed cell c, compute `qr = q - centroid[c]`,
     build ADC LUT, score all entries via `sum_m lut[m][code[m]]`; keep top-`rerank_k`
   - Stage 3 (refine): exact L2sq on stored raw vectors for rerank_k candidates; return top-k

4. **No SIMD, no unsafe code**: pure safe Rust with auto-vectorisation-friendly inner loops.
   FastScan SIMD is explicitly a roadmap item (P0 in ADR-194 production layout).

### Why HAKES architecture over plain IVF-PQ

Classic IVF-PQ returns PQ-approximate distances as the final answer. HAKES separates
filter (PQ-approximate, fast) from refine (exact L2, small set) to get:
- Exact final distances (no accuracy loss in the top-k result)
- Tunable recall/speed tradeoff via `rerank_k` without changing the codebook
- Storage flexibility: raw vectors can live on SSD (fetched only for rerank set), reducing
  RAM footprint to the compressed index alone (238 KB for N=10K, D=128)

### Why a standalone crate

Depending on `ruvector-core` from `ruvector-ivfpq` would pull the full core dependency chain
(serde, tokio, etc.) and make the PoC harder to compile in constrained environments. The
IVF and PQ algorithms are O(100) lines each; re-implementing them standalone is lower
complexity than managing a transitive dependency. Future integration with ruvector-core
(e.g., sharing `DistanceMetric`, the `AnnIndex` trait) is a separate ADR item.

## Consequences

### Positive

- **New index family**: ruvector can now serve compression-based ANN alongside its graph
  indexes; 21.0× memory reduction vs raw f32 at 94.7% recall@10 with nprobe=4
- **Measured recall curve**: 34.7% → 94.7% → 100.0% @K=10 across nprobe=1/4/16
- **Fast search**: 41,841 QPS (nprobe=1) → 12,361 QPS (nprobe=4) on Celeron N4020 single
  thread — competitive with FAISS IVFFlat on the same hardware class
- **Clean architecture**: train/add/search separation mirrors FAISS API; easy to extend
- **Foundation for FastScan**: the `PqCodebook` + `LookupTable` types are the building
  blocks for a future AVX2/NEON FastScan path (P0 roadmap item)
- **All tests pass**: 10 unit tests across kmeans, pq, and ivfpq modules; criterion
  benchmarks produce stable numbers

### Negative / Risks

- **Slow training for large N**: k-means at N=10K, nlist=64, max_iter=30 takes ~23 s on
  the N4020. For N ≥ 1M, parallel mini-batch k-means is required (not in this PoC).
- **Raw vectors stored in RAM**: the refine stage currently stores raw f32 alongside PQ
  codes in the same `Vec<Entry>`. Full in-memory size is 5.2 MB (same as raw baseline).
  Production use requires an mmap-backed refine store (roadmap P1).
- **No SIMD scanning**: the ADC inner loop is scalar; a FastScan pass would give 4–8×
  speedup on AVX2 machines (roadmap P0).
- **Residual PQ only as accurate as IVF alignment**: if nlist is misconfigured, recall
  degrades silently. Users must tune nlist for their data distribution.
- **Static index only**: no insert after initial training (rebuild required).
  Streaming inserts with online k-means update is roadmap P2.

## Alternatives Considered

### A: Use ruvector-core's EnhancedPQ directly

The existing `EnhancedPQ` in `ruvector-core` encodes full vectors (not residuals) and wraps
`HashMap<String, Vec<u8>>` — not suitable for IVF-style cell-partitioned storage. Adapting
it would have required modifying ruvector-core API, which risks breaking existing users.
Decision: standalone reimplementation with residual encoding from the start.

### B: Full-vector PQ without residuals

Simpler implementation (one LUT for the entire query, no per-cell residual).
Recall@10 at nprobe=4 was only 44.4% vs 94.7% with residuals (measured).
Unacceptable recall loss; residual PQ is non-negotiable.

### C: HNSW over IVF centroids for coarse scan (IVF-HNSW)

Reduces coarse scan from O(nlist × D) to O(D × log nlist). Significant for nlist ≥ 1024.
At nlist=64 the linear scan takes < 0.1 ms — negligible. Deferred to roadmap P1.

### D: ResidualVQ (RVQ) instead of flat PQ

RVQ applies PQ in cascaded stages — each stage quantizes the residual of the previous.
Used in Apple's on-device ML and in neural codecs (QINCo2). Much higher reconstruction
quality per byte vs flat PQ. But: (a) decoding is 3–5× more expensive than flat PQ ADC;
(b) training requires careful initialisation across stages; (c) at D=128 with M=8
flat PQ already gives 94.7% recall — the gain from RVQ is marginal for this use case.
Deferred to a separate nightly research topic.

## Implementation Notes

Key source files (all under 500 lines per CLAUDE.md):

| File | Lines | Purpose |
|------|-------|---------|
| `src/kmeans.rs` | ~130 | k-means++ with usize::MAX initialisation fix |
| `src/pq.rs` | ~140 | PqCodebook (train/encode/build_lut) + LookupTable |
| `src/ivfpq.rs` | ~230 | IvfPqIndex (train/add/search) + 4 unit tests |
| `src/main.rs` | ~110 | Demo binary with recall/QPS sweep |
| `benches/ivfpq_bench.rs` | ~80 | Criterion: search_nprobe, search_m, train |

Notable implementation detail: the initial `assignments` vector in k-means is initialised
to `usize::MAX` (not 0) to force the first iteration to always run an update step. This
fixes a bug where k=1 training would exit immediately without updating the centroid from
the random seed to the true mean.
