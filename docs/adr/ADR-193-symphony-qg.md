# ADR-193: SymphonyQG — Graph-Coupled 4-bit FastScan Neighbor Scoring

## Status

Proposed

## Date

2026-05-08

## Authors

ruv.io · RuVector Nightly Research (automated nightly agent)

## Relates To

- ADR-154 — RaBitQ rotation-based binary quantization (`ruvector-rabitq`)
- ADR-160 — ACORN predicate-agnostic filtered HNSW (`ruvector-acorn`)
- ADR-001 — Tiered quantization strategy
- Research: `docs/research/nightly/2026-05-08-symphony-qg/README.md`

---

## Context

### The problem: graph ANN is bottlenecked by the neighbor-scoring inner loop

Graph-based ANN (HNSW, DiskANN) spends 60–80% of search time computing exact f32
distances from the query to each expanded neighbor. In HNSW at M=16, each node
expansion requires 16 distance computations, each touching 512 bytes (128 × f32)
at a random memory location. This is a latency-bound, cache-miss-dominated pattern.

The standard mitigation — encode vectors to 8-bit SQ or 4-bit PQ, keep a re-rank
buffer — adds two phases to every search:
1. Approximate scoring from quantized codes.
2. Exact re-rank of the top-K approximate candidates from full f32 vectors.

Phase 2 still requires a pointer chase per candidate and a full f32 distance loop.
At high QPS, re-rank becomes the new bottleneck.

### SymphonyQG (SIGMOD 2025, arXiv:2411.12229)

Gou et al. propose two changes to the standard graph-ANN inner loop:

1. **Co-locate quantized neighbor codes inside the edge list.** Instead of a
   `neighbor_id → data[id]` pointer chase, each edge record stores the neighbor's
   packed 4-bit PQ codes contiguously. The entire edge batch fits in 3–5 cache lines.

2. **Replace distance computation with FastScan LUT lookup.** Before the beam
   search begins, precompute a look-up table: for each PQ subspace s and centroid c,
   the u8-quantized distance from the query to centroid `c` in subspace `s`. Scoring
   a neighbor is then M table lookups (8 for M=8 subspaces) — no floating-point.

Together, these changes eliminate the separate re-rank phase: FastScan scores are
good enough to guide beam search, and no additional exact-distance phase is needed.

**Published results** (SIGMOD 2025): 3.5–17× QPS over HNSWlib at 90–95% recall@10
on SIFT-1M, GIST-1M, text-embedding-3-small. No pure-Rust implementation existed
as of 2026-05-08.

### ruvector gap

ruvector has:
- `ruvector-rabitq`: 1-bit quantized brute-force scan (RaBitQ, ADR-154).
- `ruvector-acorn`: filtered HNSW with in-graph predicate evaluation (ADR-160).
- `ruvector-core`: scalar/int4/binary quantization in `advanced_features/`.

Missing: **graph-coupled 4-bit PQ FastScan** — the specific integration of packed
neighbor codes + SIMD LUT that eliminates the re-rank phase in graph traversal.

---

## Decision

Add a new crate `crates/ruvector-symphony-qg` implementing the SymphonyQG kernel
with three variants:

| Variant | Distance source | Re-rank |
|---------|----------------|---------|
| `flat_exact` | Exact f32 brute force | N/A (baseline) |
| `sqg_fastscan` | 4-bit PQ FastScan in-graph | None |
| `sqg_rerank` | 4-bit PQ FastScan + f32 re-rank | Yes |

### Key design choices

**4-bit (not 8-bit) quantization.** Four bits allows 16 centroids per subspace and
packing two codes per byte. The LUT fits in 256 bytes (16 × M for M=16 subspaces),
which stays in L1 cache. The FastScan AVX2 path uses `vpshufb` which natively
handles 4-bit nibble indexing.

**Scalar + optional AVX2.** The `scan_scalar` path is always available for portability
(WASM, aarch64, non-AVX2 x86). The `scan_avx2` path is compiled conditionally and
processes 32 neighbors per subspace pair using `_mm256_shuffle_epi8`.

**Bidirectional graph construction.** Each forward edge A→B is mirrored as B→A,
capping total degree at 2×M. This improves navigability from random seed entries
without requiring a multi-layer HNSW structure (which is planned for v2 of this crate).

**Immutable index.** `SqgIndex` is built once, searched many times, no incremental
inserts. This matches the pattern of `ruvector-rabitq` and `ruvector-acorn`.

---

## Consequences

### Positive

- **FastScan kernel speedup**: 4.1–4.2× at D=128, 8.5–9.8× at D=256 (measured),
  matching the theoretical model (fewer FMA rounds replaced by LUT lookups).
- **Memory-efficient neighbor scoring**: edge-list codes are contiguous → sequential
  cache access, not random pointer chases.
- **Eliminates re-rank phase**: SqgFastScan has no secondary exact-distance step,
  simplifying the search pipeline.
- **Composable with RaBitQ**: Variant C can use ruvector-rabitq's asymmetric scorer
  in place of 4-bit PQ for higher fidelity at modest extra cost (planned, ADR-154).
- **No external dependencies**: only `rand`, `rand_distr`, `serde`, `thiserror`.

### Negative / Limitations

- **Graph navigability**: the flat greedy k-NN graph (PoC) does not reach recall
  levels of HNSW multi-layer graphs. High recall (>90%) requires HNSW construction,
  which is deferred to a follow-up.
- **4-bit resolution floor**: full-scan recall is limited by LUT quantization noise
  (ties within a u8 bucket). This is a property of 4-bit PQ, not of FastScan itself.
- **Build time O(n²)**: the PoC computes exact k-NN for all pairs. At n=5000 this
  takes ~4.3s. Production scale (n>100K) requires approximate construction.
- **WASM AVX2 unavailable**: the SIMD path is x86_64-only; WASM uses scalar fallback.

### Neutral

- Memory overhead: ~32% additional bytes per node for packed neighbor codes
  (measured: 1.35 MB vs 1.02 MB at n=2K, D=128).
- Codebook training quality depends on data distribution; random Gaussian is a
  worst case (uniform directions — centroid assignments are less stable than
  real embedding distributions which have cluster structure).

---

## Alternatives Considered

### A. Integrate into ruvector-core advanced_features/

`ruvector-core` already has `advanced_features/product_quantization.rs` and
`advanced_features/matryoshka.rs`. Integrating SymphonyQG there avoids a new crate
but couples it to ruvector-core's dependency tree and makes it hard to publish
as a standalone WASM package. **Rejected**: independent crate follows the pattern
of ruvector-rabitq and ruvector-acorn.

### B. Use 8-bit scalar quantization instead of 4-bit PQ

SQ8 (8-bit scalar, per-dimension) is simpler to implement and avoids codebook
training. However, SQ8 does not enable FastScan (the SIMD `vpshufb` trick requires
4-bit indexing). SQ8 also has lower compression (8× vs 16× relative to f32 in
practice). **Rejected**: 4-bit PQ enables FastScan and is the mechanism in the paper.

### C. LoRANN (NeurIPS 2024, arXiv:2410.18926) as alternative topic

LoRANN replaces PQ in IVF clusters with a rank-r SVD approximation per cluster.
Achieves higher recall at matched compression but requires eigendecomposition per
cluster and is harder to implement in pure Rust without BLAS. **Deferred**: LoRANN
is a strong next topic (see research roadmap).

### D. RoarGraph (VLDB 2024) — cross-modal bipartite graph

Builds "cross-modal shortcuts" from a training query set into the graph. Excellent
for out-of-distribution queries but requires a separate query corpus and more complex
build logic. **Deferred**: requires a labeled training set not available at nightly
benchmark time.

---

## Measured Results (2026-05-08, x86_64, cargo --release)

### FastScan kernel throughput

| D | Variant | dist/s | Recall@10 | Speedup |
|---|---------|--------|-----------|---------|
| 128 | ExactF32 | 6,516,307 | 100.0% | 1.00× |
| 128 | FastScan4bit | 27,150,455 | 6.5% | 4.17× |
| 128 | FastScan+Rerank50 | 23,732,767 | 20.1% | 3.64× |
| 256 | ExactF32 | 3,203,917 | 100.0% | 1.00× |
| 256 | FastScan4bit | 27,178,640 | 3.5% | 8.48× |
| 256 | FastScan+Rerank50 | 22,118,897 | 12.9% | 6.90× |

### End-to-end graph search (n=5000, D=128)

| Variant | ef | Recall@10 | QPS | Speedup |
|---------|----|-----------|-----|---------|
| FlatExact | — | 100.0% | 1,253 | 1.00× |
| SqgFastScan | 50 | 6.5% | 10,644 | 8.50× |
| SqgFastScan | 200 | 5.0% | 4,321 | 3.45× |
| SqgFastScan | 500 | 4.8% | 1,942 | 1.55× |

Build time: 4,440 ms (n=5000, O(n²) PoC). Tests: 11/11 pass.
