---
adr: 193
title: "Add SOAR-IVF: partition-based ANN with orthogonality-amplified residual spilling"
status: accepted
date: 2026-05-08
authors: [ruvnet, claude-flow]
related: []
tags: [ivf, ann, quantization, soar, nightly-research, product-quantization, nearest-neighbor]
---

# ADR-193 — SOAR-IVF: Inverted File Index with Orthogonality-Amplified Residual Spilling

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-08-soar-ivf` as
`crates/ruvector-soar`. See `docs/research/nightly/2026-05-08-soar-ivf/README.md`
for SOTA survey, algorithm walkthrough, and benchmark numbers.

## Context

Every existing ruvector index is **graph-based**:

| Crate | Algorithm | Build cost | Best at |
|-------|-----------|------------|---------|
| `ruvector-core` | HNSW | O(n log n) | Balanced recall/QPS |
| `ruvector-diskann` | DiskANN/Vamana | O(n log n) | Billion-scale SSD |
| `ruvector-acorn` | ACORN (filtered HNSW) | O(n²) PoC | Low-selectivity filtering |
| `ruvector-hyperbolic-hnsw` | Hyperbolic HNSW | O(n log n) | Hierarchical data |

**No partition-based (IVF) index** exists in the workspace. IVF complements
graph-based indices in several scenarios:
- **Memory budget is tight**: IVF-PQ compresses to M bytes per vector (M=8 for
  D=128 gives 16× vs flat f32).
- **Batch workloads**: IVF centroid lookup is cache-friendly and SIMD-vectorisable
  at scale.
- **Production index rebuild**: k-means is parallelisable and deterministic;
  graph indices have random elements that complicate reproducible builds.

The IVF boundary problem — boundary vectors missing from searches at low nprobe
— is addressed by SOAR (Sun et al., NeurIPS 2023), which won the Big-ANN
Benchmarks 2023 OOD and streaming tracks and is deployed in Google Cloud Vertex
AI Vector Search.

**Gap**: No Rust implementation of SOAR existed on crates.io or GitHub prior
to this ADR.

## Decision

Introduce `crates/ruvector-soar` implementing three index variants under a single
`SoarIndex` struct governed by `IndexKind`:

| Variant | Description |
|---------|-------------|
| `IndexKind::Flat` | Brute-force exact scan (always-recall baseline) |
| `IndexKind::IvfPq` | IVF k-means partitioning + product quantization (ADC) |
| `IndexKind::SoarIvfPq` | Above + SOAR secondary assignment via orthogonality-amplified residual loss |

**SOAR secondary assignment rule** for vector `v` with primary centroid `c`:

```
L(c') = ‖v − c'‖² + λ · [ (v−c) · (v−c') ]² / ‖v−c‖²
```

The secondary centroid is `argmin_{c' ≠ c} L(c')` over the `n_secondary_candidates`
nearest centroids. This penalises secondary residuals that are parallel to the
primary residual, guaranteeing that the secondary centroid's "blind direction"
is orthogonal to the primary's blind direction.

**File structure** (all files < 500 lines):

```
crates/ruvector-soar/
  Cargo.toml
  src/lib.rs       —  public API + 5 unit tests
  src/error.rs     —  SoarError enum
  src/kmeans.rs    —  k-means++, Lloyd iterations, top-k centroid query
  src/pq.rs        —  ProductQuantizer, encode, distance_table, adc_distance
  src/index.rs     —  SoarIndex::build, SoarIndex::search, soar_secondary_assign
  src/main.rs      —  benchmark harness with 3 variants × 3 nprobe settings
  benches/soar_bench.rs — Criterion micro-benchmarks
```

## Consequences

### Positive

- **First IVF-based index in ruvector**: fills a structural gap; enables
  memory-budget-constrained deployments not well served by graph indices.
- **SOAR recall advantage at low nprobe**: +10.4pp recall@10 at nprobe=1 on
  2K/D=64 benchmark; +1.8pp at nprobe=2 on 10K/D=128.
- **Trait-based design**: swapping Flat → IvfPq → SoarIvfPq requires one field
  change in `SoarConfig`; no code duplication.
- **Zero external dependencies beyond workspace**: only `rand`, `rand_distr`,
  `thiserror`, `serde`, `rayon`.
- **All 5 unit tests pass**: `cargo test -p ruvector-soar` green.
- **`cargo build --release -p ruvector-soar` succeeds** with zero errors.

### Negative / Trade-offs

- **17% memory overhead** of secondary lists vs plain IVF-PQ.
- **SOAR QPS ~20–28% lower** than IVF-PQ at same nprobe due to secondary list
  scanning. Net result: at equal recall target, QPS is similar; SOAR earns its
  memory overhead by needing lower nprobe for the same recall.
- **Build time dominated by k-means**: Lloyd iterations O(n × nlist × D × iter).
  For n=10K, D=128, nlist=64: ~4.2 s single-threaded. Acceptable for PoC;
  must be parallelised via rayon before production use at n > 1M.
- **Recall ceiling from PQ**: at nprobe ≥ 8 on 10K corpus, both IVF-PQ and
  SOAR-IVF-PQ plateau at ~46% recall. Root cause: M=16 subspaces × 20 training
  iterations is under-trained for 10K vectors at D=128. Residual reranking
  (future work) removes this ceiling.

### Neutral

- Crate is workspace-local only; not published to crates.io in this PR.
- No WASM or Node.js bindings in this PR (`wasm32` falls through to sequential
  path via `cfg(not(target_arch = "wasm32"))` on rayon dep).

## Alternatives Considered

### A: Standard IVF-PQ without secondary spilling

Implement only `IndexKind::IvfPq` without SOAR. Simpler but misses the recall
gain at low nprobe that motivates the new crate. Since SOAR adds ~50 lines of
code to IVF-PQ, the marginal complexity is low.

### B: SeRF (SIGMOD 2024)

Segment graph for range-filtering ANNS. High value for range queries; however
the 2D segment graph structure has O(n log n) index size and partially overlaps
with `ruvector-acorn`'s filtered search story. Deferred.

### C: GleanVec (arXiv 2410.22347)

Piecewise linear dimensionality reduction per cluster. Requires SVD per cluster
(ndarray-linalg/LAPACK linkage). Deferred to avoid C-library dependencies in
what is otherwise a pure-Rust crate.

### D: MUVERA (NeurIPS 2024)

Multi-vector FDE encoding for ColBERT-style retrieval. Already shipped in
Weaviate 1.31 (2025). Deferred; lower marginal differentiation.

## References

- Sun et al. "SOAR: Improved Indexing for Approximate Nearest Neighbor Search."
  NeurIPS 2023. arXiv:2404.00774.
- Jégou et al. "Product quantization for nearest neighbor search." TPAMI 2011.
- Johnson et al. "Billion-scale similarity search with GPUs." IEEE Trans. Big
  Data 2019.
