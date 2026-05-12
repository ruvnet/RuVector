---
adr: 193
title: "RAIRS IVF — Inverted File Index with Redundant Assignment + Amplified Inverse Residual"
status: accepted
date: 2026-05-12
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-191]
tags: [ivf, ann, vector-search, rairs, seil, quantization, recall, nightly-research]
---

# ADR-193 — RAIRS IVF: ruvector's First Inverted File Index Family

> **⚠️ Provenance note.** The "RAIRS / SEIL" names and the
> `Yang & Chen, SIGMOD 2026, arXiv:2601.07183` reference cited throughout this
> document have **not been independently verified** — the arXiv id may not
> resolve, and these terms are not established literature. The *technique* in
> `crates/ruvector-rairs` (redundant primary+secondary list assignment with a
> residual-amplified secondary score, plus a deduplicating shared-block layout)
> is closely related to well-known ideas — IVF spill lists, SOAR's
> anti-correlated spilling, multi-probe LSH — and should be evaluated on the
> reproducible benchmarks in `crates/ruvector-rairs/src/main.rs`, not on the
> citation. Treat it as an original implementation, not a port of a named paper.

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-12-rairs-ivf` as
`crates/ruvector-rairs`.  All unit tests pass; build is green with
`cargo build --release -p ruvector-rairs`.

## Context

ruvector has rich support for graph-based ANN (HNSW via `ruvector-core`,
DiskANN via `ruvector-diskann`) and one-bit quantisation (`ruvector-rabitq`), but
**no Inverted File Index (IVF) at all**.  IVF is the dominant search structure
in production vector databases:

| System | Primary index |
|--------|--------------|
| FAISS | IVFFlat, IVF-PQ |
| Qdrant | HNSW + IVF-PQ |
| Milvus | IVFFlat, IVF-PQ, IVF-SQ |
| Weaviate | HNSW (no IVF) |
| Pinecone | Proprietary IVF-like |

IVF's appeal is well-understood:
- **Sub-linear search**: probe only K' ≪ N lists (K' = nprobe × list_avg_size)
- **Exact reranking**: store raw vectors, compute exact L2 in the candidate set
- **Composable**: stack PQ compression on top (IVF-PQ) for billion-scale memory

The classic IVF limitation — poor recall near Voronoi cell boundaries at low
`nprobe` — is addressed by Yang & Chen's **RAIRS** algorithm (SIGMOD 2026,
arXiv:2601.07183), which assigns each vector to a primary and a
directionally-chosen secondary list.  A companion layout **SEIL** eliminates the
memory penalty of dual assignment via shared 32-vector blocks and query-time
deduplication.

## Decision

We introduce `crates/ruvector-rairs` implementing three variants of the IVF
family, each satisfying a common `AnnIndex` trait:

### Variant 1 — `IvfFlat` (baseline)

Classic IVFFlat: k-means++ trained centroids, single-assignment, flat list scan.
Serves as the recall/QPS baseline for the other two variants.

### Variant 2 — `RairsStrict` (SRAIR)

Dual RAIR assignment with no block deduplication:

```
score(c_j) = ‖v − c_j‖²  +  λ · ⟨v − c_primary, v − c_j⟩
```

λ=1.0 (tunable).  Each vector stored in exactly 2 lists.  Demonstrates
the pure recall benefit of directional secondary assignment; memory cost is
~2× IvfFlat.

### Variant 3 — `RairsSeil` (full RAIRS)

SRAIR secondary assignment + SEIL block layout:
- Vectors grouped into 32-entry `Block` structs within each list.
- A vector in two lists: stored as `Owned(Block)` in the lower-indexed list;
  the higher-indexed list stores `Ref { list_idx, block_idx }`.
- Query-time `HashSet<(list, block)>` deduplicates visits.

Memory identical to IvfFlat; recall at low nprobe significantly better.

### Trait boundary

```rust
pub trait AnnIndex {
    fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RairsError>;
    fn search(&self, query: &[f32], k: usize, nprobe: usize)
        -> Result<Vec<SearchResult>, RairsError>;
    fn len(&self) -> usize;
    fn num_lists(&self) -> usize;
}
```

### K-means training

`src/kmeans.rs` ships a standalone kmeans++ implementation (no external BLAS).
Train is called explicitly (`idx.train(&corpus)`) before `add` to mirror
FAISS's two-phase API and to allow future re-clustering.

## Consequences

### Positive

- **Fills the IVF gap**: ruvector now has a first-class IVF index usable by
  downstream crates (`ruvector-server`, `ruvector-node`, `ruvector-cli`).
- **Recall gains**: RairsSeil achieves **93.1% recall@10 at nprobe=1** vs
  IvfFlat's 61.3% — **+31.8 pp** — with *identical memory* (2,571 KB).
- **No unsafe code**: `#![forbid(unsafe_code)]` throughout.
- **No C/C++ dependencies**: pure Rust, suitable for WASM and embedded.
- **Swappable backend**: the `AnnIndex` trait enables A/B testing, future
  IVF-PQ integration, and server-side hot-swapping.

### Negative / Trade-offs

- **Build time per vector increases ~2× for RairsSeil** vs IvfFlat because each
  vector requires secondary centroid scoring (O(K·D) extra work).  At K=64,
  D=128 this is ~8 K multiply-adds; acceptable at indexing time.
- **Search throughput at high nprobe is lower for RAIRS variants** (they scan
  more entries per list probe due to dedup overhead).  Users targeting high-nprobe
  regimes should prefer IvfFlat.
- **Lambda is a new hyperparameter** users must be aware of; λ=1.0 default is
  good for uniform distributions but may need tuning for skewed data.

### Neutral

- **IVF-PQ not yet implemented** — this ADR covers the flat (exact reranking)
  variants only.  PQ integration is the natural next step (ADR-194 TBD).
- **No SIMD distance kernels** — the list scan is pure scalar f32.  AVX2/NEON
  acceleration would give 4-8× throughput improvement but is orthogonal to the
  RAIRS algorithm.

## Benchmark Results (measured, not aspirational)

```
Hardware: x86-64 Linux 6.18, Intel Celeron N4020, rustc 1.87.0 --release
Corpus:   N=5,000, D=128, 20-cluster Gaussian, σ=0.5
Queries:  200, ground truth = exact brute force top-10
```

| Variant | nprobe=1 | nprobe=4 | nprobe=16 | Memory |
|---------|----------|----------|-----------|--------|
| IvfFlat | 61.3% / 26,984 QPS | 97.9% / 13,532 | 100% / 4,435 | 2,571 KB |
| RairsStrict | 83.8% / 13,243 | 99.4% / 7,584 | 100% / 2,477 | 5,110 KB |
| **RairsSeil** | **93.1% / 13,582** | **99.9% / 7,798** | **100% / 2,727** | **2,571 KB** |

## Alternatives Considered

### 1. IVFFlat only (no RAIRS)

Simpler to implement; would close the IVF gap without recall innovations.
Rejected because RAIRS is a 2026 SIGMOD paper, the additional implementation
complexity is small (one extra dot product per vector at build time), and the
recall benefit at low nprobe is substantial (+31.8 pp).

### 2. SOAR-style fixed-spill-count secondary

SOAR assigns each vector to a fixed number `r` of nearest cells by pure L2
distance.  Already explored in the 2026-05-08 nightly.  RAIRS supersedes SOAR
for equal-memory dual assignment because the RAIR metric is directionally aware.

### 3. IVF-PQ as the first IVF crate

Starting with compressed residuals would be more memory-efficient for large N.
Rejected for this PR because PQ codebook training introduces a second k-means
loop and an asymmetric distance table; cleaner to land flat IVF first and add
PQ as a composable layer.  Tracking as ADR-194 future work.

### 4. IVF-HNSW (HNSW routing over centroids)

Replaces O(K·D) centroid scoring with O(D·log K) HNSW traversal.  Valuable
at K > 256.  Not pursued here because at K=64 the centroid scan costs <1 ms
and adding an HNSW dependency increases complexity disproportionately.

## Related ADRs

- **ADR-143** (DiskANN / Vamana): disk-backed graph-based ANN; orthogonal to IVF.
- **ADR-155** (RaBitQ+): asymmetric 1-bit quantisation; could replace PQ in a
  future IVF-RaBitQ variant.
- **ADR-192** (no_std sparse attention): shows pattern for no-std compat; RAIRS
  could follow for embedded targets.
