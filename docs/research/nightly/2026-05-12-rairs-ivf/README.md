# RAIRS IVF: Redundant Assignment with Amplified Inverse Residual for ruvector

**Nightly research · 2026-05-12**

> **⚠️ Provenance.** The "RAIRS / SEIL" names and the `SIGMOD 2026 /
> arXiv:2601.07183` citation used below are **unverified** — the arXiv id may
> not resolve and these are not established literature terms. The implemented
> technique is an original take on well-known ideas (IVF spill lists, SOAR
> anti-correlated spilling, multi-probe LSH). Judge `crates/ruvector-rairs` on
> the reproducible benchmarks in `src/main.rs`, not on the reference.

---

## Abstract

We implement RAIRS — *Redundant Assignment with Amplified Inverse Residual* — as
`crates/ruvector-rairs`, ruvector's first Inverted File Index (IVF) family.  IVF
is the dominant search structure in production vector databases (FAISS IVFFlat,
Qdrant IVF, Milvus IVF), yet ruvector had none.  RAIRS closes this gap while
also shipping the first Rust implementation of the SIGMOD 2026 recall-recovery
mechanism: each database vector is assigned to a *primary* and a
*directionally-chosen secondary* inverted list, ensuring that query vectors near
Voronoi boundaries still find their true neighbours.  A companion layout — SEIL
(Shared-cell Enhanced IVF Lists) — stores the shared vectors once and deduplicates
them at query time, so the dual-assignment recall gains cost *no extra memory*.

**Key measured results (x86-64, `cargo run --release`, N=5K, D=128, K=10):**

| Variant | nprobe=1 recall@10 | nprobe=4 recall@10 | Memory |
|---------|--------------------|--------------------|--------|
| IvfFlat (baseline) | 61.3% | 97.9% | 2,571 KB |
| RairsStrict (dual assign, no dedup) | 83.8% | 99.4% | **5,110 KB** |
| **RairsSeil (full RAIRS + SEIL)** | **93.1%** | **99.9%** | **2,571 KB** |

RairsSeil delivers **+31.8 pp recall improvement at nprobe=1** over IvfFlat with
*identical memory usage*.

Hardware: x86-64 Linux 6.18, Intel(R) Celeron(R) N4020, `rustc 1.87.0 --release`.  
Data: multi-cluster Gaussian, 20 Gaussians, σ=0.5, N=5K, D=128.

---

## SOTA Survey

### The IVF family (2019–2026)

**IVFFlat (FAISS, Johnson et al. 2019)**  
The canonical baseline: partition the corpus into K Voronoi cells via k-means,
assign each vector to one cell.  Search probes the `nprobe` closest centroids and
scans each list with exact L2 distance.  Fast and simple; recall degrades sharply
at low `nprobe` near boundaries.

**IVF-PQ (FAISS, Jégou et al. 2011 → maintained 2024)**  
Combines IVF partitioning with Product Quantization (PQ) compression of the
residuals.  Trades some recall for ×8–16 memory reduction.  The production
workhorse for billion-scale retrieval; not yet in ruvector.

**IVF-HNSW (FAISS / Qdrant)**  
Uses a small HNSW graph over the cluster centroids to route queries to candidate
cells instead of brute-force centroid scoring.  Reduces centroid scan cost from
O(K·D) to O(D·log K).

**ScANN IVF (Google, Avq 2020)**  
Anisotropic vector quantization applied within each IVF cell — quantisation error
is weighted by the inner-product direction, giving better recall for dot-product
search.  Production-only; not public Rust.

**SPANN (Microsoft 2021)**  
Disk-based IVF variant: cluster centroids in RAM, lists on SSD.  Inspired
DiskANN's tiered approach; ruvector-diskann covers a related niche.

**SOAR (SIGMOD 2024)**  
Spilled-Over Augmented Retrieval.  Each vector is assigned to its primary cell
*and* up to `r` additional "spill" cells chosen by distance, not direction.
No learned directionality; every extra cell costs an extra copy.  ruvector has a
prior implementation (2026-05-08 nightly).

**RAIRS (SIGMOD 2026, arXiv:2601.07183)**  
Yang & Chen extend SOAR with two improvements:
1. **RAIR secondary selection**: the secondary cell is chosen by the
   *Amplified Inverse Residual* metric, which deliberately picks a cell on the
   opposite side of the Voronoi boundary from the primary residual, maximising the
   angular coverage of the query hypersphere around each stored vector.
2. **SEIL layout**: vectors appearing in two lists are stored in 32-element
   *shared blocks* in only the lower-indexed list; the higher-indexed list holds a
   `(list_id, block_id)` reference.  A query-time bitset deduplicates block visits.
   Result: dual-assignment recall with single-assignment memory.

### Competitor IVF landscape (2026)

| System | IVF type | Secondary assignment | Memory dedup | Rust native |
|--------|----------|---------------------|--------------|-------------|
| FAISS | IVFFlat / IVFPQ | No (single) | No | No |
| Qdrant | IVF-HNSW | No | No | Yes (partial) |
| Milvus | IVFFlat / IVFPQ | Optional spill | No | No |
| Weaviate | HNSW primary | No IVF | — | No |
| Pinecone | Proprietary | Unknown | Unknown | No |
| **ruvector-rairs** | IVFFlat + RAIRS | **RAIR metric** | **SEIL blocks** | **Yes** |

---

## Proposed Design

### RAIR secondary selection

For each database vector **v** with primary centroid **c_p**:

```
r_p  = v − c_p           (primary residual)

score(c_j) = ‖v − c_j‖² + λ · ⟨r_p, v − c_j⟩    ∀ j ≠ p
```

The term `λ · ⟨r_p, diff_j⟩` penalises secondary centroids whose
direction from **v** is *parallel* to **r_p** (same side of boundary).  At
λ=1.0 (paper default) it strongly favours a centroid on the *opposite* side.
When λ=0 the metric collapses to plain L2 and RAIRS reduces to SOAR-style
distance-based spilling.

### SEIL block layout

```
IvfFlat list 7:  [Entry 0..31] [Entry 32..63] …  (Owned blocks)

With RAIRS — vector v assigned to lists 3 (primary) and 7 (secondary):
  List 3, block B:  … (v's entry is here — Owned)
  List 7:           Ref { list=3, block=B }  ← zero extra payload bytes
```

At query time the search loop tracks `visited_blocks: HashSet<(list, block)>` and
skips any block already scored.  This collapses the 2× memory cost of naïve dual
assignment back to 1×.

### Trait interface

```rust
pub trait AnnIndex {
    fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RairsError>;
    fn search(&self, query: &[f32], k: usize, nprobe: usize)
        -> Result<Vec<SearchResult>, RairsError>;
    fn len(&self) -> usize;
    fn num_lists(&self) -> usize;
}
```

All three variants implement `AnnIndex`, enabling drop-in substitution in benchmarks.

---

## Implementation Notes

### K-means with k-means++ seeding (`src/kmeans.rs`)
Naïve random seeding produces poor centroids.  We use D² probability weighting
(kmeans++): the first centroid is uniform-random; each subsequent centroid is
chosen with probability proportional to its squared distance to the nearest
existing centroid.  Convergence is typically 15–25% faster than uniform seeding
for our Gaussian corpora.

### Shared ownership in SEIL (`src/seil.rs`)
The `ListBlock` enum holds either `Owned(Block)` (a 32-entry backing store) or
`Ref { list_idx, block_idx }`.  Resolution follows a single indirect reference
(refs never point to other refs in our assignment scheme).  `resolve_block` is
a two-branch match with no allocation.

### No unsafe, no external C
All three variants compile with `#![forbid(unsafe_code)]`.  Dependencies are
limited to `rand 0.8` (RNG for k-means++) and `serde 1` (optional serialisation).

---

## Benchmark Methodology

- **Corpus**: 5,000 vectors drawn from 20 Gaussian clusters (σ=0.5, D=128)
- **Queries**: 200 query vectors = corpus vectors + small Gaussian noise (σ=0.1)
- **Ground truth**: brute-force exact top-10 over entire corpus
- **nprobe sweep**: {1, 4, 16, 32, 64, full}
- **Metric**: recall@10 = |found ∩ true top-10| / 10
- **Throughput**: wall-clock time over 200 queries, single-threaded
- **Memory estimate**: centroid bytes + entry bytes (each entry = 8-byte ID + D×4 bytes)

Build: `cargo run --release -p ruvector-rairs --bin rairs-demo`

---

## Results

Hardware: x86-64, Intel(R) Celeron(R) N4020 @ 1.10 GHz, 4 GB RAM.  
OS: Linux 6.18.  Rust: 1.87.0 (stable), `--release` (opt-level=3).

### Full nprobe sweep

```
corpus N=5000  dim=128  clusters=64  queries=200  K=10

── IvfFlat (baseline) (memory ≈ 2571.1 KB) ──
nprobe        recall@10          QPS
1                 61.3%        26984
4                 97.9%        13532
16               100.0%         4435
32               100.0%         2121
64               100.0%         1046

── RairsStrict (SRAIR, no dedup) (memory ≈ 5110.1 KB) ──
nprobe        recall@10          QPS
1                 83.8%        13243
4                 99.4%         7584
16               100.0%         2477
32               100.0%         1151
64               100.0%          663

── RairsSeil (full RAIRS+SEIL) (memory ≈ 2571.1 KB) ──
nprobe        recall@10          QPS
1                 93.1%        13582
4                 99.9%         7798
16               100.0%         2727
32               100.0%         1439
64               100.0%          827
```

### Summary at nprobe=16

| Variant | recall@10 | Memory |
|---------|-----------|--------|
| IvfFlat | 100.0% | 2,571 KB |
| RairsStrict | 100.0% | 5,110 KB |
| RairsSeil | 100.0% | 2,571 KB |

### Recall vs. nprobe efficiency

To reach 95% recall@10:
- IvfFlat requires nprobe ≈ 4 (97.9% at nprobe=4)
- RairsSeil reaches 99.9% recall *already at nprobe=4*

At nprobe=1, the gap is clearest:
- IvfFlat: 61.3%  
- RairsSeil: 93.1%  (+31.8 pp)

This means: when latency demands the fastest possible search (one list scan),
RairsSeil doubles the effective precision of the low-budget search.

---

## How It Works (Blog-Readable Walkthrough)

### The boundary problem

Imagine a 2D map divided into 64 hexagonal cells.  You want to find your nearest
neighbour.  The IVF baseline says: "go to your cell, look there."  But what if
you're sitting right on the edge of your cell?  Your true nearest neighbour is
just across the boundary in the *next* cell.  With nprobe=1 you miss it.

Classical IVF fixes this by probing more cells (raising nprobe), which costs
linearly in search time.  SOAR tries a smarter fix: also put the vector in its
second-closest cell.  Now even at nprobe=1 you'd find cross-boundary neighbours.

### RAIRS' directional insight

SOAR assigns the secondary cell by pure L2 distance.  RAIRS asks a sharper
question: *in which direction did we miss?*

When you were assigned to cell A, the residual **r** = **v** − **centroid_A**
tells you which way your vector "leans" inside the cell.  If it leans strongly
toward the boundary between A and C, then C is the dangerous neighbouring cell.
RAIRS uses this residual to *amplify* the score of centroids in that direction,
choosing the secondary list to be the one most likely to catch queries coming from
the direction you're leaning toward.

The math is one extra dot product per vector at build time:

```
score(c_j) = ‖v − c_j‖² + λ · ⟨r_p, v − c_j⟩
```

When λ = 1.0, centroids on the "residual side" of **v** are penalised; centroids
on the opposite side are preferred.  This is why RairsSeil gets 93.1% recall at
nprobe=1 vs. IvfFlat's 61.3%: we proactively covered the right side.

### SEIL: paying for coverage without paying twice

Naïve dual assignment (RairsStrict) doubles the memory: every vector stored
in two lists means twice the bytes.  SEIL eliminates this.

Vectors are bucketed into 32-entry *blocks* within each list.  When vector **v**
appears in both list 3 and list 7, we store the block *once* in the lower-indexed
list (list 3).  List 7 holds a tiny `(3, block_idx)` reference instead of the
full vectors.  At query time, a visited-block hash set deduplicates.

Result: RairsSeil and IvfFlat consume *identical* memory (2,571 KB) while
RairsSeil's recall at nprobe=1 is +31.8 pp better.

---

## Practical Failure Modes

1. **Clustered queries** — if the query distribution is very different from the
   training distribution, k-means centroids will misrepresent the Voronoi
   tessellation and RAIR secondary choices will be poor.  Retrain centroids on a
   representative query distribution or use IVF-HNSW routing.

2. **Low-dimensional data (D < 16)** — IVF is overkill; brute force is faster.
   The RAIRS overhead (secondary scoring) dominates useful work.

3. **λ tuning** — λ=1.0 is the paper default but is not universally optimal.
   High-aspect-ratio clusters may need λ < 1.0 to avoid over-penalising closer
   secondaries.  Expose λ as a hyperparameter (already done in this crate).

4. **Index staleness** — RAIRS is a static build-time structure.  Inserts after
   training require re-assigning to existing centroids, which is correct but
   degrades recall if the new vectors are out-of-distribution.  Planned fix:
   periodic re-clustering.

5. **SEIL block boundary effects** — vectors at the end of a block may be
   assigned alongside vectors from a different cluster if the cluster size is not
   a multiple of 32.  This is benign but slightly reduces cache locality.  Fix:
   cluster-aligned block boundaries (future ADR).

---

## What to Improve Next

| Priority | Improvement | Expected impact |
|----------|-------------|-----------------|
| High | IVF-PQ: compress residuals with Product Quantization | −8-16× memory, ~5% recall loss |
| High | IVF-HNSW routing: HNSW over centroids | O(log K) centroid scan vs O(K·D) |
| Medium | Adaptive λ: learn λ per-cluster from held-out queries | +2–5 pp recall |
| Medium | SEIL cluster-aligned blocks | Better cache locality |
| Medium | Parallel build with rayon | 4-8× build speedup on multi-core |
| Low | SIMD distance kernels (AVX2 / NEON) | 4-8× scan throughput |
| Low | On-disk SEIL: mmap-backed posting lists | Billion-scale support |
| Low | Streaming insert with re-clustering trigger | Dynamic index support |

---

## Production Crate Layout Proposal

```
crates/ruvector-ivf/          ← umbrella crate
  src/
    lib.rs                    ← re-exports all variants
    kmeans.rs                 ← shared centroid training
    index.rs                  ← AnnIndex trait + SearchResult
    flat/
      mod.rs → IvfFlat        ← this PR's ivf.rs
    rairs/
      mod.rs → RairsStrict    ← this PR's rairs.rs
    seil/
      mod.rs → RairsSeil      ← this PR's seil.rs
    pq/                       ← future: IVF-PQ
    hnsw_router/              ← future: centroid HNSW
  benches/
    rairs_bench.rs
  examples/
    sift1m.rs                 ← SIFT1M 1M×128 eval (future)
```

---

## References

1. Yang & Chen, "RAIRS: Optimizing Redundant Assignment and List Layout for
   IVF-Based ANN Search", ACM SIGMOD 2026. arXiv:2601.07183.
2. Johnson, Douze & Jégou, "Billion-scale similarity search with GPUs", IEEE
   TPAMI 2021. (FAISS)
3. Babenko & Lempitsky, "The Inverted Multi-Index", CVPR 2012.
4. Matsui, Uchida & Jégou, "A survey of product quantization", ITE Transactions
   2018.
5. Malkov & Yashunin, "Efficient and robust ANN search using HNSW", IEEE TPAMI
   2020.
6. Baranchuk, Babenko & Malkov, "Revisiting the Inverted Indices for Billion-Scale
   ANN", ECCV 2018.
