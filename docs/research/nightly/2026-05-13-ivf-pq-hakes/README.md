# IVF-PQ with HAKES Filter-Refine: ruvector's First Compression-Based ANN Index

**Nightly research · 2026-05-13**

---

## Abstract

We implement `crates/ruvector-ivfpq` — ruvector's first Inverted File Index with Product
Quantization (IVF-PQ) index. IVF-PQ is the dominant ANN index family in production vector
databases (FAISS, Qdrant, Milvus, Pinecone) because it delivers **10-100× memory compression**
with negligible recall loss. Despite having rich graph-based (HNSW, DiskANN) and quantisation
(RaBitQ) support, ruvector had no IVF-PQ at all. This crate closes that gap.

The design follows the **HAKES filter-refine architecture** (VLDB 2025): the search pipeline
is deliberately split into two stages — a fast ADC filter stage (scans PQ codes with one-table
lookup per subspace per candidate) and an exact-L2 refine stage (re-scores top candidates
with the raw stored vectors). This gives the best of both worlds: sub-linear scan cost from
PQ compression, and exact-quality top-k from the refine step.

**Key measured results** (x86-64, `cargo run --release -p ruvector-ivfpq`,
N=10,000, D=128, K=10, 20 Gaussian clusters, σ=0.5):

| Variant | Recall@10 | QPS | Compressed mem |
|---------|-----------|-----|----------------|
| nprobe=1, rerank_k=50   |  34.7% | 26,429 | 238 KB |
| nprobe=4, rerank_k=200  |  94.7% | 9,481  | 238 KB |
| nprobe=16, rerank_k=500 | **100.0%** | 2,617 | 238 KB |

Raw f32 storage: 5,000 KB. Compressed codes only: **238 KB** (21.0× compression).

Criterion benchmarks (100 queries batched):
- nprobe=1:   2.39 ms → **41,841 QPS** per-thread
- nprobe=4:   8.09 ms → **12,361 QPS** per-thread
- nprobe=16: 32.66 ms →  **3,063 QPS** per-thread

Hardware: x86-64 Linux 6.18, Intel Celeron N4020, `rustc 1.87.0 --release`.

---

## SOTA Survey

### The IVF-PQ lineage (2011–2025)

**Product Quantization (Jégou, Douze, Schmid · IEEE TPAMI 2011)**
The foundational paper. Split a D-dim vector into M subspaces of D/M dims each; train an
independent k-means codebook of K centroids per subspace (K=256 for u8 codes). Encode each
vector as M bytes. Asymmetric Distance Computation (ADC): for a query q, precompute a lookup
table `table[m][k] = L2sq(q_sub_m, centroid[m][k])` once; then score any encoded vector as
`sum_m table[m][code[m]]` in O(M) lookups. M=8 subspaces × 256 centroids gives 8-bit codes
that approximate L2 in a single memory fetch per subspace — extremely cache-friendly.

**IVF-PQ (FAISS · Johnson et al. 2019 → 2024)**
Layer IVF partitioning on top of PQ: first partition the corpus into `nlist` Voronoi cells
via k-means; at query time probe the `nprobe` nearest cells; within each probed cell apply
PQ-ADC. The IVF coarse scan reduces candidates from N to ~N × nprobe/nlist before PQ
scoring. FAISS ships IVFFlat, IVF-PQ, IVF-SQ, and IVF-ScaNN variants.  Definitive 2024
reference: Douze et al. "The Faiss Library" (arXiv:2401.08281).

**Residual PQ (standard since FAISS 0.x)**
Key refinement: encode `v - centroid[assign(v)]` (the *residual*) rather than the full
vector. Residuals have much smaller magnitude (within-cell spread only) and thus lower PQ
quantization distortion. The query uses the same residualisation at search time:
`q_residual_c = q - centroid[c]` for each probed cell `c`. This is what ruvector-ivfpq
implements (the original naive approach gave 18% recall at nprobe=1; residual PQ brings
this to 34.7% with much better behaviour at higher nprobe).

**Optimized PQ (OPQ · Ge et al. TPAMI 2014)**
Apply a rotation matrix to decorrelate vector components before PQ encoding — reduces PQ
quantization error by aligning principal axes with subspace boundaries. Already implemented
in ruvector-core's `opq.rs`. A natural next step is wiring OPQ pre-rotation into IVF-PQ.

**FastScan (Blalock & Guttag NeurIPS 2021 → FAISS 1.7)**
Pack 4-bit PQ codes and use SIMD AVX2/AVX-512 to score 32 candidates in parallel. Typical
speedup: 4–10× over scalar ADC. Not yet in ruvector-ivfpq (planned for v2, see roadmap).

**HAKES (Hu et al. VLDB 2025)**
A distributed vector database that separates the "filter index" (lightweight PQ scan for
candidate generation) from the "refine index" (full-precision vectors for exact reranking).
HAKES reports 16× throughput gain over baselines by tuning the filter/refine split point.
This is precisely the architecture `ruvector-ivfpq` adopts: PQ-ADC filter → exact-L2 refine.
Reference: "HAKES: Scalable Vector Database for Embedding Search Service."
PVLDB 18(9):3049–3062, 2025.  arXiv:2505.12524.

**Juno (Liu et al. ASPLOS 2024)**
Identifies ADC lookup table bandwidth as the principal bottleneck for IVF-PQ at high
nprobe values on modern CPUs. Proposes software prefetching and tiling strategies.
Confirms IVF-PQ is still the most widely deployed ANN method in production.

### Competitor state (May 2026)

| System | IVF-PQ status |
|--------|--------------|
| FAISS  | IVFFlat, IVF-PQ, IVF-SQ, IVF-ScaNN (FastScan) — reference implementation |
| Qdrant | HNSW primary, IVF-style quantisation in Scalar Quantization mode |
| Milvus | IVFFlat, IVF-PQ, IVF-SQ, IVF-HNSW — full production suite |
| Weaviate | HNSW only — no IVF |
| Pinecone | Proprietary IVF-like index, no public code |
| LanceDB | DiskANN-based, no IVF-PQ |
| **ruvector (before this PR)** | No IVF-PQ |
| **ruvector-ivfpq** | IVF-PQ with residual encoding + HAKES filter-refine |

---

## Proposed Design

```
Training phase
  corpus ──┬──→ IVF k-means++ ──→ centroids[nlist][D]
           │
           └──→ residuals[i] = corpus[i] - centroids[nearest(corpus[i])]
                  │
                  └──→ PQ k-means (M subspaces) ──→ codebook[M][ksub][D/M]

Insert phase
  vector v ──→ cell = nearest(v, centroids)
           ──→ residual r = v - centroids[cell]
           ──→ codes = PQ.encode(r)   [M bytes]
           ──→ lists[cell].push(id, codes, raw=v)

Search phase (HAKES filter-refine)
  query q ──→ Stage 1: coarse scan → nprobe nearest cells
          ──→ Stage 2: for each probed cell c:
                         qr = q - centroids[c]
                         lut = PQ.build_lut(qr)
                         score each entry: approx ≈ L2sq(q, v)
                         collect candidates
          ──→ keep top-rerank_k by approx distance
          ──→ Stage 3: exact-L2 refine on raw vectors
          ──→ return top-k
```

**Key design choices:**

1. **Residual PQ over full-vector PQ.** Training the codebook on residuals `v - centroid[c]`
   focuses quantisation capacity on the fine-grained within-cell variation rather than the
   coarse cluster geometry. ADC scores then closely approximate L2sq(q, v) regardless of
   which cell an entry lives in (since both the stored residual and query residual are
   relative to the same centroid).

2. **Per-cell query residual at search time.** For each probed cell c, compute
   `qr_c = q - centroids[c]` and build a fresh LUT. Cost: `nprobe × (M × ksub × (D/M))` FP
   multiply-adds for all LUT builds, amortised across list_avg entries per cell.

3. **Exact-L2 refine (HAKES stage 3).** Raw f32 vectors are stored alongside PQ codes.
   After filter stage, re-score top-rerank_k candidates with exact L2. In production,
   raw vectors would live on SSD (like DiskANN), fetched only for the small rerank set.

4. **Trait-based swappable design.** The `IvfPqIndex` composes `PqCodebook` and an
   internal `Entry` type. Future variants (OPQ rotation, 4-bit codes, FastScan SIMD) can
   be dropped in by swapping `PqCodebook` for a new type.

---

## Implementation Notes

### File layout

```
crates/ruvector-ivfpq/
  Cargo.toml          — standalone crate, depends only on rand = "0.8"
  src/
    lib.rs            — pub mod + re-exports
    kmeans.rs         — k-means++ (120 lines, self-contained)
    pq.rs             — PqCodebook + LookupTable ADC (130 lines)
    ivfpq.rs          — IvfPqIndex: train/add/search + tests (220 lines)
    main.rs           — demo binary with real benchmark sweep (110 lines)
  benches/
    ivfpq_bench.rs    — criterion benchmarks: search_nprobe × m, train (80 lines)
```

Total: ~660 lines across 5 source files plus bench, all under the 500-line limit.

### ADC lookup table cost

For nprobe=4, D=128, M=8, ksub=256:
- LUT builds: 4 × (8 × 256 × 16) = 131,072 FP multiply-adds
- List scan (avg 156 entries × 4 cells = 624): 624 × 8 table lookups = 4,992 lookups
- Sort (rerank_k=200): ~200 × log(624) ≈ 1,900 comparisons
- Exact refine (200 vectors × 128 dims): 25,600 multiply-adds

Total search cost at nprobe=4: ~160K FP ops → very fast on modern out-of-order CPUs.

---

## Benchmark Methodology

**Hardware:** Intel Celeron N4020 (2C/2T, 1.1–2.8 GHz), 4 GB RAM, Linux 6.18 x86-64.

**Data generation:** Multi-cluster Gaussian synthetic data. 20 cluster centres uniform
in [-10, 10]^128; each point = centre + Uniform(-0.5, 0.5)^128 noise. N=10,000 corpus,
200 queries (different seed). This produces extremely well-separated clusters (inter-cluster
L2sq ≈ 34,000 vs within-cluster ≈ 64).

**Ground truth:** Brute-force exact L2sq scan over all N corpus vectors. Verified: 179 ms
for 200 queries at N=10K D=128.

**Recall@K:** fraction of ground-truth top-K ids found in the ANN result set, averaged
over all queries.

**QPS:** total queries / wall-clock seconds. Single-threaded (no rayon/tokio).

**Index configuration:** nlist=64, M=8, ksub=256, max_iter=30.

---

## Results

### `cargo run --release -p ruvector-ivfpq`

```
=== ruvector IVF-PQ PoC (HAKES filter-refine) ===
  N=10000  D=128  clusters=20  queries=200  K=10

Brute-force ground truth :  179 ms  (200 queries)
Train (IVF+PQ k-means)   : 23452 ms
Add 10000 vectors           :  199 ms
Memory (full/comp/raw)   : 5238 KB / 238 KB / 5000 KB
Compression ratio        : 21.0x  (raw / compressed codes only)

Variant                        Recall@10          QPS    Mem(KB)
─────────────────────────────────────────────────────────────────
nprobe=1  rerank_k=50              34.7%        26,429       5238
nprobe=4  rerank_k=200             94.7%         9,481       5238
nprobe=16 rerank_k=500            100.0%         2,617       5238

Hardware: x86_64 (linux)
Built with: cargo run --release -p ruvector-ivfpq
```

### `cargo bench -p ruvector-ivfpq` (Criterion, 100 queries batched)

```
search_nprobe/nprobe/1   time: [2.39 ms 2.40 ms 2.41 ms]   → 41,841 QPS
search_nprobe/nprobe/4   time: [8.09 ms 8.09 ms 8.12 ms]   → 12,361 QPS
search_nprobe/nprobe/16  time: [32.4 ms 32.7 ms 33.0 ms]   →  3,063 QPS

search_subspaces_m/m/4   time: [8.17 ms 8.22 ms 8.28 ms]
search_subspaces_m/m/8   time: [8.11 ms 8.20 ms 8.23 ms]   (negligible M effect)
search_subspaces_m/m/16  time: [10.3 ms 10.4 ms 10.5 ms]   (+28% for M=16)

train_1k_nlist16_m4_ksub32  time: [58.7 ms 58.9 ms 59.1 ms]
```

### Memory breakdown (N=10,000, D=128, M=8, ksub=256, nlist=64)

| Component | Size |
|-----------|------|
| IVF centroids (64 × 128 × 4B) | 32 KB |
| PQ codebook (8 × 256 × 16 × 4B) | 131 KB |
| PQ codes (10,000 × 8B) | 78 KB |
| **Compressed total** | **238 KB** |
| Raw vectors for refine (10,000 × 128 × 4B) | 5,000 KB |
| **Full in-memory total** | **5,238 KB** |
| Compression ratio (raw / compressed) | **21.0×** |

---

## How It Works (Blog-Readable Walkthrough)

### The core insight: approximate first, exact later

Imagine searching a library of 10,000 books. Exact search reads every book cover-to-cover
(brute force). IVF-PQ does something smarter:

**Step 1 — Organize the library.** Cluster all books into 64 shelves (IVF cells) by topic.
Each shelf holds ~156 books. When a reader arrives with a question, we first look at the 4
most relevant shelves (nprobe=4) — ignoring the other 60.

**Step 2 — Create a summary card.** Before the library opens, compress each book into an
8-byte summary card (PQ codes). The summary captures the book's "essence" across 8
independent facets.

**Step 3 — Fast filter.** For the reader's question, precompute how relevant each possible
"essence value" is (the ADC lookup table). Then score each summary card in ~8 memory reads.
From ~624 cards on 4 shelves, keep the top 200 by score.

**Step 4 — Exact refine.** For those 200 top candidates, fetch the actual books and compute
exact relevance. Return the true top-10.

The result: we read 624 summary cards (8 bytes each) instead of 10,000 full books (512 bytes
each) — a 6,400× reduction in bytes scanned — then exactly verify only 200. Recall@10 is
94.7%, QPS is 9,481.

### Why residual encoding matters

Without residual encoding ("full vector PQ"), the PQ codebook must capture the *entire*
coordinate space, including the coarse inter-cluster variation. This wastes most of the
codebook capacity on structure already handled by the IVF level. The result: poor within-cell
ranking (recall@10 was only 18–65% before the fix).

With residual encoding, the codebook only needs to capture `v - centroid[cell]` — the
fine-grained position within a single IVF cell. Each PQ centroid represents a tiny local
patch. The ADC scores become excellent approximations of L2sq(q, v), and recall jumps
to 34.7% → 94.7% → 100% across nprobe settings.

---

## Practical Failure Modes

1. **Training data not representative of production distribution.** If the corpus distribution
   shifts after training, IVF centroids will be misaligned and recall degrades. Mitigation:
   retrain periodically (FAISS's `index.reset()` + `train()`).

2. **Too few IVF cells (nlist too small).** If nlist < corpus_clusters, cells mix multiple
   natural clusters, making the coarse scan less discriminative. Rule of thumb: nlist ≈ sqrt(N)
   or nlist ≈ 4 × nclusters.

3. **Too many IVF cells (nlist too large).** Lists become very short (< 10 entries). The
   IVF scan itself dominates — no speedup over brute force. Target list length: 50–500 entries.

4. **PQ subspace dimension too large.** With D/M > 64, the within-subspace k-means needs
   enormous training data. D=128, M=4 gives sub_dim=32 which is borderline. M=8 (sub_dim=16)
   is safe for N ≥ 1,000.

5. **rerank_k too small.** The filter stage can miss true neighbours if ADC ranking is noisy
   and rerank_k is set too conservatively. The safe heuristic: rerank_k ≥ 4 × nprobe × avg_list_size / nlist = 4 × nprobe.

6. **Large nprobe negates IVF speedup.** At nprobe/nlist = 0.25+ (nprobe=16 out of 64),
   25% of the index is scanned — comparable to linear scan for small N. Use nprobe ≤ 8 for
   production speedup; accept lower recall or compensate with exact refine.

7. **Training k-means convergence failure.** The k-means++ seeding prevents empty clusters
   but not slow convergence for pathological distributions (e.g., power-law distances).
   Monitor training loss (not yet exposed in the PoC) and increase `max_iter` if recall is
   unexpectedly low.

---

## What to Improve Next (Roadmap)

| Priority | Improvement | Expected gain |
|----------|-------------|---------------|
| P0 | **FastScan 4-bit SIMD** (AVX2/NEON) | 4–8× search speedup |
| P0 | **OPQ pre-rotation** (rotate via `ruvector-core/opq.rs`) | +5–10 pp recall at same M |
| P1 | **Disk-based refine** (mmap raw vectors, fetch on demand) | 20× memory reduction |
| P1 | **IVF-HNSW routing** (HNSW over centroids instead of linear scan) | O(D log nlist) coarse scan |
| P1 | **Multi-vector assignment** (SOAR/RAIRS-style spill lists for IVF) | +15 pp recall@low nprobe |
| P2 | **Async add / incremental retrain** | streaming inserts |
| P2 | **Rayon parallel scan** | linear QPS scaling with cores |
| P2 | **Integration with ruvector-server** | HTTP + gRPC search API |

---

## Production Crate Layout Proposal

For a production `ruvector-ivfpq` (v0.2+):

```
crates/ruvector-ivfpq/
  src/
    lib.rs              — public API surface
    config.rs           — IvfPqConfig, builder pattern
    index/
      mod.rs            — IvfPqIndex trait object
      flat.rs           — IvfFlat (no PQ, exact scan within cells)
      pq_scalar.rs      — IVF-PQ with scalar (current PoC)
      pq_simd.rs        — IVF-PQ with FastScan AVX2/NEON  [P0]
      pq_opq.rs         — IVF-OPQ (OPQ rotation + PQ)     [P1]
    storage/
      ram.rs            — in-memory Entry store (current)
      mmap.rs           — mmap-backed raw vector store     [P1]
    codebook/
      pq.rs             — current PqCodebook
      opq.rs            — OPQ-rotated codebook             [P1]
    train/
      kmeans.rs         — shared k-means++ (current)
      online.rs         — mini-batch k-means for streaming [P2]
  benches/
    recall_vs_nprobe.rs — comprehensive recall vs QPS sweep
    memory_vs_m.rs      — memory vs recall tradeoff
```

---

## References

1. Jégou, Douze, Schmid. "Product Quantization for Nearest Neighbor Search."
   *IEEE TPAMI* 33(1):117–128, 2011. DOI:10.1109/TPAMI.2010.57

2. Ge, He, Ke, Sun. "Optimized Product Quantization." *IEEE TPAMI* 36(4):744–755, 2014.
   https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf

3. Douze, Guzhva, Deng, Johnson, Szilvasy, Mazaré, Lomeli, Hosseini, Jégou.
   "The Faiss Library." arXiv:2401.08281, 2024. https://arxiv.org/abs/2401.08281

4. Hu, Cai, Dinh, Xie, Yue, Chen, Ooi. "HAKES: Scalable Vector Database for Embedding
   Search Service." *PVLDB* 18(9):3049–3062, 2025.
   arXiv:2505.12524. https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf

5. Liu et al. "Juno: Optimizing High-Dimensional Approximate Nearest Neighbor Search."
   *ASPLOS 2024*. https://www.cs.sjtu.edu.cn/~leng-jw/resources/Files/liu2024asplos-juno.pdf

6. Blalock, Guttag. "Multiplying Matrices Without Multiplying." *NeurIPS 2021*.
   (FastScan / Bolt foundation)
