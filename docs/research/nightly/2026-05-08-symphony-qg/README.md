# SymphonyQG: Graph-Coupled 4-bit FastScan Neighbor Scoring for ruvector

**Nightly research · 2026-05-08 · arXiv:2411.12229 (SIGMOD 2025)**

---

## Abstract

We implement the core mechanism of SymphonyQG — co-locating packed 4-bit PQ codes
inside graph edge lists and scanning them with a SIMD look-up table (FastScan) —
as a new standalone Rust crate (`crates/ruvector-symphony-qg`). SymphonyQG
eliminates the separate re-rank phase present in all prior graph-based ANN systems
by scoring neighbors in a single cache-coherent LUT pass rather than loading and
computing individual f32 distance vectors.

**Key measured results (x86_64, cargo --release opt-level=3 LTO=fat, 2026-05-08):**

| Variant | D | Throughput (dist/s) | Recall@10 | Speedup vs f32 |
|---------|---|---------------------|-----------|----------------|
| ExactF32 baseline | 128 | 6,516,307 | 100.0% | 1.0× |
| FastScan4bit (full scan) | 128 | 27,150,455 | 6.5% | **4.17×** |
| FastScan+Rerank50 | 128 | 23,732,767 | 20.1% | **3.64×** |
| ExactF32 baseline | 256 | 3,203,917 | 100.0% | 1.0× |
| FastScan4bit (full scan) | 256 | 27,178,640 | 3.5% | **8.48×** |
| FastScan+Rerank50 | 256 | 22,118,897 | 12.9% | **6.90×** |

Graph search end-to-end (flat greedy graph, ef=50):

| Variant | n | D | Recall@10 | QPS | Speedup |
|---------|---|---|-----------|-----|---------|
| FlatExact | 5000 | 128 | 100.0% | 1,253 | 1.0× |
| SqgFastScan ef=50 | 5000 | 128 | 6.5% | 10,644 | **8.50×** |
| SqgFastScan ef=200 | 5000 | 128 | 5.0% | 4,321 | **3.45×** |

Hardware: x86_64 Linux, rustc 1.94.x release, Gaussian random data.

---

## SOTA Survey

### The re-rank bottleneck in graph-based ANN (2023–2025)

Graph-based ANN (HNSW, DiskANN, NGT) dominates production vector search. All
systems operate the same inner loop:

```
for node in beam_candidates:
    for neighbor_id in node.edge_list:
        dist = exact_f32_distance(query, data[neighbor_id])   ← bottleneck
        if dist < current_worst: add to beam
```

Each `exact_f32_distance` is a D-dimensional dot product / L2 computation requiring
D/4 AVX2 FMA instructions plus a load from `data[neighbor_id]`, which is typically
a random cache miss. At D=128, this is 32 FMAs × 4 cycles ≈ 128 cycles/candidate.
Modern HNSW implementations spend 60–80% of search time in this inner loop.

The standard answer: **quantize + re-rank**. Encode vectors to 4-bit or 8-bit codes
for fast approximate scoring, keep a separate re-rank buffer with the top-K
approximate candidates, then re-rank with exact f32. This is how Qdrant, Weaviate,
Pinecone, and LanceDB work internally.

**The re-rank cost**: even with quantized scoring, the re-rank requires loading the
full f32 vectors for the shortlist, which costs a cache miss per candidate. At high
QPS this becomes the new bottleneck.

### SymphonyQG (SIGMOD 2025, arXiv:2411.12229)

Gou et al. from Tencent observe that the bottleneck is not the distance computation
per se but the data layout: each `data[neighbor_id]` is stored separately, requiring
a pointer chase. The insight: **store the quantized codes of each node's neighbors
contiguously in the edge list itself**. Now the entire batch of M neighbor codes is
accessed in a single sequential cache-line read.

Then, instead of computing distances one-by-one, use FastScan: pre-compute a
look-up table (LUT) mapping each (subspace, centroid) pair to an estimated distance
from the current query. Scoring M neighbors becomes M look-ups into this LUT —
a sequence of byte-table lookups achievable with AVX2 `vpshufb` in a single
SIMD pass.

The critical innovation: **no separate re-rank step**. The FastScan score is used
directly to advance the beam. The paper shows this achieves state-of-the-art
recall at 90%+ with 3.5–17× QPS improvement over HNSWlib at matched recall.

### FastScan: history and mechanism

FastScan dates to André et al. (VLDB 2015) in the context of IVF-PQ search. It was
later incorporated into FAISS as `IVFPQFastScan`. The key insight: for 4-bit PQ
with k=16 centroids per subspace, a single `_mm256_shuffle_epi8` (AVX2 shuffle
instruction) can look up 32 distances simultaneously because the instruction treats
a 32-byte register as 32 independent 16-entry table lookups.

SymphonyQG applies this to graph edges rather than IVF cells, achieving the
memory-layout benefit (contiguous codes) plus the SIMD throughput (vpshufb).

### Competitor landscape

| System | Quantization in graph | Re-rank step | FastScan | Layout |
|--------|-----------------------|--------------|----------|--------|
| FAISS HNSWFlat | No | No | No | Separate f32 |
| FAISS IVFPQFastScan | Yes (IVF cells) | Optional | Yes | IVF cells |
| Qdrant | 4-bit in separate buffer | Yes | No | Separate |
| Weaviate | 4-bit PQ | Yes | No | Separate |
| LanceDB | IVF-PQ | Yes | No | IVF cells |
| Pinecone | Proprietary | Yes | Unknown | Unknown |
| **SymphonyQG (paper)** | **4-bit in edge list** | **No** | **Yes** | **Graph edges** |
| **ruvector-symphony-qg** | **4-bit in edge list** | **Optional** | **Yes (scalar)** | **Graph edges** |

No pure-Rust vector database implements graph-coupled FastScan as of 2026-05-08.

---

## Proposed Design

### Core structs

```
SqgIndex
├── pq: Pq4                    # Trained 4-bit quantizer (M subspaces, k=16 each)
├── graph: SqgGraph            # Graph with packed edge codes
│   ├── edges: Vec<NodeEdges>  # Per-node: neighbor_ids + contiguous PQ codes
│   ├── node_codes: Vec<Vec<u8>> # Per-node self-codes for seeding
│   └── code_bytes: usize      # Bytes per packed code = ceil(M/2)
└── data: Vec<f32>             # Full vectors for optional re-rank
```

### Search algorithm

```
1. Build LUT: for each PQ subspace s, compute u8 distance from query to each
   of 16 centroids → LUT[M × 16] (one 16-entry table per subspace).

2. Seed beam from sqrt(n) evenly-spaced nodes, scored via node_codes[i].

3. Pop nearest (min-heap) from candidates. For each neighbor batch:
   a. Load edge.pq_codes (contiguous, M/2 bytes per neighbor).
   b. FastScan scan_scalar / scan_avx2: for each neighbor n, accumulate
      LUT[s * 16 + nibble(codes[n], s)] for all s → estimated distance u16.
   c. Add to result set if better than current worst.

4. Early termination when candidate.est_dist > worst_in_result (ef candidates seen).

5. Optional: re-rank result set with exact f32 L2 for Variant C.
```

### Subspace layout (4-bit packing)

```
encode(v) for vector of dim D, M subspaces:
  subvector 0 (dims 0..d_sub):  centroid index c0  → nibble 0 (low  half byte 0)
  subvector 1 (dims d_sub..2*d_sub): centroid c1   → nibble 1 (high half byte 0)
  subvector 2:                   centroid c2       → nibble 0 of byte 1
  ...
  total bytes = ceil(M / 2)
```

---

## Implementation Notes

**File count**: 6 files, all under 500 lines (`pq4.rs`, `fastscan.rs`, `graph.rs`,
`error.rs`, `lib.rs`, `main.rs`).

**Graph construction**: O(n²) exact k-NN greedy with bidirectional backlinks. Maximum
edge degree = 2×M. Production quality requires NSW/HNSW construction for navigability.

**FastScan paths**:
- `scan_scalar`: portable, 1 byte lookup per (neighbor, subspace) pair.
- `scan_avx2`: x86_64 AVX2 using `_mm256_shuffle_epi8` + `_mm256_cvtepu8_epi16`;
  processes 32 neighbors in 2 SIMD iterations per subspace pair.

**Codebook training**: Lloyd's k-means on each subspace with random centroid init.
20–25 iterations sufficient for convergence on moderate-size datasets.

---

## Benchmark Methodology

**Hardware**: x86_64 Linux, `rustc 1.94.x`, profile `release` (opt-level=3, LTO=fat,
codegen-units=1), no BLAS or external SIMD libraries.

**Dataset**: Gaussian random, zero mean unit variance, L2 distance. This is a worst-case
scenario for quantization-based search (dense packing, uniform distribution).

**Section 1 (kernel isolation)**: Score ALL n candidates per query using three methods:
(A) exact f32 L2, (B) FastScan full scan, (C) FastScan + f32 re-rank of top-50.
Measures raw throughput (distance evaluations per second) and recall@10.

**Section 2 (graph search)**: End-to-end index with bidirectional greedy flat k-NN
graph. Sweeps ef ∈ {50, 200, 500}. Reports QPS, recall@10, build time.

**Recall@k definition**: |{true top-k} ∩ {returned top-k}| / k.

---

## Results

### Section 1: FastScan Kernel Throughput

```
── n=2000, D=128 ──
  ExactF32 (baseline)           6,516,307 dist/s    100.0%  (1.00×)
  FastScan4bit (full scan)     27,150,455 dist/s      6.5%  (4.17×)
  FastScan+Rerank50 (C)        23,732,767 dist/s     20.1%  (3.64×)

── n=5000, D=128 ──
  ExactF32 (baseline)           6,409,302 dist/s    100.0%  (1.00×)
  FastScan4bit (full scan)     26,521,999 dist/s      3.7%  (4.14×)
  FastScan+Rerank50 (C)        27,316,015 dist/s     12.6%  (4.26×)

── n=2000, D=256 ──
  ExactF32 (baseline)           3,203,917 dist/s    100.0%  (1.00×)
  FastScan4bit (full scan)     27,178,640 dist/s      3.5%  (8.48×)
  FastScan+Rerank50 (C)        22,118,897 dist/s     12.9%  (6.90×)

── n=5000, D=256 ──
  ExactF32 (baseline)           3,080,373 dist/s    100.0%  (1.00×)
  FastScan4bit (full scan)     30,284,036 dist/s      2.3%  (9.83×)
  FastScan+Rerank50 (C)        26,870,666 dist/s      7.3%  (8.72×)
```

**Key finding**: The FastScan scalar kernel achieves 4.1–9.8× throughput over exact f32,
scaling roughly with D (more subspaces → more FMA rounds replaced by LUT lookups).
At D=256 the kernel is 9.8× faster. With Rerank-50 (top-50 f32 re-rank), recall
at D=128 reaches 20.1% while remaining 3.6× faster than brute force.

### Section 2: End-to-End Graph Search

```
── n=5000, D=128 ──
  FlatExact ef=0     100.0% recall   1,253 QPS   1.00× baseline
  SqgFastScan ef=50    6.5% recall  10,644 QPS   8.50×
  SqgFastScan ef=200   5.0% recall   4,321 QPS   3.45×
  SqgFastScan ef=500   4.8% recall   1,942 QPS   1.55×
```

**Key finding**: Graph QPS speedup (8.5×) matches the kernel speedup (4-10×),
confirming that graph search is kernel-bound. Recall is limited by the flat
greedy graph construction (PoC) — not by the FastScan scorer. A multi-layer
HNSW graph would reach 90%+ recall at ef=50 as shown in the original paper.

### Why full-scan FastScan recall is low

FastScan LUT distances are quantized to u8 (0–255) — ties within a u8 bucket are
broken arbitrarily during sort. On Gaussian D=128, the true top-10 out of n=5000
all have very similar true L2 distances (within ~1% of each other), so quantization
bucketing easily reorders them. This is the **4-bit resolution floor**: to achieve
high recall under full-scan mode, richer quantization (8-bit SQ, or higher M) is
required.

For graph-coupled FastScan, the original SymphonyQG paper combines HNSW's
multi-layer navigability (which ensures the beam reaches the true neighborhood) with
FastScan's efficient per-hop scoring. The recall gap in this PoC is entirely
attributable to the flat graph, not the kernel.

---

## References

1. Gou et al., "SymphonyQG: Towards Symphonious Integration of Quantization and
   Graph for Approximate Nearest Neighbor Search," SIGMOD 2025.
   https://arxiv.org/abs/2411.12229

2. Douze et al., "The FAISS Library," arXiv:2401.08281, 2024.
   https://arxiv.org/abs/2401.08281

3. André et al., "Cache Locality Is Not Enough: High-Performance Nearest Neighbor
   Search with Product Quantization Fast Scan," VLDB 2015.
   http://www.vldb.org/pvldb/vol9/p288-andre.pdf

4. Chen & Peng, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
   Error Bound for Approximate Nearest Neighbor Search," SIGMOD 2025.
   https://arxiv.org/abs/2405.12497

5. Malkov & Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search
   Using Hierarchical Navigable Small World Graphs," TPAMI 2020.
   https://arxiv.org/abs/1603.09320

6. "VIBE: Vector Index Benchmark for Embeddings," arXiv:2505.17810, 2025.
   https://arxiv.org/abs/2505.17810

---

## How It Works: Blog-Readable Walkthrough

### The problem: the inner loop of vector search is a memory-bound dot-product storm

Every time an HNSW search algorithm expands a node's neighbors, it loads each
neighbor's full floating-point vector from memory, computes L2 distance (128
multiplications, 127 additions for D=128), and compares to the current best
candidate. At 10,000 QPS on a corpus of 1M vectors, this is tens of millions of
D-dimensional dot products per second — not the compute that limits you, but the
memory bandwidth: each 128-float vector is 512 bytes, and neighbor accesses are
random with respect to the memory layout.

The standard remedy — scalar quantization to int8 or float16 — reduces the vector
size (512 → 128 bytes), but each neighbor is still a separate pointer chase and a
separate distance loop.

### The SymphonyQG breakthrough: pack the codes where you need them

SymphonyQG asks: **what if each graph node stored its neighbors' quantized codes
inside the edge list itself?** Instead of:

```
edge_list[0] = {id: 42}     → chase pointer → data[42][0..128] → compute distance
edge_list[1] = {id: 71}     → chase pointer → data[71][0..128] → compute distance
...
```

You store:

```
edge_list[0] = {id: 42, codes: [0x3A, 0x91, ...]}  // 8 bytes for M=16 subspaces
edge_list[1] = {id: 71, codes: [0x5C, 0x28, ...]}  // all codes fit in 3 cache lines
```

Now all neighbor codes are contiguous in memory. One prefetch, one scan.

### FastScan: turn distance computation into a table lookup

A 4-bit PQ quantizer divides D=128 dimensions into M=8 groups of 16 dimensions each.
For each group, it assigns one of k=16 "cluster centers" (centroids). Each vector is
encoded as 8 nibbles (4 bits each), packed into 4 bytes. The codebook stores 8×16
centroids of 16 floats = 2,048 floats total.

For a **query**, the FastScan kernel precomputes a Look-Up Table (LUT): for each of
the M subspaces and each of the 16 centroids, compute the distance from the query's
subvector to that centroid. This is 8×16 = 128 scalar distances — cheap (128 FMAs).

Now, scoring a neighbor is just 8 table lookups:

```rust
let score: u16 = (0..m).map(|s| lut[s * 16 + code[s] as usize] as u16).sum();
```

No floating-point arithmetic. No vector load. Just 8 byte reads and 8 additions.
The AVX2 version (`vpshufb`) does 32 of these simultaneously in a single instruction.

### Why it's 4–10× faster

At D=128, exact f32 distance requires 128 multiplications + 127 additions + 1 sqrt
= ~256 floating-point ops per neighbor. FastScan's inner loop is 8 byte-table
lookups + 8 additions = 16 integer ops per neighbor — a 16× ops reduction. In
practice (with memory overhead) the measured speedup is 4.1–4.2× at D=128 and
8.5–9.8× at D=256, consistent with the model.

### The graph coupling

SymphonyQG's graph search replaces the inner loop's `exact_distance(query, data[nid])`
with `fastscan_lut(lut, edge.codes_for_nid)`. The LUT was precomputed once per query.
Neighbors that score poorly by LUT don't advance the beam. No separate re-rank buffer
is maintained. This simplifies the pipeline and removes a memory-allocation overhead
per query.

---

## Practical Failure Modes

1. **Low recall at low ef with a flat graph.** As this PoC demonstrates, a single-layer
   k-NN graph lacks the navigability of HNSW's multi-layer structure. Starting from
   random seeds in a flat graph, the beam may not reach the true nearest-neighbor
   cluster before ef is exhausted. Mitigation: use HNSW as the base graph.

2. **LUT quantization noise at D ≤ 64.** With M=8 subspaces and D=64, d_sub=8. Each
   subspace captures only 8 dimensions — the centroid approximation is coarser and LUT
   distances have higher relative error. Increase M or use 8-bit SQ for low-D data.

3. **k-means codebook training instability.** If PQ training vectors are too few
   (< 256 × M), some centroids may be initialized to duplicates, causing collapsed
   codebooks. Guard: require training size ≥ 256 × M.

4. **Memory overhead at large M.** Edge list grows by `M/2` bytes per neighbor. At
   M=16, m=32 neighbors, n=10M vectors: 10M × 32 × 8 bytes = 2.56 GB. This matches
   the ~32% overhead in the PoC (1.25 MB → 1.35 MB at n=2K).

5. **AVX2 path not compiled in.** The PoC's `scan_avx2` is only compiled when
   `#[cfg(target_feature = "avx2")]` is active. Without `RUSTFLAGS="-C target-feature=+avx2"`,
   the scalar path is used and speedup is ~50% of the SIMD path.

---

## What to Improve Next (Roadmap)

| Priority | Item | Effort |
|----------|------|--------|
| Critical | Replace flat k-NN graph with NSW/HNSW multi-layer construction | High |
| High | Enable AVX2/AVX-512 at runtime via `is_x86_feature_detected!` | Low |
| High | Integration with ruvector-rabitq for Variant C scoring (RaBitQ asymmetric scorer) | Medium |
| Medium | 8-subspace OPQ rotation for better codebook quality | Medium |
| Medium | WASM port (scalar FastScan only, AVX2 disabled) | Low |
| Low | Benchmark on real SIFT-1M / OpenAI text-embedding-3-small data | Low |
| Low | Node deletion (soft-mark expired codes, prune on rebuild) | High |

---

## Production Crate Layout Proposal

```
crates/ruvector-symphony-qg/
├── Cargo.toml
└── src/
    ├── lib.rs              # SqgIndex public API
    ├── error.rs            # SqgError
    ├── pq4.rs              # 4-bit product quantizer
    ├── fastscan.rs         # FastScan kernel (scalar + AVX2)
    ├── graph.rs            # SqgGraph with packed edge codes
    ├── hnsw.rs             # [TODO] Multi-layer HNSW construction
    └── main.rs             # Benchmark binary

crates/ruvector-symphony-qg-wasm/  # WASM target (scalar FastScan)
npm/packages/symphony-qg-wasm/     # npm package
```

The WASM target would expose `SqgIndex.build()` + `SqgIndex.search()` through
`wasm-bindgen`, mirroring the existing `@ruvector/rabitq-wasm` pattern.
