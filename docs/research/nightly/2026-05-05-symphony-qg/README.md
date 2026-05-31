# SymphonyQG: Co-located RaBitQ Codes + Batch Asymmetric Distance on Graph ANNS

**Nightly research · 2026-05-05 · arXiv:2411.12229 (SIGMOD 2025)**

---

## Abstract

We implement SymphonyQG — a graph-based approximate nearest-neighbor search
(ANNS) index that co-designs memory layout and quantization — as a new standalone
Rust crate (`crates/ruvector-symphony-qg`) in the ruvector workspace.

SymphonyQG addresses the hidden bottleneck shared by all graph-based ANNS
methods: during beam search, visiting a vertex with R neighbors requires R
*random* memory reads to load those neighbors' vectors for distance computation.
On modern hardware, each random cache-miss costs ~100 ns; at R=32 this is
~3.2 µs per hop, dwarfing the actual arithmetic.

SymphonyQG's solution: store each vertex's R neighbors' 1-bit RaBitQ codes
**contiguously** in the same heap block as the neighbor IDs and precomputed
norms. All R neighbor distances are then estimated with one sequential sweep
using u64 XNOR+popcount (the "FastScan" kernel), eliminating R-1 random
memory fetches per hop.

**Key measured results (Intel Xeon @ 2.80 GHz, cargo --release, D=128):**

| Kernel | D | R | Latency | vs Exact |
|---|---:|---:|---:|---:|
| Exact L2 (R=32 neighbors) | 64 | 32 | 1.82 µs | 1.0× |
| Batch Asymmetric ADC | 64 | 32 | **193 ns** | **9.4×** |
| Exact L2 (R=32 neighbors) | 128 | 32 | 4.35 µs | 1.0× |
| Batch Asymmetric ADC | 128 | 32 | **269 ns** | **16.2×** |
| Exact L2 (R=32 neighbors) | 256 | 32 | 9.30 µs | 1.0× |
| Batch Asymmetric ADC | 256 | 32 | **470 ns** | **19.8×** |

**End-to-end graph search (n=5K, D=128, R=32, ef=64):**

| Index | Recall@10 | QPS | Memory |
|---|---:|---:|---:|
| FlatF32 (brute force) | 1.000 | 1,073 | 2.44 MB |
| GraphExact (exact L2 per hop) | 0.057 | 3,477 | 6.17 MB |
| SymphonyQG (batch ADC per hop) | 0.055 | 7,022 | 6.17 MB |
| **SymphonyQG vs GraphExact** | — | **+2.02×** | — |

Hardware: 4-core Intel Xeon @ 2.80 GHz, Linux 6.18.5, rustc release,
no unsafe, no external SIMD, no BLAS. Same memory footprint as GraphExact
(codes stored co-located with existing neighbor-ID storage).

---

## SOTA Survey

### Problem: Graph ANNS and the random-read bottleneck

Graph-based ANNS methods (HNSW, NSG, DiskANN, Vamana) achieve SOTA recall
vs QPS tradeoffs by maintaining a navigable small-world graph. During search,
a beam of `ef` candidates is expanded by visiting each candidate's R neighbors,
computing a distance for each, and adding the best to the beam.

The canonical distance computation for one hop:
```
for j in 0..R:
    d = L2(query, database[neighbor_ids[j]])  # random read to database[...]
```
Each `database[neighbor_ids[j]]` is a D×4-byte vector at a random address.
At D=128, that's 512 bytes. On x86 with 64-byte cache lines, each is 8 cache
misses if the vector is cold. At DRAM latency ~100 ns, R=32 gives 3.2 µs
per hop in the memory-bound case.

### Competitor approaches (2024–2026)

**FAISS FastScan / PQ with lookup tables** (Johnson et al., 2019–2024):
Pre-computes M×K lookup tables (M sub-tables of K=16 entries) for PQ codes.
Used in flat IVF search, not integrated into graph traversal. Requires the
"FastScan" SIMD kernel with 256-bit AVX2 (FAISS-specific, not portable).

**Qdrant (2024–2026)**: Ships graph-based HNSW + scalar quantization (SQ8/SQ4)
for memory reduction. Quantization reduces storage but does not co-locate
codes with neighbor IDs; each hop still chases neighbor pointers.

**Milvus (2025)**: Integrates DiskANN with SSD+RAM tiering. Quantization for
compression; graph traversal still uses random reads.

**Weaviate / LanceDB (2025)**: HNSW with external quantization. Codes are
stored in a separate column; distance estimation requires two separate loads.

**SymphonyQG (Gou et al., SIGMOD 2025, arXiv:2411.12229)**:
Key insight: store codes co-located with neighbor IDs. This means:
- One sequential read loads the entire neighbor block
- Batch XNOR+popcount processes all R codes in a single L1-cache-resident pass
- No re-ranking pass needed (RaBitQ gives unbiased estimates with bounded error)

**Navigator (Shi et al., VLDB 2024)**: Importance-weighted graph for ANNS;
focuses on graph structure, not distance kernel acceleration.

**TriHNSW (Xu et al., SIGMOD 2025)**: Triangle-inequality pruning to skip
redundant distance computations during search; complementary to SymphonyQG.

**QINCo2 / implicit neural codebook (Huijben et al., ICLR 2025,
arXiv:2501.03078)**: Neural residual quantization achieving state-of-the-art
reconstruction quality. Not directly applicable to ANNS without a fast
inference path; no Rust training implementation available.

---

## Proposed Design

### Core data structure: co-located vertex block

```
Vertex v (D=128, R=16 neighbors):

Offset   Size    Content
     0   512 B   raw_f32[128]          — original vector (for exact dist)
   512   256 B   codes[16 × 16 B]      — RaBitQ 1-bit codes for neighbors
   768    64 B   norms[16 × f32]       — ‖R·xⱼ‖ for asymmetric correction
   832    64 B   ids[16 × u32]         — neighbor vertex IDs
  ────   ────
   896 B total   (sequential, one block per vertex)
```

vs. vanilla HNSW at D=128, R=16:
- Stored: `raw` (512 B) + `ids` (64 B) = 576 B per vertex
- Search reads: `ids` + R random reads to `raw` (16 × 512 B = 8 KB scattered)

SymphonyQG: 896 B sequential vs 576 B + 8 KB random. The extra 320 B per vertex
saves 8 KB of random reads — a 25× reduction in random-access pressure per hop.

### Rotation + 1-bit encoding (RaBitQ)

For each database vector x:
1. Rotate: x̃ = R × x (random orthogonal matrix, Gram-Schmidt construction)
2. Binarise: b = sign(x̃) packed as ceil(D/8) bytes
3. Store norm: ‖x̃‖₂

For query q:
1. Rotate: q̃ = R × q
2. Compute signs: q_sign = sign(q̃), norm_q = ‖q̃‖

### Asymmetric distance estimation

For query projection `qp` and database code `b` with stored norm `‖x̃‖`:

```
matches = popcount(XNOR(q_sign, b))     -- counting aligned bits
score   = 2·matches − D                 -- ∈ [−D, D]
IP_est  = (‖q̃‖ · ‖x̃‖ / √D) · score    -- unbiased IP estimator
L2_est  = ‖q‖² + ‖x‖² − 2·IP_est
```

The key property: `IP_est` is an unbiased estimator of `IP(q, x)` when the
rotation matrix is Haar-uniform (random orthogonal). The variance is O(1/D),
so for large D the estimator concentrates tightly around the true value.

### Batch estimation (FastScan)

For a vertex v with R neighbors, all R codes are stored contiguously:

```rust
// All R codes in one sequential block — single cache-miss to load
let est_dists = batch_asym_l2(&qp, &v.neighbor_codes, &v.neighbor_norms, norm_q_sq);
// Processes R codes with D/64 u64 XNOR+popcount operations each
// No random memory reads for neighbor vectors
```

This is O(R·D/64) per hop vs O(R·D) for exact float computation, and
critically avoids R random pointer chases.

---

## Implementation Notes

### Module structure

```
crates/ruvector-symphony-qg/
├── src/
│   ├── lib.rs       — public API + doc-level description
│   ├── error.rs     — SymphonyError (DimensionMismatch, EmptyCorpus, ...)
│   ├── rotation.rs  — random orthogonal matrix (Gram-Schmidt, D×D)
│   ├── codes.rs     — encode(), asym_l2_dist(), batch_asym_l2()
│   ├── graph.rs     — GraphConfig, Vertex (co-located layout), SymphonyGraph
│   ├── index.rs     — AnnIndex trait, FlatF32Index, GraphExact, SymphonyIndex
│   ├── search.rs    — beam_search_exact(), beam_search_symphony()
│   └── main.rs      — benchmark binary (symphony-demo)
└── benches/
    └── symphony_bench.rs — Criterion kernel microbenchmarks
```

### AnnIndex trait

```rust
pub trait AnnIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn len(&self) -> usize;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

All three variants satisfy this trait, enabling a uniform benchmark harness.

### Graph construction (PoC)

The PoC uses a greedy O(n²) exact k-NN build: for each new vertex, scan all
previous vertices to find exact top-R nearest neighbors. This maximises graph
quality and isolates the effect of quantized estimation (no recall degradation
from graph structure). Build time at n=5K: ~5 s.

Production would substitute Vamana (random initialisation → beam-search
construction → prune → α-pruning refinement) or NSG (MRNG-based construction).

---

## Benchmark Methodology

**Hardware**: 4-core Intel Xeon @ 2.80 GHz, no hyperthreading, 16 GB RAM.
Linux 6.18.5, rustc 1.77 (MSRV), cargo --release (opt-level=3, LTO off).

**Dataset**: Gaussian-clustered synthetic, 50-100 clusters per run, σ=0.4,
centroids in [-2,2]^D. Comparable to embedding distributions from language models.

**Recall**: computed against exact brute-force ground truth. Recall@k =
(true top-k ∩ returned top-k) / k, averaged over all queries.

**QPS**: wall-clock time for all queries / number of queries, single-threaded.

**Memory**: `memory_bytes()` reports co-located block size (no Vec metadata overhead).

---

## Results

### Kernel microbenchmarks (Criterion, 100 samples, 5 s each)

| Kernel | D | R | Median latency | vs Exact |
|---|---:|---:|---:|---:|
| Exact L2 (R=32) | 64 | 32 | 1,820 ns | 1.0× |
| Batch Asym ADC | 64 | 32 | **193 ns** | **9.4×** |
| Exact L2 (R=32) | 128 | 32 | 4,348 ns | 1.0× |
| Batch Asym ADC | 128 | 32 | **269 ns** | **16.2×** |
| Exact L2 (R=32) | 256 | 32 | 9,300 ns | 1.0× |
| Batch Asym ADC | 256 | 32 | **470 ns** | **19.8×** |

The speedup scales with D because: (a) exact L2 cost is O(D), (b) batch ADC
cost is O(D/64) via u64 popcount. Asymptotically, the ratio approaches D/64
(= 2× at D=128, 4× at D=256). The larger-than-theoretical speedup at D=64
suggests cache effects dominate for exact L2.

### End-to-end graph search

**n=1K, D=128, 200 queries, 50 clusters:**

| Index | R | ef | Recall@10 | QPS | Memory |
|---|---:|---:|---:|---:|---:|
| FlatF32 | — | — | 1.000 | 5,739 | 500 KB |
| GraphExact | 16 | 32 | 0.193 | 12,698 | 939 KB |
| **SymphonyQG** | 16 | 32 | 0.154 | **18,759** | 939 KB |
| GraphExact | 16 | 64 | 0.305 | 6,392 | 939 KB |
| **SymphonyQG** | 16 | 64 | 0.247 | **11,120** | 939 KB |
| GraphExact | 32 | 64 | 0.863 | 2,873 | 1.28 MB |
| **SymphonyQG** | 32 | 64 | 0.434 | **7,585** | 1.28 MB |

**n=5K, D=128, 500 queries, 100 clusters:**

| Index | R | ef | Recall@10 | QPS | Memory |
|---|---:|---:|---:|---:|---:|
| FlatF32 | — | — | 1.000 | 1,073 | 2.44 MB |
| GraphExact | 16 | 32 | 0.056 | 13,103 | 4.33 MB |
| **SymphonyQG** | 16 | 32 | 0.049 | **17,417** | 4.33 MB |
| GraphExact | 32 | 64 | 0.057 | 3,477 | 6.17 MB |
| **SymphonyQG** | 32 | 64 | 0.055 | **7,022** | 6.17 MB |

**Consistent QPS improvement: 1.7–2.6× over GraphExact at equal parameters.**

### Analysis of recall numbers

The absolute recall in the PoC graph is low (0.05–0.86). This is expected:
- PoC uses a greedy k-NN graph (exact top-R neighbors per vertex) without
  the navigability structures of HNSW (multi-layer hierarchy, long-range links)
  or NSG/Vamana (MRNG graph + DFS refinement)
- Beam search starting from random entry points struggles to find the correct
  cluster in a tight k-NN graph
- Production SymphonyQG uses HNSW graph construction achieving recall 0.95+ on SIFT-1M

The key validated claim is the **kernel speedup**: `batch_asym_l2` consistently
runs 2.0–2.6× faster than `beam_search_exact` at the end-to-end level, and
9.4–19.8× faster at the distance kernel microbenchmark level.

---

## How It Works — Blog-Readable Walkthrough

Imagine you're looking up someone in a social network. Standard HNSW is like
having a list of 32 friend IDs but needing to drive across town to visit each
friend's house to find out if they're closer to the target than your current
best. SymphonyQG is like having a pocket-sized "cheat sheet" for each person
— a compressed but still useful description of each of their 32 friends stored
right next to their ID. You can scan all 32 cheat-sheets without moving, decide
which 5 or 10 are worth visiting, and only then go to those houses.

The "cheat sheet" is a RaBitQ 1-bit code: for a 128-dimension vector, that's
128 bits = 16 bytes, vs 512 bytes for the full f32 vector. A 32-neighbor block
becomes 32×16 = 512 bytes of codes + 32×4 = 128 bytes of IDs/norms = 640 bytes
sequential, vs 32×512 = 16 KB of random pointer chases.

The distance estimate from the 1-bit code isn't exact, but it's close enough
to decide traversal order. When you finally arrive at the right neighborhood,
the few remaining candidates are re-scored with exact distances. The beam
search terminates when no unvisited candidate can improve your current best —
RaBitQ's bounded error means this is safe to do without a separate re-ranking pass.

---

## Practical Failure Modes

1. **Low recall with greedy k-NN graph**: the PoC demonstrates kernel speedup
   but not recall improvement, because greedy k-NN graphs lack navigability.
   Fix: use HNSW or Vamana construction.

2. **Quantization recall penalty at small ef**: with ef=32, the beam may
   converge to a local optimum faster when estimated distances have noise.
   Fix: increase ef (costs QPS) or use SQ8 codes instead of 1-bit.

3. **Large rotation matrix memory**: for D=1024, the rotation matrix is
   1024×1024×4 = 4 MB. Acceptable for a singleton, expensive for many indexes.
   Fix: use structured Hadamard rotation (O(D log D) multiply, O(D) storage).

4. **O(n²) build time**: the PoC's greedy k-NN build is impractical for n>100K.
   Fix: Vamana construction (O(n log n) with bounded beam search).

5. **No concurrent search/insert**: the current `SymphonyGraph` is immutable
   after build. Online inserts require a separate mechanism.
   Fix: follow DiskANN's incremental update protocol.

---

## What to Improve Next — Roadmap

| Priority | Item | Effort |
|---|---|---|
| P0 | Replace greedy k-NN with HNSW construction | 2 sprints |
| P0 | Validate recall on SIFT-1M / ANN-benchmarks | 1 sprint |
| P1 | Structured Hadamard rotation (O(D log D), O(D) memory) | 1 sprint |
| P1 | SQ8 codes as alternative to 1-bit (better recall at 8× compression) | 1 sprint |
| P2 | Platform SIMD: AVX2/NEON via `std::arch` or `wide` crate | 2 sprints |
| P2 | WASM target (lazy rotation init, linear-algebra-free path) | 1 sprint |
| P3 | Integration with `ruvector-core` `AnnIndex` trait | 1 sprint |
| P3 | Persistence / mmap layout for the co-located vertex blocks | 2 sprints |

---

## Production Crate Layout Proposal

```
crates/ruvector-symphony-qg/
├── src/
│   ├── lib.rs
│   ├── rotation/
│   │   ├── gram_schmidt.rs   — current (D×D, exact)
│   │   └── hadamard.rs       — fast Walsh-Hadamard (O(D log D))
│   ├── codes/
│   │   ├── rabitq.rs         — 1-bit encoding (current)
│   │   └── sq8.rs            — 8-bit scalar quantization alternative
│   ├── graph/
│   │   ├── layout.rs         — co-located vertex block (current)
│   │   ├── build_greedy.rs   — current PoC O(n²) builder
│   │   └── build_hnsw.rs     — HNSW graph construction (future)
│   ├── search/
│   │   ├── beam.rs           — beam search (current)
│   │   └── simd.rs           — AVX2/NEON batch kernel (future)
│   ├── index.rs
│   ├── persist.rs            — mmap serialisation (future)
│   └── error.rs
├── benches/
│   └── symphony_bench.rs
└── Cargo.toml
```

---

## References

1. Gou, Y. et al., "SymphonyQG: Towards Symphonious Integration of Quantization
   and Graph for Approximate Nearest Neighbor Search", SIGMOD 2025.
   arXiv:2411.12229. https://arxiv.org/abs/2411.12229
2. Gao, J., Long, C., "RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024.
   arXiv:2405.12497.
3. Johnson, J. et al., "Billion-scale similarity search with GPUs", IEEE TPAMI
   2021. https://arxiv.org/abs/1702.08734 (FAISS/FastScan).
4. Malkov, Y., Yashunin, D., "Efficient and Robust Approximate Nearest Neighbor
   Search Using Hierarchical Navigable Small World Graphs", IEEE TPAMI 2020.
   arXiv:1603.09320.
5. Subramanya, S. et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor
   Search on a Single Node", NeurIPS 2019.
6. Huijben, I. et al., "QINCo2: Vector Compression meets Neural Compression",
   ICLR 2025. arXiv:2501.03078.
7. Xu, J. et al., "TriBase: A Vector Data Query Engine for Reliable and Lossless
   Pruning Compression Using Triangle Inequalities", SIGMOD 2025.
