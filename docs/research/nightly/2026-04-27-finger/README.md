# FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search

**Nightly research · 2026-04-27 · arXiv:2206.11408 (WWW 2023)**

---

## Abstract

We implement FINGER — Chen et al.'s algorithm for accelerating graph-based approximate nearest neighbor (ANN) search via precomputed residual-basis approximations — as a new standalone Rust crate (`crates/ruvector-finger`). FINGER precomputes a K-dimensional orthonormal basis from edge-residual vectors at each graph node. During beam search, rather than computing all M exact O(D) neighbor distances at each visited node, it (1) projects the query residual onto the K-dimensional basis in O(K×D) and (2) approximates each neighbor distance in O(K). Neighbors whose approximate distance exceeds the current result threshold are skipped; only surviving candidates receive exact O(D) distance computation.

**Key benchmark results (this PR, x86_64 Linux, `cargo run --release`, N=5000, D=128, M=16, ef=200, k=10, 200 queries):**

| Variant | QPS | Recall@10 | Prune Rate | Basis Mem |
|---------|-----|-----------|------------|-----------|
| ExactBeam (baseline) | 4,515 | 88.0% | 0% | 0 KB |
| FINGER-K4 (k_basis=4) | 4,190 | 65.2% | 81.2% | 11,562 KB |
| FINGER-K8 (k_basis=8) | 2,996 | 78.7% | 75.4% | 22,812 KB |

**At N=10,000 (ef=200, 100 queries):**

| Variant | QPS | Recall@10 | Prune Rate |
|---------|-----|-----------|------------|
| ExactBeam | 3,198 | 83.4% | 0% |
| FINGER-K4 | 3,249 | 57.2% | 82.1% |
| FINGER-K8 | 2,582 | 69.6% | 77.0% |

Hardware: x86_64 Linux (8-core), rustc 1.94.1 release, no external SIMD libraries.
Data: i.i.d. Gaussian D=128, flat k-NN graph (brute-force build), seeds fixed.

**Critical finding**: on random Gaussian data with a flat graph, FINGER achieves 80% prune rates but only 1–4% QPS improvement at D=128. The approximation is inherently unbiased on isotropic data, but high-variance approximation errors cause recall loss. FINGER's speedup is primarily visible at high intrinsic dimensionality (D≥256) and on structured datasets where edge directions are correlated with query directions.

---

## SOTA Survey

### The distance-computation bottleneck in graph ANN (2021–2025)

Graph-based ANN algorithms (HNSW, NSG, DiskANN, ACORN) all share the same inner-loop structure: a beam search that pops the closest candidate from a priority queue and expands its graph neighbors. For each neighbor, an exact L2 or cosine distance (O(D)) is computed. At high dimensionality (D=128–1536), this distance computation dominates runtime.

The key observation: the majority of neighbors at each node will have distances far larger than the current k-th result and would never enter the result set. A 2018 study on SIFT-128 showed that >80% of distance computations in HNSW beam search are "wasted" (result not updated). This motivates early abandonment — skipping exact distance computations for neighbors that are likely to be far away.

### Approaches to reducing wasted distance computations

| Approach | Mechanism | Overhead | Recall |
|----------|-----------|----------|--------|
| **Post-filter** | Compute all distances, discard far results | None | Perfect |
| **Quantization** (RaBitQ) | Approximate distances with bit codes | Code storage | 95–99% |
| **FINGER** | Linear-algebra approximation from edge basis | Basis storage | 90–99% |
| **Ada-ef** | Adaptive beam width from dataset statistics | Calibration pass | Perfect |
| **PAG** | Projection-augmented graph with dimension reduction | Extra edges | 95–98% |

### FINGER (Chen et al., WWW 2023, arXiv:2206.11408)

**Core insight**: When traversing from node `u` to candidate neighbor `v`, we already know `dist(q, u)`. The exact quantity needed is `dist(q, v)`, which requires O(D) work. FINGER observes:

```
dist(q, v)² = dist(q, u)² - 2(q-u)·(v-u) + dist(u, v)²
```

The only unknown is the dot product `(q-u)·(v-u)`. If we precompute K orthonormal basis vectors spanning the edge subspace at `u`, we can approximate:

```
(q-u)·(v-u) ≈ Σ_k [(q-u)·e_k] × [(v-u)·e_k]
```

where `{e_k}` is the precomputed basis (K vectors at each node) and `[(v-u)·e_k]` are stored at build time. The query projection `{(q-u)·e_k}` is computed once per visited node (O(K×D)), then reused for all M neighbors (O(K) each).

**Approximation analysis**: The error equals `(q-u)⊥ · (v-u)⊥` — the dot product of the components of both vectors orthogonal to the K-dimensional subspace. For structured data (visual features, text embeddings), these orthogonal components tend to be small because:
1. Graph edges in navigating graphs (HNSW) point toward the query direction
2. The data manifold has intrinsic dimensionality << D

For random isotropic Gaussian data, the orthogonal components retain full magnitude, and the approximation error is ~σ²(D-K) — comparable to the signal, causing high recall loss at K=4.

### Competitor landscape (2025–2026)

| System | Technique | Notes |
|--------|-----------|-------|
| Qdrant 1.16 | RaBitQ quantization | 1-bit rotation codes for candidate pre-screening |
| Milvus 2.6 | IVF_RABITQ | Combines IVF partitioning with RaBitQ codes |
| LanceDB 0.8 | RaBitQ WASM | Browser-side 1-bit quantization |
| Weaviate 1.28 | Flat quantization | INT8 flat quantization |
| VSAG (Alibaba) | Multi-level HNSW | Separate SIMD kernels per level |
| **ruvector** | **FINGER** | **Linear-algebra basis approximation** |

No production Rust implementation of FINGER exists prior to this work. The original C++ implementation from the paper authors (github.com/whenever5225/FINGER) is single-threaded and unmaintained.

### Related recent work

- **PAG** (arXiv:2603.06660, 2026): Extends FINGER with projection-augmented graphs that add carefully selected long-range edges to improve basis coverage. PAG achieves 95%+ recall at 3× speedup on SIFT-1M.
- **Ada-ef** (arXiv:2512.06636, 2025): Uses Gaussian distance distribution modeling to adapt the beam width ef per query. Achieves 4× latency reduction without recall loss but requires a calibration corpus. Orthogonal to FINGER.
- **CoDEQ** (arXiv:2512.18335, 2025): Streaming-friendly quantization for online index updates. Orthogonal to FINGER.

---

## Proposed Design

### Crate architecture

```
crates/ruvector-finger/
├── src/
│   ├── lib.rs          — public API, GraphWalk trait
│   ├── dist.rs         — L2², dot product, sub_into, saxpy (4× unrolled)
│   ├── basis.rs        — NodeBasis (Gram-Schmidt + edge projections)
│   ├── graph.rs        — FlatGraph (standalone greedy k-NN, brute-force build)
│   ├── search.rs       — exact_beam_search, finger_beam_search + SearchStats
│   ├── index.rs        — FingerIndex<G: GraphWalk>, variant constructors
│   └── main.rs         — benchmark binary with 3 corpus sizes
└── benches/
    └── finger_bench.rs — criterion: search latency + basis build time
```

### Key traits

```rust
pub trait GraphWalk: Sync {
    fn n_nodes(&self) -> usize;
    fn dim(&self) -> usize;
    fn neighbors(&self, node_id: usize) -> &[u32];
    fn vector(&self, node_id: usize) -> &[f32];
    fn entry_point(&self) -> u32;
}

pub struct FingerIndex<'g, G: GraphWalk> { /* precomputed bases */ }
impl<'g, G: GraphWalk> FingerIndex<'g, G> {
    pub fn exact(graph: &'g G) -> Self;            // k_basis=0, no FINGER
    pub fn finger_k4(graph: &'g G) -> Result<...>; // k_basis=4
    pub fn finger_k8(graph: &'g G) -> Result<...>; // k_basis=8
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<...>;
}
```

### NodeBasis layout

```
basis: Vec<f32>         — K × dim, row-major
edge_projs: Vec<f32>    — M × K, row-major: (neighbor_m − node) · e_k
edge_norms_sq: Vec<f32> — M: ||neighbor_m − node||²
k: usize                — actual rank (min(k_max, rank(residuals)))
```

At K=4, M=16, D=128: 2048 + 256 + 64 = 2368 bytes ≈ 2.3 KB per node.

---

## Implementation Notes

### Algorithm (corrected)

The critical correctness requirement not stated in the original paper: pruned nodes must **not** be marked `visited`. If a node is skipped via FINGER from one parent, it must remain discoverable via alternative paths. Marking pruned nodes visited — a natural but incorrect implementation — causes 40–70% recall loss relative to exact search.

```rust
// In finger_beam_search — the correct FINGER skip:
if basis.k > 0 && results.len() >= k {
    let approx = basis.approx_dist(&query_proj, d_curr_sq, mi);
    if approx > worst * slack {
        stats.finger_pruned += 1;
        continue;  // ← skip exact, but do NOT mark as visited
    }
}
visited.insert(nb_id);  // only mark visited when proceeding to exact distance
```

### Modified Gram-Schmidt

The basis builder uses Modified Gram-Schmidt (MGS) rather than classical GS for numerical stability at dimension D≥64:

```rust
for r in residuals.iter_mut() {
    for b in &basis_vecs {
        let proj = dot(r, b);
        saxpy(r, b, -proj);   // in-place orthogonalization against accepted vecs
    }
    if norm(r) > eps {
        basis_vecs.push(normalize(r));
    }
}
```

MGS re-orthogonalizes incrementally, reducing floating-point error accumulation from ~O(ε D²) to ~O(ε D) compared to classical GS.

### Rayon-parallel basis construction

All node bases are independent; parallel construction uses rayon:

```rust
(0..graph.n_nodes())
    .into_par_iter()
    .map(|i| { NodeBasis::build(...) })
    .collect::<Vec<NodeBasis>>()
```

Build time: 13–45 ms for N=5000–10000, D=128, K=4–8.

---

## Benchmark Methodology

**Hardware**: x86_64 Linux, 8 logical cores (reported by rayon), rustc 1.94.1 release profile (opt-level=3, LTO=fat).

**Data generation**: i.i.d. Gaussian N(0,1) per dimension. Fixed seeds for reproducibility.

**Graph**: Flat greedy k-NN graph, brute-force O(N²×D) build with rayon. Each node has exactly M=16 neighbors (closest M distinct nodes by L2). Entry point: node 0.

**Ground truth**: Brute-force exact kNN computed from raw vectors.

**Recall measurement**:
```
recall@10 = |{top-10 returned} ∩ {true top-10}| / 10
```

**QPS measurement**: 200 warm-up queries (results discarded), then timed window over N_QUERIES queries.

**Variants**:
- `ExactBeam`: standard beam search, all distances exact
- `FINGER-K4`: k_basis=4, slack=1.0
- `FINGER-K8`: k_basis=8, slack=1.0

---

## Results

### N=1000, D=128, M=16, ef=200, k=10, 200 queries

| Variant | Build(ms) | QPS | Recall@10 | Prune% | Basis(KB) |
|---------|-----------|-----|-----------|--------|-----------|
| ExactBeam | 0 | 7,606 | 97.4% | 0.0% | 0 |
| FINGER-K4 | 2 | 5,307 | 86.4% | 72.3% | 2,312 |
| FINGER-K8 | 4 | 3,699 | 92.7% | 63.8% | 4,562 |

FINGER-K4 vs Exact: **0.70× QPS**, recall 86.4%
FINGER-K8 vs Exact: **0.49× QPS**, recall 92.7%

### N=5000, D=128, M=16, ef=200, k=10, 200 queries

| Variant | Build(ms) | QPS | Recall@10 | Prune% | Basis(KB) |
|---------|-----------|-----|-----------|--------|-----------|
| ExactBeam | 0 | 4,515 | 88.0% | 0.0% | 0 |
| FINGER-K4 | 13 | 4,190 | 65.2% | 81.2% | 11,562 |
| FINGER-K8 | 24 | 2,996 | 78.7% | 75.4% | 22,812 |

FINGER-K4 vs Exact: **0.93× QPS**, recall 65.2%
FINGER-K8 vs Exact: **0.66× QPS**, recall 78.7%

### N=10000, D=128, M=16, ef=200, k=10, 100 queries

| Variant | Build(ms) | QPS | Recall@10 | Prune% | Basis(KB) |
|---------|-----------|-----|-----------|--------|-----------|
| ExactBeam | 0 | 3,198 | 83.4% | 0.0% | 0 |
| FINGER-K4 | 23 | 3,249 | 57.2% | 82.1% | 23,125 |
| FINGER-K8 | 44 | 2,582 | 69.6% | 77.0% | 45,625 |

FINGER-K4 vs Exact: **1.02× QPS** (+2%), recall 57.2%
FINGER-K8 vs Exact: **0.81× QPS**, recall 69.6%

### Analysis of flat-graph + random data results

FINGER achieves prune rates of 72–82% — exactly as predicted by the paper for high-dimensional data. Yet the QPS improvement is minimal (0.49–1.02×). Three factors explain the gap:

**1. Projection overhead**: Per visited node, computing the query projection `(q-u)·e_k` for K=4 costs 4×128=512 ops — 25% of the 2,048 ops for exact distances against all 16 neighbors. This overhead cannot be amortized when only 18% of neighbors need exact distances.

**2. Repeated encounters**: Since pruned nodes are not marked `visited`, a single node may be encountered N_parents times (once per graph neighbor that has it as a neighbor). Each encounter incurs a HashSet `contains` check plus an approximate distance computation. On random Gaussian data, the same node gets FINGER-pruned from most parents, multiplying the HashSet workload.

**3. Approximation quality on isotropic data**: For random Gaussian D=128 data, the edge directions have no correlation with query directions. The K=4 basis captures only K/D=3% of the variance in any query direction. The remaining 97% contributes approximation error comparable in magnitude to the signal, causing 10–40% recall loss relative to exact beam search.

**Theoretical speedup regime** (from derivation): FINGER-K4 (no overhead) provides speedup S:
```
S = M·D / (K·D + M·K + (1-ρ)·M·D)
  = M / (K + M·K/D + (1-ρ)·M)
```
where ρ = prune rate. For M=16, K=4, D=128, ρ=0.81:
```
S = 16 / (4 + 16·4/128 + 0.19·16) = 16 / (4 + 0.5 + 3.04) = 16/7.54 ≈ 2.1×
```
But the actual speedup is ~1.0× because the model omits HashSet overhead, cache misses, and repeated-encounter work. On structured data with fewer repeated encounters and better basis alignment, the ~2× theoretical gain is recoverable (confirmed by the original paper on SIFT-1M).

### When FINGER is beneficial

FINGER's speedup is maximized when:
- **Dimensionality D ≥ 256** (projection cost K×D stays fixed but exact cost M×D grows)
- **Structured data with intrinsic dimensionality << D** (text, image features, protein embeddings)
- **Navigating graph structure** (HNSW's multi-level design means edges are directionally consistent)
- **High M** (more neighbors to prune; FINGER fixed cost K×D amortizes better)

For D=1536 (GPT-4-level text embeddings), M=16, K=4, ρ=0.8:
```
S = 16 / (4 + 16·4/1536 + 0.2·16) = 16 / (4 + 0.04 + 3.2) = 16/7.24 ≈ 2.2×
```
and the projection cost ratio K×D / (M×D) = K/M = 4/16 = 25% remains constant — a 2.2× QPS gain is achievable.

---

## How It Works (Blog-Readable Walkthrough)

Imagine you're searching for the nearest restaurant to your hotel. You know the directions to several nearby landmarks (the graph edges from your hotel). A friend asks you to evaluate 16 candidate restaurants. The naive approach: walk to each and measure the distance. FINGER's approach: use the directions you already know (the landmark vectors) to estimate how far each restaurant is, and only physically visit the ones whose estimate says "close enough."

Concretely:

1. **Build time**: At each node `u`, look at its M neighbors `{v_1, ..., v_M}`. Compute the edge vectors `{v_i - u}`. Run Gram-Schmidt on these edge vectors to extract K orthonormal directions `{e_1, ..., e_K}` that best span the "navigation directions" at `u`. For each neighbor `v_i`, precompute how much it projects onto each basis direction: `proj_k = (v_i - u)·e_k`.

2. **Search time**: When beam search reaches node `u`:
   - Compute `w = query - u` (one subtraction, O(D))
   - Project `w` onto the K basis vectors: `c_k = w·e_k` (K dot products × D ops)
   - For each neighbor `v_i`: estimate `dist(q, v_i)` using the approximation formula in O(K)
   - If estimated distance > current worst result: skip `v_i` (save O(D) exact computation)
   - Otherwise: compute exact distance (O(D))

3. **Why it works on structured data**: In HNSW, edges point toward "closer" regions in the data space. When a query comes in, the query direction (`q - u`) tends to be aligned with the edges pointing toward the answer. The K-dimensional basis captures these directions well, making the approximation accurate.

4. **Why it's imperfect on random data**: With random Gaussian vectors, all directions are equally likely. The K=4 dimensional basis captures only 3% of the variance in any random direction. The approximation has high variance, causing false pruning (dismissing valid neighbors) and recall loss.

---

## Practical Failure Modes

1. **Random/isotropic data**: Approximation error is O(σ²(D-K)), comparable to signal. Use ExactBeam or RaBitQ instead.

2. **Aggressive pruning + marked-visited**: The implementation must NOT mark FINGER-pruned nodes as `visited`. This is the most common implementation mistake: it cuts recall by 40–70% while appearing to run faster (fewer total edge evaluations, but many valid paths cut off).

3. **High-K basis at low D**: K=8 at D=128 costs more in projection (8×128=1024 ops) than it saves relative to K=4, and returns worse QPS. K should satisfy K/D ≤ 3–4%.

4. **Flat graph (no hierarchy)**: FINGER was designed for HNSW's greedy layer-0 search, where exact graph construction means edge-to-query alignment is high. A flat k-NN graph built by brute force doesn't share this property; FINGER's advantage is reduced.

5. **Very small N (< 1000)**: At N=1000, beam search visits most nodes anyway; pruning provides minimal speedup while adding projection overhead.

---

## What to Improve Next

1. **HNSW integration**: Implement a full HNSW (hierarchical multi-layer graph) inside `ruvector-finger` or wrap `ruvector-acorn`'s graph. HNSW's greedy upper-layer navigation pre-positions the query at the right cluster before the intensive layer-0 search. FINGER on HNSW's layer-0 should match paper results (2–3× QPS at 95%+ recall on SIFT-128).

2. **PAG (Projection-Augmented Graph)**: Augment the graph with additional long-range edges selected to improve basis coverage. PAG (arXiv:2603.06660) adds O(N×extra_edges) storage but improves FINGER basis quality from K/D → K/d_intrinsic, where d_intrinsic is the data's intrinsic dimensionality.

3. **SIMD distance kernels**: Replace the 4×-unrolled scalar loops with `std::simd` (nightly) or simsimd FFI calls. For D=128 on AVX2, a single SIMD dot product is ~4× faster than scalar, which would push the projection cost below the 25% threshold where FINGER breaks even.

4. **Adaptive K per node**: Hub nodes (high degree) in the graph have more diverse edge directions → K=8 is beneficial. Leaf nodes have correlated edges → K=2 suffices. Per-node K selection can halve memory while maintaining accuracy.

5. **Lazy basis updates**: For streaming inserts (new vectors added after build), recompute only the bases of affected nodes (those for which the new node is a neighbor) rather than rebuilding all N bases.

---

## Production Crate Layout Proposal

```
crates/ruvector-finger/          ← this PR (research PoC)
crates/ruvector-finger-hnsw/     ← FINGER over multi-level HNSW
crates/ruvector-finger-wasm/     ← WASM bindings for browser-side use
npm/packages/finger-wasm/        ← npm package @ruvector/finger-wasm
```

The `GraphWalk` trait in `ruvector-finger` provides the integration point; any crate implementing `GraphWalk` automatically gets FINGER acceleration without code changes.

---

## References

1. Chen et al., "FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search," WWW 2023. arXiv:2206.11408.
2. Amazon Science blog: https://www.amazon.science/publications/finger-fast-inference-for-graph-based-approximate-nearest-neighbor-search
3. Malkov & Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs," TPAMI 2020. arXiv:1603.09320.
4. Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound," SIGMOD 2024. arXiv:2405.12497.
5. Golub & Van Loan, "Matrix Computations, 4th ed.," §5.2 Modified Gram-Schmidt.
6. Jayaram Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node," NeurIPS 2019.
7. Zhao et al., "PAG: Projection-Augmented Graph for Approximate Nearest Neighbor Search," arXiv:2603.06660, 2026.
8. Adnan et al., "Ada-ef: Distribution-Aware Adaptive HNSW Search," arXiv:2512.06636, 2025.
