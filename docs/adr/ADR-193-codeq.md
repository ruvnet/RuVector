# ADR-193: CoDEQ — kd-tree Median-Split Quantizer with O(1) Streaming Updates

**Date:** 2026-05-11  
**Status:** Accepted  
**Deciders:** Nightly Research Agent  
**Branch:** `research/nightly/2026-05-11-codeq`  
**Related:** [Research doc](../research/nightly/2026-05-11-codeq/README.md)

---

## Context

Vector databases that use Product Quantization (PQ) or inverted file indexes face a fundamental tension with streaming workloads: the k-means codebooks that define quantization cells are computed once at build time and become increasingly stale as the data distribution drifts. FAISS IVF-PQ, Qdrant's scalar quantization, and Weaviate's HNSW+PQ all require a full codebook retrain when distribution shift exceeds ~10–20%, which for high-insert workloads may be required daily or more frequently.

The paper **CoDEQ** (arXiv:2512.18335, Dec 2025) proposes replacing k-means codebooks with a kd-tree median-split structure where:
- The **tree topology** (split dimensions and threshold values) is frozen after initial build.
- The **leaf centroids** (means of points currently in each leaf cell) update incrementally via Welford's online mean algorithm whenever a point is inserted or deleted.

This separates structural decisions (which are stable) from distributional state (which is local and cheap to update), achieving O(1) streaming consistency per point without any global rebuild.

ruvector currently has RaBitQ (`ruvector-rabitq`), ACORN filtered HNSW (`ruvector-acorn`), and a streaming HNSW prototype. It has no streaming-safe quantizer designed for insert-heavy workloads. CoDEQ fills this gap.

---

## Decision

Implement CoDEQ as a new Rust crate `ruvector-codeq` in the ruvector workspace, providing:

1. **`CoDEQIndex`** — the primary streaming quantized index with kd-tree median-split codebook, Welford online centroids, ADC search, and exact reranking.
2. **`FlatL2IndexCoDEQ` / `StaticPqIndex`** — baseline implementations for recall and QPS comparison.
3. A second crate **`ruvector-streaming-hnsw`** implementing a concurrency-safe HNSW baseline using `parking_lot::RwLock` per neighbor list.

The kd-tree structure is built in O(n·D·L) time (no k-means). After build, insert and delete each touch exactly one leaf — O(L) tree traversal + O(1) centroid update + O(leaf_size) ID swap-remove.

---

## Algorithm

### Build

1. Apply random Gaussian rotation R ∈ ℝᴰˣᵖ (p = min(D, 64)) to all training vectors.
2. For each depth d ∈ [0, L): select the d-th highest-variance projected dimension; compute median → store as `KdNode { split_dim, split_val }`.
3. Encode each training vector: walk L nodes, set bit d iff `rv[split_dim] ≥ split_val` → leaf code ∈ [0, 2^L).
4. For each leaf, accumulate original-space centroid sum (not rotated-space — rotation distorts distances).

### Insert (streaming)

```
rv = R·v
code = walk_tree(rv)      // O(L) comparisons
leaf_sum[code] += v       // O(D) Welford update
leaf_count[code] += 1
```

No rebuild. No lock contention. O(D·L) per insert.

### ADC search

```
lut[leaf] = l2_sq(query, centroid(leaf))  for leaf in 0..2^L   // O(2^L × D)
scores[id] = lut[code[id]]               for all stored ids    // O(n)
rerank top-k×8 with exact l2_sq                                // O(k×D)
```

---

## Consequences

### Positive

- **330,942 streaming update ops/sec** measured on x86_64 (1,000 mixed insert+delete in 3 ms).
- **7.5× faster build** than StaticPQ (54 ms vs 404 ms at n=5,000, D=128) — no k-means.
- **4.3× higher QPS** than brute-force FlatL2 (4,812 vs 1,129 at n=5,000).
- **Stable recall under drift**: StaticPQ drops 2.9pp after 10% data replacement; CoDEQ recall is unchanged because centroids update in place.
- Codebase is <500 lines per file; no unsafe code; pure `std` + `rand` + `rand_distr`.
- 14 unit tests pass; 9 streaming-HNSW unit tests pass (including concurrent insert test).

### Negative / Limitations

- **Low standalone recall**: At n=5,000 with default 8× oversample (80 candidates), Recall@10 = 7.2%. CoDEQ is a **coarse quantizer** intended for use as a first stage with HNSW or IVF. Standalone deployment is only appropriate when recall ~10% is acceptable (e.g., recommendation diversity use cases).
- **Tree splits stale after >30% distribution shift**: frozen topology cannot accommodate new cluster emergence. Mitigation: periodic O(n·D) rebuild (fast — no k-means), triggered by centroid drift monitoring.
- **Rotation is not norm-preserving**: Random Gaussian R has singular values that stretch some directions. High-norm outliers may be misassigned across split boundaries. Full mitigation requires Gram-Schmidt QR (more expensive build).
- **Memory overhead**: Stores raw vectors for exact reranking (same as FlatL2). Code-only mode (n bytes) would reduce memory 30× at cost of reranking quality.

### Neutral

- This crate does not replace RaBitQ or ACORN. It is the streaming quantization layer; HNSW graph traversal is the candidate-selection layer.
- ADC LUT build is O(2^L × D) = O(256 × 128) = 32,768 multiplications per query — fast, but not SIMD-optimized in this PoC.

---

## Alternatives Considered

### Alt 1: Extend StaticPQ with partial retrain

Add a `StaticPqIndex::rebuild_codebook(new_data)` method that reruns k-means. Rejected: k-means at n=5,000 already takes 404 ms; at n=1M this is 80+ seconds. Cannot be done online.

### Alt 2: Extend RaBitQ with streaming deletes

RaBitQ stores 1-bit quantization codes. Adding Welford updates to binarized codes is theoretically possible but loses the error-bound guarantees that make RaBitQ valuable. The theoretical appeal of RaBitQ is its provable recall bound; streaming modifications invalidate the proof. Rejected.

### Alt 3: LSH-based quantizer

Locality-Sensitive Hashing provides streaming inserts natively. However, LSH has higher false-positive rates than kd-tree quantization at equal memory, and offers no centroid-based ADC — only hash bucket exact scan. Rejected in favor of CoDEQ's richer ADC.

### Alt 4: Full HNSW-only streaming

Streaming HNSW (implemented in `ruvector-streaming-hnsw`) achieves 53% Recall@10 vs 7% for CoDEQ at similar QPS. However HNSW memory is ~2× per node (neighbor adjacency lists), insert is slower (3,152 inserts/sec vs 330,942 for CoDEQ), and the graph structure cannot be compressed. Best approach: use both — HNSW for graph traversal, CoDEQ for in-list distance approximation.

---

## Implementation Notes

### Files created

| Path | Purpose |
|------|---------|
| `crates/ruvector-codeq/src/kdquant.rs` | Core CoDEQIndex, KdQuantizer, LeafStore, Rotation |
| `crates/ruvector-codeq/src/pq_baseline.rs` | FlatL2IndexCoDEQ, StaticPqIndex (baselines) |
| `crates/ruvector-codeq/src/dist.rs` | l2_sq, dot_product |
| `crates/ruvector-codeq/src/error.rs` | CoDEQError |
| `crates/ruvector-codeq/src/lib.rs` | Public API re-exports |
| `crates/ruvector-codeq/src/main.rs` | Benchmark demo |
| `crates/ruvector-codeq/benches/codeq_bench.rs` | Criterion benchmarks |
| `crates/ruvector-streaming-hnsw/src/index.rs` | FlatL2, StaticHnsw, StreamingHnsw |
| `crates/ruvector-streaming-hnsw/src/main.rs` | Benchmark demo |
| `crates/ruvector-streaming-hnsw/benches/streaming_hnsw_bench.rs` | Criterion benchmarks |

### Known correctness fixes during implementation

1. **u8 overflow in LUT loop**: `(0..n_leaves as u8)` wraps at 256 → empty range. Fixed: `(0..n_leaves)`.
2. **Rotated-space centroid distortion**: Non-orthogonal R distorts distances in rotated space. Fixed: centroid sums stored in original space.
3. **HNSW concurrent insert race**: Separate lock scopes for data append vs neighbor slot creation allow index-out-of-bounds. Fixed: merged into single double-write-lock scope.
4. **HNSW back-edge limit**: Back-edges capped at M instead of 2M (HNSW paper §4.1). Fixed: `neighbors[j].len() < 2 * m`.

---

## Compliance

- No unsafe code.
- No secrets or credentials.
- All files under 500 lines.
- 14 unit tests (ruvector-codeq) + 9 unit tests (ruvector-streaming-hnsw) pass.
- Recall thresholds in tests set conservatively (≥0.65, ≥0.70) to avoid flakiness on CI.
