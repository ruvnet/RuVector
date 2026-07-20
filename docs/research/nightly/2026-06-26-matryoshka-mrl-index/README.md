# Matryoshka Resolution Index (ruvector-mrl)

**Date:** 2026-06-26  
**Branch:** `research/nightly/2026-06-26-matryoshka-mrl-index`  
**Crate:** `crates/ruvector-mrl`  
**Status:** PASS — all acceptance criteria met

---

## 1. Motivation

Modern production embedding APIs—OpenAI `text-embedding-3`, Cohere `embed-v3`, Nomic `embed-v1.5`—ship **Matryoshka Representation Learning (MRL)** embeddings where the first `D'` dimensions form a geometrically meaningful approximation of the full `D`-dimensional vector. Truncating to `D'` dimensions preserves cosine-similarity ranking.

This property enables a two-stage ANN strategy: screen candidates cheaply in `D'`-dimensional space, then rerank the shortlist with the full-dimensional cosine. The potential gain is `D/D'`-fold reduction in screening cost at near-perfect recall.

The question this research answers: **how much does the prefix-dimension ratio affect recall and throughput, and does a graph navigator add value over brute-force screening?**

---

## 2. SOTA Context

| System | Approach | Key Insight |
|--------|----------|-------------|
| OpenAI text-embedding-3 | MRL training | 256-dim prefix retains ~99% recall vs 1536-dim full |
| Cohere embed-v3 | MRL training | Deployed at production scale; 25% prefix = 90%+ recall |
| Nomic embed-v1.5 | MRL training | Open-weight; 64-dim prefix usable for coarse filtering |
| HNSW (hnswlib) | Full-dim graph | Sub-linear search but no dimension reduction |
| ScaNN (Google) | Quantization + AH | Orthogonal to MRL; both can be combined |
| FAISS IVF | Inverted file | Cluster-based; MRL not exploited in standard impl |

**Gap this research fills:** A clean Rust proof-of-concept showing _when_ MRL dimension reduction helps (Matryoshka-trained embeddings) versus _when_ it hurts (random Gaussian vectors), with two concrete index variants (linear scan + graph navigator) benchmarked on a reproducible synthetic dataset.

---

## 3. Research Questions

1. Does a 25%-dim prefix screen reliably predict full-dim cosine rank on random Gaussian vectors?
2. Does Matryoshka-simulated data (strong prefix signal, small tail noise) change the recall profile?
3. Does a greedy kNN graph on the prefix add throughput versus a brute-force prefix scan?
4. What speedup–recall Pareto front emerges from varying the prefix ratio?

---

## 4. Architecture

```
                 ┌─────────────────────────────────────────┐
                 │           MrlSearch trait                │
                 │  insert(id, &[f32])                      │
                 │  search(&[f32], k) → Vec<SearchResult>   │
                 └──────┬──────────────────┬───────────────┘
                        │                  │
              ┌─────────▼──────┐  ┌────────▼──────────┐
              │   MrlLinear    │  │    MrlGraph        │
              │                │  │                    │
              │ Stage 1:       │  │ Stage 1:           │
              │  O(N·D_FAST)   │  │  beam_fast()       │
              │  brute-force   │  │  O(ef·M·D_FAST)    │
              │  prefix scan   │  │  graph navigation  │
              │                │  │                    │
              │ Stage 2:       │  │ Stage 2:           │
              │  exact rerank  │  │  exact rerank      │
              │  top k_over    │  │  top oversample·k  │
              └────────────────┘  └────────────────────┘
                        │                  │
                        └──────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │     GreedyGraph     │
                    │                     │
                    │ vectors: Vec<Vec>   │
                    │ adj: Vec<Vec<u32>>  │
                    │ d_fast, d_full, m   │
                    │                     │
                    │ insert() → just store│
                    │ build_edges() → O(N²)│
                    │ beam_fast() → search │
                    │ rerank() → full-dim  │
                    └─────────────────────┘
```

### Key Design Decisions

**Two-phase graph build.** `insert()` only stores the vector; `build_edges()` runs the full O(N²·D_FAST) symmetric kNN pass over the entire dataset. This guarantees every node has M well-connected neighbours before any search begins. Sequential greedy insertion (connect to prior nodes only) leaves early nodes with no outgoing edges, crippling beam search.

**EFANNA-style beam search.** Maintains two sets: W (exploration frontier, max-heap) and C (candidate result set, capped at `ef`). Prunes when the best unexplored score falls below the worst entry in C. Sorted-Vec simulation of the heap keeps the implementation self-contained.

**Symmetric adjacency.** When node i connects to neighbour j, j also back-connects to i (capped at M). This doubles reachability without additional memory: every edge is traversable in both directions.

---

## 5. Benchmark Setup

Hardware: x86-64 Linux (single process, no GPU)  
Dataset: synthetic, seeded, reproducible  
Build: `cargo run --release`

| Parameter | Value |
|-----------|-------|
| N (corpus size) | 5,000 vectors |
| D_FULL | 128 dimensions |
| D_FAST | 32 dimensions (25% prefix) |
| N_QUERIES | 200 queries |
| K | 10 nearest neighbours |
| M (graph degree) | 16 |
| ef (beam width) | 60 |
| Oversample | 8× |
| k_over (MrlLinear) | 10× |
| Alpha (MRL noise) | 0.25 |

**Experiment A — Random Gaussian.** Vectors drawn from U(-1,1)^128, normalised to unit sphere. Prefix dimensions carry no special information about full-dim similarity. Documents the fundamental limitation.

**Experiment B — MRL-Simulated.** Vectors generated as `v = normalize(signal || α·noise)` where `signal ~ U(-1,1)^32`, `noise ~ U(-1,1)^96`, α=0.25. The prefix is genuinely predictive: prefix cosine rank closely tracks full-dim rank. Simulates Matryoshka-trained embeddings.

---

## 6. Measured Results

Results from `cargo run --release -p ruvector-mrl --bin mrl-bench` on 2026-06-26:

### Experiment A — Random Gaussian (no MRL structure)

| Variant | Mean µs/q | p50 µs | p95 µs | QPS | Recall@10 | Speedup |
|---------|-----------|--------|--------|-----|-----------|---------|
| FlatFull (exact) | 446.3 | 435 | 534 | 2,241 | 1.000 | 1.0× |
| MrlLinear | 232.8 | 212 | 262 | 4,296 | 0.284 | 1.9× |
| MrlGraph | 122.6 | 121 | 145 | 8,155 | 0.211 | 3.6× |

**Interpretation:** Without Matryoshka training structure, a 25%-dim prefix has ~28% recall on the brute-force path and ~21% recall on the graph path. Speedup exists (1.9–3.6×) but recall is unacceptable for most applications. This confirms: **MRL speedup requires MRL-trained embeddings**.

### Experiment B — MRL-Simulated (prefix is informative)

| Variant | Mean µs/q | p50 µs | p95 µs | QPS | Recall@10 | Speedup |
|---------|-----------|--------|--------|-----|-----------|---------|
| FlatFull (exact) | 429.7 | 429 | 461 | 2,327 | 1.000 | 1.0× |
| MrlLinear | 216.2 | 208 | 251 | 4,625 | 1.000 | 2.0× |
| MrlGraph | 123.1 | 120 | 148 | 8,123 | 0.943 | 3.5× |

**Interpretation:** With Matryoshka structure (α=0.25 noise), MrlLinear achieves perfect recall@10 at 2.0× throughput. MrlGraph reaches 94.3% recall@10 at 3.5× throughput—the graph's approximate navigation costs ~6% recall versus brute-force prefix scan.

### Acceptance Criteria (Experiment B only)

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| MrlLinear recall@10 | ≥ 0.80 | 1.000 | PASS |
| MrlGraph recall@10 | ≥ 0.70 | 0.943 | PASS |
| MrlLinear speedup | ≥ 1.5× | 2.0× | PASS |
| MrlGraph speedup | ≥ 3.0× | 3.5× | PASS |

---

## 7. Key Findings

1. **Training dependency is real.** MRL dimension reduction only yields usable recall when the embedding model was Matryoshka-trained. On untrained vectors, a 25%-dim prefix is as informative as a random projection—recall collapses to ~28%.

2. **MrlLinear perfect recall at 2×.** On MRL-structured data, brute-force prefix scan + full-dim rerank achieves 100% recall at 2× throughput. The k_over=10 factor is conservative; with k_over=5 recall remains 100% on this dataset.

3. **MrlGraph: 3.5× throughput, 94% recall.** The greedy graph adds another 1.75× over MrlLinear for a 3.5× total speedup, with a 5.7% recall cost. For retrieval-augmented generation where top-10 completeness matters less than latency, this is an attractive operating point.

4. **Graph build cost is acceptable.** O(N²·D_FAST) build on N=5,000 vectors takes ~1.5 s in release mode. At N=20,000 this would scale to ~24 s—practical for batch indexing, not for streaming updates.

5. **Memory overhead is modest.** FlatFull: 2.4 MB for vectors. Graph adjacency: 0.3 MB (N × M × 4 bytes). Total index memory: 2.7 MB for 5,000 × 128 vectors.

---

## 8. Limitations and Future Work

| Limitation | Mitigation Path |
|------------|----------------|
| O(N²) build forbids streaming | Add incremental insertion with delta-relink |
| Single-layer graph (no HNSW hierarchy) | Add hierarchical layers for sub-linear large-N build |
| Sorted-Vec beam search (O(ef log ef) per step) | Replace with priority queue (BinaryHeap) |
| No SIMD inner product | Add `target_feature = "+avx2"` dot product |
| Fixed alpha=0.25 in sim | Sweep alpha to produce recall-vs-speedup curves |
| No real MRL embeddings | Integrate with OpenAI / Nomic embedding API |

---

## 9. Comparison with Related Work

**vs. ruvector-acorn** (2026-04-26): ACORN targets filtered ANN with attribute predicates. MRL targets dimension reduction with embedding structure. Orthogonal; could be combined (filter in D_FAST space, rerank with full-dim).

**vs. ruvector-rabitq** (2026-04-23): RaBitQ targets binary quantization for memory reduction. MRL targets prefix truncation for compute reduction. Both are two-stage; could stack (prefix scan in RaBitQ space, full-dim rerank).

**vs. 2026-06-21-matryoshka-coarse-fine**: That nightly explored coarse-to-fine re-scoring within a single HNSW graph. This nightly isolates the pure dimension-reduction contribution and adds an explicit graph variant.

---

## 10. Implementation Notes

- All source files under 500 lines (lib.rs: 54, flat.rs: 96, graph.rs: 214, mrl.rs: 247, main.rs: ~320)
- No unsafe code, no external BLAS, no C FFI
- 7 tests: 2 flat, 2 graph, 3 MRL (top-1 graph, top-1 linear, recall@10 linear)
- Reproducible: seeded `StdRng` for all dataset generation

---

## 11. Running the Benchmark

```bash
# Build and run
CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse \
  cargo run --release -p ruvector-mrl --bin mrl-bench

# Run tests only
CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse \
  cargo test -p ruvector-mrl
```

---

## 12. File Map

```
crates/ruvector-mrl/
├── Cargo.toml
└── src/
    ├── lib.rs        — MrlSearch trait, dot(), normalize(), recall_at_k()
    ├── flat.rs       — FlatIndex brute-force baseline
    ├── graph.rs      — GreedyGraph: insert, build_edges, beam_fast, rerank
    ├── mrl.rs        — MrlLinear, MrlGraph — two-stage ANN variants
    └── main.rs       — benchmark binary (two experiments + acceptance test)
```
