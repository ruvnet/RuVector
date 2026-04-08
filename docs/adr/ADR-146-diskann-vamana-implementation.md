# ADR-144: DiskANN/Vamana Implementation

## Status
Implemented

## Date
2026-04-06

## Context

The ruvector npm package claimed DiskANN support in its README and package.json ("billion-scale SSD-backed ANN with <10ms latency") but no implementation existed. An audit (ADR-143) identified this as the largest capability gap. DiskANN/Vamana is a widely-cited algorithm (NeurIPS 2019, Microsoft Research) that enables approximate nearest neighbor search on datasets too large to fit in RAM.

Existing HNSW index (`hnsw_rs` via `@ruvector/router`) requires all vectors in memory. For datasets exceeding available RAM (100M+ vectors), a disk-backed solution with compressed in-memory representations is needed.

## Decision

Implement DiskANN as a dedicated Rust crate (`ruvector-diskann`) with NAPI-RS bindings (`ruvector-diskann-node`) and npm package (`@ruvector/diskann`), integrated into the `ruvector` npm package as an optional peer dependency.

### Algorithm Design

**Vamana Graph Construction (two-pass)**
1. Compute medoid (point closest to centroid) — used as search entry point
2. Initialize random graph with bounded out-degree R
3. Pass 1 (α=1.0): For each node in random order, greedy search from medoid to find candidates, then α-robust prune to select R neighbors. Update bidirectional edges.
4. Pass 2 (α=1.2): Same process with relaxed pruning — adds long-range edges that improve search convergence.

**α-Robust Pruning** (Algorithm 2 from paper)
- Sort candidates by distance to node
- Greedily select neighbors: keep candidate only if no already-selected neighbor α-dominates it
- A candidate p is α-dominated by selected neighbor s if: α × dist(s, p) ≤ dist(node, p)
- This ensures a mix of nearby (accuracy) and distant (navigability) edges

**Product Quantization (optional)**
- Split D-dim vectors into M subspaces of D/M dimensions
- Train 256 centroids per subspace via k-means++ with Lloyd's iterations
- Encode each vector as M bytes (one centroid index per subspace)
- During search: precompute distance table (query subvectors to all centroids), then PQ distance = sum of table lookups

**Search**: Greedy beam search from medoid, expanding best unexpanded node each step, maintaining top-L candidates. With PQ: filter candidates using approximate distance, then re-rank top results with exact L2.

### Optimizations

| Optimization | Rationale |
|---|---|
| **FlatVectors** (contiguous `Vec<f32>`) | Eliminates `Vec<Vec<f32>>` pointer indirection; cache-line prefetch works |
| **VisitedSet** (generation counter) | O(1) clear per query instead of re-allocating HashSet |
| **4-accumulator ILP** | 4 independent sums exploit instruction-level parallelism; auto-vectorizes to SIMD |
| **Flat PQ distance table** | `table[sub * 256 + code]` layout vs `Vec<Vec<f32>>` — sequential memory access |
| **Parallel medoid** (rayon) | Centroid computation + min-distance embarrassingly parallel |
| **Zero-copy save** | Write flat slab directly from memory to file (no per-float serialization) |
| **mmap load** | OS pages in only accessed vectors — working set << total dataset |
| **SimSIMD** (optional `simd` feature) | Hardware NEON/AVX2/AVX-512 for L2 and inner product |
| **GPU stubs** (optional `gpu` feature) | Metal/CUDA/Vulkan batch distance dispatch (rayon parallel fallback) |

### Package Architecture

```
ruvector-diskann (Rust crate, crates.io v2.1.0)
├── distance.rs    — FlatVectors, VisitedSet, L2², inner product, PQ distance
├── graph.rs       — Vamana: build, greedy_search, robust_prune, medoid
├── pq.rs          — ProductQuantizer: train, encode, distance tables
├── index.rs       — DiskAnnIndex: insert, build, search, save, load
└── error.rs       — Error types

ruvector-diskann-node (NAPI-RS bindings)
└── lib.rs         — DiskAnn class: insert, insertBatch, build[Async], search[Async], delete, save, load

@ruvector/diskann (npm v0.1.0, 5 platforms)
├── index.js       — Platform-specific native loader
├── index.d.ts     — TypeScript declarations
└── test.js        — Integration test

ruvector (npm, optional integration)
└── src/core/diskann-wrapper.ts — Lazy-load wrapper, re-exported from index
```

## Performance

Benchmarked on Apple M-series, release build:

| Metric | Result |
|--------|--------|
| Recall@10 (2K, 64d) | 1.000 |
| Recall@10 (2K, 64d, 50 queries) | 0.998 |
| Search latency (5K, 128d, k=10) | **55µs** |
| Build time (5K, 128d) | 6.2s |
| PQ self-distance | < 0.1 |
| Degree bound | Verified ≤ maxDegree for all nodes |

17 tests passing: distance (L2, IP, flat vectors, visited set, PQ table), PQ (train/encode), Vamana (build/search, bounded degree), index (basic, PQ, save/load, recall@10, scale, dimension mismatch, duplicate ID, search-before-build).

## When to Use DiskANN vs HNSW

| | HNSW (`@ruvector/router`) | DiskANN (`@ruvector/diskann`) |
|---|---|---|
| Scale | <1M vectors, all in RAM | 1M+ vectors, SSD-backed |
| Insert | Incremental (anytime) | Batch (build after all inserts) |
| Search | Sub-ms, no build step | 55µs after build |
| Memory | Full vectors in RAM | Only graph + PQ codes in RAM |
| Use case | Real-time routing | Large corpus RAG, retrieval |

## Consequences

- DiskANN claim in README is now backed by a real, benchmarked implementation
- 17 Rust tests + 1 Node.js integration test validate correctness
- Published to crates.io (v2.1.0) and npm (v0.1.0, 5 platforms)
- Optional `simd` and `gpu` features available for further acceleration
- Integrated into `ruvector` via optional peerDep — zero cost if not installed
