# Rust Crate Design: ruvector-sparsifier

**Document ID**: spectral-sparsification/02-rust-crate-design
**Date**: 2026-03-19
**Status**: Implemented
**Classification**: Crate Architecture — Implementation Design
**Series**: [00](./00-executive-summary.md) | [01](./01-algorithms-sota.md) | **02** | [03](./03-ruvector-integration.md) | [04](./04-companion-systems.md)

---

## 1. Rust Ecosystem Analysis

### 1.1 Existing Crates (Building Blocks)

| Crate | Purpose | Relevance |
|-------|---------|-----------|
| **petgraph** | Graph data structures (CSR, adjacency list, traversals) | Graph representation patterns |
| **sprs** | Sparse matrix operations (CSR/CSC, sparse BLAS) | Matrix storage reference |
| **faer** | High-performance dense/sparse linear algebra | Eigendecomposition backend |
| **nalgebra** | General-purpose linear algebra | Vector operations |
| **ruvector-solver** | 7 sublinear solver engines, CsrMatrix, WASM-ready | Direct integration target |

### 1.2 Gap Analysis

**No Rust crate for spectral graph sparsification exists.** The `ruvector-sparsifier` crate is first-to-market in the Rust ecosystem, providing:
- Dynamic spectral sparsification (insert/delete/point-move)
- Effective resistance-based importance scoring
- Spectral auditing via Laplacian quadratic form probes
- WASM compilation for browser/edge deployment

---

## 2. Module Architecture

```
ruvector-sparsifier/
├── Cargo.toml
├── README.md
├── benches/
│   └── sparsifier_bench.rs          # Criterion benchmarks
├── tests/
│   └── integration_tests.rs         # Integration + property tests
└── src/
    ├── lib.rs                       # Module declarations, re-exports, prelude
    ├── error.rs                     # SparsifierError enum, Result alias
    ├── types.rs                     # SparsifierConfig, EdgeImportance, AuditResult, Stats
    ├── graph.rs                     # SparseGraph (adjacency + CSR + Laplacian QF)
    ├── backbone.rs                  # Backbone spanning forest (union-find)
    ├── importance.rs                # Effective resistance estimation, importance scoring
    ├── sampler.rs                   # Spectral edge sampling (probability-proportional)
    ├── sparsifier.rs                # AdaptiveGeoSpar (main dynamic sparsifier)
    ├── audit.rs                     # SpectralAuditor (quadratic form, cuts, conductance)
    └── traits.rs                    # Sparsifier, ImportanceScorer, BackboneStrategy traits

ruvector-sparsifier-wasm/
├── Cargo.toml
├── README.md
└── src/
    └── lib.rs                       # wasm-bindgen bindings
```

---

## 3. Core Types

### 3.1 SparseGraph

```rust
pub struct SparseGraph {
    adj: Vec<HashMap<usize, f64>>,   // adjacency lists
    num_edges: usize,                // undirected edge count
    total_weight: f64,               // sum of all weights
}
```

Key operations:
- `insert_edge(u, v, weight)` / `delete_edge(u, v)` — O(1) amortized
- `laplacian_quadratic_form(x)` — O(m) for full graph, O(m') for sparsifier
- `to_csr()` / `from_csr()` — CSR export for solver integration
- `neighbors(u)` / `edges()` — iteration

### 3.2 SparsifierConfig

```rust
pub struct SparsifierConfig {
    pub epsilon: f64,                    // 0.2 default
    pub edge_budget_factor: usize,       // 8 default (budget = factor × n)
    pub audit_interval: usize,           // 1000 updates between audits
    pub walk_length: usize,              // 6 hops for random walks
    pub num_walks: usize,                // 10 walks per edge
    pub n_audit_probes: usize,           // 30 random probe vectors
    pub auto_rebuild_on_audit_failure: bool,
    pub local_rebuild_fraction: f64,
}
```

### 3.3 AdaptiveGeoSpar

The main sparsifier struct:

```rust
pub struct AdaptiveGeoSpar {
    g_full: SparseGraph,             // full graph (ground truth)
    g_spec: SparseGraph,             // compressed sparsifier
    backbone: Backbone,              // spanning forest
    scorer: LocalImportanceScorer,   // importance estimation
    sampler: SpectralSampler,        // probability-proportional sampling
    auditor: SpectralAuditor,        // quality verification
    config: SparsifierConfig,
    stats: SparsifierStats,
    backbone_edge_set: HashSet<(usize, usize)>,
    snapshot: RwLock<SparseGraph>,    // thread-safe reader snapshot
}
```

---

## 4. Feature Flags

| Flag | Default | Dependencies | Description |
|------|---------|-------------|-------------|
| `static-sparsify` | yes | None | One-shot static sparsification |
| `dynamic` | yes | None | Dynamic insert/delete support |
| `simd` | no | None | SIMD-accelerated operations |
| `wasm` | no | None | WASM-compatible code paths |
| `audit` | no | None | Extended audit diagnostics |
| `full` | no | All above | Everything enabled |

---

## 5. Algorithm Selection

### Phase 1 (Shipped): Practical Effective Resistance

Instead of exact effective resistance (which requires solving Laplacian systems), we use **random walk commute time** as a practical approximation:

```
R_eff(u,v) ≈ commute_time(u,v) / (2 × total_weight)
```

This is computed via short random walks (6 hops, 10 walks per edge) and provides a good importance ranking without needing a full Laplacian solver.

### Phase 2 (Planned): ADKKP-Style Dynamic Maintenance

Replace the random walk estimator with the ADKKP layered sampling:
1. Maintain O(log n) resistance levels
2. Per-level independent sparsifiers
3. Amortized polylog updates via link-cut trees

### Phase 3 (Planned): GPU-Accelerated Importance

Use localized random walks on GPU for importance scoring. Recent practical work shows ~10x speedup over CPU incremental methods.

### The "Shippable Update Rule"

The product version trades theoretical purity for engineering pragmatism:

```
For each insert/delete/embedding move:
  1. Update G_full
  2. Update backbone (union-find)
  3. Score touched edges (random walk importance)
  4. Keep edge with probability ∝ weight × importance × log(n) / ε²
  5. Reweight kept edges by inverse probability
  6. Every audit_interval updates: verify via random probes
  7. If audit fails: rebuild affected region
```

---

## 6. WASM Strategy

### 6.1 Memory Budget

For a graph with n vertices and m edges:

| Component | Memory | At n=10K |
|-----------|--------|----------|
| Adjacency lists (full) | ~24m bytes | 3.6 MB |
| Adjacency lists (sparsifier) | ~24m' bytes | ~0.4 MB |
| Backbone union-find | 16n bytes | 160 KB |
| Config + stats | ~200 bytes | 200 B |
| **Total** | | **~4.2 MB** |

Target: <5 MB WASM memory for 10K-node graphs.

### 6.2 Web Worker Integration

```typescript
// spectral-worker.ts
import init, { WasmSparsifier } from 'ruvector-sparsifier-wasm';

await init();
const spar = WasmSparsifier.buildFromEdges(edgesJson, configJson);

self.onmessage = (event) => {
    switch (event.data.type) {
        case 'insert':
            spar.insertEdge(event.data.u, event.data.v, event.data.w);
            self.postMessage({ type: 'stats', value: spar.stats() });
            break;
        case 'audit':
            self.postMessage({ type: 'audit', value: spar.audit() });
            break;
    }
};
```

---

## 7. API Design

### Core Trait

```rust
pub trait Sparsifier: Send + Sync {
    fn insert_edge(&mut self, u: usize, v: usize, weight: f64) -> Result<()>;
    fn delete_edge(&mut self, u: usize, v: usize) -> Result<()>;
    fn audit(&self) -> AuditResult;
    fn rebuild_local(&mut self, nodes: &[usize]) -> Result<()>;
    fn rebuild_full(&mut self) -> Result<()>;
    fn sparsifier(&self) -> &SparseGraph;
    fn compression_ratio(&self) -> f64;
    fn stats(&self) -> &SparsifierStats;
}
```

### Construction

```rust
// From existing graph
let spar = AdaptiveGeoSpar::build(&graph, config)?;

// Empty, for streaming construction
let spar = AdaptiveGeoSpar::new(config);

// Dynamic updates
spar.insert_edge(u, v, weight)?;
spar.delete_edge(u, v)?;
spar.update_embedding(node, &old_neighbors, &new_neighbors)?;
```

---

## 8. Performance Characteristics

### Benchmarked (cargo bench)

| Operation | n=100 | n=500 | n=1000 |
|-----------|-------|-------|--------|
| Build | 161 µs | ~4 ms | ~16 ms |
| Insert edge | 81 µs | ~0.4 ms | ~0.8 ms |
| Audit (30 probes) | 39 µs | ~0.5 ms | ~2 ms |
| Laplacian QF | 4.2 µs | ~50 µs | ~200 µs |

---

## Document Navigation

- **Previous**: [01 - Algorithms & SOTA](./01-algorithms-sota.md)
- **Next**: [03 - RuVector Integration](./03-ruvector-integration.md)
