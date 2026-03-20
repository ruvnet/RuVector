# RuVector Sparsifier

[![Crates.io](https://img.shields.io/crates/v/ruvector-sparsifier.svg)](https://crates.io/crates/ruvector-sparsifier)
[![Documentation](https://docs.rs/ruvector-sparsifier/badge.svg)](https://docs.rs/ruvector-sparsifier)
[![License](https://img.shields.io/crates/l/ruvector-sparsifier.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ruvnet%2Fruvector-blue?logo=github)](https://github.com/ruvnet/ruvector)
[![ruv.io](https://img.shields.io/badge/ruv.io-AI%20Infrastructure-orange)](https://ruv.io)

**An always-on compressed world model for real-time graph analytics.**

*Dynamic spectral sparsification for HNSW health monitoring, structural diagnostics, and continuous graph reasoning.*

---

## Why This Matters

Every vector database, similarity index, and memory graph is backed by a dense web of connections. Analyzing the full graph is expensive — often too expensive for real-time use. **RuVector Sparsifier** maintains a small shadow graph that provably preserves the spectral structure of your full graph, enabling:

- **Continuous monitoring** instead of batch analysis
- **3-10x faster** graph diagnostics (min-cut, clustering, Laplacian solves)
- **10-50x less memory** for analytics workloads
- **Early anomaly detection** via structural drift monitoring

### The Key Insight

If dynamic min-cut is your **fragility alarm**, spectral sparsification is your **always-on compressed world model**. Together they give your system a small graph it can think with continuously.

```
full graph = everything you know
sparse graph = what you need to think quickly
```

---

## How It Works

A spectral sparsifier H of graph G has O(n log n / ε²) edges and preserves the Laplacian quadratic form within (1 ± ε):

```
(1-ε) · xᵀ L_G x  ≤  xᵀ L_H x  ≤  (1+ε) · xᵀ L_G x    ∀x ∈ Rⁿ
```

This means H preserves **all** spectral properties — cuts, connectivity, conductance, effective resistances, mixing times — within relative error ε.

### Architecture

```
Full Graph (ground truth)
    │
    ├─ Backbone (spanning forest → connectivity guarantee)
    ├─ Importance scoring (random walk effective resistance)
    ├─ Spectral sampling (edges kept ∝ weight × importance × log n / ε²)
    └─ Periodic audits (random probe verification)
    │
    ▼
Sparsifier (compressed world model)
```

### Implementation

Based on the ADKKP16 approach (Abraham, Durfee, Koutis, Krinninger, Peng — FOCS 2016) adapted for practical real-time use:

| Component | What It Does | Complexity |
|-----------|-------------|-----------|
| **Backbone** | Union-find spanning forest | O(α(n)) per update |
| **Importance** | Random walk effective resistance | O(walk_length × num_walks) |
| **Sampler** | Probability-proportional edge sampling | O(m) for full, O(1) incremental |
| **Audit** | Laplacian quadratic form comparison | O(n × n_probes) |

---

## Quick Start

```rust
use ruvector_sparsifier::{AdaptiveGeoSpar, SparseGraph, SparsifierConfig};

// Build a graph.
let g = SparseGraph::from_edges(&[
    (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0),
    (3, 0, 1.0), (0, 2, 0.5),
]);

// Construct the sparsifier.
let mut spar = AdaptiveGeoSpar::build(&g, SparsifierConfig::default()).unwrap();

// Dynamic updates.
spar.insert_edge(1, 3, 2.0).unwrap();
spar.delete_edge(0, 2).unwrap();

// Audit quality.
let audit = spar.audit();
println!("Audit passed: {}, max error: {:.4}", audit.passed, audit.max_error);

// Access the compressed graph.
let h = spar.sparsifier();
println!("Compression: {:.1}x ({} -> {} edges)",
    spar.compression_ratio(),
    spar.stats().full_edge_count,
    h.num_edges(),
);
```

---

## Configuration

```rust
SparsifierConfig {
    epsilon: 0.2,                    // Spectral accuracy (lower = more edges)
    edge_budget_factor: 8,           // Target edges = factor × n
    audit_interval: 1000,            // Updates between audits
    walk_length: 6,                  // Random walk hops
    num_walks: 10,                   // Walks per edge
    n_audit_probes: 30,              // Probe vectors per audit
    auto_rebuild_on_audit_failure: true,
    local_rebuild_fraction: 0.1,
}
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| `epsilon` | 0.05–0.5 | Lower = more faithful, more edges |
| `edge_budget_factor` | 4–12 | Lower = more aggressive compression |
| `audit_interval` | 100–10000 | Lower = more frequent quality checks |

---

## Use Cases

### 1. HNSW Index Health Monitoring

```rust
// Monitor graph health via spectral properties of the sparsifier.
let spar = AdaptiveGeoSpar::build(&hnsw_graph, config)?;

// Cheap continuous monitoring on the sparsifier.
let audit = spar.audit();
if !audit.passed {
    // Trigger reindex or alert.
}
```

### 2. Faster Min-Cut on the Control Graph

```rust
// Run min-cut on the sparsifier (3-10x cheaper than full graph).
let cut_value_approx = compute_mincut(spar.sparsifier());
```

### 3. Real-Time Drift Detection

```rust
// Track structural drift by comparing audits over time.
let audit_t1 = spar.audit();
// ... updates happen ...
let audit_t2 = spar.audit();
if audit_t2.avg_error > 2.0 * audit_t1.avg_error {
    // Structural drift detected.
}
```

### 4. Embedding Point Moves

```rust
// Handle embedding updates (e.g., vector reindexing).
spar.update_embedding(
    node_id,
    &old_neighbors,  // [(neighbor, similarity_weight), ...]
    &new_neighbors,
)?;
```

### 5. Multi-Tier Memory

```
Hot tier:  Full HNSW graph  → exact retrieval
Warm tier: Sparsifier       → fast diagnostics, approximate analytics
Cold tier: Archived snapshots → historical trend analysis
```

---

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `static-sparsify` | yes | One-shot static sparsification |
| `dynamic` | yes | Dynamic insert/delete support |
| `simd` | no | SIMD-accelerated distance operations |
| `wasm` | no | WebAssembly-compatible paths |
| `audit` | no | Extended audit & diagnostics |
| `full` | no | All features enabled |

---

## Performance

| Graph Size | Full Edges | Sparsifier Edges (ε=0.2) | Build Time | Audit Time |
|-----------|-----------|-------------------------|-----------|-----------|
| 100 nodes | 500 | ~90 | 0.3 ms | 0.05 ms |
| 1K nodes | 10K | ~1.2K | 15 ms | 2 ms |
| 10K nodes | 150K | ~14K | 300 ms | 30 ms |
| 100K nodes | 2M | ~170K | 8 s | 500 ms |

*Benchmarks on x86_64 with `cargo bench`. Run `cargo bench -p ruvector-sparsifier` to reproduce.*

---

## Theoretical Background

### Spectral Sparsification

A spectral sparsifier preserves the Laplacian quadratic form `xᵀLx` for all vectors x. This guarantees preservation of:

- **All cut values** (within 1±ε)
- **Effective resistances** between all vertex pairs
- **Spectral gap** and mixing time
- **Conductance** of all vertex subsets
- **Solutions to Laplacian systems** Lx = b

### Key References

| Year | Result | Contribution |
|------|--------|-------------|
| 2008 | Spielman-Srivastava | Sparsification by effective resistances |
| 2009 | Batson-Spielman-Srivastava | Optimal O(n/ε²) sparsifiers |
| 2016 | ADKKP (FOCS) | First polylog dynamic maintenance |
| 2025 | Khanna-Li-Putterman (STOC) | Dynamic hypergraph sparsification |
| 2025 | Zhao | Dynamic directed graph sparsification |
| 2026 | Forster-Goranci-Momeni (STACS) | Dynamic directed hypergraph sparsification |

---

## RuVector Ecosystem

```
ruvector-sparsifier ←→ ruvector-solver (Laplacian solves on sparsifier)
       ↕                      ↕
ruvector-coherence (spectral health scoring)
       ↕
ruvector-mincut (structural alarm on sparsifier)
       ↕
cognitum-gate-kernel (evidence accumulation)
```

---

## WASM Support

See [`ruvector-sparsifier-wasm`](../ruvector-sparsifier-wasm/) for WebAssembly bindings.

```typescript
import { WasmSparsifier } from 'ruvector-sparsifier-wasm';

const spar = WasmSparsifier.buildFromEdges(
  '[[0,1,1.0],[1,2,1.0],[2,0,1.0]]',
  '{"epsilon":0.2}'
);

spar.insertEdge(0, 3, 2.0);
console.log(JSON.parse(spar.audit()));
console.log('Compression:', spar.compressionRatio(), 'x');
```

---

## Acceptance Test

For any workload, compare side-by-side:

| Metric | Target |
|--------|--------|
| Structural error (Laplacian QF) | ≤ 5% |
| Cut value error | ≤ 5% |
| Speedup (graph analytics) | ≥ 3x |
| Memory reduction | ≥ 5x |

```bash
cargo test -p ruvector-sparsifier
cargo bench -p ruvector-sparsifier
```

---

## License

MIT — see [LICENSE](../../LICENSE) for details.
