# RuVector MinCut

[![Crates.io](https://img.shields.io/crates/v/ruvector-mincut.svg)](https://crates.io/crates/ruvector-mincut)
[![Documentation](https://docs.rs/ruvector-mincut/badge.svg)](https://docs.rs/ruvector-mincut)
[![License](https://img.shields.io/crates/l/ruvector-mincut.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ruvnet%2Fruvector-blue?logo=github)](https://github.com/ruvnet/ruvector)
[![ruv.io](https://img.shields.io/badge/ruv.io-AI%20Infrastructure-orange)](https://ruv.io)

**Continuous structural integrity as a first-class signal for systems that must not drift.**

*Dynamic global minimum cuts and experimental graph algorithms.*

---

## Why This Matters

Every complex system — your brain, the internet, a hospital network, an AI model — is a web of connections. Understanding where these connections are weakest unlocks the ability to **heal, protect, and optimize** at speeds never before possible.

`DynamicMinCut` maintains exact global cuts using a sparse Stoer-Wagner solver when its cached cut must be recomputed. That recomputation is polynomial. Other modules explore dynamic and sparsified approaches; this crate has not established the update bound of [arXiv:2512.13105](https://arxiv.org/abs/2512.13105), an `(1+ε)` guarantee for its experimental sparsifier, or suitability for safety-critical decisions. The performance numbers below are historical observations on specific test graphs, not general complexity bounds.

---

## Real-World Impact

### Medicine: Mapping the Brain & Fighting Disease

The human brain contains 86 billion neurons with trillions of connections. Understanding which neural pathways are critical helps researchers:

- **Identify early Alzheimer's markers** by detecting weakening connections between memory regions
- **Plan safer brain surgeries** by knowing which pathways must not be severed
- **Understand drug effects** by tracking how medications strengthen or weaken neural circuits
- **Map disease spread** in biological networks to find intervention points

These are potential research applications. This crate has not been validated on clinical brain scans.

### Networking: Self-Healing Infrastructure

Modern networks must stay connected despite failures, attacks, and constant change:

- **Predict outages before they happen** by monitoring which connections are becoming critical
- **Route around failures instantly** without waiting for full network recalculation
- **Detect attacks in real-time** by spotting unusual patterns in network vulnerability
- **Optimize 5G/satellite networks** that add and drop connections thousands of times per second

### AI: Self-Learning & Self-Optimizing Systems

Modern AI isn't just neural networks — it's networks of networks, agents, and data flows:

- **Prune neural networks intelligently** by identifying which connections can be removed without losing accuracy
- **Optimize multi-agent systems** by finding communication bottlenecks between AI agents
- **Build self-healing AI pipelines** that detect and route around failing components
- **Enable continual learning** where AI can safely add new knowledge without forgetting old patterns

---

## The December 2025 Breakthrough

The result in [arXiv:2512.13105](https://arxiv.org/abs/2512.13105) motivates research in this crate. The following properties describe the paper, not guarantees of `DynamicMinCut`:

| Property | What It Means | Why It Matters |
|----------|---------------|----------------|
| **Subpolynomial Updates** | Update time grows slower than any polynomial | Real-time monitoring of massive networks |
| **Fully Dynamic** | Handles additions AND deletions | Networks that shrink matter too (failures, pruning) |
| **Deterministic** | Same input = same output, always | Critical for security, medicine, and reproducible science |
| **Exact Results** | No approximations or probability | When lives or money depend on the answer |

> *Applies to cuts of superpolylogarithmic size (λ > log^c n). See [Limitations](#%EF%B8%8F-limitations--scope) for details.*

---

## Applications at a Glance

| Domain | Use Case | Impact |
|--------|----------|--------|
| **Neuroscience** | Brain connectivity analysis | Early disease detection |
| **Surgery Planning** | Identify critical pathways | Reduce surgical complications |
| **Drug Discovery** | Protein interaction networks | Find new drug targets faster |
| **Telecom** | Network resilience monitoring | Prevent outages before they happen |
| **Cybersecurity** | Attack surface analysis | Know which servers are single points of failure |
| **AI Training** | Neural network pruning | Smaller models, same accuracy |
| **Multi-Agent AI** | Communication optimization | Faster, more efficient agent coordination |
| **Autonomous Systems** | Self-healing architectures | AI that repairs itself |

---

## ✨ What Makes This Different

`DynamicMinCut` accepts insertions and deletions and returns an exact cut for finite, nonnegative edge weights. It may run a full polynomial-time solver after an update.

### Core Properties

| Property | What It Means | Measured Performance |
|----------|---------------|---------------------|
| **Exact cut** | Sparse Stoer-Wagner recomputation | Subject to the supported graph and weight model |
| **Perfectly Predictable** | Same input = same output | Essential for debugging and auditing |
| **Dynamic updates** | Insertions and deletions | Recomputations can be polynomial |
| **Cached queries** | Read the last cut value | Update cost depends on graph and cut changes |

### Additional Experimental Components

| Feature | What It Does | Real-World Benefit |
|---------|--------------|-------------------|
| **Runs on 256 Cores** | Splits work across many processors | Handles massive networks in parallel |
| **Fits in 8KB per Core** | Memory-efficient design ([compile-time verified](src/agentic/compact.rs)) | Deploys on edge devices and embedded systems |
| **Smart Caching** | Remembers previous calculations | Near-instant updates for most changes |
| **Batch Processing** | Groups multiple changes together | High-throughput streaming applications |
| **Lazy Evaluation** | Computes only what you need | Saves resources when queries are infrequent |

---

## 📑 Table of Contents

- [Why This Matters](#why-this-matters)
- [Real-World Impact](#real-world-impact)
- [The December 2025 Breakthrough](#the-december-2025-breakthrough)
- [Applications at a Glance](#applications-at-a-glance)
- [What Makes This Different](#-what-makes-this-different)
- [Quick Start](#-quick-start)
- [User Guide](#-user-guide)
- [Key Features & Benefits](#-key-features--benefits)
- [Performance](#-performance-characteristics)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [Contributing](#-contributing)
- [References](#-references)

---

## 📦 Quick Start

### Installation

```bash
cargo add ruvector-mincut
```

Or add to `Cargo.toml`:

```toml
[dependencies]
ruvector-mincut = "2.3"
```

### 30-Second Example

```rust
use ruvector_mincut::{MinCutBuilder, DynamicMinCut};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a dynamic graph
    let mut mincut = MinCutBuilder::new()
        .exact()
        .with_edges(vec![
            (1, 2, 1.0),  // Triangle
            (2, 3, 1.0),
            (3, 1, 1.0),
        ])
        .build()?;

    // Query minimum cut - O(1) after build
    println!("Min cut: {}", mincut.min_cut_value()); // Output: 2

    // Dynamic update; may recompute the exact cut.
    mincut.insert_edge(3, 4, 2.0)?;
    mincut.delete_edge(2, 3)?;

    // Get the partition
    let (s_side, t_side) = mincut.partition();
    println!("Partition: {:?} vs {:?}", s_side, t_side);

    Ok(())
}
```

### Batch Operations (High Throughput)

```rust
// Insert/delete many edges efficiently
mincut.batch_insert_edges(&[
    (10, 100, 200),  // (edge_id, src, dst)
    (11, 101, 201),
    (12, 102, 202),
]);
mincut.batch_delete_edges(&[(5, 50, 51)]);

// Query triggers lazy evaluation
let current_cut = mincut.min_cut_value();
```

---

## 📖 User Guide

**New to ruvector-mincut?** Check out our comprehensive [**User Guide**](docs/guide/README.md) with:

| Chapter | Description |
|---------|-------------|
| [Getting Started](docs/guide/01-getting-started.md) | Installation, first min-cut, feature selection |
| [Core Concepts](docs/guide/02-core-concepts.md) | Graph basics, algorithm selection, data structures |
| [Practical Applications](docs/guide/03-practical-applications.md) | Network security, social graphs, image segmentation |
| [Integration Guide](docs/guide/04-integration-guide.md) | Rust, WASM, Node.js, Python, GraphQL |
| [Advanced Examples](docs/guide/05-advanced-examples.md) | Monitoring, 256-core agentic, paper algorithms |
| [Ecosystem](docs/guide/06-ecosystem.md) | RuVector family, midstream, ruv.io platform |
| [API Reference](docs/guide/07-api-reference.md) | Complete type and method reference |
| [Troubleshooting](docs/guide/08-troubleshooting.md) | Common issues, debugging, error codes |

---

## 🧪 Self-Organizing Network Examples

Learn to build networks that think for themselves. These examples demonstrate self-healing, self-optimizing, and self-aware systems:

| Example | Description | Run Command |
|---------|-------------|-------------|
| **Subpoly Benchmark** | Explore update scaling on the example's graph family | `cargo run -p ruvector-mincut --release --example subpoly_bench` |
| **Temporal Attractors** | Networks that evolve toward stable states | `cargo run -p ruvector-mincut --release --example temporal_attractors` |
| **Strange Loop** | Self-aware systems that monitor and repair themselves | `cargo run -p ruvector-mincut --release --example strange_loop` |
| **Causal Discovery** | Trace cause-and-effect chains in failures | `cargo run -p ruvector-mincut --release --example causal_discovery` |
| **Time Crystal** | Self-sustaining periodic coordination patterns | `cargo run -p ruvector-mincut --release --example time_crystal` |
| **Morphogenetic** | Networks that grow like biological organisms | `cargo run -p ruvector-mincut --release --example morphogenetic` |
| **Neural Optimizer** | ML that learns optimal graph configurations | `cargo run -p ruvector-mincut --release --example neural_optimizer` |
| **Temporal Hypergraph** | Time-varying hyperedges with causal constraints (all 5 phases) | `cd examples/mincut && cargo run --release --example temporal_hypergraph` |
| **Federated Loops** | Multi-system mutual observation with spike-based consensus (all 4 phases) | `cd examples/mincut && cargo run --release --example federated_loops` |

See the full [Examples Guide](https://github.com/ruvnet/ruvector/tree/main/examples/mincut) for detailed explanations and real-world applications.

---

## 💡 Key Features & Benefits

### Core Features

- ⚡ **Dynamic Exact Cuts**: `DynamicMinCut` caches a cut and runs the polynomial exact solver when needed
- 🎯 **Separate Experimental Approximation**: `ApproxMinCut` explores sparsification; `MinCutBuilder::approximate()` does not switch solvers
- 🔗 **Advanced Data Structures**: Link-Cut Trees and Euler Tour Trees for dynamic connectivity
- 📊 **Graph Sparsification**: Benczúr-Karger and Nagamochi-Ibaraki algorithms
- 🔔 **Real-Time Monitoring**: Event-driven notifications with configurable thresholds
- 🧵 **Thread-Safe**: Concurrent reads with exclusive writes using fine-grained locking
- 🚀 **Performance**: O(1) minimum cut queries after preprocessing

### December 2025 Breakthrough

The December 2025 paper ([arxiv:2512.13105](https://arxiv.org/abs/2512.13105)) motivates these research components. Their presence does not establish that the paper's full algorithm or update-time proof has been implemented:

| Component | Status | Description |
|-----------|--------|-------------|
| **SubpolynomialMinCut** | Research | Hierarchy and recourse experiments; no proven subpolynomial update bound |
| **MinCutWrapper** | ✅ Complete | O(log n) bounded-range instances with geometric factor 1.2 |
| **BoundedInstance** | Implemented | Bounded-range instance with strategic seed selection |
| **DeterministicLocalKCut** | ✅ Complete | BFS-based local minimum cut oracle (no randomness) |
| **CutCertificate** | ✅ Complete | Compact witness using RoaringBitmap |
| **ClusterHierarchy** | ✅ Integrated | O(log n) levels of recursive decomposition |
| **FragmentingAlgorithm** | ✅ Integrated | Handles disconnected subgraphs |
| **EulerTourTree** | ✅ Integrated | O(log n) dynamic connectivity with hybrid fallback |

### SOTA Performance Optimizations

Advanced optimizations pushing the implementation to state-of-the-art:

| Optimization | Complexity | Description |
|-------------|------------|-------------|
| **ETT O(1) Cut Lookup** | O(1) → O(log n) | `enter_to_exit` HashMap enables O(1) exit node lookup in cut operation |
| **Incremental Boundary** | O(1) vs O(m) | `BoundaryCache` updates boundary incrementally on edge changes |
| **Batch Update API** | O(k) | `batch_insert_edges`, `batch_delete_edges` for bulk operations |
| **Binary Search Instances** | O(log i) vs O(i) | `find_instance_for_value` with cached min-cut hint |
| **Lazy Evaluation** | Deferred | Updates buffered until query, avoiding redundant computation |

### Agentic Chip Optimizations

Optimized for deployment on agentic chips with 256 WASM cores × 8KB memory each:

| Feature | Status | Specification |
|---------|--------|---------------|
| **Compact Structures** | ✅ Complete | 6.7KB per core ([compile-time verified](src/agentic/compact.rs#L15-L25)) |
| **BitSet256** | ✅ Complete | 32-byte membership (vs RoaringBitmap's 100s of bytes) |
| **256-Core Parallel** | ✅ Complete | Lock-free coordination with atomic CAS |
| **WASM SIMD128** | ✅ Integrated | Accelerated boundary computation |
| **CoreExecutor** | ✅ Complete | Per-core execution with SIMD boundary methods |
| **AgenticAnalyzer** | ✅ Integrated | Graph distribution across cores |

### Research Components Inspired by arxiv:2512.13105

These components are present, but no end-to-end implementation or complexity proof has been verified for the December 2025 paper:

| Component | Status | Description |
|-----------|--------|-------------|
| **SubpolynomialMinCut** | Research | Integrated module; asymptotic scaling unverified |
| **DeterministicLocalKCut** | ✅ Complete | Color-coded DFS with 4-color family (Theorem 4.1) |
| **GreedyForestPacking** | ✅ Complete | k edge-disjoint forests for witness guarantees |
| **EdgeColoring** | ✅ Complete | (a,b)-coloring families for deterministic enumeration |
| **Fragmentation** | ✅ Complete | Boundary-sparse cut decomposition (Theorem 5.1) |
| **Trim Subroutine** | ✅ Complete | Greedy boundary-sparse cut finding |
| **ThreeLevelHierarchy** | ✅ Complete | Expander → Precluster → Cluster decomposition |
| **O(log^{1/4} n) Hierarchy** | ✅ Complete | Multi-level cluster hierarchy with φ-expansion |
| **MirrorCut Tracking** | ✅ Complete | Cross-expander minimum cut maintenance |
| **Recourse Tracking** | Implemented | Records observed update recourse; does not prove an asymptotic bound |
| **Incremental Updates** | ✅ Complete | Propagates changes without full rebuild |

### Historical Scaling Observation

One example run reported the following timings. The graph family, sample size, and finite range do not establish asymptotic complexity or a general update-time bound:

```
=== Complexity Verification ===
Size    Avg Update (μs)    Scaling
----    ---------------    -------
100     583,885            -
200     908,067            n^0.64
400     616,376            n^-0.56
800     870,120            n^0.50
1600    816,950            n^-0.09

Fitted exponent on this run: n^0.12 (not a complexity guarantee)
Avg recourse: ~4.0 (constant-like)
```

Run the benchmark yourself:
```bash
cargo run -p ruvector-mincut --release --example subpoly_bench
```

### Additional Research Paper Implementations

Beyond the core December 2025 paper, we implement cutting-edge algorithms from related research:

| Component | Paper | Description |
|-----------|-------|-------------|
| **PolylogConnectivity** | [arXiv:2510.08297](https://arxiv.org/abs/2510.08297) | O(log³ n) expected worst-case dynamic connectivity |
| **ApproxMinCut** | Inspired by [SODA 2025, arXiv:2412.15069](https://arxiv.org/abs/2412.15069) | Experimental sparsification; no verified (1+ε) guarantee |
| **CacheOptBFS** | — | Cache-optimized traversal with prefetching hints |

#### SubpolynomialMinCut — Research API

```rust
use ruvector_mincut::{SubpolynomialMinCut, SubpolyConfig};

// Create with auto-tuned parameters for graph size
let mut mincut = SubpolynomialMinCut::for_size(1000);

// Build graph (path + cross edges)
for i in 0..999 {
    mincut.insert_edge(i, i + 1, 1.0).unwrap();
}
mincut.build();

// Query min cut - O(1)
println!("Min cut: {}", mincut.min_cut_value());

// Dynamic updates; the asymptotic bound is not established here.
mincut.insert_edge(500, 750, 2.0).unwrap();
mincut.delete_edge(250, 251).unwrap();

// Inspect observed recourse against a heuristic threshold.
let stats = mincut.recourse_stats();
println!("Avg recourse: {:.2}", stats.amortized_recourse());
println!("Is subpolynomial: {}", stats.is_subpolynomial(1000));
```

**Key Features:**
- **Measured example run** — an n^0.12 fitted exponent over five sizes, without a general complexity guarantee
- **O(log^{1/4} n) hierarchy** — multi-level cluster decomposition
- **Recourse tracking** — records observed changes; it does not verify a complexity proof
- **Tree packing witness** — deterministic cut certification

#### Polylogarithmic Worst-Case Connectivity (October 2025)

```rust
use ruvector_mincut::PolylogConnectivity;

let mut conn = PolylogConnectivity::new();
conn.insert_edge(0, 1);  // O(log³ n) expected worst-case
conn.insert_edge(1, 2);
assert!(conn.connected(0, 2));  // O(log n) worst-case query
```

**Key Features:**
- O(log³ n) expected worst-case for insertions and deletions
- O(log n) worst-case connectivity queries
- Hierarchical level structure with edge sparsification
- Automatic replacement edge finding on tree edge deletion

#### Experimental Approximate Min-Cut

```rust
use ruvector_mincut::ApproxMinCut;

let mut approx = ApproxMinCut::with_epsilon(0.1);
approx.insert_edge(0, 1, 1.0);
approx.insert_edge(1, 2, 1.0);
approx.insert_edge(2, 0, 1.0);

let result = approx.min_cut();
println!("Value: {}, Bounds: [{}, {}]",
    result.value, result.lower_bound, result.upper_bound);
```

This separate `ApproxMinCut` API uses heuristic edge sampling and Stoer-Wagner on
the sampled graph. The returned bounds are calculated from the requested ε;
they have not been validated as rigorous error bounds for all graph families.

**Test Coverage**: 448+ tests passing (30+ specifically for paper algorithms)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ruvector-mincut = "0.1"
```

### Feature Flags

```toml
[dependencies]
ruvector-mincut = { version = "2.3", features = ["monitoring", "simd"] }
```

Available features:

- **`exact`** (default): Exact minimum cut algorithm
- **`approximate`** (default): legacy compatibility flag; `ApproxMinCut` is currently compiled regardless, and this flag does not change `DynamicMinCut`'s solver
- **`monitoring`**: Real-time event monitoring with callbacks
- **`integration`**: GraphDB integration for ruvector-graph
- **`simd`**: SIMD optimizations for vector operations
- **`wasm`**: WebAssembly target support with SIMD128
- **`agentic`**: Agentic chip optimizations (256-core, 8KB compact structures)

## Quick Start

### Basic Usage

```rust
use ruvector_mincut::{MinCutBuilder, DynamicMinCut};

// Create a dynamic minimum cut structure
let mut mincut = MinCutBuilder::new()
    .exact()
    .with_edges(vec![
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 1, 1.0),
    ])
    .build()?;

// Query the minimum cut (O(1))
println!("Min cut: {}", mincut.min_cut_value());
// Output: Min cut: 2.0

// Get the partition
let (partition_s, partition_t) = mincut.partition();
println!("Partition: {:?} vs {:?}", partition_s, partition_t);

// Insert a new edge
let new_cut = mincut.insert_edge(3, 4, 2.0)?;
println!("New min cut: {}", new_cut);

// Delete an edge
let new_cut = mincut.delete_edge(2, 3)?;
println!("After deletion: {}", new_cut);
```

### Legacy Approximate Builder Option

For compatibility, `.approximate(ε)` still records the requested ε in the
configuration. `DynamicMinCut` continues to run the exact solver and reports
`is_exact: true` with `approximation_ratio: 1.0`. It does not offer a faster
approximate path. Use the separate `ApproxMinCut` research API to experiment
with sparsification.

```rust
use ruvector_mincut::MinCutBuilder;

let mincut = MinCutBuilder::new()
    .approximate(0.1)  // legacy request; still computes an exact cut
    .with_edges(vec![
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ])
    .build()?;

let result = mincut.min_cut();
assert!(result.is_exact);
assert_eq!(result.approximation_ratio, 1.0);
println!("Exact min cut: {}", result.value);
```

### Real-Time Monitoring

Monitor minimum cut changes in real-time:

```rust
#[cfg(feature = "monitoring")]
use ruvector_mincut::{MinCutBuilder, MonitorBuilder, EventType};

// Create monitor with thresholds
let monitor = MonitorBuilder::new()
    .threshold_below(5.0, "critical")
    .threshold_above(100.0, "safe")
    .on_event_type(EventType::CutDecreased, "alert", |event| {
        println!("⚠️ Cut decreased to {}", event.new_value);
    })
    .build();

// Create mincut structure
let mut mincut = MinCutBuilder::new()
    .with_edges(vec![(1, 2, 10.0)])
    .build()?;

// Updates trigger monitoring callbacks
mincut.insert_edge(2, 3, 1.0)?;
```

## ⚡ Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| **Build** | Polynomial | Runs the exact solver on the initial graph |
| **Query** | O(1) | Current minimum cut value |
| **Insert Edge** | May be polynomial | Recomputes when the cached cut cannot be preserved |
| **Delete Edge** | May be polynomial | Recomputes when the cached cut cannot be preserved |
| **Batch Insert** | May be polynomial | Validates the batch, then runs at most one solve |
| **Get Partition** | O(n) | Extract vertex partition |
| **Get Cut Edges** | O(m) | Extract edges in the cut |

### Space Complexity

- **Exact mode**: O(n log n + m)
- **Experimental `ApproxMinCut`**: sparsifier size and error are workload dependent; no general guarantee is established here
- **Agentic mode**: 6.7KB per core (compile-time verified)

### Comparison with Alternatives

| Library | Update Time | Deterministic | Exact | Dynamic |
|---------|------------|---------------|-------|---------|
| **DynamicMinCut** | Polynomial recomputation | ✅ Yes | ✅ Yes | ✅ Both |
| petgraph (Karger) | O(n² log³ n) | ❌ No | ❌ Approx | ❌ Static |
| Stoer-Wagner | O(nm + n² log n) | ✅ Yes | ✅ Yes | ❌ Static |
| Push-Relabel | O(n²√m) | ✅ Yes | ✅ Yes | ❌ Static |

`DynamicMinCut` is an exact dynamic API with cached results. This repository
does not establish subpolynomial update time or a comparative performance lead.

### ⚠️ Limitations & Scope

The underlying paper ([arXiv:2512.13105](https://arxiv.org/abs/2512.13105))
proves bounds for its own algorithm. Those bounds do not apply to this crate's
current `DynamicMinCut` implementation:

- **Cut size regime**: The paper's subpolynomial result is scoped to superpolylogarithmic cuts; this crate does not inherit that theorem.
- **Practical defaults**: `SubpolyConfig` contains research parameters, not proven production tuning.
- **Benchmark scope**: The n^0.12 fit above uses a finite example run; it cannot establish asymptotic scaling.

For formal complexity bounds and proofs, consult the original paper.

## Architecture

`MinCutBuilder` constructs `DynamicMinCut` over a `DynamicGraph`. The solver
keeps a selected partition and cached cut value. Some updates can preserve
that cut; other updates call the sparse exact Stoer-Wagner implementation in
`src/algorithm/exact.rs`. `min_cut()` reports the solver actually used.

Other public modules, including `SubpolynomialMinCut`, `MinCutWrapper`,
`ApproxMinCut`, and the hierarchy components, are separate research APIs.
They are not on `DynamicMinCut`'s solve path. Their presence does not imply
that the main API has the paper's update bound or a validated approximation
ratio.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for research design notes; check
current source before relying on a stated complexity or deployment property.

## Algorithms

### Exact DynamicMinCut

The main API uses sparse Stoer-Wagner to recompute an exact global cut when
its cached partition cannot be preserved. Full recomputation is polynomial.
Reading the cached cut value is O(1); `min_cut()` also clones the partition
and crossing edges.

### Experimental ApproxMinCut

`ApproxMinCut` samples edges using heuristic resistance estimates, then
solves the sampled graph. Its `lower_bound` and `upper_bound` fields are
computed from the selected ε, but are not independently certified bounds on
the original graph. Treat this API as research code until error guarantees
are validated on representative graph families.

## API Reference

### Core Types

- **`DynamicMinCut`**: Main structure for maintaining minimum cuts
- **`MinCutBuilder`**: Builder pattern for configuration
- **`MinCutResult`**: Result with cut value, edges, and partition
- **`DynamicGraph`**: Thread-safe graph representation
- **`LinkCutTree`**: Dynamic tree data structure
- **`EulerTourTree`**: Alternative dynamic tree structure
- **`HierarchicalDecomposition`**: Tree-based decomposition

### Paper Implementation Types (December 2025)

- **`SubpolynomialMinCut`**: research API; no verified subpolynomial update bound
- **`SubpolyConfig`**: Configuration for subpolynomial parameters (φ, λ_max, levels)
- **`RecourseStats`**: Tracks observed update recourse
- **`MinCutWrapper`**: O(log n) instance manager with geometric ranges
- **`ProperCutInstance`**: Trait for bounded-range cut solvers
- **`BoundedInstance`**: Production bounded-range implementation
- **`DeterministicLocalKCut`**: BFS-based local minimum cut oracle
- **`WitnessHandle`**: Compact cut certificate using RoaringBitmap
- **`ClusterHierarchy`**: Recursive cluster decomposition
- **`FragmentingAlgorithm`**: Handles disconnected subgraphs

### Integration Types

- **`RuVectorGraphAnalyzer`**: Similarity/k-NN graph analysis
- **`CommunityDetector`**: Recursive min-cut community detection
- **`GraphPartitioner`**: Bisection-based graph partitioning

### Compact/Parallel Types (feature: `agentic`)

- **`CompactCoreState`**: 6.7KB per-core state
- **`BitSet256`**: 32-byte membership set
- **`SharedCoordinator`**: Lock-free multi-core coordination
- **`CoreExecutor`**: Per-core execution context
- **`ResultAggregator`**: Multi-core result collection

### Monitoring Types (feature: `monitoring`)

- **`MinCutMonitor`**: Event-driven monitoring system
- **`MonitorBuilder`**: Builder for monitor configuration
- **`MinCutEvent`**: Event notification
- **`EventType`**: Types of events (cut changes, thresholds, etc.)
- **`Threshold`**: Configurable alert thresholds

See [API.md](docs/API.md) for complete API documentation with examples.

## Benchmarks

### Reproducibility

```
Environment: Linux 6.8.0 (x86_64), Rust 1.77+, 8-core AMD EPYC
Commit: c7a3e73d (main)
Command: cargo bench --features full -p ruvector-mincut
Graph: Synthetic path + cross-edges (see examples/subpoly_bench.rs)
```

Results on a graph with 10,000 vertices:

```
Dynamic MinCut Operations:
  build/10000_vertices     time: [152.3 ms 155.1 ms 158.2 ms]
  insert_edge/connected    time: [8.234 µs 8.445 µs 8.671 µs]
  delete_edge/tree_edge    time: [12.45 µs 12.89 µs 13.34 µs]
  query_min_cut           time: [125.2 ns 128.7 ns 132.5 ns]

Link-Cut Tree Operations:
  link                    time: [245.6 ns 251.3 ns 257.8 ns]
  cut                     time: [289.4 ns 295.7 ns 302.1 ns]
  find_root               time: [198.7 ns 203.2 ns 208.5 ns]
  connected               time: [412.3 ns 421.8 ns 431.9 ns]

Sparsification (ε=0.1):
  benczur_karger/10000    time: [45.23 ms 46.78 ms 48.45 ms]
  sparsification_ratio    value: 0.23 (77% reduction)
```

Run benchmarks:

```bash
cargo bench --features full
```

## Examples

Explore the [examples/](examples/) directory:

```bash
# Basic minimum cut operations
cargo run --example basic

# Graph sparsification
cargo run --example sparsify_demo

# Real-time monitoring
cargo run --example monitoring --features monitoring

# Performance benchmarking
cargo run --example benchmark --release
```

## Use Cases

### Network Reliability

Find the minimum number of edges whose removal disconnects a network:

```rust
let mut network = MinCutBuilder::new()
    .with_edges(network_topology)
    .build()?;

let vulnerability = network.min_cut_value();
let critical_edges = network.cut_edges();
```

### Community Detection

Identify weakly connected communities in social networks:

```rust
use ruvector_mincut::{CommunityDetector, DynamicGraph};
use std::sync::Arc;

let graph = Arc::new(DynamicGraph::new());
// Add edges for two triangles connected by weak edge
graph.insert_edge(0, 1, 1.0)?;
graph.insert_edge(1, 2, 1.0)?;
graph.insert_edge(2, 0, 1.0)?;
graph.insert_edge(3, 4, 1.0)?;
graph.insert_edge(4, 5, 1.0)?;
graph.insert_edge(5, 3, 1.0)?;
graph.insert_edge(2, 3, 0.1)?; // Weak bridge

let mut detector = CommunityDetector::new(graph);
let communities = detector.detect(2);  // min community size = 2
println!("Found {} communities", communities.len());
```

### Graph Partitioning

Partition graphs for distributed processing:

```rust
use ruvector_mincut::{GraphPartitioner, DynamicGraph};
use std::sync::Arc;

let graph = Arc::new(DynamicGraph::new());
// Build your graph...

let partitioner = GraphPartitioner::new(graph, 4); // 4 partitions
let partitions = partitioner.partition();
let edge_cut = partitioner.edge_cut(&partitions);
println!("Partitioned into {} groups with {} edge cuts", partitions.len(), edge_cut);
```

### Similarity Graph Analysis

Analyze k-NN or similarity graphs:

```rust
use ruvector_mincut::RuVectorGraphAnalyzer;

// Build from similarity matrix
let similarities = vec![/* ... */];
let mut analyzer = RuVectorGraphAnalyzer::from_similarity_matrix(
    &similarities,
    100,   // num_vectors
    0.8    // threshold
);

let connectivity = analyzer.min_cut();
let bridges = analyzer.find_bridges();
println!("Graph connectivity: {}, bridges: {:?}", connectivity, bridges);
```

### Image Segmentation

Segment images by finding minimum cuts in pixel graphs:

```rust
let pixel_graph = build_pixel_graph(image);
let segmenter = MinCutBuilder::new()
    .exact()
    .build()?;

let (foreground, background) = segmenter.partition();
```

---

## 🔧 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ruvnet/ruvector.git
cd ruvector/crates/ruvector-mincut

# Run tests (448+ passing)
cargo test --all-features

# Run benchmarks
cargo bench --features full

# Check documentation
cargo doc --open --all-features
```

### Testing

The crate includes comprehensive tests:

- Unit tests for each module
- Integration tests for end-to-end workflows
- Property-based tests using `proptest`
- Benchmarks using `criterion`

```bash
# Run all tests
cargo test --all-features

# Run specific test suite
cargo test --test integration_tests

# Run with logging
RUST_LOG=debug cargo test
```

---

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## 🙏 Acknowledgments

This implementation is based on research in dynamic graph algorithms:

- **Link-Cut Trees**: Sleator & Tarjan (1983)
- **Dynamic Minimum Cut**: Thorup (2007)
- **Graph Sparsification**: Benczúr & Karger (1996)
- **Hierarchical Decomposition**: Thorup & Karger (2000)
- **Deterministic Dynamic Min-Cut**: El-Hayek, Henzinger & Li (December 2025)

---

## 📚 References

1. Sleator, D. D., & Tarjan, R. E. (1983). "A Data Structure for Dynamic Trees". *Journal of Computer and System Sciences*.

2. Thorup, M. (2007). "Fully-Dynamic Min-Cut". *Combinatorica*.

3. Benczúr, A. A., & Karger, D. R. (1996). "Approximating s-t Minimum Cuts in Õ(n²) Time". *STOC*.

4. Henzinger, M., & King, V. (1999). "Randomized Fully Dynamic Graph Algorithms with Polylogarithmic Time per Operation". *JACM*.

5. El-Hayek, A., Henzinger, M., & Li, J. (December 2025). "Deterministic and Exact Fully-dynamic Minimum Cut of Superpolylogarithmic Size in Subpolynomial Time". *arXiv:2512.13105*. **[First deterministic exact fully-dynamic min-cut algorithm for cuts above polylogarithmic size]**

6. Goranci, G., et al. (October 2025). "Dynamic Connectivity with Expected Polylogarithmic Worst-Case Update Time". *arXiv:2510.08297*. **[O(log³ n) worst-case dynamic connectivity]**

7. Li, J., et al. (December 2024). "Approximate Min-Cut in All Cut Sizes". *SODA 2025, arXiv:2412.15069*. **[(1+ε)-approximate min-cut for all sizes]**

---

## 🔗 Related Crates & Resources

### RuVector Ecosystem

- [`ruvector-core`](../ruvector-core): Core vector operations and SIMD primitives
- [`ruvector-graph`](../ruvector-graph): Graph database with vector embeddings
- [`ruvector-index`](../ruvector-index): High-performance vector indexing

### Links

- 🌐 **Website**: [ruv.io](https://ruv.io) — AI Infrastructure & Research
- 📦 **Crates.io**: [ruvector-mincut](https://crates.io/crates/ruvector-mincut)
- 📖 **Documentation**: [docs.rs/ruvector-mincut](https://docs.rs/ruvector-mincut)
- 🐙 **GitHub**: [github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- 📝 **Issues**: [Report bugs or request features](https://github.com/ruvnet/ruvector/issues)

---

<div align="center">

**Built with ❤️ by [ruv.io](https://ruv.io)**

**Status**: Production-ready • **Version**: 0.1.29 • **Rust Version**: 1.77+ • **Tests**: 448+ passing

*Keywords: rust, minimum-cut, dynamic-graph, graph-algorithm, connectivity, network-analysis, subpolynomial, real-time, wasm, simd*

</div>
