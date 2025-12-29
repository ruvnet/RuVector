# RuVector DAG - Neural Self-Learning DAG

A high-performance neural DAG learning system for query optimization in RuVector.

## Features

- **7 DAG Attention Mechanisms**: Topological, Causal Cone, Critical Path, MinCut Gated, Hierarchical Lorentz, Parallel Branch, Temporal BTSP
- **SONA Learning**: Self-Optimizing Neural Architecture with MicroLoRA adaptation
- **Subpolynomial MinCut**: O(n^0.12) bottleneck detection
- **Self-Healing**: Autonomous anomaly detection and repair
- **QuDAG Integration**: Quantum-resistant distributed pattern learning

## Quick Start

```rust
use ruvector_dag::{QueryDag, OperatorNode, OperatorType};
use ruvector_dag::attention::{TopologicalAttention, DagAttention};

// Build a query DAG
let mut dag = QueryDag::new();
let scan = dag.add_node(OperatorNode::hnsw_scan(0, "vectors_idx", 64));
let filter = dag.add_node(OperatorNode::filter(1, "score > 0.5"));
let result = dag.add_node(OperatorNode::new(2, OperatorType::Result));

dag.add_edge(scan, filter).unwrap();
dag.add_edge(filter, result).unwrap();

// Compute attention scores
let attention = TopologicalAttention::new(Default::default());
let scores = attention.forward(&dag).unwrap();
```

## Modules

- `dag` - Core DAG data structures and algorithms
- `attention` - 7 attention mechanisms for node importance scoring
- `sona` - Self-Optimizing Neural Architecture with adaptive learning
- `mincut` - Subpolynomial bottleneck detection and optimization
- `healing` - Self-healing system with anomaly detection
- `qudag` - QuDAG network integration for distributed learning

## Core Components

### DAG (Directed Acyclic Graph)

The `QueryDag` structure represents query execution plans as directed acyclic graphs. Each node represents an operator (scan, filter, join, etc.) and edges represent data flow.

```rust
use ruvector_dag::{QueryDag, OperatorNode, OperatorType};

let mut dag = QueryDag::new();
let scan = dag.add_node(OperatorNode::seq_scan(0, "users"));
let filter = dag.add_node(OperatorNode::filter(1, "age > 18"));
dag.add_edge(scan, filter).unwrap();
```

### Attention Mechanisms

Seven different attention mechanisms to compute node importance:

1. **Topological**: Position-based importance with depth decay
2. **Causal Cone**: Focus on downstream dependencies
3. **Critical Path**: Emphasize execution bottlenecks
4. **MinCut Gated**: Flow-aware importance gating
5. **Hierarchical Lorentz**: Hyperbolic geometry for hierarchies
6. **Parallel Branch**: Multi-branch execution awareness
7. **Temporal BTSP**: Temporal backward trajectory sampling

```rust
use ruvector_dag::attention::{TopologicalAttention, DagAttention};

let attention = TopologicalAttention::new(Default::default());
let scores = attention.forward(&dag)?;
```

### SONA (Self-Optimizing Neural Architecture)

Adaptive learning engine that improves query optimization over time:

```rust
use ruvector_dag::sona::DagSonaEngine;

let mut sona = DagSonaEngine::new(256);

// Pre-query: Get enhanced embedding
let enhanced = sona.pre_query(&dag);

// Execute query...

// Post-query: Record trajectory
sona.post_query(&dag, execution_time, baseline_time, "topological");

// Background learning
sona.background_learn();
```

### MinCut Optimization

Subpolynomial bottleneck detection:

```rust
use ruvector_dag::mincut::{DagMinCutEngine, MinCutConfig};

let mut engine = DagMinCutEngine::new(MinCutConfig::default());
let analysis = engine.analyze_bottlenecks(&dag)?;

for bottleneck in &analysis.bottlenecks {
    println!("Bottleneck at nodes {:?}: capacity {}",
        bottleneck.cut_nodes, bottleneck.capacity);
}
```

### Self-Healing

Autonomous anomaly detection and repair:

```rust
use ruvector_dag::healing::{HealingOrchestrator, AnomalyConfig};

let mut orchestrator = HealingOrchestrator::new();

orchestrator.add_detector("query_latency", AnomalyConfig {
    z_threshold: 3.0,
    window_size: 100,
    min_samples: 10,
});

// Observe metrics
orchestrator.observe("query_latency", latency);

// Run healing cycle
let result = orchestrator.run_cycle();
println!("Detected: {}, Repaired: {}",
    result.anomalies_detected, result.repairs_succeeded);
```

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_usage.rs` - DAG creation and basic operations
- `attention_selection.rs` - Using different attention mechanisms
- `learning_workflow.rs` - SONA learning workflow
- `self_healing.rs` - Self-healing system demonstration

Run examples with:
```bash
cargo run --example basic_usage
cargo run --example attention_selection
cargo run --example learning_workflow
cargo run --example self_healing
```

## Performance Targets

| Component | Target |
|-----------|--------|
| Attention (100 nodes) | <100μs |
| MicroLoRA adaptation | <100μs |
| Pattern search (10K) | <2ms |
| MinCut update | O(n^0.12) |
| Anomaly detection | <50μs |

## Architecture

```
┌─────────────────────────────────────────────────┐
│            Query DAG Layer                       │
│  (Operators, Edges, Topological Sort)           │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                      │
┌───────▼─────────┐    ┌──────▼──────────┐
│   Attention     │    │    MinCut       │
│   Mechanisms    │    │   Optimization  │
│   (7 types)     │    │  (Bottlenecks)  │
└───────┬─────────┘    └──────┬──────────┘
        │                      │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │    SONA Engine      │
        │ (Neural Learning)   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Self-Healing      │
        │  (Orchestrator)     │
        └─────────────────────┘
```

## Development

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench

# Check documentation
cargo doc --open
```

## Integration with RuVector

This crate is part of the RuVector ecosystem and integrates with:

- `ruvector-core` - Core vector operations
- `ruvector-qudag` - Quantum-resistant distributed learning
- `ruvector-hooks` - Intelligence hooks for adaptive behavior

## License

Apache-2.0 OR MIT
