# exo-temporal

Temporal memory coordinator with causal structure for the EXO-AI 2025 cognitive substrate.

## Overview

This crate implements a biologically-inspired temporal memory system with:

- **Short-term buffer**: Volatile memory for recent patterns
- **Long-term store**: Consolidated memory with strategic forgetting
- **Causal graph**: Tracks antecedent relationships between patterns
- **Memory consolidation**: Salience-based filtering (frequency, recency, causal importance, surprise)
- **Predictive anticipation**: Pre-fetching based on sequential patterns, temporal cycles, and causal chains

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  TemporalMemory                         │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│ │ Short-Term  │  │ Long-Term   │  │   Causal    │      │
│ │   Buffer    │→ │    Store    │  │    Graph    │      │
│ └─────────────┘  └─────────────┘  └─────────────┘      │
│        ↓                ↑                 ↑             │
│ ┌─────────────────────────────────────────────┐         │
│ │          Consolidation Engine               │         │
│ │  (Salience computation & filtering)         │         │
│ └─────────────────────────────────────────────┘         │
│        ↓                                                │
│ ┌─────────────────────────────────────────────┐         │
│ │       Anticipation & Prefetch               │         │
│ └─────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Modules

- **`types`**: Core type definitions (Pattern, Query, SubstrateTime, etc.)
- **`causal`**: Causal graph for tracking antecedent relationships
- **`short_term`**: Volatile short-term memory buffer
- **`long_term`**: Consolidated long-term memory store
- **`consolidation`**: Memory consolidation with salience computation
- **`anticipation`**: Predictive pre-fetching and query anticipation

## Key Algorithms

### Causal Cone Query (Pseudocode 3.1)

Retrieves patterns within causal light-cone constraints:

```rust
let results = memory.causal_query(
    &query,
    reference_time,
    CausalConeType::Past,
);
```

- Filters by time range (Past, Future, or LightCone)
- Computes causal distance via graph traversal
- Ranks by combined similarity, temporal, and causal relevance

### Memory Consolidation (Pseudocode 3.2)

Transfers patterns from short-term to long-term based on salience:

```rust
let result = memory.consolidate();
```

Salience factors:
- **Frequency**: Access count (logarithmic scaling)
- **Recency**: Exponential decay from last access
- **Causal importance**: Out-degree in causal graph
- **Surprise**: Novelty compared to existing patterns

### Predictive Anticipation (Pseudocode 3.3)

Pre-fetches likely future queries:

```rust
memory.anticipate(&[
    AnticipationHint::SequentialPattern { recent: vec![id1, id2] },
    AnticipationHint::CausalChain { context: id3 },
]);
```

Strategies:
- **Sequential patterns**: If A then B learned sequences
- **Temporal cycles**: Time-of-day / day-of-week patterns
- **Causal chains**: Downstream effects in causal graph

## Usage Example

```rust
use exo_temporal::{TemporalMemory, TemporalConfig, Pattern, Metadata};

// Create temporal memory
let memory = TemporalMemory::new(TemporalConfig::default());

// Store pattern with causal context
let pattern = Pattern::new(vec![1.0, 2.0, 3.0], Metadata::new());
let id = memory.store(pattern, &[]).unwrap();

// Retrieve pattern
let retrieved = memory.get(&id).unwrap();

// Causal query
let query = Query::from_embedding(vec![1.0, 2.0, 3.0])
    .with_origin(id)
    .with_k(10);

let results = memory.causal_query(
    &query,
    SubstrateTime::now(),
    CausalConeType::Past,
);

// Trigger consolidation
let consolidation_result = memory.consolidate();

// Strategic forgetting
memory.forget();

// Get statistics
let stats = memory.stats();
println!("Short-term: {} patterns", stats.short_term.size);
println!("Long-term: {} patterns", stats.long_term.size);
println!("Causal edges: {}", stats.causal_graph.num_edges);
```

## Implementation Notes

### Pseudocode Alignment

This implementation follows the pseudocode in `PSEUDOCODE.md`:

- **Section 3.1**: `causal_query` method implements causal cone filtering
- **Section 3.2**: `consolidate` function implements salience-based consolidation
- **Section 3.3**: `anticipate` function implements predictive pre-fetching

### Concurrency

- Uses `DashMap` for concurrent access to patterns and indices
- `parking_lot::RwLock` for read-heavy workloads
- Thread-safe throughout for multi-threaded substrate operations

### Performance

- **O(log n)** temporal range queries via binary search on sorted index
- **O(k × d)** similarity search where k = results, d = embedding dimension
- **O(n²)** worst-case for causal distance via Dijkstra's algorithm
- **O(1)** prefetch cache lookup

## Dependencies

- `exo-core`: Core traits and types (to be implemented)
- `dashmap`: Concurrent hash maps
- `parking_lot`: Efficient synchronization primitives
- `chrono`: Temporal handling
- `petgraph`: Graph algorithms for causal distance
- `serde`: Serialization support

## Future Enhancements

- [ ] Temporal Knowledge Graph (TKG) integration (mentioned in ARCHITECTURE.md)
- [ ] Relativistic light cone with spatial distance
- [ ] Advanced consolidation policies (sleep-inspired replay)
- [ ] Distributed temporal memory via CRDT synchronization
- [ ] GPU-accelerated similarity search

## References

- ARCHITECTURE.md: Section 2.5 (Temporal Memory Coordinator)
- PSEUDOCODE.md: Section 3 (Temporal Memory Coordinator)
- Research: Zep-inspired temporal knowledge graphs, IIT consciousness metrics

## License

MIT OR Apache-2.0
