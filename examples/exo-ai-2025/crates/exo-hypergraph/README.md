# exo-hypergraph

Hypergraph substrate for higher-order relational reasoning in the EXO-AI cognitive substrate.

## Features

- **Hyperedge Support**: Relations spanning multiple entities (not just pairwise)
- **Topological Data Analysis**: Persistent homology and Betti number computation (interface ready, full algorithms to be implemented)
- **Sheaf Theory**: Consistency checks for distributed data structures
- **Thread-Safe**: Lock-free concurrent access using DashMap

## Architecture

This crate implements the hypergraph layer as described in the EXO-AI architecture:

```
HypergraphSubstrate
├── HyperedgeIndex        # Efficient indexing for hyperedge queries
├── SimplicialComplex     # TDA structures and Betti numbers
└── SheafStructure        # Sheaf-theoretic consistency checking
```

## Usage

```rust
use exo_hypergraph::{HypergraphSubstrate, HypergraphConfig};
use exo_core::{EntityId, Relation, RelationType};

// Create hypergraph
let config = HypergraphConfig::default();
let mut hypergraph = HypergraphSubstrate::new(config);

// Add entities
let e1 = EntityId::new();
let e2 = EntityId::new();
let e3 = EntityId::new();

hypergraph.add_entity(e1, serde_json::json!({"name": "Alice"}));
hypergraph.add_entity(e2, serde_json::json!({"name": "Bob"}));
hypergraph.add_entity(e3, serde_json::json!({"name": "Charlie"}));

// Create 3-way hyperedge (beyond pairwise!)
let relation = Relation {
    relation_type: RelationType::new("collaboration"),
    properties: serde_json::json!({"project": "EXO-AI"}),
};

let hyperedge_id = hypergraph.create_hyperedge(
    &[e1, e2, e3],
    &relation
).unwrap();

// Query topology
let betti = hypergraph.betti_numbers(2);  // Get Betti numbers β₀, β₁, β₂
println!("Topological structure: {:?}", betti);
```

## Topological Queries

### Betti Numbers

Betti numbers are topological invariants that describe the structure:

- **β₀**: Number of connected components
- **β₁**: Number of 1-dimensional holes (loops)
- **β₂**: Number of 2-dimensional holes (voids)

```rust
let betti = hypergraph.betti_numbers(2);
// β₀ = connected components
// β₁ = loops (currently returns 0 - stub)
// β₂ = voids (currently returns 0 - stub)
```

### Persistent Homology (Interface Ready)

The persistent homology interface is implemented, with full algorithm to be added:

```rust
use exo_core::TopologicalQuery;

let query = TopologicalQuery::PersistentHomology {
    dimension: 1,
    epsilon_range: (0.0, 1.0),
};

let result = hypergraph.query(&query).unwrap();
// Returns persistence diagram (currently empty - stub)
```

## Implementation Status

✅ **Complete**:
- Hyperedge creation and indexing
- Entity-to-hyperedge queries
- Simplicial complex construction
- Betti number computation (β₀)
- Sheaf consistency checking
- Thread-safe concurrent access

🚧 **Stub Interfaces** (Complex algorithms, interfaces ready):
- Persistent homology computation (requires boundary matrix reduction)
- Higher Betti numbers (β₁, β₂, ...) require Smith normal form
- Filtration building for persistence

## Dependencies

- `exo-core`: Core types and traits
- `petgraph`: Graph algorithms
- `dashmap`: Concurrent hash maps
- `serde`: Serialization

## License

MIT OR Apache-2.0
