# ruvix-vecgraph

Kernel-resident vector and graph stores for the RuVix Cognition Kernel (ADR-087).

## Overview

This crate implements the Vector/Graph Kernel Objects from ADR-087 Section 4.3. Unlike conventional kernels where all data structures are userspace constructs, RuVix makes vector stores and graph stores kernel-resident objects.

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Kernel-resident** | Vector data lives in kernel-managed slab regions |
| **Capability-protected** | All access requires valid capabilities |
| **Proof-gated** | All mutations require cryptographic proof |
| **Coherence-aware** | Metadata co-located with each vector |
| **Witness-emitting** | Every mutation emits an attestation |

## Syscalls Implemented

| Syscall | Description | Rights Required |
|---------|-------------|-----------------|
| `vector_get` | Read vector data and coherence metadata | READ |
| `vector_put_proved` | Write vector with proof verification | WRITE + PROVE |
| `graph_apply_proved` | Apply graph mutation with proof | WRITE + PROVE |

## Components

### KernelVectorStore

Main vector storage engine:

```rust
use ruvix_vecgraph::{KernelVectorStore, VectorStoreBuilder};
use ruvix_types::{VectorKey, ProofToken};

// Create a vector store with 768 dimensions and 10000 capacity
let store = VectorStoreBuilder::new(768, 10000)
    .with_proof_policy(ProofPolicy::standard())
    .build(backing)?;

// Read a vector (no proof required, only READ capability)
let (data, meta) = store.vector_get(key, cap)?;

// Write a vector (proof required, WRITE + PROVE capability)
let attestation = store.vector_put_proved(key, &data, proof, cap)?;
```

### KernelGraphStore

Graph storage with mincut-aware mutations:

```rust
use ruvix_vecgraph::{KernelGraphStore, GraphStoreBuilder};
use ruvix_types::GraphMutation;

let store = GraphStoreBuilder::new(1000) // 1000 nodes max
    .with_proof_policy(ProofPolicy::deep())
    .build(backing)?;

// Apply graph mutation with proof
let mutation = GraphMutation::AddEdge { from: 0, to: 1, weight: 0.95 };
let attestation = store.graph_apply_proved(mutation, proof, cap)?;
```

### HnswRegion

HNSW index backed by slab allocation:

```rust
use ruvix_vecgraph::{HnswRegion, HnswConfig};

let config = HnswConfig {
    m: 16,           // Max connections per node
    ef_construction: 200,
    max_elements: 10000,
};
let hnsw = HnswRegion::new(config, slab_region)?;

// Insert vector
hnsw.insert(key, &vector, cap)?;

// Search nearest neighbors
let neighbors = hnsw.search(&query, k, ef_search)?;
```

### CoherenceTracker

Track coherence metadata for vectors:

```rust
use ruvix_vecgraph::{CoherenceTracker, CoherenceConfig};

let tracker = CoherenceTracker::new(CoherenceConfig::default());

// Get coherence score (0.0 - 1.0)
let score = tracker.score(key)?;

// Update on mutation
tracker.update(key, timestamp_ns)?;
```

### WitnessLog

Record attestations for all mutations:

```rust
use ruvix_vecgraph::{WitnessLog, WitnessEntry};

let mut log = WitnessLog::new();

// Every successful mutation returns an attestation
let entry = log.append(attestation)?;
println!("Witness: {:?}", entry.hash);
```

## SIMD Distance Functions

Optimized distance computations:

```rust
use ruvix_vecgraph::{cosine_similarity, euclidean_distance_squared, dot_product, l2_norm};

let a = &[1.0f32; 768];
let b = &[0.5f32; 768];

let cos_sim = cosine_similarity(a, b);
let l2_dist = euclidean_distance_squared(a, b);
let dot = dot_product(a, b);
let norm = l2_norm(a);
```

## Statistics

With the `stats` feature enabled:

```rust
use ruvix_vecgraph::VecGraphStats;

let stats = store.stats();
println!("Vector reads: {}", stats.vector_reads);
println!("Vector writes: {}", stats.vector_writes);
println!("Writes rejected: {}", stats.vector_writes_rejected);
println!("Witness entries: {}", stats.witness_entries);
```

## Features

- `std` (default): Enable standard library support
- `alloc`: Enable alloc crate support
- `stats`: Enable statistics collection
- `coherence`: Enable coherence scoring

## Integration with RuVix

This crate integrates with:

- `ruvix-types`: Core type definitions (`VectorKey`, `GraphHandle`, `CoherenceMeta`)
- `ruvix-cap`: Capability checking for READ/WRITE/PROVE rights
- `ruvix-region`: Slab regions for vector storage
- `ruvix-proof`: Proof verification for mutations
- `ruvector-coherence`: Spectral coherence scoring (optional)

## License

MIT OR Apache-2.0
