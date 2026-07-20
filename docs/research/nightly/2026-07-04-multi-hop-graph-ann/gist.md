# MHGAR: Multi-Hop Graph-Anchored Retrieval for Rust Vector DBs

## The Problem

Approximate nearest-neighbor (ANN) search fails for entities that are
graph-connected to relevant hubs but live in a semantically distant region of
vector space — the *CrossCluster regime*.  Every structured knowledge base has
this: drug–adverse-effect pairs, legal case citations, product accessories.

## What We Built

`ruvector-mhgar`: three retrieval variants in a single in-process Rust crate.
No Python. No RPC. No multi-process orchestration.

```rust
// 79 pp recall improvement over pure ANN in CrossCluster
let retriever = OneHopExpander {
    index: &flat_index,
    graph: &knowledge_graph,
    initial_k: 10,
    num_seeds_to_expand: 1,   // only top ANN result drives graph traversal
    hop_discount: 0.5,         // graph-found entities score at 50% of raw dist
};
```

## The Key Finding

**Graph expansion without graph-weight scoring provides ZERO recall gain**
in CrossCluster.  Random satellites are indistinguishable from any other random
entity by cosine distance alone.  The `hop_discount` parameter creates the
score gap that makes graph-reachable entities rank above random noise.

This is reproducible and tested (`naive_expansion_no_discount_matches_vector_only`).

## Benchmark Results (50 hubs × 10 sats, D=64, 200 queries)

| Variant | Recall@10 (CrossCluster) | Mean latency |
|---------|--------------------------|--------------|
| VectorOnly | 0.113 | 37 µs |
| **OneHopExpander** | **0.900** | **42 µs** |
| CoherenceGatedHopper | 0.898 | 56 µs |

**7.97× recall improvement at 1.12× latency cost.**

## SOTA Gap

All 2026 GraphRAG papers (HippoRAG2, PathRAG, BridgeRAG, HopRAG) use Python,
multi-process pipelines, and pure cosine reranking after graph expansion.
None provide graph-edge–weighted reranking in a single in-process binary.

## Codebase

- `src/retriever.rs` — three variants implementing the `Retriever` trait
- `src/synth.rs` — deterministic hub-satellite synthetic dataset
- `src/coherence.rs` — adaptive stopping criterion
- `src/bin/benchmark.rs` — full benchmark with two scenarios and acceptance gates
- `tests/integration.rs` — 12 tests including the naive-expansion null result
