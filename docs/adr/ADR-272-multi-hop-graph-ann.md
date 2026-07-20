# ADR-272: Multi-Hop Graph-Anchored Retrieval (MHGAR)

**Status:** Accepted  
**Date:** 2026-07-04  
**Deciders:** Nightly Research Agent  
**Tags:** retrieval, graph, ann, graphrag, mhgar

---

## Context

Approximate nearest-neighbor search is the dominant retrieval primitive in
RuVector.  It works well when the target entity is semantically close to the
query in embedding space.  However, many real knowledge structures have entities
that are *relationally* connected to a relevant hub while occupying a different
region of vector space (CrossCluster regime).  Examples:

- A drug molecule (hub) and its adverse-effect compounds (satellites)
- A legal case (hub) and its cited precedents (satellites)
- A product (hub) and its compatible accessories (satellites)

In these cases, graph traversal from a vector-found hub is the only reliable
path to the satellite entities.  All 2026 GraphRAG research systems (HippoRAG2,
PathRAG, BridgeRAG, HopRAG, AtomicRAG) address this but do so in Python with
multi-process pipelines.  RuVector has no in-process solution.

---

## Decision

Implement `crates/ruvector-mhgar` with three retrieval variants that can be
compared head-to-head in benchmarks and integrated into the RuVector stack:

### 1. VectorOnlyRetriever
Pure cosine ANN baseline.  Establishes the null hypothesis.

### 2. OneHopExpander

Parameters:
- `initial_k`: number of ANN seeds
- `num_seeds_to_expand`: how many top seeds drive graph traversal (≤ initial_k)
- `hop_discount`: score multiplier for graph-found entities

Scoring: seeds by raw cosine distance; graph-found entities by
`dist × (1 - hop_discount)`.

**Design principle:** `num_seeds_to_expand = 1` is the recommended CrossCluster
setting.  Expanding from all initial seeds floods the candidate pool with
entities from wrong-cluster hubs, eliminating the benefit of graph traversal.

### 3. CoherenceGatedHopper

Extends OneHopExpander with adaptive stopping: measure the mean query-distance
of the visited candidate set after each hop; stop if it falls below
`expansion_threshold`.

**Threshold calibration:** ANN-selected seeds are biased toward entities with
below-average cosine distance (they are the most similar random vectors from
the full pool).  `expansion_threshold = 0.50` accounts for this selection bias
and reliably triggers expansion in CrossCluster (mean ≈ 0.65–0.75) while
correctly stopping in NearHub (mean ≈ 0.05–0.20).

---

## Consequences

### Positive

- **79 pp recall improvement** in CrossCluster at 1.12× latency overhead
  (OneHopExpander, 50 hubs × 10 sats, D=64).
- **Single in-process binary**: no RPC, no Python, no graph database required
  at benchmark scale.
- **Research claim validated and tested**: the naive-expansion null result
  (`hop_discount=0.0` → zero recall gain) is a reproducible, committed test
  (`naive_expansion_no_discount_matches_vector_only`).
- Establishes the `Retriever` trait as the uniform interface for future
  retrieval variants.

### Negative / Trade-offs

- `hop_discount` and `expansion_threshold` require per-dataset calibration;
  no automatic tuning is implemented.
- `FlatIndex` is O(n) scan; production use requires HNSW or DiskANN backing.
- The coherence gate is ineffective if the expansion_threshold is not tuned
  for the specific dataset's ANN selection bias.

### Neutral

- Memory overhead of 62% over vector-only at benchmark scale (85 KB graph
  for 550 entities × 10 sats).
- Graph stored as adjacency list (`HashMap<u32, Vec<Edge>>`); suitable for
  millions of nodes but requires an on-disk format for >10M nodes.

---

## Alternatives Considered

| Alternative | Reason Rejected |
|-------------|-----------------|
| Integrate with external graph DB (Neo4j, Nebula) | Breaks in-process constraint; RPC latency |
| PPR (Personalized PageRank) à la HippoRAG | Requires full graph materialization; O(n²) precompute |
| Re-rank by path length only | Equivalent to hop_discount=1.0 which doesn't account for vector quality |
| Multi-hop without frontier limiting | Floods candidate pool with cross-cluster noise (tested and documented) |

---

## Implementation

- **Crate**: `crates/ruvector-mhgar`
- **Research doc**: `docs/research/nightly/2026-07-04-multi-hop-graph-ann/README.md`
- **Tests**: 12 integration tests, all green
- **Benchmark**: `cargo run --release -p ruvector-mhgar --bin benchmark`
- **Example**: `cargo run --release -p ruvector-mhgar --example mhgar_demo`
