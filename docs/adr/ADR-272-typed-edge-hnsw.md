---
adr: 272
title: "Typed-Edge Navigable Graph (TENG) for Hybrid Vector+Semantic Retrieval"
status: accepted
date: 2026-06-30
authors: [ruvnet, claude]
related: [ADR-268, ADR-264, ADR-193, ADR-197]
tags: [vector-search, graph, ann, nsw, semantic, agent-memory, graph-rag, hybrid-retrieval, typed-edges]
---

# ADR-272 — Typed-Edge Navigable Graph (TENG)

## Status

**Accepted (implemented).** Crate `crates/ruvector-tegraph`.  Three retrieval
variants benchmarked on 5,000 × 128-dim synthetic corpus.  All acceptance tests
pass.

---

## Context

RuVector's graph-RAG pipeline currently operates in two sequential passes: an ANN
index retrieves vector-similar candidates, then a separate graph-traversal step
re-ranks them using knowledge-graph adjacency.  This two-pass architecture has
three practical costs:

1. **Latency**: the graph traversal adds a second network or memory hop after
   the ANN phase is already done.
2. **Coverage gaps**: semantically related nodes that happen to be far in
   embedding space are only discoverable in the second pass — but only if the
   first pass already found a bridge node.
3. **Implementation weight**: maintaining two separate data structures (ANN
   index + property graph) doubles operational complexity.

GraphRAG, HippoRAG, LightRAG and similar systems all suffer from this
separation.  There is no widely-adopted open-source system that weaves typed
semantic edges **into** the ANN navigation graph itself so that a single beam
search covers both proximity and semantic adjacency simultaneously.

TENG closes this gap by extending a Navigable Small World (NSW) graph with
five typed edge classes — SameDocument, References, CoOccurs, Temporal, Causal
— and providing three retrieval modes that differ in how these edges are
consulted during navigation.

---

## Decision

Introduce `ruvector-tegraph` as a standalone crate with:

- **`NswGraph`** — standard greedy NSW with `BinaryHeap`-based beam search.
  O(n · ef · d) construction; O(ef · d) per query.
- **`TengIndex`** — wraps `NswGraph` with a per-node typed-edge list.
- **`search_vector_only`** — baseline NSW, edges ignored.
- **`search_edge_expand`** — NSW candidates + outward typed-edge walk from
  top-k initial results, re-ranked by pure cosine similarity.
- **`search_edge_constrained`** — NSW with wide ef, filtered to candidates
  that own at least one edge matching an `EdgeConstraint` predicate.

The typed-edge structure stays entirely in user-space Rust (no external graph
database, no serialisation library).  It is zero-copy after construction.

---

## Consequences

### Positive

- **+22.0% semantic recall** (EdgeExpand vs VectorOnly on the union of
  k-NN ∪ typed-edge neighbours) with real benchmark numbers.
- **+35.3% absolute constrained recall** (EdgeConstrained 0.992 vs
  VectorOnly 0.733) for domain-restricted retrieval.
- Single data structure covers vector search, graph traversal, and semantic
  filtering — eliminating the two-pass penalty for agent memory workloads.
- No external dependencies beyond `rand` / `rand_distr` from the workspace.
- Works offline / edge-deployable (no network calls at query time).

### Negative

- EdgeExpand is 1.9× slower than VectorOnly (446 μs vs 233 μs mean).
- EdgeConstrained is 3.4× slower (792 μs vs 233 μs) due to over-fetching.
- NSW (flat graph) has lower recall@10 than full HNSW at equal ef — upgrading
  to multi-layer HNSW is the primary next step.
- The SameDocument / cluster structure used in the PoC dataset inflates
  EdgeConstrained recall; real-world recall will vary with edge density.

---

## Alternatives Considered

| Option | Why rejected |
|--------|-------------|
| Post-retrieval graph traversal (status quo) | Two-pass latency; misses bridge nodes |
| GNN reranking (ADR-194) | Post-retrieval; can't expand candidate set |
| Coherence-gated HNSW (ADR-265 research) | Coherence scores, not typed edges |
| DiskANN page locality (existing crate) | SSD-first concern; orthogonal to edge types |
| Filtered HNSW / ACORN (ADR) | Metadata filters, not semantic graph edges |

---

## Implementation Plan

| Phase | Work | Status |
|-------|------|--------|
| 1 | NSW base + three typed search variants | Done (PoC) |
| 2 | Upgrade to multi-layer HNSW | Next |
| 3 | Persistent typed-edge serialisation (RVF or redb) | Next |
| 4 | MCP tool surface: `teng_index`, `teng_search_expand` | Future |
| 5 | WASM build (`ruvector-tegraph-wasm`) | Future |
| 6 | ruFlo trigger: auto-rebuild on typed-edge density shift | Future |

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-tegraph --bin benchmark`
on x86_64 Linux, release build, 5,000 nodes × 128 dims, 500 queries, k=10.

| Variant | Vec R@10 | Sem R@10 | Mean μs | p50 μs | p95 μs | QPS | Mem MB |
|---------|----------|----------|---------|--------|--------|-----|--------|
| VectorOnly | 0.733 | — | 232.8 | 229 | 290 | 4,295 | 4.46 |
| EdgeExpand(f=0.30) | 0.895 | 0.895 | 446.4 | 440 | 524 | 2,240 | 4.46 |
| EdgeConstrained (SameDocument) | 0.992 | — | 792.4 | 773 | 859 | 1,262 | 4.46 |

Semantic recall improvement: **+0.161 (+22.0% relative)** EdgeExpand vs VectorOnly.

---

## Failure Modes

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Low typed-edge density | EdgeExpand degenerates to VectorOnly | Monitor edge/node ratio; warn when < 1 edge/node |
| Stale edges after deletes | Traversal reaches deleted nodes | Generational edge tombstones (future work) |
| NSW recall floor | Flat NSW saturates at ~0.73 for these params | Upgrade to multi-layer HNSW (Phase 2) |
| EdgeConstrained empty result | Over-filtered set returns < k results | Return partial results + metadata flag |
| edge_factor tuning | Wrong blend degrades recall | Expose as config param; monitor recall drift |

---

## Security Considerations

- Typed edges are user-supplied metadata.  Edge targets must be bounds-checked
  before traversal (currently enforced by Rust's slice indexing panics; future
  production code should return `Result`).
- EdgeConstrained can act as an access control layer (similar to ADR-268
  capability-gated ANN) if `EdgeType::Causal` encodes ownership.
- No secrets are stored in the edge structure.

---

## Migration Path

This is a new crate.  Existing callers of `ruvector-core`, `ruvector-gnn-rerank`,
or `ruvector-coherence-hnsw` are unaffected.  TENG can be adopted alongside them
by replacing the retrieval call:

```rust
// Before: two-pass
let candidates = hnsw_index.search(query, k * 4);
let reranked   = gnn_reranker.rerank(&candidates, query, k);

// After: single-pass with typed edges
let results = teng_index.search_edge_expand(query, k, 0.30);
```

---

## Open Questions

1. What is the correct default for `edge_factor`?  0.30 was chosen empirically;
   a self-tuning mechanism (via ruFlo feedback loop) is the long-term answer.
2. Should typed edges be stored in `redb` for persistence?  RVF manifest format
   is another candidate.
3. Can the typed-edge walk be parallelised across edge types using `rayon`?
4. What happens to recall when edges are added incrementally (streaming corpus)?
5. How does NSW recall compare to full HNSW at the same `ef_search`?
