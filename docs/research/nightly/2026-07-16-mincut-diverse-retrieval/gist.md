# Graph-Partition MMR: 3.25× More Diverse Vector Retrieval in Pure Rust

**Tags**: `rust` `vector-search` `ann` `diversity` `mmr` `graph-cut` `rag` `agent-memory`

Standard top-K ANN retrieval returns the k closest vectors — but in clustered
corpora all k results often come from the same semantic neighbourhood.  This
makes RAG pipelines repeat themselves and causes agent-memory recall to fixate
on one past event.

This nightly research note introduces **PartitionMMR**: Maximal Marginal
Relevance extended with a graph-cut-inspired partition penalty.  Implemented
in safe Rust, no dependencies beyond `rand`, benchmark results measured with
`cargo run --release`.

## The Core Idea

```
score(c) = −λ·dist(c, q)                  ← relevance
           + (1−λ)·min_dist_to_selected   ← diversity (standard MMR)
           − penalty · same_partition_count  ← NEW: partition penalty
```

Before greedy selection, we build an ephemeral connectivity graph over the
C-candidate pool (edges where L2 < data-derived threshold), extract connected
components via Union-Find, and add a per-selection deduction whenever a
candidate shares its component with an already-chosen result.

## Why the Threshold Must Use the Full Pool

The threshold is set to `0.55 × mean_pairwise_L2` of the pool.  If you sample
only the nearest N candidates for this mean, you capture intra-cluster distances
only — the threshold lands below the intra-cluster mean and every candidate
becomes a singleton, so the penalty never fires.

Using the full pool gives a bimodal distribution (intra-sub ≈ 2.83, inter-sub
≈ 13.57), whose mean (~11.94) yields threshold ≈ 6.57 — correctly connecting
within-sub pairs while leaving between-sub pairs in separate components.

## Benchmark (real `cargo run --release`, x86_64 Linux)

Dataset: 10 super-clusters × 6 sub-clusters × 50 vectors = 3,000 total, 64 dims.

```
Variant              Mean µs    QPS    MeanDiv    MeanRel
─────────────────────────────────────────────────────────
TopK (baseline)        271.3   3686     2.574      2.234
MMR (λ=0.5)            443.8   2253     5.696      4.017
PartitionMMR           725.8   1378     8.377      6.207
```

- PartitionMMR: **3.25× more diverse** than TopK
- PartitionMMR: **47% more diverse** than plain MMR
- Relevance cost vs MMR: **1.55×** (bounded, accepted)

All 4 acceptance tests PASS.  26 unit tests PASS.

## Ecosystem Fit

```
ruFlo recall step
    └─► ruvector-diverse-retrieval
            ├─► TopKRetriever      (relevance baseline)
            ├─► MmrRetriever       (λ-tunable MMR)
            └─► PartitionMmrRetriever
                    └─► graph.rs (union-find, shared with ruvector-mincut)
                    └─► MCP tool: vector/search_diverse
```

Connects: vector search, graph primitives, agent memory, ruFlo workflows, MCP.

## Key Files

- `crates/ruvector-diverse-retrieval/src/partition_mmr.rs` — core algorithm
- `crates/ruvector-diverse-retrieval/src/graph.rs` — Union-Find + threshold
- `crates/ruvector-diverse-retrieval/src/main.rs` — benchmark binary
- `docs/adr/ADR-272-mincut-diverse-retrieval.md` — architecture decision

## What's Next

1. Replace brute-force pool selection with HNSW beam search
2. Expose as `vector/search_diverse` MCP tool with `diversity_mode` param
3. ruFlo `recall_diverse` workflow step with feedback-loop penalty tuning
4. WASM wrapper for Cognitum Seed / browser

**Branch**: `research/nightly/2026-07-16-mincut-diverse-retrieval`  
**Crate**: `ruvector-diverse-retrieval v0.1.0`
