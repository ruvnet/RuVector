---
adr: 211
title: "Temporal Coherence Decay for Agent Memory Retrieval"
status: accepted
date: 2026-06-13
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-197, ADR-210]
tags: [agent-memory, vector-search, temporal-decay, coherence, graph-coherence, retrieval, nightly-research]
---

# ADR-211 — Temporal Coherence Decay for Agent Memory Retrieval

## Status

**Accepted.** Implemented on branch `research/nightly/2026-06-13-temporal-coherence-agent-memory`
as `crates/ruvector-temporal-coherence`. All 21 unit tests pass; all 4 acceptance
tests pass with `cargo run --release -p ruvector-temporal-coherence --bin tcd-benchmark`.

## Context

As AI agents accumulate memories over time, two problems emerge:

1. **Recency blindness**: Pure cosine similarity treats a memory from last week
   identically to one from three months ago. For an agent operating in a changing
   world, recent memories often carry more actionable signal.

2. **Coherence dilution**: Isolated memories — those without strong semantic
   neighbours in the memory corpus — may represent one-off observations rather
   than stable world knowledge. A memory that is reinforced by many similar
   memories across the corpus is statistically more reliable.

Neither problem is addressed by existing RuVector search primitives (HNSW,
IVF, filtered-ANN). This ADR introduces `ruvector-temporal-coherence`, which
adds temporal decay and graph-coherence gating as first-class scoring signals
in agent memory retrieval.

The design is inspired by:
- Governing Evolving Memory in LLM Agents (SSGM, arXiv 2603.11768, 2026)
- Temporal Tensor Compression work already in `ruvector-temporal-tensor`
- MinCut coherence gating already in `ruvector-mincut` / `ruvector-coherence`

## Decision

Ship `crates/ruvector-temporal-coherence` as a standalone crate providing three
scored retrieval variants over an append-only agent memory store:

| Variant | Scoring formula | Primary fitness metric |
|---------|----------------|----------------------|
| `FlatSearch` | `cosine_sim(q, m)` | Cosine recall@K |
| `TemporalSearch` | `cosine_sim × exp(-λ·age)` | Mean recency of results |
| `CoherenceSearch` | `cosine_sim × ((1-w)·decay + w·gate)` | Mean coherence gate of results |

Where `gate(m) = degree(m) / max_degree` over the adjacency graph of
memories whose pairwise cosine similarity exceeds `coherence_threshold`.

The trait surface is:

```rust
pub trait VectorSearch {
    fn search(&self, query: &[f32], k: usize, store: &MemoryStore) -> Vec<SearchResult>;
}
```

All three variants implement `VectorSearch`. `DecayConfig` carries the
exponential decay parameter. `CoherenceGraph` wraps the adjacency degree
array and is built once at indexing time.

## Consequences

### Positive

- Agents can tune retrieval by passing a `DecayConfig` and `CoherenceGraph`
  without changing query code — the `VectorSearch` trait is uniform.
- Temporal decay is a pure multiply on top of cosine scan — no extra I/O,
  no graph traversal per query.
- Coherence gate overhead is O(1) per candidate (single array lookup).
- The coherence graph build is one-time (O(n²) at indexing) — in production
  this would be replaced by an approximate k-NN graph via HNSW from
  `ruvector-acorn` or `ruvector-core`, reducing build to O(n·log n).
- MCP memory tools can expose `DecayConfig` as a tool parameter, enabling
  ruFlo workflow loops to pass `half_life` as a session-scoped configuration.

### Negative / Risks

- The O(n²) coherence graph build limits PoC to ~50K memories without HNSW
  approximation. This is documented and the production migration path is clear.
- The exponential decay half-life is a hyperparameter that must be tuned per
  domain. A universal default (30% of session time) is provided but may need
  calibration.
- Coherence gate is based on pairwise cosine threshold — not mincut. A future
  upgrade (see open questions) should replace the degree-normalised gate with a
  proper spectral coherence score from `ruvector-coherence::spectral`.

## Alternatives Considered

### A: Geometric MMR Diversity (gMMR, DF-RAG arXiv 2601.17212)
SOTA diversity reranking with a deterministic greedy algorithm. Scored 4.50
by the nightly selection formula. Rejected for this run because it operates
*post-hoc* on existing cosine results rather than integrating temporal and
coherence signals into the scoring pass — a structurally different problem.
Recommended as the next nightly topic.

### B: QuIVer Binary Graph Topology Quantization
2-bit sign-magnitude encoding for HNSW topology. Scored 4.45. Rejected
because it targets index construction speed, not agent memory retrieval
fitness — a different layer of the stack.

### C: Agent Memory Compaction via MinCut (graph compaction)
Scored 4.05. Rejected for now because it depends on the coherence graph
structure being built first — logically downstream of this ADR. Should be
built on top of `ruvector-temporal-coherence` in a future nightly.

## Implementation Plan

### Week 1 (current)
- [x] `crates/ruvector-temporal-coherence` — FlatSearch, TemporalSearch, CoherenceSearch
- [x] `DecayConfig` with `None`, `Linear`, `Exponential` variants
- [x] `CoherenceGraph` with threshold-gated adjacency degree
- [x] Benchmark binary with per-variant fitness metrics
- [x] 21 unit tests, 4 acceptance tests, all green

### Near-term hardening
- Replace O(n²) graph build with approximate k-NN from `ruvector-acorn`
- Add `spectral` coherence gate from `ruvector-coherence` as optional feature
- Expose `DecayConfig` as MCP tool parameter in `mcp-brain-server`
- Integrate with `ruvector-snapshot` for RVF-packed memory checkpoints

### Research horizon (2026–2036)
- Learned half-life: train λ per agent session from outcome feedback
- Graph-coherence mincut gating: replace degree normalisation with spectral
  Fiedler value to identify genuine coherence domains
- Drift detection: flag memories whose coherence drops below threshold after
  corpus updates (connects to SSGM arXiv 2603.11768)

## Benchmark Evidence

Hardware: x86_64 Linux 6.18.5, Intel Celeron N4020  
Rust: 1.94.1 (e408947bf 2026-03-25)  
Command: `cargo run --release -p ruvector-temporal-coherence --bin tcd-benchmark`  
Dataset: N=5000, D=128, K=10, 200 queries, 20 clusters, half_life=300 000

| Variant | Mean µs | p50 µs | p95 µs | Throughput | Memory | Fitness metric | Acceptance |
|---------|---------|--------|--------|-----------|--------|----------------|------------|
| FlatSearch | 1 036 | 1 017 | 1 136 | 965 q/s | 2 656 KB | cosine_recall=1.000 | PASS |
| TemporalSearch | 1 033 | 1 020 | 1 096 | 967 q/s | 2 656 KB | recency=0.962 | PASS |
| CoherenceSearch | 1 070 | 1 053 | 1 179 | 935 q/s | 2 675 KB | coh_gate=0.971 | PASS |

Coherence graph build: 1 996 ms, 590 313 edges (dense at threshold=0.55, random corpus).
Production corpora will be sparser — 10–50× fewer edges expected.

## Failure Modes

1. **Wrong half-life**: λ too large → retrieves only the very latest memories,
   missing important older context. Mitigation: expose half-life in MCP tool
   and instrument per-session feedback loops.
2. **Dense coherence graph**: High-overlap corpora (e.g., duplicate-heavy logs)
   produce near-uniform gate values, eliminating coherence signal. Mitigation:
   dedup before building the coherence graph, or raise `coherence_threshold`.
3. **Clock skew**: If timestamps are not monotonic (e.g., agent memory ingested
   from an external replay), the decay formula produces incorrect ordering.
   Mitigation: enforce strictly monotonic ingestion timestamps in `MemoryStore`.
4. **Negative cosine scores**: When cosine_sim < 0 and the temporal factor > 0,
   `TemporalSearch` scores stay negative — they are still correctly ranked below
   positive-scoring memories. Acceptance test verifies scores >= -0.01.

## Security Considerations

- Memory content is stored as raw `f32` vectors — no PII in the vector layer.
- `MemoryMetadata.source` is a string field; callers must sanitise before
  inserting from untrusted origins.
- Coherence graph edges reveal which memories are semantically similar to which;
  in multi-tenant deployments the coherence graph must be per-tenant.

## Migration Path

1. Existing code using `ruvector-core` cosine scan can wrap results with
   `FlatSearch` — identical behaviour, no migration required.
2. To enable temporal decay: construct `DecayConfig::exponential(now, half_life)`
   and swap `FlatSearch` → `TemporalSearch`.
3. To enable coherence gating: build `CoherenceGraph::build(&store, threshold)`
   once at session start, then swap to `CoherenceSearch::new(decay, graph, w)`.
4. The production upgrade path replaces the O(n²) graph build with
   `ruvector-acorn` approximate k-NN construction — the `CoherenceGraph` API
   is unchanged.

## Open Questions

1. What is the right default `coherence_weight` (currently 0.30)? Should it
   be calibrated per domain or per agent session?
2. Should `CoherenceGraph` store the full adjacency list or just the degree
   array? Full adjacency enables edge-level mincut pruning but costs O(n·deg) RAM.
3. Is exponential decay the right family? SSGM uses Weibull decay (two-parameter)
   — should `DecayKind` add a `Weibull` variant?
4. Should the coherence gate be computed against the full corpus or only
   against the memories in the current query's temporal window?
