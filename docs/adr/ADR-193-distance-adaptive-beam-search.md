---
adr: 193
title: "Distance-Adaptive Beam Search for Provably Accurate Graph-Based ANN"
status: accepted
date: 2026-05-10
authors: [ruvnet, claude-flow]
related: [ADR-160, ADR-170, ADR-185]
tags: [ann, beam-search, adaptive, provable-guarantee, graph-search, diskann, hnsw, stopping-criterion]
---

# ADR-193 — Distance-Adaptive Beam Search

## Status

**Accepted.** Implemented as new standalone crate `ruvector-adaptive-beam` on branch
`research/nightly/2026-05-10-distance-adaptive-beam-search`.
Full integration into `ruvector-core` (DiskANN and HNSW search paths) is tracked in the roadmap below.

## Context

Every graph-based ANN search in ruvector uses a fixed count-based stopping rule:
the inner beam search loop expands at most `search_list_size` (DiskANN, `VamanaConfig`) or
`ef` (HNSW) candidates before terminating. This is the universal pattern across the entire
vector database industry (FAISS, Qdrant, Milvus, Weaviate, usearch, LanceDB).

Two problems with this approach were identified:

**Problem 1 — No approximation guarantee.**  
`FixedWidth(bw=64)` achieves 73.6% Recall@10 on our benchmark dataset; `bw=4096` achieves
99.0%. There is no formula relating `bw` to recall: users must grid-search per dataset.
If the data distribution changes (embedding model upgrade, new data domain), recall silently
degrades unless `bw` is re-tuned.

**Problem 2 — Wasted distance evaluations on converged frontiers.**  
When the search has already found the true top-k neighbours, FixedWidth continues expanding
stale candidates until the count is exhausted. These evaluations contribute nothing to recall
but consume 30-50% of search time (measured on HNSW graphs in arXiv:2505.15636).

In May 2025, Mussmann et al. (arXiv:2505.15636) published the first graph-based ANN stopping
criterion with a provable approximation guarantee:

> **Theorem 1 (Distance-Adaptive Stopping)**: On a δ-navigable graph, if beam search
> terminates when the closest unvisited candidate c satisfies
> `d(q, c) > (1 + γ) · d(q, p_k)`, the returned set is a `(1 + γ/2)`-approximation
> to the true k nearest neighbours.

No open-source Rust implementation existed as of May 2026. All major vector databases
(Qdrant, Milvus, Weaviate, LanceDB, pgvector, usearch) continue to use FixedWidth.

## Decision

We introduce a `BeamStopPolicy` enum as the canonical stopping abstraction for all
graph-based search in ruvector, and implement it in a new standalone PoC crate
(`crates/ruvector-adaptive-beam`) with full tests and benchmarks.

### Policy enum

```rust
pub enum BeamStopPolicy {
    /// Current behaviour: expand at most `beam_width` nodes (no guarantee).
    FixedWidth { beam_width: usize },

    /// arXiv:2505.15636: stop when d(q,c) > (1+gamma)*d(q,k-th result).
    /// Gives provable (1+gamma/2)-approximation on any navigable graph.
    DistanceAdaptive { gamma: f32 },

    /// Hybrid: same as DistanceAdaptive but never stop before min_expansions.
    /// Protects against sparse entry regions.
    AdaptiveWithFloor { gamma: f32, min_expansions: usize },
}
```

### Recommended defaults

| Use case | Policy | Rationale |
|----------|--------|-----------|
| High-recall production (≥99%) | `DA(γ=1.0)` | Provable 1.5× bound; self-tuning |
| Balanced production (≥97%) | `DA(γ=0.5)` | Provable 1.25× bound; 6% fewer dist/q vs FW |
| Low-latency / approximate | `DA(γ=0.1)` | Provable 1.05× bound; matched QPS to FW(64) |
| Backwards compatibility | `FixedWidth { beam_width: search_list_size }` | Identical to pre-ADR-193 |

### Benchmark results (PoC, k-NN graph, N=5 000, D=128)

```
Policy                   QPS   Recall@10   Dist/q  Guarantee
FixedWidth(bw=64)       6313      73.6%     595    none
FixedWidth(bw=256)      2376      91.0%    1403    none
FixedWidth(bw=1024)      975      97.4%    2612    none
FixedWidth(bw=4096)      413      99.0%    3859    none
DA(γ=2.0)                413      99.0%    3859    ≤2.0× optimal
DA(γ=1.0)                414      99.0%    3859    ≤1.5× optimal
DA(γ=0.5)                482      98.8%    3635    ≤1.25× optimal  ← recommended
DA(γ=0.1)               5999      75.4%     622    ≤1.05× optimal
AdaptiveFloor(γ=0.5,16)  490      98.8%    3635    ≤1.25× optimal
```

Hardware: x86_64 Linux, 4 CPUs, rustc 1.94.1 `--release`.

Note: on flat k-NN graphs (no hierarchical layers), DA explores similarly to FixedWidth(n)
at high-recall targets. The 30-50% distance computation savings reported in arXiv:2505.15636
apply to HNSW/Vamana graphs with hierarchical entry points and are expected on integration
into `ruvector-core`'s existing HNSW and DiskANN search paths.

### Integration path

**Phase 1 (this ADR)**: Standalone PoC crate with correct algorithm, tests, benchmarks.

**Phase 2** (follow-on): Extend `VamanaConfig` in `ruvector-core/diskann.rs`:
```rust
pub struct VamanaConfig {
    pub beam_stop: BeamStopPolicy,  // replaces/wraps search_list_size
    ...
}
```
Default: `BeamStopPolicy::FixedWidth { beam_width: self.search_list_size }` — zero breaking change.

**Phase 3** (follow-on): Same for HNSW ef parameter in `ruvector-core`.

## Consequences

### Positive

- **Provable quality**: users can specify a quality level (γ) and receive a mathematical guarantee, eliminating per-dataset hyperparameter tuning for recall targets.
- **Self-adaptive**: DA naturally stops earlier on well-connected graphs (dense neighbourhoods), spending compute only where needed.
- **Zero breaking change**: existing code using `search_list_size` defaults to `FixedWidth { beam_width: search_list_size }`, identical behaviour.
- **Future-proof**: works with any graph structure (k-NN, NSW, HNSW, Vamana, NSG) without modification.
- **Production readiness**: AdaptiveWithFloor handles degenerate entry points that trip pure DA.

### Negative / Risks

- **Flat graph limitation**: on flat k-NN graphs without hierarchical navigation, DA requires more distance evaluations than FixedWidth at low beam widths. Full benefit requires HNSW/Vamana integration (Phase 2-3).
- **Approximation, not exact**: users expecting true nearest neighbours (e.g., distance-sensitive similarity thresholds) must use γ=0 or exact search.
- **New parameter surface**: γ is more principled than `bw` but is still a parameter. Users unfamiliar with approximation ratios may choose poorly.
- **Proof requires navigability**: the guarantee applies to δ-navigable graphs. Degenerate graph builds (M too small, disconnected components) can violate navigability.

## Alternatives Considered

### A — Keep FixedWidth, tune per dataset

**Rejected**: provides no approximation guarantee; requires expensive recall-vs-latency sweeps per data distribution update. Every embedding model upgrade requires re-tuning.

### B — Implement exhaustive search with early exit on exact k-NN convergence

**Rejected**: exact convergence detection requires brute-force verification of all nodes, negating the purpose of graph-based ANN. O(n·D) per query.

### C — Confidence-based stopping (estimate recall from graph properties)

**Considered**: heuristic methods estimate recall from degree distribution or graph density. Rejected because these produce no provable bound; they are essentially calibrated guesses, not theorems.

### D — NSG (Navigating Spreading-out Graph) with adaptive ef

**Partially adopted**: NSG's construction (RNG pruning, angle-diverse edges) combined with DA stopping is synergistic and is captured in the roadmap. NSG construction is a separate concern from the stopping criterion.

### E — Per-query FixedWidth calibration (predict recall from query features)

**Considered**: ML-guided beam width selection per query. Rejected for now: adds inference latency and training complexity. DA(γ) achieves similar goals with a single parameter and a mathematical guarantee.
