---
adr: 193
title: "Distance Adaptive Beam Search (DABS) for provably-accurate graph-based ANN"
status: accepted
date: 2026-05-08
authors: [ruvnet, claude-flow]
related: [ADR-001, ADR-041]
tags: [vector-search, hnsw, ann, beam-search, dabs, nightly-research, neurips-2025]
---

# ADR-193 — Distance Adaptive Beam Search (DABS) for Graph-Based ANN

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-08-dabs-hnsw` as
standalone crate `crates/ruvector-dabs`. Pending integration into `ruvector-core`
HNSW search path as a follow-on PR.

## Context

ruvector's graph-based ANN search (HNSW in `ruvector-core`, greedy graph in
`ruvector-acorn`) uses the standard fixed-ef termination criterion: beam search
continues until `ef` candidates have been evaluated. This approach has two
well-known weaknesses:

1. **No provable recall guarantee**: there is no formal relationship between the
   ef parameter and the approximation error of returned results. Users must tune
   ef empirically per dataset.

2. **Wasted exploration**: when k good results are found early (dense query
   region), fixed-ef continues evaluating candidates that cannot improve results.
   Al-Jazzazi et al. (NeurIPS 2025, arXiv:2505.15636) measured 10–50% wasted
   distance computations on SIFT1M, DEEP96, GloVe, GIST, and MNIST.

The paper proposes Distance Adaptive Beam Search (DABS): replace the fixed-ef
loop condition with a distance-ratio test that carries a formal approximation
guarantee.

Our benchmark (N=10K, D=128, M=16) confirms the adaptive advantage:
- Fixed-ef peaks at 84.85% recall (ef=256, 1,814 ops/query)
- DABS γ=0.20 reaches 90.25% recall (2,433 ops/query) — **+5.4 pp above the
  fixed-ef ceiling** at only 34% more ops

No other Rust ANN crate (hnsw\_rs, hnswlib-rs, swarc) has implemented DABS as
of 2026-05-08.

## Decision

We introduce `crates/ruvector-dabs` as a standalone Rust crate implementing:

1. **`DabsGraph`**: flat row-major vector store with greedy k-NN adjacency list
   (rayon-parallelised build, O(n²) PoC).

2. **`search_dabs(graph, query, k, gamma)`**: DABS Algorithm 1 (arXiv:2505.15636).
   Termination condition: `d(q, x_closest) > (1+γ) × d_k` where d_k is the
   current k-th nearest discovered distance. Results heap bounded to exactly k
   entries. Neighbor enqueueing gated by the same γ-window.

3. **`search_fixed_ef(graph, query, k, ef)`**: standard ef-bounded beam search
   for comparison and compatibility.

4. **`SearchMode` enum**: `Flat | FixedEf { ef } | Dabs { gamma }`. Adding
   future search strategies requires only dispatching on a new variant.

5. **Formal guarantee**: on any navigable graph, DABS with parameter γ returns
   results satisfying `d(q, result_i) ≤ (1+γ)² × d(q, true_i)` for each rank i
   (Theorem 1, Al-Jazzazi et al.).

The DABS search loop replaces the single condition `curr_d > worst_in_results`
with `curr_d > (1.0 + gamma) * kth_d`. This is the complete algorithmic change.

## Consequences

### Positive

- **Provable recall bounds**: users can select γ based on their SLA
  (γ=0.1 → 1.21× approximation; γ=0.2 → 1.44× approximation).
- **Higher recall without graph rebuild**: DABS γ=0.20 exceeds fixed-ef recall
  ceiling by +5.4 pp on the benchmark dataset.
- **Adaptive efficiency**: on clustered data (common for embedding spaces),
  DABS terminates earlier than fixed-ef when results converge quickly.
- **Backward compatible**: γ=0 recovers greedy descent; FixedEf mode is retained
  for users who need deterministic ef-bounded behavior.

### Negative

- **Worse QPS at high recall**: DABS γ=0.50 (recall=0.9835) runs at 490 QPS
  vs fixed\_ef=256 (recall=0.8485) at 2,222 QPS. To achieve 98%+ recall, more
  exploration is needed regardless of termination criterion.
- **Graph quality matters more**: DABS relies on the γ-window neighbor enqueue
  to find good nodes. A poorly-constructed graph (low M, no back-edges) will
  limit DABS recall regardless of γ.
- **γ tuning required**: the optimal γ value is dataset-dependent. We recommend
  providing a `calibrate_gamma(sample_queries, target_recall)` utility in the
  follow-on integration PR.

### Neutral

- Crate stands alone with no dependency on `ruvector-core`. Integration into
  core HNSW search path is a follow-on task (see roadmap in research doc).
- Build time (O(n²) greedy graph) is acceptable at PoC scale; production
  integration will reuse the existing HNSW multi-layer construction.

## Alternatives Considered

### 1. LoRANN (NeurIPS 2024) IVF with reduced-rank regression

Strong recall/memory tradeoffs but requires a fundamentally different index
architecture (IVF clusters vs. navigable graph). Does not improve existing HNSW
search paths. Deferred to a future IVF-focused nightly.

### 2. Probabilistic Edge Order Sampling (PEOs, ICML 2024)

Reduces per-evaluation cost via inner-product hashing (skip expensive evals for
unpromising edges). Complementary to DABS, not exclusive. Could be layered on
top of DABS in a follow-on PR to further reduce ops per evaluation.

### 3. Matryoshka Adaptive Precision Search

Uses truncated embeddings for coarse candidate selection, full embeddings for
reranking. Requires Matryoshka-trained embeddings (not always available). DABS
works on any navigable graph without embedding assumptions.

### 4. Increasing ef ceiling

Simply raising ef from 256 to 512 or 1024 in fixed-ef mode would increase
recall but provides no provable bound and scales linearly with ef. DABS achieves
higher recall at γ=0.20 with 2,433 ops vs an estimated ~3,000+ ops for ef=512
on this dataset.
