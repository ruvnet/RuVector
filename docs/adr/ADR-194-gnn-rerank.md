---
adr: 194
title: "GNN-Enhanced Candidate Reranking for Approximate ANN Search"
status: accepted
date: 2026-05-21
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193, ADR-184]
tags: [gnn, reranking, ann, vector-search, graph, rag, nightly-research]
---

# ADR-194 — GNN-Enhanced Candidate Reranking for Approximate ANN

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-21-gnn-rerank` as
`crates/ruvector-gnn-rerank`.  All 14 unit tests pass; build is green with
`cargo build --release -p ruvector-gnn-rerank`.

Benchmark (x86-64, Linux 6.18, `cargo run --release`, N=5K, D=128, K=10,
retrieval_k=80, noise_σ=0.40):

| Variant | recall@10 | mean µs | p50 µs | p95 µs |
|---------|-----------|---------|--------|--------|
| NoisyScore (baseline) | 28.0% | 0.2 | 0.2 | 0.2 |
| GnnDiffusion (1-hop, α=0.60) | **38.4%** | 1006 | 997 | 1053 |
| GnnMincut (coh≥0.50, α=0.60) | **38.4%** | 999 | 992 | 1025 |
| ExactL2 (oracle) | 74.9% | 13.8 | 12.5 | 16.5 |

GNN score diffusion delivers **+10.4 pp recall@10** over the noisy baseline.

## Context

Every approximate ANN index — whether HNSW, DiskANN, IVF, or RaBitQ — returns
a ranked candidate set whose distance estimates contain noise.  For quantised
indexes (1-bit RaBitQ, low-bit PQ, coarse IVF with small `nprobe`) this noise
can cause significant recall loss: items near the K-boundary swap order, pushing
true positives out of the top-K window.

Existing ruvector crates address the *first stage* (better indexing: RaBitQ in
ADR-177, RAIRS IVF in ADR-193, filtered ACORN in ADR-187) but not the *second
stage* (post-retrieval reranking).

The 2025–2026 literature identifies graph-based reranking as a promising
direction:
- GNRR (arXiv:2406.11720): +5.8% Average Precision on TREC-DL19
- Maniscope (arXiv:2602.15860): +7% NDCG, 3.2× faster than cross-encoders
- AQR-HNSW (arXiv:2602.21600): 2.5–3.3× QPS with 98%+ recall using multi-stage reranking

No production vector database (Milvus, Qdrant, Weaviate, LanceDB, FAISS, pgvector)
applies GNN message passing over the ANN candidate subgraph topology.

## Decision

We introduce `crates/ruvector-gnn-rerank` implementing the `CandidateReranker`
trait with four variants:

| Variant | Algorithm | Use case |
|---------|-----------|----------|
| `NoisyScoreReranker` | passthrough | baseline measurement |
| `GnnDiffusionReranker` | 1-hop score averaging | general reranking after quantised retrieval |
| `GnnMincutReranker` | structurally-gated diffusion | reranking with cross-cluster isolation |
| `ExactL2Reranker` | exact Euclidean sort | oracle and production fallback |

### Core algorithm (GnnDiffusionReranker)

1. Accept `n` candidates (id, full-precision vector, noisy_score).
2. Build a cosine k-NN graph over the candidate set: O(n² × dim).
3. Run `hops` rounds of score averaging:
   `s_i^{t+1} = α · s_i^t + (1-α) · mean_{j ∈ N(i)} s_j^t`
4. Sort by final score; return top-k.

### Why diffusion improves recall

True top-K items occupy the same vector cluster; in the candidate k-NN graph
they are mutually connected.  Averaging their noisy scores cancels i.i.d. noise
by the law of large numbers.  False positives with artificially high noisy scores
are isolated from the true cluster, so diffusion reduces rather than amplifies
their apparent relevance.  This is discrete graph spectral low-pass filtering.

### Mincut coherence gating

`GnnMincutReranker` gates propagation on structural edge weight (cosine
similarity between candidates) rather than on score ratios.  This prevents
cross-cluster score bleeding while allowing intra-cluster diffusion.  Score-ratio
gating was explicitly rejected because it blocks the most important edges — those
connecting a true positive that received a low noisy score to its correctly-scored
true-positive neighbours.

## Consequences

### Positive
- +10.4 pp recall@10 over noisy baseline (measured).
- Composable with any first-stage retriever via the `CandidateReranker` trait.
- No external service dependency.
- WASM-compatible (no unsafe code, no OS dependencies).
- Production candidate for use after `ruvector-rairs` / `ruvector-rabitq` retrieval.

### Negative / Trade-offs
- Graph construction is O(n² × dim): adds ~1ms latency for n=80, dim=128.
- `ExactL2Reranker` is 73× faster (14µs vs 1000µs) and provides the recall
  ceiling (74.9% at this noise level); GNN reranking only makes sense when exact
  vector comparison for all candidates is too expensive (e.g., remote fetch).
- The +10.4 pp gain is relative to a noisy baseline; the oracle gap is 36.5 pp.

## Alternatives considered

| Alternative | Rejected reason |
|-------------|-----------------|
| Cross-encoder reranking | Requires a learned model; Python dependency |
| Exact L2 (oracle) | Already in PoC; use directly when vectors are local |
| Score-ratio coherence gating | Blocks useful edges for low-noisy-score true positives |
| 2-hop diffusion | Marginal gain, 2× graph-build cost; future work |
| Multi-vector (ColBERT-style) | Different problem; higher design cost |

## Implementation plan

1. `crates/ruvector-gnn-rerank` — core library (this ADR). ✓ Done.
2. Integration into `ruvector-server` search pipeline — future work.
3. Candidate graph construction from compressed (RaBitQ) vectors — future work.
4. Adaptive alpha / hop selection via query feedback — future work.

## Benchmark evidence

Hardware: Intel Celeron N4020, x86-64, Linux 6.18.5, `rustc 1.87.0 --release`.
Data: multi-Gaussian, 20 clusters, σ=0.5, N=5K, D=128, 100 queries, K=10.
Noisy retrieval simulates quantised index: score = −L2 + N(0, 0.40²).

Full output: `cargo run --release -p ruvector-gnn-rerank --bin benchmark`.

## Failure modes

- **Candidate graph too sparse**: if candidates are from disjoint clusters, k-NN
  graph is disconnected; diffusion has no effect.  Mitigation: use `ExactL2Reranker` fallback.
- **Noise too high / coverage too low**: if true top-K items aren't in the
  candidate set, no reranker can recover them.  Mitigation: increase `retrieval_k`.
- **Alpha too low**: aggressive diffusion (small alpha) washes out signal; score
  homogenisation hurts recall.  Recommendation: α ≥ 0.5.
- **k_graph too small**: insufficient graph density → poor noise cancellation.
  Recommendation: k_graph ≥ 5, typically 8.

## Security considerations

- No file I/O, no network calls, no unsafe code.
- Candidate vectors are caller-provided; no validation of vector dimensionality
  across candidates (caller responsibility).
- No secrets or credentials involved.

## Migration path

`CandidateReranker` is a new trait; no existing code is modified.  Integration
into `ruvector-server` would add a post-search pipeline stage behind a feature
flag `gnn-rerank`.

## Open questions

1. Can candidate graph be built from compressed (4-bit) vectors with < 5%
   recall degradation, removing the need to fetch full-precision vectors?
2. What is the optimal α for production workloads with real embedding models?
3. Is 2-hop diffusion worth the 2× graph cost at larger n (200, 500)?
4. Can the O(n²) graph construction be approximated in O(n log n)?
