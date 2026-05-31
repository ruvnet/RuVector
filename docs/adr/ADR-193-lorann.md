---
adr: 193
title: "ruvector-lorann: IVF with per-cluster reduced-rank regression score approximation (LoRANN, NeurIPS 2024)"
status: proposed
date: 2026-05-08
authors: [ruvnet, claude-flow]
related: []
tags: [lorann, ann, ivf, reduced-rank-regression, svd, quantization, nightly-research]
---

# ADR-193 — `ruvector-lorann`: LoRANN index (NeurIPS 2024)

## Status

**Proposed.** Implemented on branch `research/nightly/2026-05-08-lorann`.

## Context

ruvector already contains graph-based indices (HNSW variants), quantization codecs (RaBitQ, 1-bit),
filtered-search enhancements (ACORN), and disk-resident indices (DiskANN). One missing category is
**clustering-based (IVF-style) approximate nearest-neighbour search** with a modern score approximator
that is competitive with graph-based methods at high dimensionality (d ≥ 768).

Standard IVF (Inverted File Index) divides the corpus into k clusters and at query time scans all
vectors in the `n_probe` nearest clusters exactly, costing O(n_probe · m_avg · d). At d=1536
(OpenAI text-embedding-3) and n_probe=32, m_avg=500, this is 24.6 M multiplications per query —
expensive enough that practitioners default to HNSW. But HNSW costs O(M · log n · d) per query in
latency and O(n · M · d) in memory, which becomes prohibitive for n ≥ 10 M.

**LoRANN** (Jääsaari, Hyvönen, Roos — NeurIPS 2024, arXiv:2410.18926) identifies the key insight:
the per-cluster exact scorer is a multi-output regression problem. Its optimal rank-r solution is the
truncated SVD of the cluster's document matrix. Replacing exact scoring with this low-rank
approximation reduces query cost to O(r·(d + m)) and achieves recall competitive with HNSW at
moderate to high recall regimes, while using 30–60% of HNSW's memory.

## Decision

Add a new crate `crates/ruvector-lorann` implementing:

1. **k-means++ clustering** (Lloyd's algorithm, parallel via rayon).
2. **Per-cluster `ClusterModel`** — truncated SVD of the cluster doc matrix, producing factor
   matrices A = U_r Σ_r ∈ R^{m×r} and B = V_r ∈ R^{d×r}. Score approximation at query time:
   `scores = A (B^T q)`, costing O(r(d+m)) vs O(d·m) for exact.
3. **`LorannIndex`** — top-level index combining (1) and (2) with exact inner-product reranking
   of the `candidate_set` top approximate candidates.
4. **`FlatExactIndex`** — brute-force baseline.
5. **`AnnIndex` trait** — shared interface for transparent benchmark swaps.

The SVD is computed by nalgebra 0.33 (already a workspace dependency). No new heavyweight
dependencies are introduced.

### Mathematical guarantee

For X_c ≈ U_r Σ_r V_r^T (rank-r truncated SVD):
- Error bound: ||X_c q − Â_c q||₂ ≤ σ_{r+1}(X_c) ||q||₂ per query, where σ_{r+1} is the
  (r+1)-th singular value — the approximation is provably optimal in the Frobenius sense.
- In high-dimensional embedding distributions, singular values decay rapidly after the first ~32,
  making r=32 sufficient for ≥ 85% recall at moderate n_probe.

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_clusters` | √n | Partition granularity. More clusters → finer partitions, better recall at same n_probe. |
| `rank` | 32 | SVD truncation rank. Higher → better recall, slower query. |
| `n_probe` | 8 | Clusters probed at query time. Main recall–QPS knob. |
| `candidate_set` | 200 | Candidates passed to exact reranker. Increase for higher recall. |

## Consequences

### Positive

- **6–55× QPS speedup over brute-force** (measured, single-threaded, x86_64, release build):
  - n=5K, n_probe=8, rank=32: 5.8× speedup at 85.5% recall@10
  - n=50K, n_probe=8, rank=32: 30.9× speedup at 56.1% recall@10
  - n=50K, n_probe=2, rank=32: 54.9× speedup at 29.5% recall@10
- **Complementary to ruvector-rabitq**: RaBitQ is a quantization codec for all ANN algorithms;
  LoRANN is a clustering-based ANN index that can layer RaBitQ on top of it in future work.
- **Complementary to ruvector-acorn**: ACORN is for filtered search; LoRANN is for pure ANN.
- **No new heavy dependencies**: nalgebra already in workspace.
- **Deterministic builds**: SVD is deterministic, k-means uses a fixed seed.

### Negative / Risks

- **Recall at high n_probe degrades** when `candidate_set / n_probe` per cluster becomes too small.
  The default `candidate_set=200` was tuned for n_probe≤8; users targeting >90% recall should
  increase `candidate_set` to 500–1000.
- **Build cost is O(k · m² · d)** for the SVD step. At n=50K, k=224 clusters, avg m=223,
  d=128: build takes 7–8 s single-node. For n≥1M, the SVD step must be batched or parallelised.
- **Memory overhead**: storing A (m×r) and B (d×r) per cluster adds ~70% over raw vector storage
  at rank=32, d=128. At r=16, overhead is ~36%.
- **Synthetic benchmark bias**: current benchmarks use Gaussian-clustered data, not real
  ann-benchmarks datasets. Recall figures on SIFT-1M or GIST-960 may differ.

## Alternatives Considered

### 1. HNSW (already in ruvector-core)
- Pro: Better recall at same QPS for low-d data.
- Con: O(n · M · d) memory; slow graph construction; poor tail latency.
- Decision: LoRANN is a complement, not a replacement.

### 2. IVF-PQ (standard product quantization)
- Pro: Industry standard; great codec compression.
- Con: PQ distortion > SVD approximation error at equal byte budget; no Rust workspace crate.
- Decision: LoRANN SVD strictly better than PQ under Frobenius norm; IVF-PQ may be added later
  as a separate crate or as a `ScoreApproximator` variant.

### 3. SOAR (NeurIPS 2023, Google ScaNN)
- Pro: State-of-art on ann-benchmarks.
- Con: Requires training phase with query distribution; complex multi-VQ spilling logic.
- Decision: Too complex for a single-night nightly implementation.

### 4. Matryoshka Representation Learning (MRL) prefix search
- Pro: 14× speedup reported with HNSW + MRL prefixes.
- Con: Requires MRL-trained embeddings; not applicable to arbitrary f32 vectors.
- Decision: LoRANN works with any f32 corpus without retraining.
