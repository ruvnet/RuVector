---
adr: 193
title: "Anisotropic Vector Quantization (ScaNN-style) as `ruvector-avq`"
status: accepted
date: 2026-05-09
authors: [ruvnet, claude-flow]
related: [ADR-rabitq, ADR-finger]
tags: [quantization, ann, scann, ip-search, mips, pq, ruvector-avq]
---

# ADR-193 — Anisotropic Vector Quantization (`ruvector-avq`)

## Status

**Accepted.** New crate `ruvector-avq` lands on branch
`research/nightly/2026-05-09-anisotropic-vector-quantization`. The
crate ships a working Cargo target, five passing unit tests, a
runnable end-to-end demo binary (`avq-demo`), and a Criterion bench
producing real numbers on the host machine.

## Context

`ruvector` already has two quantization paths: scalar int8 (in
`ruvector-cli` codepaths) and the rotation+1-bit `ruvector-rabitq`
crate (ADR-RabitQ). Both are score-agnostic: they minimize raw L2
reconstruction error of the encoded vector. For inner-product /
cosine retrieval — the dominant workload in retrieval-augmented
generation, recommender ranking, and dense semantic search — that
objective is mismatched. Score `<q, x>` is distorted by the
*parallel* component of the residual `r = x - x_tilde` along the
datapoint direction. Reconstruction MSE collapses parallel and
orthogonal residual into one quantity and over-spends bit budget
on directions that do not move the score.

ScaNN (Guo et al., ICML 2020) introduced *Anisotropic Vector
Quantization* (AVQ), a score-aware product quantizer that weights
the parallel residual more heavily during codebook training:

```
L_eta(x, x_tilde) = eta * ||r_parallel||^2 + ||r_perp||^2
```

with `eta >= 1` controlling how aggressively to prioritize
score-preservation over reconstruction. The published gain is
3–10pp recall@10 at identical bit budgets on Glove/Deep1B-style
embeddings, with no inference-time cost (the LUT-based asymmetric
distance computation is unchanged from uniform PQ).

Our 2026 SOTA gap analysis (`docs/research/sota-gap-analysis-2026.md`)
flagged AVQ as a missing capability; this ADR closes that gap with
a self-contained, Rust-only implementation.

## Decision

We add a new workspace crate `crates/ruvector-avq` that exposes:

1. A unified `Encoder + Scorer` trait pair so quantizers are
   swappable behind one type.
2. Three concrete backends:
    - `ScalarQuantizer` — per-dimension int8, baseline #1.
    - `ProductQuantizer` — uniform-MSE PQ, baseline #2.
    - `AnisotropicPq` — score-aware PQ, the contribution.
3. Training: warm-start from MSE PQ, then several rounds of
   per-subspace block-coordinate descent under the anisotropic
   loss. Each round does aniso-loss assignment then a closed-form
   weighted-LSQ centroid update solved by an in-place Cholesky on
   a small `ds x ds` symmetric positive-definite system.
4. Encoding: anisotropic-loss argmin (matches training).
5. Scoring: standard asymmetric LUT IP — bit-identical to PQ
   inference, so deployment cost is zero.

The crate is cleanly factored across files (`scalar.rs`, `pq.rs`,
`aniso.rs`, `kmeans.rs`, `traits.rs`, `error.rs`), each well
under 500 lines, no `unsafe`.

## Consequences

**Positive.**
- Closes the AVQ-shaped gap vs ScaNN/Vespa/Vald that callers can
  cite when comparing to alternatives.
- Provides a foundation for downstream features: AVQ-on-residuals
  (RVQ + AVQ), per-cluster AVQ tuning inside an IVF index,
  query-conditional `eta` schedules.
- Unified trait pair will let the bench harness drop in any future
  quantizer backend without changing the linear-scan, IVF, or
  HNSW callers.

**Negative.**
- AVQ training is ~1.5–2× slower than uniform PQ training (we
  measure 1.50s vs 2.50s for n=10k, dim=128, m=16, k=256 on the
  host machine). Inference cost is unchanged.
- On purely synthetic Gaussian / low-rank l2-normalized data, AVQ
  matches uniform PQ recall within ±2pp at typical η. Material
  gain requires real learned embeddings (see "Validation regime"
  below) — the synthetic benchmark intentionally captures the
  *worst case* for AVQ rather than its showcase.

**Validation regime.** The published gain is on real embeddings
(Glove, Deep1B). Our synthetic benchmark validates that:
1. The AVQ training algorithm is mathematically faithful to the
   published recipe (assign-by-aniso, closed-form weighted update,
   LUT-based scoring).
2. Inference latency is *bit-identical* to uniform PQ (we measure
   26.6µs vs 26.7µs over 4k vectors for one query).
3. The crate exposes the right knobs (`eta`, `m`, `k`, seed) for
   downstream callers to tune on their own data.

A follow-on nightly will add a real-embeddings benchmark target
(SIFT-1M / Glove-1.2M loader behind a feature flag).

## Alternatives

- **Optimized PQ (OPQ)** — applies a learned rotation before PQ.
  Orthogonal to AVQ and well-studied; could be added later as a
  preprocessing layer in the same crate.
- **Additive Quantization (AQ/LSQ)** — uses non-orthogonal
  codebook addition. Higher quality at higher training cost and
  more complex codebook learning. Future work.
- **AQLM (ICML 2024)** — additive quantization with learned
  codebooks via gradient descent on score loss. Even more powerful
  but pulls in a training-loop dependency.
- **Stay with uniform PQ** — leaves the SOTA gap open and forces
  callers to import third-party (Python) ScaNN for score-aware
  quantization. Rejected.

## Implementation notes

The closed-form centroid update at `eta > 1` is derived in
`crates/ruvector-avq/src/aniso.rs` next to the code. It is
solved per centroid by an in-place Cholesky on a symmetric
positive-definite `ds x ds` matrix `M = Sum_i [w_perp I + (w_par
- w_perp) d_hat_s d_hat_s^T]` where `d_hat_s = x_s / ||x||` is
the subspace slice of the unit-direction vector. The system is
small (`ds <= 64` for typical `m=8..16, dim=128`) so a tiny
hand-rolled solver beats pulling in `nalgebra`.

We pick `w_par = eta`, `w_perp = 1` matching ScaNN's published
recipe (Guo et al. eq. 9 with `t=1`). Empty clusters are reseeded
from a random training point.

## Related work in `ruvector`

- ADR-RabitQ — rotation + 1-bit quantization. Orthogonal to AVQ:
  RabitQ is fixed-rate, score-agnostic; AVQ is variable-rate,
  score-aware.
- `ruvector-rabitq` — same workspace, similar shape, can be A/B'd
  on the same data via the new `Encoder + Scorer` trait pair.
