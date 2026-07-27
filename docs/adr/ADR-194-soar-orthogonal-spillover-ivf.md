---
adr: 194
title: "SOAR — Spilling Orthogonal Anti-correlated Refinement for IVF assignment"
status: proposed
date: 2026-05-08
authors: [claude-nightly]
related: [ADR-193]
tags: [ivf, ann, vector-search, soar, scann, anisotropic-quantization, nightly-research]
---

# ADR-194 — SOAR: Spilling Orthogonal Anti-correlated Refinement for IVF

## Status

**Proposed.** Implemented as PoC on branch
`research/nightly/2026-05-08-soar-orthogonal-spillover-ivf` in crate
`crates/ruvector-soar`. `cargo build -p ruvector-soar --release` and
`cargo test -p ruvector-soar` pass on Apple M4 Max (rustc 1.89.0).

## Context

ruvector ships an IVF-style ANN path via several crates (`ruvector-cluster`,
the IVF helpers in `ruvector-core`). Today, posting-list assignment is
single-nearest-centroid. Boundary recall — vectors near a Voronoi face
between two cells — is the dominant recall-loss source for IVF on real
embeddings.

The classical fix is **2× spillover**: write each vector to its top-2
nearest centroids. This costs 2× posting storage, and in practice on
real distributions the second copy is *highly correlated* with the first —
both quantization error vectors point in nearly the same direction. The
second posting adds little new query-side coverage.

Sun et al. (NeurIPS 2024, "SOAR: Improved Indexing for Approximate
Nearest Neighbor Search") propose replacing "second-nearest" with an
**anti-correlated** secondary chosen to minimize:

```
loss(c) = ||x - c||^2 + lambda * ((x - c) . r_hat)^2
```

where `r_hat = (x - c1)/||x - c1||` is the unit residual after the
primary assignment. The penalty term suppresses centroids whose error
vector is parallel to the primary residual, forcing the two assignments
to cover *complementary* error directions. The technique is shipping in
production in Google's ScaNN.

## Decision

Add a new workspace member `crates/ruvector-soar` exposing:

- `enum Assignment { Single, Spillover, Soar { lambda: f32 } }` —
  pluggable strategies, identical query path.
- `struct IvfIndex` with `build(vectors, n_centroids, assignment, seed)`
  and `search(query, k, n_probe)`.
- A pure-Rust deterministic k-means (k-means++ init + 12 Lloyd iters),
  no `unsafe`, no external math deps beyond `rand`.
- A `mean_residual_correlation()` KPI to validate the orthogonalization
  objective independently of recall.
- Demo binary `soar-demo` printing real recall@10, build time, query
  latency, and residual correlation across all three strategies on three
  synthetic anisotropic-cluster benchmarks.
- Criterion bench `soar_bench` for build + query latency.
- Four integration tests asserting (a) replication factors, (b) sorted
  unique top-k, (c) SOAR ≥ Single recall at equal probe budget,
  (d) SOAR residual correlation ≤ Spillover.

The PoC keeps storage as raw `Vec<f32>` per posting (no quantization)
to isolate the assignment-strategy variable. Composition with
ruvector-rabitq / ruvector-lvq is left to a follow-on ADR.

## Consequences

**Positive**

- Mean residual correlation drops monotonically with `lambda` —
  measured **+0.231 → +0.143 (-38%)** at N=10k, dim=32, k=128. Confirms
  faithful implementation of the SOAR objective.
- Query latency is consistently lower than plain Spillover at the same
  posting cost — measured **52.1 µs → 42.9 µs (-18%)** at N=20k, dim=64,
  k=256, n_probe=4. Cause: SOAR's secondaries land in genuinely different
  cells, reducing post-dedup candidate set size.
- Clean trait-shaped enum lets us slot SOAR into existing IVF paths
  without breaking other backends.
- No new external dependencies. Pure-Rust, deterministic, no `unsafe`.

**Neutral / known limits**

- On synthetic isotropic+anisotropic Gaussians with 200 uniform queries,
  SOAR matches Spillover's recall to within ±0.005, not the +3–8 pp
  improvement reported in the paper. The paper's gains appear on real
  high-dim embedding distributions (deep1B, glove, Cohere). Real-dataset
  validation is queued as a follow-up (see "What to improve next" in the
  research doc).
- Build time is **~30–45% slower** than Spillover (extra centroid scan
  per vector). For N ≥ 1M the constant matters; mitigations include the
  rotation trick from §4 of the paper or batched GPU scoring.

**Negative**

- 2× posting cost vs. plain `Single` IVF. Same as plain spillover —
  not a new cost, but worth stating.
- Adds one workspace crate (~600 LoC across src + tests + bench).

## Alternatives considered

1. **Do nothing (Single only)** — leaves boundary recall on the table.
   Rejected; ANN literature has converged on multi-assignment as
   essentially free at high-recall operating points.
2. **Plain 2× spillover** — simpler, but our measurements show SOAR
   delivers the same recall at lower query latency, and the orthogonality
   KPI is empirically better. Spillover stays in-tree as `Assignment::Spillover`
   for ablation and as the natural fallback.
3. **Anisotropic quantization (ScaNN-style loss)** — addresses a different
   axis of the problem (what gets stored in a posting, not which postings
   a vector lives in). Complementary to SOAR, not a substitute. Out of
   scope for this ADR.
4. **3+ assignments** — extension of SOAR with multiple `r_hat` penalty
   terms. Diminishing returns past 2 per the paper; left as future work.

## References

- Sun, Simhadri, Guo, Kumar. *SOAR: Improved Indexing for Approximate
  Nearest Neighbor Search.* NeurIPS 2024. arXiv:2404.00774.
- Guo et al. *Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization (ScaNN).* ICML 2020.
- Research doc: `docs/research/nightly/2026-05-08-soar-orthogonal-spillover-ivf/README.md`.
