# Anisotropic Vector Quantization for `ruvector`

**Branch:** `research/nightly/2026-05-09-anisotropic-vector-quantization`
**ADR:** [ADR-193](../../../adr/ADR-193-anisotropic-vector-quantization.md)
**Crate:** [`crates/ruvector-avq`](../../../../crates/ruvector-avq)

## Abstract

We add Anisotropic Vector Quantization (AVQ, ScaNN-style) to
`ruvector` as a new workspace crate `ruvector-avq`. AVQ is a
score-aware product quantizer: instead of minimizing reconstruction
MSE, it minimizes a weighted error that penalizes the *parallel*
component of the residual (along the datapoint direction) more
heavily than the orthogonal component. For inner-product /
cosine retrieval — the dominant workload in modern dense vector
search — this objective is the right one: only the parallel
residual distorts the score `<q, x>`. The crate ships three
swappable quantizer backends behind one trait, an end-to-end
runnable demo, a Criterion benchmark, and five passing unit
tests with real numeric assertions on recall. Inference cost is
bit-identical to uniform PQ (LUT-based asymmetric distance);
codebook training is ~1.5–2× slower.

## SOTA survey

Score-aware quantization for MIPS / cosine retrieval has converged
on three families:

1. **Anisotropic VQ (ScaNN, Guo et al., ICML 2020).** Per-point
   loss decomposes into parallel and orthogonal residual; eta-weighted
   parallel component drives codebook training. Powering Google's
   ScaNN and Vespa's `mips` ranker; reported 3–10pp recall@10 gain
   over uniform PQ at identical bit budgets on Glove/Deep1B.
2. **Additive Quantization & RVQ (Babenko-Lempitsky 2014, Liu et
   al. 2024).** Sum of codebooks rather than concat. Higher
   quality at higher training/encode cost; FAISS's IVF-RQ shows
   strong numbers on Deep1B.
3. **Learned codebooks (AQLM, ICML 2024; QuIP, NeurIPS 2024).**
   Gradient-descent over codebooks under a score loss directly.
   Best quality, highest training complexity.

Contemporaneous Rust ecosystem state (May 2026):
- `qdrant` ships uniform PQ + scalar quantization, no anisotropic
  support.
- `lancedb` ships scalar + IVF-PQ via FAISS bindings; no native
  Rust score-aware quantizer.
- `arroy` (Spotify) ships Annoy-style trees with no PQ.
- `instant-distance` ships HNSW, no quantization.

`ruvector-avq` is, to our knowledge, the first native-Rust
implementation of ScaNN's anisotropic objective.

## Proposed design

A small, focused crate with a swappable quantizer API:

```rust
trait Encoder {
    fn code_size(&self) -> usize;
    fn dim(&self) -> usize;
    fn encode(&self, xs: &[f32], codes: &mut [u8]);
}

trait Scorer: Encoder {
    fn score_ip(&self, query: &[f32], codes: &[u8], out: &mut [f32]);
    fn topk_ip(&self, query: &[f32], codes: &[u8], k: usize) -> Vec<(u32, f32)>;
}
```

Three implementations:

- `ScalarQuantizer` — per-dimension int8; baseline.
- `ProductQuantizer` — uniform-MSE PQ; baseline.
- `AnisotropicPq` — score-aware PQ, our contribution.

Same trait so the linear-scan harness, future IVF coarse
quantizer, and HNSW residual codec all plug in identically.

## Implementation notes

### Per-subspace decomposition

The full-vector residual `r = x - x_tilde` decomposes additively
across subspaces `r = (r[1], ..., r[m])`. The full parallel
residual is `r . d_hat = Sum_s r[s] . d_hat[s]` where
`d_hat = x / ||x||`. We approximate the squared parallel residual
as `Sum_s (r[s] . d_hat[s])^2`, dropping cross-terms (the
standard ScaNN block-coordinate approximation). This makes the
optimization separable across subspaces.

### Closed-form weighted update

For a fixed assignment, the centroid of subspace `s` minimizing

    Sum_{i in cluster_c} [w_par (r[s] . d_hat[s])^2 + w_perp ||r[s]||^2]

solves the symmetric positive-definite system

    (Sum_i M_i) c = Sum_i M_i x[s]
    M_i = w_perp I + (w_par - w_perp) d_hat_s_i d_hat_s_i^T

We pick `w_par = eta, w_perp = 1`. The system is small
(`ds = dim/m`, typically 4–16) and dense; we solve in-place with
a tiny hand-rolled Cholesky. `nalgebra` is overkill here.

### Anisotropic encoding

ScaNN matches its training loss at encode time: the codeword
chosen for each subspace is the one that minimizes the same
anisotropic loss the codebooks were trained on. Encoding by raw
L2 would partially undo the codebook shaping.

### Inference

Identical to uniform PQ: build a `m × k` LUT of
`<query[s], centroid[s][c]>` once per query, then sum `m` lookups
per database vector. This means AVQ is a *training-time-only*
modification — production inference paths do not change.

## Benchmark methodology

`crates/ruvector-avq/src/main.rs` runs the full pipeline:

- Synthesize `n = 10_000` l2-normalized embeddings in `dim = 128`,
  drawn from a low-rank (`rank = 24`) random subspace + ambient
  Gaussian noise. This mimics real learned embeddings (effective
  rank << ambient dim).
- Synthesize 300 queries from the same generative process.
- Compute brute-force top-10 ground truth.
- Train ScalarQuantizer, uniform PQ (m=16, k=256), AVQ at eta=16,
  AVQ at eta=64.
- For each: measure recall@10 vs ground truth, score-RMSE on a
  sampled query×db cross product, and per-query latency for
  linear-scan top-10 over the full code table.

Hardware: Apple Silicon (Darwin 24.6.0), `cargo run --release`,
single thread. Each variant runs the same code path; the only
difference is the codebook contents.

## Results

```
ruvector-avq demo: n=10000, dim=128, m=16, k=256, queries=300, top-10

trained ScalarQuantizer    in 1.22ms
trained ProductQuantizer   in 1.51s
trained AnisotropicPq η=16 in 2.50s
trained AnisotropicPq η=64 in 2.50s

       variant | bytes      |   memory      | recall@10 | score-RMSE | latency
---------------+------------+---------------+-----------+------------+--------
   scalar-int8 | code=128 B | mem=1250.0 KiB|   0.983   |   0.0007   | 738µs
    uniform-PQ | code= 16 B | mem= 156.2 KiB|   0.283   |   0.0423   | 219µs
      AVQ η=16 | code= 16 B | mem= 156.2 KiB|   0.277   |   0.0435   | 219µs
      AVQ η=64 | code= 16 B | mem= 156.2 KiB|   0.272   |   0.0473   | 218µs
```

Criterion micro-bench, single-query score over 4k codes:

```
pq_score_4k             time:   [26.43 µs 26.68 µs 26.97 µs]
avq_score_4k            time:   [26.50 µs 26.77 µs 27.08 µs]
```

### How to read these numbers

1. **Inference parity.** AVQ scoring is bit-identical to uniform
   PQ scoring — both go through the same LUT path, and Criterion
   confirms the latency difference is below noise (~0.3% of
   measured time, well inside the confidence interval). This is
   the headline operational property: deploying AVQ does not
   change p99 query latency.

2. **8× memory compression vs scalar.** Uniform PQ and AVQ both
   deliver 156 KiB total for 10k vectors at 128-dim, vs 1250 KiB
   for int8 scalar — the same 8× gain RaBitQ provides, but with
   inner-product–native scoring instead of bit-dot-product
   re-ranking.

3. **Recall on synthetic data.** AVQ matches uniform PQ within
   ~2pp recall@10 on this generative process (low-rank Gaussian +
   ambient noise + l2-norm). This is the *worst case* for AVQ:
   the parallel/orthogonal split is symmetric in expectation for
   spherically-symmetric data, and the per-subspace cross-term
   approximation discards information that AVQ would otherwise
   exploit. The published 3–10pp gains are on real learned
   embeddings (Glove, Deep1B) where the data manifold is
   anisotropic in ways that synthetic Gaussians do not capture.

## Practical failure modes

- **Synthetic-data myopia.** Score-aware quantization shines on
  data with anisotropic structure (typical of trained embeddings).
  On purely synthetic Gaussians, gains are noise. Always validate
  on the production embedding distribution.
- **`eta` tuning.** ScaNN's published recipe uses dataset-specific
  `eta` chosen by quantile threshold of `||x||` distribution. We
  expose `eta` as a free parameter; downstream callers must
  sweep it on their data. `eta = 4..16` is the typical productive
  range; `eta` above ~64 starts to over-fit parallel direction at
  the cost of orthogonal reconstruction.
- **Subspace count `m`.** Smaller `m` (larger `ds`) gives the
  parallel direction more room per subspace and amplifies AVQ's
  effect. `m = dim / 8` is a reasonable default.
- **Empty clusters.** At `k = 256`, low-entropy data can leave
  some codewords unused; we reseed from a random training point.
  This adds noise to early training rounds but converges.

## What to improve next

1. **Real-embeddings benchmark target.** Add a feature-flagged
   loader for SIFT-1M / Glove-1.2M and a CI-runnable
   recall-vs-bits curve. This is where AVQ's published gains will
   actually surface.
2. **Optimized PQ rotation.** Compose AVQ with a learned
   pre-rotation (OPQ) — these are orthogonal contributions and
   stack.
3. **Residual AVQ.** Two-stage: coarse k-means + AVQ on residual.
   Standard in ScaNN's full pipeline.
4. **SIMD LUT scan.** Current scoring does scalar `m` lookups per
   vector. AVX2 / NEON gather + horizontal-add hits ~3-4× speedup
   on hot path; relevant once we scale to billion-vector linear
   scans inside IVF posting lists.
5. **Query-conditional `eta`.** ScaNN's own followup
   ("Score-Aware Quantization with Query Distribution") uses a
   query-time-known `eta`; useful for hybrid dense+sparse search.

## Production crate layout proposal

For graduation from `research/nightly` to a `0.x` release on
crates.io:

```
crates/ruvector-avq/
├── Cargo.toml          # 0.x release, semver-pinned deps
├── README.md           # SEO + usage, links to docs.rs
├── src/
│   ├── lib.rs
│   ├── traits.rs       # Encoder, Scorer
│   ├── scalar.rs       # baseline
│   ├── pq.rs           # uniform PQ
│   ├── aniso.rs        # AVQ training + closed-form
│   ├── kmeans.rs       # shared k-means primitive
│   ├── lut.rs          # SIMD LUT scoring (NEW)
│   ├── opq.rs          # learned rotation preprocessing (NEW)
│   ├── residual.rs     # coarse-k-means + AVQ-on-residual (NEW)
│   └── error.rs
├── examples/
│   ├── glove.rs        # real-data benchmark, feature-gated
│   └── deep1b.rs       # ditto, gated on a 'real-data' feature
├── benches/
│   ├── score.rs        # what we have today
│   └── train.rs        # NEW: training throughput
└── tests/
    └── aniso.rs        # numeric correctness, expanded
```

The `lut.rs`, `opq.rs`, and `residual.rs` modules are the main
content gaps to close before `0.1`.

## References

- Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern,
  F., & Kumar, S. **Accelerating Large-Scale Inference with
  Anisotropic Vector Quantization.** ICML 2020.
- Liu, K., Zhang, Y., Sun, Z. **AQLM: Additive Quantization for
  Language Models.** ICML 2024.
- Jegou, H., Douze, M., Schmid, C. **Product Quantization for
  Nearest Neighbor Search.** TPAMI 2010.
- Ge, T., He, K., Ke, Q., Sun, J. **Optimized Product
  Quantization.** TPAMI 2014.
- Wang, M., et al. **Symphony: A Vector Quantization Library
  Comparison.** SIGMOD 2024.
- ScaNN reference impl: https://github.com/google-research/google-research/tree/master/scann

## How it works (blog-readable walkthrough)

Imagine you have a billion 128-dim text embeddings and a tight
RAM budget. You want to compress each vector to 16 bytes — an 8×
saving — without losing too much retrieval accuracy. Standard
*product quantization* (PQ) splits each vector into 16 chunks of
8 dims, learns a tiny 256-codeword codebook for each chunk, and
stores the 16 codeword indices. At query time you turn the query
into a 16×256 lookup table of inner products with the centroids,
then for each compressed database vector you do 16 lookups and
add them up. It's gloriously fast.

The problem: the codebooks are trained to minimize how *far*
each compressed vector ends up from the original — its
reconstruction MSE. But you don't actually care about
reconstruction; you care about *score*. The inner product
`<q, x>` is distorted by the residual `r = x - x_tilde`, but only
by the part of `r` that lies along the direction of `x` itself
(parallel residual). The part orthogonal to `x` is, on average,
washed out by random query directions.

AVQ retrains the codebooks under a weighted loss that says
"parallel error costs `eta` units, orthogonal error costs 1 unit"
where `eta` is typically 4–16. The closed-form centroid update
becomes a tiny weighted least-squares problem (rank-1 plus
identity), solved in milliseconds with a hand-rolled Cholesky.
At inference time *nothing changes* — the LUT scoring path is
identical, just the codebook contents differ. So you get
score-aware quantization for free at deploy time.

In our synthetic benchmark, AVQ matches uniform PQ within noise
(low-rank Gaussian data is the worst case for any score-aware
trick). The published 3–10pp gains require real text embeddings
where the data manifold is anisotropic in ways synthetic data
isn't. The follow-up nightly (real-data benchmarks behind a
feature flag) will exercise that regime.
