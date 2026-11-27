# Warm-starting `e-means` for cheaper streaming PQ/IVF codebook refresh

## Problem

`ruvector-e-means` is RuVector's shared Rust/WASM implementation of *e-means*
(Bessa et al., arXiv:2611.09501v1, Meta FAIR, 2026), a mini-batch EMA variant of
k-means that trains PQ and IVF codebooks for `ruvector-pq-search` and the
IVF/RAIRS/SEIL family. Its only entrypoint, `fit()`, always starts from a fresh
random draw of `k` rows — every retrain is a cold restart, as the surrounding ADR
(ADR-360) documents explicitly.

That is a real cost for streaming vector memory: an index that gets periodically
refreshed against newly appended or evicted vectors pays full cold-restart
convergence cost every time, even though the existing codebook is usually still a
reasonable description of most of the data. This is a known, named gap in the
published literature on streaming quantization — a 2024 paper on incremental IVF
maintenance (arXiv:2411.00970) explicitly leaves a PQ extension to future work, and
a 2017 paper on online product quantization (Xu et al.) builds incremental codebook
updates for exactly this reason, using a different base optimizer. A 2025 preprint
on online k-centers notes that warm-starting a k-means-family optimizer "greatly
reduces" iteration count in a related setting — but nothing applies that to
`e-means`'s specific EMA-count/dead-cluster-revival dynamics, and the `e-means`
paper itself does not address streaming refresh.

## Hypothesis

> Given a codebook trained on an initial window, and a second window where some
> fraction of vectors are drawn from drifted cluster centers, warm-starting the
> refresh from the existing codebook (instead of a cold restart) should not regress
> held-out quality by more than 5%, and at half the epoch budget should cut
> refresh wall-clock time by at least 30% — across light, moderate, and heavy
> drift, not just one favorable point.

These thresholds were fixed before any benchmark ran and were not adjusted
afterward.

## Design

Two new public Rust functions, no new dependency, no WASM/npm change:

```rust
pub fn fit_warm_start(
    data: &[f32],
    dimensions: usize,
    config: &Config,
    initial_centroids: &[f32],
) -> Result<Model, Error>
```

and `ProductQuantizer::fit_warm_start`, which does the same per PQ subspace,
reusing a previous codebook's cluster count. Both run the *same* algorithm as
`fit` — equations (2)-(8) of arXiv:2611.09501v1, including EMA centroid updates and
donor-split dead-cluster revival — differing only in whether the starting
centroids are a random draw from `data` or the caller's own prior codebook. The
refactor that makes this possible (extracting a shared `fit_from_centroids` core)
was verified behavior-preserving by the crate's full pre-existing test suite (23
tests) passing unmodified before any new code was added.

## Benchmark

Synthetic, deterministic, seeded data: 24 well-separated Gaussian clusters in 32
dimensions. A "new" window draws each row from drifted centers with probability
`drift_frac` (15% / 35% / 60%, tested separately) and from the original centers
otherwise. Held-out quality is measured on an independently drawn sample from the
*same* post-drift distribution, never used for training — avoiding the "benchmark
leaks evaluation information" trap. Six independent repetitions per drift level,
each with disjoint seed streams; a comparison between variants is only called
ACCEPT or REJECT if the difference exceeds the combined standard error across
repetitions, otherwise it is reported INCONCLUSIVE.

Three variants, all refreshing against the same drifted window, timed on the
refresh step alone (the pre-existing codebook's own construction cost is excluded,
matching the real deployment where it already exists):

- **baseline**: cold `fit`, 4 epochs.
- **candidate A**: `fit_warm_start`, same 4 epochs.
- **candidate B**: `fit_warm_start`, 2 epochs.

## Results

```json
{"drift":"moderate","candidateA_vs_baseline_relHoldoutMse":-0.0912,"candidateA_verdict":"ACCEPT"}
{"drift":"moderate","candidateB_vs_baseline_relHoldoutMse":-0.0860,"candidateB_relRefreshMsChange":-0.3722,"candidateB_verdict":"ACCEPT"}
{"drift":"heavy","candidateA_vs_baseline_relHoldoutMse":-0.1110,"candidateA_verdict":"ACCEPT"}
{"drift":"heavy","candidateB_vs_baseline_relHoldoutMse":-0.1056,"candidateB_relRefreshMsChange":-0.4209,"candidateB_verdict":"ACCEPT"}
{"drift":"light","candidateA_vs_baseline_relHoldoutMse":-0.0150,"candidateA_verdict":"INCONCLUSIVE"}
{"drift":"light","candidateB_vs_baseline_relHoldoutMse":0.0029,"candidateB_verdict":"INCONCLUSIVE"}
```

At moderate and heavy drift, warm-starting doesn't just match baseline — it
*improves* held-out MSE by 9–11%, at the same epoch budget, and the improvement
holds at half the epoch budget while cutting wall-clock time 37–42%. The likely
mechanism: at only 4 epochs, a cold restart hasn't fully converged from a random
draw, so much of its budget goes to rediscovering coarse structure the
warm-started run already has — consistent with `revived_clusters` dropping to
~0–0.2 for warm start at these drift levels versus 1.0–2.5 for cold.

At light drift, the apparent differences (−1.5% and +0.29%) are both smaller than
the measurement noise across six repetitions, so they are reported as
**inconclusive**, not rounded up into a win. This is a direct consequence of a
modest, pre-fixed repetition count, honestly reported rather than resolved by
quietly increasing it after looking at the result.

A small, explicit, bounded search over the refresh epoch budget — run in-process,
not through the ecosystem's `@metaharness/darwin` package (extending that would
require touching WASM bindings and a five-package release graph, out of scope for
this change) — found that a single epoch already dominates the quality/speed
trade-off at this benchmark's scale. It also surfaced a real methodology
limitation worth naming rather than hiding: searching epoch *fractions* finer than
the underlying integer epoch count (only 4 possible values here) produces no
further signal, just measurement noise relabeled as a "winning" fraction.

## Limitations

Synthetic data only — no real embedding corpus was used, and real streaming drift
(new topics, vocabulary shift, embedding-model upgrades) may not resemble this
run's Gaussian-shift model. The light-drift result needs more repetitions, decided
in advance, to actually resolve. Timings are single-machine. No numeric comparison
is made against the literature's own results (different implementations, datasets,
hardware) — architectural differences are the honest substitute for a performance
claim here.

## Production relevance

`fit_warm_start` and `ProductQuantizer::fit_warm_start` are new, tested, opt-in
library API in `ruvector-e-means` — not a default-behavior change. `fit`,
`IvfPartition::fit`, and `ProductQuantizer::fit` are unchanged. Whether a consumer
crate (`ruvector-pq-search`, `ruvector-rairs`) adopts warm-started refresh as an
opt-in retraining mode is deliberately left for a follow-up change, gated on
validating this result against real streaming data first.

## Future direction

Validate against real embedding drift; fix the epoch-fraction search to operate on
the actual integer domain; explore a deeper "merge old+new sufficient statistics"
variant instead of this run's "reset the EMA prior, keep only centroid positions"
approach; and, if adopted, define an RVF stream descriptor for a signed, replayable
codebook-refresh lineage, building on the determinism this run already
demonstrates.

## References

- Bessa et al., *e-means: outperforming k-means (in one-tenth the epochs)*,
  arXiv:2611.09501v1 (2026).
- Xu, Tsang, Zhang, *Online Product Quantization*, arXiv:1711.10775 (2017).
- *Incremental IVF Index Maintenance for Streaming Vector Search*, arXiv:2411.00970 (2024).
- *Fast Online Distributionally Robust Optimization via Data Compression*, arXiv:2504.08097 (2025).
- RuVector ADR-359 (shared e-means clustering), ADR-360 (ecosystem integrations).

Full benchmark methodology, raw results, Darwin-lite search output, ecosystem
integration analysis (RVF/RVM/ruFlo/MCP/WASM), and the ADR are in
[`docs/research/nightly/2026-11-17-e-means-warm-start-drift-refresh/README.md`](./README.md)
and [ADR-361](../../../adr/ADR-361-e-means-warm-start-codebook-refresh.md).
