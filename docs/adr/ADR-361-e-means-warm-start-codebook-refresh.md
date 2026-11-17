# ADR-361: Warm-started `e-means` refresh for streaming PQ/IVF codebooks

- **Status**: Accepted as opt-in library API; not wired into any consumer's default
  retraining path; promotion to a consumer default is future work pending real-data
  validation (see [Open questions](#open-questions))
- **Date**: 2026-11-17
- **Related**: [ADR-359](./ADR-359-e-means-shared-clustering.md) (shared e-means
  core), [ADR-360](./ADR-360-e-means-ecosystem-integrations.md) (PQ/IVF/MetaHarness/
  Cloudflare consumers), nightly research report
  [`docs/research/nightly/2026-11-17-e-means-warm-start-drift-refresh`](../research/nightly/2026-11-17-e-means-warm-start-drift-refresh/README.md)

## Context

`ruvector-e-means`'s only training entrypoint, `fit()`, always initializes centroids
from a fresh seeded-random draw of `k` rows (ADR-359 §Decision, item 1). ADR-360
documents the consequence explicitly: "Successful explicit retraining resets
inserted vectors and IDs" — every retrain of a `ruvector-pq-search` codebook or an
IVF/RAIRS/SEIL partition is a cold restart, even when the index is being refreshed
against a window of mostly-unchanged data with only incremental drift. Published
work on streaming quantization (Xu et al. 2017's Online Product Quantization;
arXiv:2411.00970 on incremental IVF maintenance, which explicitly leaves a PQ
extension to future work) treats this cold-restart cost as a real, open problem for
PQ/IVF specifically; a 2025 preprint on online k-centers (arXiv:2504.08097) found
warm-starting a related k-means-family optimizer "greatly reduces" iteration count.
Nothing found applies a warm start to `e-means`'s specific EMA-count/dead-cluster-
revival dynamics, and the `e-means` paper (arXiv:2611.09501v1) itself does not
address streaming refresh.

## Hypothesis

Warm-starting a refresh (initializing `e-means` from an existing codebook rather
than a random draw) reduces the epoch budget needed to reach acceptable held-out
quality on a drifted data window, relative to a cold restart, without regressing
quality beyond a small, fixed tolerance. See the nightly report for the full
formal Given/When/Then statement and the pre-registered acceptance thresholds
(`QUALITY_TOLERANCE = 0.05`, `PARITY_TOLERANCE = 0.02`, `SPEED_TARGET = 0.30`).

## Decision

Add two new public functions to `ruvector-e-means` 0.1.0, additive and
non-breaking:

```rust
pub fn fit_warm_start(
    data: &[f32],
    dimensions: usize,
    config: &Config,
    initial_centroids: &[f32],
) -> Result<Model, Error>;

impl ProductQuantizer {
    pub fn fit_warm_start(
        data: &[f32],
        dimensions: usize,
        subspaces: usize,
        config: &Config,
        previous: &ProductQuantizer,
    ) -> Result<Self, Error>;
}
```

`fit_warm_start` runs the identical algorithm to `fit` (ADR-359's five-step
recipe, unchanged) against a shared, newly extracted private core
(`fit_from_centroids`), differing only in whether `centroids: Vec<f32>` comes from
a random draw or the caller. `fit`'s refactor onto that shared core was verified
non-breaking by the crate's full pre-existing test suite (23 tests, unmodified)
passing before any new test was added.

`ProductQuantizer::fit_warm_start` trains each subspace via `fit_warm_start` against
the corresponding subspace of `previous`'s codebook, and — deliberately — ignores
`config.clusters` in favor of `previous.clusters()`, so a stale `Config` cannot
silently resize a codebook already in use. It rejects a `previous` whose
`dimensions`/`subspaces` don't match the call.

`IvfPartition` needs no new code: `IvfPartition::from_model` already accepts any
valid `Model`, including one produced by `fit_warm_start`.

**Not decided here:** whether `ruvector-pq-search` or `ruvector-rairs` should call
`fit_warm_start` by default when refreshing a live codebook. That is left for a
follow-up change once real-data evidence exists (see
[Open questions](#open-questions)); this ADR only adds the primitive.

## Evidence

Benchmarked in `crates/ruvector-e-means/examples/warm_start_drift_benchmark.rs`
against a deterministic, synthetic streaming-drift workload (24 Gaussian clusters,
32 dimensions, three drift levels, six repetitions each, held-out MSE never used
for training). Measured result: at moderate (35%) and heavy (60%) simulated drift,
warm-starting at the same epoch budget as baseline improves held-out MSE by
9–11% and cuts refresh wall-clock time by 37–42% at half the epoch budget, both
within pre-registered thresholds (**ACCEPT**). At light (15%) drift, the
difference from baseline is smaller than the measurement noise across six
repetitions (**INCONCLUSIVE**, not rounded up). Full raw output, methodology, and
the standard-error-aware accept/reject rule are in the nightly report linked
above; this ADR does not restate the numbers.

A bounded, in-process search over the refresh epoch budget (explicitly documented
as *not* an invocation of `@metaharness/darwin` — see the nightly report's
"MetaHarness and Darwin scoping" section) found that a single epoch already
dominates the measured quality/speed trade-off at this benchmark's scale, and also
surfaced a real methodology limitation (searching epoch fractions finer than the
underlying 4-valued integer domain adds no signal) that is reported, not hidden.

## Consequences

- New, opt-in, additive API surface in `ruvector-e-means`; zero behavior change to
  any existing public function or persisted format.
- `ruvector-pq-search`/`ruvector-rairs` gain an available-but-unused capability;
  no release coordination required for this ADR alone (contrast ADR-360's
  multi-package release graph, which this change deliberately does not touch).
- The crate gains a second example binary (`warm_start_drift_benchmark`) that
  doubles as a regression check: any future change to `fit_from_centroids` that
  breaks warm-start quality parity will show up when this benchmark is re-run.
- No change to WASM binary size, npm packages, or the Cloudflare Worker surface
  (none of `src/wasm.rs`, `npm/packages/e-means/*`, or
  `npm/packages/e-means/integrations/metaharness.mjs` were modified).

## Alternatives considered

1. **Extend `npm/packages/e-means/integrations/metaharness.mjs`'s evaluator genome
   with an initial-centroid hook**, so the real `@metaharness/darwin` could tune
   the refresh epoch budget directly. Rejected for this change: it requires
   touching the WASM bridge and the pinned-version release graph across four npm
   facade packages plus `ruv-fann` (ADR-360's "release coordination" section) — a
   legitimate, larger, separately reviewable change.
2. **True incremental training** (persist and continue `counts`/`sums` as new
   `Model` fields, rather than re-seeding the `1e-10` EMA prior around the warm
   centroids). Likely a better warm start, but requires a public `Model`
   field/serialization change affecting every existing consumer. Deferred as a
   follow-up, not rejected on quality grounds.
3. **Do nothing; accept cold-restart cost as a `ruvector-pq-search`/`ruvector-rairs`
   problem to solve independently.** Rejected: the primitive belongs in the shared
   kernel (ADR-359's whole premise is that clustering numerics should not diverge
   per consumer), and the measured evidence here supports building it once,
   centrally.

## Implementation plan

Done in this change: `trainer.rs` refactor + `fit_warm_start`;
`quantization.rs`'s `ProductQuantizer::fit_warm_start`; `lib.rs` export; 5 new
tests in `tests/warm_start.rs`; the drift benchmark example; README update.

Not done, left for follow-up: wiring into `ruvector-pq-search`/`ruvector-rairs`;
real-embedding-drift validation; the WASM/npm/MetaHarness extension from
Alternative 1; the RVF stream-descriptor lineage format described in the nightly
report's RVF integration analysis.

## API shape

See [Decision](#decision). Both new functions return the existing `Model`/
`ProductQuantizer` types — no new public types were added. `Error` messages for
the new validation paths: `"warm-start centroids length must equal clusters *
dimensions"` and `"PQ warm-start codebook must match this call's dimensions and
subspaces"`.

## Feature flags

None added. `fit_warm_start` is unconditional, matching `fit`'s existing
unconditional availability (only the `wasm` module is feature-gated in this
crate).

## Benchmark evidence

See [Evidence](#evidence) and the nightly report's "Benchmark methodology" and
"Benchmark results" sections for the full raw output, exact command
(`cargo run -p ruvector-e-means --release --example warm_start_drift_benchmark`),
and reproducibility notes (MSE bit-identical across two runs; wall-clock figures
vary by ordinary measurement noise).

## Security

No new attack surface: pure functions over caller-provided slices, same
`validate_rows` finiteness/shape checks as existing `fit`/`ProductQuantizer::fit`,
no I/O, no new dependency, no unsafe code. `ProductQuantizer`'s only public
constructors all validate shape, so no reachable value can reach
`fit_warm_start`'s subspace-matching check with an internally inconsistent
`previous`.

## Governance

Not promoted to any consumer default (see [Decision](#decision)). No MCP surface
added (there is no "current production codebook" concept yet for a tool to act
on — see the nightly report's MCP analysis for why one would be premature). No
RVM or witness-signing integration added, for the same "nothing to gate yet"
reason, documented rather than silently skipped.

## Failure modes

See the nightly report's "Failure modes and rejected alternatives" and
"Limitations and falsification" sections: light-drift statistical
inconclusiveness, the Darwin-lite search's sub-epoch resolution degeneracy, and
reliance on synthetic (not real) drift data are all reported there in full.

## Migration

None required. No existing call site changes; `fit`, `IvfPartition::fit`, and
`ProductQuantizer::fit` are unchanged.

## Rollback

Trivial: the new functions are additive. Rollback means not calling
`fit_warm_start`/`ProductQuantizer::fit_warm_start`; no persisted format, config
schema, or existing API changed.

## Rejection criteria

This ADR's decision (ship the opt-in primitive) would be reversed if: the
refactor extracting `fit_from_centroids` is found to have changed `fit`'s
observable output for any input (it did not, per the unmodified pre-existing test
suite, but this remains the standing bar for any further change to that shared
core); or if real-embedding-drift validation (Next research item 1 in the nightly
report) finds the synthetic benchmark's conclusion does not hold in practice —
in which case the primitive would likely be kept (it is harmless and correctly
scoped as opt-in) but not recommended for adoption.

## Open questions

1. Does the measured benefit hold on real, logged streaming-embedding drift
   (not synthetic Gaussian shift)?
2. Should `ruvector-pq-search`/`ruvector-rairs` adopt `fit_warm_start` as an
   opt-in retraining mode, and with what fallback-to-cold-restart policy when a
   warm-started refresh's held-out quality fails the tolerance check?
3. Is the "reset EMA prior, keep centroid positions" warm start (implemented
   here) close enough to optimal, or does true incremental `counts`/`sums`
   continuation (Alternative 2) matter enough to justify its larger API surface?
4. What is the actual WASM binary-size and on-device timing cost of exposing
   `fit_warm_start` through `@ruvector/e-means`'s WASM bridge — estimated, not
   measured, in the nightly report?
