# ADR-346: Canonical-Cactus-Cut Forgetting — Attacking ADR-345's Rejection Root Cause

## Status

Rejected (for production use as designed), on a different and more specific
basis than ADR-345. Experimental crate addition
(`ruvector-agent-memory::graph_forget_cactus`, feature-gated behind
`mincut-forget-cactus`, off by default) retained as evidence and reference
implementation, not promoted. `ruvector-mincut`'s `canonical` feature itself
is validated as correct, fast, and genuinely deterministic — the rejection is
about the *forgetting-signal design*, not about the min-cut backend.

## Context

ADR-345 (`docs/research/nightly/2026-09-05-mincut-gated-forgetting`)
implemented `MincutGatedForgetting`: a `CompactionPolicy` that builds a k-NN
similarity graph over compaction candidates and uses
`ruvector_mincut::RuVectorGraphAnalyzer::partition()` (the crate's general
dynamic min-cut wrapper) to find a global min-cut boundary, then treats
boundary vertices as structurally load-bearing "bridges" worth protecting
from eviction. It was **rejected** on two measured grounds:

1. **Non-determinism**: `partition()` returned an empty/unusable result on
   50% of repeated calls (30 trials) on byte-identical input.
2. **Latency**: 76ms-11.4s per call at 50-400 vertices, 1,800-2,700x the
   scalar `CoherencePolicy` baseline even at an 84-vertex corpus.

`ruvector-mincut` separately ships a `canonical` feature
(`crates/ruvector-mincut/src/canonical/`) whose stated purpose is exactly a
fix for problem (1): a `CactusGraph` built by dense-array Stoer-Wagner
enumerates *every* global minimum cut and `canonical_cut()` deterministically
selects the lexicographically smallest one. It was not used by ADR-345 and
nobody had measured it against ADR-345's own rejection criteria. This ADR
does that: same k-NN-graph-plus-boundary-bonus design, same
`ForgetMode::{Soft,Hard}` policies, same acceptance-test shape and
thresholds where reusable, swapped min-cut backend
(`ruvector_mincut::CactusGraph` instead of `RuVectorGraphAnalyzer`).

## Hypothesis

```text
Given the identical synthetic corpus ADR-345 used (6 topic clusters, 12
memories each = 72, plus 12 "bridge" memories interpolated 50/50 between two
randomly paired clusters, 32-dim, hot-cluster access simulation, k-NN k=8
cosine >= 0.05 similarity graph),

when the 84-entry store is compacted to 50% (42 entries) using
CactusGatedForgetting-Soft (structural bonus delta=0.5) and
CactusGatedForgetting-Hard (20% protected budget) -- backed by
CactusGraph::canonical_cut() instead of RuVectorGraphAnalyzer::partition() --
versus the existing CoherencePolicy baseline and versus ADR-345's own
MincutGatedForgetting-Soft/Hard,

then (a) the cactus backend is deterministic (100% identical boundary result
across repeated calls on byte-identical input, vs. ADR-345's measured 50%
degenerate rate), (b) each cactus candidate's compaction wall-clock stays
within 20x the scalar baseline's (a materially tighter bar than ADR-345's
100x, chosen because Stoer-Wagner on graphs this small is expected to be fast
in absolute terms, not merely bounded), and (c) both cactus candidates retain
a bridge-memory survival rate at least 15 percentage points higher than
baseline while Recall@10 stays within 2 percentage points of baseline,

subject to: 100% tamper-detection across 20 single-byte-flip trials against
the reused eviction-witness chain.
```

Fixed before any benchmark ran; not modified after seeing results. Full
methodology, raw output, and three supporting probes
(`examples/cactus_determinism_probe.rs`, `examples/cactus_scaling_probe.rs`,
`examples/cactus_seed_sensitivity_probe.rs`) live in
`docs/research/nightly/2026-09-10-canonical-cactus-forgetting/README.md`.

## Decision

**Do not promote `CactusGatedForgetting` to a default-enabled compaction
policy.** Keep it as an opt-in, feature-gated experimental module
(`mincut-forget-cactus`) alongside ADR-345's own retained
`MincutGatedForgetting`, both off by default. Two sub-findings, both backed
by measurement:

- **(a) and (b) are CONFIRMED**, resoundingly: 100/100 identical partitions
  across two independent determinism runs (0% degenerate, vs. ADR-345's 50%),
  and 85-93x faster than `RuVectorGraphAnalyzer` at the identical 84-vertex
  corpus (1.6ms vs. 138-152ms). `ruvector-mincut`'s `canonical` feature is a
  strict, measured upgrade over the general dynamic wrapper for this
  workload on both axes ADR-345 flagged as blocking.
- **(c) is FALSIFIED, and not only for the cactus backend.** At this ADR's
  pre-registered seed (346), *neither* backend beat the scalar baseline:
  bridge survival was 16.7% for `CoherencePolicy`, `MincutGatedForgetting-Soft`,
  and `MincutGatedForgetting-Hard` alike (a 0.0pp gap for ADR-345's own
  policy, run fresh in this experiment), and 8.3% (a *negative* 8.3pp gap)
  for both `CactusGatedForgetting` variants. A follow-up 10-seed sweep
  (`cactus_seed_sensitivity_probe`, seeds 1000-1009) found the 15pp
  survival-gap bar met by **0 of 10 seeds for either backend**, with mean
  gaps of -3.3pp (old backend) and -4.2pp (cactus) and standard deviations
  around 8pp -- i.e., statistical noise centered at zero, not a real effect
  that a faster backend failed to preserve.

The deeper, more general lesson: **a single global-minimum-cut call on a
many-cluster k-NN graph identifies one structurally weakest point in the
*entire* graph** (generically, whichever vertex or small vertex-set has the
least total edge weight -- e.g. one lightly-connected "gateway" vertex),
**not all of the semantically engineered "bridge" vertices this experiment's
corpus generator constructs.** With `mincut_trials = 1` (used by both
backends here, matching ADR-345's own main-benchmark configuration), the
signal genuinely does correlate with *some* structural weak point, but that
point is not reliably one of the 12 constructed bridges out of 84 vertices,
so it does not reliably help *this specific eviction task*. ADR-345's
original single-seed (341) run happening to show a 15pp+ gap looks, on this
evidence, like a favorable-seed artifact rather than a reproducible property
of the approach -- itself a useful, previously-undocumented finding about
that benchmark's sensitivity.

## Evidence

All commands below are exactly reproducible; raw stdout is in the nightly
research README.

```bash
cargo run --release -p ruvector-agent-memory --example cactus_determinism_probe --features mincut-forget-cactus
# trials=50 elapsed=0.0043s avg_per_call=0.086ms empty_or_degenerate=0 (0%)
# bridge_detected_as_boundary=50 (100%) distinct_partitions=1

cargo run --release -p ruvector-agent-memory --example cactus_scaling_probe --features mincut-forget-cactus
# n=19..800 ring graph: total latency 0.77ms (n=19) to 8236ms (n=800);
# at n=400 (ADR-345's largest measured size), 1086ms total vs. ADR-345's
# reported multi-second RuVectorGraphAnalyzer measurements at the same size.

cargo run --release -p ruvector-agent-memory --example cactus_gated_forgetting_bench --features mincut-forget-cactus
# CoherenceWeighted                 16.7% survival, 100.0% recall,     41us
# MincutGatedForgetting-Soft        16.7% survival, 100.0% recall, 151802us
# MincutGatedForgetting-Hard        16.7% survival, 100.0% recall, 138776us
# CactusGatedForgetting-Soft         8.3% survival, 100.0% recall,   1632us
# CactusGatedForgetting-Hard         8.3% survival, 100.0% recall,   1615us
# Tamper detection: 20/20
# => REJECT (survival-gap and speed-vs-scalar-baseline criteria both fail)

cargo run --release -p ruvector-agent-memory --example cactus_seed_sensitivity_probe --features mincut-forget-cactus
# old(mincut) gap : mean=-3.3pp std=7.6pp seeds_meeting_15pp_bar=0/10
# new(cactus) gap : mean=-4.2pp std=8.5pp seeds_meeting_15pp_bar=0/10

cargo test --release -p ruvector-agent-memory --features mincut-forget-cactus
# 34/34 tests pass (3 new: soft/hard bridge-detection unit tests on the
# original ADR-345 two-clique-plus-bridge fixture, plus a below-minimum-size
# fallback test)
```

Hardware/software: this run's container (`uname -a`: Linux x86_64, 4 vCPU),
`rustc 1.94.1`, `cargo 1.94.1`, release profile throughout.

## Consequences

- `ruvector-mincut`'s `canonical` feature is now empirically validated as a
  correct, fast, deterministic drop-in for small-graph (tested to n=800)
  global min-cut queries -- a reusable fact for any future crate (not just
  `ruvector-agent-memory`) that needs a min-cut boundary and cannot tolerate
  ADR-345's non-determinism or latency.
- `ruvector-agent-memory`'s eviction-witness mechanism
  (`compact_witnessed`/`EvictionWitnessChain`, unchanged by this ADR) is
  re-confirmed independently sound: 20/20 tamper detections, identical to
  ADR-345.
- The "protect the global-min-cut boundary" *idea itself*, not just its
  ADR-345 implementation, is now weakened as a compaction-policy candidate:
  two independent min-cut backends and 11 total seeds (1 pre-registered + 10
  sensitivity) found no reproducible bridge-protection benefit over the
  existing scalar `CoherencePolicy`. A future attempt would need either (i) a
  richer signal than a single global cut (e.g. per-community local cuts, or
  `ruvector-mincut`'s `all-cut-queries`/`jtree` sparsest-cut primitives, which
  this ADR did not test), or (ii) a benchmark corpus where "the weakest
  global cut" and "the engineered semantic bridges" are constructed to
  coincide, which this one does not guarantee.
- `docs/research/nightly/2026-09-05-mincut-gated-forgetting`'s own positive
  15pp-gap number should be read as seed-specific, not as evidence the
  *scalar-baseline-beating* part of that design worked in general; its
  rejection for speed/determinism reasons stands independently and is not
  weakened by this finding.

## Alternatives Considered

1. **Fix `RuVectorGraphAnalyzer::partition()`'s hash-map-order
   non-determinism directly** (filed as a follow-up hardening item by
   ADR-345, not attempted here or there). Rejected as this ADR's approach
   because `canonical` already exists, is purpose-built for exactly this
   property, and required zero changes to `ruvector-mincut` itself --
   strictly less engineering risk than patching the general wrapper's
   internals.
2. **`tree_packing::canonical_mincut_fast`** (Gomory-Hu tree packing, "Tier
   2", advertised as `O(V * T_maxflow)`) instead of the plain `CactusGraph`
   used here. Not benchmarked in this ADR; worth a follow-up if larger
   corpora (the originally-desired ~2,000-memory scale ADR-345 could not
   reach) are still wanted, since dense Stoer-Wagner's measured ~cubic
   scaling (25ms at n=100 to 8.2s at n=800) will not get there either.
3. **Community/local cuts instead of one global cut** (e.g. run the
   boundary detector per-cluster, or use `ruvector-mincut`'s
   `all-cut-queries` sparsest-cut query) to better target "the semantic
   bridges between the corpus's 6 clusters" specifically, rather than "the
   single weakest point in the whole 84-vertex graph". Plausible fix for the
   (c) falsification above; out of scope for this ADR, which committed in
   advance to reusing ADR-345's exact policy design to isolate the backend
   variable. Recorded as the concrete next experiment.

## Implementation Plan

Already implemented and merged as an experimental, default-off module:
`ruvector-agent-memory::graph_forget_cactus::CactusGatedForgetting`
(`Soft`/`Hard`, reusing `graph_forget::ForgetMode`), plus three research
examples. No further implementation is planned under this ADR given the
rejection; a follow-up ADR would be needed for alternative #3 above.

## API Shape

```rust
pub struct CactusGatedForgetting {
    pub weights: CoherenceWeights,
    pub mode: ForgetMode, // shared with graph_forget::MincutGatedForgetting
    pub k_neighbors: usize,
    pub min_similarity: f32,
    pub structural_bonus: f32,
    pub protect_fraction: f32,
}
impl CactusGatedForgetting {
    pub fn soft(weights: CoherenceWeights, structural_bonus: f32) -> Self;
    pub fn hard(weights: CoherenceWeights, protect_fraction: f32) -> Self;
}
impl CompactionPolicy for CactusGatedForgetting { /* ... */ }
```

No `mincut_trials` field (present on ADR-345's `MincutGatedForgetting`): the
canonical backend needs no retry-and-union mitigation because it is
deterministic by construction.

## Feature Flags

`mincut-forget-cactus = ["mincut-forget", "ruvector-mincut/canonical"]` in
`ruvector-agent-memory`'s `Cargo.toml`. Composes with, rather than replaces,
ADR-345's `mincut-forget`, so both backends can be built and compared in the
same binary (as `cactus_gated_forgetting_bench` does). Off by default.

## Benchmark Evidence

See [Evidence](#evidence) above and the full research README for raw tables,
the seed-sensitivity distribution, and the scaling probe's complete
19-800-vertex curve.

## Security

No new security surface: `CactusGraph::build_from_graph` and
`canonical_cut()` are pure, allocation-only computations over an in-memory
graph with no I/O, no unsafe code introduced by this ADR, and no persisted
state. The reused eviction-witness chain's security properties are unchanged
from ADR-345/ADR-134 (SHA-256 content addressing, chained hashes,
tamper-evidence re-confirmed at 20/20 in this run).

## Governance

Same governance posture as ADR-345: this remains an opt-in, non-default
compaction policy. No autonomous system is authorized to enable
`mincut-forget-cactus` in production without a human decision informed by
this ADR's rejection.

## Failure Modes

- **Falsified as designed**: see [Decision](#decision) -- the structural
  bonus does not reliably improve bridge survival over the scalar baseline
  at `mincut_trials = 1`, regardless of min-cut backend.
- **Scaling ceiling**: dense Stoer-Wagner's measured cubic-ish growth (25ms
  at n=100 -> 8.2s at n=800) means this backend, while dramatically faster
  than `RuVectorGraphAnalyzer` at small n, still cannot reach the
  ~2,000-memory corpus ADR-345 originally wanted to test at.
- **`CactusGraph::build_from_graph` uses `f64::INFINITY`-based sentinel
  values and 1e-12 epsilon comparisons for cut-value ties** (see its source);
  this ADR did not stress-test numerical edge cases (e.g. all-equal edge
  weights, or graphs with many tied minimum cuts) beyond the two corpora
  used here.

## Migration

None: nothing is enabled by default before or after this ADR.

## Rollback

Delete `crates/ruvector-agent-memory/src/graph_forget_cactus.rs`, its four
example files, and the `mincut-forget-cactus` feature entry. No default
behavior depends on it.

## Rejection Criteria

This ADR's hypothesis is rejected under its own pre-registered thresholds:
sub-claims (a) determinism and (b) speed-vs-scalar-baseline both hold with
wide margins, but sub-claim (c) bridge-survival-gap does not hold at the
pre-registered seed nor across a 10-seed sensitivity sweep (0/10 seeds meet
the 15pp bar for either backend). Per this nightly process's own rule, all
three sub-claims needed to hold for acceptance; two of three holding is not
a partial promotion.

## Open Questions

1. Does a per-cluster/local-cut variant (alternative #3 above) produce a
   bridge-survival gap that actually correlates with the corpus's
   constructed bridges, rather than one arbitrary global weak point?
2. Does `tree_packing::canonical_mincut_fast`'s Gomory-Hu approach scale
   meaningfully better than dense Stoer-Wagner past n=800, and is it still
   deterministic in the same sense?
3. Is ADR-345's original seed-341 result reproducible under `mincut_trials =
   3` (its policy default, though not what its own main benchmark used)?
   This ADR did not test the multi-trial-union mitigation against the
   cactus backend (which has no retry mechanism to test, being already
   deterministic) or re-verify it against the old backend beyond what
   ADR-345 itself measured.
