# ADR-346: Direct `DynamicMinCut` Backend for Mincut-Gated Forgetting

## Status

Accepted (performance sub-hypothesis only) / Rejected (overall production
use, unchanged from ADR-345). `MincutBackend::Direct` is merged as an
opt-in, feature-gated addition to `ruvector-agent-memory::graph_forget`,
default-off, alongside the existing `MincutBackend::Wrapper` (unchanged
default). `MincutGatedForgetting` remains **not promoted** to a recommended
or default compaction policy.

## Context

ADR-345 (nightly 2026-09-05) implemented `MincutGatedForgetting`, a
`CompactionPolicy` that uses `ruvector-mincut`'s `RuVectorGraphAnalyzer` to
find structurally load-bearing "bridge" memories before compaction. It was
rejected on two independent axes: a ~1,800-2,700x compaction slowdown
against the required <=100x gate, and a measured 0.0pp bridge-survival
improvement over the scalar baseline (required >=15pp). Its "Next Research"
item 1 asked whether the performance failure was specific to
`RuVectorGraphAnalyzer` — which rebuilds a `MinCutWrapper` and replays every
edge into each of O(log range) bounded-range instances on every call — or
inherent to `ruvector-mincut` itself. It named `ruvector_mincut::DynamicMinCut`
(used directly, bypassing the wrapper) as the specific alternative to try.

Reading `ruvector-mincut`'s source confirms the two entry points are
architecturally different, not just different call sites on the same
algorithm: `RuVectorGraphAnalyzer` → `MinCutWrapper::query()` walks up to
100 geometrically-scaled bounded-range instances, lazily replaying the
*entire* edge set into each newly-instantiated one
(`process_instances`, `crates/ruvector-mincut/src/wrapper/mod.rs`).
`DynamicMinCut` (`crates/ruvector-mincut/src/algorithm/mod.rs`), built via
`MinCutBuilder`, instead runs one sparse, deterministic Stoer-Wagner-style
exact global-min-cut solve per (re)build
(`crates/ruvector-mincut/src/algorithm/exact.rs`: "Full recomputation is
polynomial, not subpolynomial").

## Hypothesis

This ADR reuses ADR-345's exact corpus, hypothesis text, and acceptance
thresholds without modification (its own "don't move the goalposts" Next
Research item 3), adding one variable: the backend.

```text
Given the same 84-memory synthetic corpus (6 clusters x 12 + 12 bridges,
32-dim, k-NN k=5, cosine >= 0.05) and the same MincutGatedForgetting-Soft /
-Hard policies,

when boundary detection uses MincutBackend::Direct (ruvector_mincut::
DynamicMinCut via MinCutBuilder, one exact solve per call) instead of
MincutBackend::Wrapper (RuVectorGraphAnalyzer, unchanged),

then Direct-backed candidates' compaction wall-clock stays under 100x
baseline's (the gate Wrapper failed by ~18-27x last run),

subject to: bridge-survival gap and Recall@10 delta versus baseline
reproducing the Wrapper backend's own numbers (i.e., the backend swap must
not itself change effectiveness — only speed).
```

## Decision

1. Add `graph_forget::MincutBackend` (`Wrapper` | `Direct`, default
   `Wrapper`) and `MincutGatedForgetting::with_backend()`. Both backends
   build the identical k-NN graph via one shared private helper
   (`build_knn_graph`) so topology and edge weights are byte-identical
   between them — the only difference is which `ruvector-mincut` API
   computes the partition. `Direct`'s disconnected-graph case
   (`min_cut_value() <= 0.0`) is mapped to the same "no signal" empty
   partition `Wrapper` returns for `MinCutResult::Disconnected`, so a
   real behavioral difference (`DynamicMinCut::partition()` can report a
   nontrivial split even when the min cut is 0) is deliberately not
   exercised by this policy — that would be a different, uncontrolled
   experiment.
2. Add three new examples reusing prior nightly probes' exact
   methodology, swapped to the new backend:
   `mincut_direct_backend_bench` (main hypothesis, adds Soft-Direct /
   Hard-Direct rows next to reproduced Wrapper rows),
   `mincut_direct_scaling_probe` (mirrors `mincut_scaling_probe`'s ring
   topology and sizes), `mincut_direct_determinism_probe` (mirrors
   `mincut_determinism_probe`'s fixed 19-vertex bridge topology).
3. Add one exploratory, non-gating probe,
   `mincut_direct_scale_effectiveness_probe`, answering the follow-on
   question this fix makes affordable to ask: does the 0.0pp effectiveness
   gap persist at corpus sizes larger than the performance-forced 84? (It
   does — see Evidence.)
4. **Do not promote `MincutGatedForgetting`.** The performance sub-hypothesis
   is confirmed; the overall ADR-345 hypothesis remains rejected on the
   effectiveness axis, now with materially stronger evidence that the
   effectiveness failure is not a corpus-size artifact.

## Evidence

All numbers from `cargo run --release`, this repository, this session
(raw stdout preserved verbatim in the nightly research README).

**Scaling probe** (`mincut_direct_scaling_probe`, ring k-NN, k=8 — same
methodology as ADR-345's `mincut_scaling_probe`):

| n | Wrapper `partition()` (ADR-345) | Direct build+solve (this run) | Speedup |
|---:|---:|---:|---:|
| 19  | 69,269.9ms | 0.290ms | ~238,000x |
| 50  | 76.8ms | 0.708ms | ~108x |
| 100 | 481.3ms | 1.895ms | ~254x |
| 200 | 2,712.9ms | 5.466ms | ~496x |
| 400 | 11,415.0ms | 18.620ms | ~613x |
| 800 (new) | not measured | 70.5ms | n/a |
| 1600 (new) | not measured | 309.0ms | n/a |
| 3200 (new) | not measured | 1,337.9ms | n/a |

**Determinism probe** (`mincut_direct_determinism_probe`, same fixed
19-vertex two-clique-plus-bridge graph as ADR-345's `mincut_determinism_probe`,
30 trials): 1 distinct min-cut value across all 30 trials (Wrapper measured
15/30 empty/unusable results on the same graph); 30/30 correctly flagged the
bridge as boundary; 0.119ms/call mean (Wrapper: 841ms/call).

**Main benchmark** (`mincut_direct_backend_bench`, the 84-memory corpus,
seed=341 — identical to ADR-345's benchmark):

| Policy | Bridge Surv. | Recall@10 | Compaction | Gap vs baseline | Slowdown |
|---|---:|---:|---:|---:|---:|
| CoherencePolicy (baseline) | 66.7% | 100.0% | 74us | — | — |
| Soft-Wrapper (reproduced) | 66.7% | 100.0% | 117,589us | +0.0pp FAIL | 1,589.0x FAIL |
| Hard-Wrapper (reproduced) | 66.7% | 100.0% | 116,171us | +0.0pp FAIL | 1,569.9x FAIL |
| Soft-Direct (new) | 66.7% | 100.0% | 2,081us | +0.0pp FAIL | **28.1x PASS** |
| Hard-Direct (new) | 66.7% | 100.0% | 2,112us | +0.0pp FAIL | **28.5x PASS** |

Direct vs Wrapper speedup on this exact corpus: Soft 56.5x, Hard 55.0x. The
Wrapper reproduction (66.7% survival, 100.0% recall, ~1,570-1,589x slowdown)
matches ADR-345's originally reported 66.7% survival and "1,800-2,700x"
range closely enough (same order of magnitude, same seed, expected run-to-run
variance already documented in ADR-345) to confirm this implementation did
not silently change anything about the existing Wrapper path.

**This ADR's registered performance sub-hypothesis: CONFIRMED.** Both
Direct-backed candidates clear the 100x slowdown gate (28.1x, 28.5x) that
Wrapper failed by 15-16x. The effectiveness sub-hypothesis ("must reproduce
Wrapper's own numbers") is also confirmed — bridge survival and recall are
bit-identical between Wrapper and Direct at this corpus size, evidence the
two backends compute the same answer, just at very different cost.

**Exploratory follow-up** (`mincut_direct_scale_effectiveness_probe`, not
gating, Soft/Direct only, one seed per size):

| n (memories) | Baseline survival | Soft-Direct survival | Gap | Compaction |
|---:|---:|---:|---:|---:|
| 84   | 66.7% | 66.7% | +0.0pp | 2.4ms |
| 168  | 45.8% | 45.8% | +0.0pp | 7.4ms |
| 336  | 37.5% | 31.2% | **-6.2pp** | 28.3ms |
| 672  | 13.5% | 13.5% | +0.0pp | 45.4ms |
| 1344 | 67.2% | 67.2% | +0.0pp | 170.7ms |

The 0.0pp gap is not a corpus-size artifact of the performance-forced
84-memory corpus: it persists (and briefly reverses) across a 16x range
that Direct now makes cheap enough to test in seconds. This strengthens
ADR-345's own alternative explanation ("the global min-cut isolates an
outlier, not the intended bridge") over the "we never got to test at real
scale" explanation its Open Question 3 left unresolved for synthetic data.

## Consequences

- ADR-345's Open Question 1 is answered: yes, bypassing
  `RuVectorGraphAnalyzer` for `DynamicMinCut` fixes the measured
  performance failure, by 55-613x depending on graph size, with no change
  to computed effectiveness at the sizes tested.
- ADR-345's Open Question 2 (non-determinism) gets a strong, if indirect,
  answer: `DynamicMinCut`'s Stoer-Wagner-style solve is fully deterministic
  on the exact graph that made `RuVectorGraphAnalyzer` non-deterministic
  15/30 times, consistent with the non-determinism being specific to
  `MinCutWrapper`'s bounded-range instance/witness machinery (unchanged,
  not investigated further here — still a `ruvector-mincut` hardening item,
  not something this ADR fixes upstream).
- ADR-345's Open Question 3 gets a partial answer for synthetic data: the
  zero-effect finding generalizes across a 16x corpus-size range, making
  "it's just too small a sample" a materially weaker explanation than
  before. Whether it holds on real (non-Gaussian-cluster) embeddings is
  still open.
- `MincutGatedForgetting` is now performance-viable as a background/offline
  compaction pass (low tens of ms at ~1,300 memories) if a future run finds
  a boundary-detection method that actually correlates with intended
  bridges — the blocking issue is now demonstrably effectiveness, not
  latency.
- No existing behavior changes: `MincutBackend::Wrapper` remains the
  default; all ADR-345 call sites and tests are untouched.

## Alternatives Considered

- **`ClusterHierarchy::boundary_size`** (ADR-345 Open Question 1's other
  named candidate). Not implemented: no method by that name exists on
  `ruvector_mincut::ClusterHierarchy` in this codebase (verified by source
  inspection, not documentation); `DynamicMinCut` was available, simpler,
  and directly comparable via the same `partition()`-shaped API
  `RuVectorGraphAnalyzer` already exposed, so it was used instead.
- **Fixing `MinCutWrapper`'s instance-replay cost directly** (e.g., caching
  edge state across instances instead of full replay). Rejected for this
  ADR's scope: that is a `ruvector-mincut` internal optimization, not a
  `ruvector-agent-memory` integration change, and ADR-345 already
  identified `DynamicMinCut` as the more direct fix to try first.
- **Also fixing the `MinCutWrapper` non-determinism in place.** Rejected
  for this ADR's scope: `Direct`'s determinism is a side effect of using a
  different, already-deterministic algorithm, not a fix to `Wrapper`
  itself; `Wrapper` remains exactly as non-deterministic as ADR-345 found
  it, since it is unchanged.
- **Promoting `MincutGatedForgetting` now that it is fast enough.**
  Rejected: the effectiveness gate is still unmet, and the new
  scale-effectiveness evidence makes "just needs more data" a weaker
  argument for eventual promotion than before, not a stronger one.

## Implementation Plan

Already implemented in this PR:

- `crates/ruvector-agent-memory/src/graph_forget.rs`: `MincutBackend` enum,
  `backend` field (default `Wrapper`), `with_backend()`, `build_knn_graph`,
  `partition_via_wrapper`, `partition_via_direct`, `boundary_from_sides`.
- `crates/ruvector-agent-memory/src/lib.rs`: re-export `MincutBackend`.
- `crates/ruvector-agent-memory/examples/mincut_direct_backend_bench.rs`,
  `mincut_direct_scaling_probe.rs`, `mincut_direct_determinism_probe.rs`,
  `mincut_direct_scale_effectiveness_probe.rs`.
- `crates/ruvector-agent-memory/Cargo.toml`: registers the four new
  examples under the existing `mincut-forget` feature.

No further implementation is planned under this ADR.

## API Shape

```rust
// Behind `mincut-forget` (additive — existing API unchanged):
pub enum MincutBackend {
    Wrapper, // default; unchanged 2026-09-05 behavior
    Direct,  // new: ruvector_mincut::DynamicMinCut via MinCutBuilder
}
impl Default for MincutBackend { /* Wrapper */ }

impl MincutGatedForgetting {
    pub fn with_backend(self, backend: MincutBackend) -> Self;
    // existing soft()/hard()/select_survivors() unchanged
}
```

## Feature Flags

No new feature flag: `MincutBackend` lives behind the existing
`mincut-forget` flag (unchanged, off by default).

## Benchmark Evidence

See "Evidence" above; raw stdout for all four runs is preserved in
`docs/research/nightly/2026-09-17-mincut-direct-backend/README.md`.

## Security

No new cryptographic primitive; no change to `witnessed_compaction`
(untouched by this ADR). `MincutBackend::Direct`'s disconnected-graph
mapping (see Decision item 1) is a deliberate correctness/parity choice,
not a security control.

## Governance

None beyond ADR-345's existing "no witness, no mutation" invariant
(unaffected — this ADR only changes which primitive computes the boundary
signal, not the eviction or witness path).

## Failure Modes

- `Direct`'s Stoer-Wagner-style solve is polynomial (empirically
  worse-than-linear in the scaling probe: n=400->800 roughly 3.8x, 800->1600
  roughly 4.4x, 1600->3200 roughly 4.3x, consistent with a quadratic-ish
  term), so it will eventually become the bottleneck again at large enough
  n — just several orders of magnitude later than `Wrapper`. Not
  characterized past n=3,200 in this run.
- The effectiveness failure (0.0pp gap, occasionally negative) is
  unresolved by this ADR and is the actual blocker to any future
  promotion.

## Migration

None: `MincutBackend::Direct` is additive and opt-in; every existing
`MincutGatedForgetting::soft()`/`hard()` call site keeps `Wrapper` by
default with unchanged behavior.

## Rollback

Remove `MincutBackend`, `with_backend()`, `build_knn_graph`,
`partition_via_wrapper`, `partition_via_direct`, and
`boundary_from_sides`, restoring `boundary_from_one_partition` calling
`RuVectorGraphAnalyzer::from_knn` directly (ADR-345's original code) with
no impact on any caller, since no existing call site uses
`with_backend()`.

## Rejection Criteria

The overall `MincutGatedForgetting` production hypothesis remains rejected
because the bridge-survival gap is still +0.0pp (occasionally negative)
against the required >=15pp threshold, now confirmed across a 16x
corpus-size range. This ADR's own, narrower performance sub-hypothesis is
accepted: Direct clears the previously-failed 100x slowdown gate by a wide
margin (28.1-28.5x measured, vs. a 100x limit).

## Open Questions

1. Does the "global min-cut isolates an outlier, not the intended bridge"
   explanation (ADR-345 Open Question 3) hold on real, non-synthetic agent
   embeddings? Still open; this ADR only strengthens the synthetic-data
   evidence for it.
2. Is there a boundary-detection formulation (e.g., a *local* min-cut
   around each candidate-eviction vertex, rather than one *global* min-cut
   over the whole candidate set) that would correlate better with intended
   bridges than global min-cut does? Not attempted here.
3. `ruvector-mincut`'s `MinCutWrapper` non-determinism (ADR-345 Open
   Question 2) is still unfixed at its source; this ADR only shows that
   routing around it via `DynamicMinCut` avoids the symptom for this
   specific integration.
