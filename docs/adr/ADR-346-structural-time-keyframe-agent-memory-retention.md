# ADR-346: Structural-Time Keyframe Retention for Agent Memory Compaction

## Status

Accepted (experimental, feature-gated). New module
`ruvector-agent-memory::structural_recency`, feature `structural-time`
(optional path dependency on `emergent-time`), off by default. Includes one
promoted mechanism (`StructuralKeyframeRetention`) and one retained, documented
negative result (`StructuralTimeRecency`).

## Context

`ruvector-agent-memory::compaction::CoherencePolicy` (2026-06-14 nightly)
scores each memory's `recency` term from `MemoryEntry::last_accessed_at` — a
**logical clock** incremented by exactly one tick per insertion or access,
regardless of what happened. This is, in effect, `emergent-time`'s own
`WallClock`/`AgentWallClock`: `Clock::tick` returns `1.0` unconditionally,
which that crate's own benchmarks (`structural_clock.rs`,
`agentic_time.rs`) treat as the null hypothesis its Structural Proper Time
clock is built to beat, because a uniform-tick clock cannot distinguish a
burst of redundant, low-information events (retries, duplicate
re-observations) from one genuinely new, information-bearing event — both
advance the clock identically.

`emergent-time` (`crates/emergent-time`) already implements this alternative
clock — `StructuralProperTime`, the arc length of a system's trajectory
through embedding + entropy + graph + coherence + prediction-error space —
and ships it, in the same file, with the primitive built specifically for
budget-constrained retention: `structural_clock::keyframes(clock, traj,
budget)`, which samples `budget` indices spaced evenly in *structural* time
rather than by rank. Neither crate used the other. This ADR's question: does
wiring `ruvector-agent-memory`'s compaction to `emergent-time`'s clock give it
a recency signal that resists churn inflation, and if so, through which of
`emergent-time`'s two mechanisms (rank score vs. keyframe sampling)?

## Hypothesis

```text
Given a synthetic agent memory stream of 40 "regime-shift" events (fresh,
mutually unrelated 32-dim topic vectors) each immediately followed by 8
near-duplicate "churn" memories (small perturbations of the same vector —
redundant retries/re-observations that advance the logical clock without
carrying new information), a 360-entry store,

when the store is aggressively compacted to exactly 40 entries (recency-only
weights, no context window) using CoherencePolicy (baseline, tick recency),
DedupGatedRecency (fair cheap competitor: ticks suppressed on near-duplicate
transitions, no emergent-time dependency), StructuralTimeRecency (candidate
B1: cumulative structural time turned into a `[0,1]` rank score), and
StructuralKeyframeRetention (candidate B2: `emergent_time::structural_clock
::keyframes` used directly as the retention sample),

then candidate B2's regime-shift-memory survival rate should exceed
baseline's by at least 15 percentage points,

subject to: (a) a 50%-compaction, production-weights, 5-vector-context
variant shows no more than a 3pp Recall@10 regression for any candidate; (b)
every candidate's compaction wall-clock stays within 20x baseline's; (c) B2's
survivor selection is deterministic across repeated runs with the same seed.
```

Full methodology and raw output are in
`docs/research/nightly/2026-09-06-structural-time-agent-memory/README.md`.

## Decision

1. Add `ruvector-agent-memory::structural_recency` with:
   - `DedupGatedRecency` / `dedup_gated_recency` (always compiled, no new
     dependency): the fair cheap competitor.
   - `StructuralTimeRecency` / `structural_recency` (feature `structural-time`):
     the score-based integration. **Retained as a documented negative
     result**, not deleted — see "Evidence" below for why it does not work.
   - `StructuralKeyframeRetention` (feature `structural-time`): the promoted
     mechanism, using `emergent_time::structural_clock::keyframes` for
     budget-constrained sampling, with a frequency+coherence top-up for any
     shortfall against `target_size`.
2. Feature-gate the `emergent-time` dependency behind `structural-time`,
   off by default, mirroring `mincut-forget`'s pattern (ADR-345).
3. **Do not** replace `CoherencePolicy`'s default recency term. This ADR adds
   `StructuralKeyframeRetention` as an available, opt-in `CompactionPolicy`,
   not a default-path change — the measured benefit is corpus-shape-dependent
   (see "Limitations").

## Evidence

Benchmark: `cargo run --release -p ruvector-agent-memory --example
structural_time_recency_bench --features structural-time` (360-memory corpus,
40 segments × (1 regime-shift + 8 churn), 32-dim, seed fixed, 3 independent
process runs cross-checked for determinism).

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| B2 survival-rate gap vs. tick-recency baseline | ≥ +15pp | +25.0pp (35.0% vs 10.0%) | PASS |
| B2 survival-rate gap vs. fair dedup baseline | (reported, not gating) | +22.5pp (35.0% vs 12.5%) | edge over fair baseline |
| B2 Recall@10 delta (50% compaction, production weights) | ≥ −3pp | −0.5pp (99.5% vs 100.0%) | PASS |
| B2 compaction slowdown vs. baseline | ≤ 20x | 2.7x – 3.3x (Exp. 1 and 2) | PASS |
| B2 determinism (3 independent process runs) | identical | identical | PASS |
| B1 (score-based) survival-rate gap vs. baseline | — | +0.0pp | negative result, informative |

**Overall acceptance: ACCEPT.**

### Why B1 (score-based) fails and B2 (keyframe-based) works

Cumulative structural time is, by construction, monotone non-decreasing in
insertion order — exactly like the raw tick count it was meant to replace.
Ranking entries by any strictly monotone reparametrization of the same order
produces (up to ties) the *same* top-K selection; and ties resolve toward the
later index, which is always a churn duplicate (churn is inserted after the
memory it duplicates). B1 therefore measured statistically indistinguishable
from the baseline (+0.0pp) — a real, reproducible negative result, not a bug,
documented in the module and left in the tree as working, tested code per the
nightly process's evidence-retention rule.

`emergent-time` itself defines the correct primitive for retention under a
fixed budget: `keyframes()` samples indices *evenly spaced in accumulated
structural time*, not by rank. Because a churn run contributes ≈0 structural
time, few or no keyframe samples land inside it — the mechanism structurally
prefers the regime-shift memory that precedes the run over any of its
duplicates, which no monotone rank score can express. Unit tests
(`structural_recency::structural_tests::keyframe_retention_prefers_regime_shift_over_churn`
et al.) verify this directly with an exact-duplicate corpus where the
tie-break mechanics are unambiguous.

## Consequences

- `ruvector-agent-memory` gains an opt-in compaction policy that is
  materially more churn-resistant than tick-based recency on a corpus shaped
  like bursty, redundant agent activity (a realistic pattern: retries, tool
  re-calls, repeated confirmations).
- The crate now has a second real (not toy) consumer of `emergent-time`,
  connecting agent memory, graph/vector, and the calculus-of-emergent-time
  crates in the ecosystem map.
- The negative result on B1 is a reusable lesson for future nightly work:
  turning a monotone internal clock into a *rank score* for top-K retention
  is a dead end in general; budget-constrained *sampling* along the clock is
  the correct integration pattern for this class of problem.

## Alternatives

- **Mincut-gated forgetting** (ADR-345): a different structural signal
  (graph bridges) for the same "protect what matters" problem; rejected for
  cost (1,800x–2,700x compaction slowdown at even 84 entries). This ADR's
  mechanism is 20–1,000x cheaper (2.7–3.3x baseline at 360 entries, O(n log n)
  dominated by the sort, no graph min-cut computation) and solves a
  different failure mode (churn-inflated recency, not bridge-memory loss).
- **Entropy/graph/prediction-error channels for the structural metric**:
  `StructuralMetric` supports 5 channels; this integration uses only 2
  (embedding movement, coherence loss) because `ruvector-agent-memory` has no
  honest observable for the other 3 today. Fabricating one was rejected as
  dishonest per the nightly process's constraints.

## Implementation Plan

Implemented in this nightly run:
`crates/ruvector-agent-memory/src/structural_recency.rs`,
`crates/ruvector-agent-memory/examples/structural_time_recency_bench.rs`,
feature `structural-time` in `Cargo.toml`. No changes to default-path code.

## API Shape

```rust
pub struct DedupGatedRecency { pub weights: DedupGatedWeights }
pub fn dedup_gated_recency(entries: &[MemoryEntry], dedup_threshold: f32) -> Vec<f32>;

#[cfg(feature = "structural-time")]
pub struct StructuralTimeRecency { pub weights: StructuralTimeWeights } // negative result, kept
#[cfg(feature = "structural-time")]
pub struct StructuralKeyframeRetention { pub base: CoherenceWeights, pub metric: StructuralMetric }
```

Both implement `CompactionPolicy` and slot into the existing
`ruvector_agent_memory::compact()` entry point unchanged.

## Feature Flags

`structural-time` (optional `dep:emergent-time`), off by default — additive,
no change to any existing default-feature behavior or public API.

## Benchmark Evidence

See "Evidence" above and the full run transcript in
`docs/research/nightly/2026-09-06-structural-time-agent-memory/README.md`.

## Security

No new attack surface: pure in-memory scoring over caller-provided vectors,
no I/O, no new serialization format, no witness/ledger interaction. Does not
touch `ledger.rs`, `ops.rs`, or the proof-gate path.

## Governance

Opt-in via Cargo feature; no default-path behavior change; no promotion to
`CoherencePolicy`'s default recency term. A future promotion to default would
require a corpus-representative benchmark (this ADR's synthetic churn/regime
corpus is a stress test, not a claim of representativeness — see
"Limitations").

## Failure Modes

- **Non-churny corpora**: if memory insertion has no redundant-duplicate
  structure, `StructuralKeyframeRetention` degrades toward
  `keyframes()`'s even-spacing-by-position behavior, i.e. similar to a
  stratified-sample-by-recency baseline — not measured to regress recall in
  Experiment 2 (production weights, mixed corpus), but not validated on a
  churn-free corpus either.
- **Degenerate exact-duplicate tails**: documented in the module's
  `select_survivors` comments — a trailing run of *exactly* identical
  entries can, in a corner case, cause the endpoint-forcing rule in
  `keyframes()` to add a redundant frame that the trim step then removes
  from the wrong end. Not observed on the (noisy, non-exact) benchmark
  corpus; covered by a dedicated unit test using an exact-duplicate corpus
  small enough to exhibit the mechanism deterministically.

## Migration

None — additive, feature-gated, opt-in policy. No default-path migration.

## Rollback

Remove the `structural-time` feature and its dependency; no other code
depends on it.

## Rejection Criteria (met, for B1 only)

`StructuralTimeRecency` (B1) was pre-registered as the primary candidate and
falsified: it does not exceed the tick-recency baseline (+0.0pp against a
+15pp threshold). It is retained as evidence and as the reference
implementation that motivated B2, not deleted.

## Open Questions

- Does `StructuralKeyframeRetention`'s advantage hold on a real (not
  synthetic) agent trace with organic churn structure (tool-call retries,
  repeated tool outputs) rather than constructed near-duplicates?
- Would extending the structural metric with a real entropy/graph channel
  (e.g. from `ruvector-mincut`'s bridge signal, ADR-345) change the
  survival-rate/performance tradeoff, given ADR-345's cost findings?
- Is there a principled way to fix the exact-duplicate-tail trimming corner
  case beyond documenting it?
