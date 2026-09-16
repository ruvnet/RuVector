# Why a Monotone Clock Makes a Bad Ranking Score (and a Good Sampler)

## Problem

Long-running AI agents accumulate memories faster than they accumulate
insight. Most of what an agent "remembers" during a real task is redundant:
retried tool calls, repeated confirmations, near-identical re-observations
of a slow-changing environment. A memory-compaction policy that scores
recency by counting *events* — every insertion or access ticks a clock by
exactly one, regardless of what happened — cannot tell a burst of redundant
churn apart from a burst of genuinely new information. Both inflate the tick
count identically, and identically make everything inserted before the burst
look stale.

`ruvector-agent-memory`'s existing `CoherencePolicy` scores recency exactly
this way. It works well in general (that's why it's the crate's headline
result), but it inherits this specific blind spot from its tick-based clock.

## Hypothesis

`emergent-time`, a separate crate in the same monorepo, implements
**Structural Proper Time**: instead of counting events, it measures the arc
length a system's state traces through its own state manifold — embedding
movement, entropy change, coherence loss, and so on. A quiet period (nothing
really changing) contributes almost no structural time; a genuine regime
shift contributes a lot. This is exactly the property agent-memory recency
is missing. The natural move: wire `ruvector-agent-memory`'s recency term to
`emergent-time`'s clock instead of the raw tick count.

## What Actually Happened

The first, most obvious way to do this — turn cumulative structural time
into a `[0,1]` score and rank memories by it, exactly like the existing
tick-count recency term — **does not work**. Measured on a synthetic 360-memory
corpus (40 topic shifts, each followed by 8 near-duplicate "churn" memories),
this candidate matched the tick-based baseline exactly: 10.0% survival for
both, when the goal was distinguishing genuine topic shifts from churn under
an aggressive compaction budget.

The reason is structural, not a tuning problem: cumulative structural time is
*monotone non-decreasing* in insertion order — exactly the same
order-preserving property the raw tick count has. Ranking by any strictly
monotone reparametrization of a fixed order picks the same top-K set, except
at ties — and ties resolve toward the later index, which is always a churn
duplicate (it was inserted after the memory it copies). A fancier clock
turned into the same kind of score just reproduces the same ranking.

`emergent-time` already ships the right primitive for this problem, though —
it's just not a ranking function. `structural_clock::keyframes(clock, traj,
budget)` samples `budget` positions evenly spaced *along accumulated
structural time*, built originally for compressing a trajectory to a fixed
number of representative samples. Applied to memory retention instead of
compression: because a churn run contributes ≈0 structural time, the
sampling algorithm's "next target level" lands on or near the regime-shift
memory that precedes the churn, not inside it. This is a different
mechanism — nearest-sample-to-a-budget, not top-K-by-score — and it
measures a real effect: **35.0% survival, +25 percentage points over the
tick-recency baseline**, and +22.5pp over a fair, dependency-free "only tick
on a real change" competitor built specifically to rule out a strawman
comparison.

## Technical Design

Two new `CompactionPolicy` implementations in
`ruvector-agent-memory::structural_recency` (feature `structural-time`,
optional dependency on `emergent-time`):

- `StructuralTimeRecency` — the score-based approach. Kept in the tree,
  tested, documented as a negative result. Deleting it would just mean some
  future engineer rediscovers the same dead end.
- `StructuralKeyframeRetention` — the promoted mechanism, using
  `emergent_time::structural_clock::keyframes` directly, with a
  frequency/coherence top-up for any shortfall against the target budget.

The structural clock is configured with only the two channels this crate can
honestly compute — embedding movement and coherence loss against the active
context window — leaving entropy, graph, and prediction-error at zero rather
than inventing signals for them.

## Actual Implementation

- `crates/ruvector-agent-memory/src/structural_recency.rs` — both policies,
  the fair cheap `DedupGatedRecency` baseline, and unit tests including a
  small exact-duplicate-churn corpus that isolates the tie-breaking mechanism
  deterministically.
- `crates/ruvector-agent-memory/examples/structural_time_recency_bench.rs` —
  the full benchmark: two experiments (aggressive recency-only ablation;
  production-weights 50%-compaction Recall@10 check), reporting all four
  policies side by side with pre-declared PASS/FAIL gates.

## Actual Benchmark Evidence

360-memory synthetic corpus (40 regime-shift segments × 9 memories each),
`cargo run --release`, 3 independent process runs cross-checked for
determinism:

| Policy | Survival rate (Exp. 1) | Recall@10 (Exp. 2) |
|---|---|---|
| CoherenceWeighted (baseline) | 10.0% | 100.0% |
| DedupGatedRecency (fair baseline) | 12.5% | 100.0% |
| StructuralTimeRecency (score-based) | 10.0% | — |
| **StructuralKeyframeRetention** | **35.0%** | **99.5%** |

All pre-declared acceptance gates passed for `StructuralKeyframeRetention`
(survival gap ≥ +15pp: measured +25.0pp; Recall@10 regression ≤ 3pp: measured
−0.5pp; compaction slowdown ≤ 20x: measured 2.7–3.3x; determinism: identical
across 3 process runs). Overall result: **ACCEPT**.

## Limitations

Synthetic corpus, not a real agent trace. Only 2 of the structural metric's 5
channels are used. A degenerate exact-duplicate-tail corner case in the
budget-trimming logic is documented and unit-tested but not fully resolved.
Not benchmarked head-to-head against the prior nightly's mincut-gated
forgetting approach (a related but distinct mechanism for a different
failure mode).

## Production Relevance

`StructuralKeyframeRetention` is an opt-in Cargo feature, not a default-path
change — it does not alter `CoherencePolicy`'s existing behavior. It is a
genuine 2.7–3.3x-overhead alternative (not the 1,800x+ seen in the prior
mincut-gated-forgetting nightly) that a production agent-memory deployment
with known bursty/redundant activity patterns could adopt today, behind the
feature flag, with the caveat that it has only been validated on a
synthetic stress-test corpus so far.

## RuVector Ecosystem Implications

This is the first production-code-adjacent use of `emergent-time` outside
its own crate anywhere in the monorepo, connecting agent memory, the
calculus-of-emergent-time library, and (via the crate's existing
`witnessed_compaction` module) the witness/provenance layer. It also leaves
behind a specific, reusable engineering lesson for the rest of the
ecosystem: when reaching for an internal/structural clock to solve a
retention or eviction problem, reach for budget-constrained *sampling*
along the clock, not a *rank score* derived from it — the latter is a
structural dead end whenever the underlying clock (like most internal-time
constructions) is monotone in the same order it's meant to improve on.

## Future Direction

Validate on a real captured agent trace; extend the structural metric with a
real graph channel sourced from the prior nightly's mincut/bridge-detection
work; wire the witnessed-compaction path through end-to-end toward an RVF
portable-memory-snapshot integration.

## References

- `crates/emergent-time/src/structural_clock.rs`
- `crates/ruvector-agent-memory/src/compaction.rs`
- `docs/adr/ADR-346-structural-time-keyframe-agent-memory-retention.md`
- `docs/adr/ADR-345-mincut-gated-forgetting.md` (prior related nightly)
