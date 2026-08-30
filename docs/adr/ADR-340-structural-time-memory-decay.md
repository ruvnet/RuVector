# ADR-340: Structural-Time Memory Decay — Evaluated, Not Promoted

## Status

**Rejected** (of the pre-registered acceptance threshold). Experimental
crate `crates/ruvector-structural-memory` retained in the workspace as
evidence and as a reusable benchmark harness; not wired into
`ruvector-agent-memory` or any production compaction path.

## Context

`ruvector-agent-memory` (nightly 2026-06-14) scores stored-memory retention
using, among other signals, a wall-clock recency term: a memory's
"freshness" decays as a function of turns/steps elapsed since it was written
or last accessed. `emergent-time` (ADR-251) is a mature, already-merged
crate implementing several internal-time formalisms, including
`StructuralProperTime` — internal time defined as accumulated,
metric-weighted arc length through a system's own state manifold, rather
than external step count. As of this ADR, `StructuralProperTime` had only
been benchmarked on generic anomaly-early-warning and trajectory-compression
tasks (its own module's test suite); it had not been applied to agent memory
retention scoring, despite memory decay being an obvious candidate use case
(the whole point of the formalism is distinguishing "much wall-clock time
passed" from "much actually changed").

This ADR records the outcome of testing that specific application.

## Hypothesis

```text
Given synthetic agent sessions of 4 topic plateaus separated by sharp
switches (plateau lengths 20, 60, or 150 steps; one memory written per
step, embedding dim 32),

when compaction retention score uses StructuralEmbeddingClock's
accumulated context drift (emergent_time::StructuralProperTime, embedding
channel only) instead of WallClock step count as the age signal,

then mean recall@15 of the oracle nearest-neighbour set — averaged over 10
independent seeds — after compacting to a fixed budget of 25 memories
improves by >= 5 percentage points in the long-plateau (150 steps/topic)
configuration, without regressing by more than 2 percentage points in the
short-plateau (20 steps/topic) configuration,

subject to compaction compute time staying within 5x WallClock's, and
causal order being preserved by every clock on every seed.
```

## Decision

**Do not promote structural-time decay to `ruvector-agent-memory` or any
production path at this time.** The mandatory acceptance clause on
long-plateau recall lead (measured +2.00pp against a fixed +5.00pp bar,
averaged over 10 non-cherry-picked seeds) failed. Retain
`crates/ruvector-structural-memory` in the workspace: its benchmark harness
(deterministic multi-seed scenario generator, oracle-recall methodology,
compute-overhead measurement) is directly reusable for a follow-up attempt
with a revised scenario (see Consequences), and the negative result itself
is evidence future nightlies should not blindly rediscover by re-testing the
same configuration.

## Evidence

10-seed mean recall@15 after compaction to a fixed 25-memory budget (full
raw benchmark output in the nightly README):

| plateau_len | WallClock | StructuralEmbedding | StructuralFull | lead (pp) |
|---|---|---|---|---|
| 20 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 |
| 60 | 0.3867 ± 0.1147 | 0.4000 ± 0.1075 | 0.4133 ± 0.1024 | 1.33 |
| 150 | 0.1600 ± 0.0442 | 0.1800 ± 0.0600 | 0.1800 ± 0.0600 | 2.00 |

Compute overhead: `StructuralEmbedding` compaction took 1.30x `WallClock`'s
wall-clock time, summed across all three plateau-length configurations
(well under the 5x acceptance bound).

Per-seed detail at the deciding (150) configuration: 3/10 seeds show
`StructuralEmbedding` beating `WallClock` by exactly +6.67pp; 7/10 tie
exactly; 0/10 show a regression. See the nightly README's
[Why the Effect Is Real But Small](../research/nightly/2026-08-24-structural-time-memory-decay/README.md#why-the-effect-is-real-but-small)
section for the mechanism this pattern is consistent with.

Acceptance clauses:

| Clause | Threshold | Measured | Result |
|---|---|---|---|
| (a) long-plateau mean lead | ≥ 5.00pp | 2.00pp | **FAIL** |
| (b) short-plateau mean regression | ≥ -2.00pp | 0.00pp | PASS |
| (c) compute overhead ratio | ≤ 5.00x | 1.30x | PASS |
| (d) causal order preserved | all cells | all cells | PASS |

Reproducible via `cargo run --release -p ruvector-structural-memory --bin
benchmark`; deterministic given the fixed seed-generation formula
(`0xC0FFEE + i * 0x9E3779B9`).

## Consequences

**What this ADR does NOT claim:** that `StructuralProperTime` is unsuitable
for agent memory decay in general. The measured effect is directionally
consistent (never a regression across 30 seed×plateau_len cells) but too
small and too seed-dependent, at this specific scenario configuration, to
clear the bar set before benchmarking. Two concrete, unexplored variables
could change that: (1) the ratio of within-plateau embedding noise to
topic-switch jump size was fixed at one value (≈1:700 over the longest
plateau) rather than swept — a real embedding source might sit at a
materially different point on that ratio; (2) `StructuralFull`'s entropy
channel added no benefit here, but its `ΔC`/`ΔG` channels were left
unweighted for lack of an honest signal source in this synthetic harness —
a real coherence signal (e.g. from `ruvector-coherence`) is untested.

**What does NOT become stable API:** nothing. No public interface in
`ruvector-agent-memory` changes. `ruvector-structural-memory`'s own types
(`ScenarioConfig`, `MemoryItem`, `Session`, `CompactionWeights`) are
research-tier and may change freely in a follow-up.

**What remains experimental:** the entire `ruvector-structural-memory`
crate, including its benchmark methodology, which is the actual reusable
asset from this nightly.

## Alternatives Considered

1. **Fractional compaction budget** (30% of corpus size) — tried first;
   made the experiment trivial (recall@15 = 1.0000 for every clock,
   uninformative) because the budget always comfortably contained an entire
   topic's memory pool. Rejected as a methodology bug, not as a result.
2. **Single-seed evaluation** — the first fixed-budget run (seed
   `0xC0FFEE`) showed a comfortable +6.67pp lead that would, reported alone,
   have read as an unambiguous ACCEPT. Rejected per this repository's
   explicit prohibition on cherry-picked seeds; replaced with the 10-seed
   mean gating this ADR's decision.
3. **Lowering the acceptance threshold post-hoc** to convert the 2.00pp
   measured lead into a pass — not done. The 5pp bar was fixed alongside the
   budget and noise parameters before the first benchmark run and is
   reported as failed, per this repository's rule against weakening
   acceptance criteria to force a pass.

## Implementation Plan

None at this time — no production change is being made. A follow-up
implementation plan, contingent on a future ACCEPT, would extend
`ruvector-agent-memory`'s existing scoring path with a `Clock` type
parameter defaulting to today's wall-clock behavior (additive, not
breaking).

## API Shape

N/A — `ruvector-structural-memory` is a standalone research crate with no
production consumer. Its public surface (`clocks`, `scenario`, `compaction`
modules) is documented in `src/lib.rs` and may change without a deprecation
cycle.

## Feature Flags

None. No production crate depends on this one.

## Benchmark Evidence

See [Evidence](#evidence) above and the full raw output in
`docs/research/nightly/2026-08-24-structural-time-memory-decay/README.md`.

## Security

No new attack surface introduced (synthetic, in-process, no I/O). The
security-relevant risk for a *future* production integration — an agent or
injected tool result that keeps its reported context embedding artificially
static to make a structural clock under-forget stale/compromised memories —
is identified but out of scope for this synthetic benchmark; flagged as a
required precondition for any future promotion.

## Governance

REJECT outcomes are retained, not deleted, per this repository's nightly
research process: the crate, its tests, and this ADR stand as the record
that this specific configuration was tried and did not clear its
pre-registered bar, so a future nightly does not have to rediscover that
independently.

## Failure Modes

See the nightly README's
[Failure Modes](../research/nightly/2026-08-24-structural-time-memory-decay/README.md#failure-modes-and-things-that-almost-made-this-look-better-than-it-is)
section for the three methodology issues found and fixed during this
nightly (trivial fractional budget, noise-scale collapse, single-seed
cherry-picking risk) before the final result was gated.

## Migration

N/A — no production code changes.

## Rollback

N/A — nothing was promoted to roll back. Reverting this ADR means removing
`crates/ruvector-structural-memory` from the workspace; not recommended, as
the benchmark harness is reusable evidence for the follow-up work listed
below.

## Rejection Criteria

Already applied: this ADR's own hypothesis was rejected by its own
pre-registered clause (a). Documented here rather than silently discarded,
per this repository's rule that a falsified hypothesis with good evidence is
a valid nightly outcome.

## Open Questions

1. Does a real (non-synthetic) embedding source sit at a noise-to-drift
   ratio where the effect reliably clears 5pp, or is 2pp closer to the
   mechanism's true ceiling?
2. Would wiring a genuine coherence signal (`ruvector-coherence`) into
   `StructuralFull`'s `ΔC` channel — untested here — change the exploratory
   comparison's null result?
3. Is a fixed-budget compaction regime (used here to force within-topic
   competition) representative of how `ruvector-agent-memory` actually
   triggers compaction in production, or does that differ enough to warrant
   a different benchmark design entirely?
