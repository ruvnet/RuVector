# ADR-346: Structural-Time-Gated Memory Compaction Scheduling

## Status

**Rejected** (for production use as designed) — measured evidence retained.
Experimental module (`ruvector-agent-memory::structural_gate`, `structural-gate`
feature), not wired into any production write path. See
[nightly research report](../research/nightly/2026-09-18-structural-time-gated-memory-compaction/README.md)
for full evidence.

## Context

`ruvector-agent-memory` has three `CompactionPolicy` implementations that
answer *what* survives a compaction pass, but nothing in the crate has ever
answered *when* a compaction pass should run. Every existing benchmark
(`src/main.rs`, `examples/mincut_gated_forgetting_bench.rs`) invokes
`compact()` exactly once, on demand, against a store that never receives
further writes. A long-running agent has no such luxury: memories arrive
continuously, and *something* has to decide how often to pay the cost of a
compaction pass.

Two obvious real-world defaults exist and are used elsewhere in the
ecosystem: a fixed wall-clock/write-count cadence, or a fixed capacity
ceiling. Neither looks at what the writes actually contain. `emergent-time`
(v2.2.4), an existing, independently-developed and independently-tested
workspace crate, ships `StructuralProperTime` — an endogenous clock whose
tick is the metric-weighted arc length a system's state traces through its
own state manifold (embedding drift + coherence loss + entropy change +
graph change + prediction-error change), used elsewhere in that crate for
early-warning and history-compression tasks. This ADR asks whether reusing
that clock, unmodified, as a compaction *trigger* signal reduces the
operational cost of keeping a growing memory store near its target size,
relative to the two naive defaults.

## Hypothesis

```text
Given a deterministic 1,400-write agent-memory stream alternating four
300-write "quiet" epochs (near-duplicate writes into the most recently
introduced topic cluster, embedding noise sigma=0.02) with four 50-write
"burst" epochs (writes into a brand-new, never-before-seen topic cluster),
compacted to target_size=200 by the same CoherencePolicy at every trigger
fire,

when compaction is scheduled by StructuralGateTrigger (fires once
accumulated StructuralProperTime since the last compaction, computed over a
24-write sliding-window summary, crosses a threshold calibrated as 20x the
mean per-write tick observed on an independent 50-write quiet baseline
slice) instead of FixedIntervalTrigger(50) (fires every 50 writes,
unconditionally),

then StructuralGateTrigger's compaction-call count and time-integrated
excess-store-size (writes-since-last-compaction summed while store.len() >
target_size) should both be at least 20% lower than FixedIntervalTrigger's,

subject to: final Recall@10 on hot-cluster (most-recent-topic) queries
staying within 2 percentage points of FixedIntervalTrigger's, and
wall-clock trigger+compaction overhead staying within 2x of
FixedIntervalTrigger's (StructuralGateTrigger does O(window*dims) work per
write vs. FixedIntervalTrigger's O(1)).
```

## Decision

**Do not promote `StructuralGateTrigger` to production use with the
calibration procedure tested.** The signal is real and measurably
discriminates regime — 70 of its 100 fires landed on the 200 "burst"
writes (14.3% of the stream), a ~14x higher fire-density than during the
1,200 "quiet" writes — but the pre-registered calibration (20x the mean
quiet-baseline tick) was not conservative enough: quiet-regime noise alone
crosses it roughly once every 40 writes, so total call count (100) came in
4.2x *higher* than the fixed-interval baseline (24), the opposite of the
primary hypothesis. Retain the trigger machinery (`CompactionTrigger`
trait, `FixedIntervalTrigger`, `CapacityTrigger`, `StructuralGateTrigger`)
as a tested, honest negative result and a base for recalibration, but it is
not wired into any default path.

## Evidence

Single deterministic benchmark (`examples/structural_gated_compaction_bench.rs`,
`cargo run --release -p ruvector-agent-memory --features structural-gate
--example structural_gated_compaction_bench`), 3 repeated release-mode runs
(algorithmic outputs are exactly reproducible across runs given the fixed
seed; only wall-clock varied):

| Trigger | Compactions | Excess-size integral | Recall@10 | Wall (ms, 3-run range) | Fires@quiet / Fires@burst |
|---|---:|---:|---:|---:|---:|
| FixedInterval(50) | 24 | 30,551 | 1.0000 | 6.2–6.3 | 20 / 4 |
| Capacity(400) | 5 | 120,615 | 1.0000 | 2.4–2.5 | 4 / 1 |
| StructuralGate | 100 | 18,913 | 1.0000 | 22.2–23.5 | 30 / 70 |

Acceptance evaluation vs. FixedInterval (thresholds fixed before the run):

| Criterion | Result | Threshold | Pass? |
|---|---:|---:|:---:|
| Compaction-call reduction | −316.7% (4.2x *more* calls) | ≥ 20% reduction | **No** |
| Excess-size-integral reduction | 38.1% | ≥ 20% reduction | Yes |
| Recall@10 gap | 0.0000pp | ≤ 2pp | Yes |
| Wall-clock ratio (struct/fixed) | 3.6–3.9x | ≤ 2.0x | **No** |

Two of four mandatory criteria failed → **REJECT** per the pre-registered
rule (all four must pass).

**Root cause.** `StructuralProperTime`'s coherence channel only accumulates
on *loss* (`(prev.coherence - cur.coherence).max(0.0)`), by original design
for irreversible-drift semantics. A 24-write sliding-window coherence
estimate fluctuates from sampling noise alone even under a stationary
source; loss-only accumulation means every downward fluctuation adds
internal time and no upward fluctuation ever cancels it, so quiet-regime
noise reads as monotone drift. A threshold calibrated purely from a quiet
baseline's *mean* tick is still crossed by that same baseline's own
variance roughly every 40 writes — well under FixedInterval's 50-write
cadence. The diagnostic fire-location split (70/100 fires on the 14.3% of
writes that are genuine bursts) shows the signal is not noise-only —
regime does drive a real, large excess in fire density — but the specific
calibration tested does not suppress quiet-regime false-positives enough
to win on absolute call count.

## Consequences

- No change to any existing `ruvector-agent-memory` compaction path;
  `structural_gate` is additive and feature-gated (`structural-gate`,
  off by default).
- A genuine, previously-latent correctness bug was found and fixed as part
  of this work: `MemoryStore::insert` assigned `id = entries.len()`,
  which silently collides with a surviving entry's id once compaction and
  further insertion interleave (every prior benchmark only ever compacted
  once, terminally, so the bug was unreachable). Fixed to a monotonic
  `next_id` counter. This is a pure correctness fix, independent of which
  trigger is used, verified by the full existing `ruvector-agent-memory`
  test suite (all features) passing unchanged.
- The `CompactionTrigger` trait and its three implementations are retained
  as tested code and as the base for recalibration, not deleted — per the
  nightly process's rule that a rejected candidate stays in the lineage so
  a future run does not rediscover the same dead end blindly.

## Alternatives

- **Symmetric structural metric** (accumulate on coherence *change*, not
  just *loss*): would reduce the noise-driven bias but changes
  `StructuralProperTime`'s existing, already-tested semantics used
  elsewhere in `emergent-time`; out of scope for a *reuse-as-is* experiment
  and left as a candidate for a follow-up that forks the metric instead of
  reusing the shared one.
- **Larger window / burst-aware calibration** (calibrate the threshold from
  a mix of quiet *and* one representative burst, not quiet alone): the
  most promising untested direction — see Next Research.
- **EMA-smoothed centroid** instead of a hard sliding window: would reduce
  per-write jitter at the cost of a decay-rate hyperparameter; not tested
  here to keep the trigger a direct, unmodified reuse of
  `emergent-time::structural_clock`.

## Implementation Plan

Not applicable — rejected for production. The module remains as
`#[cfg(feature = "structural-gate")]`, opt-in, with no default-feature or
production call site depending on it.

## API Shape

```rust
pub trait CompactionTrigger {
    fn name(&self) -> &str;
    fn on_write(&mut self, entries: &[MemoryEntry]) -> bool;
    fn on_compacted(&mut self, entries: &[MemoryEntry]);
}

pub struct FixedIntervalTrigger { /* .. */ }
pub struct CapacityTrigger { /* .. */ }
pub struct StructuralGateTrigger { /* .. */ }
```

## Feature Flags

`structural-gate` (off by default) — pulls in `emergent-time` as a path
dependency. No effect on any other feature or the crate's default build.

## Benchmark Evidence

`crates/ruvector-agent-memory/examples/structural_gated_compaction_bench.rs`;
raw run captured in the nightly research report (README.md, §Benchmark
Results). Deterministic (fixed seed `0x5EED_C0DE`); algorithmic outputs
(call counts, excess-size integral, recall) are bit-identical across 3
repeated runs, wall-clock varied 6.2–6.3ms (fixed) vs 22.2–23.5ms
(structural).

## Security

No new attack surface: the trigger only reads `MemoryEntry` vectors already
in the store and a scalar threshold; no untrusted input parsing, no new
serialization format, no witness/signature involvement. `structural-gate`
is an additive, opt-in Cargo feature with no default-path exposure.

## Governance

No autonomous promotion occurred; this ADR documents a rejection with
retained evidence, as the nightly process requires for a falsified
hypothesis. Re-running the same experiment with the same acceptance
thresholds against a materially different calibration procedure (see Next
Research) would be a new, independently evaluated hypothesis, not a
retroactive change to this one.

## Failure Modes

Documented above (root cause). Additionally: the `StructuralGateTrigger`'s
`O(window * dims)` per-write cost (measured at 3.6–3.9x
`FixedIntervalTrigger`'s wall time at window=24, dims=32) would scale
linearly with `window` and `dims`; at much higher dims (e.g. 1536 for a
typical embedding model) or larger windows this ratio would grow further
without algorithmic changes (e.g. incremental centroid/entropy updates
instead of full window rescans).

## Migration

None; nothing existing changes behavior except the `MemoryStore::insert`
id-collision fix (see Consequences), which is a strict correctness
improvement with no observable behavior change for any existing
single-terminal-compaction call site.

## Rollback

Not applicable — feature is opt-in and not used by any default path. To
remove: delete `structural_gate.rs`, the `structural-gate` feature and
`emergent-time` dependency from `Cargo.toml`, and the corresponding
example. The `MemoryStore::insert` id fix should be kept regardless (it is
an independent correctness fix, not part of the rejected candidate).

## Rejection Criteria

Applied as pre-registered: all four acceptance criteria (call-count
reduction, excess-size reduction, recall gap, wall-clock ratio) must pass;
two failed (call-count reduction, wall-clock ratio) → REJECT. No threshold
was weakened after seeing results.

## Open Questions

1. Does a threshold calibrated from *both* a quiet slice and a
   representative burst slice (rather than quiet alone) bring the call
   count under FixedInterval's while preserving the burst-vs-quiet fire
   discrimination already observed (70/100 fires on 14.3% of writes)?
2. Does replacing the window rescan with incremental centroid/entropy
   updates close enough of the wall-clock gap to matter, independent of
   the calibration question?
3. Is the coherence-loss-only asymmetry the dominant noise source, or does
   the entropy channel's 8-bin histogram contribute comparably? (Not
   isolated in this run — see Next Research.)
