# When should an agent compact its memory? We tried an endogenous clock instead of a timer, and it lost — instructively.

## Problem

Every agent-memory system that compacts/evicts old entries answers "what
survives?" with some importance score. Almost none of them ask "when
should this run at all?" — the answer is usually a cron tick, a request
counter, or a capacity ceiling, none of which know or care what the agent
has actually been writing. A quiet agent and a bursty agent get compacted
on the same schedule.

RuVector's `ruvector-agent-memory` crate already had three ways to decide
*what* to keep (LRU, LFU, and a coherence-weighted policy). It had zero
ways to decide *when* to run compaction — every existing benchmark
compacts a store exactly once, on demand, and never writes to it again.

## Hypothesis

The same workspace ships an unrelated, already-tested crate,
`emergent-time`, built around one idea: **time is the arc length a
system's state traces through its own state manifold**, not a background
tick. It's used elsewhere for early-warning detection (structural drift
precedes thermodynamic drift) and history compression (sample where things
change, skip where they don't).

We asked: can that clock, reused unmodified, decide *when* to compact
agent memory — staying quiet through near-duplicate writes and firing
promptly on a burst of genuinely new material — better than a naive
fixed-interval timer?

## Technical Design

Three interchangeable `CompactionTrigger` implementations, all answering
"should compaction run now?":

```rust
pub trait CompactionTrigger {
    fn name(&self) -> &str;
    fn on_write(&mut self, entries: &[MemoryEntry]) -> bool;
    fn on_compacted(&mut self, entries: &[MemoryEntry]);
}
```

- `FixedIntervalTrigger` — every 50 writes, `O(1)`.
- `CapacityTrigger` — once the store exceeds 2x target size, `O(1)`.
- `StructuralGateTrigger` — accumulates
  `emergent_time::structural_clock::StructuralProperTime` over a
  **24-write sliding window** (deliberately *not* the whole store — more
  on why below), and fires once the accumulator crosses a threshold
  calibrated from an independent quiet baseline.

The sliding window matters. A previous nightly experiment in this same
repository tried a structural signal for a different purpose (deciding
*what* to evict, via graph min-cut) and was rejected because its signal
cost 1,800–2,700x the scalar baseline even at trivial corpus sizes — an
`O(n)`-or-worse design. This trigger was built from the start to be
`O(window · dims)`, independent of store size, specifically to not repeat
that mistake.

## Implementation

All three triggers share one `MemoryStore`, one `CoherencePolicy` for
*what* survives (held constant across the comparison — only *when* varies),
and get exercised against the same deterministic, seeded 1,400-write
stream: four 300-write "quiet" epochs (near-duplicate writes into the
current topic cluster) alternating with four 50-write "burst" epochs
(writes into a brand-new, never-before-seen cluster).

While building this, the benchmark's own Recall@10 metric came back
`0.0000` for every trigger — not a research finding, a bug. `MemoryStore`
assigned entry ids as `entries.len()` at insert time. Every prior use of
this crate only ever compacted once, terminally, so that always matched a
monotonic counter. This experiment's write-then-maybe-compact loop was the
first usage pattern to insert *after* a compaction had already shrunk the
store — at which point a new id could silently collide with a surviving
older entry's id. Fixed to an actual monotonic counter; a real, generally
useful correctness fix, independent of which trigger wins.

## Actual Benchmark Evidence

```
cargo run --release -p ruvector-agent-memory --features structural-gate \
  --example structural_gated_compaction_bench
```

```
trigger           compactions excess_size_integral      recall@10      wall_ms final_size
FixedInterval              24                30551         1.0000        6.161        200
Capacity                    5               120615         1.0000        2.415        200
StructuralGate            100                18913         1.0000       22.227        200

=== Diagnostic: fire location (1200 Quiet writes, 200 Burst writes) ===
trigger             fires@quiet    fires@burst
FixedInterval                20              4
Capacity                      4              1
StructuralGate               30             70
```

Pre-registered acceptance required **all four** of: ≥20% fewer compaction
calls, ≥20% less time-integrated excess store size, recall within 2
points, wall-clock within 2x — all decided before running the comparison.
Two passed (excess-size reduction: 38.1%; recall gap: 0). Two failed
(compaction-call count: 4.2x *higher*, not lower; wall-clock: 3.6–3.9x,
not ≤2x).

**Verdict: REJECT**, per the pre-registered thresholds, unchanged after
seeing the result.

## The Interesting Part

The diagnostic table is why this is a useful negative result rather than
just a negative result. 70 of the structural trigger's 100 fires landed on
the 200 writes that were genuine bursts — 14.3% of the stream accounting
for 70% of the fires, a roughly 14x fire-density skew toward the writes
that actually mattered. The signal is real.

It over-fired anyway, because of how `StructuralProperTime`'s coherence
channel is defined: it accumulates only on *coherence loss*
(`(prev - cur).max(0.0)`), which is the right design for genuine
irreversible drift, but a small sliding-window coherence estimate
fluctuates from sampling noise alone even when nothing is actually
changing. Every downward fluctuation adds "time"; no upward fluctuation
ever cancels it out. A threshold calibrated from that same quiet
baseline's *mean* tick was still well within its *variance* — the quiet
regime alone crossed it every ~40 writes, faster than the 50-write
fixed-interval baseline it was supposed to beat.

## Limitations

- One synthetic dataset shape; not validated against a real write trace.
- Wall-clock numbers conflate per-write trigger cost and the cost of
  running 100 vs. 24 actual `compact()` passes — not decomposed here.
- No deletes, no concurrent writers, no adversarial streams.
- Two of the clock's five channels (`graph`, prediction error) were left
  at zero rather than fed a fabricated signal.

## Production Relevance

Not production-ready as calibrated — rejected. What is production-ready
today: the `MemoryStore` id-collision fix (unconditionally useful, already
merged into this change), and the `CompactionTrigger` trait itself, which
is a clean extension point regardless of which implementation eventually
wins. The next real attempt should calibrate from both quiet *and* burst
reference data, not quiet alone, and should isolate whether the loss-only
coherence asymmetry or the entropy-histogram noise is the larger
contributor before touching the threshold again.

## RuVector Ecosystem Implications

This is the first attempt in the repository to reuse `emergent-time`'s
clock formalism outside its own crate, as a scheduling signal for
*maintenance*, not just anomaly detection or history compression. It
connects `ruvector-agent-memory`'s existing compaction-policy layer to
that clock, and — via the pattern, not yet the implementation — points at
the same approach being reusable for graph-maintenance scheduling
(`ruvector-mincut`), re-embedding/re-indexing schedules, or any other
agent-maintenance task whose cost should track actual churn rather than
wall time. The rejection is retained in the ADR/nightly lineage
specifically so a future attempt starts from "loss-only asymmetry +
small-window noise is the failure mode to design around" instead of
rediscovering it from scratch.

## Future Direction

Recalibrate from a mixed quiet+burst reference slice; ablate the
coherence-loss-only vs. entropy-histogram noise sources independently;
decompose wall-clock cost by source; only then reconsider promotion,
against the same pre-registered acceptance thresholds this run used.

## References

- `crates/emergent-time/src/structural_clock.rs` — `StructuralProperTime`,
  reused unmodified.
- `crates/ruvector-agent-memory/src/structural_gate.rs` — this
  experiment's trigger implementations.
- `crates/ruvector-agent-memory/examples/structural_gated_compaction_bench.rs`
  — the exact, reproducible benchmark.
- ADR-346 and the full nightly research report (README.md in this
  directory) for the complete evidence trail, including the prior
  (2026-09-05) structural-signal-for-compaction rejection this design was
  built to avoid repeating.
