# Why cumulative-mean drift is the wrong signal for checkpoint scheduling

## Problem

Long-running agent-memory stores need durable checkpoints for crash
recovery and migration. The obvious scheduler is periodic: snapshot every
N writes. It's simple, but it doesn't adapt to *when* the store's content
actually changed — it wastes storage during quiet periods and gives no
guarantee a snapshot lands promptly after the store's semantic content
genuinely shifted.

RuVector already has a coherence-drift signal (`ruvector-temporal-coherence`
uses cosine-distance-based centroid comparison to weight retrieval by how
much a memory has "aged" relative to current context). The obvious next
idea: reuse that signal to *trigger checkpoints* instead of gating
retrieval — snapshot when the store's centroid has moved far enough since
the last snapshot, rather than on a fixed schedule.

## Hypothesis

At an equal snapshot budget (same total number of snapshots taken),
drift-triggered scheduling should reduce the *worst-case replay gap* —
the number of writes that would need replaying, from the nearest
snapshot, to recover exact state at the worst possible failure point —
relative to fixed-interval scheduling. The intuition: fixed intervals
waste snapshots during calm periods and might miss covering a burst
promptly; a content-aware trigger should place snapshots where they
matter.

Acceptance bar, fixed before measuring: ≥20% reduction in max replay gap
at matched snapshot budget, with 100% exact-replay correctness and 100%
witness-chain integrity for every variant.

## Technical Design

Three checkpoint policies behind a shared trait:

```rust
pub trait CheckpointPolicy {
    fn should_snapshot(&mut self, ctx: &PolicyContext<'_>) -> bool;
    fn name(&self) -> &'static str;
}
```

- `FixedInterval` — snapshot every `interval` events.
- `DriftTriggered` — snapshot when `drift(last_snapshot_centroid,
  current_centroid) >= threshold`, where `drift` is `1 - cosine_sim` and
  `current_centroid` is a `RunningCentroid` — an O(dims)-per-insert
  incremental mean over *every* event since the store began.
- `DriftTriggeredCapped` — `DriftTriggered` plus a hard `max_interval`
  ceiling.

Every snapshot copies the full store state (exact recovery, not lossy
compaction), computes a SHA-256 digest over it, and admits
`(centroid, digest)` as a `WritePayload` through a real
`ruvector_proof_gate::HashChainGate`, producing a witness-chained
`WriteReceipt`. Correctness is checked by actually reconstructing state —
snapshot entries plus replayed events — and comparing it vector-for-vector
against independently-computed ground truth at 40 sampled points per run,
not by trusting the digest alone.

## Actual Implementation

`crates/ruvector-coherence-checkpoint`, real path dependencies on
`ruvector-agent-memory` (for `MemoryStore`) and `ruvector-proof-gate` (for
the witness chain) — no mocks. 19 tests, including two adversarial tamper
tests that flip a byte in a receipt commitment or a stored digest and
confirm the corresponding verification fails.

## Actual Benchmark Evidence

Deterministic seeded workload: 6,000 events, 48 dims, alternating calm
phases (250 events clustered around a fixed centroid, ±0.04 noise) and
burst phases (50 events linearly walking the centroid to a fresh random
direction). `cargo run --release -p ruvector-coherence-checkpoint --example
benchmark -- 6000 48 <seed> <threshold>`, x86-64/4 cores/Linux
6.18.5/rustc 1.94.1.

| threshold | seed | baseline max_gap | candidate_A max_gap | reduction vs. hypothesis's +20% bar |
|---|---|---|---|---|
| 0.02 | 2026 | 205 | 329 | -60.5% |
| 0.05 | 2026 | 374 | 588 | -57.2% |
| 0.08 | 2026 | 499 | 839 | -68.1% |
| 0.15 | 2026 | 749 | 1570 | -109.6% |
| 0.25 | 2026 | 1199 | 2663 | -122.1% |
| 0.08 | 7 | 374 | 791 | -111.5% |
| 0.08 | 4242 | 499 | 1122 | -124.8% |
| 0.08 | 99 | 427 | 1019 | -138.6% |

Every row: `REJECT`. Not one of 8 (threshold, seed) combinations came
close to the +20% bar — all landed on the wrong side of zero. Replay
correctness and witness integrity held at 100% in every run
(`exact_replay=40/40`, `chain_rederivation_ok=true`,
`receipt_structural_ok=true`) — the checkpoint mechanism itself works;
only the trigger heuristic fails.

**Why:** printing candidate_A's actual snapshot event indices at
threshold=0.08 shows inter-snapshot gaps growing from ~300-470 early in
the stream to 753-840 late in the stream:

```text
snapshot event indices: [0, 384, 655, 1043, 1390, 1730, 2173, 2647, 3121, 3874, 4714, 5494]
inter-snapshot gaps:    [384, 271, 388, 347, 340, 443, 474, 474, 753, 840, 780]
```

`RunningCentroid` is a mean over an ever-growing sample count. A 50-event
burst contributes a shrinking fraction to that mean as the total event
count grows, so the drift-since-last-snapshot signal takes longer to
cross `threshold` the further into the stream you go. Raising the
threshold makes this strictly worse (see table) — consistent with the
diagnosis: a higher bar takes even longer for an increasingly damped
signal to reach. A whole-history cumulative mean is the wrong content
signal for a trigger that needs to stay responsive to *recent* events on
an unbounded stream.

## Limitations

Single synthetic workload family; single dimensionality/event-count
scale; no concurrent-write or delete scenario tested. The rejection is
scoped to what was measured, not claimed universal — see the ADR's full
Limitations section.

## Production Relevance

None yet, and that's the honest result. `FixedInterval` remains the
recommended default checkpoint policy for RuVector agent-memory stores.
What *does* ship from this work: a real, tested, witness-chained
checkpoint-and-exact-replay mechanism, decoupled from the (rejected)
trigger heuristic and ready to drive whatever trigger a follow-up
experiment does validate.

## RuVector Ecosystem Implications

The negative result is itself useful ecosystem knowledge: it rules out
naive whole-history-mean drift as a scheduling signal anywhere in
RuVector, not just for checkpointing — the same failure mode would hit
any other feature considering a cumulative-mean "how much has this
changed" trigger on a long-lived stream.

## Future Direction

The concrete, falsifiable follow-up: swap `RunningCentroid` for a
fixed-size windowed mean or an exponentially-weighted moving centroid,
which should stay responsive to recent bursts regardless of total stream
length, and re-run this exact benchmark harness unmodified. That is next
week's hypothesis, not this week's — Step 10 of the nightly research
process is explicit that a hypothesis is not silently changed after
seeing results; this one is closed as REJECTED, and the corrected version
gets measured fresh.

## References

- ADR-305 (full decision record for this rejection).
- ADR-227 / ADR-304 (`ruvector-proof-gate`, `ruvector-retrieval-receipt` —
  the witness-chain infrastructure this crate reuses).
- `docs/research/nightly/2026-06-13-temporal-coherence-agent-memory/` —
  origin of the coherence-drift concept applied here to a new question.
