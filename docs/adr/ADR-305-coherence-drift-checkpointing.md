# ADR-305: Coherence-Drift-Triggered Checkpointing for Agent Memory (Hypothesis Rejected)

## Status

**Rejected** (hypothesis falsified by measurement). Experimental crate
(`ruvector-coherence-checkpoint`) retained: the witness-chained snapshot
*mechanism* is correct and reusable, but its motivating drift-trigger
heuristic must not be promoted or reused as specified. See Rejection
Criteria and Open Questions for what a corrected follow-up would need.

## Context

`ruvector-agent-memory` (ADR-linked nightly 2026-06-14) answers *which*
memories to keep during compaction. `ruvector-temporal-coherence` (nightly
2026-06-13) answers *how* a query should weight memories by recency and
graph coherence. `ruvector-proof-gate` (ADR-227) gives every individual
*write* a tamper-evident receipt. `ruvector-retrieval-receipt` (ADR-304)
gives every *query result set* a tamper-evident receipt. None of these
answer a fourth, distinct scheduling question every durable agent-memory
deployment faces: **when should the running store's full state be
checkpointed into a signed, portable snapshot** (the RVF portable-artifact
use case), so a crash, migration, or fleet handoff can recover without
replaying the entire write history from the beginning?

The naive answer — snapshot every N writes — is what every reviewed
system implicitly does when it checkpoints on a schedule. It wastes
storage during quiet periods and, more importantly, gives no guarantee
that a checkpoint lands promptly after a period where the agent's working
context has genuinely shifted (a topic change, a new task). This ADR asks
whether reusing RuVector's existing coherence-drift concept — already
used by `ruvector-temporal-coherence` to gate retrieval — as a checkpoint
*trigger* instead of a query filter, can do better than periodic
checkpointing on the metric that actually matters for recovery: worst-case
replay gap (how many writes must be replayed after the nearest snapshot to
reconstruct exact state).

## Hypothesis

```text
Given a memory store fed a deterministic, seeded event stream alternating
calm phases (samples clustered tightly around a fixed centroid) and burst
phases (the centroid linearly walked to a fresh random direction over the
phase),

when snapshot scheduling is driven by centroid drift (cosine distance
between the store's whole-history running centroid and the centroid
captured at the last snapshot) instead of a fixed event interval,

then, at an equal (±1) snapshot budget, the drift-triggered policy's
worst-case replay gap (max events since the nearest snapshot, sampled
across the entire stream) should be at least 20% lower than a fixed-
interval baseline tuned to the same budget,

subject to every variant reconstructing exact state on replay (100%
vector-for-vector match, not an approximation) and every snapshot's
witness receipt re-deriving cleanly from genesis.
```

This acceptance threshold (20% max-gap reduction at matched budget, 100%
exact replay, 100% witness integrity) was fixed in the benchmark's
acceptance-check code before the first measurement was taken and was not
adjusted after seeing results.

## Decision

Add `crates/ruvector-coherence-checkpoint`, implementing three snapshot
policies over a shared `CheckpointPolicy` trait, all witness-chained
through a real `ruvector_proof_gate::HashChainGate`:

- `FixedInterval` (baseline) — snapshot every `interval` events.
- `DriftTriggered` (candidate A) — snapshot when cosine distance between
  the current running centroid and the last snapshot's centroid exceeds
  `threshold`.
- `DriftTriggeredCapped` (candidate B) — `DriftTriggered` plus a hard
  `max_interval` ceiling, bounding worst-case gap even if drift never
  crosses threshold during a long calm stretch.

Each snapshot stores a full copy of the store's vectors (this is a
checkpoint mechanism, not lossy compaction — recovery must be exact),
a SHA-256 `state_digest` over that copy, and a `WriteReceipt` from
admitting `(centroid, state_digest)` through `HashChainGate`. Recovery
correctness is checked by actually reconstructing state (snapshot +
replayed events) and comparing it vector-for-vector against ground truth
at 40 sampled points per run — not merely comparing digests, which would
only prove the *claim* was self-consistent, not that reconstruction is
lossless.

## Evidence

Measured via `cargo run --release -p ruvector-coherence-checkpoint
--example benchmark -- 6000 48 <seed> <threshold>` (n_events=6000, dims=48,
calm_phase_len=250, burst_phase_len=50, noise=±0.04). Hardware: x86-64, 4
logical CPUs, Linux 6.18.5, `rustc` 1.94.1, release build. Full raw output
tables are in the nightly research README; summary:

| threshold | seed | baseline max_gap | candidate_A max_gap | reduction |
|---|---|---|---|---|
| 0.02 | 2026 | 205 | 329 | **-60.5%** |
| 0.05 | 2026 | 374 | 588 | **-57.2%** |
| 0.08 | 2026 | 499 | 839 | **-68.1%** |
| 0.15 | 2026 | 749 | 1570 | **-109.6%** |
| 0.25 | 2026 | 1199 | 2663 | **-122.1%** |
| 0.08 | 7 | 374 | 791 | **-111.5%** |
| 0.08 | 4242 | 499 | 1122 | **-124.8%** |
| 0.08 | 99 | 427 | 1019 | **-138.6%** |

Negative "reduction" means candidate_A's worst-case gap is *larger* than
baseline's — the opposite of the hypothesis, at every threshold tested and
every seed tested. `ACCEPTANCE_RESULT: REJECT` in all 8 runs.

Diagnostic (`examples/diag_snapshot_indices.rs`) explains the mechanism:
for threshold=0.08, seed=2026, candidate_A's snapshot event indices are
`[0, 384, 655, 1043, 1390, 1730, 2173, 2647, 3121, 3874, 4714, 5494]` —
inter-snapshot gaps grow from ~300-470 early in the stream to 753-840 late
in the stream. A whole-history running centroid is a mean over an
ever-growing sample count; each new burst contributes a shrinking fraction
to that mean as the store accumulates history, so the drift signal
becomes progressively less sensitive to recent bursts over the stream's
lifetime — the opposite of what an adaptive trigger needs. `threshold`
does not fix this: raising it only makes the effect worse (see table),
because a higher bar takes even longer for a damped signal to cross.

Correctness and witness integrity held at 100% for all three variants,
all 8 runs, all 40 sampled replay targets per run: `exact_replay=40/40`,
`chain_rederivation_ok=true`, `receipt_structural_ok=true` throughout. The
checkpoint/witness/replay *mechanism* is sound; only the drift-trigger
*heuristic* is falsified.

`DriftTriggeredCapped` (candidate B) never underperformed baseline's
max_gap (the cap makes it structurally impossible to), but it also never
beat it — every measured run shows candidate B's max_gap tying or equal to
baseline's, while consuming more snapshots (and more storage) to get
there. It is dominated by simply running `FixedInterval` at the cap's
interval; the drift component contributes no measured benefit in any run.

## Consequences

**Positive:**
- Falsifies, with reproducible evidence across 5 thresholds and 4 seeds, a
  plausible-sounding design (reuse an existing coherence-drift signal as a
  checkpoint trigger) before it could be built into a production snapshot
  scheduler. This is exactly the kind of mistake that "it uses the same
  drift concept as `ruvector-temporal-coherence`" intuition would not have
  caught without measurement.
- Identifies the specific mechanism (whole-history running-mean dilution)
  responsible, which is a general lesson for any future drift-based
  trigger in this codebase, not specific to checkpointing: a cumulative
  mean is the wrong signal once a stream is long-lived; a bounded-window
  or exponentially-weighted mean is very likely necessary instead (see
  Open Questions).
- Delivers real, tested infrastructure regardless of the negative result:
  a witness-chained, exact-replay-verified checkpoint mechanism generic
  over any `CheckpointPolicy`, immediately reusable once a better trigger
  signal is found.

**Negative / costs:**
- No production improvement ships from this ADR. `FixedInterval` remains
  the recommended checkpoint policy for `ruvector-agent-memory`-style
  stores until a corrected trigger is measured.
- The synthetic workload (alternating calm/burst phases) is a specific,
  documented model of "coherence drift," not a general one; a different
  drift pattern (e.g. continuous slow drift with no calm phases) was not
  tested and could behave differently. The rejection is scoped to the
  tested workload family, not claimed as universal.

## Alternatives Considered

- **Windowed drift (fixed-size lookback, e.g. last 500 events only).**
  Not implemented this run — it is the leading hypothesis for why the
  measured design failed, and per Step 10 of the nightly process the
  hypothesis under test may not be silently swapped after seeing results.
  Recorded as the top candidate for a follow-up nightly (see Open
  Questions), to be run as its own hypothesis with its own fresh
  measurement.
- **Exponentially-weighted moving centroid** (recent events weighted more
  than old ones without a hard window boundary). Same reasoning as above:
  a real candidate, not measured tonight, not claimed.
- **Snapshot on every burst-phase boundary directly (oracle policy).**
  Deliberately not implemented: the workload generator's phase boundaries
  are not available to a real checkpoint policy at write time (a real
  agent memory store does not know in advance when a "burst" starts or
  ends) — an oracle policy would not be a fair comparison to a runtime-
  observable trigger and was excluded to avoid an unfalsifiable, unusable
  result.

## Implementation Plan

Because the hypothesis was rejected, there is no promotion plan for
`DriftTriggered`/`DriftTriggeredCapped` as specified. If a windowed or
EWMA drift signal is measured in a follow-up nightly and clears the same
20%-reduction / 100%-correctness bar:

1. Swap `RunningCentroid` (whole-history mean) for a bounded-window or
   EWMA variant behind the same `CheckpointPolicy` trait — no other crate
   surface changes.
2. Re-run this ADR's exact benchmark command against the new trigger to
   get a directly comparable number.
3. If it clears acceptance, integrate as an optional checkpoint scheduler
   for `ruvector-agent-memory`, gated behind a feature flag.
4. RVF integration: snapshots already carry a `state_digest` and a
   witness `WriteReceipt` — the two fields an RVF portable-package
   manifest needs to make a checkpoint independently verifiable outside
   the process that produced it. Wiring `Snapshot` into an actual RVF
   container format is separate future work, not attempted here.

## API Shape

```rust
let events = generate_workload(&cfg, seed);
let (run, gate) = run_checkpoint_policy(&events, dims, DriftTriggered { threshold });
assert!(run.gate_integrity_ok);
assert!(verify_exact_replay(&run, &events, target_index));
```

## Feature Flags

None — the crate is opt-in by virtue of not being a dependency of any
other crate in the workspace.

## Benchmark Evidence

See `docs/research/nightly/2026-08-18-coherence-drift-checkpointing/README.md`
for the full methodology and raw `cargo run --release` output across all
8 measured (threshold, seed) combinations.

## Security

- Reuses `ruvector-proof-gate`'s `HashChainGate` unmodified — no new
  cryptographic primitives introduced.
- Every snapshot's receipt is checked two ways: structural chain
  consistency (`HashChainGate::verify_receipt` / `verify_integrity`) and
  payload rehash-and-compare (`WritePayload::payload_hash()` recomputed
  from the claimed snapshot content) — a corrupted stored digest is
  caught by the second check even if the chain structure alone would
  miss a downstream mutation (tested:
  `tampering_stored_digest_is_caught_by_payload_rehash`,
  `tampering_a_receipt_commitment_fails_structural_check`).
- No new `unsafe` code. No network calls. Dependency surface is `sha2` +
  `rand`, matching the WASM-compatible shape of `ruvector-proof-gate`.

## Governance

A rejected hypothesis is retained in-tree (this ADR + the crate) rather
than deleted, per the nightly process's flywheel requirement: future
agents must not re-propose whole-history cumulative-centroid drift as a
checkpoint trigger without first reading this ADR's evidence.

## Failure Modes

- If `events` is empty, `run_checkpoint_policy` produces a run with zero
  snapshots and an empty `gap_at_event`; `max_gap()`/`mean_gap()` return
  `0`/`0.0` rather than panicking (`unwrap_or(0)` / empty-check guards).
- `verify_exact_replay` returns `false` (not an error/panic) when no
  snapshot exists at or before the target index, or when reconstruction
  diverges from ground truth by even one component of one vector.

## Migration

N/A — new, unintegrated, rejected-hypothesis crate. Not depended on by any
other workspace member.

## Rollback

Delete `crates/ruvector-coherence-checkpoint` and its workspace member
entry in the root `Cargo.toml`; nothing else depends on it. Given the
hypothesis is already rejected, rollback would only be warranted if the
crate itself (the reusable witness/replay mechanism) is judged not worth
keeping as infrastructure for a follow-up nightly.

## Rejection Criteria

Already met — restated for clarity, since this ADR documents a rejection
rather than a promotion:

- `DriftTriggered`'s max replay gap was *larger* than the matched-budget
  `FixedInterval` baseline's in all 8 measured (threshold, seed)
  combinations (range: 57%-139% larger), against an acceptance bar of
  "at least 20% smaller."
- No threshold in `{0.02, 0.05, 0.08, 0.15, 0.25}` produced a passing
  result; the effect strictly worsens as threshold increases.

## Open Questions

- Does a fixed-size windowed centroid (e.g. last W events) or an
  exponentially-weighted moving centroid fix the diminishing-sensitivity
  problem identified here? This is the concrete, falsifiable follow-up
  hypothesis for a future nightly, with the same benchmark harness
  reusable unmodified (only `RunningCentroid` needs a windowed variant).
- Is centroid drift the right *content* signal at all, or would a
  coherence-graph-density signal (mirroring `ruvector-temporal-coherence`'s
  `CoherenceGraph` more directly, rather than only borrowing its "drift"
  vocabulary) behave differently under the same workload?
- Does the calm/burst synthetic workload model realistic agent-memory
  drift, or would a real agent transcript corpus show different
  degradation characteristics? Not addressed here — the workload is
  synthetic and deterministic by design (Step 12 of the nightly process),
  which trades realism for reproducibility.
