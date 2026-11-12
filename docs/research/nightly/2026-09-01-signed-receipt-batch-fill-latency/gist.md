# Batching signatures is cheap. Waiting for a batch to fill is not free — here's the number.

## Problem

Signing every retrieval receipt individually costs one Ed25519 signature
per query. Batching B receipts under one signature amortizes that cost by
roughly B. This is well known and was already measured, in CPU-only
terms, for `ruvector-retrieval-receipt`'s signed-anchoring layer in an
earlier iteration (ADR-340): amortized signing cost at batch size 128
dropped to 5.8–7.7% of batch size 1.

What that earlier measurement explicitly declined to claim: whether
batching is safe to turn on for a real query stream. A batch doesn't
exist until it closes. If a fixed-size-only policy is waiting for the
32nd query to arrive and queries are arriving slowly, the first query in
that batch waits — potentially a long time — for its signed receipt to
become available. The CPU-cost benchmark assumed a batch was already
fully assembled in memory; it never modeled how long assembly itself
takes.

## Hypothesis

Given a stream of queries arriving under three load regimes — a target
rate high enough to fill a 32-query batch quickly, a light rate too slow
to fill it inside a reasonable window, and a bursty on/off pattern
approximating real agent traffic — compare three batch-fill policies:

- **B1**: sign immediately, no batching.
- **Fixed-size-only (B32)**: wait for exactly 32 queries, however long
  that takes.
- **Hybrid (B32, 50ms timeout)**: close at 32 queries *or* after 50ms
  since the oldest pending query, whichever comes first.

Prediction: the hybrid policy bounds worst-case latency close to the
timeout at every load regime; fixed-size-only does not, and the gap
should be dramatic at light load.

## Technical Design

Two new pieces, both additive to the existing `ruvector-retrieval-receipt`
crate:

1. **`BatchScheduler`** — pure, clock-free logic deciding when a batch of
   pending receipt roots closes. It takes arrival events and hands back a
   closed batch when the policy's size or timeout condition is met. No
   cryptography, no I/O, fully unit-testable with plain integers.

2. **A discrete-event simulation** — generates real Poisson and bursty
   arrival timelines with a seeded RNG, produces a real `MerkleReceipt`
   root for every arrival via the actual production search + receipt
   code path, and — whenever the scheduler closes a batch — performs a
   **real** `BatchAnchor::build` + `Issuer::sign_root` call, timed with
   `Instant`. That measured wall time is added to the batch's simulated
   close time to compute each member's real availability time.

The key methodological point: arrival *timing* is simulated (this is a
discrete-event simulation, not a live real-time system test with actual
`sleep`), but every operation that contributes cost to a reported latency
number — search, receipt construction, batch construction, signing,
verification — is real, measured work. This is what makes it possible to
test five orders of magnitude of load (50 to 2000 queries/second) in a
few seconds of wall-clock CPU time, deterministically and reproducibly,
rather than requiring a live test running for as long as the slowest
regime takes in real time.

## Actual Implementation

```rust
pub struct BatchFillPolicy {
    pub max_members: usize,
    pub max_wait_ns: Option<u64>, // None = fixed-size-only
}

impl BatchFillPolicy {
    pub const fn fixed_size(max_members: usize) -> Self { .. }
    pub const fn hybrid(max_members: usize, max_wait_ns: u64) -> Self { .. }
}
```

`BatchScheduler::arrive()` returns `Some(batch)` when an arrival fills
the batch; `oldest_pending_arrival_ns()` lets a caller derive the next
timeout deadline (`oldest + max_wait_ns`) without the scheduler owning a
clock; `close_on_timeout()` force-closes whatever's pending when that
deadline fires. The simulation's event loop merges two sorted event
sources — the next arrival, and the current timeout deadline (recomputed
after every scheduler mutation) — and always processes whichever is
earlier, so a timeout that should have fired *before* the next arrival
closes the batch at the correct simulated time, not at the next
arrival's time.

## Real Benchmark Evidence

n=2,000 vectors, dims=64, k=10, 3,000 queries per regime, mean of 3
independent process runs (raw output preserved in this run's companion
`raw-runs.txt`):

| regime | policy | p99 latency | mean batch size |
|---|---|---:|---:|
| target load (1000 q/s) | fixed-size-only | 36.80ms | 31.9 |
| target load (1000 q/s) | hybrid (50ms) | 36.81ms | 31.9 |
| light load (50 q/s) | fixed-size-only | **756.13ms** | 31.9 |
| light load (50 q/s) | hybrid (50ms) | **50.03ms** | 3.5 |
| bursty (on/off) | fixed-size-only | 415.69ms | 31.9 |
| bursty (on/off) | hybrid (50ms) | 49.30ms | 29.4 |

At target load, the hybrid policy costs nothing versus fixed-size-only —
batches fill well inside the timeout, so the timeout essentially never
fires. At light load, fixed-size-only's tail latency is over an order of
magnitude worse than the hybrid policy's, while the hybrid policy's p99
sits right where it should: at the configured 50ms bound plus a small
signing-cost epsilon. Every closed batch, in every regime, in every run,
verified correctly — bounding latency did not cost any correctness.

All four pre-registered acceptance thresholds passed in all 3 runs;
result: **ACCEPT**.

## Limitations

- The simulation assumes a single serialized signer with no queueing
  delay for the sign operation itself — valid here because real signing
  cost (single-digit to tens of microseconds) is three-plus orders of
  magnitude below every tested fill window (32ms to 640ms mean), but
  untested at arrival rates high enough to invalidate that assumption.
- Traffic is synthetic (Poisson and on/off bursty), not measured
  production traffic. The specific 50ms timeout value demonstrates the
  mechanism; it is not a recommended universal default.
- This is a discrete-event simulation combining real cryptographic
  operation costs with simulated arrival timing — stated explicitly
  rather than presented as a live real-time deployment test.

## Production Relevance

Any system attaching signed provenance to retrieval — agent-memory audit
trails, regulatory RAG, cross-agent attestations in a multi-agent swarm —
needs to know not just "how cheap is signing" but "how long might a
receipt be unavailable." A fixed-size-only batching policy answers the
first question well and the second question badly, in a way that gets
worse, unboundedly, as load drops. A bounded-timeout hybrid policy is
this experiment's answer: a small, measured amortization cost at low
load, in exchange for a latency guarantee an operator actually chooses
rather than one that emerges however traffic happens to behave.

## RuVector Ecosystem Implications

This is a direct extension of `ruvector-retrieval-receipt` (ADR-304,
ADR-340), not a new island: no new crate, no new dependency, and every
pre-existing test in the crate still passes unchanged. It closes a gap
the crate's own prior research explicitly named rather than speculating
about a new capability — the kind of "attack its primary bottleneck"
follow-up this repository's nightly research process treats as
preferable to starting a fresh, unrelated topic.

## Future Direction

The next honest step is replacing synthetic Poisson/bursty traffic with
real agent-memory query-arrival traces and re-deriving an
appropriate timeout from actual data, plus finding where the single-
serialized-signer assumption breaks down at higher simulated rates. Both
are listed as open items in the accompanying ADR and nightly report
rather than claimed as already answered.

## References

- ADR-340: Signed Retrieval-Receipt Anchoring (this repository).
- ADR-343: Signed-Receipt Batch-Fill Latency (this run).
- 2026-08-31 nightly research report, whose "Next Research" item 1 is
  this experiment's direct origin.
