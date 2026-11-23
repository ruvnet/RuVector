# ADR-361: Bounded Fill-Timeout for `BatchTail` Witness Signing

## Status

Accepted. New, opt-in, additive module (`ruvector-agent-memory::batch_tail_timeout`,
`TimedBatchTailSink`). No change to `witness_signing.rs`'s existing,
ADR-347-reviewed `SignedWitnessSink`/`SigningStrategy` API; one small
accessor (`TransactionalLedger::witness_sink_mut`) added to `ledger.rs`
for live integration. Nightly research run: `docs/research/nightly/2026-11-23-witness-batch-tail-timeout/README.md`.

## Context

ADR-347 (`witness_signing.rs`) added `SigningStrategy::BatchTail{batch_size}`:
an Ed25519-signed span closes once `batch_size` witness records have
arrived, amortizing signature cost across many records. It has no notion
of wall-clock time — a deployment receiving records slower than
`batch_size` fills can leave a span open indefinitely, with no bound on
how long a write stays signature-unavailable. The 2026-09-16 nightly
(`docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`)
built and benchmarked `BatchTail`, explicitly named this gap as its own
Next Research item 1, and pointed at the fix already built for an
analogous gap in `ruvector-retrieval-receipt` (ADR-340 / its own nightly
follow-up, ADR-343): `batch_fill::{BatchFillPolicy, BatchScheduler}`
closes a batch at `max_members` or after `max_wait_ns`, whichever comes
first.

## Hypothesis

Given a `BatchTail`-signed witness log receiving records slower than
`batch_size` would naturally fill, wrapping it in a wall-clock fill
timeout (porting the cited policy-decision methodology) bounds p99
signature-availability latency to approximately `max_wait_ns`, at
negligible signing-cost overhead when load is healthy enough that the
timeout rarely fires.

## Decision

1. Add `TimedBatchTailSink<S: WitnessSink, C: Clock = SystemClock>` in a
   new module, `batch_tail_timeout.rs`, wrapping an already-constructed
   `SignedWitnessSink<S>` **unchanged**. It implements `WitnessSink`
   itself (drop-in for `SignedWitnessSink` inside `TransactionalLedger`)
   and adds exactly: `opened_at_ns: Option<u64>` tracking when the
   current pending span opened, `check_timeout()`/`check_timeout_at(now_ns)`
   to seal it once `max_wait_ns` has elapsed, and `pending_deadline_ns()`
   to tell a caller the latest time it must check by.
2. Time comes from an injected `Clock` trait (`now_ns() -> u64`):
   `SystemClock` (wall clock, production) or `LogicalClock` (manually
   advanced, deterministic simulation/tests). This is dependency
   injection for time, not a new time-measurement primitive.
3. `witness_signing.rs` is **not modified**. The wrapper calls only its
   existing public API (`unsigned_pending()`, `seal()`). This was
   deliberate: that module is security-reviewed (ADR-347's revision
   history), already near the repository's 500-line file guidance, and
   the new capability needs none of its internals.
4. Add `TransactionalLedger::witness_sink_mut(&mut self) -> &mut S`,
   mirroring the existing `witness_sink(&self) -> &S`. Required for a
   host application to call `check_timeout()` on a live ledger's sink
   without tearing the ledger down — tearing it down and rebuilding a
   fresh `TransactionalLedger` around the recovered sink resets
   `witness_seq` to 0, which the sink's sequence-continuity check then
   correctly rejects (discovered and fixed during this ADR's own test
   development, not a theoretical concern).
5. Scheduling `check_timeout`/`check_timeout_at` is the caller's
   responsibility; nothing here spawns a background thread. A sketched
   ruFlo periodic-tick integration is Next Research, not implemented.

## API Shape

```rust
pub trait Clock {
    fn now_ns(&self) -> u64;
}
pub struct SystemClock;      // wall clock
pub struct LogicalClock(..); // manually advanced, for tests/sims

pub struct TimedBatchTailSink<S: WitnessSink, C: Clock = SystemClock> { .. }

impl<S: WitnessSink, C: Clock> TimedBatchTailSink<S, C> {
    pub fn new(inner: SignedWitnessSink<S>, max_wait_ns: u64, clock: C) -> Self;
    pub fn check_timeout(&mut self) -> bool;
    pub fn check_timeout_at(&mut self, now_ns: u64) -> bool;
    pub fn pending_deadline_ns(&self) -> Option<u64>;
    pub fn timeout_seals(&self) -> usize;
    // + public_key/spans/anchor/unsigned_pending/inner_log/seal, mirroring SignedWitnessSink
}

impl<S: WitnessSink, C: Clock> WitnessSink for TimedBatchTailSink<S, C> { .. }
```

## Feature Flags

None added. No new Cargo dependency (reuses `ed25519-dalek`/`rvf-types`
already required by `witness_signing.rs`). The capability is reachable
simply by constructing `TimedBatchTailSink` instead of using
`SignedWitnessSink` directly at a call site; existing call sites are
unaffected.

## Benchmark Evidence

Full methodology and raw output:
`docs/research/nightly/2026-11-23-witness-batch-tail-timeout/README.md`
and its `benchmark_raw_output.txt`. Summary (4000 synthetic arrivals/regime,
`batch_size=64`, release build, `rustc 1.97.0`, Linux x86_64):

| regime | policy | p99 availability | amortized sign cost | verified |
|---|---|---|---|---|
| light_load_50qps | baseline (no timeout) | 1402.1ms | 612ns | yes |
| light_load_50qps | `max_wait_ns=50ms` | 50.1ms | 10.8µs | yes |
| light_load_50qps | `max_wait_ns=10ms` | 10.0ms | 19.5µs | yes |
| target_load_5000qps | baseline | 14.5ms | 620ns | yes |
| target_load_5000qps | `max_wait_ns=50ms` | 14.5ms | 605ns | yes |

The 50ms timeout cuts light-load p99 by ~28x and stays within its
declared bound; at healthy load it never fires and costs <3% extra.
Acceptance result: **ACCEPT**.

## Security

No new cryptographic primitive; a timeout-sealed span is produced by the
same `SignedWitnessSink::seal()` path as a size-sealed one and is
indistinguishable to `verify_signed_chain` (confirmed by unit test and by
every benchmark regime reporting `verified=true`). `witness_sink_mut()`
exposes no new trust boundary: a caller already holding `&mut
TransactionalLedger` could already recover full ownership of the sink via
`into_witness_sink()`.

## Governance

Additive, backward-compatible, opt-in. No existing caller's behavior
changes. Not promoted as the new default for `BatchTail` — see
Consequences.

## Failure Modes

1. Crash while a span (size-pending or timeout-pending) is open: unsigned
   tail, unchanged from ADR-347's existing documented limit.
2. If the host application never calls `check_timeout`/`check_timeout_at`,
   behavior silently reverts to baseline's unbounded wait — the mechanism
   is provided, scheduling is not. Flagged as a production footgun; see
   Open Questions.
3. `check_timeout_at` uses `saturating_sub`, so a clock moving backward
   only delays a seal, never panics or seals early.

## Migration

No migration required for existing code. To adopt: construct
`TimedBatchTailSink::new(signed_sink, max_wait_ns, SystemClock)` instead
of using `SignedWitnessSink` directly, and arrange a periodic
`check_timeout()` call (see Open Questions / Next Research).

## Rollback

Delete `batch_tail_timeout.rs` and the `witness_sink_mut` accessor; no
other code depends on either (verified: `grep` for
`TimedBatchTailSink`/`witness_sink_mut` outside this module and its own
tests/benchmark returns nothing). Zero blast radius.

## Rejection Criteria

Would have been rejected (REJECT, not ACCEPT) had any of: a timeout-sealed
span failed `verify_signed_chain`; `candidate_a`/`candidate_b`'s p99
latency exceeded its declared bound plus the fixed 5ms slack; baseline's
unbounded-tail failure mode failed to materialize (i.e. baseline's p99
did not meaningfully exceed the timeout variants'); or amortized signing
cost at healthy load exceeded 2x baseline's. None of these occurred — see
Benchmark Evidence.

## Open Questions

1. Who calls `check_timeout` in production — a ruFlo-owned periodic tick
   is sketched (nightly README's ruFlo Implications) but not implemented.
2. Whether `(batch_size, max_wait_ns)` should eventually be Darwin-tuned
   per deployment rather than hand-chosen (two points sampled tonight:
   50ms, 10ms) — no Darwin/MetaHarness/Flywheel CLI was available in this
   environment to run that sweep tonight (verified absent; see the
   nightly README's capability-discovery table).
3. Real traffic-trace validation of the 5ms slack constant and the two
   sampled `max_wait_ns` points, versus this run's synthetic
   Poisson/bursty arrival processes.

## Consequences

- **Positive:** a measured, bounded signature-availability latency option
  for `BatchTail`, at near-zero cost when unneeded, with the tradeoff
  against signing throughput explicitly quantified rather than asserted.
- **Positive:** `witness_signing.rs` gained a new capability without a
  single line of its own code changing, demonstrating the
  decorator-over-modification pattern is viable for this crate's other
  `WitnessSink` implementors too.
- **Neutral:** `TransactionalLedger` gained one more public accessor
  (`witness_sink_mut`), slightly widening its surface for symmetry with
  the existing `witness_sink`.
- **Negative (accepted):** scheduling `check_timeout` is left to the
  caller; a caller that forgets gets silent baseline behavior rather than
  a compile-time or runtime error. Judged acceptable because the
  alternative (spawning a thread from inside a library sink) is a worse
  surprise for an embedding application.

## What should become stable API

`TimedBatchTailSink`'s public methods (`new`, `check_timeout`,
`check_timeout_at`, `pending_deadline_ns`, `timeout_seals`,
`max_wait_ns`) and `TransactionalLedger::witness_sink_mut` — these are
the complete, tested, documented integration surface.

## What should remain experimental

The specific `(batch_size, max_wait_ns)` operating points (50ms/10ms) are
this run's sampled anchors, not a recommendation; no default value is
set, and callers must choose explicitly.

## Why does this belong in RuVector?

It closes a gap RuVector's own prior nightly research named explicitly,
using a methodology RuVector's own prior nightly research already built
and validated in a sibling crate — this is ecosystem-internal knowledge
reuse, exactly the kind the nightly process exists to accumulate.

## Why is this more than an experiment?

It is a tested (6 new unit tests, 54/54 existing tests still green),
benchmarked (real Ed25519 signs, reproducible seeds), zero-new-dependency
library capability already usable at any `BatchTail` call site today.

## What evidence supports promotion (as an available capability)?

The benchmark table above: bounded latency achieved within declared
thresholds, baseline failure mode demonstrated at ~28x margin, healthy-load
overhead statistically indistinguishable from baseline, 100% correctness
across all 9 regime×policy combinations, reproducible across two runs.

## What evidence would cause rejection?

See Rejection Criteria above — none of those conditions occurred.
