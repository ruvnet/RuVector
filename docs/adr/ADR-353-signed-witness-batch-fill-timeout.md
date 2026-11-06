# ADR-353: Signed Witness Batch-Fill Timeout — Bounding `BatchTail`'s Unsigned-Tail Latency

## Status

Proposed. Additive extension to `ruvector-agent-memory::witness_signing`
(a new `SigningStrategy::BatchTailTimeout` variant plus
`SignedWitnessSink::oldest_pending_arrival_ns` /
`SignedWitnessSink::check_timeout`), not wired into any default ledger
construction path. Does not modify `SigningStrategy::PerRecord`,
`SigningStrategy::BatchTail`, or any existing test.

## Context

ADR-347 added `SignedWitnessSink` with two strategies: `PerRecord` (every
record gets its own signature) and `BatchTail { batch_size }` (one
signature amortized over `batch_size` records). `BatchTail` has no
wall-clock bound: its pending span closes only when `batch_size` records
have arrived, however long that takes. The 2026-09-16 nightly research
report (`docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`)
named this gap verbatim as Next Research item #1:

> A wall-clock batch-fill timeout for `BatchTail` (reusing the
> `BatchFillPolicy`/`BatchScheduler` methodology already built and
> measured in `ruvector-retrieval-receipt::batch_fill`, without adding
> that crate as a dependency — port the pattern, not the code), so a
> deployment gets a bounded worst-case signature-availability latency
> instead of "whenever the batch happens to fill."

This is the second instance of the same gap: ADR-340
(`ruvector-retrieval-receipt`'s signed-receipt anchoring) named it first;
ADR-343 closed it there with `BatchFillPolicy::hybrid` and a discrete-event
`batch_latency` simulation. ADR-343's own Open Questions item 1 asked
"What `max_wait_ns` values are appropriate for real agent-memory query
traffic shapes" — anticipating exactly this follow-up.

`ruvector-agent-memory`'s `SignedWitnessSink` cannot reuse
`ruvector-retrieval-receipt::batch_fill`'s `BatchScheduler` directly: that
type hands closed batches of opaque `PendingMember`s to an external
signer, whereas `SignedWitnessSink::emit_batch` signs witness *spans*
in-place, inline with forwarding records to the inner `WitnessSink`, and
every `LedgerWitnessRecord` already carries a `timestamp_ns` field that
can serve as the arrival clock with no new parameter threaded through
`WitnessSink::emit_batch`. The fix is therefore a port of the *pattern*
(size-or-timeout, caller-driven deadline, no internal clock) directly
into `witness_signing.rs`'s existing pending-span bookkeeping, not a new
dependency on the sibling crate.

## Hypothesis

```text
Given a stream of ledger witness records (timestamp_ns as the arrival
clock) closed into signed BatchTail spans of batch_size=32, under three
load regimes — target Poisson (2000 records/s), light Poisson
(50 records/s), and bursty on/off Poisson (1500 records/s for 100ms,
silent for 400ms, repeating) —

when each record's signature-availability latency is measured as (the
span-close decision time under real arrival timing, plus the real
measured Ed25519 span-sign wall time) minus the record's arrival time,
comparing baseline BatchTail{batch_size:32} (no timeout) against
candidate_a BatchTailTimeout{batch_size:32, max_wait_ns:50ms},

then candidate_a's p99 latency should stay within a fixed bound of 70ms
(the 50ms timeout plus a fixed, not-tuned-post-hoc 20ms slack) at every
tested regime, while baseline's p99 latency should exceed 2x candidate_a's
p99 at the light-load regime — demonstrating the unbounded-tail failure
mode this ADR closes,

subject to: every closed span verifying under verify_signed_chain (100%),
and candidate_a's amortized signing cost at the target-load regime
staying within 2x of baseline's amortized cost at that same regime (the
timeout safety net must not destroy most of the amortization benefit
when load is sufficient to fill spans anyway).
```

Acceptance thresholds, fixed before this run:

1. 100% of closed spans verify (`verify_signed_chain` succeeds), every
   regime and strategy.
2. candidate_a's p99 latency ≤ 70ms at all three regimes.
3. At light load, baseline's p99 latency > 2× candidate_a's p99 at the
   same regime.
4. At target load, candidate_a's amortized signing cost (ns/record) ≤ 2×
   baseline's amortized cost at the same regime.

## Decision

1. Add `SigningStrategy::BatchTailTimeout { batch_size: usize, max_wait_ns: u64 }`
   alongside the existing `PerRecord` and `BatchTail` variants (additive;
   neither existing variant's struct shape changes, so none of the ~15
   existing call sites across `witness_signing_tests.rs` and
   `examples/witness_signing_bench.rs` need updating).
2. `PendingSpan` gains `opened_at_ns: u64`, set to the first record's
   `timestamp_ns` when a new pending span opens. `emit_batch`'s
   `BatchTail`/`BatchTailTimeout` arm is unified via an or-pattern (same
   `batch_size`-reached-seals-immediately behavior for both).
3. `SignedWitnessSink::oldest_pending_arrival_ns() -> Option<u64>` exposes
   the pending span's arrival time, mirroring
   `BatchScheduler::oldest_pending_arrival_ns()`'s role: a caller derives
   its next timeout deadline as `oldest_pending_arrival_ns() + max_wait_ns`
   without the sink owning a clock.
4. `SignedWitnessSink::check_timeout(now_ns: u64) -> bool` seals the
   pending span if `BatchTailTimeout` is active and `now_ns` is at or past
   the deadline; a no-op otherwise. Unlike
   `BatchScheduler::close_on_timeout` (which trusts a discrete-event
   simulation driver to call it only once the deadline has passed),
   `check_timeout` re-checks the deadline itself — a real caller here
   polls on its own schedule (a timer tick, the next write attempt)
   rather than driving a simulation clock, so defending against an early
   call is cheap and avoids a footgun.
5. Add a new example, `witness_signing_batch_fill_latency`, a
   discrete-event simulation parallel to
   `ruvector-retrieval-receipt::bin::batch_latency`: real
   `SignedWitnessSink::emit_batch`/`check_timeout`/`seal` calls (real
   Ed25519 signs over real witness-record digests) driven by synthetic,
   seeded arrival timelines, reporting per-record availability latency
   and a fixed acceptance gate.

## Threat Model

Unchanged from ADR-347: signature authenticity and chain integrity, not
signer honesty. This ADR adds one purely operational property — a
**latency bound** — with no new cryptographic primitive and no change to
`SignedSpan`'s message layout, `verify_signed_chain`'s guarantees, or any
existing strategy's behavior. A deployment choosing `BatchTailTimeout`
trades some amortization (spans close smaller/more often under light
load — observed mean span size dropped from 32.0 to 3.5 at light load in
this run's own evidence) for a guarantee that no record waits longer than
`max_wait_ns` plus one signing operation for its signature to become
available.

The simulation models a single serialized signer with no queueing delay
for the sign operation itself — accurate here because real span-sign cost
(roughly 1.1–1.5 microseconds amortized at `batch_size=32`, per this run's
own measurements) is 4+ orders of magnitude below the shortest fill
window tested (light-load's ~32ms mean fill wait under `BatchTailTimeout`).
This would stop holding at arrival rates high enough that signing itself
becomes the bottleneck — not evaluated here, same limitation ADR-343
disclosed for the sibling crate.

## Evidence

Full methodology, raw output across 3 independent runs, and the complete
results table are in
`docs/research/nightly/2026-10-10-signed-witness-batch-fill-timeout/README.md`.
Summary of the headline result (4000 records per regime, mean of 3 runs,
`batch_size=32`, `max_wait_ns=50ms`):

| regime | strategy | p99 latency | verified |
|---|---|---:|---|
| target (2000 rec/s) | baseline (BatchTail) | 19.05ms | 100% |
| target (2000 rec/s) | candidate_a (BatchTailTimeout) | 19.05ms | 100% |
| light (50 rec/s) | baseline (BatchTail) | **761.09ms** | 100% |
| light (50 rec/s) | candidate_a (BatchTailTimeout) | **50.06ms** | 100% |
| bursty (on/off) | baseline (BatchTail) | 423.36ms | 100% |
| bursty (on/off) | candidate_a (BatchTailTimeout) | 49.67ms | 100% |

All three runs: **ACCEPT** on every acceptance threshold above.
`candidate_b` (`PerRecord`, included as a reference upper bound) measures
~0.04ms mean latency at every regime, at roughly 30–40x the per-record
signing cost of `candidate_a` at target load.

## Consequences

- **Positive:** a deployment can bound worst-case witness-signature
  latency under `BatchTail`-style amortized signing, with a measured (not
  assumed) amortization-loss cost at low write-rate. The gap ADR-347 and
  the 2026-09-16 nightly run both named is now closed by an implemented,
  tested strategy rather than left open a second time.
- **Positive:** reuses `LedgerWitnessRecord::timestamp_ns` as the arrival
  clock, so no change to `WitnessSink::emit_batch`'s signature and no new
  parameter threaded through `TransactionalLedger` — the timeout is
  entirely internal to `SignedWitnessSink`'s own bookkeeping plus one new
  caller-driven method.
- **Negative:** `BatchTailTimeout` requires a caller to pick `max_wait_ns`
  and to actually call `check_timeout` on some schedule (a background
  timer, or opportunistically before the next write) — unlike `BatchTail`,
  which needs nothing beyond `emit_batch` calls. A deployment that never
  calls `check_timeout` gets `BatchTail`'s exact unbounded behavior with
  extra unused fields.
- **Negative:** the single-serialized-signer assumption (Threat Model)
  is untested at arrival rates where signing itself would queue.
- **Neutral:** no change to `PerRecord`, `BatchTail`, `SignedSpan`,
  `verify_signed_chain`, or any of the 16 pre-existing tests in
  `witness_signing_tests.rs` (all still pass unchanged).

## Alternatives Considered

- **Give `BatchTail` itself an optional `max_wait_ns` field** instead of a
  new variant: rejected because it would break every existing struct-literal
  call site (`BatchTail { batch_size: N }`, used ~15 times across tests and
  the existing benchmark) for a field most callers don't need — a new
  variant is purely additive.
- **Have `emit_batch` take an explicit `now_ns` parameter** instead of
  deriving arrival time from `LedgerWitnessRecord::timestamp_ns`: rejected
  because it would change `WitnessSink::emit_batch`'s trait signature
  (implemented by `MemoryWitnessLog` and any future sink), and
  `timestamp_ns` is already present on every record for exactly this
  purpose.
- **An internal background timer/task inside `SignedWitnessSink`** that
  calls `check_timeout` on its own: rejected to keep the crate
  synchronous and dependency-free (no async runtime, no thread spawned by
  a library type) — matching `ruvector-retrieval-receipt::batch_fill`'s
  explicit design choice to keep its scheduler clock-free and
  caller-driven.
- **BLS aggregate signatures** (ADR-340's and ADR-343's own named
  follow-up, restated here): would let signatures be aggregated after the
  fact without a fill-timeout at all. Not implemented here — still
  requires a pairing-friendly curve dependency not currently in the
  workspace, and orthogonal to the specific `BatchTail` gap this ADR
  closes.

## Implementation Plan

1. `witness_signing.rs`: `SigningStrategy::BatchTailTimeout`, `PendingSpan::opened_at_ns`,
   `oldest_pending_arrival_ns`, `check_timeout`, and the unified
   `BatchTail | BatchTailTimeout` match arm in `emit_batch`. Zero-batch-size
   validation extended to cover the new variant.
2. `witness_signing_tests.rs`: 9 new unit tests (scheduling-only, built via
   direct `emit_batch` calls with a `minimal_record` helper rather than
   through `TransactionalLedger`, since the ledger's real wall-clock
   timestamps aren't deterministic) plus extension of the existing
   `STRATEGIES` table-driven tests to cover `BatchTailTimeout`'s honest-chain
   round trip.
3. `examples/witness_signing_batch_fill_latency.rs`: Poisson/bursty arrival
   generators (seeded xorshift, same construction as
   `ruvector-retrieval-receipt::bin::batch_latency`), a discrete-event loop
   driving real `SignedWitnessSink` calls, hash-chain-correct synthetic
   record construction (`prev_hash`/`record_hash`/evidence-grade flags all
   consistent, so `verify_chain` accepts them), and an acceptance section.

No changes to `ledger.rs`, `ops.rs`, or `SigningStrategy::{PerRecord,BatchTail}`'s
existing behavior. All 76 pre-existing tests in `ruvector-agent-memory`
(53 `witness_signing` unit tests after the 9 additions, plus the crate's
other suites) pass unchanged.

## API Shape

```rust
pub enum SigningStrategy {
    PerRecord,
    BatchTail { batch_size: usize },
    BatchTailTimeout { batch_size: usize, max_wait_ns: u64 },
}

impl<S: WitnessSink> SignedWitnessSink<S> {
    pub fn oldest_pending_arrival_ns(&self) -> Option<u64>;
    pub fn check_timeout(&mut self, now_ns: u64) -> bool;
    // unchanged: new, from_keypair, public_key, spans, inner, anchor,
    // unsigned_pending, seal
}
```

`SignedWitnessSink` owns no clock and performs no background work — a
caller supplies `now_ns` (wall-clock or any monotonic source) and is
responsible for calling `check_timeout` no earlier than
`oldest_pending_arrival_ns() + max_wait_ns`, though unlike
`BatchScheduler::close_on_timeout`, calling it early is harmless (it
simply returns `false`). This keeps the crate's only
async/real-time dependency at the call site.

## Feature Flags

None. `BatchTailTimeout` is unconditionally compiled, matching
`PerRecord`/`BatchTail`'s existing unconditional-compilation posture.

## Benchmark Evidence

- **Command:** `cargo run --release -p ruvector-agent-memory --example witness_signing_batch_fill_latency`
- **Hardware/toolchain:** Linux x86_64, 4 vCPUs, rustc 1.97.0, cargo
  1.97.0, release profile. See the paired nightly report for the full
  per-run table.
- **Repetitions:** 3 full process runs; every acceptance threshold held
  in all 3.

## Security

- No new cryptographic primitive: `BatchTailTimeout` reuses
  `SignedWitnessSink`'s existing Ed25519 signing path (`close`) unchanged;
  it only changes *when* `close` is called.
- No new dependency: the new example adds a binary target under
  `examples/`, not a crate dependency.
- The simulation's arrival-time RNG is a plain xorshift for reproducible
  *timing*, not a security-relevant random source — the signing keypair
  still comes from `Ed25519Keypair`, unchanged from ADR-347.

## Governance

Experimental, matching ADR-347's posture: not wired into any default
`TransactionalLedger` or `SignedWitnessSink` construction path in this
repository. A promotion decision for `BatchTailTimeout` at a specific
`max_wait_ns`, or for wiring a background `check_timeout` caller into a
production deployment, requires benchmark evidence against that
deployment's actual write-rate distribution — this run's regimes remain
synthetic Poisson/bursty approximations, as ADR-343's Open Questions item
1 anticipated.

## Failure Modes

- **Nobody calls `check_timeout`:** `BatchTailTimeout` degrades silently
  to `BatchTail`'s exact unbounded behavior — not a correctness bug (every
  span that does close still verifies), but a deployment-integration
  failure mode worth flagging prominently in the type's docs (done).
- **`max_wait_ns` chosen too small for the deployment's actual write
  rate:** degrades toward near-`PerRecord` amortization (mean span size
  dropped from 32.0 to 3.5 at light load in this run), without becoming
  *incorrect*.
- **`max_wait_ns` chosen too large for the deployment's latency SLA:** the
  mirror-image misconfiguration — the bound is only as good as the
  timeout value an operator actually chooses and actually enforces via
  `check_timeout`.
- **Simulation-boundary flush:** as in ADR-343, the very last partial span
  in a finite simulation run closes via `seal()` at the final arrival's
  timestamp rather than its own timeout — disclosed, not corrected.
- **Signer becomes the bottleneck at extreme arrival rates:** not
  modeled; see Threat Model.

## Migration

None — purely additive. No existing type, function, or test is modified;
`SigningStrategy::{PerRecord,BatchTail}` behave exactly as before.

## Rollback

Remove the `BatchTailTimeout` variant, `PendingSpan::opened_at_ns`,
`oldest_pending_arrival_ns`, `check_timeout`, the unified match arm (revert
to the original `BatchTail`-only arm), the 9 new tests in
`witness_signing_tests.rs`, and
`examples/witness_signing_batch_fill_latency.rs`. No other code in the
repository references these additions.

## Rejection Criteria (Not Yet Triggered)

Production adoption of `BatchTailTimeout` at any specific `max_wait_ns`
should be rejected if: a target deployment's real write-rate distribution
produces a timeout-hit rate that destroys amortization below an
acceptable cost threshold; the single-serialized-signer assumption is
invalidated by the deployment's actual write rate relative to real
signing throughput; a production-representative benchmark (this run's
workload remains synthetic) fails to reproduce the bound; or no caller in
the actual deployment path ever invokes `check_timeout` (see Failure
Modes), making the feature inert in practice. None of these were
evaluated against a real deployment in this run.

## Open Questions

1. What `max_wait_ns` values are appropriate for real agent-memory write
   traffic shapes, as opposed to this run's synthetic Poisson/bursty
   approximations? Requires production traffic traces, not available to
   this run — the same open question ADR-343 raised for this exact
   follow-up, now still open after closing the implementation half of it.
2. Should `TransactionalLedger` (or a wrapper around it) drive
   `check_timeout` automatically via a background scheduler, or should
   that remain the integrator's responsibility? This run implements only
   the caller-driven primitive, not an integration.
3. Does wiring `SignedWitnessSink::check_timeout` through
   `witnessed_compaction::compact_witnessed` (2026-09-16's Next Research
   item #6, still open) interact with this timeout — e.g. does a
   compaction event make a reasonable `check_timeout` call site?
