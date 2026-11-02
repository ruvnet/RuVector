# ADR-355: Bounded Batch-Fill Timeout for `SignedWitnessSink::BatchTail`

## Status

Accepted. Non-breaking within `ruvector-agent-memory` (0.1.x, pre-1.0; the
only in-tree consumers of `SigningStrategy::BatchTail`'s struct literal are
this crate's own tests and benchmark example, both updated in this change).

## Context

The 2026-09-16 nightly research run (no ADR of its own; see
`docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`)
built `SignedWitnessSink` to close the ADR-134 §9 / ADR-347 `WitnessSigner`
gap in `ruvector-agent-memory`'s TARL witness ledger. Its `BatchTail`
strategy amortizes one Ed25519 signature over `batch_size` records for
throughput, but that run's own Next Research item 1 named the gap this ADR
closes:

> A wall-clock batch-fill timeout for `BatchTail` (reusing the
> `BatchFillPolicy`/`BatchScheduler` methodology already built and measured
> in `ruvector-retrieval-receipt::batch_fill`, without adding that crate as
> a dependency — port the pattern, not the code), so a deployment gets a
> bounded worst-case signature-availability latency instead of "whenever
> the batch happens to fill."

Concretely: under `BatchTail { batch_size: 64, max_wait: None }` (the only
option before this change), a deployment whose arrival rate is slower than
64 records — the common case for an interactive agent-memory workload, not
a bulk-ingest one — leaves every pending record unsigned indefinitely. The
sink's own `unsigned_pending()` reports *how many* records are unsigned but
nothing reports *for how long*, and nothing closes them short of a caller
remembering to call `seal()` (which, per the existing module docs, is only
reachable after consuming the `SignedWitnessSink` out of its
`TransactionalLedger`, i.e. at shutdown).

## Hypothesis

```text
Given a SignedWitnessSink<S> configured with
SigningStrategy::BatchTail { batch_size, max_wait: Some(d) },
fed a trickle of records strictly fewer than batch_size,

when a caller polls SignedWitnessSink::seal_if_overdue_now() on a fixed
real-clock tick until it returns true,

then the oldest pending record's measured wall-clock age at the moment it
is signed is bounded by d plus the polling tick's scheduling slack,

subject to: the resulting signed chain still verifies under
verify_signed_chain with no SignedChainError, and the existing size-
triggered BatchTail path (max_wait: None, or a burst that reaches
batch_size before any timeout matters) is measurably unaffected.
```

## Decision

1. `SigningStrategy::BatchTail` (`crates/ruvector-agent-memory/src/witness_signing.rs`)
   gains a second field, `max_wait: Option<Duration>`. `None` preserves the
   exact prior behavior (fixed-size-only, unbounded wait) byte-for-byte —
   this is a pure addition, not a behavior change, for every existing call
   site once updated to pass `max_wait: None`.
2. `PendingSpan` records `opened_at: Instant` when the first record of a
   new pending span arrives.
3. Three new public methods on `SignedWitnessSink`:
   - `pending_deadline() -> Option<Instant>` — `opened_at + max_wait` when
     a span is pending and `max_wait` is set; `None` otherwise. Mirrors
     `BatchScheduler::oldest_pending_arrival_ns`'s "next deadline" contract
     from the sibling `ruvector-retrieval-receipt::batch_fill` module.
   - `seal_if_overdue(now: Instant) -> bool` — seals the pending span iff
     `now` is at or past its deadline; otherwise a no-op. Does not itself
     re-check the deadline against a live clock (mirrors
     `BatchScheduler::close_on_timeout`'s documented contract: the caller
     controls exactly when the timeout fires), which keeps it
     deterministically testable with `Instant` arithmetic and no real
     sleeps in the unit tests.
   - `seal_if_overdue_now() -> bool` — `seal_if_overdue(Instant::now())`,
     the convenience a real caller actually calls on a timer.
4. The sink never arms a timer itself. The bound only holds if a
   deployment actually drives `seal_if_overdue_now()` periodically — a
   maintenance tick, not a passive guarantee. This is documented explicitly
   in the module's "Limits" section as a new bullet, consistent with that
   section's existing style of naming what is *not* guaranteed.
5. No change to the signed message format, the span-chaining scheme, or
   `verify_signed_chain`: a timeout-closed span is signed with
   `SignPurpose::BatchTail`, identical to a size-closed span, and verifies
   identically. This ADR adds a second trigger for an existing state
   transition; it does not add new cryptographic surface.

## Evidence

- `cargo test -p ruvector-agent-memory --lib`: 50 passed, 0 failed (6 new
  tests added: `no_max_wait_means_no_deadline_ever`,
  `max_wait_seals_at_but_not_before_the_deadline`,
  `seal_if_overdue_with_nothing_pending_is_a_noop`,
  `size_trigger_preempts_the_timeout_bound`,
  `size_closed_then_timeout_closed_span_both_verify`, plus the existing
  44 tests updated for the new field and confirmed still green).
- `cargo clippy -p ruvector-agent-memory --all-targets`: no warnings.
- `cargo run --release -p ruvector-agent-memory --example
  witness_signing_timeout_bench` (new benchmark, raw output in the nightly
  research document): baseline trickle of 8 records under
  `max_wait: None` confirms `unsigned_pending=16`,
  `pending_deadline=None` — unbounded, as documented. The same trickle
  under `max_wait: Some(50ms)`, polled every 5ms, measured a real
  signed-at age of ~51.5–52.3ms across 4 runs (bound: 60ms, i.e.
  `max_wait` + 2 poll ticks) — PASS every run, no flakes observed. The
  size-triggered control path (64-record burst) closed in ~0.1–0.2ms,
  1 span, unaffected.
- `cargo run --release -p ruvector-agent-memory --example
  witness_signing_bench` (pre-existing benchmark, unmodified logic):
  re-run after this change, `correctness=PASS` for every variant — no
  regression to the already-measured signing cost or amortization ratio.

## Consequences

- A deployment that wants the latency bound in practice must schedule
  `seal_if_overdue_now()` itself (see "RuFlo Integration" in the nightly
  research document for a concrete shape). This ADR provides the
  mechanism, not an automatic scheduler — consistent with this crate's
  existing no-hidden-concurrency style (`TransactionalLedger` is already
  single-threaded by construction, per the 2026-09-16 run's Production
  Path item 3).
- `max_wait: Option<Duration>` makes `SigningStrategy::BatchTail`'s struct
  literal non-exhaustive-breaking for any out-of-tree consumer. None exist
  in this repository today (`rvf-types`'s only ed25519 consumer path is
  internal to this crate); a future out-of-tree break is possible but
  unaddressed by this ADR, matching how the crate is versioned (0.1.x).

## Alternatives

- **Spawn a background thread inside `SignedWitnessSink` that calls
  `seal()` on a timer.** Rejected: `SignedWitnessSink` and
  `TransactionalLedger` are deliberately single-threaded (`&mut self`
  throughout); adding internal concurrency here would be a larger, riskier
  change than the problem calls for, and it duplicates whatever scheduling
  a host application (or ruFlo) already does for its own maintenance
  ticks.
- **A `tokio`/async timeout future.** Rejected: this crate has no async
  dependency today and the existing module is synchronous throughout;
  adding one feature-gated async API surface for a single timeout check is
  disproportionate.
- **Depend on `ruvector-retrieval-receipt::batch_fill` directly instead of
  porting the concept.** Rejected per the originating Next Research item's
  own instruction ("port the pattern, not the code") and because that
  module's `BatchScheduler` is arrival-indexed (`Vec<PendingMember>`) for a
  query-stream simulation, not record-sequence-indexed like
  `SignedWitnessSink`'s `PendingSpan` — the two data shapes differ enough
  that a shared dependency would need its own abstraction layer for no
  measured benefit here.

## Implementation Plan

Done in this change: the `witness_signing.rs` field/method additions, the
`witness_signing_tests.rs` regression + new tests, the
`witness_signing_bench.rs` literal updates (no logic change), and the new
`witness_signing_timeout_bench.rs` example.

## API Shape

```rust
pub enum SigningStrategy {
    PerRecord,
    BatchTail { batch_size: usize, max_wait: Option<Duration> },
}

impl<S: WitnessSink> SignedWitnessSink<S> {
    pub fn pending_deadline(&self) -> Option<Instant>;
    pub fn seal_if_overdue(&mut self, now: Instant) -> bool;
    pub fn seal_if_overdue_now(&mut self) -> bool;
}
```

## Feature Flags

None added; this is unconditional crate API, matching the rest of
`witness_signing.rs`.

## Benchmark Evidence

See "Evidence" above and the full raw benchmark output in
`docs/research/nightly/2026-11-02-witness-batch-fill-timeout/README.md`.

## Security

No new cryptographic surface (see Decision item 5). The timeout trigger
produces exactly the same `SignPurpose::BatchTail`-tagged span a size
trigger would; `verify_signed_chain` cannot distinguish how a span was
closed, by design — that is not part of what it verifies or needs to.

## Governance

None of this run's changes touch key management, access control, or MCP
authority surfaces.

## Failure Modes

- **Caller never polls `seal_if_overdue_now()`.** The bound silently does
  not hold — behaves exactly like `max_wait: None`. No new failure mode
  versus before this change; it is the same "crash/omission before seal"
  limitation the module already documents, now with one more way to
  trigger it (forgetting to schedule a tick, not just forgetting to call
  `seal()` at shutdown).
- **Polling tick is coarser than `max_wait`.** The measured bound becomes
  `max_wait` + up to one tick, which is exactly what this run's benchmark
  measures and reports rather than hides (see `bound = max_wait + 2 *
  poll_tick` in the benchmark's acceptance check).
- **Clock changes mid-wait (e.g. system suspend/resume).** `Instant` on
  the platforms this crate targets is monotonic and not affected by
  wall-clock adjustments, but a suspend/resume cycle can still make a
  deadline appear to arrive "late" relative to real elapsed time if the
  polling loop itself was suspended — unaddressed here, inherited from
  `std::time::Instant`'s own platform guarantees.

## Migration

None required: every existing in-tree `BatchTail { batch_size: N }` call
site is updated to `BatchTail { batch_size: N, max_wait: None }` in this
same change, which is behaviorally identical to before.

## Rollback

Revert this commit; `max_wait` and the three new methods are additive and
unused by anything outside this change.

## Rejection Criteria

This hypothesis would have been rejected if any of:

- The hybrid variant's measured unsigned age exceeded its declared bound
  (`max_wait` + 2 poll ticks) in any of the repeated runs — it did not, in
  4/4 runs.
- The timeout-sealed span failed `verify_signed_chain` — it did not.
- The size-triggered control path changed its span count, or regressed its
  measured closing time versus the pre-existing benchmark's numbers for
  the same batch size — it did not (1 span, sub-millisecond, as before).
- Any pre-existing test in the crate regressed — none did (50/50 pass,
  including 44 pre-existing tests updated only for the new field, not for
  behavior).

## Open Questions

- Should a convenience helper spawn the polling loop itself behind an
  optional, explicitly-opt-in feature (e.g. a `std::thread`-based ticker)?
  Deferred — this ADR deliberately keeps the mechanism synchronous and
  leaves scheduling to the host, matching the crate's existing style.
- The exact production polling tick is a deployment choice this ADR does
  not prescribe; the benchmark's 5ms tick against a 50ms `max_wait` is
  illustrative, not a recommended default.
