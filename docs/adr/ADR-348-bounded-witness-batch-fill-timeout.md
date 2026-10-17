# ADR-348: A Bounded Wall-Clock Fill Timeout for `WitnessSigner`'s `BatchTail`

## Status

Accepted. New, opt-in `SigningStrategy` variant
(`ruvector-agent-memory::witness_signing::SigningStrategy::BatchTailTimeout`)
alongside the existing `PerRecord`/`BatchTail`, non-breaking and
non-default. Directly answers ADR-347's Open Question 1 ("Should
`BatchTail` gain a wall-clock fill timeout, and at what default?") and its
"Alternatives Considered" entry "Wall-clock `BatchTail` fill timeout.
Deferred.", and implements the 2026-09-16 nightly's "Next Research" item 1.

## Context

ADR-347 added `SignedWitnessSink<S: WitnessSink>`, a decorator that signs
spans of TARL-ledger witness records either per-record (`PerRecord`) or in
amortized batches (`BatchTail { batch_size }`). `BatchTail`'s own doc
comment and ADR-347's Limits section are explicit about what it does not
bound: "up to `batch_size - 1` committed records are unauthenticated until
the batch closes or `seal()` runs. No wall-clock fill timeout in this
pass." If arrivals stop — a quiet agent-memory ledger, a crash, a
deployment with bursty traffic — a partially-filled batch stays
unauthenticated indefinitely. `verify_signed_chain` already reports this
state correctly (`SignedChainError::UnsignedTail`), but nothing in the
sink itself ever resolves it short of either the batch filling or a
caller explicitly calling `seal()` — and nothing tells the caller *when*
to call it.

`ruvector-retrieval-receipt::batch_fill` solved the structurally identical
problem for signed retrieval-receipt batches: `BatchFillPolicy::hybrid(max_members,
max_wait_ns)` and `BatchScheduler` close a batch at `max_members` or after
`max_wait_ns` since the oldest pending member, whichever comes first, with
the caller responsible for driving the timeout check
(`close_on_timeout`). That module was measured and accepted (ADR-343) but
is architecturally a standalone scheduler with no cryptographic state —
it cannot be reused as-is by `SignedWitnessSink`, which commits spans
through the `WitnessSink` trait and signs inline. This ADR ports the
*policy*, not the code, into `witness_signing.rs` directly, keeping
`ruvector-agent-memory` free of a new intra-workspace dependency edge
(consistent with ADR-347's own rejection of depending on
`ruvector-retrieval-receipt`).

## Hypothesis

```text
Given a SignedWitnessSink<MemoryWitnessLog> under SigningStrategy::BatchTail,
a partially-filled pending span's signature-availability latency is
unbounded under sparse/halted arrivals (no mechanism exists to close it
except reaching batch_size or an explicit seal() call).

When SigningStrategy::BatchTailTimeout { batch_size, max_wait_ns } is used
instead, and a caller drives check_timeout()/check_timeout_at() at least
once per polling interval,

Then the worst-case signature-availability latency for any record is
bounded by max_wait_ns plus the polling interval,

Subject to: dense-arrival throughput and signature-amortization staying
within 10% of plain BatchTail at the same batch_size, and every existing
SignedWitnessSink/verify_signed_chain guarantee (coverage, integrity,
unsigned-tail detection, rollback-anchor detection) continuing to hold.
```

## Decision

1. Add `SigningStrategy::BatchTailTimeout { batch_size: usize, max_wait_ns: u64 }`.
   Both fields must be `>= 1`; construction returns
   `WitnessSignerError::ZeroBatchSize` / `::ZeroMaxWait` otherwise (no new
   panics).
2. `PendingSpan` gains `opened_at_ns: u64` — the wall-clock time (relative
   to the sink's own `clock_start: Instant`, so it is monotonic and
   immune to clock-skew adjustments) at which the span's *first* record
   was forwarded.
3. New methods on `SignedWitnessSink<S>`:
   - `emit_batch_at(&mut self, records, now_ns: u64) -> Result<(), LedgerError>`
     — the real logic behind `WitnessSink::emit_batch`, with `now_ns`
     supplied explicitly instead of read from the wall clock. The trait
     method is now a one-line wrapper (`self.emit_batch_at(records,
     self.wall_clock_now_ns())`), preserving the existing public API and
     every other `WitnessSink` implementor untouched.
   - `next_timeout_deadline_ns(&self) -> Option<u64>` — `None` unless
     `BatchTailTimeout` has a span pending; otherwise
     `pending.opened_at_ns + max_wait_ns`.
   - `check_timeout_at(&mut self, now_ns: u64) -> bool` — force-seals the
     pending span (calling the existing `seal()`) iff `now_ns` has
     reached the deadline; otherwise a no-op. Deterministic: tests drive
     exact boundary cases without sleeping.
   - `check_timeout(&mut self) -> bool` — the real-wall-clock-driven
     production entry point for a periodic maintenance tick.
   - `strategy(&self) -> SigningStrategy` — a small introspection
     accessor used by the tests and available to any caller.
4. **Ordering, not just presence, matters.** `emit_batch_at`'s
   `BatchTailTimeout` arm calls `check_timeout_at(now_ns)` **before**
   merging the incoming records into the (possibly stale) pending span.
   Checking after merging would silently fold a late-arriving record into
   a span whose nominal availability deadline had already passed,
   stretching the record it was supposed to protect past the configured
   bound instead of bounding it — the first version of this patch had
   exactly that bug, caught by
   `batch_tail_timeout_opportunistic_seal_on_a_late_arrival`.
5. Nothing inside `SignedWitnessSink` spawns a timer or background thread.
   The bound is a contract with the caller: call `check_timeout` (or
   `check_timeout_at`) at least once per polling interval, or the only
   guarantee left is the pre-existing opportunistic one (a fresh arrival
   re-checks the deadline against the stale span before merging — see
   decision 4). A deployment with no further arrivals and no periodic
   tick is back to `BatchTail`'s original unbounded-wait behavior; this
   is a caller-discipline requirement, not something the type system
   enforces, and is recorded under Limits.
6. Not feature-gated. `BatchTailTimeout` is additive; `PerRecord` and
   `BatchTail` are unchanged bit-for-bit (same spans, same signatures,
   confirmed by the full pre-existing test suite staying green).

## Evidence

`witness_signing_tests.rs` grew from 17 to 27 witness-signing tests (45
total in the crate), all passing, `cargo clippy --all-targets -p
ruvector-agent-memory` clean. New coverage:

- Construction errors (`ZeroMaxWait`, `ZeroBatchSize` for the new variant).
- `plain_batch_tail_has_no_timeout_bound_at_any_elapsed_time` — a
  structural regression guard for the problem this ADR fixes:
  `next_timeout_deadline_ns()` is `None` under `BatchTail` at
  `now_ns = u64::MAX`, i.e. literally no amount of elapsed time bounds it.
- `batch_tail_timeout_behaves_like_batch_tail_under_dense_arrivals` —
  identical span count and pending state to plain `BatchTail` when
  `now_ns` never reaches the deadline.
- Deadline-boundary tests (`does_not_force_seal_before`,
  `force_seals_exactly_at`, `count_close_wins_when_it_happens_first`,
  `opportunistic_seal_on_a_late_arrival`).
- `batch_tail_timeout_bounds_worst_case_pending_age_under_a_poller` — a
  200-trial, xorshift64-seeded property test with randomized
  `batch_size` and randomized sparse/bursty inter-arrival gaps (0 to
  `4*max_wait_ns`) and a fixed-cadence poller; asserts the observed
  pending age never exceeds `max_wait_ns + poll_tick` and that the
  resulting signed chain still verifies end to end for every trial.
- `honest_chain_verifies_under_batch_tail_timeout_with_forced_partial_seals`
  — a timeout-forced partial seal mixed with a count-close still produces
  a chain `verify_signed_chain` accepts.

Benchmark (`cargo run --release -p ruvector-agent-memory --example
witness_signing_timeout_bench`; three runs on a shared, containerized
workstation, so treat figures as indicative, not a guaranteed SLA):

**Measurement 1 — dense-burst overhead** (N_ENTRIES=20,000 add+accept
pairs, 40,000 witness records, `batch_size=64`, no idle gaps; mean
per-op latency over three runs):

| Variant | Mean latency/op | Throughput | Signatures |
|---|---|---|---|
| `BatchTail{64}` | 3.39–3.64µs | 272k–292k ops/s | 625 |
| `BatchTailTimeout{64, 5ms}` | 3.34–3.44µs | 288k–297k ops/s | 625 |

Overhead across three runs: -1.39%, -0.19%, -7.11% (all **negative**,
i.e. no measurable regression — within run-to-run noise on a shared
container, well inside the ±10% gate). Signature count identical in
every run (the timeout path never fires under dense arrivals, as
expected: count-close always wins first). **Dense-burst overhead gate:
PASS** on all three runs.

**Measurement 2 — sparse-arrival worst-case latency** (10 add+accept
pairs = 20 records, `batch_size=64` so far under the count threshold,
then a real 15ms `std::thread::sleep` idle gap, `max_wait_ns=5ms`,
`check_timeout` polled every 1ms):

| Variant | Pending before gap | Pending after gap/poll | Has timeout bound | Sealed after |
|---|---|---|---|---|
| `BatchTail{64}` | 20 | 20 (unchanged) | `false` | never |
| `BatchTailTimeout{64, 5ms}` | 20 | 0 | `true` | 5.53–5.65ms |

`BatchTail` leaves all 20 records permanently unauthenticated through the
entire 15ms gap — confirmed structurally, not just empirically:
`next_timeout_deadline_ns()` has no concept of a deadline for it at all.
`BatchTailTimeout` force-seals within 5.53–5.65ms in every run, inside
the `max_wait_ns (5ms) + poll_interval (1ms)` = 6ms bound derived in the
Decision section. **Sparse-arrival bounded-latency gate: PASS** on all
three runs. **Overall acceptance: ACCEPT** on all three runs.

## Consequences

- A production `ruvector-agent-memory` deployment can now choose a
  bounded worst-case signature-availability latency for `BatchTail`-style
  amortized signing, trading a small, measured amount of dense-arrival
  throughput for a hard ceiling on how long any record can sit
  unauthenticated during a lull — directly useful wherever a reader
  (ADR-342's periodic anchoring, a ruFlo staleness-alerting workflow, a
  cross-repo witness consumer) needs a latency SLA rather than a "signs
  eventually" promise.
- Purely additive. `PerRecord`, `BatchTail`, every other `WitnessSink`
  implementor, and every existing caller of
  `SignedWitnessSink`/`verify_signed_chain` are untouched.
- The bound is caller-enforced, not self-enforced: `SignedWitnessSink`
  still does not own a thread, a timer, or an async runtime. Production
  use requires wiring an actual periodic tick (see ruFlo Integration
  below) — shipping the policy does not by itself ship the tick.

## Alternatives Considered

- **Depend on `ruvector-retrieval-receipt::batch_fill` directly.**
  Rejected for the same reason ADR-347 rejected depending on that crate's
  `Issuer`/`BatchAnchor`: a new intra-workspace dependency edge for a
  dozen lines of policy logic that needs to be adapted to
  `SignedWitnessSink`'s commit-through-a-trait shape anyway (`batch_fill`
  has no concept of signing or of a `WitnessSink`).
- **A background thread inside `SignedWitnessSink` that wakes on its own
  timer.** Rejected: `ruvector-agent-memory` has no async runtime
  dependency today, and a self-spawned thread per sink is a surprising
  cost for a library type embedded in arbitrary host processes (tests,
  WASM targets, embedded callers). Caller-driven polling composes with
  whatever scheduling primitive the host already has (a ruFlo tick, a
  Tokio interval, a cron).
  checked second against the wall clock with no opportunistic help from
  arrivals. Rejected as an unnecessary complication for this pass: the
  deadline is a simple linear function of `opened_at_ns`, computable on
  demand via `next_timeout_deadline_ns`; an event-queue abstraction can be
  layered on top by a caller that wants one, without this crate committing
  to its shape.
- **Reset the deadline on every new arrival (sliding window), instead of
  anchoring it to the span's first record.** Rejected: a sliding window
  can be starved indefinitely by a steady trickle of arrivals each
  resetting the clock, which defeats the whole point of a *worst-case*
  bound. Anchoring to `opened_at_ns` (the oldest pending record) is what
  `ruvector-retrieval-receipt::batch_fill` already validated and is what
  the property test asserts.

## API Shape

```rust
pub enum SigningStrategy {
    PerRecord,
    BatchTail { batch_size: usize },
    BatchTailTimeout { batch_size: usize, max_wait_ns: u64 }, // new
}

pub enum WitnessSignerError { ZeroBatchSize, ZeroMaxWait /* new */ }

impl<S: WitnessSink> SignedWitnessSink<S> {
    pub fn strategy(&self) -> SigningStrategy;                       // new
    pub fn emit_batch_at(&mut self, records: &[LedgerWitnessRecord],
        now_ns: u64) -> Result<(), LedgerError>;                     // new
    pub fn next_timeout_deadline_ns(&self) -> Option<u64>;           // new
    pub fn check_timeout_at(&mut self, now_ns: u64) -> bool;         // new
    pub fn check_timeout(&mut self) -> bool;                         // new
    // unchanged: new, from_keypair, public_key, spans, anchor,
    // unsigned_pending, seal, inner
}
```

## Feature Flags

None. `BatchTailTimeout` is a plain enum variant selected at construction
time, like `PerRecord`/`BatchTail`; no Cargo feature gate is introduced or
required.

## Security

- No new cryptographic surface: `check_timeout_at`'s force-seal calls the
  existing `seal()` → `close()` path, producing a normal `BatchTail`-purpose
  signed span over whatever records are pending, hashed and signed
  exactly as a count-triggered seal would be. A verifier cannot
  distinguish a count-closed span from a timeout-closed one (by design —
  `SignPurpose::BatchTail` is shared), so this adds no new forgery
  surface to `verify_signed_chain`.
- `check_timeout`'s wall clock is `Instant`-based (monotonic, immune to
  system-clock adjustment), so an attacker who can move the system clock
  cannot shrink or stretch the bound; they would need to control process
  scheduling itself, at which point far stronger attacks are already
  available.
- Does not change `ops.rs`'s "no witness, no mutation" invariant: the
  inner sink still commits before anything is signed or sealed.

## Governance

- No change to who may construct a `SignedWitnessSink` or hold its
  signing key; this is purely a scheduling policy for an already-trusted
  signer.
- A deployment choosing `BatchTailTimeout` takes on an operational
  obligation (wire a periodic `check_timeout` tick) that `BatchTail` does
  not have. This should be called out in any deployment runbook that
  adopts it — an unmet obligation silently degrades to `BatchTail`'s
  original behavior rather than failing loudly, which is a known,
  documented Limit, not a bug.

## Failure Modes

- **No periodic tick wired up, and arrivals also stop.** Degrades exactly
  to `BatchTail`: the pending span stays open indefinitely. Detectable
  operationally via `unsigned_pending() > 0` persisting across an
  expected polling interval, or via `next_timeout_deadline_ns()` being
  `Some(d)` with `d` far in the past relative to the real clock.
- **Polling interval much larger than `max_wait_ns`.** The effective bound
  is `max_wait_ns + poll_interval`, not `max_wait_ns` alone — this is
  documented on `check_timeout`/`next_timeout_deadline_ns` and asserted
  by the property test's tolerance term. A caller choosing
  `max_wait_ns` should pick a polling interval materially smaller than it.
- **Process crash with a span pending.** Identical to `BatchTail`'s
  existing documented limit ("Crash before seal"): the pending tail is
  in-memory only and is lost; `verify_signed_chain` fails closed with
  `UnsignedTail` on the surviving prefix, same as today.

## Migration

Opt-in only. An existing `BatchTail { batch_size }` caller migrates by
constructing `BatchTailTimeout { batch_size, max_wait_ns }` instead and
adding a periodic `sink.check_timeout()` call (e.g. from a ruFlo
maintenance workflow or any existing tick already present in the host
process). No data migration: the on-disk/in-memory record and span
formats are byte-for-byte identical to `BatchTail`'s.

## Rollback

Remove the `BatchTailTimeout` variant, its `WitnessSignerError::ZeroMaxWait`
arm, `PendingSpan::opened_at_ns`, `emit_batch_at`/`check_timeout_at`/
`check_timeout`/`next_timeout_deadline_ns`/`strategy`, the new tests, and
`examples/witness_signing_timeout_bench.rs`. No other caller in the
repository depends on any of this; `PerRecord`/`BatchTail` callers are
unaffected either way.

## Rejection Criteria

This change should be reverted or reworked if any of:

- The dense-burst overhead gate in
  `witness_signing_timeout_bench`(mean-latency or signature-count delta
  vs. plain `BatchTail`) exceeds 10% on a representative run, indicating
  the opportunistic per-arrival timeout check is not actually cheap.
- The property test's bound (`max_wait_ns + poll_tick`) is ever violated
  on CI hardware under realistic tick granularities, rather than only
  under the pathological `u64::MAX`-style edge cases already covered.
- A future caller needs the deadline reset-on-arrival (sliding window)
  semantics this ADR explicitly rejected — that is a different policy and
  should be its own variant, not a silent change to this one.

## Open Questions

1. What `max_wait_ns` default (if any) should `ruvector-agent-memory`'s
   higher-level ledger constructors recommend for typical agent-memory
   workloads? This ADR ships the mechanism, not a recommended default —
   left to the production-path integration below.
2. Should the opportunistic per-arrival check in decision 4 be removed in
   favor of requiring callers to always drive an explicit tick, trading a
   small latency win (sealing early on a late arrival, without waiting
   for the next poll) for a simpler mental model ("only `check_timeout`
   ever force-seals")? Kept for now because it is strictly a latency
   improvement with no correctness cost, per the property test.
3. Durable (non-`MemoryWitnessLog`) sinks need their own verifier, same
   open question ADR-347 already carries — unaffected by this ADR either
   way.
