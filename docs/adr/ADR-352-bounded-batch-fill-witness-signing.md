# ADR-352: Bounded Batch-Fill (Count-or-Timeout) Signing for `SignedWitnessSink`

## Status

**Hypothesis REJECTED; strategy retained as an opt-in with a corrected
guarantee.** Nightly research run 2026-09-29
(`docs/research/nightly/2026-09-29-bounded-batch-fill-witness-signing/`).

The pre-registered hypothesis said write-path timeout checks would bound
worst-case signature-availability latency by `max_wait`. They do not: on
the bursty benchmark the measured maximum was 37.3 ms against an 11 ms gate
(`max_wait` 10 ms + 1 ms tolerance), reproduced in 3/3 runs. What the new
strategy does measurably provide is a bound of `max_wait` plus the time
until the next write or `seal_expired()` call. It cut the maximum latency
from ~250 ms to ~37 ms. With a caller-owned 1 ms poll, the maximum was
11.2–11.8 ms. Steady-state throughput and every correctness gate were
unaffected. `BatchTailTimeout` ships as an opt-in, non-default variant, and
its documentation claims only the bound that was measured.

> **Numbering note.** The nightly brief proposed ADR-348. Other refs had
> already claimed 348–351 (348: TwinKV, PR #946; 349: `ruvbrain` on
> `main`; 350: several `claude/focused-darwin-*` branches; 351:
> `feat/ruvector-edge-*`), so this ADR uses 352. It was the first number
> with no file on any local or remote ref (`git log --all -- docs/adr/ADR-352-*`
> is empty).

## Context

ADR-347 added `SignedWitnessSink<S>` with `SigningStrategy::{PerRecord,
BatchTail { batch_size }}`. `BatchTail` signs only when `batch_size`
records have accumulated, or when the caller calls `seal()`. Under a slow
or bursty write rate, a partial batch can therefore stay unsigned for an
unbounded time. `verify_signed_chain` reports those records as
`UnsignedTail` (unauthenticated), and a crash loses the chance to sign
them. ADR-347 Failure Mode 2 and the 2026-09-16 nightly's Next Research
item 1 both name the missing wall-clock fill timeout.

`ruvector-retrieval-receipt` already solved the same problem for
retrieval receipts with `BatchFillPolicy::hybrid` / `BatchScheduler`
(ADR-343): close at N members or T since the oldest pending member,
whichever comes first. That work was measured in a discrete-event
simulation. The brief was to port the *pattern* without adding a
dependency on that crate.

## Hypothesis

Recorded verbatim from the nightly brief, before any code was written:

```
Given a SignedWitnessSink<S> configured with SigningStrategy::BatchTail{batch_size},
under a bursty/slow write workload where batches sometimes stay open indefinitely
below batch_size,
when a new SigningStrategy::BatchTailTimeout{batch_size, max_wait} variant is added
that closes and signs a partially-filled batch once max_wait wall-clock time has
elapsed since the batch opened (even if batch_size has not been reached),
then worst-case signature-availability latency (time from a record being witnessed
to a signature covering it existing) should be bounded by max_wait,
subject to: correctness of verify_signed_chain unchanged (5/5 PASS as before),
diligent-forgery rejection unchanged (still rejects), the existing tests in
ruvector-agent-memory remaining green, and steady-state throughput (fast, dense
writes that fill batches before timeout) not regressing materially relative to
plain BatchTail{batch_size}.
```

The brief says the sink has no background timer and that timeouts fire
only on a later write or explicit flush. This ADR took that as a design
constraint and did **not** relax the `max_wait` bound to fit it.

## Decision

1. Add `SigningStrategy::BatchTailTimeout { batch_size: usize, max_wait: Duration }`.
   `batch_size == 0` is rejected with `WitnessSignerError::ZeroBatchSize`.
   `max_wait == 0` is allowed and signs at the end of every `emit_batch` call.
2. Each open `PendingSpan` records `opened_at: Option<Instant>`, the time
   its oldest record was witnessed. Only `BatchTailTimeout` reads the clock,
   once per `emit_batch` call. Plain `BatchTail` stores `None` and never
   calls `Instant::now()`.
3. `emit_batch` keeps its order: check sequence contiguity, forward to the
   inner sink (witness-first), then append to the open batch, closing it at
   `batch_size`. After appending, if the open batch is at least `max_wait`
   old, it is signed. The records of the write that finds the expiry go into
   the same signature rather than opening a new batch.
4. New `pub fn seal_expired(&mut self) -> bool` is the hook for a
   caller-owned timer. It signs the open batch only if it has expired, and
   is a no-op under other strategies. New `pub fn pending_age(&self) ->
   Option<Duration>` supports monitoring.
5. A span closed by timeout is signed exactly like a size close
   (`SignPurpose::BatchTail`). The signed statement ("these records, after
   that span") does not depend on why the span closed, so the message
   format, the verifier and `verify_signed_chain` are all unchanged.
6. The code lives in a child module (`src/witness_batch_fill.rs`, loaded
   with `#[path]` from `witness_signing.rs`) so it can reach the sink's
   private state while `witness_signing.rs` stays at about 500 lines.

## Evidence

All numbers come from `cargo run --release -p ruvector-agent-memory
--example bounded_batch_fill_bench`, run 3 times on 4 vCPU, Linux
6.18.44-fc-v37 x86_64, rustc/cargo 1.94.1. Settings: `batch_size=64`,
`max_wait=10ms`. The bursty schedule uses seed `0x5eed20260929`: 80 bursts
of 1–6 add+accept pairs (554 records), with idle gaps of 2–30 ms, 51 of
which are longer than `max_wait`. Latency runs from the instant before the
`emit_batch` call to the instant after the call or poll that closed the
record's span.

| Variant (bursty) | max latency (run 1 / 2 / 3) | p99 (run 1) | p50 (run 1) | signed during run | unsigned at end (oldest age) |
|---|---|---|---|---|---|
| `BatchTail{64}` | 249.854 / 249.880 / 250.306 ms | 249.848 ms | 62.439 ms | 512/554 | 42 (47.9 ms) |
| `BatchTailTimeout{64,10ms}` (write-driven) | 37.292 / 37.366 / 37.322 ms | 37.289 ms | 16.462 ms | 543/554 | 11 (13.1 ms) |
| `BatchTailTimeout{64,10ms}` + 1 ms `seal_expired` poll (supplementary) | 11.218 / 11.849 / 11.173 ms | 11.212 ms | 10.227 ms | 554/554 | 0 |

Dense workload: 20,000 pairs, 7 interleaved reps. The median ratio of
`BatchTailTimeout{64,10ms}` to `BatchTail{64}` total time was 1.007 / 0.986
/ 0.864 across the 3 runs, and both produced 625 signatures in every rep.
The run-to-run noise (±~14%) is larger than the cost of the extra clock
read. A scratch `rustc -O` probe measured that read at ~35–46 ns, or ~2%
of a ~4 µs pair.

## Consequences

- **Positive, measured:** maximum signature-availability latency on the
  bursty workload fell ~6.7× (250 → 37 ms) with no writes added. With a
  1 ms caller poll it fell to ≤ 11.85 ms, and no record was left unsigned
  at the end of the run.
- **Positive, measured:** no measurable steady-state cost. Batches that
  fill before the deadline are byte-identical to plain `BatchTail`,
  signatures included (unit test
  `size_reached_before_timeout_is_identical_to_plain_batch_tail`).
- **Negative:** more spans on slow workloads (65 vs 9 signatures on the
  bursty schedule), which is the amortization traded for bounded latency.
  The same trade appears in ADR-343.
- **Negative (the rejection):** with write-driven checks alone, the bound
  is `max_wait + time to next write`. That is unbounded if writes stop. A
  deployment that needs a hard bound must drive `seal_expired()` from its
  own timer.

## Alternatives

1. **Background timer thread inside the sink.** Rejected for this ADR.
   The sink is a single-owner, `&mut self`, runtime-free decorator. A timer
   thread would need `Arc<Mutex<_>>` around the sink and the inner sink,
   would change `WitnessSink`'s threading contract, and would sign
   concurrently with `emit_batch`. `seal_expired()` leaves that choice to
   the caller (tokio interval, event loop tick, cron) without imposing it.
2. **Depend on `ruvector-retrieval-receipt::BatchScheduler`.** Rejected
   because the brief forbids the dependency, and because that type is a
   clock-free, `Vec`-of-members scheduler built for simulation. Here the
   "batch" is a single streaming SHA-256 `PendingSpan`, so a membership
   list would be dead weight.
3. **Close the stale batch *before* appending the new write's records.**
   Rejected. It costs one extra signature per timeout, and the new records
   would wait a further `max_wait` instead of being signed immediately.
4. **Distinct `SignPurpose::BatchTimeout`.** Rejected. It would change the
   verifier (the hypothesis requires it unchanged) and would authenticate
   *why* a span closed, which no verifier needs.
5. **Injectable clock trait (`SignedWitnessSink<S, C: Clock>`).**
   Rejected as too much API weight. Tests use a `#[cfg(test)]
   emit_batch_at(records, Instant)` / `seal_expired_at` instead.

## Implementation Plan

Done in this run:

- `crates/ruvector-agent-memory/src/witness_signing.rs`: new variant,
  constructor validation, `opened_at` on `PendingSpan`, and `emit_batch`
  routed through `emit_batch_inner` (505 lines).
- `crates/ruvector-agent-memory/src/witness_batch_fill.rs` (new):
  `push_batched`, `seal_expired`, `seal_expired_at`, `pending_age`,
  `clock_if_timed`, and test-only `emit_batch_at`.
- `crates/ruvector-agent-memory/src/witness_signing_tests.rs`: the
  `STRATEGIES` matrix now also includes `BatchTailTimeout{4, 1h}` and
  `BatchTailTimeout{8, 0}`.
- `crates/ruvector-agent-memory/src/witness_signing_timeout_tests.rs`
  (new): 8 tests.
- `crates/ruvector-agent-memory/examples/bounded_batch_fill_bench.rs`
  (new).

Not done: an MCP/ruFlo-driven poller, or any runtime integration.

## API Shape

```rust
pub enum SigningStrategy {
    PerRecord,
    BatchTail { batch_size: usize },
    BatchTailTimeout { batch_size: usize, max_wait: std::time::Duration }, // new
}

impl<S: WitnessSink> SignedWitnessSink<S> {
    pub fn seal_expired(&mut self) -> bool;              // new: caller-owned timer hook
    pub fn pending_age(&self) -> Option<Duration>;       // new: BatchTailTimeout only
    // unchanged: new, from_keypair, seal, spans, anchor, unsigned_pending, ...
}
// unchanged: verify_signed_chain, SignedSpan, SignPurpose, SignedAnchor, errors
```

Adding a variant to a public, non-`#[non_exhaustive]` enum is a
source-breaking change for downstream code that matches `SigningStrategy`
exhaustively. No such match exists in this workspace (checked with
`grep -rn "SigningStrategy::" crates/`). The crate is at 0.1.0 and
unpublished.

## Feature Flags

None. The variant is always compiled. It is opt-in by construction:
nothing changes unless a caller chooses `BatchTailTimeout`.

## Benchmark Evidence

See the Evidence section above. The full raw output of run 1 and the key
lines of runs 2 and 3 are in the nightly README. Gates were fixed in
`examples/bounded_batch_fill_bench.rs` before its first execution:

| Gate | Threshold | Measured (runs 1/2/3) | Result |
|---|---|---|---|
| A1 dense throughput | median(timeout)/median(plain) ≤ 1.10 | 1.007 / 0.986 / 0.864 | PASS |
| A2 timeout inert when batches fill | identical signature counts | 625 = 625 (all reps) | PASS |
| A3 correctness | `verify_signed_chain` OK, all dense and bursty runs | all OK | PASS |
| A4 diligent forgery | rejected on a log with timeout-closed spans | rejected | PASS |
| **B1 hypothesis bound** | write-driven max latency ≤ `max_wait` + 1 ms = 11 ms | 37.29 / 37.37 / 37.32 ms | **FAIL** |
| B2 workload validity | plain `BatchTail` max > 11 ms (the problem exists) | 249.9 / 249.9 / 250.3 ms | PASS |
| Existing tests | 45 pre-existing lib tests green | 45 + 8 new = 53/53 | PASS |
| Existing bench | `witness_signing_bench` 5/5 correctness, forgery rejected | 5/5, 2/2 | PASS |

**Verdict: REJECT.** B1 is the hypothesis's own bound, and it failed in
every run.

## Security

- No new cryptographic surface. The primitives (Ed25519 via
  `ed25519-dalek` `SigningKey`, SHA-256 via `rvf-types`), the message
  format, the domain tags and `verify_signed_chain` are unchanged.
- Witness-first is preserved: a sequence-mismatch or inner-sink refusal
  returns `Err` before any signing, and does not close a stale batch
  either (test `refused_write_neither_signs_nor_closes_the_stale_batch`).
- Timing side channel: span boundaries now reveal roughly *when* writes
  went quiet (a partial span means an idle period of at least `max_wait`).
  Span boundaries were already public, and the witness records carry
  their own ordering. Record timestamps are not signed or exposed. This is
  noted, not mitigated.
- The clock is only used locally to decide when to sign. It is not
  authenticated, and a skewed or manipulated clock can only make signing
  earlier or later. It cannot make an unsigned record verify, because
  `UnsignedTail` still fails closed.

## Governance

No change to the "no witness, no mutation" invariant. The sink still
cannot turn a successful inner commit into an `Err`. This run was carried
out by a single session acting as researcher, implementer and adversarial
reviewer. No MetaHarness, Darwin or Flywheel CLI is installed in this repo
(see the nightly README).

## Failure Modes

1. **Idle writer, no poller.** The last partial batch stays unsigned
   until the next write, `seal_expired()` or `seal()`. This is the B1
   failure, and it is by design.
2. **Poller slower than `max_wait`.** The bound becomes `max_wait +
   poll_interval + sign time`. The measured 1 ms poll gave ≤ 11.85 ms.
3. **Tiny `max_wait` on a slow stream.** This degrades toward per-write
   signing (with `max_wait = 0`, one signature per `emit_batch`). It is
   correct but loses amortization.
4. **Crash before an expired batch is noticed.** Same as ADR-347 Limit 2:
   the tail is permanently unsigned and fails closed.
5. **Non-monotonic clock.** `Instant` is monotonic, and
   `saturating_duration_since` prevents panics.

## Migration

No migration is needed. Existing `PerRecord` and `BatchTail` users see
identical behavior and identical bytes. To adopt, replace `BatchTail {
batch_size }` with `BatchTailTimeout { batch_size, max_wait }`. For a hard
bound, also call `sink.seal_expired()` from a periodic timer the caller
owns.

## Rollback

Revert the feature commit. Signed logs produced under `BatchTailTimeout`
stay verifiable by the unchanged `verify_signed_chain`, because their spans
are ordinary `BatchTail` spans.

## Rejection Criteria

These were fixed before the benchmark ran. The hypothesis is rejected if
any of A1–A4 or B1 fails while B2 passes, and the result is INCONCLUSIVE
if B2 fails. B1 failed while B2 passed, so the result is **REJECT**. The
code is kept only because every non-hypothesis gate passed and the
guarantee it documents is the one measured, not the one hypothesized.

## Open Questions

1. Should `seal_expired` be driven by a first-party helper, such as an
   optional `tokio` interval behind a feature, or remain caller-owned?
2. Adaptive `max_wait`, trading amortization for latency by observed
   write rate. This is a Darwin-style tuning target, and ADR-343 names the
   same question.
3. Should `SigningStrategy` become `#[non_exhaustive]` before the crate is
   published?
4. Measure the bound on a real deployment's arrival distribution rather
   than a synthetic uniform-gap schedule.
