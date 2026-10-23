# ADR-353: Wall-Clock Fill Timeout for `WitnessSigner`'s `BatchTail` Strategy

## Status

Accepted. Additive, backward-compatible change to the existing opt-in
module `ruvector-agent-memory::witness_signing` (ADR-347). No new crate,
no new dependency.

## Context

ADR-347 shipped `SignedWitnessSink` with two signing strategies:
`PerRecord` (one signature per record) and `BatchTail { batch_size }`
(one amortized signature per `batch_size` records). `BatchTail` only
closes a span when `batch_size` more records arrive. ADR-347's own
module docs name this explicitly as a limit, and its Next Research item
1 asked for the fix directly:

> A wall-clock batch-fill timeout for `BatchTail`... so a deployment
> gets a bounded worst-case signature-availability latency instead of
> "whenever the batch happens to fill."

The sibling crate `ruvector-retrieval-receipt` had already solved the
identical structural problem for a different artifact (signed retrieval
receipts, ADR-340/343): its `batch_fill::BatchFillPolicy::hybrid` closes
a batch at `max_members` records *or* after `max_wait_ns` since the
oldest pending member, whichever is first, with the caller driving the
timeout from `oldest_pending_arrival_ns()`. This ADR ports that pattern
(not the code — the two crates have unrelated batch payloads) into
`witness_signing`.

Without this fix, a `BatchTail`-signed TARL ledger in a low-write-volume
deployment (the realistic case for most agent-memory tenants: writes
are sporadic, not a continuous firehose) can carry an unsigned tail for
an unbounded time — in the worst case, forever, if fewer than
`batch_size` records are ever written after the last close. Any witness
record in that tail has no signed evidence until an operator forces a
`seal()` or enough further writes arrive. That is a real governance gap
for proof-gated or audited deployments: "has this write been
authenticated yet" should have a time-bounded answer.

## Hypothesis

```text
Given a BatchTail-signed witness sink receiving records in bursts
separated by idle periods shorter than its deployment's audit-latency
requirement,

when SigningStrategy::batch_tail_timed(batch_size, max_wait_ns) replaces
SigningStrategy::batch_tail(batch_size) and a caller drives
seal_on_timeout at pending_deadline_ns(),

then the maximum observed time from a record's own timestamp_ns to its
covering span's close should be bounded by max_wait_ns,

subject to: verify_signed_chain continuing to accept the resulting
chain exactly as it does today, and no measurable regression in
signing throughput/signature count when load is high enough that
batch_size, not the timeout, is the real trigger.
```

## Decision

1. `SigningStrategy::BatchTail` gains a second field,
   `max_wait_ns: Option<u64>`. `None` preserves the exact original
   behavior (fixed-size-only, no early close). `Some(w)` adds a second
   trigger: close once `w` nanoseconds have elapsed since the oldest
   record in the pending span, measured against that record's own
   `timestamp_ns`.
2. Two const constructors replace bare struct literals at call sites:
   `SigningStrategy::batch_tail(batch_size)` (old behavior) and
   `SigningStrategy::batch_tail_timed(batch_size, max_wait_ns)` (new).
3. `SignedWitnessSink` gains `pending_deadline_ns() -> Option<u64>`
   (the deadline, if any, derived from the pending span's oldest
   record) and `seal_on_timeout(now_ns: u64) -> bool` (closes the
   pending span, via the exact same `close()` the size trigger and
   explicit `seal()` use, iff the deadline has elapsed; otherwise a
   no-op). Mirrors
   `batch_fill::BatchScheduler::{oldest_pending_arrival_ns, close_on_timeout}`'s
   caller-driven shape.
4. `PendingSpan` gains `oldest_timestamp_ns: u64`, set from the first
   record folded into a fresh pending span.
5. No internal timer, thread, or async runtime is added — consistent
   with the rest of the crate being synchronous. The deadline only
   bounds latency if some caller actually invokes `seal_on_timeout` at
   or after it; this is documented as a named limit, not silently
   assumed away.

## API Shape

```rust
pub enum SigningStrategy {
    PerRecord,
    BatchTail { batch_size: usize, max_wait_ns: Option<u64> },
}
impl SigningStrategy {
    pub const fn batch_tail(batch_size: usize) -> Self;
    pub const fn batch_tail_timed(batch_size: usize, max_wait_ns: u64) -> Self;
}
impl<S: WitnessSink> SignedWitnessSink<S> {
    pub fn pending_deadline_ns(&self) -> Option<u64>;
    pub fn seal_on_timeout(&mut self, now_ns: u64) -> bool;
    // seal(), emit_batch, anchor(), unsigned_pending(), ... unchanged
}
```

## Feature Flags

None. `witness_signing` remains part of `ruvector-agent-memory`'s
default build (no feature gate), same as ADR-347.

## Implementation Plan

- `crates/ruvector-agent-memory/src/witness_signing.rs`: enum/struct
  changes above (±60 lines on top of ADR-347's 487-line module).
- `crates/ruvector-agent-memory/src/witness_signing_tests.rs`: 10 new
  unit tests (deadline tracking, timeout-vs-size-trigger precedence,
  no-timeout backward compatibility, verification parity for a
  timeout-closed span including the existing diligent-forgery check).
- `crates/ruvector-agent-memory/examples/witness_signing_fill_timeout_bench.rs`:
  new benchmark — see Benchmark Evidence.
- Existing call sites (`witness_signing_tests.rs`'s `STRATEGIES`
  constant, `witness_signing_bench.rs`'s three `BatchTail` benches)
  updated from the old struct-literal form to `SigningStrategy::batch_tail(n)`.

## Benchmark Evidence

`cargo run --release -p ruvector-agent-memory --example witness_signing_fill_timeout_bench`
(see `docs/research/nightly/2026-10-23-witness-signer-batch-tail-fill-timeout/README.md`
for full methodology, raw output, and three repeated runs of Experiment
B). Headline numbers from one representative run:

```
Experiment A (4 bursts of 10 records, 2s idle gaps, batch_size=64):
  baseline     unbounded: pending staleness at final checkpoint = 8,002,000,000ns (8.0s), never closed on its own
  candidate_a  (max_wait_ns=10ms):  max observed availability latency = 10,000,000ns   -- bound HELD
  candidate_b  (max_wait_ns=100ms): max observed availability latency = 100,000,000ns  -- bound HELD
  verify_signed_chain: PASS for baseline, candidate_a, candidate_b after a final seal()

Experiment B (2,000 records, 1us apart, no idle gaps -- batch_size always wins):
  signatures: 32/32/32 (identical across baseline/candidate_a/candidate_b)
  mean per-op latency: within -12.7%..+1.2% of baseline across 3 repeated runs
  (noise, not a systematic regression -- see README for the 3-run table)
```

## Security

`seal_on_timeout` calls the identical `close()` function the size
trigger and explicit `seal()` already call: it signs the same
`records_digest`/chained-link message, over the same SHA-256 digest of
the full record bytes, with the same Ed25519 key. It introduces **no
new way to produce a signature**, only a new condition under which the
existing signing path fires. `verify_signed_chain`'s coverage, rollback,
and forgery-rejection guarantees are therefore unchanged by construction
— re-verified directly by
`timeout_closed_span_verifies_exactly_like_a_size_closed_one`, which
repeats ADR-347's diligent-forgery test against a timeout-closed span.

The one new attack-relevant question this ADR introduces: can a
deployment be starved into an unexpectedly *early* close that costs more
signatures than it should? No — `max_wait_ns` only ever closes a span
*sooner* than the old fixed-size-only behavior; it cannot be used to
delay a close, and closing early produces a smaller, still-correctly
verified span, not a security weakening. The failure mode is cost
(more, smaller spans under adversarial sparse writes), not correctness.

## Governance

For proof-gated or audited deployments, this turns "when is a write's
signature available" from an unanswerable question (depends on
unrelated future traffic) into an operational SLO parameter
(`max_wait_ns`) that an operator sets and can reason about. It does not
change what is signed or how verification works; it only changes when
signing happens.

## Failure Modes

- **No driver.** If nothing calls `seal_on_timeout`, behavior is
  identical to `max_wait_ns: None` regardless of its configured value.
  This must be surfaced operationally (e.g. a periodic maintenance
  task — see the ruFlo integration note in the nightly README) or the
  ADR's whole benefit is lost silently.
- **Scheduler jitter.** The benchmark's discrete-event driver wakes
  exactly at the deadline — the best case. A real scheduler's own
  wake-up latency adds to the effective bound; this ADR does not
  measure that (see Open Questions).
- **Pathologically tiny `max_wait_ns` under bursty load.** If
  `max_wait_ns` is set far below the real inter-arrival time of a busy
  deployment, nearly every span closes by timeout instead of by size,
  producing many small spans and losing most of `BatchTail`'s
  amortization benefit (this is the same size/latency tradeoff
  `ruvector-retrieval-receipt`'s `batch_fill` sibling already documents
  for receipts; it is a configuration responsibility, not a bug).

## Migration

Backward compatible. Existing `SigningStrategy::BatchTail { batch_size }`
struct-literal construction sites must switch to
`SigningStrategy::batch_tail(batch_size)` (a one-line mechanical change;
all in-tree call sites were updated in this change). No persisted data
format changes: `SignedSpan`/`SignedAnchor` encodings are unchanged.

## Rollback

Revert the diff; `batch_tail`/`batch_tail_timed` and
`pending_deadline_ns`/`seal_on_timeout` are purely additive on top of
ADR-347's shipped behavior, so rollback is a plain revert with no data
migration.

## Rejection Criteria

Would have been rejected if: the timeout path required a different
signing/verification code path (new attack surface) instead of reusing
`close()`; the high-load benchmark had shown a systematic (not noise-
level) throughput regression; or `verify_signed_chain` had needed
changes to accept timeout-closed spans. None of these occurred.

## Open Questions

1. Real scheduler wake-up jitter on top of `max_wait_ns` is not
   measured here (ADR-347's Next Research item 4's WASM
   binary-size/latency question is a related, still-open instance of
   "measure this outside an idealized simulation").
2. Concurrent-writer / fault-injection hardening (ADR-347's Next
   Research item 3) is orthogonal and still open.
3. Whether `seal_on_timeout` should be wired to a concrete ruFlo
   scheduled workflow in this crate, versus left as a library primitive
   for the embedding application to drive, is left to the embedding
   deployment (see the nightly README's ruFlo section) — no workflow is
   shipped in this change.

## References

- `docs/adr/ADR-347-witness-signer-tarl-ledger.md` — the strategy and
  module this ADR extends.
- `crates/ruvector-retrieval-receipt/src/batch_fill.rs` — the
  `BatchFillPolicy`/`BatchScheduler` pattern this ADR ports (not reuses
  as code) into a different crate's signing path.
- `docs/research/nightly/2026-10-23-witness-signer-batch-tail-fill-timeout/README.md` —
  full benchmark methodology and raw results.
