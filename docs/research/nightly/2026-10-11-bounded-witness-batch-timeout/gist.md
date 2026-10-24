# Bounding the Worst-Case Latency of an Amortized Signature Scheme

## Problem

`ruvector-agent-memory`'s `SignedWitnessSink` signs a tamper-evident
witness log (Ed25519 over SHA-256 digests, chained span-to-span) either
per record or in amortized batches (`SigningStrategy::BatchTail {
batch_size }`). Amortized batching is the obvious win for throughput —
one signature instead of up to `batch_size` — but it has an unstated
cost: a batch that never fills never signs. If a ledger goes quiet after
20 of a 64-record batch, those 20 records sit unauthenticated
indefinitely. The sink's own prior design (ADR-347) named this and
deferred it: "No wall-clock fill timeout in this pass."

This is not a hypothetical gap. `verify_signed_chain` already has a
dedicated error variant for it (`SignedChainError::UnsignedTail`) — the
system was built to detect the problem, just not to prevent it from
growing without bound.

## Hypothesis

A caller-driven, wall-clock timeout — close the batch at `batch_size`
records *or* after `max_wait_ns` since the oldest unsealed record,
whichever comes first — bounds the worst case to `max_wait_ns` plus
however often the caller checks, while costing almost nothing when
traffic is dense enough that the count-based close always wins anyway.

This is not a new idea in this codebase: `ruvector-retrieval-receipt`'s
`batch_fill` module validated exactly this hybrid policy for a different
signed-batch problem (receipt anchoring, ADR-343). The question this run
answers is whether the *same policy*, ported into `SignedWitnessSink`'s
different architecture (it commits through a `WitnessSink` trait and
signs inline, rather than running as a standalone scheduler), still
holds up — and whether it can be implemented without introducing
`ruvector-retrieval-receipt` as a dependency of `ruvector-agent-memory`.

## Design

```rust
pub enum SigningStrategy {
    PerRecord,
    BatchTail { batch_size: usize },
    BatchTailTimeout { batch_size: usize, max_wait_ns: u64 }, // new
}
```

The sink tracks `opened_at_ns` — the wall-clock time the current pending
span's *first* record arrived, measured against a monotonic
`Instant`-based clock private to the sink. Two entry points drive the
deadline:

- `check_timeout_at(now_ns)` — deterministic, takes the current time
  explicitly. Returns `true` and force-seals iff `now_ns` has reached
  `opened_at_ns + max_wait_ns`.
- `check_timeout()` — the real-wall-clock wrapper, meant to be called
  from a periodic maintenance tick (a ruFlo workflow, a cron, any
  scheduler the host process already has — the sink does not spawn its
  own timer).

The one subtlety that actually mattered: **order of operations inside
`emit_batch`.** The first implementation checked the timeout *after*
merging a newly-arrived record into the pending span. That's backwards —
if the span was already stale, merging a late record into it before
checking the deadline means that record's own signing wait gets silently
extended to cover however long the *previous* span had already been
waiting. The fix is to check first, merge second: a stale span gets
force-sealed on its own, and the new record opens a fresh span with its
own fresh deadline. This was caught by a property test before it ever
reached a benchmark, which is the point of writing the property test
before trusting the benchmark.

```rust
// Decision 4 in ADR-348 — order matters.
SigningStrategy::BatchTailTimeout { batch_size, .. } => {
    self.check_timeout_at(now_ns);  // seal anything already stale, FIRST
    for r in records {
        // ... merge into (possibly fresh) pending span, count-close if full
    }
}
```

## Evidence

Everything below is from `cargo run --release -p ruvector-agent-memory
--example witness_signing_timeout_bench`, three repetitions, on a
shared, containerized workstation (treat as indicative, not an SLA).

**Dense arrivals** (20,000 `add`+`accept` pairs, `batch_size=64`, no idle
gaps — the regime where the timeout should never fire and the cost
should be ~zero):

| Variant | Mean latency/op | Signatures |
|---|---|---|
| `BatchTail{64}` | 3.39–3.64µs | 625 |
| `BatchTailTimeout{64, 5ms}` | 3.34–3.44µs | 625 |

Overhead across three runs: **-1.4%, -0.2%, -7.1%** — every run came in
*faster*, not slower, which is run-to-run noise on shared hardware
rather than a real effect, but it rules out any measurable regression
against the ±10% threshold set before running the benchmark. Signature
counts are identical in every run: the count-based close wins every
single time under dense traffic, exactly as designed.

**Sparse arrivals** (10 pairs = 20 records, well under the 64-record
threshold, then a real 15ms idle gap, `max_wait_ns=5ms`, polled every
1ms):

| Variant | Pending after 15ms gap | Timeout bound exists | Force-sealed after |
|---|---|---|---|
| `BatchTail{64}` | 20 (unchanged) | no | never |
| `BatchTailTimeout{64, 5ms}` | 0 | yes | 5.53–5.65ms |

`BatchTail` isn't slow to resolve this — it structurally cannot resolve
it at all. `next_timeout_deadline_ns()` returns `None` for it regardless
of how much time passes, which is the precise, checkable form of "no
timeout concept exists" rather than an informal claim. `BatchTailTimeout`
resolves it in every run within the `max_wait_ns (5ms) + poll_interval
(1ms) = 6ms` bound the design derives.

A randomized property test (`witness_signing_tests.rs`) drives 8 trials
× 20 steps with randomized batch sizes and randomized sparse/bursty
inter-arrival gaps, asserting the observed pending age never exceeds
`max_wait_ns + poll_tick` and that the resulting signed chain still
verifies end to end — a check against the specific class of bug the
ordering fix above addresses, not just the two hand-picked scenarios
above.

Full regression: 55/55 tests passing (was 45/45 before this change — 10
new, 0 broken), `clippy --all-targets -- -D warnings` clean.

## Limitations

- The bound requires a caller to actually drive `check_timeout`
  periodically. The sink does not enforce this itself — ship the
  mechanism without the tick, and it silently behaves like plain
  `BatchTail`. That's a deliberate design choice (no self-spawned thread
  in a library type with no async runtime dependency), documented as an
  operational obligation, not hidden as a footnote.
- No durable-sink, multi-issuer, or WASM measurement in this run — all
  explicitly open, not assumed to transfer.
- Benchmark numbers are from one shared, containerized workstation, 3
  repetitions; treat as indicative.

## Production Relevance

This is a small, additive, opt-in change to one existing crate — no new
dependency, no new crate, no behavior change for existing callers. It
turns "signs eventually" into "signs within a stated bound, if you wire
up the tick" for any `ruvector-agent-memory` deployment that wants a
latency SLA on its witness chain, which matters anywhere a reader
(a staleness-alerting workflow, a cross-repo witness consumer, an
auditor) needs to know not just that a record *will* be authenticated,
but by when.

## RuVector Ecosystem Implications

Connects agent memory, witness chains/signed provenance, proof-gated
writes, the retrieval-receipt crate's already-validated batching policy,
and ruFlo (as the natural home for the periodic tick this mechanism
needs in production — described concretely in the companion README's
"ruFlo Workflow" section, not asserted generically).

## Future Direction

Pick a production default for `max_wait_ns`, build the ruFlo tick
workflow, extend verification to a durable sink, and resolve the
still-open multi-issuer verification question this inherits from
ADR-347. Full detail and rejection criteria in
[ADR-348](../../../adr/ADR-348-bounded-witness-batch-fill-timeout.md).

## References

- ADR-347 — Ed25519 `WitnessSigner` for the TARL Ledger's Witness Chain.
- ADR-343 — Signed-Receipt Batch-Fill Latency (source of the ported
  hybrid count-or-timeout policy).
- ADR-348 — this run's full design record.
