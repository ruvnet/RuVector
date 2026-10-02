# A Batch Timeout That Only Fires on the Next Write Is Not a Latency Bound

## Problem

`ruvector-agent-memory` signs its TARL ledger's witness chain with
Ed25519 (`SignedWitnessSink`, ADR-347). Signing every record costs ~100 µs
per record, so the practical mode is `BatchTail { batch_size }`: one
signature covers a whole run of records. The catch is that a batch closes
only when it fills. During quiet periods a partially filled batch can
stay unsigned indefinitely. Until it closes, a verifier reports those
records as an `UnsignedTail` (unauthenticated), and a crash loses them for
good.

A sibling crate, `ruvector-retrieval-receipt`, had already solved this
for retrieval receipts (ADR-343). Its fix was to close a batch at N
members *or* after T has elapsed since the oldest member arrived,
whichever comes first. The task was to port that pattern.

## Hypothesis

Add `SigningStrategy::BatchTailTimeout { batch_size, max_wait }`. Under a
bursty workload, worst-case signature-availability latency (from a record
being witnessed to a signature covering it existing) should then be
bounded by `max_wait`. Correctness, forgery rejection, existing tests and
dense-workload throughput must all be unaffected.

The design constraint was that the sink has no background timer, so the
timeout is checked when writes arrive.

## Design

- Each open batch (`PendingSpan`) records `opened_at: Option<Instant>`.
  Only the timeout strategy reads the clock, once per `emit_batch` call.
- `emit_batch` works in three steps:
  1. Commit to the inner sink first (witness-first; a refused batch is
     never signed).
  2. Append the records, closing the batch at `batch_size`.
  3. Afterwards, sign the open batch if its oldest record is at least
     `max_wait` old. The records of the write that discovered the expiry
     go into the same signature.
- `seal_expired()` is a new public hook that lets a caller-owned timer
  close an expired batch without writing anything.
- A timeout close is signed exactly like a size close. The signed
  statement is "these records, following that span", so the message
  format and `verify_signed_chain` are unchanged.
- The pattern was ported, not the code. ADR-343's scheduler is a
  clock-free list of members, built for a discrete-event simulation. Here
  the batch is a single streaming SHA-256 digest inside a live sink. No
  new dependency was added.

## Implementation

- `crates/ruvector-agent-memory/src/witness_signing.rs`: the new variant
  and the clock read.
- `crates/ruvector-agent-memory/src/witness_batch_fill.rs`: new child
  module containing the batch-closing logic.
- Tests: 8 new, using synthetic `Instant`s; only one test sleeps, for
  3 ms. The existing strategy matrix now also includes two timeout
  configurations: one whose timeout never fires, and one whose timeout
  fires on every write. All 53 tests pass (45 pre-existing and 8 new), and
  `clippy -D warnings` is clean.

## Evidence

Command: `cargo run --release -p ruvector-agent-memory --example
bounded_batch_fill_bench`, run 3 times. Environment: 4 vCPU, Linux
6.18.44, rustc 1.94.1. Settings: `batch_size=64`, `max_wait=10ms`. The
bursty schedule is deterministic (fixed seed): 80 bursts of 1–6 ledger
add+accept pairs, with real `thread::sleep` gaps of 2–30 ms (51 of the 80
gaps are longer than 10 ms).

| Variant | max latency (3 runs) | p50 | unsigned at end | signatures |
|---|---|---|---|---|
| `BatchTail{64}` | 249.9 / 249.9 / 250.3 ms | 62.4 ms | 42 records | 9 |
| `BatchTailTimeout{64,10ms}`, checked on write | 37.29 / 37.37 / 37.32 ms | 16.4 ms | 11 records | 65 |
| same + caller-owned 1 ms `seal_expired()` poll | 11.22 / 11.85 / 11.17 ms | 10.2 ms | 0 | 65 |

On the dense workload (20,000 pairs, 7 interleaved reps), the ratio of
median run times between timeout and plain was 1.007, 0.986 and 0.864
across the three runs. Both produced 625 signatures, so the timeout never
fired. Correctness passed and the diligent forgery was rejected in every
run.

**Verdict: REJECT.** The pre-registered gate was max latency ≤ 11 ms
(`max_wait` plus 1 ms tolerance). The measured maximum was 37.3 ms in
every run. It sits just under the analytical ceiling of `max_wait` + the
largest gap = 39 ms, so the failure is structural, not noise.

## Limitations

- A check that runs only on write gives a bound of `max_wait` plus the
  time until the next write. If writes stop, that is unbounded.
- The polled variant was a supplementary row, not a pre-registered gate.
  It was not tested under CPU contention.
- The gaps are synthetic and uniformly distributed, not taken from a real
  agent trace.
- The VM is shared, so timings are noisy (±14% on dense medians).
- On `wasm32-unknown-unknown`, `Instant::now()` panics, so the timeout
  variant needs a clock-injection API there. The other strategies never
  read the clock.

## Production Relevance

`BatchTailTimeout` has no measurable steady-state cost and cut maximum
latency 6.7× on the bursty schedule, so it is worth turning on. To get
the latency bound people actually want, the host must also call
`sink.seal_expired()` from its own periodic tick: an event loop, an
interval timer or an RVM scheduler tick. Kafka's `linger.ms` and CT's
Maximum Merge Delay both work because a scheduler owns the deadline, not
the next client request. This result is the in-process version of the
same lesson.

## Ecosystem Implications

- The signed span format is unchanged, so RVF exports, the verifier and
  `SignedAnchor` rollback protection are unaffected.
- `pending_age()` gives MCP and ruFlo monitors a direct "oldest unsigned
  record" gauge.
- ADR-343's simulated bound assumed the timeout fires exactly at its
  deadline. This run measures what that assumption costs when nothing
  fires it.

## Future Direction

1. An optional timer driver behind a feature flag, re-measured against
   the same seed and gate.
2. A public clock-injection API, for WASM and simulation.
3. Evaluation on a real agent trace.
4. Crash-injection tests for expired batches that no write has
   discovered yet.

## References

- ADR-352 (this work); ADR-347 (witness signer); ADR-343 (batch-fill
  latency simulation).
- `docs/research/nightly/2026-09-29-bounded-batch-fill-witness-signing/README.md`
- Apache Kafka producer configuration docs (`batch.size`, `linger.ms`).
- PostgreSQL WAL configuration (`commit_delay`).
- RFC 896 (Nagle); RFC 6962 (Certificate Transparency, Maximum Merge
  Delay).
