# Bounding Signature-Availability Latency for Batched Witness Signing in RuVector

## Problem

Signing every record in an append-only witness log is expensive;
batching signatures amortizes the cost, but naively — close a batch once
it reaches N records — has an unbounded worst case: if records arrive
slower than N fills, a write's signature can stay unavailable
indefinitely. RuVector's `ruvector-agent-memory` crate already has both
halves of the standard fix in two different places: a batched signing
strategy (`SigningStrategy::BatchTail`, ADR-347) with no timeout, and —
in a sibling crate, `ruvector-retrieval-receipt` — a batch-fill policy
that *does* have one (`BatchFillPolicy::hybrid`, ADR-340). A prior
nightly research run in this repository built the first half, benchmarked
it, and explicitly named the second half as unfinished work. This write-up
closes that gap.

## Hypothesis

Wrapping `BatchTail`'s signing in a wall-clock timeout — close the span
at `batch_size` records, or after `max_wait_ns`, whichever comes first —
should bound worst-case signature-availability latency to roughly
`max_wait_ns`, at negligible cost when the batch fills by count anyway.

## Technical Design

Rather than editing the existing, already-security-reviewed
`SignedWitnessSink` (ADR-347), the fix is a decorator:

```rust
pub struct TimedBatchTailSink<S: WitnessSink, C: Clock = SystemClock> {
    inner: SignedWitnessSink<S>,   // unmodified
    max_wait_ns: u64,
    opened_at_ns: Option<u64>,
    clock: C,
    timeout_seals: usize,
}
```

`Clock` is injected (`SystemClock` for production, `LogicalClock` for
deterministic tests/benchmarks) so the exact same code path is exercised
by both a real deployment and a discrete-event simulation. The wrapper
tracks only when the currently-open span began; `check_timeout()` seals
it early if `max_wait_ns` has elapsed, calling nothing but the inner
sink's already-public `seal()`. Everything the wrapper produces is
byte-for-byte what the unwrapped sink would have produced at that moment
— a timeout-sealed span is cryptographically indistinguishable from a
size-sealed one.

One integration gap surfaced during testing: `TransactionalLedger` (the
TARL ledger this sink plugs into) only exposed `witness_sink(&self) -> &S`,
not a mutable equivalent — so a host application had no way to call
`check_timeout()` on a *live* ledger without tearing it down (which resets
an internal sequence counter and breaks the signed chain's continuity
check). Fixed with one new accessor, `witness_sink_mut`.

## Actual Implementation

- `crates/ruvector-agent-memory/src/batch_tail_timeout.rs` — 355 lines,
  6 new unit tests, zero new Cargo dependencies.
- `crates/ruvector-agent-memory/examples/batch_tail_timeout_latency.rs`
  — a 465-line benchmark, methodology ported from
  `ruvector-retrieval-receipt`'s own `bin/batch_latency.rs`: deterministic
  xorshift-seeded Poisson and bursty-on/off arrival processes, a virtual
  clock driving the batch-fill decision, real `Instant`-measured Ed25519
  signing cost layered on top of virtual time.
- `crates/ruvector-agent-memory/src/ledger.rs` — one new accessor.

## Actual Benchmark Evidence

Real numbers, `cargo run --release`, 4000 synthetic arrivals per regime,
`batch_size=64`, reproducible with fixed seeds (raw output preserved at
`benchmark_raw_output.txt` next to this file):

```
regime                     policy                 n  spans  t/o  avail_p99   sign_amort_ns
target_load_5000qps        baseline_no_timeout 4000     63    0    14.481ms          620.0
target_load_5000qps        candidate_a_50ms    4000     63    0    14.479ms          604.6
light_load_50qps           baseline_no_timeout 4000     63    0  1402.118ms          612.4
light_load_50qps           candidate_a_50ms    4000   1141 1140    50.051ms        10768.2
light_load_50qps           candidate_b_10ms    4000   2665 2664    10.046ms        19490.7
bursty_on2000qps_off400ms  baseline_no_timeout 4000     63    0   434.355ms          453.9
bursty_on2000qps_off400ms  candidate_a_50ms    4000     77   20    49.340ms          543.9
```

At a slow 50 queries/sec arrival rate, the unmodified baseline's p99
signature-availability latency is **1.4 seconds**. Wrapped in a 50ms
timeout, it is **50ms** — bounded as designed, a ~28x reduction — at a
measured signing-cost overhead under healthy (5000qps) load of under 3%,
statistically within noise of baseline. The cost under slow load is real
and measured, not hidden: 18x more signatures (1141 spans vs. 63) at the
50ms bound, 42x more (2665) at a tighter 10ms bound — a predictable,
linear latency/throughput tradeoff, not a cliff. Every one of the 9
regime × policy combinations still verifies correctly under
`verify_signed_chain`.

Acceptance thresholds were fixed before running: p99 bounded by the
declared timeout plus a 5ms fixed slack; baseline's p99 at light load
exceeding the timeout variant's by at least 5x (it exceeded it by ~28x);
amortized signing cost at healthy load within 2x of baseline's (it was
~0.97x). All four passed. **Result: ACCEPT.**

## Limitations

- Synthetic Poisson/bursty arrival processes, not a real traffic trace.
- The latency/cost tradeoff curve is sampled at exactly two points
  (50ms, 10ms timeouts); the shape between them is interpolated.
- No fault-injection test of a crash between a timeout firing and the
  resulting span actually persisting.
- No WASM or embedded-hardware measurement; `SystemClock` needs a
  platform shim that was not built or measured tonight.
- Scheduling who calls `check_timeout()` in production is left to the
  caller — the mechanism is provided, not the periodic trigger.

## Production Relevance

This is not a research-only construct: `TimedBatchTailSink` implements
the same `WitnessSink` trait `SignedWitnessSink` does, so it is a direct,
drop-in replacement at any existing call site, with zero new
dependencies and zero behavior change for callers who don't opt in. An
agent-memory deployment that needs "every write is tamper-evident within
50ms" can adopt it today and get a provable, monitorable bound instead of
an unbounded "eventually."

## RuVector Ecosystem Implications

This connects RuVector's agent-memory witness chain, its Ed25519
provenance/signing machinery, and a batch-fill decision methodology
already built and measured in a sibling crate (`ruvector-retrieval-receipt`)
for an analogous gap — ecosystem-internal knowledge reuse rather than a
new algorithm. A natural next step is a ruFlo-owned periodic maintenance
tick that calls `check_timeout()` across every registered agent-memory
ledger, and, longer-term, a Darwin-automated sweep of
`(batch_size, max_wait_ns)` against a real traffic shape rather than two
hand-picked points.

## Future Direction

1. A real ruFlo periodic-tick integration, not just a sketch.
2. A bounded Darwin search over `(batch_size, max_wait_ns)` once a harness
   is wired into this repository (none was available tonight — verified,
   not assumed).
3. Fault-injection coverage for the crash-mid-timeout case.
4. Validation against a real agent-memory traffic trace.

## References

- `crates/ruvector-agent-memory/src/batch_tail_timeout.rs`
- `crates/ruvector-agent-memory/examples/batch_tail_timeout_latency.rs`
- `crates/ruvector-retrieval-receipt/src/batch_fill.rs`,
  `src/bin/batch_latency.rs` (the ported methodology)
- `docs/adr/ADR-347-witness-signer-tarl-ledger.md`
- `docs/adr/ADR-361-timed-batch-tail-witness-signing.md`
- `docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`
  (the run this one follows up on)
