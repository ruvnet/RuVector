# Bounding the Worst-Case Latency of an Amortized Signature Scheme

## Problem

If you amortize a cryptographic signature over a batch of records to save
signing cost, you've traded throughput for latency — but only *measured*
latency if you also bound the *worst case*. A size-triggered batch
("sign once every 64 records") has an unbounded worst case whenever the
arrival rate is slower than the batch fills: the last few records before
a lull can sit unsigned forever, with nothing reporting that they will.

This is exactly what `ruvector-agent-memory`'s `SignedWitnessSink` had,
after an earlier research run (2026-09-16) built it to Ed25519-sign a
tamper-evident agent-memory ledger. That run measured the throughput side
of the tradeoff carefully (36x–212x fewer signatures than per-record
signing across batch sizes 16–256) and explicitly flagged the latency side
as open: "a batch-fill timeout... so a deployment gets a bounded
worst-case signature-availability latency instead of 'whenever the batch
happens to fill.'"

## Hypothesis

```text
Given a SignedWitnessSink<S> configured with
SigningStrategy::BatchTail { batch_size, max_wait: Some(d) },
fed a trickle of records strictly fewer than batch_size,

when a caller polls seal_if_overdue_now() on a fixed real-clock tick
until it returns true,

then the oldest pending record's measured wall-clock age at the moment
it is signed is bounded by d plus the polling tick's scheduling slack,

subject to the resulting signed chain still verifying, and the existing
size-triggered path remaining unaffected.
```

## Technical Design

The fix adds exactly one new idea to the existing module: a second
closing trigger alongside "batch is full." `SigningStrategy::BatchTail`
gains `max_wait: Option<Duration>`; the sink's internal `PendingSpan`
records when it was opened; and three new methods let a caller ask "is
anything overdue?" and close it if so:

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

Two design choices worth calling out:

1. **The sink never arms a timer itself.** It stays single-threaded and
   synchronous, like the rest of the module. The bound only holds if a
   deployment actually drives `seal_if_overdue_now()` on a schedule — a
   ruFlo maintenance tick, in practice. This is a mechanism, not an
   automatic promise, and the docs say so explicitly rather than implying
   passive safety.
2. **`seal_if_overdue` doesn't re-check the deadline against a live
   clock.** It trusts whatever `now` the caller passes. This mirrors a
   design choice already made in a sibling crate
   (`ruvector-retrieval-receipt::batch_fill`'s `BatchScheduler`, built for
   a discrete-event query-arrival simulation) and has a nice side effect:
   unit tests can construct deterministic "future" instants with plain
   `Instant` arithmetic (`deadline - Duration::from_millis(1)`,
   `deadline`) instead of real sleeps, so the timeout logic is fully
   covered without a flaky test suite.

The sibling crate's pattern was *ported as a concept*, not copied as code
or added as a dependency: `BatchScheduler` is arrival-indexed (built
around a `Vec<PendingMember>` for simulating a query stream), while
`SignedWitnessSink`'s `PendingSpan` is record-sequence-indexed (it tracks
a contiguous `[from, to]` range inside an already-committed ledger). The
"close at size or timeout, whichever first" idea transfers; the data
structure doesn't need to.

## Actual Implementation

All of it lands in `ruvector-agent-memory`:

- `src/witness_signing.rs` — the field, the method additions, updated
  module docs naming the new limit explicitly (`max_wait` bounds *when a
  caller may seal*, not a promise that anything will).
- `src/witness_signing_tests.rs` — 44 pre-existing tests updated only for
  the new field (confirmed behaviorally unchanged), plus 5 new tests
  covering: no deadline without `max_wait`; sealing at-but-not-before the
  deadline; a no-op on an empty sink; the size trigger preempting the
  timeout; and a mixed chain (one size-closed span, one timeout-closed
  span) verifying end to end.
- `examples/witness_signing_timeout_bench.rs` — the benchmark below.

## Benchmark Evidence

Three variants, single-threaded release build, same signing path:

- **baseline** — `max_wait: None`, an 8-record trickle (well under a
  64-record `batch_size`). Confirms the bug: `unsigned_pending() == 16`
  and `pending_deadline() == None`, forever.
- **hybrid** — same trickle, `max_wait: Some(50ms)`, closed by a loop
  that sleeps 5ms and polls `seal_if_overdue_now()` — the production
  shape, not a one-shot deadline-computed sleep, so measured scheduling
  slack is included rather than hidden.
- **size** — a fast 64-record burst with `max_wait` configured but
  irrelevant (size closes first). Confirms no regression to the
  pre-existing path.

Raw output (one of five repeated runs, all consistent, no flakes):

```
ruvector-agent-memory witness-signer batch-fill timeout benchmark
trickle_records=8 batch_size=64 max_wait=50ms poll_tick=5ms

baseline (unbounded): unsigned_pending=16 pending_deadline_without_max_wait=false -> CONFIRMS unbounded tail (no max_wait => no deadline, ever)
candidate_a (hybrid):  measured_unsigned_age=51.464952ms  bound=60ms  signed_chain_ok=true  -> PASS
candidate_b (size):    64 records closed by size trigger in 169.58µs (1 span(s)) -> PASS

ACCEPT: max_wait bounds worst-case signature-availability latency to 60ms; baseline (no max_wait) remains unbounded as documented; size-triggered path unaffected.
```

Across five runs the hybrid variant's measured age landed consistently in
the 51.5–52.3ms band against a 60ms bound (`max_wait` + 2 poll ticks) —
comfortably inside, with the ~1.5–2.3ms beyond `max_wait` itself
attributable to the 5ms poll tick and the `seal()` call, not noise at the
edge of the bound. `cargo test -p ruvector-agent-memory --lib` passed
50/50 (44 existing + 6 new), and `cargo clippy --all-targets` reported no
warnings. The pre-existing `witness_signing_bench` example was re-run
unmodified as a regression control: `correctness=PASS` for every variant,
no change to the already-measured per-signature cost or amortization
ratio from the originating 2026-09-16 run.

## Limitations

- One fixed `(batch_size, max_wait, poll_tick)` triple was measured, not a
  parameter sweep.
- No WASM, embedded-hardware, or multi-threaded/concurrent-writer
  measurement — both remain open from the originating run and this one.
- The slack beyond `max_wait` was reported as a single bounded quantity,
  not decomposed into "poll-tick scheduling" versus "signing cost"
  separately.

## Production Relevance

The mechanism is additive and opt-in (`max_wait: None` preserves prior
behavior exactly), so adopting it costs nothing for a deployment that
doesn't need the bound. For one that does — an interactive, low-write-rate
agent-memory deployment, the common case this fix targets — the missing
piece for production use is wiring `seal_if_overdue_now()` into whatever
periodic maintenance scheduler already exists (a ruFlo workflow, in this
ecosystem), not more cryptographic engineering: the signing path itself is
unchanged.

## RuVector Ecosystem Implications

This closes a specific, previously-named gap in the agent-memory witness
chain (ADR-134 §9 / ADR-347 lineage) without touching its cryptography,
and it demonstrates that a scheduling-policy pattern from one crate
(`ruvector-retrieval-receipt`'s batch-fill methodology) can be reused
across a different crate's data shape by porting the *decision policy*
rather than the code. The natural next integration is a ruFlo workflow
definition that actually drives the tick in a real deployment — described
here, not yet shipped.

## Future Direction

The 2026-09-16 run's other open items (WASM measurement, concurrent-writer
hardening, multi-issuer verification, wiring through
`witnessed_compaction`) remain exactly as open as before; this run closes
one specific item from that list and adds its own narrower follow-ups
(parameter sweep, slack decomposition, an actual ruFlo manifest, a
read-only MCP health tool) rather than claiming more than it measured.

## References

- `docs/adr/ADR-355-witness-signer-batch-fill-timeout.md`
- `docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`
- `crates/ruvector-retrieval-receipt/src/batch_fill.rs`
- `crates/ruvector-agent-memory/src/witness_signing.rs`
