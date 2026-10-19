# ADR-353: Bounded-Latency `BatchTail` Signing for the TARL Witness Chain

## Status

Accepted. New, opt-in, non-default `SigningStrategy` variant
(`ruvector-agent-memory::witness_signing::SigningStrategy::BoundedBatchTail`).
`PerRecord`/`BatchTail` are unchanged; no default behavior changes for
existing callers.

## Context

ADR-347 added `SignedWitnessSink` to `ruvector-agent-memory`'s TARL
witness chain, with two strategies: `PerRecord` (immediate, ~69µs/sig)
and `BatchTail { batch_size }` (amortized, ~3.8µs/op at `batch_size=64`
in that run's measurement). `BatchTail`'s own ADR text and the
2026-09-16 nightly run that shipped it both name an explicit, unaddressed
limitation: a pending span only closes once `batch_size` records have
arrived or `SignedWitnessSink::seal()` is called manually — if arrivals
pause, the newest records stay unsigned for as long as the pause lasts,
with no upper bound. That run's Next Research item 1 names the fix
directly: "a wall-clock batch-fill timeout for `BatchTail`... reusing the
`BatchFillPolicy`/`BatchScheduler` methodology already built and measured
in `ruvector-retrieval-receipt::batch_fill`... port the pattern, not the
code." Its Next Research item 6 separately names an unwired composition:
`SignedWitnessSink` had never been demonstrated end to end through
`witnessed_compaction::compact_witnessed` (ADR-345), the module that
witnesses memory *evictions* rather than admissions.

This ADR closes both.

## Hypothesis

```text
Given SignedWitnessSink's existing fixed-size-only BatchTail strategy,
which leaves a pending span unsigned indefinitely whenever record
arrivals pause before filling batch_size,

when a new BoundedBatchTail strategy additionally closes the pending span
once max_wait_ns has elapsed since it opened — checked inline against
each new record's existing timestamp_ns field, and externally via a new
seal_if_due(now_ns) method a caller drives from a wall-clock timer —

then the worst-case signature-availability latency over a synthetic
bursty-then-idle arrival trace should be bounded by max_wait_ns plus a
caller-controlled poll interval for BoundedBatchTail, while remaining
unbounded (growing with total elapsed deployment time) for plain
BatchTail, without materially regressing steady-state throughput (<=20%)
relative to BatchTail at the same batch_size,

subject to: verify_signed_chain passing on every produced chain in every
trial, and every latency/throughput number coming from an actual
cargo run --release execution.
```

## Decision

1. Add `SigningStrategy::BoundedBatchTail { batch_size: usize, max_wait_ns:
   u64 }` alongside the unchanged `PerRecord`/`BatchTail` variants.
2. `PendingSpan` gains `opened_at_ns: u64`, set from the first record's
   own `timestamp_ns` field (already part of the ADR-134 schema and
   already inside the hashed, signed span content) — **no change** to the
   `WitnessSink::emit_batch` trait signature shared by every sink in this
   crate.
3. `emit_batch`'s `BoundedBatchTail` arm closes the pending span when
   either trigger fires on a newly arrived record: size (`count >=
   batch_size`, identical logic to plain `BatchTail`) or inline time
   (`r.timestamp_ns - opened_at_ns >= max_wait_ns`).
4. Add `SignedWitnessSink::seal_if_due(&mut self, now_ns: u64) -> bool` —
   closes the pending span if its deadline has passed; a no-op for
   `PerRecord`/`BatchTail`, no pending span, or before the deadline. This
   is the mechanism that bounds latency when arrivals stop entirely: a
   caller runs any periodic timer it already has (cron, async interval,
   background thread — none built here) and calls this on a schedule.
5. Add `SignedWitnessSink::pending_opened_at_ns() -> Option<u64>`, mirroring
   `ruvector-retrieval-receipt::batch_fill`'s
   `BatchScheduler::oldest_pending_arrival_ns()`, for a caller to derive
   the next deadline.
6. The signed statement itself (`SignedSpan`, `SignPurpose::BatchTail`,
   the message layout, `verify_signed_chain`) is **completely
   unmodified** — a verifier cannot tell, and does not need to, which
   trigger closed a given span.
7. Add a test proving `SignedWitnessSink` composes with
   `witnessed_compaction::compact_witnessed` end to end, including a
   `seal_if_due`-triggered close for an eviction batch that no further
   eviction ever arrives for — closing the 2026-09-16 run's Next Research
   item 6.

## Evidence

Real `cargo run --release -p ruvector-agent-memory --example
bounded_batch_tail_bench` execution
(`rustc 1.97.0`, `cargo 1.97.0`, Ubuntu 24.04.5, `Linux 6.18.44-fc-v114
x86_64`):

```
== Experiment 1: steady-state throughput (N_ENTRIES=20000, batch_size=64, max_wait_ns=5000000) ==
baseline         total=   30.758ms  mean=   1.497us  throughput=  650227.9 ops/s  signatures=      0  correctness=PASS
candidate_a (BatchTail)         total=   76.449ms  mean=   3.780us  throughput=  261613.0 ops/s  signatures=    625  correctness=PASS
candidate_b (BoundedBatchTail)  total=   76.206ms  mean=   3.769us  throughput=  262447.4 ops/s  signatures=    625  correctness=PASS
overhead vs candidate_a = -0.29% (4-run range across this session: -0.29% to +13.25%, noise-dominated, all within the +-20% gate)

== Experiment 2: worst-case availability latency (10 cycles, 5-record bursts, 20ms idle gaps, batch_size=1e6 [isolates timeout path], max_wait_ns=5ms, 1ms poll) ==
candidate_a (BatchTail)         max_latency=200.050ms  mean_latency=110.025ms  spans_produced=1   correctness=PASS
candidate_b (BoundedBatchTail)  max_latency=  5.995ms  mean_latency=  5.875ms  spans_produced=10  correctness=PASS
```

All 5 numeric acceptance gates computed by the benchmark binary PASS
(steady-state correctness, overhead <=20%, legacy unboundedness
demonstrated, bounded variant's worst case <= computed upper bound,
latency-experiment correctness). `cargo test -p ruvector-agent-memory
--lib`: 53 passed, 0 failed (45 pre-existing + 8 new). `cargo clippy -p
ruvector-agent-memory --all-targets -- -D warnings`: clean. `cargo fmt -p
ruvector-agent-memory -- --check`: clean. Full raw output and methodology:
`docs/research/nightly/2026-10-14-bounded-witness-batch-tail/README.md`.

**ACCEPT.**

## Consequences

- Positive: a deployment can now state a bounded worst-case
  signature-availability latency for the TARL witness chain, provided it
  drives `seal_if_due` on a schedule — a real, auditable property plain
  `BatchTail` could not offer.
- Positive: `witnessed_compaction` eviction witnesses now have a tested,
  working path to the same bound, closing a named gap from ADR-345's
  lineage.
- Neutral: zero behavior change for existing `PerRecord`/`BatchTail`
  callers; fully opt-in.
- Negative: the bound is only as good as the caller's timer discipline —
  a deployment that adopts `BoundedBatchTail` but never calls
  `seal_if_due` gets `BatchTail`'s original unbounded behavior, silently.
  This is a usability risk, not a correctness risk (verification still
  fails closed on an unsigned tail either way).
- Negative: `BoundedBatchTail` does not change the crash-before-seal
  limitation inherited unchanged from ADR-347 — still in-memory-only
  pending state.

## Alternatives Considered

- **Internal `std::time::Instant` clock inside the sink**, instead of
  deriving the deadline from each record's own `timestamp_ns`. Rejected:
  would require changing the shared `WitnessSink::emit_batch` trait
  signature used by every sink in the crate, for no benefit over data
  already present and already signed.
- **Add `max_wait_ns` as an `Option<u64>` field on the existing
  `BatchTail` variant**, instead of a new variant. Rejected: would force
  6 existing call sites in this crate's own tests/example to add an
  unused field, for no behavioral gain, and risks silently changing
  already-reviewed `BatchTail` semantics; a new variant is strictly
  additive.
- **Spawn a background timer thread inside the crate.** Rejected as out
  of scope: this crate has no existing threading/async dependency, and a
  library-owned thread-spawning policy is a worse default across this
  crate's range of embedders (WASM, edge, single-threaded CLI, server)
  than exposing a cheap, pure primitive the deployment drives itself.
- **Depend on `ruvector-retrieval-receipt::batch_fill`'s
  `BatchScheduler` directly.** Rejected: that type separates
  batch-fill-decision bookkeeping from cryptography for a crate whose
  signer is architected that way; `SignedWitnessSink` already owns both
  pending-span state and signing in one type, so grafting a second
  scheduler on top would duplicate rather than share state. The pattern
  (hybrid size-or-deadline closing, caller-driven checks) was reused; the
  code, deliberately, was not — no new dependency edge between the two
  crates.

## Implementation Plan

1. `witness_signing.rs`: add the `BoundedBatchTail` variant,
   `PendingSpan::opened_at_ns`, `seal_if_due`, `pending_opened_at_ns`, and
   the generalized zero-batch-size validation. (Done.)
2. `witness_signing_tests.rs`: 7 new unit tests covering validation, size
   trigger, inline time trigger, `seal_if_due` bounding latency with zero
   further arrivals, `seal_if_due` no-op correctness, and post-`seal_if_due`
   chain verification. (Done.)
3. `witnessed_compaction.rs`: 1 new test proving end-to-end composition
   with `compact_witnessed`. (Done.)
4. `examples/bounded_batch_tail_bench.rs`: real two-experiment benchmark
   with explicit numeric acceptance gates. (Done.)
5. Not done, deferred (see Next Research in the nightly README): a real
   OS-thread/async timer driver replacing the deterministic
   simulated-clock driver used in Experiment 2; a joint
   `(batch_size, max_wait_ns, poll_interval)` sweep against a real
   production arrival-rate distribution.

## API Shape

```rust
pub enum SigningStrategy {
    PerRecord,
    BatchTail { batch_size: usize },              // unchanged
    BoundedBatchTail { batch_size: usize, max_wait_ns: u64 }, // new
}

impl<S: WitnessSink> SignedWitnessSink<S> {
    // unchanged: new, from_keypair, public_key, spans, inner, anchor,
    // unsigned_pending, seal

    /// New: oldest pending record's timestamp_ns, for a caller to derive
    /// the next seal_if_due deadline.
    pub fn pending_opened_at_ns(&self) -> Option<u64>;

    /// New: force-close the pending span if its deadline has passed.
    /// No-op (false) for PerRecord/BatchTail, no pending span, or before
    /// the deadline.
    pub fn seal_if_due(&mut self, now_ns: u64) -> bool;
}
```

`verify_signed_chain`'s signature and semantics are **unchanged**.
`WitnessSignerError::ZeroBatchSize` now also covers
`BoundedBatchTail { batch_size: 0, .. }`.

## Feature Flags

None. No `Cargo.toml` change — zero new dependencies, same as ADR-347
(`rvf-types::ed25519` was already an unconditional dependency). The new
variant is additive Rust API, not gated behind a feature.

## Benchmark Evidence

See "Evidence" above and the full raw output/methodology in
`docs/research/nightly/2026-10-14-bounded-witness-batch-tail/README.md`.
Reproduce:

```bash
cargo test -p ruvector-agent-memory --lib
cargo clippy -p ruvector-agent-memory --all-targets -- -D warnings
cargo run --release -p ruvector-agent-memory --example bounded_batch_tail_bench
```

## Security

- No new cryptographic primitive; `verify_signed_chain` is unmodified.
  `BoundedBatchTail` spans use the identical `SignPurpose::BatchTail`
  domain tag and message layout as plain `BatchTail` spans — a verifier
  cannot distinguish which trigger closed a span, by design (the trigger
  is not part of the signed statement, only content and coverage are).
- `timestamp_ns` trust model is unchanged from ADR-347 (already hashed,
  signed content; attacker-influenceable only by whoever already
  controls record generation). `seal_if_due`'s `now_ns` should come from
  a trusted clock; the function does not itself validate it, matching
  `ruvector-retrieval-receipt::batch_fill::BatchScheduler`'s equivalent
  trust assumption.
- `seal_if_due` can only no-op or call the existing, already-reviewed
  `seal()` — no new failure mode, no partial-signing or corruption path.
- "No witness, no mutation" is preserved unchanged: `emit_batch`'s
  witness-first ordering is untouched by this change.

## Governance

None beyond the existing "no witness, no mutation" invariant, unchanged.

## Failure Modes

1. `seal_if_due` requires an actual caller-driven timer; without one, a
   `BoundedBatchTail` deployment silently gets `BatchTail`'s original
   unbounded behavior.
2. The realized bound is `max_wait_ns + poll_interval` — a coarse timer
   (e.g. a constrained edge device's power-saving tick) inflates the
   realized worst case proportionally.
3. Crash-before-seal is unchanged and NOT fixed: in-memory-only pending
   state, identical exposure to ADR-347's `BatchTail`.
4. `max_wait_ns = 0` is accepted (closes on the very next check), not
   rejected by a dedicated error — documented, not guarded.

## Migration

None required. Existing `PerRecord`/`BatchTail` callers are unaffected.
Adopting `BoundedBatchTail` is a call-site change
(`SigningStrategy::BatchTail { batch_size }` →
`SigningStrategy::BoundedBatchTail { batch_size, max_wait_ns }`) plus
wiring a periodic `seal_if_due` call into whatever scheduler the
deployment already runs.

## Rollback

Revert to `SigningStrategy::BatchTail { batch_size }` at the call site;
no data migration needed since the on-chain/signed `SignedSpan` shape is
identical either way. The new code (`BoundedBatchTail` variant,
`seal_if_due`, `pending_opened_at_ns`) can be deleted without touching
`PerRecord`/`BatchTail` behavior at all, since it is additive.

## Rejection Criteria

This would have been REJECTED if any of: a produced chain under
`BoundedBatchTail` failed `verify_signed_chain` in any trial;
`BoundedBatchTail`'s steady-state overhead exceeded the pre-registered
+20% gate; the legacy `BatchTail` worst-case latency in the gap
experiment did NOT grow with elapsed idle time (which would have meant
the "unbounded" premise itself was wrong); or `BoundedBatchTail`'s
worst-case latency in the same experiment exceeded its computed upper
bound (`max_wait_ns + poll_interval + burst_span`). None of these
occurred; see Evidence.

## Open Questions

1. Does the bound hold under a real OS timer thread with real scheduling
   jitter, not this run's deterministic simulated-clock driver?
2. What `(batch_size, max_wait_ns, poll_interval)` combination is
   actually right for a given production arrival-rate distribution? Not
   answered here by design (see Alternatives/Implementation Plan).
3. Should `ruvector-retrieval-receipt`'s own `BatchFillPolicy::hybrid`
   eventually be reimplemented in terms of this simpler inline/
   `seal_if_due` split, unifying the two crates' now-parallel
   bounded-batching designs behind one shared pattern description (not
   shared code)?

## References

- ADR-347 — `SignedWitnessSink` base this extends.
- ADR-345 — `witnessed_compaction`, now composed end to end with signing.
- ADR-134 — witness record schema (`timestamp_ns` field reused here).
- `docs/research/nightly/2026-10-14-bounded-witness-batch-tail/README.md` — full evidence and methodology.
- `docs/research/nightly/2026-10-14-bounded-witness-batch-tail/gist.md` — standalone technical write-up.
- `docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md` — source of this ADR's Next Research items 1 and 6.
- `crates/ruvector-retrieval-receipt/src/batch_fill.rs` — the pattern this ADR's design mirrors (not a dependency).
