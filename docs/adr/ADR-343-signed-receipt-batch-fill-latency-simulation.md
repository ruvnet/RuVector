# ADR-343: Signed-Receipt Batch-Fill Latency — A Bounded Alternative to Fixed-Size-Only Batching

## Status

Proposed. Experimental crate extension (`ruvector-retrieval-receipt::batch_fill`
plus the `batch_latency` simulation binary), not wired into the default
query path of any production index. Adds a scheduling module alongside
ADR-340's `signing` module without modifying it, ADR-304's unsigned
receipts, or either module's existing tests.

## Context

ADR-340 implemented and benchmarked Ed25519 signing of `MerkleReceipt`
roots, both per-query and batched under one signature. It measured the
*CPU* cost of signing/verifying an already-assembled batch and explicitly
scoped out the *wall-clock* cost of assembling that batch from a live
query stream, naming this in its own Limitations section:

> No wall-clock batch-fill model. This benchmark measures CPU cost of
> signing/verifying assuming a batch is already fully assembled in memory.
> A real streaming deployment's end-to-end receipt-availability latency
> also includes however long it takes B queries to actually arrive — not
> modeled...

and again in Failure Modes:

> Batch never closes: a streaming deployment where queries arrive slower
> than the target batch size fills would delay signed-anchor availability
> indefinitely without a fill-timeout.

The 2026-08-31 nightly research README's "Next Research" section named
this as the first follow-up item verbatim:

> Model wall-clock batch-fill latency under a realistic query
> arrival-rate distribution, to turn the CPU-only amortization result
> here into an end-to-end latency claim.

This ADR implements that follow-up: a batch-fill scheduling policy that
bounds worst-case wait with a fill-timeout, and a discrete-event
simulation that combines real query-arrival timing with real
`Issuer`/`BatchAnchor` signing operations to produce a genuine end-to-end
receipt-availability latency measurement — not a CPU-only proxy for it.

## Hypothesis

```text
Given a stream of retrieval queries producing MerkleReceipt roots,
arriving under three tested load regimes — light Poisson (lambda=50 q/s),
target Poisson (lambda=1000 q/s), and bursty on/off Poisson (2000 q/s for
100ms, silent for 400ms, repeating) — each closed into a batch for
Ed25519 anchoring under one of three policies: baseline B=1 (immediate,
no wait), fixed-size-only B=32 (no timeout), and hybrid B=32 with a 50ms
fill-timeout,

when end-to-end receipt-availability latency is measured per query as
(batch-close decision time, chosen by the policy under real arrival
timing, plus the real measured Ed25519 batch-sign wall time) minus the
query's arrival time,

then the hybrid policy's p99 latency should stay within a fixed bound of
70ms (the 50ms timeout plus a fixed, not-tuned-post-hoc 20ms slack) at
every tested load regime, while fixed-size-only's p99 latency should
exceed twice the hybrid policy's p99 at the light-load regime —
demonstrating the exact unbounded-tail failure mode ADR-340 named but did
not measure,

subject to: every closed batch's signature and every member's inclusion
proof verifying correctly (100%), and the hybrid policy's amortized
signing cost at the target-load regime staying within 2x of
fixed-size-only's amortized cost at that same regime (the timeout safety
net must not destroy most of the amortization benefit when load is
sufficient to fill batches anyway).
```

Acceptance thresholds, fixed before this run:

1. 100% of closed batches verify (signature + all inclusion proofs), every
   regime and policy.
2. Hybrid p99 latency ≤ 70ms at all three regimes.
3. At light load, fixed-size-only's p99 latency > 2× hybrid's p99 at the
   same regime.
4. At target load, hybrid's amortized signing cost (ns/query) ≤ 2× fixed-
   size-only's amortized cost at the same regime.

## Decision

Add a new pure module, `batch_fill`, that decides *when* a batch of
pending receipt roots closes, independent of the cryptography:

- `BatchFillPolicy::fixed_size(n)` — closes only at `n` members (ADR-340's
  implicit policy, made explicit and reusable).
- `BatchFillPolicy::hybrid(n, max_wait_ns)` — closes at `n` members, or
  after `max_wait_ns` since the oldest pending member, whichever is first.
- `BatchScheduler` drives the policy against a stream of `arrive(...)`
  calls, exposing `oldest_pending_arrival_ns()` so a caller can derive the
  next timeout deadline without the scheduler owning a clock itself (it
  has none — this keeps the module synchronous, dependency-free, and
  unit-testable with plain integers).

Pair this with a new binary, `batch_latency`, that is a discrete-event
simulation: it generates real Poisson/bursty arrival timelines with a
seeded RNG, produces a real `MerkleReceipt` root per arrival via
`RetrievalIndex::search` + `RetrievalReceipt::build` (the same production
code path as ADR-304/ADR-340's benchmarks), and — when the scheduler
closes a batch — performs a **real** `BatchAnchor::build` +
`Issuer::sign_root` call, timed with `std::time::Instant`, whose measured
wall time is added to the batch's virtual close time to produce each
member's real availability time. No latency number in this ADR's evidence
is synthesized; every signing operation that contributes to a reported
latency is a real Ed25519 sign performed during the run.

## Threat Model

This ADR does not change ADR-340's threat model (origin authentication,
not issuer honesty; see that ADR). It adds one purely operational
property: a **latency bound**. A caller choosing `BatchFillPolicy::hybrid`
trades some amortization (batches close smaller/more often under light
load) for a guarantee that no query waits longer than `max_wait_ns` plus
one signing operation for its receipt to become available — closing the
"batch never closes" gap ADR-340 named as a failure mode rather than
fixed.

The simulation models a single serialized signer with no queueing delay
for the sign operation itself. This is accurate for every regime tested
here because real batch-sign cost (single-digit to tens of microseconds,
per ADR-340) is 3+ orders of magnitude below the shortest fill window
tested (the light-load hybrid regime's ~35ms mean batch-fill wait). It
would stop being accurate at arrival rates high enough that signing
itself becomes the bottleneck — not evaluated here; see Limitations in
the companion nightly research report.

## Evidence

Full methodology, raw output across 3 independent runs, and the complete
results table are in
`docs/research/nightly/2026-09-01-signed-receipt-batch-fill-latency/README.md`.
Summary of the headline result (n=2000, dims=64, k=10, 3000 queries per
regime, mean of 3 runs):

| regime | policy | p99 latency | verified |
|---|---|---:|---|
| target (1000 q/s) | B32_fixed_only | 36.80ms | 100% |
| target (1000 q/s) | B32_hybrid_50ms | 36.81ms | 100% |
| light (50 q/s) | B32_fixed_only | **756.13ms** | 100% |
| light (50 q/s) | B32_hybrid_50ms | **50.03ms** | 100% |
| bursty (on/off) | B32_fixed_only | 415.69ms | 100% |
| bursty (on/off) | B32_hybrid_50ms | 49.30ms | 100% |

All three runs: **ACCEPT** on every acceptance threshold above.

## Consequences

- **Positive:** a deployment can now choose a fill-timeout that bounds
  worst-case signed-receipt latency, with a measured (not assumed) cost
  in amortization loss at low load. The Failure Mode ADR-340 named is now
  mitigated by an implemented, tested policy rather than left open.
- **Positive:** the simulation methodology (real crypto ops driven by a
  virtual-time event schedule) is reusable for future ADR-340-adjacent
  questions — e.g. BLS aggregate signatures' batch-fill behavior, per
  ADR-340's Next Research item 2.
- **Negative:** `BatchFillPolicy::hybrid` requires a caller to pick
  `max_wait_ns`, a deployment-specific tuning parameter with a real
  tradeoff (this ADR measured one value, 50ms, at three regimes — it is
  not a universal default).
- **Negative:** the single-serialized-signer assumption (see Threat
  Model) means this ADR's latency numbers do not bound behavior at
  arrival rates where signing itself queues; that regime is untested.
- **Neutral:** no change to any existing public API in `signing` or
  `receipt`; `batch_fill` is additive.

## Alternatives Considered

- **Model batch-fill latency analytically** (e.g., an M/G/1-type queueing
  formula for the hybrid policy) instead of simulating it: rejected for
  this ADR because the discrete-event simulation already reuses real
  signing costs and real arrival generation with negligible extra
  engineering cost, and an analytical model would still need empirical
  validation against *something* — the simulation *is* that validation.
  An analytical model remains a reasonable follow-up to cross-check
  these numbers cheaply at parameter values not directly simulated.
- **Adaptive batch size** (grow/shrink `max_members` based on observed
  arrival rate) instead of a fixed hybrid timeout: a materially different
  policy requiring its own control-loop design and stability analysis —
  out of scope for a single nightly run; noted as a candidate follow-up.
- **BLS aggregate signatures** (ADR-340's own Rejected Alternative,
  restated as this run's own Next Research item 2): would let signatures
  be aggregated after the fact without a fill-timeout at all, potentially
  eliminating the tradeoff this ADR measures rather than mitigating it.
  Not implemented here — still requires a pairing-friendly curve
  dependency not currently in the workspace.

## Implementation Plan

1. `batch_fill.rs`: `BatchFillPolicy`, `PendingMember`, `BatchScheduler`
   (`new`, `arrive`, `close_on_timeout`, `flush`,
   `oldest_pending_arrival_ns`, `pending_len`). Seven unit tests covering
   fixed-size closure, hybrid timeout derivation and closure, flush
   draining, and fresh-batch-after-close behavior.
2. `bin/batch_latency.rs`: Poisson and bursty arrival generators (seeded
   xorshift, matching `bin/benchmark.rs`'s existing RNG pattern), a
   discrete-event loop driving `BatchScheduler` against real arrivals,
   real per-batch `BatchAnchor`/`Issuer` signing and verification, and a
   results table with an acceptance section mirroring
   `bin/benchmark.rs`'s existing format.
3. `lib.rs`: `pub mod batch_fill;` plus re-exports of
   `BatchFillPolicy`/`BatchScheduler`/`PendingMember`.
4. `Cargo.toml`: new `[[bin]] name = "batch_latency"` entry. No new
   dependencies — reuses `ed25519-dalek`/`sha2`/`rand` already declared
   for `signing`.

No changes to `signing.rs`, `receipt.rs`, or `index.rs`. ADR-304's and
ADR-340's existing 30 tests (23 pre-existing + 7 new in this ADR) all
pass unchanged — re-run and re-confirmed as a regression check, not
re-litigated.

## API Shape

```rust
pub struct BatchFillPolicy { pub max_members: usize, pub max_wait_ns: Option<u64> }
impl BatchFillPolicy {
    pub const fn fixed_size(max_members: usize) -> Self;
    pub const fn hybrid(max_members: usize, max_wait_ns: u64) -> Self;
}

pub struct PendingMember { pub query_index: usize, pub arrived_at_ns: u64 }

pub struct BatchScheduler { /* private */ }
impl BatchScheduler {
    pub fn new(policy: BatchFillPolicy) -> Self;
    pub fn arrive(&mut self, query_index: usize, arrived_at_ns: u64) -> Option<Vec<PendingMember>>;
    pub fn close_on_timeout(&mut self) -> Option<Vec<PendingMember>>;
    pub fn flush(&mut self) -> Option<Vec<PendingMember>>;
    pub fn oldest_pending_arrival_ns(&self) -> Option<u64>;
    pub fn pending_len(&self) -> usize;
    pub const fn policy(&self) -> BatchFillPolicy;
}
```

`BatchScheduler` owns no clock and performs no I/O or cryptography — a
caller supplies arrival timestamps and is responsible for calling
`close_on_timeout()` no earlier than
`oldest_pending_arrival_ns() + max_wait_ns`. This keeps the crate's only
async/real-time dependency (a clock source) at the call site, matching
the rest of this crate's synchronous, dependency-minimal design.

## Feature Flags

None. `batch_fill` is unconditionally compiled, matching `signing`'s
existing unconditional-compilation posture in this crate.

## Benchmark Evidence

- **Command:** `cargo run --release -p ruvector-retrieval-receipt --bin
  batch_latency -- 2000 64 10 3000`
- **Hardware/toolchain:** same environment as the paired nightly report;
  see that report for full hardware/rustc/repetition details.
- **Repetitions:** 3 full process runs; every acceptance threshold held
  in all 3. See the nightly report for the complete per-run table.

## Security

- No new cryptographic primitive: `batch_fill` contains no cryptography
  at all, and `batch_latency` uses `signing`'s existing `Issuer`/
  `BatchAnchor`/`verify_root` unmodified.
- No new dependency: `batch_latency`'s `Cargo.toml` entry adds a binary
  target, not a dependency.
- The simulation's arrival-time RNG is a plain xorshift for reproducible
  *timing*, not a security-relevant random source — key generation still
  uses `Issuer::generate()`'s `OsRng`, unchanged from ADR-340.

## Governance

Experimental, matching ADR-304's and ADR-340's posture: not on any
default query path, no production index adopts a fill-timeout as a
result of this ADR alone. A promotion decision for `BatchFillPolicy::
hybrid` at a specific `max_wait_ns` requires benchmark evidence against a
target deployment's actual arrival-rate distribution, not just this
synthetic Poisson/bursty workload.

## Failure Modes

- **Signer becomes the bottleneck at extreme arrival rates:** not
  modeled — see Threat Model. A deployment approaching this regime needs
  a queueing model for the signer itself, not just the batch-fill
  scheduler.
- **`max_wait_ns` chosen too small for the deployment's actual query
  rate:** degrades toward `B1_baseline`-like amortization (as observed at
  light load in this run's own evidence: mean batch size drops from 31.9
  to 3.5) without becoming *incorrect* — every closed batch, however
  small, still verifies. This is a performance/cost tradeoff, not a
  correctness risk.
- **`max_wait_ns` chosen too large for the deployment's latency SLA:**
  the mirror-image misconfiguration; the hybrid policy's bound is only as
  good as the timeout value operators actually choose.
- **Simulation-boundary flush:** the last partial batch in any finite run
  closes at the final arrival's timestamp rather than after its own
  timeout — a simulation artifact that does not affect steady-state
  behavior but means the very last batch's members do not exercise the
  simulated policy's actual close condition. Disclosed rather than
  corrected by discarding the tail (that would bias the sample instead).

## Migration

None — purely additive. No existing type, function, or test is modified.

## Rollback

Remove `batch_fill.rs`, the `pub mod batch_fill;` line and its re-exports
in `lib.rs`, `bin/batch_latency.rs`, and the `batch_latency` `[[bin]]`
entry in `Cargo.toml`. No other code references these additions.

## Rejection Criteria (Not Yet Triggered)

Production promotion of `BatchFillPolicy::hybrid` at any specific
`max_wait_ns` should be rejected if: a target deployment's real arrival
distribution produces a fill-timeout hit rate that destroys amortization
below an acceptable cost threshold; the single-serialized-signer
assumption is invalidated by the deployment's actual arrival rate
relative to real signing throughput; or a production-representative
benchmark (this ADR's workload remains synthetic Poisson/bursty, not
measured traffic) fails to reproduce the bound. None of these were
evaluated against a real deployment in this run.

## Open Questions

1. What `max_wait_ns` values are appropriate for real agent-memory query
   traffic shapes, as opposed to this run's synthetic Poisson/bursty
   approximations? Requires production traffic traces, not available to
   this run.
2. Does an adaptive batch-size policy (Alternatives Considered) dominate
   the fixed-hybrid policy across a wider range of arrival regimes, and
   by how much?
3. At what arrival rate does the single-serialized-signer assumption
   (Threat Model) break down, and what does end-to-end latency look like
   past that point?
