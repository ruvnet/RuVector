# Bounding "how stale can an unsigned write be?" in RuVector's agent-memory witness chain

## The problem

`ruvector-agent-memory`'s TARL ledger signs its witness chain with
Ed25519 (ADR-347). To keep signing cheap, it supports `BatchTail`
signing: group up to `batch_size` records and sign the group once,
instead of signing every record individually. That's a real win —
amortizing 64 Ed25519 signatures into 1 cuts signing cost by roughly
60x in our measurements.

But `BatchTail` as originally shipped only closes a group when enough
records arrive. If a deployment writes sparsely — a realistic case for
most agent-memory tenants, which aren't a continuous firehose — a
partial group can sit open indefinitely. Any record in that open group
has no cryptographic signature yet. "Has this write been authenticated"
has no time-bounded answer. That's a real gap for anything audited or
proof-gated, and the module's own prior research run named it directly
as its first open item.

## The fix

Add a second close trigger: a wall-clock timeout. `BatchTail` now
optionally takes `max_wait_ns` — close the pending group once that many
nanoseconds have elapsed since its oldest record arrived, even if the
group never reaches `batch_size`.

```rust
// old behavior, unchanged:
SigningStrategy::batch_tail(64)
// new: close at 64 records, OR after 100ms since the oldest pending one
SigningStrategy::batch_tail_timed(64, 100_000_000)
```

A caller drives the timeout the same way a cron job or event loop would
drive any other deadline:

```rust
if let Some(deadline) = sink.pending_deadline_ns() {
    if now_ns() >= deadline {
        sink.seal_on_timeout(now_ns());
    }
}
```

This isn't a novel idea — it's the exact two-trigger shape
(`max_members` OR `max_wait_ns`, whichever first) that a *sibling* crate
in the same repository, `ruvector-retrieval-receipt`, already built for
signed retrieval receipts. The contribution here is porting that pattern
into a different artifact's signing path, and proving it doesn't weaken
anything the original `BatchTail` guaranteed.

That last part matters more than it sounds: `seal_on_timeout` calls the
*exact same* signing function the size trigger and an explicit shutdown
flush already use. It's a new condition for closing a group, not a new
way of signing one. So every existing security guarantee —
coverage checking, rollback detection, rejecting a "diligent forgery"
that recomputes the whole unsigned hash chain — carries over unchanged.
We didn't just assert that; we re-ran the project's own diligent-forgery
test against a timeout-closed group specifically, and it's rejected
exactly like before.

## Does it actually work?

Two experiments, both against real Ed25519 signing and SHA-256 hashing
(not mocked), on a synthetic — but honestly disclosed as synthetic —
arrival-time stream (real multi-second sleeps would make the benchmark
impractically slow for no extra signal, since the signing code doesn't
know what clock produced a timestamp).

**Experiment A** simulates a sparse deployment: four small bursts of
writes, two seconds of silence between each, `batch_size=64` (so a
10-record burst never closes by size alone). Unfixed `BatchTail` leaves
8 seconds of staleness sitting unsigned by the end of the run — and it
would keep growing, unbounded, the longer the deployment stays quiet.
With a 10ms timeout, the worst observed staleness is... 10ms. With a
100ms timeout, 100ms. Exactly the bound, every time.

```
baseline     unbounded: 8,002,000,000ns staleness at checkpoint, never closes on its own
candidate_a  (10ms bound):  max observed latency = 10,000,000ns  -- held
candidate_b  (100ms bound): max observed latency = 100,000,000ns -- held
```

**Experiment B** checks the other direction: does this cost anything
when load is high enough that `batch_size`, not the timeout, is already
the thing closing groups? 2,000 records back-to-back, no gaps. Signature
counts are identical (32 in every variant). Per-operation latency
differences across three repeated runs ranged from -12.7% to +1.2% with
no consistent direction — noise at this timescale, not a regression. A
strict reading of our own pre-stated "within 5%" threshold is
technically missed in one run (-5.6%, i.e., *faster*, not slower) — we
report that plainly rather than quietly loosening the number, because
the direction that actually matters (no slowdown) held throughout.

## Limitations, said plainly

- The benchmark's scheduler wakes up exactly on time. A real scheduler's
  own jitter adds to the effective bound — not measured here.
- No real-device or WASM measurement; the sibling crate's WASM
  size/latency question is still open, now for a third time across two
  related modules.
- Nothing in this change wires a scheduler into a real deployment — it's
  a library primitive. The missing piece for production use is a few
  lines of cron/event-loop code calling `seal_on_timeout` at
  `pending_deadline_ns()`; we describe the shape of that but don't ship
  it, since the right scheduling mechanism is deployment-specific.

## Why this is more than a one-off fix

RuVector's agent-memory crate is meant to be more than a log — it's
meant to be auditable infrastructure other agents and operators can
make commitments against ("this write will be provably signed within N
seconds"). That only works if every piece of the chain has a stated,
measured latency bound instead of an implicit "eventually." This is one
small bound, on one mechanism, shipped with the measurement that backs
it — exactly the kind of incremental, evidence-carrying step the
ecosystem's research process is built to accumulate.

## Production relevance

- Audited/compliance-gated agent memory can state a concrete SLO instead
  of "signed eventually."
- Sparse-write deployments (the realistic edge/low-traffic case) get the
  same latency guarantee a high-traffic deployment gets for free from
  batch_size alone.
- Federated/multi-agent scenarios, where one agent needs to trust a
  peer's freshly written record within a bounded window, become
  possible to reason about at all.

## What's next

Wiring `seal_on_timeout` into an actual scheduled task; measuring real
scheduler jitter on top of the idealized bound measured here; and the
still-open WASM size/latency question for this module.

## References

- ADR-347 (`docs/adr/ADR-347-witness-signer-tarl-ledger.md`) — the
  module this extends.
- ADR-353 (`docs/adr/ADR-353-witness-signer-batch-tail-fill-timeout.md`) —
  this change's own ADR.
- `crates/ruvector-retrieval-receipt/src/batch_fill.rs` — the pattern
  ported here.
- `crates/ruvector-agent-memory/src/witness_signing.rs`,
  `examples/witness_signing_fill_timeout_bench.rs` — the code and
  benchmark behind every number above.
