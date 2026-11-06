# Bounding the Worst Case of a Batch-Amortized Signature

*A nightly research note on `ruvector-agent-memory`'s `SignedWitnessSink`.*

## The problem

Signing every record in an append-only log is secure but slow: in
`ruvector-agent-memory`'s TARL witness chain, Ed25519-signing every
record (`SigningStrategy::PerRecord`) costs roughly 69µs per signature —
dominated entirely by the signature operation, not by hashing or
bookkeeping. A 2026-09-16 nightly run fixed this crate's previously
*unsigned* witness chain (keyless FNV-1a only — tamper-evident against
accidental corruption, not against a log-writing adversary) by adding
`SignedWitnessSink`, which can batch-amortize signing cost: sign once
every `batch_size` records instead of once per record. At `batch_size =
64`, that amortization is dramatic — roughly 36x lower mean latency per
operation in that run's measurement.

Batching has an obvious cost, though: a record isn't *signed* — isn't
covered by any verifiable statement — until its batch fills. If records
keep arriving steadily, the wait is short and bounded by the arrival
rate. But if arrivals pause — a quiet period with no new memories
admitted, no evictions, nothing — the partially-filled batch just sits
there, unsigned, for as long as the pause lasts. The prior run named this
precisely and moved on: "Failure Modes item 2: `BatchTail` availability
window... This implementation does not include a wall-clock fill
timeout... a deployment needing bounded worst-case signature latency
would need to add one." That's the gap this run closes.

## Why it matters

A batch-amortized signing scheme without a timeout makes an implicit,
unstated promise: "your record will be signed once enough other records
arrive." For a high-throughput system that's a promise that resolves in
microseconds. For a system with bursty, sparse activity — an agent that
goes idle between tasks, a compliance log that only grows when something
actually happens — that promise can resolve in minutes, hours, or
(worst case) never, if the deployment shuts down or the key holder never
calls the manual `seal()` escape hatch. An auditor asking "can you prove
this memory was signed within N seconds of being written?" gets no
answer from `BatchTail` alone — only "eventually, maybe."

## The fix

The fix doesn't touch the cryptography at all. It adds one new strategy,
`SigningStrategy::BoundedBatchTail { batch_size, max_wait_ns }`, that
closes a pending batch on whichever trigger fires first: `batch_size`
records accumulating (the existing mechanism, unchanged), or
`max_wait_ns` nanoseconds elapsing since the batch opened. The elapsed
time is derived from data each witness record already carries — its own
`timestamp_ns` field, already part of the ADR-134 record schema and
already inside the hashed, signed content — so no new field was added to
the trait that every sink in this crate implements.

Two checks enforce the bound:

- **Inline**, on every new arrival: if the new record's timestamp has
  already crossed the deadline, the batch closes immediately, signing up
  through that record.
- **Externally**, via a new `seal_if_due(now_ns)` method: a caller
  running any kind of periodic timer — a cron job, an async interval, a
  background thread — calls this on a schedule, and it closes an open
  batch whose deadline has passed even if no new record ever arrives to
  trigger the inline check. This is the case `BatchTail` alone cannot
  recover from: with nothing new coming in, nothing was ever going to
  look at the clock.

This mirrors a pattern a sibling crate, `ruvector-retrieval-receipt`,
already built and measured for a *different* signing lineage (query
retrieval receipts, not ledger admission records) — a hybrid
size-or-deadline `BatchFillPolicy`. The pattern was ported deliberately,
not the code: the two crates solve structurally different problems
(`ruvector-retrieval-receipt` separates batch-fill-decision bookkeeping
from cryptography into two collaborating types; `ruvector-agent-memory`'s
`SignedWitnessSink` already owns both in one type), and grafting one
crate's scheduler type onto the other's design would have meant
duplicating state rather than sharing it. Reusing the idea without taking
on a new dependency edge is exactly the kind of "don't force
integration" discipline this codebase's architecture decisions generally
favor.

## Measuring both the cost and the benefit

Two honest measurements were needed, and neither should be taken on
faith:

**Does the deadline check cost anything when it doesn't fire?** Run
20,000 sequential add+accept pairs through a real ledger — the same
methodology the prior nightly run used — comparing `BatchTail{64}`
against `BoundedBatchTail{64, max_wait_ns=5ms}`. In a tight loop, the
real wall-clock gap between operations is low microseconds; the 5ms
deadline essentially never fires, so this isolates the cost of the extra
comparison alone. Measured: statistically indistinguishable — across
four consecutive runs the difference ranged from −0.29% to +13.25%, with
no consistent direction, consistent with measurement noise rather than a
real regression, and every run stayed comfortably inside a 20%
acceptance threshold set in advance.

**Does the bound actually hold?** This needed a scenario `BatchTail`
would fail on. A deterministic, simulated-clock trace — no real
sleeping, fully reproducible — ran 10 cycles of a short burst of 5
records followed by a 20-millisecond silence, with `batch_size` set
deliberately enormous so it could never trigger by size alone, isolating
the timeout path entirely. A simulated periodic timer "polled"
`seal_if_due` every millisecond throughout — the same driver code for
both candidates, since `seal_if_due` is a confirmed no-op on plain
`BatchTail` (a dedicated unit test asserts this directly).

The result: `BatchTail`'s single pending span sat open for the entire
200-millisecond simulated run — every record in it, including the very
first, waited the full run length before an end-of-run `seal()` finally
signed it. `BoundedBatchTail` closed ten separate spans, one roughly per
burst, each within about six milliseconds of opening — bounded by the
5ms deadline plus the 1ms polling granularity, exactly as designed. The
gap between the two isn't fixed at whatever ratio this particular trace
produced (33x here); it grows without limit the longer a deployment runs
between bursts, because `BatchTail`'s "worst case" is, structurally,
"however long until someone happens to call `seal()`."

## What this doesn't fix

Bounding the *honest, running* worst case is not the same as making the
system durable. If the process crashes before a deadline fires — or
before `seal()` or `seal_if_due` ever runs — the open batch's records are
still unsigned, with no way to recover a signature after the fact short
of the key holder manually re-signing. That failure mode is completely
unchanged by this work, and is also unaffected by `batch_size` or
`max_wait_ns` choice — it's a property of in-memory pending state, not of
the closing policy. A deployment that cares about this needs separate
durability work (e.g. persisting pending-span state before a crash can
lose it), not a tighter timeout.

The realized bound is also only as tight as whatever timer actually
drives `seal_if_due`. This library deliberately does not spawn its own
background thread — a thread-spawning policy baked into a library used
by everything from a WASM module to an embedded edge device to a
single-threaded CLI would be a worse default than handing the deployment
a cheap, pure, idempotent-on-no-op function and letting it choose its own
timer mechanism. That choice means the bound is a *ceiling the mechanism
enforces*, not a guarantee the library can make unconditionally — a
deployment that adds `BoundedBatchTail` but never actually calls
`seal_if_due` on a schedule gets exactly `BatchTail`'s original
unbounded behavior, silently.

## Ecosystem implications

The fix connects cleanly to a second, previously-undemonstrated
composition: `ruvector-agent-memory`'s eviction-witnessing module
(`witnessed_compaction`, from an earlier — and, on its original
mincut-based design, rejected — nightly run; only the eviction-witnessing
half survived and shipped) already implements the same `WitnessSink`
trait `SignedWitnessSink` wraps. Mechanically, nothing needed to change
for `BoundedBatchTail` to protect eviction witnesses the same way it
protects admission witnesses — but nothing had tested that composition
end to end before. A new test now drives two real compaction passes
through a `BoundedBatchTail`-wrapped sink, including a deliberately
constructed "nothing else ever arrives" case for the second pass, and
confirms the resulting signed chain verifies.

For an edge or constrained deployment specifically, a bounded signing
window matters more than it might in a data center: such a device is
more likely to lose power or connectivity mid-session, and the
crash-before-seal gap (unaffected by this work) is exactly the failure a
shorter *normal-operation* exposure window partially mitigates — not by
closing the gap, but by shrinking how often a device is caught with a
large amount of as-yet-unsigned state when something goes wrong.

## Future directions

The simulated-clock driver in this benchmark proves the mechanism works;
it doesn't prove a real OS timer thread, with real scheduling jitter and
wakeup latency, holds the same bound under contention. That's the most
direct next step. A second is tuning: this run fixed `batch_size`,
`max_wait_ns`, and poll interval at one value each per experiment,
deliberately, to isolate what was being measured — a real deployment
would want these chosen against its own actual arrival-rate distribution,
which is a different (and legitimate) kind of experiment this one was
not designed to answer.
