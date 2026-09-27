# Closing a Named Security Gap: Signing RuVector's Agent-Memory Witness Chain

## Problem

`ruvector-agent-memory` implements a TARL (Transaction-Aware Reliable
Ledgers) executable memory ledger: every admission, revision, rejection,
or acceptance of a piece of agent memory emits a witness record, chained
together with FNV-1a hashing. The crate's own source code says, in its
own words, that this chain is "tamper-EVIDENT against accidental
corruption and naive edits only... Real tamper evidence against a
log-writing adversary requires ADR-134's `WitnessSigner` escape hatch."
That escape hatch — an Ed25519 signature over the chain — was named in at
least two places in the codebase and one prior nightly research run, and
never built.

## Hypothesis

Wire it up two ways — sign every witness record individually, or sign
only the last record of every N-record batch — and measure whether the
batched approach's latency savings are real, and whether both approaches
actually deliver on the security property the unsigned chain is
documented as lacking.

## Technical Design

`ruvector-agent-memory` already depends on `rvf-types`, a sibling crate
providing Ed25519 signing (used elsewhere in the same crate for signing
individual memory observations). No new cryptographic library and no new
Cargo dependency was needed — just a new module,
`witness_signing.rs`, implementing a `WitnessSink` decorator:

```rust
pub enum SigningStrategy {
    PerRecord,
    BatchTail { batch_size: usize },
}

pub struct SignedWitnessSink<S: WitnessSink> { /* wraps any WitnessSink */ }
```

Every witness record already carries a `chain_hash` — an FNV-1a hash over
its full 64 bytes, which includes a `prev_hash` field pointing at the
previous record's `chain_hash`. That means signing just the *last*
record's `chain_hash` in a run of N records transitively authenticates
all N of them, as long as a verifier also independently walks the
unsigned chain to confirm it's actually linked correctly and hasn't been
truncated. `BatchTail` exploits exactly this, amortizing one signature
over many records; `PerRecord` signs every record as it lands, for
maximum immediacy at maximum cost.

The one subtlety that matters: a signed record's `chain_hash` has to be
cross-checked against what the log's record at that position *actually
hashes to right now* — not just trusted as whatever value sits in the
signed statement. Get that wrong and the whole scheme is bypassable by
an attacker who just keeps an old, honestly-signed statement around and
claims it covers new content. `verify_signed_chain` does this cross-check
explicitly; it's the one function in the module that actually matters for
security.

## Implementation

- `crates/ruvector-agent-memory/src/witness_signing.rs` — the module
  above, plus 6 unit tests.
- `crates/ruvector-agent-memory/src/ledger.rs` — one small addition,
  `into_witness_sink(self) -> S`, so a caller can recover a wrapped
  sink's signatures after driving a ledger run.
- `crates/ruvector-agent-memory/examples/witness_signing_bench.rs` — the
  benchmark below.

Zero Cargo.toml changes. Zero existing behavior changes: nothing that
doesn't explicitly opt into `SignedWitnessSink` is affected.

## The Actual Test: Can You Fool the Unsigned Chain?

Rather than just asserting "signing makes this more secure," the test
suite constructs an actual forgery. It builds an honest, signed witness
log, then plays the adversary: pick a record halfway through, flip a bit
in its `payload` field, and then — this is the important part — correctly
recompute every `record_hash`/`prev_hash`/`chain_hash` for every record
after it, exactly the way the ledger itself would. The result is a
completely self-consistent alternate history.

Running the existing, already-shipped `MemoryWitnessLog::verify_chain()`
against this forged log: **it passes.** This isn't a surprise — it's
exactly what the crate's own documentation already said would happen —
but it's now something you can watch happen in a test, not just take on
faith from a code comment.

Running `verify_signed_chain` (the new function) against the same forged
log, with the original signatures: **it fails**, for both `PerRecord` and
`BatchTail`. That's the whole point of the exercise, and it now has an
executable proof rather than a documentation comment.

## Benchmark

20,000 sequential `add` + `accept` operations (40,000 witness records),
release build, real wall-clock timing, deterministic signing key:

```
baseline       mean=   1.220us  throughput=  800197.4 ops/s  signatures=      0
candidate_a    mean= 137.423us  throughput=    7274.7 ops/s  signatures=  40000   (PerRecord)
candidate_b16  mean=   9.651us  throughput=  103205.5 ops/s  signatures=   2500   (BatchTail, batch=16)
candidate_b64  mean=   3.774us  throughput=  262867.7 ops/s  signatures=    625   (BatchTail, batch=64)
candidate_b256 mean=   2.011us  throughput=  489882.3 ops/s  signatures=    157   (BatchTail, batch=256)
```

Per-record signing costs about 110x the unsigned baseline's throughput —
each Ed25519 signature in this environment costs roughly 69 microseconds,
and at two witness records per logical operation, that adds up fast.
Batching at 64 records per signature recovers most of that throughput
(36x faster than per-record signing) while — per the forgery test above —
providing the exact same tamper-detection guarantee against a diligent,
fully-recomputed attacker. The tradeoff `BatchTail` actually pays for that
speedup isn't weaker security; it's availability: a record inside a batch
that hasn't closed yet has no signature at all until it does.

## Limitations

This work does not attempt to determine the true difficulty of forging a
*specific* target `chain_hash` via an FNV-1a second-preimage attack — a
figure the crate's existing documentation estimates at "~2^32 work" but
which no prior work in this repository has actually verified. Getting
that number right requires real cryptanalysis, and getting it wrong in
either direction (overstating or understating the difficulty) would be
worse than not stating it at all, so this run explicitly declines to
guess. It doesn't change this work's conclusion — Ed25519 signature
unforgeability holds regardless of whatever the inner hash's own
weaknesses turn out to be — but it does mean the question "is the
*unsigned* chain safer than the doc comment suggests" remains open for a
future, properly-scoped pass.

Also out of scope: concurrent-writer behavior (the ledger is
single-threaded by construction, unchanged by this work), WASM binary
size (an already-twice-deferred question for the sibling
`ruvector-retrieval-receipt` crate's own signing work), and fault
injection to quantify `BatchTail`'s larger blast radius if a signing
process crashes mid-batch.

## Production Relevance

This closes a specific, named gap between two already-built pieces of the
RuVector ecosystem: `ruvector-retrieval-receipt` already signs *what was
retrieved*; nothing signed *what was admitted* underneath it. An operator
who wants to prove a memory store's admission history wasn't rewritten
after an incident can now do that with a single, small, static public
key instead of needing to continuously re-anchor an ever-growing
`(count, hash)` commitment pair out-of-band — which was, until this
change, the documented alternative.

## RuVector Ecosystem Implications

The new `SignedWitnessSink<S: WitnessSink>` wraps *any* `WitnessSink`,
which means it composes for free with the eviction-witness path added by
a prior nightly run (`witnessed_compaction`, ADR-345) — signed eviction
receipts, an item that prior run's own "Next Research" section asked for,
now require zero new code, just wrapping the sink at the call site.

## Future Direction

1. A wall-clock batch-fill timeout for `BatchTail`, so a deployment gets
   a bounded worst-case signature-availability latency instead of
   "whenever the batch happens to fill" — reusing the methodology (not
   the code) already built for retrieval receipts.
2. A properly-scoped attempt at the FNV-1a second-preimage question left
   open above.
3. Concurrent-writer and fault-injection hardening.
4. Multi-issuer verification, for agent-memory scenarios where more than
   one writer needs independent attribution.

## References

- `docs/adr/ADR-134-witness-schema-log-format.md`
- `docs/adr/ADR-347-witness-signer-tarl-ledger.md` (this work's ADR)
- `docs/research/nightly/2026-08-31-signed-retrieval-receipts/README.md`
- `docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md`
- Full methodology, benchmark reproduction command, and complete
  evidence: `docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`
