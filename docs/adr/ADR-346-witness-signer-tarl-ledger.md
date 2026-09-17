# ADR-346: Ed25519 `WitnessSigner` for the TARL Ledger's Witness Chain

## Status

Accepted. New, opt-in, non-default module
(`ruvector-agent-memory::witness_signing`), no feature flag needed (its
sole dependency, `rvf-types` with the `ed25519` feature, was already an
unconditional dependency of the crate). No existing behavior changes.

## Context

`ruvector-agent-memory`'s TARL ledger (`ledger.rs`, ADR-307) chains every
witness record (ADR-134 schema) with keyless FNV-1a. `ops.rs`'s own
tamper-evidence note is explicit: this chain is "tamper-EVIDENT against
*accidental corruption and naive edits only*... Real tamper evidence
against a log-writing adversary requires ADR-134's `WitnessSigner` escape
hatch (§9, HMAC/Ed25519). Wiring a `WitnessSigner` through `WitnessSink`
is an explicit follow-up gate that MUST land before WP8 cross-repo
anchoring." `ledger.rs`'s module doc repeats the same pointer. The
2026-09-05 nightly (`docs/research/nightly/2026-09-05-mincut-gated-forgetting`)
named it again as Next Research item 4, specifically for eviction
witnesses.

Meanwhile, one layer up the stack, `ruvector-retrieval-receipt` already
built and benchmarked exactly this per-record-vs-batch-amortized Ed25519
signing tradeoff for retrieval receipts (ADR-340, ADR-343). The admission
ledger beneath it — the thing a retrieval receipt's provenance chain would
ultimately need to trust — remained unsigned. This ADR closes that gap
using the same tradeoff analysis, applied to `ledger.rs`.

`ruvector-agent-memory` already depends unconditionally on `rvf-types`
with its `ed25519` feature (ADR-320, for `AtomicObservation` signatures in
`fusion.rs`/`observation.rs`), so no new dependency is required.

## Hypothesis

```text
Given the TARL ledger's existing FNV-1a witness chain and its own
documented ADR-134 WitnessSigner gap,

when witness records are Ed25519-signed either per-record or via an
amortized batch-tail-only strategy (reusing rvf-types's existing Ed25519
primitive, no new dependency),

then batch-tail signing should reduce mean per-operation latency
substantially relative to per-record signing, while both strategies
retain identical detection of a "diligent" forgery (a fully
self-consistent, forward-recomputed chain edit) that the existing
unsigned chain-walk alone does not detect,

subject to: zero false negatives and zero false positives across the full
test and benchmark matrix, with every latency/throughput number coming
from an actual `cargo run --release` execution.
```

Full methodology and raw benchmark output are in
`docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`.

## Decision

1. Add `ruvector-agent-memory::witness_signing`:
   `SignedWitnessSink<S: WitnessSink>` (a `WitnessSink` decorator),
   `SigningStrategy::{PerRecord, BatchTail { batch_size }}`,
   `SignedSpan`, `SignPurpose`, and `verify_signed_chain`. Reuses
   `rvf-types::ed25519` verbatim; introduces no new cryptographic
   primitive and no new dependency.
2. `TransactionalLedger` gains `into_witness_sink(self) -> S`, needed to
   recover a wrapped sink's signed spans after a run.
3. Not feature-gated: unlike `graph_forget` (ADR-345, gated behind
   `mincut-forget` because it pulls in `ruvector-mincut`), this module
   adds no new dependency edge, so it is always compiled — callers opt in
   by constructing a `SignedWitnessSink` at their `TransactionalLedger::new`
   call site; everyone else is unaffected.
4. Recommend `BatchTail` (tuned batch size) for throughput-sensitive
   deployments and `PerRecord` where immediate per-record signature
   availability matters more than throughput. Neither is set as a crate
   default — both require explicit opt-in.

## Evidence

Full raw output is in the linked nightly README; summarized here (one
`cargo run --release` execution, `N_ENTRIES=20,000`, 40,000 witness
records):

| Variant | Mean latency/op | Throughput | Signatures | Correctness |
|---|---|---|---|---|
| baseline (unsigned) | 1.22µs | 800,197 ops/s | 0 | PASS |
| `PerRecord` | 137.42µs | 7,275 ops/s | 40,000 | PASS |
| `BatchTail{16}` | 9.65µs | 103,206 ops/s | 2,500 | PASS |
| `BatchTail{64}` | 3.77µs | 262,868 ops/s | 625 | PASS |
| `BatchTail{256}` | 2.01µs | 489,882 ops/s | 157 | PASS |

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| Correctness (5 variants) | 5/5 pass | 5/5 | PASS |
| Diligent-forgery rejection | 2/2 strategies reject | 2/2 | PASS |
| Amortization (`BatchTail{64}` vs `PerRecord`) | >= 5x lower mean latency | 36.4x | PASS |
| Signature-count exactness | `40000/batch_size` | exact at 16 and 64 | PASS |
| No pre-existing test regressed | 28/28 pass | 28/28 | PASS |

**ACCEPT.** All mandatory gates pass. `cargo test -p ruvector-agent-memory
--lib`: 34 passed, 0 failed (28 pre-existing + 6 new).

The security-relevant result: a "diligent forgery" was constructed by
editing one interior witness record's `payload` and then recomputing
every downstream `record_hash`/`prev_hash`/`chain_hash` exactly as the
ledger would, producing a fully self-consistent alternate log. This
forgery **passes** the existing unsigned `MemoryWitnessLog::verify_chain()`
(confirming the crate's own documented gap is real, not theoretical) and
**fails** `verify_signed_chain` under both `PerRecord` and `BatchTail` in
every tested trial.

## Consequences

- `ruvector-agent-memory` gains a working, tested, always-compiled
  primitive that composes with any existing `WitnessSink`, including the
  ADR-345 `witnessed_compaction` eviction-witness path — no code change
  needed there for it to apply.
- No existing behavior changes: `TransactionalLedger` callers not opting
  in are unaffected; `MemoryWitnessLog`, `NoopWitnessSink`, and every
  existing `WitnessSink` implementor are untouched.
- `ruvector-agent-memory`'s admission history can now be signed at the
  source, closing a gap that sat directly beneath the already-signed
  `ruvector-retrieval-receipt` provenance lineage (ADR-340/343).
- Key management (generation, rotation, storage, revocation) is
  explicitly out of scope and left to callers, matching the equivalent
  disclaimer already in `ruvector-retrieval-receipt::signing`.

## Alternatives Considered

- **Depend on `ruvector-retrieval-receipt`'s `Issuer`/`BatchAnchor`
  directly.** Rejected: would add a real new dependency edge for no
  benefit, since `rvf-types` already provides Ed25519 in this crate's
  existing dependency tree, and `BatchAnchor`'s Merkle-proof machinery
  solves random-access inclusion proofs into an *unordered* batch — this
  ledger's records are already ordered and hash-chained, so `prev_hash`
  gives inclusion "proof" for free.
- **Sign `record_hash` (48-byte-input hash) instead of `chain_hash`
  (64-byte, includes `prev_hash`/`aux`).** Rejected: `chain_hash` is the
  field that propagates into the next record's `prev_hash`, so it is what
  makes a `BatchTail` signature transitively cover the whole batch;
  signing `record_hash` would leave `aux` and the chain link itself
  outside the signed statement.
- **A wall-clock batch-fill timeout for `BatchTail`, ported from
  `ruvector-retrieval-receipt::batch_fill`.** Deferred to Next Research,
  not rejected — out of scope for keeping this pass's benchmark matrix
  and acceptance gates tractable in one run.
- **Attempt to measure or bound the FNV-1a chosen-target second-preimage
  cost (`ops.rs`'s "~2^32" claim) as part of this pass.** Explicitly
  declined — see the linked README's "What This Run Does Not Claim". This
  ADR's security argument does not depend on that figure being accurate
  in either direction; it is orthogonal (Ed25519 unforgeability holds
  independent of the inner hash's own strength).

## Implementation Plan

Already implemented in this PR:

- `crates/ruvector-agent-memory/src/witness_signing.rs` (new module, 6
  unit tests)
- `crates/ruvector-agent-memory/src/ledger.rs`: added
  `TransactionalLedger::into_witness_sink`
- `crates/ruvector-agent-memory/src/lib.rs`: `pub mod witness_signing` +
  re-exports
- `crates/ruvector-agent-memory/examples/witness_signing_bench.rs`
  (benchmark binary, no feature gate required)

No further implementation is planned under this ADR; see "Next Research"
in the nightly README for follow-up scope.

## API Shape

```rust
pub enum SignPurpose { PerRecord = 1, BatchTail = 2 }

pub struct SignedSpan {
    pub purpose: SignPurpose,
    pub covers_from_seq: u64,
    pub covers_to_seq: u64,
    pub chain_hash: u64,
    pub signature: [u8; 64],
}
impl SignedSpan {
    pub fn verify(&self, public_key: &[u8; 32]) -> bool;
}

pub enum SigningStrategy {
    PerRecord,
    BatchTail { batch_size: usize },
}

pub struct SignedWitnessSink<S: WitnessSink> { /* .. */ }
impl<S: WitnessSink> SignedWitnessSink<S> {
    pub fn new(inner: S, keypair: Ed25519Keypair, strategy: SigningStrategy) -> Self;
    pub fn public_key(&self) -> [u8; 32];
    pub fn spans(&self) -> &[SignedSpan];
    pub fn inner(&self) -> &S;
    pub fn flush(&mut self);
}
impl<S: WitnessSink> WitnessSink for SignedWitnessSink<S> { /* .. */ }

pub fn verify_signed_chain(
    log: &MemoryWitnessLog,
    spans: &[SignedSpan],
    public_key: &[u8; 32],
) -> bool;

// ledger.rs addition:
impl<S: WitnessSink, G: ProofGate> TransactionalLedger<S, G> {
    pub fn into_witness_sink(self) -> S;
}
```

## Feature Flags

None added. `rvf-types`'s `ed25519` feature was already unconditionally
enabled for this crate; `witness_signing` is always compiled, with opt-in
at the call site (construct `SignedWitnessSink` or don't).

## Benchmark Evidence

See "Evidence" above and the linked nightly README for full raw output
and methodology.

## Security

- Reuses `rvf-types::ed25519` (`ed25519-dalek`) verbatim; no new
  cryptographic primitive.
- Domain-separated by a crate-specific tag plus `SignPurpose`, preventing
  a `PerRecord` signature from being replayed as a `BatchTail` signature
  or vice versa.
- `verify_signed_chain` is the sole security-load-bearing function: it
  cross-binds each signed span's `chain_hash` to what the log's record at
  that sequence hashes to *right now*, not merely to what the span claims
  — this is what defeats the diligent-forgery scenario in Evidence above.
  `SignedSpan::verify` alone (without this cross-check) is documented as
  insufficient in isolation.
- Explicitly out of scope: FNV-1a preimage resistance of the inner chain
  hash itself (see Alternatives Considered), and all key lifecycle
  management.

## Governance

None beyond the existing "no witness, no mutation" invariant:
`SignedWitnessSink::emit_batch` forwards to the inner sink before signing,
so a refused batch is never signed.

## Failure Modes

See the nightly README's "Failure Modes" section: `PerRecord`'s ~110x
throughput cost vs. baseline; `BatchTail`'s unsigned-availability window
for records inside an unclosed batch (no wall-clock timeout in this pass);
`BatchTail`'s larger blast radius if the signer crashes mid-batch
(qualitative, not fault-injection-measured in this pass).

## Migration

None: purely additive. No existing `WitnessSink` implementor or
`TransactionalLedger` caller changes behavior.

## Rollback

Remove `witness_signing.rs`, its `lib.rs` wiring, and
`into_witness_sink`. No caller in this repository currently depends on
either (this ADR introduces the first usage), so rollback has zero
blast radius.

## Rejection Criteria

This ADR's hypothesis would have been rejected had any of:

1. The diligent forgery failed to fool the unsigned baseline (would have
   contradicted the crate's own existing tamper-evidence documentation).
2. Either signing strategy failed to reject the diligent forgery.
3. `BatchTail{64}` failed to beat `PerRecord` by at least 5x mean latency.

None occurred; see Evidence.

## Open Questions

1. What is the actual cost of a chosen-target FNV-1a second-preimage
   attack against a `LedgerWitnessRecord` (the crate's own "~2^32"
   estimate, unverified by any nightly run to date)? (Next-research item
   2 in the linked README; explicitly out of this ADR's scope.)
2. Should `BatchTail` gain a wall-clock fill timeout, and at what default,
   before any deployment with a bounded-latency requirement adopts it?
   (Next-research item 1.)
3. Does `verify_signed_chain` need a multi-issuer variant for federated
   or swarm agent-memory scenarios? (Next-research item 5; out of this
   ADR's scope to answer.)
