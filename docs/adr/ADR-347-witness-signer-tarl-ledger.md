# ADR-347: Ed25519 `WitnessSigner` for the TARL Ledger's Witness Chain

## Status

Accepted. New, opt-in, non-default module
(`ruvector-agent-memory::witness_signing`). Revised 2026-09-26 after a
security review of PR #989 found four HIGH issues in the first design
(no coverage check, forgeable unsigned `BatchTail` tail, undetected
rollback, signatures over a 64-bit FNV-1a hash). This text describes the
revised design; the guarantees and limits below are the ones the code
actually enforces and tests. (Originally drafted as ADR-346; renumbered
because ADR-346 is allocated to the mincut partition ADR, PR #979.)

## Context

`ruvector-agent-memory`'s TARL ledger (`ledger.rs`, ADR-307) chains every
witness record (ADR-134 schema) with keyless FNV-1a. `ops.rs`'s
tamper-evidence note is explicit that this chain only detects accidental
corruption and naive edits: an adversary who can write the log can relink
it in one O(n) pass, and `ops.rs` estimates a targeted FNV-1a collision at
roughly 2^32 work. It names ADR-134's `WitnessSigner` (§9) as the follow-up
gate that must land before WP8 cross-repo anchoring.

`ruvector-agent-memory` already depends on `rvf-types` with its `ed25519`
feature (ADR-320) and on `rvf-types::sha256`, so both primitives are
available without adding new crates to the tree.

## Decision

1. Add `witness_signing` with `SignedWitnessSink<S: WitnessSink>` (a
   `WitnessSink` decorator), `SigningStrategy::{PerRecord, BatchTail {
   batch_size }}`, `SignedSpan`, `SignPurpose`, `SignedAnchor`,
   `verify_signed_chain`, and typed errors/report.
2. **What is signed.** Records are grouped into spans of consecutive
   sequence numbers (one record per span for `PerRecord`, up to
   `batch_size` for `BatchTail`). Per span:

   ```text
   records_digest = SHA-256(RECORDS_TAG || rec[from].to_bytes() || … || rec[to].to_bytes())
   message        = MSG_TAG || purpose || from || to || prev_link || records_digest
   link           = SHA-256(message)        // prev_link of the next span; [0;32] at genesis
   signature      = Ed25519(signing_key, message)
   ```

   Both tags are `v2` domain-separation strings specific to this crate,
   so no signature from the first (FNV-signing) design or from any other
   `rvf-types` user can be replayed. Each record's full canonical 64-byte
   encoding is inside a SHA-256 preimage, so the FNV-1a hashes carry no
   security weight in signed verification. `prev_link` chains spans with
   SHA-256, so spans cannot be reordered, dropped from the middle, or
   spliced from another log signed with the same key.
3. **Coverage (fail closed).** `verify_signed_chain` requires the spans to
   tile the log exactly: first span starts at sequence 0, each next span
   starts at the previous `to + 1`, every record's `sequence` equals its
   index, and nothing is left uncovered.
4. **Unsigned tail (fail closed, explicitly reported).** Records not yet
   covered by a closed span make verification return
   `Err(SignedChainError::UnsignedTail { signed, unsigned })`. This error
   is only produced after the signed prefix fully verified, so it
   precisely reports "prefix of `signed` records authentic, `unsigned`
   newest records NOT authenticated". Under `BatchTail` the caller must
   call `SignedWitnessSink::seal()` (signs the open partial batch) before
   verifying. We chose fail-closed over a success-with-a-count report so a
   caller that only checks `is_ok()` can never mistake unsigned records
   for verified ones.
5. **Rollback anchor.** `verify_signed_chain` takes a required
   `&SignedAnchor { record_count, head_digest }` — the signed analogue of
   `MemoryWitnessLog::head_commitment`, exported by
   `SignedWitnessSink::anchor()` and returned as
   `SignedChainReport::head`. Verification requires a span boundary at
   exactly `record_count - 1` whose chained `link` equals `head_digest`;
   the log may extend beyond it (append-only growth). Truncating log and
   span list together below the anchor fails with `AnchorMismatch`.
   `SignedAnchor::genesis()` is the explicit opt-out.
6. **Key handling.** The sink holds an `ed25519_dalek::SigningKey`
   (`new(inner, SigningKey, strategy)`; `from_keypair(&Ed25519Keypair)`
   copies the secret once) instead of copying the secret per signature.
   Verification parses the `VerifyingKey` once and uses `verify_strict`
   locally; shared `rvf-types::ed25519_verify` is unchanged for its other
   callers. `ed25519-dalek` becomes a direct dependency with the same
   spec `rvf-types` already uses (2.2.x in the lockfile) — no new crate.
7. **Sink hygiene.** `new` returns `Err(WitnessSignerError::ZeroBatchSize)`
   instead of panicking. `emit_batch` refuses (before forwarding to the
   inner sink, preserving "no witness, no mutation") any batch whose
   sequence numbers do not continue the signed chain contiguously — e.g.
   a second `TransactionalLedger` restarting at sequence 0 over the same
   sink.
8. `TransactionalLedger` gains `into_witness_sink(self) -> S` so a caller
   can `seal()` and read spans after a run.
9. Not feature-gated; callers opt in by constructing a
   `SignedWitnessSink`. Neither strategy is a default.

## Guarantees

Given a trusted public key and a trusted anchor, `Ok(report)` means:

- every record in the log (all `report.records_verified` of them) is
  byte-for-byte what the key holder signed, in the order it signed them;
- the log contains no record the key holder did not sign;
- the log is not shorter than the anchor and extends the anchored history;
- the unsigned chain walk (`MemoryWitnessLog::verify_chain`, which also
  binds the serde `evidence_grade` to the hashed `flags` nibble —
  `to_bytes()` excludes `evidence_grade`) passed.

Security reduces to SHA-256 collision resistance and Ed25519
(strict) unforgeability, given the signing key stays secret.

## Limits (explicitly not guaranteed)

- **Rollback above the anchor.** Records appended after the anchor was
  captured can be truncated together with their spans undetectably.
  Callers must persist `report.head` / `sink.anchor()` out-of-band after
  each verified run and pass it next time. With `genesis()`, any
  truncation to a span boundary verifies.
- **Crash before `seal()`.** Pending `BatchTail` state is in memory; a
  crash leaves the tail permanently unsigned. Verification fails closed
  (`UnsignedTail`); only a key holder can re-sign it. Deleting an unsigned
  tail entirely yields a log that verifies as its signed prefix — unsigned
  records were never authenticated, so their absence cannot be detected
  without a newer anchor.
- **Unsigned-availability window.** Under `BatchTail`, up to
  `batch_size - 1` committed records are unauthenticated until the batch
  closes or `seal()` runs. No wall-clock fill timeout in this pass.
- **Key lifecycle** (generation, rotation, storage, revocation,
  multi-issuer) is out of scope. A key holder can sign any history.
- **Only `MemoryWitnessLog`** is verifiable today; durable sinks need an
  equivalent reader.

## Evidence

Tests (`src/witness_signing_tests.rs`, 17 tests) include the three review
PoCs as regressions that must fail verification: (1) empty or partial
span list over a consistently re-hashed forged log; (2) forged / truncated
unsigned `BatchTail` tail, both unsealed (`UnsignedTail`) and sealed
(`RecordsMismatch` / `SpanBeyondLog`); (3) trailing spans dropped with the
log truncated to match (`AnchorMismatch` against the real anchor; verifies
with `genesis()`, documenting the limit). Also: same-key splice,
reordered spans, wrong key, naive tamper, zero batch size, non-contiguous
sequence refusal, anchor-then-grow, positive round-trips under both
strategies.

One `cargo run --release -p ruvector-agent-memory --example
witness_signing_bench` run of the revised design (N_ENTRIES=20,000,
40,000 witness records; single run on a shared workstation, so treat
figures as indicative):

| Variant | Mean latency/op | Throughput | Signatures | Correctness |
|---|---|---|---|---|
| baseline (unsigned) | 2.18µs | 452,099 ops/s | 0 | PASS |
| `PerRecord` | 87.54µs | 11,405 ops/s | 40,000 | PASS |
| `BatchTail{16}` | 6.49µs | 153,484 ops/s | 2,500 | PASS |
| `BatchTail{64}` | 3.29µs | 299,454 ops/s | 625 | PASS |
| `BatchTail{256}` | 1.92µs | 513,809 ops/s | 157 | PASS |

Diligent-forgery rejection: PASS for both strategies. Amortization
`PerRecord` vs `BatchTail{64}`: 26.7x. The nightly README
(`docs/research/nightly/2026-09-16-witness-signer-agent-memory/`) records
the first design's run and its threat-model statements, which predate
this revision.

## Consequences

- Admission history can be signed at the source, beneath the already
  signed `ruvector-retrieval-receipt` lineage (ADR-340/343) — subject to
  the anchor-persistence and seal requirements above.
- Purely additive for existing callers; `MemoryWitnessLog`,
  `NoopWitnessSink` and other `WitnessSink` implementors are untouched.

## Alternatives Considered

- **Sign the FNV-1a `chain_hash` (first design).** Rejected by review: a
  64-bit keyless hash is the thing signing must not depend on.
- **Report the unsigned tail as `Ok` with a count.** Rejected: a caller
  checking only `is_ok()` would treat unauthenticated records as
  verified.
- **Depend on `ruvector-retrieval-receipt`'s `Issuer`/`BatchAnchor`.**
  Rejected: new dependency edge; its Merkle machinery targets unordered
  batches, whereas these records are ordered and span-chained.
- **Wall-clock `BatchTail` fill timeout.** Deferred.

## API Shape

```rust
pub struct SignedSpan { pub purpose: SignPurpose, pub covers_from_seq: u64,
    pub covers_to_seq: u64, pub records_digest: [u8; 32], pub signature: [u8; 64] }
pub struct SignedAnchor { pub record_count: u64, pub head_digest: [u8; 32] }
impl SignedAnchor { pub const fn genesis() -> Self; }

impl<S: WitnessSink> SignedWitnessSink<S> {
    pub fn new(inner: S, key: SigningKey, s: SigningStrategy) -> Result<Self, WitnessSignerError>;
    pub fn from_keypair(inner: S, kp: &Ed25519Keypair, s: SigningStrategy) -> Result<Self, WitnessSignerError>;
    pub fn public_key(&self) -> [u8; 32];
    pub fn spans(&self) -> &[SignedSpan];
    pub fn anchor(&self) -> SignedAnchor;
    pub fn unsigned_pending(&self) -> usize;
    pub fn seal(&mut self);
    pub fn inner(&self) -> &S;
}

pub fn verify_signed_chain(log: &MemoryWitnessLog, spans: &[SignedSpan],
    public_key: &[u8; 32], anchor: &SignedAnchor)
    -> Result<SignedChainReport, SignedChainError>;
```

## Rollback

Remove `witness_signing.rs`, its tests, the `lib.rs` wiring, the direct
`ed25519-dalek` dependency, and `into_witness_sink`. No other caller in
the repository depends on them.

## Open Questions

1. Should `BatchTail` gain a wall-clock fill timeout, and at what default?
2. Should the anchor be signed and published (e.g. to a transparency log)
   so rollback above a locally persisted anchor becomes detectable?
3. Multi-issuer verification for federated / swarm agent memory.
