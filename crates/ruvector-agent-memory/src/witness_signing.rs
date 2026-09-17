//! Ed25519 signing for the TARL witness chain — the ADR-134 §9
//! `WitnessSigner` follow-up gate named (but not implemented) by
//! [`crate::ops`]'s tamper-evidence note and [`crate::ledger`]'s WP8
//! comment. Reuses this crate's existing `rvf-types` Ed25519 primitive
//! (ADR-320) rather than adding a new signing dependency.
//!
//! ## What signing actually buys you here
//!
//! [`crate::ops::MemoryWitnessLog::verify_chain`] already walks the FNV-1a
//! chain and catches any record whose stored bytes were edited without
//! also consistently recomputing every `record_hash`/`prev_hash` downstream
//! of it — call this a *naive* tamper. Its own doc comment is explicit
//! that it does NOT catch a *diligent* tamper: an adversary with write
//! access to the log who edits one record and then recomputes the whole
//! downstream chain to match, producing an internally self-consistent but
//! semantically different log. [`verify_signed_chain`] closes exactly that
//! gap, IF AND ONLY IF the adversary cannot also forge a valid Ed25519
//! signature over the tampered record's new `chain_hash` — which holds
//! unconditionally under standard Ed25519 unforgeability, given the
//! signing key stays secret.
//!
//! What this module does NOT claim: it does not evaluate whether an
//! adversary could instead find a *different* 64-byte record whose FNV-1a
//! `chain_hash` collides with the original (a preimage attack on the
//! per-record hash itself, independent of signing). `ops.rs` informally
//! estimates that at "~2^32 work"; this nightly run did not attempt to
//! reproduce or falsify that specific claim (see the research README's
//! Next Research section) — a diligent forgery is defined here as one that
//! changes the target record's `chain_hash` value, which is the case any
//! such preimage attack would need to avoid.
//!
//! ## Two strategies
//!
//! - [`SigningStrategy::PerRecord`]: sign every witness record's own
//!   `chain_hash` as it is emitted. Strongest coverage — a signature exists
//!   the instant a record is durable — at the cost of one signature per
//!   record.
//! - [`SigningStrategy::BatchTail`]: sign only the last record's
//!   `chain_hash` in every `batch_size`-record run, amortizing signing
//!   cost. Because `chain_hash` embeds `prev_hash`, authenticating the tail
//!   record transitively covers every record in the batch, PROVIDED the
//!   verifier also runs `verify_chain` (which checks the walked chain
//!   terminates at the log's actual newest record) — signing the tail
//!   alone, without a full chain walk, would NOT catch a truncation of the
//!   batch's own interior. The cost: records inside an unclosed batch have
//!   no signature yet ([`SignedWitnessSink::flush`] closes a partial batch
//!   at shutdown), and losing the signer mid-batch leaves the whole batch
//!   unsigned rather than partially covered — see the research README's
//!   Failure Modes section for the measured tradeoff.

use crate::ops::{LedgerError, LedgerWitnessRecord, MemoryWitnessLog, WitnessSink};
use rvf_types::ed25519::{ed25519_sign, ed25519_verify, Ed25519Keypair};

const DOMAIN_TAG: &[u8] = b"ruvector-agent-memory:witness-signer:v1:";
/// `purpose` (1 byte) + `from`/`to`/`chain_hash` (3 × 8-byte LE `u64`).
const MESSAGE_LEN: usize = DOMAIN_TAG.len() + 25;

/// Distinguishes a per-record signature from a batch-tail signature so a
/// signature produced for one purpose can never be replayed as the other,
/// even if the covered range and chain hash happened to coincide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum SignPurpose {
    PerRecord = 1,
    BatchTail = 2,
}

/// One Ed25519-signed statement: "the witness records with sequence
/// numbers `[covers_from_seq, covers_to_seq]` exist, and the record at
/// `covers_to_seq` has `chain_hash`."
#[derive(Clone, Copy, Debug)]
pub struct SignedSpan {
    pub purpose: SignPurpose,
    pub covers_from_seq: u64,
    pub covers_to_seq: u64,
    pub chain_hash: u64,
    pub signature: [u8; 64],
}

impl SignedSpan {
    fn message(purpose: SignPurpose, from: u64, to: u64, chain_hash: u64) -> [u8; MESSAGE_LEN] {
        let mut m = [0u8; MESSAGE_LEN];
        let mut off = 0;
        m[off..off + DOMAIN_TAG.len()].copy_from_slice(DOMAIN_TAG);
        off += DOMAIN_TAG.len();
        m[off] = purpose as u8;
        off += 1;
        m[off..off + 8].copy_from_slice(&from.to_le_bytes());
        off += 8;
        m[off..off + 8].copy_from_slice(&to.to_le_bytes());
        off += 8;
        m[off..off + 8].copy_from_slice(&chain_hash.to_le_bytes());
        m
    }

    /// Verify this span's signature in isolation (does not check that
    /// `chain_hash` matches any particular log's current content — use
    /// [`verify_signed_chain`] for that).
    pub fn verify(&self, public_key: &[u8; 32]) -> bool {
        let msg = Self::message(
            self.purpose,
            self.covers_from_seq,
            self.covers_to_seq,
            self.chain_hash,
        );
        ed25519_verify(public_key, &msg, &self.signature)
    }
}

/// Which records get their own signature vs. share an amortized one.
#[derive(Clone, Copy, Debug)]
pub enum SigningStrategy {
    PerRecord,
    /// Sign the last record's `chain_hash` once every `batch_size`
    /// records. `batch_size` must be at least 1.
    BatchTail {
        batch_size: usize,
    },
}

/// A [`WitnessSink`] decorator that signs every record (or amortized
/// batch tail) it forwards to an inner sink. Still satisfies the
/// `WitnessSink` contract unmodified: signing cannot cause `emit_batch` to
/// fail (it is a deterministic, infallible local computation), so this
/// wrapper neither weakens nor strengthens the ledger's "no witness, no
/// mutation" guarantee — it only adds signatures alongside.
pub struct SignedWitnessSink<S: WitnessSink> {
    inner: S,
    keypair: Ed25519Keypair,
    strategy: SigningStrategy,
    pending_from: Option<u64>,
    pending_last: Option<(u64, u64)>, // (sequence, chain_hash) of the newest unsigned record
    pending_count: usize,
    spans: Vec<SignedSpan>,
}

impl<S: WitnessSink> SignedWitnessSink<S> {
    pub fn new(inner: S, keypair: Ed25519Keypair, strategy: SigningStrategy) -> Self {
        if let SigningStrategy::BatchTail { batch_size } = strategy {
            assert!(batch_size >= 1, "batch_size must be at least 1");
        }
        Self {
            inner,
            keypair,
            strategy,
            pending_from: None,
            pending_last: None,
            pending_count: 0,
            spans: Vec::new(),
        }
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.keypair.public_key()
    }

    /// Every span signed so far, in emission order.
    pub fn spans(&self) -> &[SignedSpan] {
        &self.spans
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }

    fn sign_span(&mut self, purpose: SignPurpose, from: u64, to: u64, chain_hash: u64) {
        let msg = SignedSpan::message(purpose, from, to, chain_hash);
        let signature = ed25519_sign(&self.keypair.secret_key(), &msg);
        self.spans.push(SignedSpan {
            purpose,
            covers_from_seq: from,
            covers_to_seq: to,
            chain_hash,
            signature,
        });
    }

    /// Sign whatever `BatchTail` run is still open (end-of-run / shutdown).
    /// A no-op under `PerRecord` (nothing is ever left pending) or when
    /// nothing has been emitted since the last close.
    pub fn flush(&mut self) {
        if let (Some(from), Some((to, hash))) = (self.pending_from.take(), self.pending_last.take())
        {
            self.sign_span(SignPurpose::BatchTail, from, to, hash);
        }
        self.pending_count = 0;
    }
}

impl<S: WitnessSink> WitnessSink for SignedWitnessSink<S> {
    fn emit_batch(&mut self, records: &[LedgerWitnessRecord]) -> Result<(), LedgerError> {
        // Witness-first: the inner sink commits before anything is signed,
        // so a refused batch is never signed either.
        self.inner.emit_batch(records)?;
        match self.strategy {
            SigningStrategy::PerRecord => {
                for r in records {
                    self.sign_span(
                        SignPurpose::PerRecord,
                        r.sequence,
                        r.sequence,
                        r.chain_hash(),
                    );
                }
            }
            SigningStrategy::BatchTail { batch_size } => {
                for r in records {
                    if self.pending_from.is_none() {
                        self.pending_from = Some(r.sequence);
                    }
                    self.pending_last = Some((r.sequence, r.chain_hash()));
                    self.pending_count += 1;
                    if self.pending_count >= batch_size {
                        let from = self.pending_from.take().expect("set above");
                        let (to, hash) = self.pending_last.take().expect("set above");
                        self.sign_span(SignPurpose::BatchTail, from, to, hash);
                        self.pending_count = 0;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Verify a signed witness log: the inner FNV-1a chain walk (catches a
/// naive tamper — any edit not also consistently recomputed downstream),
/// AND every signed span, cross-checked against what the log's record at
/// `covers_to_seq` ACTUALLY hashes to right now (catches a diligent tamper
/// — a fully self-consistent recompute that changes that record's
/// `chain_hash`). A span whose covered sequence is missing from the log
/// (e.g. a truncated tail) fails closed.
pub fn verify_signed_chain(
    log: &MemoryWitnessLog,
    spans: &[SignedSpan],
    public_key: &[u8; 32],
) -> bool {
    if !log.verify_chain() {
        return false;
    }
    for span in spans {
        if !span.verify(public_key) {
            return false;
        }
        let Some(rec) = log
            .records
            .iter()
            .find(|r| r.sequence == span.covers_to_seq)
        else {
            return false;
        };
        if rec.chain_hash() != span.chain_hash {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::TransactionalLedger;
    use crate::ops::MemoryWitnessLog;

    const TEST_SECRET: [u8; 32] = [7u8; 32];

    fn keypair() -> Ed25519Keypair {
        Ed25519Keypair::from_secret(&TEST_SECRET)
    }

    fn populated_sink(strategy: SigningStrategy, n: usize) -> SignedWitnessSink<MemoryWitnessLog> {
        let sink = SignedWitnessSink::new(MemoryWitnessLog::default(), keypair(), strategy);
        let mut ledger = TransactionalLedger::new(sink, crate::ledger::AlwaysAdmitGate::default());
        for i in 0..n {
            let id = ledger
                .add(format!("memory {i}"), &[], "actor", "reason")
                .expect("add succeeds");
            ledger
                .accept(id, "actor", "verified")
                .expect("accept succeeds");
        }
        // TransactionalLedger owns the sink privately; reconstruct is not
        // possible, so tests drive `SignedWitnessSink` directly instead of
        // through the ledger for anything needing post-hoc access. See
        // below: this helper only exists to prove the composition
        // typechecks and produces a verifiable chain end-to-end.
        let mut sink = TransactionalLedger::into_witness_sink(ledger);
        sink.flush();
        sink
    }

    #[test]
    fn per_record_honest_chain_verifies() {
        let sink = populated_sink(SigningStrategy::PerRecord, 12);
        let pk = sink.public_key();
        assert!(verify_signed_chain(sink.inner(), sink.spans(), &pk));
        // Every accepted `add` emits at least the `Add` witness; `accept`
        // adds one more, so PerRecord signs strictly more spans than there
        // are ledger entries.
        assert!(sink.spans().len() >= 12);
    }

    #[test]
    fn batch_tail_honest_chain_verifies() {
        let sink = populated_sink(SigningStrategy::BatchTail { batch_size: 5 }, 23);
        let pk = sink.public_key();
        assert!(verify_signed_chain(sink.inner(), sink.spans(), &pk));
        // Amortization must actually reduce signature count vs PerRecord.
        let per_record = populated_sink(SigningStrategy::PerRecord, 23);
        assert!(sink.spans().len() < per_record.spans().len());
    }

    #[test]
    fn wrong_public_key_fails_verification() {
        let sink = populated_sink(SigningStrategy::PerRecord, 5);
        let wrong_pk = Ed25519Keypair::from_secret(&[9u8; 32]).public_key();
        assert!(!verify_signed_chain(sink.inner(), sink.spans(), &wrong_pk));
    }

    #[test]
    fn naive_tamper_is_caught_by_chain_walk_alone() {
        let sink = populated_sink(SigningStrategy::PerRecord, 10);
        let mut forged = sink.inner().clone();
        forged.records[3].payload ^= 1; // edit one record, fix nothing downstream
        assert!(
            !forged.verify_chain(),
            "naive tamper must break the chain walk"
        );
    }

    #[test]
    fn diligent_forgery_defeats_chain_walk_alone_but_not_signatures() {
        for strategy in [
            SigningStrategy::PerRecord,
            SigningStrategy::BatchTail { batch_size: 4 },
        ] {
            let sink = populated_sink(strategy, 16);
            let pk = sink.public_key();
            assert!(verify_signed_chain(sink.inner(), sink.spans(), &pk));

            // A diligent adversary: edit one interior record's payload,
            // then recompute record_hash/chain_hash forward through every
            // subsequent record exactly as the ledger would, and fix up
            // the head commitment. This produces a log that is internally
            // self-consistent end to end.
            let mut forged = sink.inner().clone();
            let tamper_at = 6usize;
            forged.records[tamper_at].payload ^= 0xDEAD_BEEF;
            let mut prev_hash = if tamper_at == 0 {
                0
            } else {
                forged.records[tamper_at - 1].chain_hash()
            };
            for r in forged.records.iter_mut().skip(tamper_at) {
                r.prev_hash = prev_hash;
                r.record_hash = r.compute_record_hash();
                prev_hash = r.chain_hash();
            }
            forged.committed_head = prev_hash;
            forged.committed_count = forged.records.len() as u64;

            assert!(
                forged.verify_chain(),
                "a diligent, fully-recomputed forgery must pass the unsigned chain walk \
                 (this is the documented residual gap `verify_chain` alone leaves open)"
            );
            assert!(
                !verify_signed_chain(&forged, sink.spans(), &pk),
                "signatures ({strategy:?}) must catch what the chain walk alone cannot"
            );
        }
    }

    #[test]
    fn flush_signs_a_partial_batch_tail() {
        let mut sink = SignedWitnessSink::new(
            MemoryWitnessLog::default(),
            keypair(),
            SigningStrategy::BatchTail { batch_size: 100 },
        );
        let mut ledger = TransactionalLedger::new(sink, crate::ledger::AlwaysAdmitGate::default());
        for i in 0..7 {
            ledger.add(format!("m{i}"), &[], "a", "r").unwrap();
        }
        sink = TransactionalLedger::into_witness_sink(ledger);
        assert!(
            sink.spans().is_empty(),
            "batch of 100 must not have closed yet"
        );
        sink.flush();
        assert_eq!(
            sink.spans().len(),
            1,
            "flush must close the partial batch exactly once"
        );
        let pk = sink.public_key();
        assert!(verify_signed_chain(sink.inner(), sink.spans(), &pk));
    }
}
