//! Ed25519 signing for the TARL witness chain — the ADR-134 §9
//! `WitnessSigner` follow-up gate named by [`crate::ops`]'s tamper-evidence
//! note and [`crate::ledger`]'s WP8 comment (ADR-347).
//!
//! ## What is signed
//!
//! Records are grouped into *spans* of consecutive sequence numbers. For
//! each span the signer computes
//!
//! ```text
//! records_digest = SHA-256(RECORDS_TAG || rec[from].to_bytes() || … || rec[to].to_bytes())
//! message        = MSG_TAG || purpose || from || to || prev_link || records_digest
//! link           = SHA-256(message)             // prev_link of the next span
//! signature      = Ed25519(message)
//! ```
//!
//! Every record's full canonical 64-byte encoding is inside a SHA-256
//! preimage, so the per-record keyless FNV-1a `record_hash` / `chain_hash`
//! are NOT load-bearing for signed verification: forging a record means
//! finding a SHA-256 collision or an Ed25519 forgery, not a 64-bit FNV
//! collision. Each span also commits to the previous span's `link`
//! (`[0; 32]` at genesis), so spans form their own SHA-256 chain and cannot
//! be reordered, dropped from the middle, or spliced in from another log
//! signed with the same key.
//!
//! ## What [`verify_signed_chain`] guarantees
//!
//! Given a trusted public key and a trusted [`SignedAnchor`], `Ok(_)` means:
//!
//! 1. **Coverage.** The spans tile the log exactly: the first span starts
//!    at sequence 0, each next span starts one past the previous span's
//!    end, the last span ends at the newest record, and every record's
//!    `sequence` equals its index. An empty or partial span list never
//!    verifies a non-empty log.
//! 2. **Integrity.** Every record's full 64 bytes match what the key holder
//!    signed, and the span order is the order it signed them in.
//! 3. **No unsigned tail.** A record not yet covered by a closed span makes
//!    verification fail with [`SignedChainError::UnsignedTail`] — that
//!    error is only returned after the signed prefix has fully verified,
//!    so it precisely reports "prefix authentic, `unsigned` newest records
//!    unauthenticated". Under [`SigningStrategy::BatchTail`] callers must
//!    [`SignedWitnessSink::seal`] before verifying.
//! 4. **Rollback floor.** The log is at least as long as the anchor and
//!    the span ending at `anchor.record_count - 1` has exactly the anchor's
//!    chained digest. Truncating the log together with its span list below
//!    the anchor fails with [`SignedChainError::AnchorMismatch`].
//!
//! ## Limits (not guaranteed)
//!
//! - **The fill-timeout needs a driver.** [`SigningStrategy::BatchTail`]'s
//!   `max_wait_ns` bounds *signature availability* latency only if some
//!   caller actually invokes [`SignedWitnessSink::seal_on_timeout`] at or
//!   after [`SignedWitnessSink::pending_deadline_ns`] — this sink runs no
//!   background thread or async timer of its own (consistent with the
//!   rest of this crate being synchronous). A caller that never calls it
//!   gets the original unbounded-wait behavior regardless of what
//!   `max_wait_ns` is set to.
//! - **Rollback above the anchor.** Records appended after the anchor was
//!   captured can be truncated (log + spans together) without detection.
//!   Callers must persist [`SignedChainReport::head`] (or
//!   [`SignedWitnessSink::anchor`]) out-of-band after every verified run and
//!   pass it as the next run's anchor. [`SignedAnchor::genesis`] disables
//!   rollback protection entirely.
//! - **Crash before seal.** Pending `BatchTail` state is in memory only; a
//!   crash leaves the unsealed tail permanently unsigned, and only a key
//!   holder can re-sign it. Verification fails closed in that case.
//! - **Key compromise / key management** (generation, rotation, storage,
//!   revocation) is out of scope. Anyone holding the signing key can sign
//!   an arbitrary alternative history.
//! - `evidence_grade` is not part of `to_bytes()`; it is bound to the
//!   hashed `flags` nibble only by [`MemoryWitnessLog::verify_chain`], which
//!   is why verification still runs the unsigned chain walk first.

use crate::ops::{LedgerError, LedgerWitnessRecord, MemoryWitnessLog, WitnessSink};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use rvf_types::ed25519::Ed25519Keypair;
use rvf_types::sha256::{sha256, Sha256};

const MSG_TAG: &[u8] = b"ruvector-agent-memory:witness-signer:v2:span\0";
const RECORDS_TAG: &[u8] = b"ruvector-agent-memory:witness-signer:v2:records\0";
/// `purpose` (1) + `from`/`to` (2 × 8 LE) + `prev_link` (32) + `records_digest` (32).
const MESSAGE_LEN: usize = MSG_TAG.len() + 1 + 16 + 32 + 32;

/// Distinguishes a per-record signature from a batch-tail signature so a
/// signature produced for one purpose can never be replayed as the other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum SignPurpose {
    PerRecord = 1,
    BatchTail = 2,
}

/// One Ed25519-signed statement: "the witness records with sequence numbers
/// `[covers_from_seq, covers_to_seq]` have SHA-256 digest `records_digest`,
/// and they directly follow the span whose chained link is `prev_link`".
/// `prev_link` is not stored: the verifier derives it from the preceding
/// span, which is what makes the span list a chain.
#[derive(Clone, Copy, Debug)]
pub struct SignedSpan {
    pub purpose: SignPurpose,
    pub covers_from_seq: u64,
    pub covers_to_seq: u64,
    pub records_digest: [u8; 32],
    pub signature: [u8; 64],
}

impl SignedSpan {
    fn message(
        purpose: SignPurpose,
        from: u64,
        to: u64,
        prev_link: &[u8; 32],
        records_digest: &[u8; 32],
    ) -> [u8; MESSAGE_LEN] {
        let mut m = [0u8; MESSAGE_LEN];
        let mut off = MSG_TAG.len();
        m[..off].copy_from_slice(MSG_TAG);
        m[off] = purpose as u8;
        off += 1;
        m[off..off + 8].copy_from_slice(&from.to_le_bytes());
        off += 8;
        m[off..off + 8].copy_from_slice(&to.to_le_bytes());
        off += 8;
        m[off..off + 32].copy_from_slice(prev_link);
        off += 32;
        m[off..off + 32].copy_from_slice(records_digest);
        m
    }
}

/// Streaming SHA-256 over the full 64-byte encodings of a span's records.
struct RecordsHasher(Sha256);

impl RecordsHasher {
    fn new() -> Self {
        let mut h = Sha256::new();
        h.update(RECORDS_TAG);
        Self(h)
    }
    fn push(&mut self, r: &LedgerWitnessRecord) {
        self.0.update(&r.to_bytes());
    }
    fn finish(self) -> [u8; 32] {
        self.0.finalize()
    }
}

/// A trusted checkpoint of the signed chain: the number of records covered
/// and the chained SHA-256 link of the span ending at `record_count - 1`.
/// Persist it out-of-band (it is the signed analogue of
/// [`MemoryWitnessLog::head_commitment`]) and pass it back to
/// [`verify_signed_chain`] to detect rollback below it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignedAnchor {
    pub record_count: u64,
    pub head_digest: [u8; 32],
}

impl SignedAnchor {
    /// The empty-chain anchor. Passing it to [`verify_signed_chain`]
    /// explicitly opts OUT of rollback detection.
    pub const fn genesis() -> Self {
        Self {
            record_count: 0,
            head_digest: [0u8; 32],
        }
    }
}

/// Which records get their own signature vs. share an amortized one.
#[derive(Clone, Copy, Debug)]
pub enum SigningStrategy {
    PerRecord,
    /// Sign once every `batch_size` records (and on [`SignedWitnessSink::seal`]).
    /// `batch_size` must be at least 1.
    BatchTail {
        batch_size: usize,
        /// Close the pending span once this many wall-clock nanoseconds
        /// have elapsed since its oldest pending record's own
        /// `timestamp_ns`, even if `batch_size` has not been reached.
        /// `None` keeps the original fixed-size-only behavior: a partial
        /// span never closes early, however long it waits — see
        /// [`SignedWitnessSink::seal_on_timeout`] and ADR-347 Next
        /// Research item 1.
        max_wait_ns: Option<u64>,
    },
}

impl SigningStrategy {
    /// Fixed-size-only `BatchTail`: a span closes only at `batch_size`
    /// records, with no wall-clock bound. Equivalent to today's original
    /// (pre-timeout) behavior.
    pub const fn batch_tail(batch_size: usize) -> Self {
        Self::BatchTail {
            batch_size,
            max_wait_ns: None,
        }
    }

    /// `BatchTail` with a wall-clock fill timeout: a span closes at
    /// `batch_size` records, or after `max_wait_ns` since the oldest
    /// pending record's `timestamp_ns`, whichever comes first — the same
    /// two-trigger shape as
    /// `ruvector_retrieval_receipt::batch_fill::BatchFillPolicy::hybrid`,
    /// applied here to witness-signing spans rather than receipt batches.
    pub const fn batch_tail_timed(batch_size: usize, max_wait_ns: u64) -> Self {
        Self::BatchTail {
            batch_size,
            max_wait_ns: Some(max_wait_ns),
        }
    }
}

/// Construction errors for [`SignedWitnessSink`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WitnessSignerError {
    ZeroBatchSize,
}

impl std::fmt::Display for WitnessSignerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroBatchSize => write!(f, "BatchTail batch_size must be at least 1"),
        }
    }
}

impl std::error::Error for WitnessSignerError {}

struct PendingSpan {
    from: u64,
    to: u64,
    count: usize,
    hasher: RecordsHasher,
    /// The `timestamp_ns` carried by this span's first (oldest) record —
    /// the basis for [`SignedWitnessSink::pending_deadline_ns`], mirroring
    /// `batch_fill::BatchScheduler::oldest_pending_arrival_ns`'s role in
    /// `ruvector-retrieval-receipt`.
    oldest_timestamp_ns: u64,
}

/// A [`WitnessSink`] decorator that signs spans of the records it forwards
/// to an inner sink. The inner sink commits first, so a refused batch is
/// never signed; a batch whose sequence numbers do not continue the signed
/// chain contiguously is refused before reaching the inner sink.
pub struct SignedWitnessSink<S: WitnessSink> {
    inner: S,
    signing_key: SigningKey,
    strategy: SigningStrategy,
    next_seq: u64,
    pending: Option<PendingSpan>,
    head: SignedAnchor,
    spans: Vec<SignedSpan>,
}

impl<S: WitnessSink> SignedWitnessSink<S> {
    pub fn new(
        inner: S,
        signing_key: SigningKey,
        strategy: SigningStrategy,
    ) -> Result<Self, WitnessSignerError> {
        if let SigningStrategy::BatchTail { batch_size: 0, .. } = strategy {
            return Err(WitnessSignerError::ZeroBatchSize);
        }
        Ok(Self {
            inner,
            signing_key,
            strategy,
            next_seq: 0,
            pending: None,
            head: SignedAnchor::genesis(),
            spans: Vec::new(),
        })
    }

    /// Convenience constructor from an `rvf-types` keypair; copies the
    /// secret out exactly once (the sink then holds only a `SigningKey`).
    pub fn from_keypair(
        inner: S,
        keypair: &Ed25519Keypair,
        strategy: SigningStrategy,
    ) -> Result<Self, WitnessSignerError> {
        let mut secret = keypair.secret_key();
        let signing_key = SigningKey::from_bytes(&secret);
        secret.fill(0);
        std::hint::black_box(&secret);
        Self::new(inner, signing_key, strategy)
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Every span signed so far, in emission order.
    pub fn spans(&self) -> &[SignedSpan] {
        &self.spans
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// The signed head: records covered by closed spans and the newest
    /// span's chained link. Persist it out-of-band as the next anchor.
    pub fn anchor(&self) -> SignedAnchor {
        self.head
    }

    /// Records forwarded to the inner sink but not yet covered by a closed
    /// span (always 0 under `PerRecord`).
    pub fn unsigned_pending(&self) -> usize {
        self.pending.as_ref().map_or(0, |p| p.count)
    }

    fn close(&mut self, purpose: SignPurpose, from: u64, to: u64, records_digest: [u8; 32]) {
        let msg = SignedSpan::message(purpose, from, to, &self.head.head_digest, &records_digest);
        let signature = self.signing_key.sign(&msg).to_bytes();
        self.spans.push(SignedSpan {
            purpose,
            covers_from_seq: from,
            covers_to_seq: to,
            records_digest,
            signature,
        });
        self.head = SignedAnchor {
            record_count: to + 1,
            head_digest: sha256(&msg),
        };
    }

    /// Sign whatever `BatchTail` span is still open. Must be called before
    /// verification (e.g. at shutdown); a no-op when nothing is pending.
    pub fn seal(&mut self) {
        if let Some(p) = self.pending.take() {
            self.close(SignPurpose::BatchTail, p.from, p.to, p.hasher.finish());
        }
    }

    /// The wall-clock deadline (nanoseconds, same clock as
    /// [`LedgerWitnessRecord::timestamp_ns`]) at which the pending span
    /// must close even though it has not reached `batch_size`. `None`
    /// when nothing is pending, the strategy is [`SigningStrategy::PerRecord`]
    /// (never pends), or `max_wait_ns` is `None` (fixed-size-only).
    ///
    /// This sink holds no timer of its own: a caller drives the timeout by
    /// scheduling a wake-up at this deadline and then calling
    /// [`Self::seal_on_timeout`] — the same caller-driven shape
    /// `ruvector_retrieval_receipt::batch_fill::BatchScheduler::oldest_pending_arrival_ns`
    /// uses for receipt batches.
    pub fn pending_deadline_ns(&self) -> Option<u64> {
        let max_wait_ns = match self.strategy {
            SigningStrategy::BatchTail {
                max_wait_ns: Some(w),
                ..
            } => w,
            _ => return None,
        };
        self.pending
            .as_ref()
            .map(|p| p.oldest_timestamp_ns.saturating_add(max_wait_ns))
    }

    /// Close the pending span if its [`Self::pending_deadline_ns`] has
    /// elapsed by `now_ns`. Returns `true` if a span was closed (and thus
    /// signed via the ordinary [`Self::close`] path — a timeout-closed
    /// span carries no weaker guarantee than a size-closed one; both call
    /// the same signing code). Returns `false` on a stale/early/no-op
    /// wake-up: nothing pending, the deadline has not elapsed yet, or the
    /// strategy has no timeout.
    ///
    /// Bridges the gap ADR-347's Next Research item 1 named: without this,
    /// a `BatchTail` span that never reaches `batch_size` stays unsigned
    /// forever, however long a deployment is idle.
    pub fn seal_on_timeout(&mut self, now_ns: u64) -> bool {
        match self.pending_deadline_ns() {
            Some(deadline) if now_ns >= deadline => {
                self.seal();
                true
            }
            _ => false,
        }
    }
}

impl<S: WitnessSink> WitnessSink for SignedWitnessSink<S> {
    fn emit_batch(&mut self, records: &[LedgerWitnessRecord]) -> Result<(), LedgerError> {
        for (i, r) in records.iter().enumerate() {
            let expected = self.next_seq + i as u64;
            if r.sequence != expected {
                return Err(LedgerError::WitnessRejected(format!(
                    "witness signer: expected sequence {expected}, got {}",
                    r.sequence
                )));
            }
        }
        // Witness-first: the inner sink commits before anything is signed.
        self.inner.emit_batch(records)?;
        self.next_seq += records.len() as u64;
        match self.strategy {
            SigningStrategy::PerRecord => {
                for r in records {
                    let mut h = RecordsHasher::new();
                    h.push(r);
                    self.close(SignPurpose::PerRecord, r.sequence, r.sequence, h.finish());
                }
            }
            SigningStrategy::BatchTail { batch_size, .. } => {
                for r in records {
                    let p = self.pending.get_or_insert_with(|| PendingSpan {
                        from: r.sequence,
                        to: r.sequence,
                        count: 0,
                        hasher: RecordsHasher::new(),
                        oldest_timestamp_ns: r.timestamp_ns,
                    });
                    p.hasher.push(r);
                    p.to = r.sequence;
                    p.count += 1;
                    if p.count >= batch_size {
                        self.seal();
                    }
                }
            }
        }
        Ok(())
    }
}

/// Why a signed witness log failed verification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignedChainError {
    InvalidPublicKey,
    /// The unsigned FNV-1a chain walk / head commitment / evidence-grade
    /// binding ([`MemoryWitnessLog::verify_chain`]) failed.
    ChainWalkFailed,
    /// The record at `index` carries a different `sequence`.
    SequenceMismatch {
        index: u64,
        found: u64,
    },
    /// Span `span_index` does not start where the previous one ended
    /// (or the first span does not start at 0).
    CoverageGap {
        span_index: usize,
        expected_from: u64,
    },
    /// Span `span_index` is inverted, or a `PerRecord` span covers >1 record.
    MalformedSpan {
        span_index: usize,
    },
    /// Span `span_index` covers records the log does not contain.
    SpanBeyondLog {
        span_index: usize,
        log_len: u64,
    },
    /// The log's records in span `span_index` do not hash to the signed digest.
    RecordsMismatch {
        span_index: usize,
    },
    /// Span `span_index`'s Ed25519 signature (strict verification) failed.
    BadSignature {
        span_index: usize,
    },
    /// No span boundary at the anchor, or its chained digest differs —
    /// a rollback below the anchor, or an anchor from another chain.
    AnchorMismatch,
    /// The signed prefix (`signed` records) verified, but the newest
    /// `unsigned` records are not covered by any span and are NOT
    /// authenticated.
    UnsignedTail {
        signed: u64,
        unsigned: u64,
    },
}

impl std::fmt::Display for SignedChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "signed witness chain verification failed: {self:?}")
    }
}

impl std::error::Error for SignedChainError {}

/// Result of a successful [`verify_signed_chain`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignedChainReport {
    pub records_verified: u64,
    pub spans_verified: usize,
    /// The verified signed head; persist it as the next anchor.
    pub head: SignedAnchor,
}

/// Verify a signed witness log against a trusted public key and a trusted
/// anchor. See the module docs for exactly what `Ok` does and does not
/// guarantee.
pub fn verify_signed_chain(
    log: &MemoryWitnessLog,
    spans: &[SignedSpan],
    public_key: &[u8; 32],
    anchor: &SignedAnchor,
) -> Result<SignedChainReport, SignedChainError> {
    use SignedChainError as E;
    let vk = VerifyingKey::from_bytes(public_key).map_err(|_| E::InvalidPublicKey)?;
    if !log.verify_chain() {
        return Err(E::ChainWalkFailed);
    }
    for (i, r) in log.records.iter().enumerate() {
        if r.sequence != i as u64 {
            return Err(E::SequenceMismatch {
                index: i as u64,
                found: r.sequence,
            });
        }
    }
    let log_len = log.records.len() as u64;
    let mut prev_link = [0u8; 32];
    let mut next = 0u64;
    let mut anchor_ok = anchor.record_count == 0 && anchor.head_digest == [0u8; 32];
    for (k, span) in spans.iter().enumerate() {
        let (from, to) = (span.covers_from_seq, span.covers_to_seq);
        if from != next {
            return Err(E::CoverageGap {
                span_index: k,
                expected_from: next,
            });
        }
        if to < from || (span.purpose == SignPurpose::PerRecord && to != from) {
            return Err(E::MalformedSpan { span_index: k });
        }
        if to >= log_len {
            return Err(E::SpanBeyondLog {
                span_index: k,
                log_len,
            });
        }
        let mut h = RecordsHasher::new();
        for r in &log.records[from as usize..=to as usize] {
            h.push(r);
        }
        let digest = h.finish();
        if digest != span.records_digest {
            return Err(E::RecordsMismatch { span_index: k });
        }
        let msg = SignedSpan::message(span.purpose, from, to, &prev_link, &digest);
        let sig = Signature::from_bytes(&span.signature);
        if vk.verify_strict(&msg, &sig).is_err() {
            return Err(E::BadSignature { span_index: k });
        }
        prev_link = sha256(&msg);
        next = to + 1;
        if next == anchor.record_count {
            if prev_link != anchor.head_digest {
                return Err(E::AnchorMismatch);
            }
            anchor_ok = true;
        }
    }
    if !anchor_ok {
        return Err(E::AnchorMismatch);
    }
    if next < log_len {
        return Err(E::UnsignedTail {
            signed: next,
            unsigned: log_len - next,
        });
    }
    Ok(SignedChainReport {
        records_verified: log_len,
        spans_verified: spans.len(),
        head: SignedAnchor {
            record_count: next,
            head_digest: prev_link,
        },
    })
}

#[cfg(test)]
#[path = "witness_signing_tests.rs"]
mod tests;
