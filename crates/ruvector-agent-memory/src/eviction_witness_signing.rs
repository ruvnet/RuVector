//! Ed25519-signed anchors over the ADR-345 eviction-witness chain
//! ([`crate::witnessed_compaction`]).
//!
//! ## The gap this closes
//!
//! [`crate::ops`]'s module docs and ADR-134 §3/§9 are explicit: the eviction
//! chain's FNV-1a linking is tamper-EVIDENT against accidental corruption
//! and naive edits only. An adversary with write access to the persisted
//! log can pick any record, mutate it, and recompute every downstream
//! `record_hash` / `prev_hash` in one O(n) pass — [`MemoryWitnessLog::verify_chain`]
//! on the result still returns `true`. Closing that gap was named as
//! explicit follow-up work in both places ("Wiring a `WitnessSigner`
//! through `WitnessSink` ... MUST land before WP8 cross-repo anchoring
//! makes this log load-bearing") and in the 2026-09-05 nightly run's "Next
//! Research" (wire an Ed25519 `WitnessSigner` so eviction receipts are
//! signed, not just hash-chained).
//!
//! ## What this does *not* do
//!
//! This does not sign every record inline on the emission path (ADR-134
//! §9's `WitnessSigner` trait shape). It anchors the chain *head* — the
//! same `(sequence, chain_hash)` commitment already returned by
//! [`crate::ops::MemoryWitnessLog::head_commitment`] — either after every
//! record (`interval_records = 1`) or periodically, reusing the
//! interval/staleness policy shape `ruvector-retrieval-receipt::state_anchor`
//! (ADR-342) already proved out for a different chain. No new signature
//! scheme is introduced: signing goes through `rvf_types::ed25519_sign` /
//! `ed25519_verify`, the same RFC 8032 Ed25519 primitive
//! [`crate::observation::AtomicObservation`] already uses (ADR-320) and
//! already an unconditional dependency of this crate.
//!
//! An anchor authenticates one specific `(sequence, chain_head)` pair. It
//! does **not** retroactively protect records between the previous anchor
//! and this one — that residual window is the staleness bound, measured
//! honestly below, not hidden.

use crate::ops::LedgerWitnessRecord;
use rvf_types::{ed25519_sign, ed25519_verify, Ed25519Keypair};

/// Domain-separation prefix for the signed statement, mirroring
/// `ruvector-retrieval-receipt::signing`'s `SIGNED_ROOT_DOMAIN` pattern so a
/// signature over one chain's anchor can never be replayed as valid for
/// another.
pub const EVICTION_ANCHOR_DOMAIN: &[u8] = b"ruvector:agent-memory:eviction-anchor:v1:";

/// The statement an [`EvictionAnchorLog`] signs: "as of `sequence` evicted
/// records, the eviction-witness chain head is `chain_head`."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictionAnchorStatement {
    pub sequence: u64,
    pub chain_head: u64,
    pub issued_at_ns: u64,
}

impl EvictionAnchorStatement {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(EVICTION_ANCHOR_DOMAIN.len() + 24);
        bytes.extend_from_slice(EVICTION_ANCHOR_DOMAIN);
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        bytes.extend_from_slice(&self.chain_head.to_le_bytes());
        bytes.extend_from_slice(&self.issued_at_ns.to_le_bytes());
        bytes
    }
}

/// A statement plus its Ed25519 signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedEvictionAnchor {
    pub statement: EvictionAnchorStatement,
    pub signature: [u8; 64],
}

/// Sign `(sequence, chain_head)` at `issued_at_ns` with `keypair`.
pub fn sign_anchor(
    keypair: &Ed25519Keypair,
    sequence: u64,
    chain_head: u64,
    issued_at_ns: u64,
) -> SignedEvictionAnchor {
    let statement = EvictionAnchorStatement {
        sequence,
        chain_head,
        issued_at_ns,
    };
    let signature = ed25519_sign(&keypair.secret_key(), &statement.canonical_bytes());
    SignedEvictionAnchor {
        statement,
        signature,
    }
}

/// Verify the signature alone (does not check the statement against any
/// external chain state — see [`verify_anchor_against_chain`] for that).
pub fn verify_anchor(public_key: &[u8; 32], anchor: &SignedEvictionAnchor) -> bool {
    ed25519_verify(
        public_key,
        &anchor.statement.canonical_bytes(),
        &anchor.signature,
    )
}

/// Verify that `anchor` is both a valid signature AND still consistent with
/// `chain_head_at_sequence` — the chain head an independent auditor
/// recomputes at `anchor.statement.sequence` from the (possibly tampered)
/// log they hold. A log-writing adversary who mutates a record at or before
/// `anchor.statement.sequence` and relinks everything downstream changes
/// the recomputed head at that sequence; the adversary cannot also forge a
/// matching signature without the private key, so this check fails even
/// though [`crate::ops::MemoryWitnessLog::verify_chain`] on the tampered
/// log alone would report `true`.
pub fn verify_anchor_against_chain(
    public_key: &[u8; 32],
    anchor: &SignedEvictionAnchor,
    chain_head_at_sequence: u64,
) -> bool {
    verify_anchor(public_key, anchor) && anchor.statement.chain_head == chain_head_at_sequence
}

/// Errors constructing an [`EvictionAnchorPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorError {
    InvalidInterval,
}

impl core::fmt::Display for AnchorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInterval => write!(f, "interval_records must be at least 1"),
        }
    }
}

impl std::error::Error for AnchorError {}

/// How often to anchor: every `interval_records` evicted (witnessed)
/// records. `interval_records == 1` anchors every record (zero staleness,
/// maximum signing cost, and — per ADR-134 §9's `WitnessSigner` trait shape
/// — the design that module docs originally pointed to). Larger values
/// bound staleness to `interval_records - 1` while amortizing signing cost,
/// mirroring `ruvector-retrieval-receipt::state_anchor::StateAnchorPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictionAnchorPolicy {
    interval_records: u64,
}

impl EvictionAnchorPolicy {
    pub fn new(interval_records: u64) -> Result<Self, AnchorError> {
        if interval_records == 0 {
            return Err(AnchorError::InvalidInterval);
        }
        Ok(Self { interval_records })
    }

    pub const fn interval_records(&self) -> u64 {
        self.interval_records
    }
}

/// Signs eviction-witness records under a fixed interval policy, keeping
/// every anchor it has produced.
#[derive(Debug, Clone)]
pub struct EvictionAnchorLog {
    policy: EvictionAnchorPolicy,
    record_count: u64,
    pub anchors: Vec<SignedEvictionAnchor>,
}

impl EvictionAnchorLog {
    pub fn new(policy: EvictionAnchorPolicy) -> Self {
        Self {
            policy,
            record_count: 0,
            anchors: Vec::new(),
        }
    }

    pub const fn policy(&self) -> EvictionAnchorPolicy {
        self.policy
    }

    pub const fn record_count(&self) -> u64 {
        self.record_count
    }

    /// Call once per newly emitted eviction-witness record, in chain order.
    /// Returns the freshly produced anchor when `record_count` lands on an
    /// interval boundary, else `None`.
    pub fn note_record(
        &mut self,
        keypair: &Ed25519Keypair,
        record: &LedgerWitnessRecord,
        issued_at_ns: u64,
    ) -> Option<&SignedEvictionAnchor> {
        self.record_count += 1;
        if !self
            .record_count
            .is_multiple_of(self.policy.interval_records)
        {
            return None;
        }
        let anchor = sign_anchor(
            keypair,
            self.record_count,
            record.chain_hash(),
            issued_at_ns,
        );
        self.anchors.push(anchor);
        self.anchors.last()
    }

    pub fn latest(&self) -> Option<&SignedEvictionAnchor> {
        self.anchors.last()
    }

    /// Records since the last anchor, given the caller is currently at
    /// `at_record_count` (usually [`Self::record_count`]). Bounded by
    /// `interval_records - 1` for a correctly operating log.
    pub fn staleness_at(&self, at_record_count: u64) -> u64 {
        let last = self.anchors.last().map_or(0, |a| a.statement.sequence);
        at_record_count.saturating_sub(last)
    }
}

/// Attack-simulation utility (not part of the honest write path): mutate
/// `records[tamper_index]` with `mutate`, then recompute `record_hash` and
/// relink `prev_hash`/downstream `record_hash`es for every record from
/// `tamper_index` onward so the resulting slice passes
/// [`crate::ops::MemoryWitnessLog::verify_chain`] unmodified. Models the
/// log-writing adversary this module's docs describe, so the defense this
/// module provides can be measured against a real attack rather than
/// asserted.
pub fn relink_tampered_suffix(
    records: &mut [LedgerWitnessRecord],
    tamper_index: usize,
    mutate: impl FnOnce(&mut LedgerWitnessRecord),
) {
    assert!(tamper_index < records.len(), "tamper_index out of bounds");
    mutate(&mut records[tamper_index]);

    let mut prev = if tamper_index == 0 {
        0
    } else {
        records[tamper_index - 1].chain_hash()
    };
    for r in &mut records[tamper_index..] {
        r.prev_hash = prev;
        r.record_hash = r.compute_record_hash();
        prev = r.chain_hash();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::LruPolicy;
    use crate::memory::MemoryStore;
    use crate::ops::{EvidenceGrade, MemoryWitnessLog};
    use crate::witnessed_compaction::{compact_witnessed, EvictionWitnessChain};
    use rand::rngs::OsRng;

    fn keypair() -> Ed25519Keypair {
        Ed25519Keypair::generate(&mut OsRng)
    }

    fn build_chain(n_inserted: usize, n_survivors: usize) -> MemoryWitnessLog {
        let mut store = MemoryStore::new(2);
        for i in 0..n_inserted {
            store.insert(vec![i as f32, 0.0]);
        }
        let mut chain = EvictionWitnessChain::new();
        let mut log = MemoryWitnessLog::default();
        compact_witnessed(
            &mut store,
            &LruPolicy,
            n_survivors,
            &[],
            "bench",
            1_000,
            &mut chain,
            &mut log,
        )
        .unwrap();
        log
    }

    #[test]
    fn sign_verify_round_trip() {
        let kp = keypair();
        let anchor = sign_anchor(&kp, 7, 0xDEAD_BEEF, 42);
        assert!(verify_anchor(&kp.public_key(), &anchor));
    }

    #[test]
    fn wrong_key_or_tampered_statement_rejected() {
        let kp = keypair();
        let impostor = keypair();
        let anchor = sign_anchor(&kp, 7, 0xDEAD_BEEF, 42);
        assert!(!verify_anchor(&impostor.public_key(), &anchor));

        let mut tampered = anchor;
        tampered.statement.chain_head ^= 1;
        assert!(!verify_anchor(&kp.public_key(), &tampered));

        let mut tampered = anchor;
        tampered.statement.sequence += 1;
        assert!(!verify_anchor(&kp.public_key(), &tampered));
    }

    #[test]
    fn periodic_policy_anchors_at_exact_boundaries() {
        let kp = keypair();
        let log = build_chain(20, 0); // 20 evictions
        let policy = EvictionAnchorPolicy::new(4).unwrap();
        let mut alog = EvictionAnchorLog::new(policy);
        for r in &log.records {
            alog.note_record(&kp, r, 1_000);
        }
        assert_eq!(alog.anchors.len(), 20 / 4);
        assert_eq!(
            alog.anchors.last().unwrap().statement.sequence,
            log.records.len() as u64
        );
    }

    #[test]
    fn staleness_never_exceeds_interval_minus_one() {
        let kp = keypair();
        let log = build_chain(37, 0);
        for interval in [1u64, 2, 5, 8, 16] {
            let mut alog = EvictionAnchorLog::new(EvictionAnchorPolicy::new(interval).unwrap());
            let mut max_seen = 0u64;
            for r in &log.records {
                alog.note_record(&kp, r, 1_000);
                max_seen = max_seen.max(alog.staleness_at(alog.record_count()));
            }
            assert!(
                max_seen <= interval - 1,
                "interval={interval} max_staleness={max_seen}"
            );
        }
    }

    #[test]
    fn fnv1a_chain_alone_does_not_detect_relinked_tamper() {
        let mut log = build_chain(10, 0);
        assert!(log.verify_chain());
        relink_tampered_suffix(&mut log.records, 2, |r| {
            r.target_object_id = 0xFFFF_FFFF; // pretend a different id was evicted
        });
        // A log-writing adversary controls the whole persisted
        // representation, including any head commitment stored alongside
        // the records (not just the records themselves) — so the
        // commitment is rewritten to match the relinked tail too.
        log.committed_head = log.records.last().unwrap().chain_hash();
        // Documents the known weakness this module exists to close: a
        // fully relinked (records + commitment) FNV-1a chain still
        // verifies as internally self-consistent.
        assert!(
            log.verify_chain(),
            "relinked chain unexpectedly failed verify_chain (weakness assumption changed)"
        );
    }

    #[test]
    fn signed_anchor_detects_what_fnv1a_chaining_misses() {
        let kp = keypair();
        let log = build_chain(10, 0);
        // Anchor after every record (interval=1) so the last record before
        // the tamper point is anchored.
        let mut alog = EvictionAnchorLog::new(EvictionAnchorPolicy::new(1).unwrap());
        for r in &log.records {
            alog.note_record(&kp, r, 1_000);
        }
        let anchor_before_tamper = *alog.anchors.get(1).unwrap(); // anchors sequence=2

        let mut tampered = log.records.clone();
        relink_tampered_suffix(&mut tampered, 1, |r| {
            r.target_object_id = 0xFFFF_FFFF;
        });
        // The relinked log still passes the unauthenticated chain check...
        let tampered_log = MemoryWitnessLog {
            records: tampered.clone(),
            committed_count: tampered.len() as u64,
            committed_head: tampered.last().unwrap().chain_hash(),
        };
        assert!(tampered_log.verify_chain());

        // ...but the auditor recomputes the head at the anchored sequence
        // from the tampered log and the signed anchor no longer matches it.
        let recomputed_head_at_2 = tampered[1].chain_hash();
        assert!(!verify_anchor_against_chain(
            &kp.public_key(),
            &anchor_before_tamper,
            recomputed_head_at_2,
        ));

        // A correct (untampered) recomputation still verifies.
        let honest_head_at_2 = log.records[1].chain_hash();
        assert!(verify_anchor_against_chain(
            &kp.public_key(),
            &anchor_before_tamper,
            honest_head_at_2,
        ));
    }

    #[test]
    fn evidence_grade_present_on_source_records() {
        let log = build_chain(4, 0);
        for r in &log.records {
            assert_eq!(r.evidence_grade, EvidenceGrade::Recomputed);
        }
    }
}
