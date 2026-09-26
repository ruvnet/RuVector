//! Chained, tamper-evident erasure-audit certificates.
//!
//! An erasure that nobody can check is a promise, not a control. This module
//! emits one record per erasure — what was erased, under which mode, how much
//! structure was rewritten, whether the payload was zeroized, and the audited
//! distinguishing advantage that the mode achieved — and chains the records so
//! that editing an earlier record invalidates every later one.
//!
//! ## Scope of the tamper-evidence
//!
//! The chain uses keyless FNV-1a, matching the existing witness machinery in
//! `ruvector-agent-memory::ops` (ADR-134). That detects naive edits to a stored
//! log; it is **not** adversary-resistant, because anyone who can edit a record
//! can recompute the hashes. Upgrading to a keyed MAC or to the signed-receipt
//! scheme of `ruvector-retrieval-receipt` (ADR-340) is a deployment decision,
//! not a change to this module's shape. Stated plainly rather than implied.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// A single erasure event, as recorded for audit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErasureCertificate {
    /// Monotonic position in the chain.
    pub seq: u64,
    /// Hash of the erased subject's identifier. The identifier itself is not
    /// stored: a log of "which records were erased" is itself personal data.
    pub subject_hash: u64,
    /// Erasure mode label, e.g. `"LocalRebuild"`.
    pub mode: &'static str,
    /// Live nodes that held an edge to the subject at erasure time.
    pub referrers: u32,
    /// Neighbour lists recomputed.
    pub rebuilt_lists: u32,
    /// Bytes of the subject's vector still resident afterwards.
    pub retained_vector_bytes: u32,
    /// Measured held-out distinguishing accuracy, in units of 1/1000.
    /// 500 = indistinguishable; 1000 = perfectly recoverable.
    pub audit_advantage_milli: u32,
    pub timestamp_ns: u64,
    pub prev_hash: u64,
    pub hash: u64,
}

impl ErasureCertificate {
    fn payload(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64);
        b.extend_from_slice(&self.seq.to_le_bytes());
        b.extend_from_slice(&self.subject_hash.to_le_bytes());
        b.extend_from_slice(self.mode.as_bytes());
        b.extend_from_slice(&self.referrers.to_le_bytes());
        b.extend_from_slice(&self.rebuilt_lists.to_le_bytes());
        b.extend_from_slice(&self.retained_vector_bytes.to_le_bytes());
        b.extend_from_slice(&self.audit_advantage_milli.to_le_bytes());
        b.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        b.extend_from_slice(&self.prev_hash.to_le_bytes());
        b
    }

    /// Recompute this record's hash from its contents.
    pub fn recompute_hash(&self) -> u64 {
        fnv1a(&self.payload())
    }
}

/// An append-only chain of erasure certificates.
#[derive(Clone, Debug, Default)]
pub struct CertificateChain {
    records: Vec<ErasureCertificate>,
}

/// Why a chain failed verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChainError {
    /// Record `seq` has a hash that does not match its contents.
    ContentTampered(u64),
    /// Record `seq` does not link to its predecessor.
    BrokenLink(u64),
    /// Sequence numbers are not contiguous from zero at `seq`.
    SequenceGap(u64),
}

impl CertificateChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[ErasureCertificate] {
        &self.records
    }

    /// Hash of the chain head — the value a remote auditor would anchor.
    pub fn head(&self) -> u64 {
        self.records.last().map(|r| r.hash).unwrap_or(FNV_OFFSET)
    }

    /// Append an erasure event. Returns the new record.
    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &mut self,
        subject_id: &str,
        mode: &'static str,
        referrers: u32,
        rebuilt_lists: u32,
        retained_vector_bytes: u32,
        audit_advantage_milli: u32,
        timestamp_ns: u64,
    ) -> &ErasureCertificate {
        let mut rec = ErasureCertificate {
            seq: self.records.len() as u64,
            subject_hash: fnv1a(subject_id.as_bytes()),
            mode,
            referrers,
            rebuilt_lists,
            retained_vector_bytes,
            audit_advantage_milli,
            timestamp_ns,
            prev_hash: self.head(),
            hash: 0,
        };
        rec.hash = rec.recompute_hash();
        self.records.push(rec);
        self.records.last().unwrap()
    }

    /// Verify every record's content hash and back-link.
    pub fn verify(&self) -> Result<(), ChainError> {
        let mut prev = FNV_OFFSET;
        for (i, rec) in self.records.iter().enumerate() {
            if rec.seq != i as u64 {
                return Err(ChainError::SequenceGap(rec.seq));
            }
            if rec.prev_hash != prev {
                return Err(ChainError::BrokenLink(rec.seq));
            }
            if rec.hash != rec.recompute_hash() {
                return Err(ChainError::ContentTampered(rec.seq));
            }
            prev = rec.hash;
        }
        Ok(())
    }

    /// Test hook: mutate a record in place without fixing up hashes.
    #[doc(hidden)]
    pub fn tamper_retained_bytes(&mut self, idx: usize, value: u32) {
        if let Some(r) = self.records.get_mut(idx) {
            r.retained_vector_bytes = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_of(n: usize) -> CertificateChain {
        let mut c = CertificateChain::new();
        for i in 0..n {
            c.append(
                &format!("subject-{i}"),
                "LocalRebuild",
                12,
                12,
                0,
                512,
                1_000 + i as u64,
            );
        }
        c
    }

    #[test]
    fn fresh_chain_verifies() {
        assert_eq!(chain_of(25).verify(), Ok(()));
    }

    #[test]
    fn subject_identifier_is_not_stored_in_the_clear() {
        let mut c = CertificateChain::new();
        c.append("user-42@example.com", "Tombstone", 3, 0, 128, 900, 1);
        let rec = &c.records()[0];
        assert_ne!(rec.subject_hash, 0);
        // The struct has no field capable of holding the raw identifier.
        assert_eq!(rec.mode, "Tombstone");
    }

    #[test]
    fn every_single_record_edit_is_detected() {
        // Exhaustive tamper sweep rather than a spot check.
        for idx in 0..25 {
            let mut c = chain_of(25);
            let original = c.records()[idx].retained_vector_bytes;
            c.tamper_retained_bytes(idx, original + 1);
            match c.verify() {
                Err(ChainError::ContentTampered(seq)) => assert_eq!(seq, idx as u64),
                other => panic!("tamper at {idx} not detected: {other:?}"),
            }
        }
    }

    #[test]
    fn head_advances_with_each_append() {
        let mut c = CertificateChain::new();
        let h0 = c.head();
        c.append("a", "Tombstone", 1, 0, 128, 900, 1);
        let h1 = c.head();
        c.append("b", "Tombstone", 1, 0, 128, 900, 2);
        let h2 = c.head();
        assert_ne!(h0, h1);
        assert_ne!(h1, h2);
    }

    #[test]
    fn truncation_is_detected_by_head_comparison() {
        // Removing the tail still verifies internally (that is the known limit
        // of a bare hash chain) — detection requires an externally anchored
        // head, which is what `head()` exists to publish.
        let full = chain_of(10);
        let anchored_head = full.head();
        let truncated = chain_of(9);
        assert_eq!(truncated.verify(), Ok(()));
        assert_ne!(truncated.head(), anchored_head);
    }
}
