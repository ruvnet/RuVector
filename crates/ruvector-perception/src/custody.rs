//! Sensor chain of custody — a tamper-evident, replayable ledger of perception
//! events.
//!
//! Every [`DeltaWitness`] already carries a SHA-256 `evidence_hash` and the
//! `prev_hash` of the witness before it, so a sequence of witnesses forms a hash
//! chain. This module wraps that chain in an append-only ledger so that every
//! action the engine takes can be *replayed* and its provenance *audited* —
//! which is what elder-care, medical, industrial, and civic-governance
//! deployments require before they trust an automated decision.
//!
//! ## Honest scope of verification
//!
//! [`CustodyLedger::verify`] checks **chain linkage**: each record's `prev_hash`
//! must equal the prior record's `evidence_hash` (and the first record's
//! `prev_hash` must be `None`). If anyone mutates a stored `evidence_hash`, the
//! link to the next record breaks and `verify` reports it. This is
//! **link-integrity**, *not* a full content re-hash: the raw signal and feature
//! bytes that produced each `evidence_hash` are not stored in the witness, so
//! this layer cannot recompute the SHA-256 from first principles. Detecting a
//! forged-but-internally-consistent hash would require those raw bytes; here we
//! detect tampering that breaks the chain.

use crate::witness::DeltaWitness;
use serde::{Deserialize, Serialize};

/// Errors raised while maintaining or auditing the chain of custody.
#[derive(Debug, thiserror::Error)]
pub enum CustodyError {
    /// A record's `prev_hash` did not match the expected prior `evidence_hash`.
    #[error("broken chain at index {index}: prev_hash {found:?} != expected {expected:?}")]
    BrokenChain {
        /// Position in the ledger where the link broke.
        index: usize,
        /// The `prev_hash` actually found on the record.
        found: Option<String>,
        /// The `evidence_hash` of the prior record (or `None` for the first).
        expected: Option<String>,
    },
    /// No record carries the requested evidence hash.
    #[error("no record with evidence hash {0}")]
    NotFound(String),
}

/// One entry in the ledger: the witnessed delta plus an optional outcome.
///
/// The `outcome` is later feedback attached after the fact (e.g. "confirmed
/// fall", "false alarm", "operator acknowledged") so the audit trail records not
/// just what was perceived but what actually happened.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustodyRecord {
    /// The perception event, including its evidence/prev hash linkage.
    pub witness: DeltaWitness,
    /// Outcome/feedback attached to this event, if any.
    pub outcome: Option<String>,
}

/// An append-only, tamper-evident ledger of perception events.
///
/// Linkage is enforced at insert time by [`CustodyLedger::append`] and can be
/// re-audited at any time by [`CustodyLedger::verify`].
#[derive(Debug, Clone, Default)]
pub struct CustodyLedger {
    records: Vec<CustodyRecord>,
}

impl CustodyLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a witness to the ledger, enforcing chain linkage.
    ///
    /// The witness's `prev_hash` MUST equal the last record's `evidence_hash`
    /// (or be `None` for the very first record); otherwise this returns
    /// [`CustodyError::BrokenChain`] and the ledger is left unchanged.
    pub fn append(&mut self, witness: DeltaWitness) -> Result<(), CustodyError> {
        let expected = self.records.last().map(|r| r.witness.evidence_hash.clone());
        if witness.prev_hash != expected {
            return Err(CustodyError::BrokenChain {
                index: self.records.len(),
                found: witness.prev_hash,
                expected,
            });
        }
        self.records.push(CustodyRecord {
            witness,
            outcome: None,
        });
        Ok(())
    }

    /// Attach an outcome/feedback to the record with the given evidence hash.
    ///
    /// Returns [`CustodyError::NotFound`] if no record carries that hash.
    pub fn record_outcome(
        &mut self,
        evidence_hash: &str,
        outcome: impl Into<String>,
    ) -> Result<(), CustodyError> {
        let record = self
            .records
            .iter_mut()
            .find(|r| r.witness.evidence_hash == evidence_hash)
            .ok_or_else(|| CustodyError::NotFound(evidence_hash.to_string()))?;
        record.outcome = Some(outcome.into());
        Ok(())
    }

    /// Re-audit the whole chain: every `prev_hash` must equal the prior record's
    /// `evidence_hash` (and the first must be `None`).
    ///
    /// This verifies **chain linkage** — tampering with a stored `evidence_hash`
    /// breaks the link to the next record and is reported here. It does **not**
    /// recompute each SHA-256 from raw signal bytes (those are not stored in the
    /// witness), so it is link-integrity, not full content re-hash. Returns the
    /// first [`CustodyError::BrokenChain`] encountered.
    pub fn verify(&self) -> Result<(), CustodyError> {
        let mut expected: Option<String> = None;
        for (index, record) in self.records.iter().enumerate() {
            if record.witness.prev_hash != expected {
                return Err(CustodyError::BrokenChain {
                    index,
                    found: record.witness.prev_hash.clone(),
                    expected,
                });
            }
            expected = Some(record.witness.evidence_hash.clone());
        }
        Ok(())
    }

    /// Number of records in the ledger.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the ledger holds no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Borrow the full record slice (read-only; the ledger stays append-only).
    pub fn records(&self) -> &[CustodyRecord] {
        &self.records
    }

    /// Return the chain of records from the start up to and including the record
    /// with `evidence_hash` — the replayable provenance of that event.
    ///
    /// Returns [`CustodyError::NotFound`] if no record carries that hash.
    pub fn replay_until(&self, evidence_hash: &str) -> Result<Vec<&CustodyRecord>, CustodyError> {
        let end = self
            .records
            .iter()
            .position(|r| r.witness.evidence_hash == evidence_hash)
            .ok_or_else(|| CustodyError::NotFound(evidence_hash.to_string()))?;
        Ok(self.records[..=end].iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modality::Modality;
    use crate::witness::Action;

    /// Build a witness with explicit hash linkage; raw scores are arbitrary but
    /// deterministic so tests focus on custody, not perception.
    fn witness(t: u64, evidence_hash: &str, prev_hash: Option<&str>) -> DeltaWitness {
        DeltaWitness {
            t,
            changed_boundary: format!("zone-{t}"),
            supporting_modalities: vec![Modality::Rf, Modality::Vibration],
            contradicting_modalities: vec![Modality::Thermal],
            novelty: 0.8,
            coherence: 0.7,
            contradiction: 0.1,
            action: Action::Alert,
            evidence_hash: evidence_hash.to_string(),
            prev_hash: prev_hash.map(str::to_string),
        }
    }

    fn three_link_ledger() -> CustodyLedger {
        let mut ledger = CustodyLedger::new();
        ledger.append(witness(0, "h0", None)).unwrap();
        ledger.append(witness(1, "h1", Some("h0"))).unwrap();
        ledger.append(witness(2, "h2", Some("h1"))).unwrap();
        ledger
    }

    #[test]
    fn three_link_chain_verifies() {
        let ledger = three_link_ledger();
        assert_eq!(ledger.len(), 3);
        assert!(!ledger.is_empty());
        assert!(ledger.verify().is_ok());
    }

    #[test]
    fn append_rejects_mismatched_prev_hash() {
        let mut ledger = CustodyLedger::new();
        ledger.append(witness(0, "h0", None)).unwrap();
        // prev_hash should be Some("h0"), but we link it to the wrong place.
        let err = ledger.append(witness(1, "h1", Some("WRONG"))).unwrap_err();
        match err {
            CustodyError::BrokenChain {
                index,
                found,
                expected,
            } => {
                assert_eq!(index, 1);
                assert_eq!(found, Some("WRONG".to_string()));
                assert_eq!(expected, Some("h0".to_string()));
            }
            other => panic!("expected BrokenChain, got {other:?}"),
        }
        // The rejected record must not have been stored.
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn first_record_must_have_no_prev_hash() {
        let mut ledger = CustodyLedger::new();
        let err = ledger.append(witness(0, "h0", Some("h-1"))).unwrap_err();
        assert!(matches!(err, CustodyError::BrokenChain { index: 0, .. }));
        assert!(ledger.is_empty());
    }

    #[test]
    fn record_outcome_then_find_it() {
        let mut ledger = three_link_ledger();
        ledger.record_outcome("h1", "confirmed fall").unwrap();
        let record = ledger
            .records()
            .iter()
            .find(|r| r.witness.evidence_hash == "h1")
            .unwrap();
        assert_eq!(record.outcome.as_deref(), Some("confirmed fall"));
        // Other records are untouched.
        assert!(ledger.records()[0].outcome.is_none());

        // An unknown hash is reported as NotFound.
        let err = ledger.record_outcome("nope", "x").unwrap_err();
        assert!(matches!(err, CustodyError::NotFound(h) if h == "nope"));
    }

    #[test]
    fn replay_until_returns_prefix_chain() {
        let ledger = three_link_ledger();
        let chain = ledger.replay_until("h1").unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].witness.evidence_hash, "h0");
        assert_eq!(chain[1].witness.evidence_hash, "h1");

        // The full chain is reachable from the last hash.
        assert_eq!(ledger.replay_until("h2").unwrap().len(), 3);

        // Unknown hashes are NotFound.
        let err = ledger.replay_until("ghost").unwrap_err();
        assert!(matches!(err, CustodyError::NotFound(h) if h == "ghost"));
    }

    #[test]
    fn verify_detects_a_corrupted_link() {
        // Hand-build a ledger whose middle record's evidence_hash has been
        // mutated *after* insertion, so its link to the next record is broken.
        // We bypass `append` (which would reject this) to simulate tampering of
        // already-stored data, exactly what `verify` must catch.
        let mut ledger = three_link_ledger();
        // Corrupt h1 -> the third record still points at "h1", so the link
        // expected at index 2 ("h1_tampered") will mismatch its found prev_hash.
        ledger.records[1].witness.evidence_hash = "h1_tampered".to_string();

        match ledger.verify() {
            Err(CustodyError::BrokenChain {
                index,
                found,
                expected,
            }) => {
                assert_eq!(index, 2);
                assert_eq!(found, Some("h1".to_string()));
                assert_eq!(expected, Some("h1_tampered".to_string()));
            }
            other => panic!("expected BrokenChain at index 2, got {other:?}"),
        }
    }
}
