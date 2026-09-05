//! Hash-chained witness log for repair actions: every repair (or would-be
//! alert) is recorded as a receipt whose hash includes the previous
//! receipt's hash, so the sequence can be verified end-to-end and any
//! tampering or dropped entry is detectable. Same technique as the
//! 2026-08-13 retrieval-receipts nightly, implemented fresh here as a small
//! self-contained log rather than a cross-crate dependency.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairReceipt {
    pub seq: u64,
    pub step: u64,
    pub nodes_touched: Vec<usize>,
    pub edges_added: usize,
    pub edges_removed: usize,
    pub scs_before: f64,
    pub scs_after: f64,
    pub prev_hash: String,
    pub hash: String,
}

#[derive(Debug, Default)]
pub struct WitnessLog {
    pub receipts: Vec<RepairReceipt>,
}

fn hash_receipt(
    seq: u64,
    step: u64,
    nodes_touched: &[usize],
    edges_added: usize,
    edges_removed: usize,
    scs_before: f64,
    scs_after: f64,
    prev_hash: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(seq.to_le_bytes());
    hasher.update(step.to_le_bytes());
    for n in nodes_touched {
        hasher.update(n.to_le_bytes());
    }
    hasher.update(edges_added.to_le_bytes());
    hasher.update(edges_removed.to_le_bytes());
    hasher.update(scs_before.to_le_bytes());
    hasher.update(scs_after.to_le_bytes());
    hasher.update(prev_hash.as_bytes());
    format!("{:x}", hasher.finalize())
}

impl WitnessLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn genesis_hash() -> String {
        "0".repeat(64)
    }

    /// Appends a repair receipt, chaining it to the previous entry's hash.
    pub fn record(
        &mut self,
        step: u64,
        nodes_touched: Vec<usize>,
        edges_added: usize,
        edges_removed: usize,
        scs_before: f64,
        scs_after: f64,
    ) -> &RepairReceipt {
        let seq = self.receipts.len() as u64;
        let prev_hash = self
            .receipts
            .last()
            .map(|r| r.hash.clone())
            .unwrap_or_else(Self::genesis_hash);
        let hash = hash_receipt(
            seq,
            step,
            &nodes_touched,
            edges_added,
            edges_removed,
            scs_before,
            scs_after,
            &prev_hash,
        );
        self.receipts.push(RepairReceipt {
            seq,
            step,
            nodes_touched,
            edges_added,
            edges_removed,
            scs_before,
            scs_after,
            prev_hash,
            hash,
        });
        self.receipts.last().unwrap()
    }

    /// Recomputes every hash in the chain and checks it against the stored
    /// value and against the next entry's `prev_hash`. Returns `Ok(())` iff
    /// the entire chain is intact.
    pub fn verify(&self) -> Result<(), String> {
        let mut prev = Self::genesis_hash();
        for r in &self.receipts {
            if r.prev_hash != prev {
                return Err(format!("seq {}: prev_hash mismatch", r.seq));
            }
            let recomputed = hash_receipt(
                r.seq,
                r.step,
                &r.nodes_touched,
                r.edges_added,
                r.edges_removed,
                r.scs_before,
                r.scs_after,
                &r.prev_hash,
            );
            if recomputed != r.hash {
                return Err(format!("seq {}: hash mismatch", r.seq));
            }
            prev = r.hash.clone();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_chain_verifies() {
        let mut log = WitnessLog::new();
        log.record(10, vec![1, 2], 2, 0, 0.3, 0.5);
        log.record(20, vec![3], 1, 1, 0.5, 0.55);
        assert!(log.verify().is_ok());
    }

    #[test]
    fn tampering_is_detected() {
        let mut log = WitnessLog::new();
        log.record(10, vec![1, 2], 2, 0, 0.3, 0.5);
        log.record(20, vec![3], 1, 1, 0.5, 0.55);
        log.receipts[0].scs_after = 0.99; // tamper without recomputing hash
        assert!(log.verify().is_err());
    }
}
