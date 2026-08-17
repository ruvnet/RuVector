//! SHA-256 hash-chain witness over `adaptive_partition`'s split decisions.
//!
//! Each recursion step commits `(prev_hash, hash(parent_vertices),
//! cut_value, hash(side_a), hash(side_b), accepted)` into the next link.
//! A verifier who only has the recorded chain (not the raw partition) can
//! confirm the chain is internally consistent — no step's recorded cut
//! value or vertex-set hash was edited after the fact without breaking
//! every subsequent link — and, given the actual vertex sets, can confirm
//! a specific step's hashes were genuinely derived from them via
//! `verify_step_matches`. This does not certify the *min cut itself* is
//! optimal (that guarantee comes from `ruvector_mincut`'s algorithm, not
//! from hashing); it certifies the partition decision used the same
//! vertex sets and cut value throughout, the same property
//! `ruvector-retrieval-receipt` (nightly 2026-08-13) established for
//! query result sets.

use ruvector_mincut::VertexId;
use serde::Serialize;
use sha2::{Digest, Sha256};

const GENESIS: &str = "genesis";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hash_vertex_set(verts: &[VertexId]) -> String {
    let mut sorted = verts.to_vec();
    sorted.sort_unstable();
    let mut hasher = Sha256::new();
    for v in &sorted {
        hasher.update(v.to_le_bytes());
    }
    hex(&hasher.finalize())
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitRecord {
    pub parent_hash: String,
    pub parent_size: usize,
    pub cut_value: f64,
    pub side_a_hash: String,
    pub side_a_size: usize,
    pub side_b_hash: String,
    pub side_b_size: usize,
    pub accepted: bool,
    pub step_hash: String,
}

fn step_hash(
    prev_hash: &str,
    parent_hash: &str,
    cut_value: f64,
    side_a_hash: &str,
    side_b_hash: &str,
    accepted: bool,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prev_hash.as_bytes());
    hasher.update(parent_hash.as_bytes());
    hasher.update(cut_value.to_le_bytes());
    hasher.update(side_a_hash.as_bytes());
    hasher.update(side_b_hash.as_bytes());
    hasher.update([accepted as u8]);
    hex(&hasher.finalize())
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PartitionWitnessChain {
    pub steps: Vec<SplitRecord>,
}

impl PartitionWitnessChain {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn record(
        &mut self,
        parent: &[VertexId],
        cut_value: f64,
        side_a: &[VertexId],
        side_b: &[VertexId],
        accepted: bool,
    ) -> &SplitRecord {
        let prev = self
            .steps
            .last()
            .map(|s| s.step_hash.clone())
            .unwrap_or_else(|| GENESIS.to_string());
        let parent_hash = hash_vertex_set(parent);
        let side_a_hash = hash_vertex_set(side_a);
        let side_b_hash = hash_vertex_set(side_b);
        let hash = step_hash(
            &prev,
            &parent_hash,
            cut_value,
            &side_a_hash,
            &side_b_hash,
            accepted,
        );
        self.steps.push(SplitRecord {
            parent_hash,
            parent_size: parent.len(),
            cut_value,
            side_a_hash,
            side_a_size: side_a.len(),
            side_b_hash,
            side_b_size: side_b.len(),
            accepted,
            step_hash: hash,
        });
        self.steps.last().unwrap()
    }

    pub fn head(&self) -> &str {
        self.steps
            .last()
            .map(|s| s.step_hash.as_str())
            .unwrap_or(GENESIS)
    }

    /// Recompute every link from the stored fields and confirm it matches
    /// the recorded `step_hash`. Detects any post-hoc edit to a step.
    pub fn verify(&self) -> bool {
        let mut prev = GENESIS.to_string();
        for step in &self.steps {
            let recomputed = step_hash(
                &prev,
                &step.parent_hash,
                step.cut_value,
                &step.side_a_hash,
                &step.side_b_hash,
                step.accepted,
            );
            if recomputed != step.step_hash {
                return false;
            }
            prev = step.step_hash.clone();
        }
        true
    }

    /// Confirm step `idx`'s recorded hashes were genuinely derived from
    /// these vertex sets (not just internally self-consistent).
    pub fn verify_step_matches(
        &self,
        idx: usize,
        parent: &[VertexId],
        side_a: &[VertexId],
        side_b: &[VertexId],
    ) -> bool {
        match self.steps.get(idx) {
            Some(step) => {
                step.parent_hash == hash_vertex_set(parent)
                    && step.side_a_hash == hash_vertex_set(side_a)
                    && step.side_b_hash == hash_vertex_set(side_b)
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_verifies_when_untampered() {
        let mut chain = PartitionWitnessChain::new();
        chain.record(&[1, 2, 3, 4], 1.5, &[1, 2], &[3, 4], true);
        chain.record(&[1, 2], 2.0, &[1], &[2], false);
        assert!(chain.verify());
        assert!(chain.verify_step_matches(0, &[1, 2, 3, 4], &[1, 2], &[3, 4]));
    }

    #[test]
    fn chain_breaks_when_a_field_is_edited_after_the_fact() {
        let mut chain = PartitionWitnessChain::new();
        chain.record(&[1, 2, 3, 4], 1.5, &[1, 2], &[3, 4], true);
        chain.steps[0].cut_value = 99.0; // tamper
        assert!(!chain.verify());
    }

    #[test]
    fn step_does_not_match_a_different_vertex_set() {
        let mut chain = PartitionWitnessChain::new();
        chain.record(&[1, 2, 3, 4], 1.5, &[1, 2], &[3, 4], true);
        assert!(!chain.verify_step_matches(0, &[1, 2, 3, 5], &[1, 2], &[3, 5]));
    }
}
