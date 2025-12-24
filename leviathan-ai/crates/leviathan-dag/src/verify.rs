//! Cryptographic verification and Merkle tree proofs
//!
//! This module provides cryptographic verification capabilities including:
//! - Merkle tree construction
//! - Proof generation and verification
//! - Tamper detection
//! - Chain integrity validation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A Merkle proof for verifying node membership without the full DAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// The leaf hash being proven
    pub leaf_hash: String,

    /// The Merkle root
    pub root_hash: String,

    /// Path of sibling hashes from leaf to root
    pub proof_path: Vec<ProofNode>,

    /// Index of the leaf in the tree
    pub leaf_index: usize,
}

/// A node in the Merkle proof path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    /// Hash of the sibling node
    pub hash: String,

    /// Whether this sibling is on the left (true) or right (false)
    pub is_left: bool,
}

impl MerkleProof {
    /// Create a new Merkle proof
    pub fn new(
        leaf_hash: String,
        root_hash: String,
        proof_path: Vec<ProofNode>,
        leaf_index: usize,
    ) -> Self {
        Self {
            leaf_hash,
            root_hash,
            proof_path,
            leaf_index,
        }
    }

    /// Verify this proof
    pub fn verify(&self) -> bool {
        verify_proof(self)
    }
}

/// Verify a Merkle proof
pub fn verify_proof(proof: &MerkleProof) -> bool {
    let mut current_hash = proof.leaf_hash.clone();

    // Walk up the tree, hashing with siblings
    for node in &proof.proof_path {
        current_hash = if node.is_left {
            // Sibling is on the left, current is on the right
            combine_hashes(&node.hash, &current_hash)
        } else {
            // Sibling is on the right, current is on the left
            combine_hashes(&current_hash, &node.hash)
        };
    }

    // Final hash should match the root
    current_hash == proof.root_hash
}

/// Combine two hashes to create a parent hash
fn combine_hashes(left: &str, right: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(left.as_bytes());
    hasher.update(right.as_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Compute the Merkle root from a list of leaf hashes
pub fn compute_merkle_root(leaf_hashes: &[String]) -> String {
    if leaf_hashes.is_empty() {
        return String::new();
    }

    if leaf_hashes.len() == 1 {
        return leaf_hashes[0].clone();
    }

    let mut current_level = leaf_hashes.to_vec();

    // Build the tree level by level
    while current_level.len() > 1 {
        let mut next_level = Vec::new();

        // Process pairs
        for chunk in current_level.chunks(2) {
            let combined = if chunk.len() == 2 {
                combine_hashes(&chunk[0], &chunk[1])
            } else {
                // Odd number: duplicate the last hash
                combine_hashes(&chunk[0], &chunk[0])
            };
            next_level.push(combined);
        }

        current_level = next_level;
    }

    current_level[0].clone()
}

/// Generate a Merkle proof for a specific leaf
pub fn generate_proof(leaf_hashes: &[String], leaf_index: usize) -> Option<MerkleProof> {
    if leaf_index >= leaf_hashes.len() {
        return None;
    }

    let leaf_hash = leaf_hashes[leaf_index].clone();
    let root_hash = compute_merkle_root(leaf_hashes);

    if leaf_hashes.len() == 1 {
        return Some(MerkleProof::new(
            leaf_hash,
            root_hash,
            vec![],
            leaf_index,
        ));
    }

    let mut proof_path = Vec::new();
    let mut current_level = leaf_hashes.to_vec();
    let mut current_index = leaf_index;

    // Build proof path level by level
    while current_level.len() > 1 {
        let sibling_index = if current_index % 2 == 0 {
            // Current is left child, sibling is right
            current_index + 1
        } else {
            // Current is right child, sibling is left
            current_index - 1
        };

        // Get sibling hash (duplicate if at end)
        let sibling_hash = if sibling_index < current_level.len() {
            current_level[sibling_index].clone()
        } else {
            current_level[current_index].clone()
        };

        proof_path.push(ProofNode {
            hash: sibling_hash,
            is_left: sibling_index < current_index,
        });

        // Move to next level
        let mut next_level = Vec::new();
        for chunk in current_level.chunks(2) {
            let combined = if chunk.len() == 2 {
                combine_hashes(&chunk[0], &chunk[1])
            } else {
                combine_hashes(&chunk[0], &chunk[0])
            };
            next_level.push(combined);
        }

        current_level = next_level;
        current_index /= 2;
    }

    Some(MerkleProof::new(
        leaf_hash,
        root_hash,
        proof_path,
        leaf_index,
    ))
}

/// Verifier for DAG integrity
#[derive(Debug, Clone)]
pub struct DagVerifier {
    /// Known good hashes for verification
    known_hashes: HashMap<Uuid, String>,
}

impl DagVerifier {
    /// Create a new DAG verifier
    pub fn new() -> Self {
        Self {
            known_hashes: HashMap::new(),
        }
    }

    /// Register a known good hash
    pub fn register_hash(&mut self, node_id: Uuid, hash: String) {
        self.known_hashes.insert(node_id, hash);
    }

    /// Verify a node's hash against known good value
    pub fn verify_node(&self, node_id: &Uuid, hash: &str) -> VerificationResult {
        match self.known_hashes.get(node_id) {
            Some(known_hash) if known_hash == hash => VerificationResult::Valid,
            Some(_) => VerificationResult::Tampered,
            None => VerificationResult::Unknown,
        }
    }

    /// Batch verify multiple nodes
    pub fn verify_batch(&self, nodes: &[(Uuid, String)]) -> BatchVerificationResult {
        let mut valid = 0;
        let mut tampered = Vec::new();
        let mut unknown = Vec::new();

        for (id, hash) in nodes {
            match self.verify_node(id, hash) {
                VerificationResult::Valid => valid += 1,
                VerificationResult::Tampered => tampered.push(*id),
                VerificationResult::Unknown => unknown.push(*id),
            }
        }

        BatchVerificationResult {
            total: nodes.len(),
            valid,
            tampered,
            unknown,
        }
    }
}

impl Default for DagVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a single node verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    /// Hash matches known good value
    Valid,

    /// Hash does not match (potential tampering)
    Tampered,

    /// No known good value to compare against
    Unknown,
}

/// Result of batch verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerificationResult {
    /// Total number of nodes verified
    pub total: usize,

    /// Number of valid nodes
    pub valid: usize,

    /// List of tampered node IDs
    pub tampered: Vec<Uuid>,

    /// List of unknown node IDs
    pub unknown: Vec<Uuid>,
}

impl BatchVerificationResult {
    /// Check if all nodes are valid
    pub fn is_fully_valid(&self) -> bool {
        self.tampered.is_empty() && self.unknown.is_empty()
    }

    /// Get the percentage of valid nodes
    pub fn valid_percentage(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.valid as f64 / self.total as f64) * 100.0
    }
}

/// Tamper detection system
#[derive(Debug, Clone)]
pub struct TamperDetector {
    /// Baseline hashes for comparison
    baseline: HashMap<Uuid, String>,
}

impl TamperDetector {
    /// Create a new tamper detector
    pub fn new() -> Self {
        Self {
            baseline: HashMap::new(),
        }
    }

    /// Establish a baseline snapshot
    pub fn create_baseline(&mut self, nodes: Vec<(Uuid, String)>) {
        self.baseline.clear();
        for (id, hash) in nodes {
            self.baseline.insert(id, hash);
        }
    }

    /// Detect tampering by comparing current state to baseline
    pub fn detect_tampering(&self, current_nodes: &[(Uuid, String)]) -> TamperReport {
        let mut modified = Vec::new();
        let mut added = Vec::new();
        let mut removed = Vec::new();

        let current_map: HashMap<_, _> = current_nodes.iter()
            .map(|(id, hash)| (*id, hash.clone()))
            .collect();

        // Check for modifications and removals
        for (id, baseline_hash) in &self.baseline {
            match current_map.get(id) {
                Some(current_hash) if current_hash != baseline_hash => {
                    modified.push(*id);
                }
                None => {
                    removed.push(*id);
                }
                _ => {} // Unchanged
            }
        }

        // Check for additions
        for id in current_map.keys() {
            if !self.baseline.contains_key(id) {
                added.push(*id);
            }
        }

        TamperReport {
            modified,
            added,
            removed,
        }
    }
}

impl Default for TamperDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Report of detected tampering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TamperReport {
    /// Nodes that were modified
    pub modified: Vec<Uuid>,

    /// Nodes that were added
    pub added: Vec<Uuid>,

    /// Nodes that were removed
    pub removed: Vec<Uuid>,
}

impl TamperReport {
    /// Check if any tampering was detected
    pub fn has_tampering(&self) -> bool {
        !self.modified.is_empty() || !self.removed.is_empty()
    }

    /// Get total number of changes
    pub fn total_changes(&self) -> usize {
        self.modified.len() + self.added.len() + self.removed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_hashes() {
        let hash1 = "abc123";
        let hash2 = "def456";

        let combined = combine_hashes(hash1, hash2);
        assert!(!combined.is_empty());

        // Order matters
        let reversed = combine_hashes(hash2, hash1);
        assert_ne!(combined, reversed);
    }

    #[test]
    fn test_merkle_root() {
        let hashes = vec![
            "hash1".to_string(),
            "hash2".to_string(),
            "hash3".to_string(),
            "hash4".to_string(),
        ];

        let root = compute_merkle_root(&hashes);
        assert!(!root.is_empty());
    }

    #[test]
    fn test_merkle_proof() {
        let hashes = vec![
            "hash1".to_string(),
            "hash2".to_string(),
            "hash3".to_string(),
            "hash4".to_string(),
        ];

        let proof = generate_proof(&hashes, 1).unwrap();
        assert!(proof.verify());
    }

    #[test]
    fn test_single_leaf_proof() {
        let hashes = vec!["hash1".to_string()];

        let proof = generate_proof(&hashes, 0).unwrap();
        assert!(proof.verify());
        assert_eq!(proof.proof_path.len(), 0);
    }

    #[test]
    fn test_dag_verifier() {
        let mut verifier = DagVerifier::new();
        let id = Uuid::new_v4();
        let hash = "abc123".to_string();

        verifier.register_hash(id, hash.clone());

        assert_eq!(
            verifier.verify_node(&id, &hash),
            VerificationResult::Valid
        );

        assert_eq!(
            verifier.verify_node(&id, "different"),
            VerificationResult::Tampered
        );

        let unknown_id = Uuid::new_v4();
        assert_eq!(
            verifier.verify_node(&unknown_id, &hash),
            VerificationResult::Unknown
        );
    }

    #[test]
    fn test_batch_verification() {
        let mut verifier = DagVerifier::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        verifier.register_hash(id1, "hash1".to_string());
        verifier.register_hash(id2, "hash2".to_string());

        let nodes = vec![
            (id1, "hash1".to_string()),
            (id2, "hash2".to_string()),
        ];

        let result = verifier.verify_batch(&nodes);
        assert!(result.is_fully_valid());
        assert_eq!(result.valid_percentage(), 100.0);
    }

    #[test]
    fn test_tamper_detector() {
        let mut detector = TamperDetector::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let baseline = vec![
            (id1, "hash1".to_string()),
            (id2, "hash2".to_string()),
        ];

        detector.create_baseline(baseline);

        // Modify one hash
        let current = vec![
            (id1, "hash1_modified".to_string()),
            (id2, "hash2".to_string()),
        ];

        let report = detector.detect_tampering(&current);
        assert!(report.has_tampering());
        assert_eq!(report.modified.len(), 1);
        assert!(report.modified.contains(&id1));
    }

    #[test]
    fn test_tamper_detection_additions() {
        let mut detector = TamperDetector::new();

        let id1 = Uuid::new_v4();
        let baseline = vec![(id1, "hash1".to_string())];

        detector.create_baseline(baseline);

        let id2 = Uuid::new_v4();
        let current = vec![
            (id1, "hash1".to_string()),
            (id2, "hash2".to_string()),
        ];

        let report = detector.detect_tampering(&current);
        assert_eq!(report.added.len(), 1);
        assert!(report.added.contains(&id2));
    }
}
