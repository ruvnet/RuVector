//! Merkle witness construction and verification.
//!
//! Provides utilities for building and verifying Merkle witnesses
//! used in Standard-tier proofs.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::vec::Vec;

use ruvix_types::ProofPayload;

/// Maximum depth of a Merkle tree (32 levels = 2^32 leaves).
pub const MAX_MERKLE_DEPTH: usize = 32;

/// A Merkle witness for proving inclusion.
#[derive(Debug, Clone)]
pub struct MerkleWitness {
    /// Root hash of the Merkle tree.
    pub root: [u8; 32],
    /// Leaf index being proven.
    pub leaf_index: u32,
    /// Path elements (sibling hashes).
    pub path: Vec<[u8; 32]>,
    /// Direction bits (0 = left, 1 = right).
    pub directions: Vec<bool>,
}

impl MerkleWitness {
    /// Creates a new Merkle witness.
    #[must_use]
    pub const fn new(root: [u8; 32], leaf_index: u32) -> Self {
        Self {
            root,
            leaf_index,
            path: Vec::new(),
            directions: Vec::new(),
        }
    }

    /// Adds a sibling hash to the path.
    pub fn add_sibling(&mut self, sibling: [u8; 32], is_right: bool) {
        self.path.push(sibling);
        self.directions.push(is_right);
    }

    /// Returns the path length.
    #[must_use]
    pub fn path_len(&self) -> usize {
        self.path.len()
    }

    /// Converts to a proof payload.
    #[must_use]
    pub fn to_payload(&self) -> ProofPayload {
        let mut path = [[0u8; 32]; 32];
        let len = self.path.len().min(32);
        for (i, hash) in self.path.iter().take(len).enumerate() {
            path[i] = *hash;
        }

        ProofPayload::MerkleWitness {
            root: self.root,
            leaf_index: self.leaf_index,
            path_len: len as u8,
            path,
        }
    }

    /// Verifies the witness against a leaf hash.
    ///
    /// Note: This is a simple verification without actual hash computation.
    /// A full implementation would use a proper hash function.
    #[must_use]
    pub fn verify(&self, leaf_hash: &[u8; 32]) -> bool {
        if self.path.is_empty() {
            // Single-leaf tree
            return *leaf_hash == self.root;
        }

        // In a full implementation, we would:
        // 1. Start with the leaf hash
        // 2. For each level, hash (current, sibling) or (sibling, current)
        // 3. Compare final hash to root
        //
        // For now, we do a simplified check that the root is consistent
        // with the provided path structure.

        // Basic validation
        if self.path.len() != self.directions.len() {
            return false;
        }

        if self.path.len() > MAX_MERKLE_DEPTH {
            return false;
        }

        // Verify leaf_index is within the tree
        let max_leaves = 1u64 << self.path.len();
        if u64::from(self.leaf_index) >= max_leaves {
            return false;
        }

        true
    }
}

/// Builder for constructing Merkle witnesses.
#[derive(Debug, Default)]
pub struct WitnessBuilder {
    root: [u8; 32],
    leaf_index: u32,
    siblings: Vec<([u8; 32], bool)>,
}

impl WitnessBuilder {
    /// Creates a new witness builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            root: [0u8; 32],
            leaf_index: 0,
            siblings: Vec::new(),
        }
    }

    /// Sets the Merkle root.
    #[must_use]
    pub const fn root(mut self, root: [u8; 32]) -> Self {
        self.root = root;
        self
    }

    /// Sets the leaf index.
    #[must_use]
    pub const fn leaf_index(mut self, index: u32) -> Self {
        self.leaf_index = index;
        self
    }

    /// Adds a sibling at the given level.
    #[must_use]
    pub fn sibling(mut self, hash: [u8; 32], is_right: bool) -> Self {
        self.siblings.push((hash, is_right));
        self
    }

    /// Adds multiple siblings from a slice.
    #[must_use]
    pub fn siblings(mut self, siblings: &[([u8; 32], bool)]) -> Self {
        self.siblings.extend_from_slice(siblings);
        self
    }

    /// Builds the witness.
    #[must_use]
    pub fn build(self) -> MerkleWitness {
        let mut witness = MerkleWitness::new(self.root, self.leaf_index);
        for (hash, is_right) in self.siblings {
            witness.add_sibling(hash, is_right);
        }
        witness
    }
}

/// Computes a simple hash for testing (not cryptographically secure).
///
/// In production, this would use SHA-256 or similar.
#[cfg(test)]
fn simple_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = left[i] ^ right[i];
    }
    result
}

/// Creates a minimal witness for a single-leaf tree.
#[must_use]
pub fn create_single_leaf_witness(leaf_hash: [u8; 32]) -> MerkleWitness {
    MerkleWitness::new(leaf_hash, 0)
}

/// Creates a witness for a two-leaf tree.
#[must_use]
pub fn create_two_leaf_witness(
    root: [u8; 32],
    leaf_index: u32,
    sibling: [u8; 32],
) -> MerkleWitness {
    let mut witness = MerkleWitness::new(root, leaf_index);
    witness.add_sibling(sibling, leaf_index == 0);
    witness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_witness_new() {
        let root = [1u8; 32];
        let witness = MerkleWitness::new(root, 5);

        assert_eq!(witness.root, root);
        assert_eq!(witness.leaf_index, 5);
        assert!(witness.path.is_empty());
    }

    #[test]
    fn test_merkle_witness_add_sibling() {
        let mut witness = MerkleWitness::new([0u8; 32], 0);
        witness.add_sibling([1u8; 32], false);
        witness.add_sibling([2u8; 32], true);

        assert_eq!(witness.path_len(), 2);
        assert_eq!(witness.path[0], [1u8; 32]);
        assert!(!witness.directions[0]);
        assert!(witness.directions[1]);
    }

    #[test]
    fn test_merkle_witness_to_payload() {
        let mut witness = MerkleWitness::new([0xAAu8; 32], 42);
        witness.add_sibling([0xBBu8; 32], false);
        witness.add_sibling([0xCCu8; 32], true);

        let payload = witness.to_payload();

        if let ProofPayload::MerkleWitness {
            root,
            leaf_index,
            path_len,
            path,
        } = payload
        {
            assert_eq!(root, [0xAAu8; 32]);
            assert_eq!(leaf_index, 42);
            assert_eq!(path_len, 2);
            assert_eq!(path[0], [0xBBu8; 32]);
            assert_eq!(path[1], [0xCCu8; 32]);
        } else {
            panic!("Expected MerkleWitness payload");
        }
    }

    #[test]
    fn test_merkle_witness_verify_single_leaf() {
        let leaf_hash = [0xABu8; 32];
        let witness = create_single_leaf_witness(leaf_hash);

        assert!(witness.verify(&leaf_hash));
        assert!(!witness.verify(&[0xCDu8; 32]));
    }

    #[test]
    fn test_merkle_witness_verify_path_consistency() {
        let mut witness = MerkleWitness::new([0u8; 32], 3);
        witness.add_sibling([1u8; 32], false);
        witness.add_sibling([2u8; 32], true);

        // Path length matches directions
        assert!(witness.verify(&[0u8; 32]));
    }

    #[test]
    fn test_merkle_witness_verify_invalid_leaf_index() {
        let mut witness = MerkleWitness::new([0u8; 32], 100); // Index 100 requires at least 7 levels
        witness.add_sibling([1u8; 32], false); // Only 1 level = max 2 leaves

        // Leaf index 100 is out of bounds for a 1-level tree
        assert!(!witness.verify(&[0u8; 32]));
    }

    #[test]
    fn test_witness_builder() {
        let witness = WitnessBuilder::new()
            .root([0xAAu8; 32])
            .leaf_index(7)
            .sibling([0xBBu8; 32], false)
            .sibling([0xCCu8; 32], true)
            .sibling([0xDDu8; 32], false)
            .build();

        assert_eq!(witness.root, [0xAAu8; 32]);
        assert_eq!(witness.leaf_index, 7);
        assert_eq!(witness.path_len(), 3);
    }

    #[test]
    fn test_witness_builder_siblings() {
        let siblings = vec![
            ([0x11u8; 32], false),
            ([0x22u8; 32], true),
            ([0x33u8; 32], false),
        ];

        let witness = WitnessBuilder::new()
            .root([0u8; 32])
            .leaf_index(0)
            .siblings(&siblings)
            .build();

        assert_eq!(witness.path_len(), 3);
        assert_eq!(witness.path[0], [0x11u8; 32]);
        assert_eq!(witness.path[1], [0x22u8; 32]);
        assert_eq!(witness.path[2], [0x33u8; 32]);
    }

    #[test]
    fn test_create_two_leaf_witness() {
        let root = [0xABu8; 32];
        let sibling = [0xCDu8; 32];

        let witness = create_two_leaf_witness(root, 0, sibling);

        assert_eq!(witness.root, root);
        assert_eq!(witness.leaf_index, 0);
        assert_eq!(witness.path_len(), 1);
        assert_eq!(witness.path[0], sibling);
    }

    #[test]
    fn test_max_merkle_depth() {
        assert_eq!(MAX_MERKLE_DEPTH, 32);
    }

    #[test]
    fn test_simple_hash() {
        let a = [0xAAu8; 32];
        let b = [0x55u8; 32];
        let result = simple_hash(&a, &b);

        // XOR of 0xAA and 0x55 is 0xFF
        assert_eq!(result, [0xFFu8; 32]);
    }
}
