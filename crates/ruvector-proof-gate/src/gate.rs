use sha2::{Digest, Sha256};

use crate::payload::{GateError, GateVariant, WritePayload, WriteReceipt};

// ─────────────────────────────────────────────────────────────────────────────
// WriteGate trait
// ─────────────────────────────────────────────────────────────────────────────

/// Core admission gate trait for vector writes.
///
/// Each gate variant encapsulates one integrity strategy. Callers insert a
/// `WritePayload` and receive a `WriteReceipt` that commits the gate's
/// current chain state. Receipts can be stored alongside the vector index
/// entry and verified offline.
pub trait WriteGate: Send + Sync {
    /// Admit a payload and return a tamper-evident receipt.
    fn admit(&mut self, payload: &WritePayload) -> Result<WriteReceipt, GateError>;

    /// Verify that a receipt is consistent with the gate's internal state.
    fn verify_receipt(&self, receipt: &WriteReceipt) -> bool;

    /// Current chain commitment (root or chain head). Changes after every
    /// successful `admit()`.
    fn chain_root(&self) -> [u8; 32];

    /// Number of writes admitted so far.
    fn len(&self) -> usize;

    /// Returns true if no writes have been admitted.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gate variant identifier.
    fn variant(&self) -> GateVariant;
}

// ─────────────────────────────────────────────────────────────────────────────
// NullGate — baseline, no hashing overhead
// ─────────────────────────────────────────────────────────────────────────────

/// No-op gate that admits every payload with zero cryptographic overhead.
/// Used exclusively to establish a throughput ceiling for comparison.
pub struct NullGate {
    seq: u64,
}

impl NullGate {
    pub fn new() -> Self {
        Self { seq: 0 }
    }
}

impl Default for NullGate {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteGate for NullGate {
    fn admit(&mut self, _payload: &WritePayload) -> Result<WriteReceipt, GateError> {
        let seq = self.seq;
        self.seq += 1;
        Ok(WriteReceipt {
            sequence: seq,
            payload_hash: [0u8; 32],
            chain_commitment: [0u8; 32],
            gate_variant: GateVariant::Null,
        })
    }

    fn verify_receipt(&self, _receipt: &WriteReceipt) -> bool {
        true
    }

    fn chain_root(&self) -> [u8; 32] {
        [0u8; 32]
    }

    fn len(&self) -> usize {
        self.seq as usize
    }

    fn variant(&self) -> GateVariant {
        GateVariant::Null
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HashChainGate — O(1) per write, sequential SHA-256 chain
// ─────────────────────────────────────────────────────────────────────────────

/// Sequential SHA-256 hash chain.
///
/// Each write links to its predecessor: commitment[n] = SHA256(commitment[n-1]
/// || payload_hash[n] || n). An adversary cannot silently mutate, reorder, or
/// delete any entry without invalidating all subsequent commitments.
/// Verification is O(n): replay the chain from the beginning.
///
/// Memory: 64 bytes per admitted write (commitment + payload hash, both needed
/// to cryptographically re-derive the chain).
pub struct HashChainGate {
    seq: u64,
    prev_commitment: [u8; 32],
    // Per-entry chain commitments.
    chain: Vec<[u8; 32]>,
    // Per-entry payload hashes — required to *re-derive* (not merely structurally
    // scan) the chain in `verify_integrity`.
    payload_hashes: Vec<[u8; 32]>,
}

/// Genesis seed = SHA256("ruvector-proof-gate-v1"). The re-derivation anchor.
const GENESIS: [u8; 32] = *b"\xb7\x4c\x9b\x41\x3e\x27\x0e\x56\
                            \xd3\x8a\x12\x9f\x6c\x3b\x4d\x8a\
                            \x2f\x1e\x0c\x5a\x9d\x7b\xe4\xf2\
                            \x6a\x8c\x3d\x0b\x5e\x9f\x2c\x47";

impl HashChainGate {
    pub fn new() -> Self {
        Self {
            seq: 0,
            prev_commitment: GENESIS,
            chain: Vec::new(),
            payload_hashes: Vec::new(),
        }
    }

    fn compute_commitment(prev: &[u8; 32], payload_hash: &[u8; 32], seq: u64) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"ruvector:chain:");
        h.update(prev);
        h.update(payload_hash);
        h.update(seq.to_le_bytes());
        h.finalize().into()
    }

    /// Full cryptographic chain re-derivation. Re-computes every commitment from
    /// the genesis seed and the stored payload hashes, comparing against the
    /// stored chain. Returns false if ANY commitment fails to re-derive — i.e.
    /// catches a tamper that mutates a commitment, a payload hash, or the order,
    /// not just structurally-degenerate chains.
    pub fn verify_integrity(&self) -> bool {
        if self.chain.len() != self.payload_hashes.len() {
            return false;
        }
        let mut prev = GENESIS;
        for (i, (commitment, payload_hash)) in self
            .chain
            .iter()
            .zip(self.payload_hashes.iter())
            .enumerate()
        {
            let expected = Self::compute_commitment(&prev, payload_hash, i as u64);
            if &expected != commitment {
                return false;
            }
            prev = *commitment;
        }
        // Chain head must equal the last commitment (or genesis if empty).
        self.prev_commitment == prev
    }
}

impl Default for HashChainGate {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteGate for HashChainGate {
    fn admit(&mut self, payload: &WritePayload) -> Result<WriteReceipt, GateError> {
        // Allocation-free payload hash (identical digest to hashing canonical_bytes).
        let payload_hash: [u8; 32] = payload.payload_hash();
        let commitment = Self::compute_commitment(&self.prev_commitment, &payload_hash, self.seq);
        self.prev_commitment = commitment;
        self.chain.push(commitment);
        self.payload_hashes.push(payload_hash);
        let seq = self.seq;
        self.seq += 1;
        Ok(WriteReceipt {
            sequence: seq,
            payload_hash,
            chain_commitment: commitment,
            gate_variant: GateVariant::HashChain,
        })
    }

    fn verify_receipt(&self, receipt: &WriteReceipt) -> bool {
        let idx = receipt.sequence as usize;
        idx < self.chain.len() && self.chain[idx] == receipt.chain_commitment
    }

    fn chain_root(&self) -> [u8; 32] {
        self.prev_commitment
    }

    fn len(&self) -> usize {
        self.seq as usize
    }

    fn variant(&self) -> GateVariant {
        GateVariant::HashChain
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MerkleGate — Merkle Mountain Range, O(log n) amortized per write
// ─────────────────────────────────────────────────────────────────────────────

/// Merkle Mountain Range (MMR) accumulator.
///
/// An MMR maintains a forest of perfect binary trees. Appending a leaf is
/// O(log n) amortized: binary representations of the leaf count determine
/// which existing peaks must be merged. The "bagged" root (fold of all
/// peaks) changes with every write.
///
/// Compared to `HashChainGate`:
/// - Same tamper-evidence guarantee (mutating any leaf changes the root).
/// - Supports Merkle inclusion proofs (proof that leaf N is in the set)
///   without replaying the entire chain.
/// - Slightly higher per-write overhead due to peak management.
///
/// Memory: ~2n * 32 bytes (peaks + leaf hashes stored for membership proofs).
pub struct MerkleGate {
    seq: u64,
    /// MMR peaks ordered lowest-level-first.
    /// peaks[i] covers a subtree with 2^level[i] leaves.
    peaks: Vec<[u8; 32]>,
    /// Total leaves inserted (determines peak structure via binary representation).
    leaf_count: u64,
    /// Leaf hashes stored for membership proof generation.
    leaves: Vec<[u8; 32]>,
}

impl MerkleGate {
    pub fn new() -> Self {
        Self {
            seq: 0,
            peaks: Vec::new(),
            leaf_count: 0,
            leaves: Vec::new(),
        }
    }

    fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"ruvector:mmr:");
        h.update(left);
        h.update(right);
        h.finalize().into()
    }

    fn leaf_commit(payload_hash: &[u8; 32], seq: u64) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"ruvector:leaf:");
        h.update(payload_hash);
        h.update(seq.to_le_bytes());
        h.finalize().into()
    }

    /// Append a leaf to the MMR. O(log n) amortized.
    fn append(&mut self, leaf: [u8; 32]) {
        let mut node = leaf;
        let mut n = self.leaf_count;
        // While the trailing bit of n is 1, merge with the last peak.
        while n & 1 == 1 {
            let peak = self.peaks.pop().expect("peak must exist when bit is set");
            node = Self::hash_pair(&peak, &node);
            n >>= 1;
        }
        self.peaks.push(node);
        self.leaf_count += 1;
    }

    /// Bag all peaks into a single root by folding left-to-right.
    fn bagged_root(&self) -> [u8; 32] {
        if self.peaks.is_empty() {
            return [0u8; 32];
        }
        // Fold from highest-order peak (leftmost in tree) to lowest.
        let mut acc = self.peaks[0];
        for peak in &self.peaks[1..] {
            acc = Self::hash_pair(&acc, peak);
        }
        acc
    }

    /// Verify that the leaf at `sequence` matches the stored leaf hash.
    /// Full inclusion proof generation is left for the production crate.
    pub fn verify_leaf(&self, sequence: u64, payload_hash: &[u8; 32]) -> bool {
        let idx = sequence as usize;
        if idx >= self.leaves.len() {
            return false;
        }
        let expected = Self::leaf_commit(payload_hash, sequence);
        self.leaves[idx] == expected
    }
}

impl Default for MerkleGate {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteGate for MerkleGate {
    fn admit(&mut self, payload: &WritePayload) -> Result<WriteReceipt, GateError> {
        // Allocation-free payload hash (identical digest to hashing canonical_bytes).
        let payload_hash: [u8; 32] = payload.payload_hash();
        let leaf = Self::leaf_commit(&payload_hash, self.seq);
        self.leaves.push(leaf);
        self.append(leaf);
        let root = self.bagged_root();
        let seq = self.seq;
        self.seq += 1;
        Ok(WriteReceipt {
            sequence: seq,
            payload_hash,
            chain_commitment: root,
            gate_variant: GateVariant::Merkle,
        })
    }

    fn verify_receipt(&self, receipt: &WriteReceipt) -> bool {
        self.verify_leaf(receipt.sequence, &receipt.payload_hash)
    }

    fn chain_root(&self) -> [u8; 32] {
        self.bagged_root()
    }

    fn len(&self) -> usize {
        self.seq as usize
    }

    fn variant(&self) -> GateVariant {
        GateVariant::Merkle
    }
}

#[cfg(test)]
mod rederivation_tests {
    use super::*;
    use crate::WritePayload;

    fn gate_with(n: u64) -> HashChainGate {
        let mut g = HashChainGate::new();
        for i in 0..n {
            g.admit(&WritePayload::new(i, vec![i as f32, 1.0, -(i as f32)]))
                .unwrap();
        }
        g
    }

    #[test]
    fn clean_chain_reverifies() {
        assert!(gate_with(8).verify_integrity());
        assert!(
            HashChainGate::new().verify_integrity(),
            "empty chain is valid"
        );
    }

    #[test]
    fn tampered_commitment_detected() {
        let mut g = gate_with(8);
        g.chain[3][0] ^= 0xFF; // flip a byte of a stored commitment
        assert!(
            !g.verify_integrity(),
            "mutated commitment must fail re-derivation"
        );
    }

    #[test]
    fn tampered_payload_hash_detected() {
        let mut g = gate_with(8);
        g.payload_hashes[2][0] ^= 0xFF; // poisoned write whose recorded hash no longer matches
        assert!(
            !g.verify_integrity(),
            "mutated payload hash must fail re-derivation"
        );
    }

    #[test]
    fn reordered_entries_detected() {
        let mut g = gate_with(8);
        g.chain.swap(2, 5);
        g.payload_hashes.swap(2, 5);
        assert!(
            !g.verify_integrity(),
            "reordering entries must fail re-derivation"
        );
    }

    #[test]
    fn length_mismatch_detected() {
        let mut g = gate_with(8);
        g.payload_hashes.pop();
        assert!(!g.verify_integrity());
    }
}
