use sha2::{Digest, Sha256};

use crate::index::ResultItem;

/// Which receipt strategy produced a given `RetrievalReceipt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptVariant {
    /// No provenance attached — establishes the search-only latency floor.
    None,
    /// Sequential SHA-256 chain over the result set, one commitment per
    /// result (mirrors `ruvector_proof_gate::HashChainGate`).
    PerResult,
    /// Binary Merkle tree over the result set with O(log k) inclusion
    /// proofs (mirrors the membership-proof property of
    /// `ruvector_proof_gate::MerkleGate`, applied to reads instead of
    /// writes).
    Merkle,
}

const QUERY_GENESIS: [u8; 32] = *b"\x71\x75\x65\x72\x79\x2d\x67\x65\
                                    \x6e\x65\x73\x69\x73\x2d\x76\x31\
                                    \x00\x00\x00\x00\x00\x00\x00\x00\
                                    \x00\x00\x00\x00\x00\x00\x00\x00";

/// SHA-256 of the query vector's raw bytes. Bound into every leaf so a
/// receipt attests to *which query* produced the result set, not just the
/// result set in isolation (prevents a receipt from being replayed against
/// a different query).
pub fn query_hash(query: &[f32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"ruvector:retrieval:query:");
    for f in query {
        h.update(f.to_le_bytes());
    }
    h.finalize().into()
}

/// Domain-separated leaf commitment for one result. Binds: the query, the
/// index state root at query time, the result's rank/id/score, and *copies*
/// of the write receipt's gate variant, chain commitment, and payload hash.
///
/// Threat model — stated precisely so it is not over-read:
///
/// - The receipt detects **post-issuance mutation** of a receipt/result
///   pair (in transit or in storage). That is the whole guarantee.
/// - It does **not** protect against a dishonest query engine: leaves are
///   engine-chosen and unsigned; nothing binds `score` to an actual cosine
///   computation or the committed set to the true top-k.
/// - It does **not** prove write-chain membership: verification recomputes
///   hashes over the caller-supplied copies of the `WriteReceipt` fields
///   and never consults the write gate, so mutating the ingestion history
///   *after* a receipt is issued leaves that receipt verifying. Making the
///   write→read link real requires an offline membership proof — i.e.
///   anchoring each leaf to `ruvector_proof_gate::MerkleGate`'s MMR
///   inclusion proofs. That is the named future-work item, not implemented
///   here (`HashChainGate::verify_receipt` requires the live gate's full
///   chain and offers no offline membership proof).
fn result_leaf(query_hash: &[u8; 32], index_root: &[u8; 32], item: &ResultItem) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"ruvector:retrieval:leaf:");
    h.update(query_hash);
    h.update(index_root);
    h.update(item.vector_id.to_le_bytes());
    h.update(item.rank.to_le_bytes());
    h.update(item.score.to_bits().to_le_bytes());
    // gate_variant is bound so a NullGate receipt (all-zero commitment and
    // payload hash) cannot masquerade as a gated one under the same leaf.
    h.update([item.write_receipt.gate_variant as u8]);
    h.update(item.write_receipt.chain_commitment);
    h.update(item.write_receipt.payload_hash);
    h.finalize().into()
}

fn chain_step(prev: &[u8; 32], leaf: &[u8; 32], seq: u64) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"ruvector:retrieval:chain:");
    h.update(prev);
    h.update(leaf);
    h.update(seq.to_le_bytes());
    h.finalize().into()
}

fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"ruvector:retrieval:node:"); // domain-separated from leaf hashing
    h.update(left);
    h.update(right);
    h.finalize().into()
}

// ─────────────────────────────────────────────────────────────────────────
// PerResultReceipt — sequential chain, one commitment per result
// ─────────────────────────────────────────────────────────────────────────

/// Sequential SHA-256 chain over a single query's result set. Verifying
/// result `i` in isolation (without trusting a locally-held gate instance)
/// requires replaying the chain from the genesis using `leaves[0..=i]`:
/// O(i) work and O(i) bytes of evidence.
#[derive(Debug, Clone)]
pub struct PerResultReceipt {
    pub leaves: Vec<[u8; 32]>,
    pub commitments: Vec<[u8; 32]>,
    pub chain_head: [u8; 32],
}

impl PerResultReceipt {
    pub fn build(qh: [u8; 32], index_root: [u8; 32], results: &[ResultItem]) -> Self {
        let mut leaves = Vec::with_capacity(results.len());
        let mut commitments = Vec::with_capacity(results.len());
        let mut prev = QUERY_GENESIS;
        for (i, item) in results.iter().enumerate() {
            let leaf = result_leaf(&qh, &index_root, item);
            let commitment = chain_step(&prev, &leaf, i as u64);
            leaves.push(leaf);
            commitments.push(commitment);
            prev = commitment;
        }
        let chain_head = prev;
        Self {
            leaves,
            commitments,
            chain_head,
        }
    }

    /// Bytes of evidence a verifier must hold to independently re-derive
    /// and check result `idx`: the genesis-anchored replay of leaves
    /// `0..=idx`, 32 bytes each.
    pub fn proof_bytes_for(&self, idx: usize) -> usize {
        (idx + 1) * 32
    }

    pub fn verify_item(
        &self,
        idx: usize,
        qh: [u8; 32],
        index_root: [u8; 32],
        item: &ResultItem,
    ) -> bool {
        if idx >= self.leaves.len() || idx >= self.commitments.len() {
            return false;
        }
        let recomputed = result_leaf(&qh, &index_root, item);
        if recomputed != self.leaves[idx] {
            return false;
        }
        let mut prev = QUERY_GENESIS;
        for (i, leaf) in self.leaves.iter().enumerate().take(idx + 1) {
            prev = chain_step(&prev, leaf, i as u64);
        }
        prev == self.commitments[idx]
    }

    /// Empty result sets fail closed: with zero leaves the length check and
    /// the per-item loop would both pass vacuously, making "no evidence"
    /// indistinguishable from "verified evidence". A caller with a
    /// legitimately empty result set has nothing to prove and must not
    /// treat a receipt as attesting to it.
    pub fn verify_full(&self, qh: [u8; 32], index_root: [u8; 32], results: &[ResultItem]) -> bool {
        if results.is_empty() || results.len() != self.leaves.len() {
            return false;
        }
        (0..results.len()).all(|i| self.verify_item(i, qh, index_root, &results[i]))
    }

    pub fn total_bytes(&self) -> usize {
        (self.leaves.len() + self.commitments.len()) * 32
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MerkleReceipt — binary Merkle tree, O(log k) inclusion proofs
// ─────────────────────────────────────────────────────────────────────────

/// Binary Merkle tree over a single query's result set.
///
/// Known limitation (documented, not hidden): odd-width levels are padded
/// by duplicating the last node, the classic scheme used by early Bitcoin
/// and Certificate Transparency implementations. That scheme has a known
/// malleability weakness (CVE-2012-2459-style) when an adversary controls
/// the leaf *set*. Here the leaf set is exactly the server's own top-k
/// output — the querying client never supplies leaves — so the weakness
/// does not create a forgeable receipt in this deployment shape, but it
/// would need to change (e.g. RFC 6962 bit-length prefixing) before this
/// scheme is reused anywhere an untrusted party can influence leaf count.
#[derive(Debug, Clone)]
pub struct MerkleReceipt {
    pub leaves: Vec<[u8; 32]>,
    levels: Vec<Vec<[u8; 32]>>, // levels[0] = padded leaves ... levels.last() = [root]
    pub root: [u8; 32],
}

impl MerkleReceipt {
    pub fn build(qh: [u8; 32], index_root: [u8; 32], results: &[ResultItem]) -> Self {
        let leaves: Vec<[u8; 32]> = results
            .iter()
            .map(|item| result_leaf(&qh, &index_root, item))
            .collect();
        let levels = Self::build_levels(&leaves);
        let root = levels.last().map(|l| l[0]).unwrap_or([0u8; 32]);
        Self {
            leaves,
            levels,
            root,
        }
    }

    fn build_levels(leaves: &[[u8; 32]]) -> Vec<Vec<[u8; 32]>> {
        if leaves.is_empty() {
            return vec![vec![[0u8; 32]]];
        }
        let mut levels = vec![leaves.to_vec()];
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity(cur.len().div_ceil(2));
            let mut i = 0;
            while i < cur.len() {
                if i + 1 < cur.len() {
                    next.push(node_hash(&cur[i], &cur[i + 1]));
                } else {
                    // odd tail: duplicate the last node (documented limitation above)
                    next.push(node_hash(&cur[i], &cur[i]));
                }
                i += 2;
            }
            levels.push(next);
        }
        levels
    }

    /// Sibling path from leaf `idx` to the root: `(sibling_hash, sibling_is_right)`.
    pub fn proof_for(&self, idx: usize) -> Vec<([u8; 32], bool)> {
        let mut proof = Vec::new();
        let mut i = idx;
        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_idx = if i % 2 == 0 { i + 1 } else { i - 1 };
            let sibling = if sibling_idx < level.len() {
                level[sibling_idx]
            } else {
                level[i] // odd-tail duplication case
            };
            proof.push((sibling, i % 2 == 0));
            i /= 2;
        }
        proof
    }

    pub fn proof_bytes_for(&self, idx: usize) -> usize {
        // root (32) + per-level sibling hash (32 each)
        32 + self.proof_for(idx).len() * 32
    }

    pub fn verify_item(
        &self,
        idx: usize,
        qh: [u8; 32],
        index_root: [u8; 32],
        item: &ResultItem,
        root: [u8; 32],
        proof: &[([u8; 32], bool)],
    ) -> bool {
        if idx >= self.leaves.len() {
            return false;
        }
        let mut node = result_leaf(&qh, &index_root, item);
        for (sibling, sibling_is_right) in proof {
            node = if *sibling_is_right {
                node_hash(&node, sibling)
            } else {
                node_hash(sibling, &node)
            };
        }
        node == root
    }

    /// Empty result sets fail closed (same rationale as
    /// [`PerResultReceipt::verify_full`]): a zero-leaf receipt would
    /// otherwise verify vacuously.
    pub fn verify_full(&self, qh: [u8; 32], index_root: [u8; 32], results: &[ResultItem]) -> bool {
        if results.is_empty() || results.len() != self.leaves.len() {
            return false;
        }
        results.iter().enumerate().all(|(i, item)| {
            let proof = self.proof_for(i);
            self.verify_item(i, qh, index_root, item, self.root, &proof)
        })
    }

    pub fn total_bytes(&self) -> usize {
        32 + self.leaves.len() * 32
    }
}
