//! Witness-chained provenance receipts for ANN retrieval results.
//!
//! `ruvector-proof-gate` answers "what was written, and can I prove it
//! hasn't been tampered with?" for the *write* path. This crate addresses
//! the read path: it commits a query's result set to a receipt so that a
//! receipt/result pair, once issued, cannot be silently mutated in transit
//! or in storage without verification failing.
//!
//! # Threat model — read this before relying on the receipts
//!
//! Receipts are **unsigned commitments produced by the query engine
//! itself**. What they do and do not guarantee:
//!
//! - They **detect post-issuance mutation** of a receipt/result pair. That
//!   is the guarantee: an auditor holding the receipt can tell whether the
//!   result set they were handed is the one the engine committed to.
//! - They do **not** protect against a dishonest query engine. Leaves are
//!   engine-chosen; nothing binds a leaf's `score` to an actual cosine
//!   computation, or the committed k-set to the true top-k.
//! - They do **not** prove write-chain membership. Each leaf commits to
//!   *copies* of the `WriteReceipt`'s fields; verification never consults
//!   the write gate, so mutating the ingestion history after a receipt is
//!   issued leaves that receipt verifying. Binding each result to an
//!   offline membership proof via `ruvector_proof_gate::MerkleGate`'s MMR
//!   is the named future-work item that would make the write→read link
//!   real.
//!
//! # Variants
//!
//! | Variant         | Per-query build cost | Evidence to verify 1 of k results | Guarantee |
//! |-----------------|----------------------|-------------------------------------|-----------|
//! | `NoReceipt`      | 0                    | none (unverifiable)                 | none |
//! | `PerResultReceipt` | O(k) hashes        | O(idx) bytes, O(idx) work            | sequential tamper-evidence |
//! | `MerkleReceipt`  | O(k) hashes          | O(log k) bytes, O(log k) work        | membership-proof tamper-evidence |
//!
//! # Signed anchoring (origin authentication)
//!
//! The variants above are all *unsigned*: they detect tamper but do not
//! authenticate an issuing key. [`signing`] adds typed, scoped Ed25519
//! anchoring on top of `PerResultReceipt`/`MerkleReceipt` roots, either
//! per query ([`signing::Issuer::sign_root`]) or amortized across a batch
//! of queries under one signature ([`signing::BatchAnchor`]). Binding a
//! key to an organization remains the responsibility of an external key
//! registry and revocation policy. See [`RetrievalReceipt::root`] and the
//! `signing` module docs.
//!
//! # Independent state-root anchoring
//!
//! Signed receipt roots (above) authenticate what a *specific query*
//! returned. [`state_anchor`] answers a decoupled question: has
//! `index_state_root` itself ever been attested, independent of any query
//! or receipt? See the module docs for the periodic-anchoring tradeoff this
//! makes measurable.

mod index;
mod receipt;
pub mod signing;
pub mod state_anchor;

pub use index::{synthetic_queries, ResultItem, RetrievalIndex};
pub use receipt::{query_hash, MerkleReceipt, PerResultReceipt, ReceiptVariant};
pub use signing::{
    verify_root, AnchorContext, AnchorError, AnchorPurpose, BatchAnchor, Issuer, RootStatement,
    SignedRoot, VerifiedRoot, SIGNED_ROOT_VERSION,
};
pub use state_anchor::{verify_state_anchor, StateAnchor, StateAnchorLog, StateAnchorPolicy};

/// A built receipt for one query's result set, in whichever variant was
/// requested. Carries enough state to answer `proof_bytes_for` /
/// `verify_item` / `verify_full` uniformly across variants for benchmarking.
pub enum RetrievalReceipt {
    None,
    PerResult(PerResultReceipt),
    Merkle(MerkleReceipt),
}

impl RetrievalReceipt {
    pub fn build(
        variant: ReceiptVariant,
        qh: [u8; 32],
        index_root: [u8; 32],
        results: &[ResultItem],
    ) -> Self {
        match variant {
            ReceiptVariant::None => RetrievalReceipt::None,
            ReceiptVariant::PerResult => {
                RetrievalReceipt::PerResult(PerResultReceipt::build(qh, index_root, results))
            }
            ReceiptVariant::Merkle => {
                RetrievalReceipt::Merkle(MerkleReceipt::build(qh, index_root, results))
            }
        }
    }

    pub fn variant(&self) -> ReceiptVariant {
        match self {
            RetrievalReceipt::None => ReceiptVariant::None,
            RetrievalReceipt::PerResult(_) => ReceiptVariant::PerResult,
            RetrievalReceipt::Merkle(_) => ReceiptVariant::Merkle,
        }
    }

    /// The signable root for this receipt: `PerResult`'s chain head or
    /// `Merkle`'s root. `None` has no root to sign — nothing was
    /// committed, so nothing can be authenticated by a signature.
    pub fn root(&self) -> Option<[u8; 32]> {
        match self {
            RetrievalReceipt::None => None,
            RetrievalReceipt::PerResult(r) => Some(r.chain_head),
            RetrievalReceipt::Merkle(r) => Some(r.root),
        }
    }

    /// Bytes of evidence required to verify a single result at `idx`
    /// without trusting a live index/gate instance. `None` returns `0` —
    /// by construction there is nothing to verify, which is the point.
    pub fn proof_bytes_for(&self, idx: usize) -> usize {
        match self {
            RetrievalReceipt::None => 0,
            RetrievalReceipt::PerResult(r) => r.proof_bytes_for(idx),
            RetrievalReceipt::Merkle(r) => r.proof_bytes_for(idx),
        }
    }

    /// Total receipt payload size (all results), independent of which item
    /// is later spot-checked.
    pub fn total_bytes(&self) -> usize {
        match self {
            RetrievalReceipt::None => 0,
            RetrievalReceipt::PerResult(r) => r.total_bytes(),
            RetrievalReceipt::Merkle(r) => r.total_bytes(),
        }
    }

    /// Verify a single result item. `NoReceipt` always returns `false`:
    /// there is no evidence to check, which is intentional and distinct
    /// from "verification passed."
    pub fn verify_item(
        &self,
        idx: usize,
        qh: [u8; 32],
        index_root: [u8; 32],
        item: &ResultItem,
    ) -> bool {
        match self {
            RetrievalReceipt::None => false,
            RetrievalReceipt::PerResult(r) => r.verify_item(idx, qh, index_root, item),
            RetrievalReceipt::Merkle(r) => {
                let proof = r.proof_for(idx);
                r.verify_item(idx, qh, index_root, item, r.root, &proof)
            }
        }
    }

    /// Verify a complete result set against the receipt. Empty result sets
    /// fail closed for every variant: "no evidence" must never be
    /// reportable as "verified evidence".
    pub fn verify_full(&self, qh: [u8; 32], index_root: [u8; 32], results: &[ResultItem]) -> bool {
        match self {
            RetrievalReceipt::None => false,
            RetrievalReceipt::PerResult(r) => r.verify_full(qh, index_root, results),
            RetrievalReceipt::Merkle(r) => r.verify_full(qh, index_root, results),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(
        n: usize,
        dims: usize,
        k: usize,
    ) -> (
        RetrievalIndex,
        Vec<f32>,
        Vec<ResultItem>,
        [u8; 32],
        [u8; 32],
    ) {
        let index = RetrievalIndex::ingest(n, dims, 0xC0FF_EE01);
        let query = synthetic_queries(1, dims, 0xA5A5_5A5A)[0].clone();
        let results = index.search(&query, k);
        let qh = query_hash(&query);
        let root = index.index_state_root();
        (index, query, results, qh, root)
    }

    #[test]
    fn ingestion_produces_verifiable_write_history() {
        let (index, _, _, _, _) = setup(200, 16, 5);
        assert!(index.verify_write_history());
        assert_ne!(index.index_state_root(), [0u8; 32]);
    }

    #[test]
    fn search_returns_k_results_in_descending_score_order() {
        let (_, _, results, _, _) = setup(200, 16, 5);
        assert_eq!(results.len(), 5);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn no_receipt_never_verifies() {
        let (_, _, results, qh, root) = setup(100, 16, 4);
        let receipt = RetrievalReceipt::build(ReceiptVariant::None, qh, root, &results);
        assert_eq!(receipt.proof_bytes_for(0), 0);
        assert!(!receipt.verify_item(0, qh, root, &results[0]));
        assert!(!receipt.verify_full(qh, root, &results));
    }

    #[test]
    fn per_result_receipt_verifies_honest_results() {
        let (_, _, results, qh, root) = setup(500, 32, 10);
        let receipt = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &results);
        assert!(receipt.verify_full(qh, root, &results));
        for (i, item) in results.iter().enumerate() {
            assert!(receipt.verify_item(i, qh, root, item));
        }
    }

    #[test]
    fn merkle_receipt_verifies_honest_results() {
        let (_, _, results, qh, root) = setup(500, 32, 10);
        let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        assert!(receipt.verify_full(qh, root, &results));
        for (i, item) in results.iter().enumerate() {
            assert!(receipt.verify_item(i, qh, root, item));
        }
    }

    #[test]
    fn per_result_receipt_detects_score_tamper() {
        let (_, _, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &results);
        let mut tampered = results.clone();
        tampered[3].score += 1.0;
        assert!(!receipt.verify_item(3, qh, root, &tampered[3]));
    }

    #[test]
    fn merkle_receipt_detects_score_tamper() {
        let (_, _, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        let mut tampered = results.clone();
        tampered[3].score += 1.0;
        assert!(!receipt.verify_item(3, qh, root, &tampered[3]));
    }

    #[test]
    fn per_result_receipt_detects_reorder() {
        let (_, _, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &results);
        let mut tampered = results.clone();
        tampered.swap(2, 5);
        assert!(!receipt.verify_full(qh, root, &tampered));
    }

    #[test]
    fn merkle_receipt_detects_reorder() {
        let (_, _, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        let mut tampered = results.clone();
        tampered.swap(2, 5);
        assert!(!receipt.verify_full(qh, root, &tampered));
    }

    #[test]
    fn merkle_receipt_detects_vector_id_substitution() {
        let (index, query, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        // Substitute a truthful-looking but different vector's write receipt
        // in place of result 0 — simulates an adversary swapping in evidence
        // for a different, unretrieved memory.
        let other = index.search(&query, 9)[8].clone();
        let mut tampered = results.clone();
        tampered[0] = other;
        assert!(!receipt.verify_item(0, qh, root, &tampered[0]));
    }

    #[test]
    fn merkle_proof_bytes_are_sublinear_vs_per_result_at_k10() {
        let (_, _, results, qh, root) = setup(2000, 64, 10);
        let per_result = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &results);
        let merkle = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        let worst_idx = results.len() - 1;
        assert!(
            merkle.proof_bytes_for(worst_idx) < per_result.proof_bytes_for(worst_idx),
            "merkle worst-case proof ({} bytes) must be smaller than per-result worst-case ({} bytes)",
            merkle.proof_bytes_for(worst_idx),
            per_result.proof_bytes_for(worst_idx)
        );
    }

    #[test]
    fn index_state_root_changes_receipts_across_reingestion() {
        let index_a = RetrievalIndex::ingest(50, 8, 1);
        let index_b = RetrievalIndex::ingest(50, 8, 2);
        assert_ne!(index_a.index_state_root(), index_b.index_state_root());
    }

    #[test]
    fn empty_result_set_fails_closed_for_all_variants() {
        let (_, _, _, qh, root) = setup(50, 8, 3);
        let empty: Vec<ResultItem> = Vec::new();
        for variant in [
            ReceiptVariant::None,
            ReceiptVariant::PerResult,
            ReceiptVariant::Merkle,
        ] {
            let receipt = RetrievalReceipt::build(variant, qh, root, &empty);
            assert!(
                !receipt.verify_full(qh, root, &empty),
                "{variant:?}: an empty result set must not verify vacuously"
            );
        }
    }

    #[test]
    fn gate_variant_is_bound_into_the_leaf() {
        use ruvector_proof_gate::GateVariant;
        let (_, _, results, qh, root) = setup(100, 16, 4);
        for variant in [ReceiptVariant::PerResult, ReceiptVariant::Merkle] {
            let receipt = RetrievalReceipt::build(variant, qh, root, &results);
            let mut tampered = results.clone();
            // Same (copied) commitment/payload hashes, but claim they came
            // from a NullGate: must not verify — otherwise an ungated
            // all-zero receipt would be indistinguishable from a gated one.
            tampered[0].write_receipt.gate_variant = GateVariant::Null;
            assert!(!receipt.verify_item(0, qh, root, &tampered[0]));
        }
    }
}
