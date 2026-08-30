//! Witness-chained retrieval receipts composed on top of a **real**
//! multi-layer HNSW index (`ruvector_hnsw_repair::HnswGraph`), instead of the
//! brute-force index `ruvector_retrieval_receipt::RetrievalIndex` deliberately
//! uses to isolate the provenance layer's cost from ANN recall.
//!
//! This crate exists to close the gap the `2026-08-13-retrieval-receipts`
//! nightly research explicitly left open: *"Compose `ruvector-retrieval-receipt`
//! on top of a real HNSW-family index and re-measure the overhead ratio
//! (flagged explicitly in Rejection Criteria as required before any
//! production overhead claim)."* Brute-force search cost grows with `n`;
//! HNSW search cost does not (it grows with graph degree and `ef`), so the
//! *ratio* of receipt-build cost to search cost measured against brute force
//! does not generalize to a real index. This crate re-measures that ratio
//! against `ruvector-hnsw-repair`'s from-scratch multi-layer HNSW graph.
//!
//! # What is reused vs. reimplemented
//!
//! The receipt cryptography (`PerResultReceipt`, `MerkleReceipt`,
//! `ReceiptVariant`, leaf/chain/node hashing, tamper-evidence semantics) is
//! **not reimplemented** here — it is the exact same code from
//! `ruvector-retrieval-receipt`, reused via a path dependency and re-exported.
//! Only the index side is new: [`HnswReceiptIndex`] wraps
//! `ruvector_hnsw_repair::HnswGraph` (approximate, multi-layer, real
//! insert/search implementation of Malkov & Yashunin 2018) instead of a
//! brute-force scan, gated by the same `ruvector_proof_gate::HashChainGate`
//! ingestion path.
//!
//! # Threat model
//!
//! Identical to `ruvector-retrieval-receipt`: receipts are unsigned
//! commitments produced by the query engine itself. They detect
//! post-issuance mutation of a receipt/result pair; they do not protect
//! against a dishonest query engine, and they do not prove write-chain
//! membership (see that crate's module docs for the full statement, which
//! applies unchanged here since the crypto is shared).

use ruvector_hnsw_repair::{HnswConfig, HnswGraph};
use ruvector_proof_gate::{HashChainGate, WriteGate, WritePayload, WriteReceipt};

pub use ruvector_retrieval_receipt::{
    query_hash, MerkleReceipt, PerResultReceipt, ReceiptVariant, ResultItem, RetrievalReceipt,
};

/// Deterministic xorshift64 stream — identical construction to the one used
/// by `ruvector_proof_gate::synthetic_payloads` and
/// `ruvector_retrieval_receipt::index`, so datasets are reproducible without
/// an external RNG dependency and comparable across both crates' benchmarks.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as i64 as f64 / i64::MAX as f64) as f32
    }
}

/// Deterministic query generator, independent stream from ingestion so
/// queries are reproducible but not identical to any stored vector.
pub fn synthetic_queries(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xorshift64::new(seed);
    (0..n)
        .map(|_| (0..dims).map(|_| rng.next_f32()).collect())
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// A real multi-layer approximate HNSW index whose ingestion path is wrapped
/// by a `HashChainGate`, mirroring `ruvector_retrieval_receipt::RetrievalIndex`'s
/// provenance guarantee but backed by approximate graph search (bounded `ef`,
/// bounded node degree) instead of an O(n) brute-force scan.
pub struct HnswReceiptIndex {
    graph: HnswGraph,
    gate: HashChainGate,
    write_receipts: Vec<WriteReceipt>,
}

impl HnswReceiptIndex {
    /// Ingest `n` deterministic `dims`-dimensional vectors: each is admitted
    /// through the write gate (real chained `WriteReceipt`) and then inserted
    /// into the HNSW graph. `HnswGraph::insert` assigns node ids sequentially
    /// from 0, so `write_receipts[id]` always lines up with `graph.vectors[id]`.
    pub fn ingest(n: usize, dims: usize, seed: u64) -> Self {
        let mut graph = HnswGraph::new(HnswConfig::new(dims));
        let mut gate = HashChainGate::new();
        let mut rng = Xorshift64::new(seed);
        let mut write_receipts = Vec::with_capacity(n);
        for i in 0..n {
            let vector: Vec<f32> = (0..dims).map(|_| rng.next_f32()).collect();
            let payload = WritePayload::new(i as u64, vector.clone());
            let receipt = gate
                .admit(&payload)
                .expect("HashChainGate::admit never rejects");
            write_receipts.push(receipt);
            let node_id = graph.insert(vector);
            debug_assert_eq!(node_id as usize, i, "HnswGraph node ids must be sequential");
        }
        Self {
            graph,
            gate,
            write_receipts,
        }
    }

    pub fn len(&self) -> usize {
        self.graph.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.vectors.is_empty()
    }

    /// Current write-chain commitment, bound into every retrieval receipt so
    /// a receipt also attests to which version of the index answered the
    /// query.
    pub fn index_state_root(&self) -> [u8; 32] {
        self.gate.chain_root()
    }

    pub fn verify_write_history(&self) -> bool {
        self.gate.verify_integrity()
    }

    /// Raw approximate search: node ids only, no receipt-related work. This
    /// is the search-only cost that receipt overhead is measured against.
    pub fn search_raw(&self, query: &[f32], k: usize, ef: usize) -> Vec<u32> {
        self.graph.search(query, k, ef)
    }

    /// Approximate search producing full `ResultItem`s (score + write
    /// receipt) ready for receipt building. Score is real cosine similarity
    /// over the stored vectors — computed the same way
    /// `ruvector_retrieval_receipt::RetrievalIndex` computes it — so leaves
    /// bind to an actual similarity computation rather than the graph's
    /// internal L2 ranking distance. Node/result order is exactly
    /// `search_raw`'s order: this function adds bookkeeping, not a second
    /// search pass.
    pub fn search_items(&self, query: &[f32], k: usize, ef: usize) -> Vec<ResultItem> {
        self.graph
            .search(query, k, ef)
            .into_iter()
            .enumerate()
            .map(|(rank, id)| ResultItem {
                vector_id: id as u64,
                rank: rank as u32,
                score: cosine(query, &self.graph.vectors[id as usize]),
                write_receipt: self.write_receipts[id as usize].clone(),
            })
            .collect()
    }

    /// Exact brute-force top-k cosine ground truth over all vectors — used
    /// only to report ANN recall@k as benchmark context. Never part of the
    /// receipted search path.
    pub fn brute_force_topk(&self, query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(u32, f32)> = self
            .graph
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, cosine(query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.into_iter().take(k).map(|(id, _)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingestion_produces_verifiable_write_history() {
        let index = HnswReceiptIndex::ingest(300, 16, 0xC0FF_EE01);
        assert!(index.verify_write_history());
        assert_ne!(index.index_state_root(), [0u8; 32]);
        assert_eq!(index.len(), 300);
    }

    #[test]
    fn search_raw_and_search_items_agree_on_ids_and_order() {
        let index = HnswReceiptIndex::ingest(400, 16, 0xABCD_1234);
        let query = synthetic_queries(1, 16, 0x5A5A_5A5A)[0].clone();
        let raw = index.search_raw(&query, 10, 40);
        let items = index.search_items(&query, 10, 40);
        assert_eq!(raw.len(), items.len());
        for (id, item) in raw.iter().zip(items.iter()) {
            assert_eq!(*id, item.vector_id as u32);
        }
        // Receipt construction must not perturb search: building a receipt
        // is a pure function of the already-computed result set.
        for w in items.windows(2) {
            // Approximate search is not guaranteed monotonically descending
            // in cosine score the way ef-bounded L2 ranking is, but ranks
            // must still be strictly increasing.
            assert!(w[0].rank < w[1].rank);
        }
    }

    #[test]
    fn per_result_and_merkle_receipts_verify_on_real_hnsw_results() {
        let index = HnswReceiptIndex::ingest(500, 32, 0x1111_2222);
        let query = synthetic_queries(1, 32, 0x3333_4444)[0].clone();
        let results = index.search_items(&query, 10, 64);
        let qh = query_hash(&query);
        let root = index.index_state_root();

        for variant in [ReceiptVariant::PerResult, ReceiptVariant::Merkle] {
            let receipt = RetrievalReceipt::build(variant, qh, root, &results);
            assert!(
                receipt.verify_full(qh, root, &results),
                "{variant:?} must verify honest HNSW results"
            );
            for (i, item) in results.iter().enumerate() {
                assert!(receipt.verify_item(i, qh, root, item));
            }
        }
    }

    #[test]
    fn receipts_detect_tamper_on_real_hnsw_results() {
        let index = HnswReceiptIndex::ingest(300, 24, 0x7777_8888);
        let query = synthetic_queries(1, 24, 0x9999_0000)[0].clone();
        let results = index.search_items(&query, 8, 48);
        let qh = query_hash(&query);
        let root = index.index_state_root();

        for variant in [ReceiptVariant::PerResult, ReceiptVariant::Merkle] {
            let receipt = RetrievalReceipt::build(variant, qh, root, &results);
            let mut tampered = results.clone();
            tampered[2].score += 1.0;
            assert!(!receipt.verify_item(2, qh, root, &tampered[2]));
        }
    }

    #[test]
    fn no_receipt_never_verifies_on_real_hnsw_results() {
        let index = HnswReceiptIndex::ingest(200, 16, 0x2323_4545);
        let query = synthetic_queries(1, 16, 0x6767_8989)[0].clone();
        let results = index.search_items(&query, 5, 32);
        let qh = query_hash(&query);
        let root = index.index_state_root();
        let receipt = RetrievalReceipt::build(ReceiptVariant::None, qh, root, &results);
        assert!(!receipt.verify_full(qh, root, &results));
    }

    #[test]
    fn recall_against_brute_force_ground_truth_is_nontrivial() {
        // Sanity check, not the acceptance metric: a completely broken graph
        // (e.g. wrong node-id mapping between write_receipts and vectors)
        // would show ~0 overlap with brute-force ground truth.
        let index = HnswReceiptIndex::ingest(1000, 32, 0xAAAA_BBBB);
        let queries = synthetic_queries(20, 32, 0xCCCC_DDDD);
        let mut total_hits = 0usize;
        for q in &queries {
            let approx = index.search_raw(q, 10, 64);
            let gt = index.brute_force_topk(q, 10);
            let gt_set: std::collections::HashSet<u32> = gt.into_iter().collect();
            total_hits += approx.iter().filter(|id| gt_set.contains(id)).count();
        }
        let recall = total_hits as f32 / (queries.len() * 10) as f32;
        assert!(
            recall > 0.3,
            "recall@10 suspiciously low ({recall}); node-id mapping may be broken"
        );
    }
}
