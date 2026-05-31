// BM25-style sparse inverted index.
// Documents and queries provide pre-computed term weights (e.g., from TF-IDF,
// SPLADE, or any learned sparse encoder). The index stores posting lists and
// computes inner-product scoring at query time.

use crate::{HybridDoc, HybridQuery, Scored, SparseVec};
use std::collections::HashMap;

/// Inverted index: term_id → posting list (doc_id, weight).
/// Scoring: score(q, d) = Σ_{t ∈ q∩d} w_q(t) · w_d(t)
/// This is the "IMPACT" scoring model used by SPLADE and uniCOIL.
pub struct SparseIndex {
    /// posting_lists[term_id] = sorted (doc_id, weight) pairs.
    posting_lists: HashMap<u32, Vec<(u32, f32)>>,
    num_docs: usize,
}

impl SparseIndex {
    pub fn new() -> Self {
        Self {
            posting_lists: HashMap::new(),
            num_docs: 0,
        }
    }

    pub fn insert(&mut self, doc_id: u32, sparse: &SparseVec) {
        for &(term_id, weight) in &sparse.terms {
            self.posting_lists
                .entry(term_id)
                .or_default()
                .push((doc_id, weight));
        }
        self.num_docs += 1;
    }

    /// Insert a full HybridDoc (uses only sparse component).
    pub fn insert_doc(&mut self, doc: &HybridDoc) {
        self.insert(doc.id, &doc.sparse);
    }

    /// Max-WAND-style retrieval: traverse only matching posting lists.
    /// Returns top-k by inner product.
    pub fn search(&self, query: &HybridQuery, k: usize) -> Vec<Scored> {
        self.search_sparse(&query.sparse, k)
    }

    pub fn search_sparse(&self, query: &SparseVec, k: usize) -> Vec<Scored> {
        // Accumulate scores across matching posting lists.
        let mut acc: HashMap<u32, f32> = HashMap::new();

        for &(term_id, q_weight) in &query.terms {
            if let Some(postings) = self.posting_lists.get(&term_id) {
                for &(doc_id, d_weight) in postings {
                    *acc.entry(doc_id).or_insert(0.0) += q_weight * d_weight;
                }
            }
        }

        top_k_from_map(acc, k)
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// Approximate memory usage: 8 bytes per posting entry.
    pub fn memory_bytes(&self) -> usize {
        let total_postings: usize = self.posting_lists.values().map(|v| v.len()).sum();
        total_postings * 8
    }

    /// Index statistics.
    pub fn stats(&self) -> SparseStats {
        let num_terms = self.posting_lists.len();
        let total_postings: usize = self.posting_lists.values().map(|v| v.len()).sum();
        let avg_posting_len = if num_terms > 0 {
            total_postings as f64 / num_terms as f64
        } else {
            0.0
        };
        SparseStats {
            num_docs: self.num_docs,
            num_terms,
            total_postings,
            avg_posting_len,
        }
    }
}

#[derive(Debug)]
pub struct SparseStats {
    pub num_docs: usize,
    pub num_terms: usize,
    pub total_postings: usize,
    pub avg_posting_len: f64,
}

fn top_k_from_map(acc: HashMap<u32, f32>, k: usize) -> Vec<Scored> {
    let mut results: Vec<Scored> = acc
        .into_iter()
        .map(|(id, score)| Scored::new(id, score))
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(k);
    results
}

/// Generate a BM25-style sparse vector for a document given:
/// - `term_ids`: the terms present in the document
/// - `tf`: term frequencies
/// - `df`: document frequencies (how many docs contain the term)
/// - `n`: total number of documents
/// - Parameters k1 and b for BM25 (k1=1.5, b=0.75 are typical)
pub fn bm25_weights(
    term_ids: &[u32],
    tf: &[f32],
    df: &[u32],
    n: u32,
    doc_len: f32,
    avg_doc_len: f32,
    k1: f32,
    b: f32,
) -> SparseVec {
    let mut terms = Vec::with_capacity(term_ids.len());
    for i in 0..term_ids.len() {
        let idf = ((n as f32 - df[i] as f32 + 0.5) / (df[i] as f32 + 0.5) + 1.0).ln();
        let tf_norm = tf[i] * (k1 + 1.0) / (tf[i] + k1 * (1.0 - b + b * doc_len / avg_doc_len));
        let weight = (idf * tf_norm).max(0.0);
        if weight > 1e-6 {
            terms.push((term_ids[i], weight));
        }
    }
    SparseVec::new(terms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SparseVec;

    #[test]
    fn sparse_dot_product() {
        let a = SparseVec::new(vec![(1, 2.0), (3, 1.0)]);
        let b = SparseVec::new(vec![(1, 1.5), (2, 0.5), (3, 2.0)]);
        let dot = a.dot(&b);
        assert!((dot - (2.0 * 1.5 + 1.0 * 2.0)).abs() < 1e-5, "dot={}", dot);
    }

    #[test]
    fn sparse_index_exact_match() {
        let mut idx = SparseIndex::new();
        // Three docs: doc 0 shares term 1 and 2 with query, doc 1 shares only 1.
        let d0 = SparseVec::new(vec![(1, 1.0), (2, 1.0)]);
        let d1 = SparseVec::new(vec![(1, 1.0)]);
        let d2 = SparseVec::new(vec![(3, 5.0)]); // no overlap with query

        idx.insert(0, &d0);
        idx.insert(1, &d1);
        idx.insert(2, &d2);

        let query = SparseVec::new(vec![(1, 1.0), (2, 1.0)]);
        let results = idx.search_sparse(&query, 3);

        assert_eq!(results[0].id, 0, "doc 0 should rank first");
        assert_eq!(results[1].id, 1, "doc 1 should rank second");
        assert!(!results.iter().any(|r| r.id == 2), "doc 2 has no overlap");
    }

    #[test]
    fn bm25_weights_positive() {
        let term_ids = vec![1u32, 2, 3];
        let tf = vec![3.0f32, 1.0, 2.0];
        let df = vec![10u32, 500, 50];
        let n = 1000u32;
        let sv = bm25_weights(&term_ids, &tf, &df, n, 6.0, 5.0, 1.5, 0.75);
        for (_, w) in &sv.terms {
            assert!(*w > 0.0, "BM25 weight must be positive");
        }
        // Rare terms (df=10) should outweigh common terms (df=500)
        let w1 = sv
            .terms
            .iter()
            .find(|&&(t, _)| t == 1)
            .map(|&(_, w)| w)
            .unwrap_or(0.0);
        let w2 = sv
            .terms
            .iter()
            .find(|&&(t, _)| t == 2)
            .map(|&(_, w)| w)
            .unwrap_or(0.0);
        assert!(w1 > w2, "rare term should have higher IDF weight");
    }
}
