//! Robertson BM25 sparse inverted index.
//!
//! ## Formula
//!
//! BM25(q, d) = Σ_{t∈q} IDF(t) · TF_norm(t, d)
//!
//! IDF(t)         = ln((N − df(t) + 0.5) / (df(t) + 0.5) + 1)
//! TF_norm(t, d)  = tf · (k1 + 1) / (tf + k1 · (1 − b + b · |d| / avgdl))
//!
//! Parameters: k1 = 1.2, b = 0.75 (Robertson defaults).
//! IDF floor: +1 inside ln prevents negative IDF for very frequent terms.

use crate::{Document, SearchResult, SparseSearch};
use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

#[derive(Debug, Clone)]
struct Posting {
    doc_id: usize,
    tf: u32,
}

/// BM25 sparse index over tokenised document corpora.
///
/// Build once with [`Bm25Index::build`], then call [`SparseSearch::search`]
/// with query tokens.  The index stores one inverted list per unique term.
pub struct Bm25Index {
    inverted: HashMap<String, Vec<Posting>>,
    doc_lengths: Vec<u32>,
    avg_dl: f32,
    n_docs: usize,
}

impl Bm25Index {
    /// Build a BM25 index from a slice of [`Document`]s.
    ///
    /// Time: O(Σ|d|) — linear in total corpus token count.
    /// Memory: O(Σ|d|) — one posting per (term, document) pair.
    pub fn build(docs: &[Document]) -> Self {
        let n_docs = docs.len();
        let mut inverted: HashMap<String, Vec<Posting>> = HashMap::new();
        let mut doc_lengths = Vec::with_capacity(n_docs);
        let mut total_len: u64 = 0;

        for doc in docs {
            let dl = doc.tokens.len() as u32;
            doc_lengths.push(dl);
            total_len += dl as u64;

            let mut tf_map: HashMap<&str, u32> = HashMap::new();
            for token in &doc.tokens {
                *tf_map.entry(token.as_str()).or_insert(0) += 1;
            }
            for (term, tf) in tf_map {
                inverted
                    .entry(term.to_string())
                    .or_default()
                    .push(Posting { doc_id: doc.id, tf });
            }
        }

        let avg_dl = if n_docs > 0 {
            total_len as f32 / n_docs as f32
        } else {
            1.0
        };
        Self {
            inverted,
            doc_lengths,
            avg_dl,
            n_docs,
        }
    }

    /// Number of documents in this index.
    pub fn doc_count(&self) -> usize {
        self.n_docs
    }

    /// Estimated memory usage in bytes (postings only, excluding HashMap overhead).
    pub fn posting_bytes(&self) -> usize {
        self.inverted.values().map(|v| v.len() * 12).sum()
    }

    fn idf(&self, df: usize) -> f32 {
        let n = self.n_docs as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    fn tf_norm(&self, tf: u32, dl: u32) -> f32 {
        let tf = tf as f32;
        let dl = dl as f32;
        (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / self.avg_dl))
    }
}

impl SparseSearch for Bm25Index {
    fn search(&self, tokens: &[&str], k: usize) -> Vec<SearchResult> {
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for &token in tokens {
            if let Some(postings) = self.inverted.get(token) {
                let idf = self.idf(postings.len());
                for p in postings {
                    let dl = self.doc_lengths[p.doc_id];
                    let tf_n = self.tf_norm(p.tf, dl);
                    *scores.entry(p.doc_id).or_insert(0.0) += idf * tf_n;
                }
            }
        }

        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    fn make_doc(id: usize, tokens: &[&str]) -> Document {
        Document {
            id,
            tokens: tokens.iter().map(|s| s.to_string()).collect(),
            vector: vec![0.0; 4],
        }
    }

    #[test]
    fn test_bm25_exact_match() {
        let docs = vec![
            make_doc(0, &["rust", "vector", "search"]),
            make_doc(1, &["python", "machine", "learning"]),
            make_doc(2, &["rust", "memory", "safety"]),
        ];
        let index = Bm25Index::build(&docs);
        let results = index.search(&["rust"], 5);
        assert_eq!(results.len(), 2, "Only docs 0 and 2 contain 'rust'");
        let ids: Vec<usize> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&0) && ids.contains(&2));
    }

    #[test]
    fn test_bm25_no_match_returns_empty() {
        let docs = vec![make_doc(0, &["alpha", "beta"])];
        let index = Bm25Index::build(&docs);
        assert!(index.search(&["gamma"], 5).is_empty());
    }

    #[test]
    fn test_bm25_higher_tf_ranks_first() {
        let docs = vec![
            make_doc(0, &["rust", "rust", "rust"]),
            make_doc(1, &["rust", "slow"]),
        ];
        let index = Bm25Index::build(&docs);
        let results = index.search(&["rust"], 2);
        assert_eq!(results[0].id, 0, "Higher TF should rank first");
    }

    #[test]
    fn test_bm25_respects_k_limit() {
        let docs: Vec<Document> = (0..20).map(|i| make_doc(i, &["keyword"])).collect();
        let index = Bm25Index::build(&docs);
        assert_eq!(index.search(&["keyword"], 5).len(), 5);
    }

    #[test]
    fn test_bm25_scores_are_positive() {
        let docs = vec![
            make_doc(0, &["alpha", "beta", "gamma"]),
            make_doc(1, &["alpha", "delta"]),
        ];
        let index = Bm25Index::build(&docs);
        for r in index.search(&["alpha", "beta"], 5) {
            assert!(
                r.score > 0.0,
                "BM25 scores must be positive for matched terms"
            );
        }
    }

    #[test]
    fn test_posting_bytes_nonzero() {
        let docs = vec![make_doc(0, &["a", "b"]), make_doc(1, &["a", "c"])];
        let index = Bm25Index::build(&docs);
        assert!(index.posting_bytes() > 0);
    }
}
