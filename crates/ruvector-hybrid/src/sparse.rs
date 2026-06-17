//! BM25 sparse inverted-index retrieval.
//!
//! Implements the Robertson-Zaragoza BM25F formula with standard parameters:
//! - k1 = 1.2  (term-frequency saturation)
//! - b  = 0.75 (document-length normalisation)
//!
//! Tokens are represented as `u32` IDs — the caller maps raw text to IDs.
//! The index stores one posting list per token, so memory is proportional to
//! the number of (doc, token) co-occurrences.

use crate::{HybridSearch, SearchResult};
use std::collections::HashMap;

/// BM25 k1 parameter (term saturation).
const K1: f32 = 1.2;
/// BM25 b parameter (length normalisation).
const B: f32 = 0.75;

#[derive(Debug)]
struct Posting {
    doc_id: u32,
    tf: f32,
}

#[derive(Debug)]
struct DocMeta {
    len: u32,
}

/// BM25 inverted index over `u32` token IDs.
pub struct SparseBm25 {
    index: HashMap<u32, Vec<Posting>>,
    docs: HashMap<u32, DocMeta>,
    total_len: u64,
}

impl SparseBm25 {
    /// Create an empty BM25 index.
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            docs: HashMap::new(),
            total_len: 0,
        }
    }

    fn n_docs(&self) -> usize {
        self.docs.len()
    }

    fn avgdl(&self) -> f32 {
        let n = self.n_docs();
        if n == 0 {
            1.0
        } else {
            self.total_len as f32 / n as f32
        }
    }

    /// Robertson-Zaragoza smooth IDF.
    fn idf(&self, df: usize) -> f32 {
        let n = self.n_docs() as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0)
    }

    fn bm25_score(&self, tf: f32, doc_len: u32, df: usize) -> f32 {
        let avgdl = self.avgdl();
        let idf = self.idf(df);
        let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * doc_len as f32 / avgdl));
        idf * tf_norm
    }
}

impl Default for SparseBm25 {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridSearch for SparseBm25 {
    fn insert(&mut self, id: u32, _vector: &[f32], tokens: &[u32]) {
        let mut tf_map: HashMap<u32, u32> = HashMap::new();
        for &t in tokens {
            *tf_map.entry(t).or_insert(0) += 1;
        }
        let doc_len = tokens.len() as u32;
        self.docs.insert(id, DocMeta { len: doc_len });
        self.total_len += doc_len as u64;
        for (token, freq) in tf_map {
            self.index.entry(token).or_default().push(Posting {
                doc_id: id,
                tf: freq as f32,
            });
        }
    }

    fn search(&self, _query_vec: &[f32], query_tokens: &[u32], k: usize) -> Vec<SearchResult> {
        let mut acc: HashMap<u32, f32> = HashMap::new();
        // deduplicate query tokens before scoring
        let mut unique: Vec<u32> = query_tokens.to_vec();
        unique.sort_unstable();
        unique.dedup();

        for &token in &unique {
            if let Some(postings) = self.index.get(&token) {
                let df = postings.len();
                for p in postings {
                    if let Some(meta) = self.docs.get(&p.doc_id) {
                        let s = self.bm25_score(p.tf, meta.len, df);
                        *acc.entry(p.doc_id).or_insert(0.0) += s;
                    }
                }
            }
        }

        let mut results: Vec<SearchResult> = acc
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

    fn len(&self) -> usize {
        self.n_docs()
    }

    fn memory_bytes(&self) -> usize {
        let postings_bytes: usize = self.index.values().map(|v| v.len() * 8).sum();
        let docs_bytes = self.docs.len() * 8;
        postings_bytes + docs_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_idx() -> SparseBm25 {
        let mut idx = SparseBm25::new();
        // doc 0: tokens [1, 1, 2]  — token 1 appears twice
        idx.insert(0, &[], &[1, 1, 2]);
        // doc 1: tokens [2, 3]
        idx.insert(1, &[], &[2, 3]);
        // doc 2: tokens [4, 5, 6]  — shares nothing with queries on 1/2/3
        idx.insert(2, &[], &[4, 5, 6]);
        idx
    }

    #[test]
    fn returns_matching_docs_only() {
        let idx = make_idx();
        // query on token 3 — only doc 1 matches
        let res = idx.search(&[], &[3], 10);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, 1);
    }

    #[test]
    fn higher_tf_ranks_first() {
        let idx = make_idx();
        // token 1 appears twice in doc 0, once in neither
        // doc 1 has token 1 zero times, doc 0 has token 1 twice
        let res = idx.search(&[], &[1], 10);
        assert_eq!(res[0].id, 0, "doc with higher TF should rank first");
    }

    #[test]
    fn no_results_for_unknown_token() {
        let idx = make_idx();
        let res = idx.search(&[], &[99], 10);
        assert!(res.is_empty());
    }

    #[test]
    fn memory_bytes_grows_with_docs() {
        let mut idx = SparseBm25::new();
        idx.insert(0, &[], &[1, 2, 3]);
        let m1 = idx.memory_bytes();
        idx.insert(1, &[], &[4, 5, 6]);
        let m2 = idx.memory_bytes();
        assert!(m2 > m1);
    }
}
