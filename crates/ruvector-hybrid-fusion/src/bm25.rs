//! BM25 Okapi sparse inverted index — the keyword leg of hybrid search.
//!
//! Formula:
//!   score(q, d) = Σ_{t∈q} IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
//!   IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

use std::collections::HashMap;

pub struct Bm25Index {
    /// term → [(doc_id, raw_tf)]
    inverted: HashMap<String, Vec<(usize, u32)>>,
    idf: HashMap<String, f32>,
    doc_lengths: Vec<u32>,
    avg_doc_len: f32,
    k1: f32,
    b: f32,
}

impl Bm25Index {
    /// Build from a slice of token lists (one per document).
    pub fn build(docs: &[Vec<String>]) -> Self {
        let n = docs.len();
        assert!(n > 0, "corpus must not be empty");

        let doc_lengths: Vec<u32> = docs.iter().map(|d| d.len() as u32).collect();
        let avg_doc_len = doc_lengths.iter().sum::<u32>() as f32 / n as f32;

        let mut inverted: HashMap<String, Vec<(usize, u32)>> = HashMap::new();

        for (doc_id, tokens) in docs.iter().enumerate() {
            let mut tf_map: HashMap<&str, u32> = HashMap::new();
            for token in tokens {
                *tf_map.entry(token.as_str()).or_insert(0) += 1;
            }
            for (term, tf) in tf_map {
                inverted.entry(term.to_string()).or_default().push((doc_id, tf));
            }
        }

        let mut idf: HashMap<String, f32> = HashMap::with_capacity(inverted.len());
        for (term, postings) in &inverted {
            let df = postings.len() as f32;
            // Robertson-Sparck Jones IDF (BM25 standard)
            let val = ((n as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            idf.insert(term.clone(), val.max(0.0));
        }

        Bm25Index {
            inverted,
            idf,
            doc_lengths,
            avg_doc_len,
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Return top-k (doc_id, bm25_score) for the given query tokens.
    pub fn score(&self, query_tokens: &[String], top_k: usize) -> Vec<(usize, f32)> {
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for token in query_tokens {
            let Some(postings) = self.inverted.get(token) else { continue };
            let idf = *self.idf.get(token).unwrap_or(&0.0);
            if idf < 1e-9 { continue; }

            for &(doc_id, tf) in postings {
                let dl = self.doc_lengths[doc_id] as f32;
                let tf_norm = tf as f32 * (self.k1 + 1.0)
                    / (tf as f32
                        + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_len));
                *scores.entry(doc_id).or_insert(0.0) += idf * tf_norm;
            }
        }

        let mut result: Vec<(usize, f32)> = scores.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(top_k);
        result
    }

    /// Number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.doc_lengths.len()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inverted.len()
    }

    /// Approximate memory usage in bytes (index-only, not docs).
    pub fn memory_bytes(&self) -> usize {
        // Each posting: 2 × usize + u32 ≈ 20 bytes; term string ≈ 12 bytes avg
        let postings_bytes: usize = self.inverted.values().map(|v| v.len() * 20).sum();
        let vocab_bytes = self.inverted.len() * (12 + 8); // key + idf entry
        let doc_len_bytes = self.doc_lengths.len() * 4;
        postings_bytes + vocab_bytes + doc_len_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_docs() -> Vec<Vec<String>> {
        vec![
            vec!["rust", "vector", "search", "fast"].iter().map(|s| s.to_string()).collect(),
            vec!["vector", "database", "ann", "hnsw"].iter().map(|s| s.to_string()).collect(),
            vec!["bm25", "keyword", "search", "sparse"].iter().map(|s| s.to_string()).collect(),
            vec!["rust", "bm25", "hybrid", "search"].iter().map(|s| s.to_string()).collect(),
        ]
    }

    #[test]
    fn build_and_score_basic() {
        let docs = make_docs();
        let idx = Bm25Index::build(&docs);
        assert_eq!(idx.doc_count(), 4);
        assert!(idx.vocab_size() > 0);

        let results = idx.score(&["rust".to_string(), "vector".to_string()], 4);
        assert!(!results.is_empty());
        // Doc 0 has both "rust" and "vector" → should rank high
        let top_id = results[0].0;
        assert!(top_id == 0 || top_id == 1, "expected doc 0 or 1 at top, got {}", top_id);
    }

    #[test]
    fn score_returns_at_most_top_k() {
        let docs = make_docs();
        let idx = Bm25Index::build(&docs);
        let results = idx.score(&["search".to_string()], 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn unknown_term_returns_empty() {
        let docs = make_docs();
        let idx = Bm25Index::build(&docs);
        let results = idx.score(&["zzznomatch".to_string()], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn scores_are_nonnegative() {
        let docs = make_docs();
        let idx = Bm25Index::build(&docs);
        for &(_, score) in &idx.score(&["rust".to_string()], 4) {
            assert!(score >= 0.0);
        }
    }

    #[test]
    fn memory_estimate_positive() {
        let docs = make_docs();
        let idx = Bm25Index::build(&docs);
        assert!(idx.memory_bytes() > 0);
    }
}
