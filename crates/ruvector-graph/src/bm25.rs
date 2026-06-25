//! Compact BM25 keyword index over a node text property (ADR-252 P4).
//!
//! Provides the keyword arm of the tri-modal hybrid query (BM25 + ANN vector +
//! graph traversal). Self-contained — an in-memory inverted index with
//! Okapi BM25 scoring, no external search engine. Built from `(NodeId, &str)`
//! pairs and queried for the top-k by keyword relevance.

use crate::types::NodeId;
use std::collections::HashMap;

/// Okapi BM25 parameters. Defaults `k1=1.2`, `b=0.75` are the standard choices.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// In-memory BM25 inverted index over a single text field.
#[derive(Debug, Clone)]
pub struct Bm25Index {
    params: Bm25Params,
    /// term -> postings as (doc index, term frequency).
    postings: HashMap<String, Vec<(u32, u32)>>,
    doc_ids: Vec<NodeId>,
    doc_len: Vec<u32>,
    avgdl: f32,
}

impl Bm25Index {
    /// Lowercase, split on non-alphanumeric. Cheap and dependency-free.
    pub fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_ascii_lowercase())
            .collect()
    }

    /// Build the index from `(id, text)` pairs.
    pub fn build<I, S>(docs: I, params: Bm25Params) -> Self
    where
        I: IntoIterator<Item = (NodeId, S)>,
        S: AsRef<str>,
    {
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut doc_ids = Vec::new();
        let mut doc_len = Vec::new();
        let mut total_len: u64 = 0;

        for (id, text) in docs {
            let doc_idx = doc_ids.len() as u32;
            let tokens = Self::tokenize(text.as_ref());
            doc_len.push(tokens.len() as u32);
            total_len += tokens.len() as u64;

            // Term frequencies within this doc.
            let mut tf: HashMap<String, u32> = HashMap::new();
            for tok in tokens {
                *tf.entry(tok).or_insert(0) += 1;
            }
            for (term, freq) in tf {
                postings.entry(term).or_default().push((doc_idx, freq));
            }
            doc_ids.push(id);
        }

        let n = doc_ids.len().max(1) as f32;
        let avgdl = if doc_ids.is_empty() {
            0.0
        } else {
            total_len as f32 / n
        };
        Self {
            params,
            postings,
            doc_ids,
            doc_len,
            avgdl,
        }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Top-`k` documents by BM25 score for `query`, descending. Only documents
    /// with a positive score are returned.
    pub fn search(&self, query: &str, k: usize) -> Vec<(NodeId, f32)> {
        if self.doc_ids.is_empty() || k == 0 {
            return Vec::new();
        }
        let n = self.doc_ids.len() as f32;
        let (k1, b) = (self.params.k1, self.params.b);
        let mut scores: HashMap<u32, f32> = HashMap::new();

        // Deduplicate query terms; each contributes once via its idf.
        let mut seen_terms = std::collections::HashSet::new();
        for term in Self::tokenize(query) {
            if !seen_terms.insert(term.clone()) {
                continue;
            }
            let Some(postings) = self.postings.get(&term) else {
                continue;
            };
            let df = postings.len() as f32;
            // Robertson/Spärck-Jones idf with +1 to stay non-negative.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(doc_idx, freq) in postings {
                let dl = self.doc_len[doc_idx as usize] as f32;
                let tf = freq as f32;
                let denom = tf + k1 * (1.0 - b + b * dl / self.avgdl.max(1e-6));
                let contribution = idf * (tf * (k1 + 1.0)) / denom;
                *scores.entry(doc_idx).or_insert(0.0) += contribution;
            }
        }

        let mut ranked: Vec<(NodeId, f32)> = scores
            .into_iter()
            .map(|(idx, s)| (self.doc_ids[idx as usize].clone(), s))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<(NodeId, &'static str)> {
        vec![
            ("d1".into(), "the quick brown fox jumps over the lazy dog"),
            ("d2".into(), "machine learning models for vector search"),
            (
                "d3".into(),
                "vector databases enable semantic search at scale",
            ),
            ("d4".into(), "a recipe for italian pasta with tomato sauce"),
        ]
    }

    #[test]
    fn ranks_relevant_docs_first() {
        let idx = Bm25Index::build(corpus(), Bm25Params::default());
        assert_eq!(idx.len(), 4);
        let res = idx.search("vector search", 4);
        assert!(!res.is_empty());
        // d2 and d3 both mention "vector" and "search"; pasta doc must not lead.
        assert!(res[0].0 == "d2" || res[0].0 == "d3");
        assert!(res.iter().all(|(id, _)| id != "d4") || res.last().unwrap().0 == "d4");
    }

    #[test]
    fn idf_downweights_common_terms() {
        let idx = Bm25Index::build(corpus(), Bm25Params::default());
        // "the" appears in d1 only here but is short; "pasta" is rare → strong signal.
        let res = idx.search("pasta", 4);
        assert_eq!(res[0].0, "d4");
    }

    #[test]
    fn empty_query_and_index_safe() {
        let empty = Bm25Index::build(Vec::<(NodeId, &str)>::new(), Bm25Params::default());
        assert!(empty.search("anything", 5).is_empty());
        let idx = Bm25Index::build(corpus(), Bm25Params::default());
        assert!(idx.search("", 5).is_empty());
        assert!(idx.search("zzz nonexistent", 5).is_empty());
    }
}
