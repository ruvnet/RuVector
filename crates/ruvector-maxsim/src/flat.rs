//! Baseline: exhaustive (flat) MaxSim search.
//!
//! Scores every document with the MaxSim kernel. O(N · Td · Tq · D) per query.
//! Provides a recall-100% ground truth oracle and illustrates the throughput
//! ceiling an approximate index must beat.

use std::collections::BinaryHeap;

use crate::{
    error::MaxSimError,
    score::maxsim,
    types::{DocId, Embedding, MultiVecDoc, MultiVecQuery, SearchResult},
    MultiVecIndex,
};

/// Flat exhaustive MaxSim index — zero approximation, O(N) per query.
#[derive(Debug, Default)]
pub struct FlatMaxSim {
    docs: Vec<(DocId, Vec<Embedding>)>,
    dims: usize,
}

impl FlatMaxSim {
    /// Create an empty index.
    pub fn new(dims: usize) -> Self {
        Self {
            docs: Vec::new(),
            dims,
        }
    }

    /// Approximate memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.docs
            .iter()
            .map(|(_, vecs)| vecs.iter().map(|v| v.len() * 4).sum::<usize>())
            .sum()
    }
}

impl MultiVecIndex for FlatMaxSim {
    fn add(&mut self, doc: MultiVecDoc) -> Result<(), MaxSimError> {
        if let Some(v) = doc.vecs.first() {
            if v.len() != self.dims {
                return Err(MaxSimError::DimensionMismatch {
                    expected: self.dims,
                    got: v.len(),
                });
            }
        }
        self.docs.push((doc.id, doc.vecs));
        Ok(())
    }

    fn search(&self, query: &MultiVecQuery, k: usize) -> Result<Vec<SearchResult>, MaxSimError> {
        if self.docs.is_empty() {
            return Ok(Vec::new());
        }
        // Use a min-heap of size k to track the top-k docs.
        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k + 1);
        for (doc_id, doc_vecs) in &self.docs {
            let score = maxsim(&query.vecs, doc_vecs);
            heap.push(SearchResult {
                doc_id: *doc_id,
                score,
            });
            if heap.len() > k {
                heap.pop(); // removes the worst (lowest score) due to Ord impl
            }
        }
        // With reversed Ord, into_sorted_vec() returns descending by score (best first).
        Ok(heap.into_sorted_vec())
    }

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn dims(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: u64, vecs: Vec<Vec<f32>>) -> MultiVecDoc {
        MultiVecDoc {
            id: DocId(id),
            vecs,
        }
    }

    fn make_query(vecs: Vec<Vec<f32>>) -> MultiVecQuery {
        MultiVecQuery { vecs }
    }

    #[test]
    fn flat_single_exact_match() {
        let mut idx = FlatMaxSim::new(2);
        idx.add(make_doc(1, vec![vec![1.0, 0.0]])).unwrap();
        idx.add(make_doc(2, vec![vec![0.0, 1.0]])).unwrap();
        let q = make_query(vec![vec![1.0, 0.0]]);
        let res = idx.search(&q, 2).unwrap();
        assert_eq!(res[0].doc_id, DocId(1), "best match should be doc 1");
    }

    #[test]
    fn flat_top_k_ordering() {
        let mut idx = FlatMaxSim::new(3);
        // Doc scores by cosine: doc3 ≈ 1.0, doc2 ≈ 0.5, doc1 ≈ 0.0
        idx.add(make_doc(1, vec![vec![0.0, 1.0, 0.0]])).unwrap();
        idx.add(make_doc(2, vec![vec![0.5_f32.sqrt(), 0.5_f32.sqrt(), 0.0]]))
            .unwrap();
        idx.add(make_doc(3, vec![vec![1.0, 0.0, 0.0]])).unwrap();
        let q = make_query(vec![vec![1.0, 0.0, 0.0]]);
        let res = idx.search(&q, 3).unwrap();
        assert_eq!(res[0].doc_id, DocId(3));
        assert!(res[0].score > res[1].score);
        assert!(res[1].score > res[2].score);
    }

    #[test]
    fn flat_multi_token_recall() {
        // Doc covers two topics. Each query token should match its topic.
        let mut idx = FlatMaxSim::new(2);
        idx.add(make_doc(1, vec![vec![1.0, 0.0], vec![0.0, 1.0]]))
            .unwrap();
        idx.add(make_doc(2, vec![vec![1.0, 0.0]])).unwrap(); // only topic A
                                                             // Query about topic B only
        let q = make_query(vec![vec![0.0, 1.0]]);
        let res = idx.search(&q, 2).unwrap();
        // Doc 1 should rank first because it covers topic B
        assert_eq!(res[0].doc_id, DocId(1));
    }
}
