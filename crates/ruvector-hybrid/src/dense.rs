//! Dense flat-scan cosine-similarity index.
//!
//! Provides an exact brute-force search over stored unit-normalised vectors.
//! O(N·D) per query, O(N·D·4) bytes memory.  Used as the dense leg of
//! [`crate::fusion::HybridRrf`] and as the standalone baseline.
//!
//! **Pre-condition**: callers must insert and query with unit-normalised vectors.
//! Re-normalising internally would create floating-point discrepancies between
//! the stored vectors and the ground-truth vectors used for recall evaluation,
//! causing spurious rank flips at the top-K boundary.

use crate::{HybridSearch, SearchResult};

struct DenseDoc {
    id: u32,
    vector: Vec<f32>,
}

/// Exact cosine flat scan — 100% recall, O(N·D) query time.
///
/// Vectors must be pre-normalised by the caller; the index stores them as-is
/// so that dot-product scores are numerically identical to any external ground-
/// truth computation over the same vectors.
pub struct DenseFlat {
    docs: Vec<DenseDoc>,
    dim: usize,
}

impl DenseFlat {
    /// Create an empty index for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            docs: Vec::new(),
            dim,
        }
    }

    #[inline]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

impl HybridSearch for DenseFlat {
    fn insert(&mut self, id: u32, vector: &[f32], _tokens: &[u32]) {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");
        self.docs.push(DenseDoc {
            id,
            vector: vector.to_vec(),
        });
    }

    fn search(&self, query_vec: &[f32], _query_tokens: &[u32], k: usize) -> Vec<SearchResult> {
        assert_eq!(query_vec.len(), self.dim, "query dimension mismatch");
        let mut scores: Vec<(f32, u32)> = self
            .docs
            .iter()
            .map(|d| (Self::dot(query_vec, &d.vector), d.id))
            .collect();
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scores
            .into_iter()
            .take(k)
            .map(|(score, id)| SearchResult { id, score })
            .collect()
    }

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn memory_bytes(&self) -> usize {
        self.docs.len() * (self.dim * size_of::<f32>() + size_of::<u32>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_exact_neighbour() {
        let dim = 4;
        let mut idx = DenseFlat::new(dim);
        let target = vec![1.0f32, 0.0, 0.0, 0.0];
        let other = vec![0.0f32, 1.0, 0.0, 0.0];
        idx.insert(0, &target, &[]);
        idx.insert(1, &other, &[]);
        let results = idx.search(&target, &[], 1);
        assert_eq!(
            results[0].id, 0,
            "nearest to target should be target itself"
        );
    }

    #[test]
    fn len_tracks_inserts() {
        let mut idx = DenseFlat::new(3);
        assert!(idx.is_empty());
        idx.insert(0, &[1.0, 0.0, 0.0], &[]);
        idx.insert(1, &[0.0, 1.0, 0.0], &[]);
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn memory_bytes_positive() {
        let mut idx = DenseFlat::new(8);
        idx.insert(0, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[]);
        assert!(idx.memory_bytes() > 0);
    }
}
