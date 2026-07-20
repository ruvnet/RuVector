//! Exact brute-force flat scan — 100 % recall baseline.

use crate::{dot, MrlSearch, SearchResult};

/// Stores vectors in a flat `Vec` and scores every vector on every query.
///
/// O(N·D) per query — the ground-truth reference for recall measurement.
pub struct FlatIndex {
    vectors: Vec<Vec<f32>>,
    dim: usize,
}

impl FlatIndex {
    /// Create an empty flat index for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        FlatIndex {
            vectors: Vec::new(),
            dim,
        }
    }
}

impl MrlSearch for FlatIndex {
    fn insert(&mut self, id: u32, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "dimension mismatch on insert");
        assert_eq!(
            id as usize,
            self.vectors.len(),
            "ids must be inserted sequentially"
        );
        self.vectors.push(vector.to_vec());
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        assert!(
            query.len() >= self.dim,
            "query must have at least {} dims",
            self.dim
        );
        let q = &query[..self.dim];
        let mut scores: Vec<(u32, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, dot(q, v)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect()
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize;

    #[test]
    fn test_flat_exact_top1() {
        let dim = 4;
        let mut idx = FlatIndex::new(dim);
        let mut v0 = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut v1 = vec![0.0f32, 1.0, 0.0, 0.0];
        normalize(&mut v0);
        normalize(&mut v1);
        idx.insert(0, &v0);
        idx.insert(1, &v1);

        // query aligned with v0
        let results = idx.search(&v0, 1);
        assert_eq!(results[0].id, 0);
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_flat_returns_k() {
        let mut idx = FlatIndex::new(8);
        for i in 0..20u32 {
            let v: Vec<f32> = (0..8)
                .map(|j| if j == i as usize % 8 { 1.0 } else { 0.0 })
                .collect();
            idx.insert(i, &v);
        }
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = idx.search(&q, 5);
        assert_eq!(results.len(), 5);
    }
}
