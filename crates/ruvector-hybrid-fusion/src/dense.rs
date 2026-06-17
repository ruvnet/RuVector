//! Flat cosine-similarity scan — the dense leg of hybrid search.
//!
//! Stores unit-normalised vectors; inner product equals cosine similarity.
//! O(N·D) per query: exact but simple, suitable for PoC at N ≤ 100 K.

pub struct DenseIndex {
    /// Unit-normalised stored vectors, row-major.
    vectors: Vec<Vec<f32>>,
    dim: usize,
}

impl DenseIndex {
    /// Build from raw (un-normalised) vectors.
    pub fn build(vectors: Vec<Vec<f32>>) -> Self {
        assert!(!vectors.is_empty(), "corpus must not be empty");
        let dim = vectors[0].len();
        let normalised = vectors.into_iter().map(|v| unit_normalise(v)).collect();
        DenseIndex { vectors: normalised, dim }
    }

    /// Return top-k (doc_id, cosine_score) for a query vector.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        let q = unit_normalise(query.to_vec());

        let mut scores: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot(&q, v)))
            .collect();

        // Partial-sort: find top_k without full sort for larger N.
        // For this PoC N is ≤ 5K so full sort is fine and provably correct.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Number of indexed vectors.
    pub fn doc_count(&self) -> usize {
        self.vectors.len()
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Approximate memory in bytes (float32 stored vectors only).
    pub fn memory_bytes(&self) -> usize {
        self.vectors.len() * self.dim * 4
    }
}

fn unit_normalise(v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return v;
    }
    v.into_iter().map(|x| x / norm).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vecs() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn search_identity_query() {
        let idx = DenseIndex::build(make_vecs());
        let q = vec![1.0_f32, 0.0, 0.0];
        let results = idx.search(&q, 4);
        assert_eq!(results[0].0, 0, "doc 0 (1,0,0) should be top for query (1,0,0)");
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_returns_at_most_top_k() {
        let idx = DenseIndex::build(make_vecs());
        let q = vec![0.5_f32, 0.5, 0.0];
        let results = idx.search(&q, 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn scores_bounded_minus_one_to_one() {
        let idx = DenseIndex::build(make_vecs());
        let q = vec![0.3_f32, 0.7, 0.0];
        for &(_, s) in &idx.search(&q, 4) {
            assert!(s >= -1.0 - 1e-6 && s <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn memory_bytes_correct() {
        let vecs = make_vecs();
        let idx = DenseIndex::build(vecs);
        // 4 docs × 3 dim × 4 bytes = 48
        assert_eq!(idx.memory_bytes(), 4 * 3 * 4);
    }

    #[test]
    fn dim_mismatch_panics() {
        let idx = DenseIndex::build(make_vecs());
        let result = std::panic::catch_unwind(|| idx.search(&[1.0, 0.0], 1));
        assert!(result.is_err());
    }
}
