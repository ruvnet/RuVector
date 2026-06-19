//! Flat exhaustive cosine ANN index.
//!
//! All vectors are stored as-is; cosine similarity is computed via dot product
//! and L2-norm.  This is a PoC baseline; production ANN would use HNSW or
//! DiskANN.

use crate::{DenseSearch, Document, SearchResult};

/// Brute-force dense ANN using cosine similarity.
///
/// Time: O(N·D) per query.
/// Memory: 4 · N · D bytes (f32 vectors only, no norms cached).
pub struct FlatDenseIndex {
    vectors: Vec<Vec<f32>>,
}

impl FlatDenseIndex {
    /// Build from a document corpus.  Vectors are NOT pre-normalised so that
    /// the index faithfully represents the raw embeddings.
    pub fn build(docs: &[Document]) -> Self {
        Self {
            vectors: docs.iter().map(|d| d.vector.clone()).collect(),
        }
    }

    /// Estimated byte cost of the vector store alone.
    pub fn byte_size(&self) -> usize {
        self.vectors.iter().map(|v| v.len() * 4).sum()
    }
}

impl DenseSearch for FlatDenseIndex {
    fn search(&self, vector: &[f32], k: usize) -> Vec<SearchResult> {
        let qnorm = l2_norm(vector);
        let mut results: Vec<SearchResult> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(id, dv)| SearchResult {
                id,
                score: cosine(vector, qnorm, dv),
            })
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

pub(crate) fn cosine(query: &[f32], qnorm: f32, doc: &[f32]) -> f32 {
    let dnorm = l2_norm(doc);
    if qnorm == 0.0 || dnorm == 0.0 {
        return 0.0;
    }
    let dot: f32 = query.iter().zip(doc.iter()).map(|(a, b)| a * b).sum();
    dot / (qnorm * dnorm)
}

pub(crate) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    fn doc(id: usize, v: Vec<f32>) -> Document {
        Document {
            id,
            tokens: vec![],
            vector: v,
        }
    }

    #[test]
    fn test_finds_closest_axis_aligned() {
        let docs = vec![
            doc(0, vec![1.0, 0.0, 0.0]),
            doc(1, vec![0.0, 1.0, 0.0]),
            doc(2, vec![0.0, 0.0, 1.0]),
        ];
        let idx = FlatDenseIndex::build(&docs);
        let r = idx.search(&[0.9, 0.1, 0.0], 1);
        assert_eq!(r[0].id, 0);
    }

    #[test]
    fn test_respects_k_limit() {
        let docs: Vec<Document> = (0..20)
            .map(|i| doc(i, vec![1.0_f32 / (i as f32 + 1.0), 0.0]))
            .collect();
        let idx = FlatDenseIndex::build(&docs);
        assert_eq!(idx.search(&[1.0, 0.0], 5).len(), 5);
    }

    #[test]
    fn test_identical_vectors_score_one() {
        let v = vec![0.6, 0.8];
        let docs = vec![doc(0, v.clone())];
        let idx = FlatDenseIndex::build(&docs);
        let r = idx.search(&v, 1);
        assert!((r[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_byte_size() {
        let docs = vec![doc(0, vec![0.0f32; 128])];
        let idx = FlatDenseIndex::build(&docs);
        assert_eq!(idx.byte_size(), 128 * 4);
    }
}
