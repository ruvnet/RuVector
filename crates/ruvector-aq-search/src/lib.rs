//! Anisotropic Product Quantization (AQ) for high-recall angular ANN search.
//!
//! Standard PQ minimises isotropic reconstruction error (L2). For cosine
//! similarity search the relevant metric is inner product, not L2. ScaNN
//! (Guo et al., NeurIPS 2020) showed that penalising residuals that are
//! *parallel* to the query vector during codebook training yields significantly
//! higher recall at the same compression ratio.
//!
//! This crate implements three variants for direct comparison:
//! - [`IsotropicFlat`]  — standard PQ trained with L2, ADC via inner product.
//! - [`AnisotropicFlat`] — AQ trained with directional penalty η, same ADC.
//! - [`AnisotropicResidual`] — AQ + f32 residual re-rank for top-k.
//!
//! All three implement [`AqSearch`]; no external service dependency.
//! All vectors are L2-normalised on insert (unit sphere, cosine = dot product).

pub mod codebook;
pub mod flat_aq;
pub mod residual_aq;

pub use codebook::{AqCodebook, AqConfig, TrainMode};
pub use flat_aq::AnisotropicFlat;
pub use flat_aq::IsotropicFlat;
pub use residual_aq::AnisotropicResidual;

/// Single search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    /// Approximate inner-product distance (higher = more similar).
    pub score: f32,
}

/// Unified trait for all AQ-based search backends.
pub trait AqSearch {
    /// Insert a vector (L2-normalised internally).
    fn insert(&mut self, vector: &[f32]);

    /// Return top-k nearest neighbours by cosine similarity.
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;

    /// Estimated heap memory in bytes.
    fn memory_bytes(&self) -> usize;

    fn name(&self) -> &'static str;
}

/// Brute-force exact cosine search; ground-truth for recall computation only.
pub struct ExactSearch {
    vectors: Vec<Vec<f32>>,
}

impl ExactSearch {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    pub fn insert(&mut self, v: &[f32]) {
        let norm = l2_norm(v);
        self.vectors.push(v.iter().map(|x| x / norm).collect());
    }

    pub fn search_exact(&self, query: &[f32], k: usize) -> Vec<usize> {
        let norm = l2_norm(query);
        let q: Vec<f32> = query.iter().map(|x| x / norm).collect();
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot(&q, v)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.into_iter().take(k).map(|(id, _)| id).collect()
    }
}

impl Default for ExactSearch {
    fn default() -> Self {
        Self::new()
    }
}

/// Recall@k: fraction of true top-k that appear in predicted top-k.
pub fn recall_at_k(predicted: &[usize], ground_truth: &[usize]) -> f32 {
    let k = ground_truth.len();
    if k == 0 {
        return 0.0;
    }
    let hits = predicted
        .iter()
        .filter(|id| ground_truth.contains(id))
        .count();
    hits as f32 / k as f32
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9)
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_perfect() {
        assert!((recall_at_k(&[0, 1, 2], &[0, 1, 2]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_zero() {
        assert!((recall_at_k(&[3, 4, 5], &[0, 1, 2]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn recall_half() {
        assert!((recall_at_k(&[0, 3], &[0, 1]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn exact_search_nearest_is_self() {
        let mut es = ExactSearch::new();
        let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
        es.insert(&v);
        let result = es.search_exact(&v, 1);
        assert_eq!(result, vec![0]);
    }
}
