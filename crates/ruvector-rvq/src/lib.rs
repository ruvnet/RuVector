//! Residual Vector Quantization (RVQ) for RuVector
//!
//! RVQ encodes each vector through K successive k-means stages. Stage 0 quantises
//! the raw vector; each later stage quantises the residual error of the previous
//! stage.  Decoding is the element-wise sum of the selected codewords.
//!
//! Three measurable variants:
//!   `ExactSearch` – brute-force f32 inner-product (ground truth)
//!   `PqSearch`    – Product Quantization 8 sub-spaces × 256 codewords (baseline)
//!   `RvqSearch`   – Residual VQ 8 stages × 256 codewords (main contribution)
//!
//! All three operate on unit-normalised f32 vectors; inner product ≡ cosine sim.

pub mod dataset;
pub mod exact;
pub mod kmeans;
pub mod pq;
pub mod rvq;

use std::collections::HashSet;

// ─── core hit type ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Hit {
    pub id: usize,
    pub score: f32,
}

// ─── shared trait ────────────────────────────────────────────────────────────

pub trait VectorIndex: Send + Sync {
    /// Return the `k` approximate nearest neighbours (by inner product) to `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;

    /// Human-readable variant name for reporting.
    fn name(&self) -> &str;

    /// Estimated heap bytes consumed by this index (codebooks + codes + raw vecs).
    fn memory_bytes(&self) -> usize;
}

// ─── recall metric ───────────────────────────────────────────────────────────

/// Recall@k: fraction of the true top-k that appear in `approx`.
pub fn recall_at_k(truth: &[Hit], approx: &[Hit], k: usize) -> f32 {
    let true_ids: HashSet<usize> = truth.iter().take(k).map(|h| h.id).collect();
    let found = approx
        .iter()
        .take(k)
        .filter(|h| true_ids.contains(&h.id))
        .count();
    found as f32 / k.min(truth.len()) as f32
}

// ─── math primitives (shared across modules) ─────────────────────────────────

/// Inner product (cosine similarity for unit vectors).
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Squared L2 distance.
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Index of the nearest centroid and its L2-squared distance to `v`.
pub fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_sq(v, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("centroids must not be empty")
}

// ─── latency helpers ─────────────────────────────────────────────────────────

pub fn percentile_ns(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * p / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

pub fn mean_us(times_ns: &[u128]) -> f64 {
    if times_ns.is_empty() {
        return 0.0;
    }
    times_ns.iter().sum::<u128>() as f64 / times_ns.len() as f64 / 1_000.0
}

pub fn qps(times_ns: &[u128]) -> f64 {
    if times_ns.is_empty() {
        return 0.0;
    }
    let total_s = times_ns.iter().sum::<u128>() as f64 / 1e9;
    times_ns.len() as f64 / total_s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_and_l2_sq_agree() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!((dot(&a, &b)).abs() < 1e-6);
        assert!((l2_sq(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn recall_perfect() {
        let t = vec![Hit { id: 0, score: 1.0 }, Hit { id: 1, score: 0.9 }];
        let a = t.clone();
        assert!((recall_at_k(&t, &a, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_zero() {
        let t = vec![Hit { id: 0, score: 1.0 }];
        let a = vec![Hit { id: 99, score: 0.5 }];
        assert!(recall_at_k(&t, &a, 1).abs() < 1e-6);
    }
}
