//! Sliding-window approximate KL divergence drift detector.
//!
//! Maintains two windows of pairwise cosine similarities:
//! - A **reference window** (historical embeddings from the warmup phase).
//! - A **detection window** (the last `window_size` embeddings).
//!
//! The reference window is summarised as a histogram of within-reference
//! pairwise cosine similarities.  The detection window is characterised by
//! cross-window similarities (detection vs reference).  The approximate KL
//! divergence between the cross histogram and the reference histogram drives
//! the drift score.  Using cross-window similarities means the detector catches
//! directional drift even when both windows are internally cohesive.
//!
//! This variant is the most sensitive to distributional shape changes, not
//! just centroid or variance shifts.  It is also the most memory-hungry:
//! O(2 * window_size * dim * 4 bytes) plus histogram storage.
//!
//! Memory: O(2 * window_size * dim * 4 + 2 * buckets * 4) bytes.

use crate::{cosine_similarity, DriftDetector};

const BUCKETS: usize = 32;

fn build_histogram(sims: &[f32]) -> [f32; BUCKETS] {
    let mut hist = [0.0_f32; BUCKETS];
    if sims.is_empty() {
        return hist;
    }
    for &s in sims {
        // cosine in [-1, 1] → bucket in [0, BUCKETS)
        let idx = ((s + 1.0) / 2.0 * BUCKETS as f32)
            .floor()
            .clamp(0.0, (BUCKETS - 1) as f32) as usize;
        hist[idx] += 1.0;
    }
    let total: f32 = hist.iter().sum();
    if total > 0.0 {
        for h in &mut hist {
            *h /= total;
        }
    }
    hist
}

/// KL(p || q) approximation with epsilon smoothing to handle zero buckets.
fn kl_divergence(p: &[f32; BUCKETS], q: &[f32; BUCKETS]) -> f32 {
    let eps = 1e-6_f32;
    p.iter()
        .zip(q)
        .map(|(&pi, &qi)| {
            let pi = pi + eps;
            let qi = qi + eps;
            pi * (pi / qi).ln()
        })
        .sum()
}

/// Compute pairwise cosine similarities from a random subsample of `matrix`.
/// `matrix` rows are embeddings of length `dim`.
/// Returns up to `max_pairs` similarity values.
fn pairwise_sims(matrix: &[Vec<f32>], max_pairs: usize) -> Vec<f32> {
    let n = matrix.len();
    if n < 2 {
        return vec![];
    }
    let mut sims = Vec::with_capacity(max_pairs.min(n * n / 2));
    'outer: for i in 0..n {
        for j in (i + 1)..n {
            sims.push(cosine_similarity(&matrix[i], &matrix[j]));
            if sims.len() >= max_pairs {
                break 'outer;
            }
        }
    }
    sims
}

/// Compute cosine similarities between every row of `a` and every row of `b`.
/// Returns up to `max_pairs` values.  Cross-window similarities capture how
/// similar new embeddings are to the reference distribution — not just how
/// cohesive the new window is internally.
fn cross_sims(a: &[Vec<f32>], b: &[Vec<f32>], max_pairs: usize) -> Vec<f32> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut sims = Vec::with_capacity(max_pairs.min(a.len() * b.len()));
    'outer: for ai in a {
        for bi in b {
            sims.push(cosine_similarity(ai, bi));
            if sims.len() >= max_pairs {
                break 'outer;
            }
        }
    }
    sims
}

/// Sliding-window KL divergence drift detector.
pub struct SlidingWindowKl {
    dim: usize,
    window_size: usize,
    warmup: usize,
    threshold: f32,
    /// Maximum pairwise pairs sampled per histogram
    max_pairs: usize,
    reference: Vec<Vec<f32>>,
    detection: Vec<Vec<f32>>,
    reference_hist: [f32; BUCKETS],
    count: usize,
    current_kl: f32,
}

impl SlidingWindowKl {
    /// Create a new sliding-window KL detector.
    ///
    /// # Parameters
    /// - `dim`: embedding dimension
    /// - `window_size`: samples in each of the two comparison windows
    /// - `warmup`: same as `window_size` (fills the reference window)
    /// - `threshold`: KL divergence value triggering drift detection
    /// - `max_pairs`: maximum pairwise comparisons per histogram (caps O(n²))
    pub fn new(dim: usize, window_size: usize, threshold: f32, max_pairs: usize) -> Self {
        assert!(dim > 0);
        assert!(window_size >= 2);
        Self {
            dim,
            window_size,
            warmup: window_size,
            threshold,
            max_pairs,
            reference: Vec::with_capacity(window_size),
            detection: Vec::with_capacity(window_size),
            reference_hist: [0.0; BUCKETS],
            count: 0,
            current_kl: 0.0,
        }
    }

    /// Estimated heap bytes (excludes Vec metadata overhead).
    pub fn memory_bytes(&self) -> usize {
        (self.reference.capacity() + self.detection.capacity()) * self.dim * 4 + 2 * BUCKETS * 4
    }

    fn recompute_kl(&mut self) {
        // Compare cross-window similarities (detection vs reference) against
        // the within-reference baseline.  This captures the case where detection
        // vectors point in a different direction than reference vectors, even when
        // both windows are internally cohesive (unit vectors, tight clusters, etc.).
        let cross_sims = cross_sims(&self.detection, &self.reference, self.max_pairs);
        let cross_hist = build_histogram(&cross_sims);
        self.current_kl = kl_divergence(&cross_hist, &self.reference_hist);
    }
}

impl DriftDetector for SlidingWindowKl {
    fn feed(&mut self, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.dim);
        self.count += 1;

        if self.count <= self.warmup {
            self.reference.push(embedding.to_vec());
            if self.count == self.warmup {
                let ref_sims = pairwise_sims(&self.reference, self.max_pairs);
                self.reference_hist = build_histogram(&ref_sims);
            }
        } else {
            if self.detection.len() >= self.window_size {
                self.detection.remove(0);
            }
            self.detection.push(embedding.to_vec());
            if self.detection.len() >= 2 {
                self.recompute_kl();
            }
        }
    }

    fn drift_score(&self) -> f32 {
        if self.count <= self.warmup {
            return 0.0;
        }
        // KL is unbounded; clamp reasonable range 0–10 to [0,1]
        (self.current_kl / 10.0).clamp(0.0, 1.0)
    }

    fn is_drifted(&self) -> bool {
        self.count > self.warmup && self.current_kl > self.threshold
    }

    fn reset_baseline(&mut self) {
        if !self.detection.is_empty() {
            self.reference.clear();
            self.reference.extend_from_slice(&self.detection);
            let ref_sims = pairwise_sims(&self.reference, self.max_pairs);
            self.reference_hist = build_histogram(&ref_sims);
            self.detection.clear();
            self.current_kl = 0.0;
        }
    }

    fn name(&self) -> &'static str {
        "SlidingWindowKL"
    }

    fn sample_count(&self) -> usize {
        self.count
    }

    fn memory_bytes(&self) -> usize {
        (self.reference.capacity() + self.detection.capacity()) * self.dim * 4
            + 2 * BUCKETS * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, dir: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[dir] = 1.0;
        v
    }

    #[test]
    fn no_score_before_warmup() {
        let mut d = SlidingWindowKl::new(8, 20, 0.3, 50);
        for _ in 0..10 {
            d.feed(&unit_vec(8, 0));
        }
        assert_eq!(d.drift_score(), 0.0);
    }

    #[test]
    fn detects_orthogonal_shift() {
        let dim = 8;
        let window = 15;
        let mut d = SlidingWindowKl::new(dim, window, 0.05, 200);
        // Reference window: direction 0
        for _ in 0..window {
            d.feed(&unit_vec(dim, 0));
        }
        // Detection window: direction 1 (orthogonal)
        for _ in 0..window {
            d.feed(&unit_vec(dim, 1));
        }
        assert!(
            d.is_drifted(),
            "KL={:.4} should exceed threshold",
            d.current_kl
        );
    }

    #[test]
    fn histogram_sums_to_one() {
        let sims = vec![-0.5_f32, 0.0, 0.5, 1.0, -1.0, 0.9];
        let h = build_histogram(&sims);
        let total: f32 = h.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "total={total}");
    }

    #[test]
    fn kl_identical_histograms() {
        let h = build_histogram(&[0.0_f32; 32]);
        // all-zero after normalisation: each bucket gets eps / (BUCKETS * eps)
        // KL(p||p) == 0 when p is uniform
        let kl = kl_divergence(&h, &h);
        assert!(kl < 1e-3, "kl={kl}");
    }
}
