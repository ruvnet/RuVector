//! Semantic drift detection for RuVector agent memory streams.
//!
//! Detects when the distributional properties of an agent's embedding stream
//! have shifted significantly from a learned baseline — a signal that memory
//! contents, topic focus, or data quality have changed.
//!
//! Three variants are provided:
//! - [`CentroidDrift`]: tracks centroid displacement (fastest, lowest memory)
//! - [`CovarianceTrace`]: tracks variance spread via Welford's online algorithm
//! - [`SlidingWindowKl`]: approximates KL divergence between historical/recent windows

mod centroid;
mod covariance;
mod sliding_window;

pub use centroid::CentroidDrift;
pub use covariance::CovarianceTrace;
pub use sliding_window::SlidingWindowKl;

/// Unified interface for all semantic drift detectors.
pub trait DriftDetector: Send + Sync {
    /// Feed one embedding into the detector.  Must be called in stream order.
    fn feed(&mut self, embedding: &[f32]);

    /// Returns a scalar drift score in [0, 1].  Higher = more drift.
    /// Returns 0.0 until the warmup window is complete.
    fn drift_score(&self) -> f32;

    /// Returns true when `drift_score()` exceeds the configured threshold.
    fn is_drifted(&self) -> bool;

    /// Resets the baseline to the current window.  Call when a drift event
    /// has been acknowledged and the new distribution should be accepted.
    fn reset_baseline(&mut self);

    /// Human-readable variant name for reporting.
    fn name(&self) -> &'static str;

    /// How many embeddings have been fed so far.
    fn sample_count(&self) -> usize;

    /// Estimated heap memory in bytes used by this detector.
    fn memory_bytes(&self) -> usize;
}

/// Cosine similarity between two equal-length vectors.
/// Returns a value in [-1, 1]; 1.0 = identical direction.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Cosine distance: 1 - cosine_similarity, in [0, 2].
pub(crate) fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
        assert!(cosine_distance(&v, &v) < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }
}
