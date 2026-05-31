//! Centroid-based drift detector.
//!
//! Tracks the running mean of the reference and current windows using Welford's
//! online algorithm.  Drift score = L2(cur_centroid - ref_centroid) / sqrt(d),
//! normalised so that random unit-variance Gaussian noise scores ≈ 0.
//!
//! Complexity: O(d) time and O(d) space per observation.

use std::collections::VecDeque;

use crate::{l2_sq, DriftDetector, DriftReport, DriftScore};

/// Centroid drift detector.
///
/// Fast and lightweight; suitable for high-throughput online monitoring.
/// Detects mean shift but cannot distinguish distributions with the same mean.
pub struct CentroidDriftDetector {
    dims: usize,
    threshold: f32,

    ref_centroid: Vec<f32>,
    ref_count: usize,

    // Current window: we keep raw vectors so promote_current can rebuild the
    // centroid exactly.  Window is bounded by `window_size`.
    cur_buffer: VecDeque<Vec<f32>>,
    cur_sum: Vec<f32>,
    window_size: usize,

    last_score: f32,
}

impl CentroidDriftDetector {
    /// Create a detector seeded with an initial reference batch.
    ///
    /// # Panics
    /// Panics if `reference` is empty or contains vectors of mixed length.
    pub fn new(reference: &[Vec<f32>], window_size: usize, threshold: f32) -> Self {
        assert!(!reference.is_empty(), "reference must not be empty");
        let dims = reference[0].len();
        for v in reference {
            assert_eq!(
                v.len(),
                dims,
                "all reference vectors must have the same dimension"
            );
        }

        let mut centroid = vec![0.0f32; dims];
        for v in reference {
            for (c, x) in centroid.iter_mut().zip(v.iter()) {
                *c += x;
            }
        }
        let n = reference.len() as f32;
        for c in &mut centroid {
            *c /= n;
        }

        Self {
            dims,
            threshold,
            ref_centroid: centroid,
            ref_count: reference.len(),
            cur_buffer: VecDeque::with_capacity(window_size),
            cur_sum: vec![0.0f32; dims],
            window_size,
            last_score: 0.0,
        }
    }

    fn cur_centroid(&self) -> Vec<f32> {
        let n = self.cur_buffer.len() as f32;
        if n == 0.0 {
            return vec![0.0; self.dims];
        }
        self.cur_sum.iter().map(|s| s / n).collect()
    }

    fn compute_score(&self) -> f32 {
        if self.cur_buffer.is_empty() {
            return 0.0;
        }
        let cur = self.cur_centroid();
        let l2 = l2_sq(&cur, &self.ref_centroid).sqrt();
        // Normalise: expected L2 between two random centroids of N(0,1) vectors
        // = sqrt(2/n * d) ≈ sqrt(2d / n).  We divide by sqrt(d) so the scale
        // stays interpretable regardless of dimension.
        l2 / (self.dims as f32).sqrt()
    }
}

impl DriftDetector for CentroidDriftDetector {
    fn observe(&mut self, vec: &[f32]) -> DriftScore {
        assert_eq!(vec.len(), self.dims, "vector dimension mismatch");

        // Evict oldest if at capacity
        if self.cur_buffer.len() == self.window_size {
            if let Some(evicted) = self.cur_buffer.pop_front() {
                for (s, x) in self.cur_sum.iter_mut().zip(evicted.iter()) {
                    *s -= x;
                }
            }
        }

        for (s, x) in self.cur_sum.iter_mut().zip(vec.iter()) {
            *s += x;
        }
        self.cur_buffer.push_back(vec.to_vec());

        self.last_score = self.compute_score();
        DriftScore {
            score: self.last_score,
            alert: self.last_score > self.threshold,
        }
    }

    fn report(&self) -> DriftReport {
        let mag = self.compute_score();
        DriftReport {
            drift_detected: mag > self.threshold,
            magnitude: mag,
            window_size: self.cur_buffer.len(),
            method: self.name(),
        }
    }

    fn reset_current(&mut self) {
        self.cur_buffer.clear();
        self.cur_sum.fill(0.0);
        self.last_score = 0.0;
    }

    fn promote_current(&mut self) {
        if self.cur_buffer.is_empty() {
            return;
        }
        let new_centroid = self.cur_centroid();
        self.ref_centroid = new_centroid;
        self.ref_count = self.cur_buffer.len();
        self.reset_current();
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn name(&self) -> &'static str {
        "centroid"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_vecs(n: usize, d: usize, mean: f32, seed: u64) -> Vec<Vec<f32>> {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0 + mean)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn no_drift_scores_low() {
        let ref_data = gaussian_vecs(500, 64, 0.0, 42);
        let mut det = CentroidDriftDetector::new(&ref_data, 200, 0.3);

        let cur = gaussian_vecs(200, 64, 0.0, 99);
        for v in &cur {
            det.observe(v);
        }

        let report = det.report();
        // Same distribution: centroid drift should be small
        assert!(
            report.magnitude < 0.3,
            "expected low drift, got {:.4}",
            report.magnitude
        );
        assert!(!report.drift_detected);
    }

    #[test]
    fn large_drift_detected() {
        let ref_data = gaussian_vecs(500, 64, 0.0, 42);
        let mut det = CentroidDriftDetector::new(&ref_data, 200, 0.3);

        // Shift centroid by 2.0 in every dimension — clearly drifted
        let cur = gaussian_vecs(200, 64, 2.0, 77);
        for v in &cur {
            det.observe(v);
        }

        let report = det.report();
        assert!(
            report.drift_detected,
            "expected drift alert, score={:.4}",
            report.magnitude
        );
        assert!(
            report.magnitude > 1.0,
            "expected large magnitude, got {:.4}",
            report.magnitude
        );
    }

    #[test]
    fn promote_resets_reference() {
        let ref_data = gaussian_vecs(200, 32, 0.0, 1);
        let mut det = CentroidDriftDetector::new(&ref_data, 100, 0.5);

        let shifted = gaussian_vecs(100, 32, 3.0, 2);
        for v in &shifted {
            det.observe(v);
        }
        let before = det.report().magnitude;
        assert!(before > 0.5, "should be drifted before promote");

        det.promote_current();

        // After promotion, reference is the shifted data; more shifted data == no drift
        let same_shifted = gaussian_vecs(100, 32, 3.0, 3);
        for v in &same_shifted {
            det.observe(v);
        }
        let after = det.report().magnitude;
        assert!(
            after < 0.3,
            "after promote, same distribution should not drift, got {:.4}",
            after
        );
    }
}
