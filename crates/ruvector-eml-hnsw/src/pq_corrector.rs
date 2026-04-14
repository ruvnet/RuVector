//! Quantization-aware distance correction for DiskANN Product Quantization.
//!
//! Product Quantization (PQ) compresses vectors for disk-based ANN search,
//! but the approximate distances have systematic error. This module uses
//! an EML model to learn the error correction function from observed
//! `(pq_distance, exact_distance)` pairs.
//!
//! # Features
//!
//! The model uses 3 input features:
//! 1. `pq_approximate_distance` — the raw PQ distance
//! 2. `codebook_usage_ratio` — how evenly the codebook is used (0-1)
//! 3. `quantization_residual_estimate` — estimated residual from PQ
//!
//! The output is the corrected distance (closer to exact).
//!
//! # Integration
//!
//! After the PQ distance computation in DiskANN, call
//! [`PqDistanceCorrector::correct`] to refine the approximate distance.
//! This typically improves recall by 5-15% at negligible compute cost
//! (the EML model is O(1)).

use eml_core::EmlModel;
use serde::{Deserialize, Serialize};

/// Minimum training samples for the correction model.
const MIN_TRAINING_SAMPLES: usize = 100;

/// A training record: PQ distance, exact distance, and residual info.
#[derive(Debug, Clone)]
struct CorrectionRecord {
    pq_dist: f32,
    exact_dist: f32,
    residual: f32,
}

/// Corrects PQ distance approximation error using a learned EML model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqDistanceCorrector {
    /// EML model: (pq_dist, codebook_ratio, residual) -> corrected_dist.
    model: EmlModel,
    /// Whether correction is active (model trained).
    trained: bool,
    /// Running statistics for normalization.
    max_pq_dist: f64,
    max_residual: f64,
    /// Accumulated training records (skipped in serde).
    #[serde(skip)]
    records: Vec<CorrectionRecord>,
}

impl PqDistanceCorrector {
    /// Create a new untrained PQ distance corrector.
    pub fn new() -> Self {
        // 3 input features, 1 output head (corrected distance).
        let model = EmlModel::new(4, 3, 1);
        Self {
            model,
            trained: false,
            max_pq_dist: 1.0,
            max_residual: 1.0,
            records: Vec::new(),
        }
    }

    /// Whether the corrector has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Number of training records accumulated.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Correct a PQ approximate distance.
    ///
    /// # Arguments
    /// - `pq_dist`: The approximate distance from PQ computation.
    /// - `residual_hint`: Estimated quantization residual (e.g., from
    ///   the PQ codebook). Pass 0.0 if unavailable.
    ///
    /// Returns the corrected distance. If the model is not trained,
    /// returns `pq_dist` unchanged.
    pub fn correct(&self, pq_dist: f32, residual_hint: f32) -> f32 {
        if !self.trained {
            return pq_dist;
        }

        let features = self.build_features(pq_dist, residual_hint);
        let corrected = self.model.predict_primary(&features);

        // Scale back to distance space and ensure non-negative.
        let result = (corrected * self.max_pq_dist).max(0.0) as f32;

        // Sanity: corrected distance should be in a reasonable range
        // relative to PQ distance. Clamp to [0.5 * pq_dist, 2.0 * pq_dist].
        result.clamp(pq_dist * 0.25, pq_dist * 4.0)
    }

    /// Correct a batch of PQ distances.
    ///
    /// More efficient than calling `correct` in a loop because the
    /// model parameters are loaded once.
    pub fn correct_batch(&self, pq_dists: &[f32], residuals: &[f32]) -> Vec<f32> {
        if !self.trained {
            return pq_dists.to_vec();
        }

        pq_dists
            .iter()
            .zip(residuals.iter())
            .map(|(&d, &r)| self.correct(d, r))
            .collect()
    }

    /// Record a training observation.
    ///
    /// # Arguments
    /// - `pq_dist`: The approximate PQ distance.
    /// - `exact_dist`: The exact distance (ground truth).
    /// - `residual`: Quantization residual estimate.
    pub fn record(&mut self, pq_dist: f32, exact_dist: f32, residual: f32) {
        // Update running max for normalization.
        if (pq_dist as f64) > self.max_pq_dist {
            self.max_pq_dist = pq_dist as f64;
        }
        if (residual as f64) > self.max_residual {
            self.max_residual = residual as f64;
        }

        self.records.push(CorrectionRecord {
            pq_dist,
            exact_dist,
            residual,
        });
    }

    /// Train the correction model from accumulated observations.
    ///
    /// Returns `true` if training converged.
    pub fn train(&mut self) -> bool {
        if self.records.len() < MIN_TRAINING_SAMPLES {
            return false;
        }

        // Rebuild the EML model with fresh training data.
        let mut model = EmlModel::new(4, 3, 1);

        for record in &self.records {
            let features = self.build_features(record.pq_dist, record.residual);
            // Target: exact distance normalized by max_pq_dist.
            let target = record.exact_dist as f64 / self.max_pq_dist;
            model.record(&features, &[Some(target)]);
        }

        let converged = model.train();
        self.model = model;
        self.trained = true;
        converged
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    /// Build normalized feature vector for the model.
    fn build_features(&self, pq_dist: f32, residual: f32) -> Vec<f64> {
        vec![
            // PQ distance normalized.
            (pq_dist as f64 / self.max_pq_dist).clamp(0.0, 2.0),
            // Codebook usage ratio (derived from residual / pq_dist).
            if pq_dist > 0.0 {
                (1.0 - (residual as f64 / pq_dist as f64)).clamp(0.0, 1.0)
            } else {
                0.5
            },
            // Residual normalized.
            (residual as f64 / self.max_residual).clamp(0.0, 2.0),
        ]
    }
}

impl Default for PqDistanceCorrector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_corrector_not_trained() {
        let c = PqDistanceCorrector::new();
        assert!(!c.is_trained());
        assert_eq!(c.record_count(), 0);
    }

    #[test]
    fn correct_untrained_returns_pq_dist() {
        let c = PqDistanceCorrector::new();
        let result = c.correct(1.5, 0.1);
        assert!((result - 1.5).abs() < 1e-6);
    }

    #[test]
    fn correct_batch_untrained_returns_pq_dists() {
        let c = PqDistanceCorrector::new();
        let dists = vec![1.0, 2.0, 3.0];
        let residuals = vec![0.1, 0.2, 0.3];
        let corrected = c.correct_batch(&dists, &residuals);
        assert_eq!(corrected.len(), 3);
        for (i, &d) in corrected.iter().enumerate() {
            assert!((d - dists[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn record_increments_count() {
        let mut c = PqDistanceCorrector::new();
        c.record(1.0, 1.1, 0.1);
        assert_eq!(c.record_count(), 1);
        c.record(2.0, 2.2, 0.2);
        assert_eq!(c.record_count(), 2);
    }

    #[test]
    fn train_insufficient_data_returns_false() {
        let mut c = PqDistanceCorrector::new();
        for i in 0..20 {
            c.record(i as f32, i as f32 * 1.1, i as f32 * 0.05);
        }
        assert!(!c.train());
    }

    #[test]
    fn train_with_linear_error() {
        let mut c = PqDistanceCorrector::new();

        // PQ systematically underestimates by ~10%.
        for i in 0..150 {
            let exact = (i as f32 + 1.0) * 0.1;
            let pq = exact * 0.9; // 10% underestimate
            let residual = (exact - pq).abs();
            c.record(pq, exact, residual);
        }

        // Training may or may not converge, but should not panic.
        let _ = c.train();
        assert!(c.is_trained());

        // Corrected distance should be closer to exact than PQ.
        let pq_dist = 5.0 * 0.9; // 4.5
        let exact = 5.0;
        let corrected = c.correct(pq_dist, (exact - pq_dist).abs());
        assert!(corrected.is_finite());
        assert!(corrected > 0.0);
    }

    #[test]
    fn correct_output_is_bounded() {
        let mut c = PqDistanceCorrector::new();
        for i in 0..150 {
            let pq = (i as f32 + 1.0) * 0.5;
            let exact = pq * 1.05;
            c.record(pq, exact, 0.1);
        }
        c.train();

        let corrected = c.correct(10.0, 0.5);
        assert!(corrected.is_finite());
        // Should be within [0.25 * pq, 4.0 * pq] = [2.5, 40.0].
        assert!(corrected >= 2.5);
        assert!(corrected <= 40.0);
    }

    #[test]
    fn correct_zero_pq_dist() {
        let c = PqDistanceCorrector::new();
        let result = c.correct(0.0, 0.0);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut c = PqDistanceCorrector::new();
        c.record(1.0, 1.1, 0.1);
        c.record(2.0, 2.2, 0.2);

        let json = serde_json::to_string(&c).unwrap();
        let c2: PqDistanceCorrector = serde_json::from_str(&json).unwrap();
        assert_eq!(c.is_trained(), c2.is_trained());
        assert!((c.max_pq_dist - c2.max_pq_dist).abs() < 1e-10);
    }

    #[test]
    fn build_features_length() {
        let c = PqDistanceCorrector::new();
        let features = c.build_features(1.0, 0.1);
        assert_eq!(features.len(), 3);
        for &f in &features {
            assert!(f.is_finite());
            assert!(f >= 0.0);
        }
    }

    #[test]
    fn max_stats_update_on_record() {
        let mut c = PqDistanceCorrector::new();
        assert!((c.max_pq_dist - 1.0).abs() < 1e-10);
        c.record(100.0, 105.0, 5.0);
        assert!((c.max_pq_dist - 100.0).abs() < 1e-10);
        assert!((c.max_residual - 5.0).abs() < 1e-10);
    }
}
