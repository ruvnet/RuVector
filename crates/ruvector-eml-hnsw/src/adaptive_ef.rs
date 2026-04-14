//! Adaptive beam width (ef) prediction per query.
//!
//! Different queries have different difficulty levels. A query near a dense
//! cluster needs a small ef; a query in a sparse region needs a large ef.
//! This model learns to predict the right ef from query features, avoiding
//! wasted work on easy queries while maintaining recall on hard ones.
//!
//! Expected speedup: 1.5-3x by avoiding overprovisioned beam width.
//!
//! # Features Used
//!
//! 1. `query_norm`: L2 norm of the query vector (normalized)
//! 2. `query_variance`: variance of query components (normalized)
//! 3. `graph_size_log`: log10(graph_size) / 8.0
//! 4. `query_max_component`: max absolute component value (normalized)

use eml_core::EmlModel;
use serde::{Deserialize, Serialize};

/// Learns optimal beam width (ef) per query for target recall.
///
/// # Example
///
/// ```
/// use ruvector_eml_hnsw::AdaptiveEfModel;
///
/// let mut model = AdaptiveEfModel::new(64, 10, 200);
///
/// // Before training, returns default_ef
/// let ef = model.predict_ef(&[0.5f32; 128], 10_000);
/// assert_eq!(ef, 64);
///
/// // Record observations during searches
/// model.record(&[0.5f32; 128], 10_000, 50, 0.98);
/// // ... record many more ...
/// // model.train();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveEfModel {
    /// EML model: 4 input features -> 1 output (predicted ef).
    model: EmlModel,
    /// Whether training is complete.
    trained: bool,
    /// Default ef to use before training.
    default_ef: usize,
    /// Minimum ef to ever return.
    min_ef: usize,
    /// Maximum ef to ever return.
    max_ef: usize,
    /// Training buffer: (query_features, ef_used, recall_achieved).
    #[serde(skip)]
    training_buffer: Vec<([f64; 4], usize, f64)>,
}

impl AdaptiveEfModel {
    /// Create a new adaptive ef model.
    ///
    /// # Arguments
    /// - `default_ef`: ef to use before training is complete.
    /// - `min_ef`: minimum ef to ever predict (safety floor).
    /// - `max_ef`: maximum ef to ever predict (budget ceiling).
    pub fn new(default_ef: usize, min_ef: usize, max_ef: usize) -> Self {
        let model = EmlModel::new(3, 4, 1);
        Self {
            model,
            trained: false,
            default_ef,
            min_ef: min_ef.max(1),
            max_ef,
            training_buffer: Vec::new(),
        }
    }

    /// Whether the model has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Number of training samples collected.
    pub fn sample_count(&self) -> usize {
        self.training_buffer.len()
    }

    /// Predict optimal ef for this query.
    ///
    /// Returns `default_ef` if the model has not been trained yet.
    pub fn predict_ef(&self, query: &[f32], graph_size: usize) -> usize {
        if !self.trained {
            return self.default_ef;
        }

        let features = Self::extract_features(query, graph_size);
        let predicted = self.model.predict_primary(&features);
        // The model predicts normalized ef; denormalize and clamp
        let ef_raw = predicted * self.max_ef as f64;
        (ef_raw as usize).clamp(self.min_ef, self.max_ef)
    }

    /// Record a training observation.
    ///
    /// # Arguments
    /// - `query`: the query vector used.
    /// - `graph_size`: number of points in the graph at search time.
    /// - `ef`: the ef value used for this search.
    /// - `recall`: the recall achieved (0.0 to 1.0).
    pub fn record(&mut self, query: &[f32], graph_size: usize, ef: usize, recall: f64) {
        let features = Self::extract_features(query, graph_size);
        self.training_buffer.push((features, ef, recall));
    }

    /// Train the model to predict minimum ef for >= 95% recall.
    ///
    /// Returns `true` if training converged.
    pub fn train(&mut self) -> bool {
        self.train_for_target_recall(0.95)
    }

    /// Train for a specific target recall threshold.
    pub fn train_for_target_recall(&mut self, target_recall: f64) -> bool {
        if self.training_buffer.len() < 100 {
            return false;
        }

        self.model = EmlModel::new(3, 4, 1);

        // Select samples with adequate recall
        let good_count = self
            .training_buffer
            .iter()
            .filter(|(_, _, recall)| *recall >= target_recall)
            .count();

        if good_count < 50 {
            // Not enough high-recall samples; train on all data
            for (features, ef, _) in &self.training_buffer {
                let ef_normalized = *ef as f64 / self.max_ef as f64;
                self.model.record(features, &[Some(ef_normalized)]);
            }
        } else {
            // Train on samples that achieved target recall
            for (features, ef, recall) in &self.training_buffer {
                if *recall >= target_recall {
                    let ef_normalized = *ef as f64 / self.max_ef as f64;
                    self.model.record(features, &[Some(ef_normalized)]);
                }
            }
        }

        let converged = self.model.train();
        self.trained = true;
        converged
    }

    /// Extract 4 normalized features from a query vector.
    pub(crate) fn extract_features(query: &[f32], graph_size: usize) -> [f64; 4] {
        let n = query.len() as f64;
        if n == 0.0 {
            return [0.0; 4];
        }

        // Feature 1: L2 norm (normalized by sqrt(dim))
        let norm: f64 = query
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        let norm_normalized = (norm / n.sqrt()).min(1.0);

        // Feature 2: standard deviation of components (normalized)
        let mean: f64 = query.iter().map(|&x| x as f64).sum::<f64>() / n;
        let variance: f64 = query
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std_normalized = variance.sqrt().min(1.0);

        // Feature 3: log graph size (normalized to ~[0, 1])
        let graph_log = if graph_size > 0 {
            (graph_size as f64).log10() / 8.0
        } else {
            0.0
        };
        let graph_normalized = graph_log.min(1.0);

        // Feature 4: max absolute component (normalized)
        let max_abs: f64 = query
            .iter()
            .map(|&x| (x as f64).abs())
            .fold(0.0f64, f64::max);
        let max_normalized = max_abs.min(1.0);

        [norm_normalized, std_normalized, graph_normalized, max_normalized]
    }

    /// Serialize the model to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("AdaptiveEfModel serialization should not fail")
    }

    /// Deserialize a model from JSON.
    pub fn from_json(json: &str) -> Option<Self> {
        serde_json::from_str(json).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_defaults() {
        let m = AdaptiveEfModel::new(64, 10, 200);
        assert!(!m.is_trained());
        assert_eq!(m.sample_count(), 0);
    }

    #[test]
    fn untrained_returns_default() {
        let m = AdaptiveEfModel::new(64, 10, 200);
        let ef = m.predict_ef(&[0.5f32; 128], 10_000);
        assert_eq!(ef, 64);
    }

    #[test]
    fn record_increments() {
        let mut m = AdaptiveEfModel::new(64, 10, 200);
        assert_eq!(m.sample_count(), 0);
        m.record(&[1.0f32; 8], 1000, 50, 0.98);
        assert_eq!(m.sample_count(), 1);
    }

    #[test]
    fn train_insufficient_data() {
        let mut m = AdaptiveEfModel::new(64, 10, 200);
        for _ in 0..10 {
            m.record(&[0.5f32; 8], 1000, 50, 0.95);
        }
        assert!(!m.train());
    }

    #[test]
    fn train_with_data() {
        let mut m = AdaptiveEfModel::new(64, 10, 200);
        let mut rng = 42u64;
        for _ in 0..200 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let t = (rng >> 33) as f32 / (u32::MAX as f32);
            let dim = 16;
            let query: Vec<f32> = (0..dim)
                .map(|i| t * (i as f32 + 1.0) / dim as f32)
                .collect();
            let ef_needed = (20.0 + t * 100.0) as usize;
            let recall = if ef_needed < 100 { 0.98 } else { 0.92 };
            m.record(&query, 10_000, ef_needed, recall);
        }

        m.train();
        assert!(m.is_trained());

        let ef = m.predict_ef(&[0.5f32; 16], 10_000);
        assert!(ef >= 10, "ef >= min_ef: got {ef}");
        assert!(ef <= 200, "ef <= max_ef: got {ef}");
    }

    #[test]
    fn clamps_predictions() {
        let mut m = AdaptiveEfModel::new(64, 10, 200);
        for _ in 0..200 {
            m.record(&[0.1f32; 8], 100, 5, 0.99);
        }
        m.train();
        let ef = m.predict_ef(&[0.1f32; 8], 100);
        assert!(ef >= 10, "clamped to min_ef: got {ef}");
    }

    #[test]
    fn feature_extraction_deterministic() {
        let query = vec![0.5f32; 8];
        let f1 = AdaptiveEfModel::extract_features(&query, 10_000);
        let f2 = AdaptiveEfModel::extract_features(&query, 10_000);
        assert_eq!(f1, f2);
    }

    #[test]
    fn feature_extraction_normalized() {
        let query = vec![0.5f32; 8];
        let f = AdaptiveEfModel::extract_features(&query, 10_000);
        for &v in &f {
            assert!(v >= 0.0 && v <= 1.0, "feature {v} not in [0, 1]");
        }
    }

    #[test]
    fn empty_query_features() {
        let f = AdaptiveEfModel::extract_features(&[], 1000);
        assert_eq!(f, [0.0; 4]);
    }

    #[test]
    fn serialization_roundtrip() {
        let m = AdaptiveEfModel::new(64, 10, 200);
        let json = m.to_json();
        let m2 = AdaptiveEfModel::from_json(&json).expect("should deserialize");
        assert_eq!(m.default_ef, m2.default_ef);
        assert_eq!(m.min_ef, m2.min_ef);
        assert_eq!(m.max_ef, m2.max_ef);
        assert_eq!(m.trained, m2.trained);
    }
}
