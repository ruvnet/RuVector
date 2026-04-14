//! Predict when an HNSW index rebuild is needed.
//!
//! As vectors are inserted and deleted, the HNSW graph quality degrades.
//! This module uses an EML model to predict the recall loss from
//! observable graph statistics, allowing proactive rebuilds before
//! search quality drops below an acceptable threshold.
//!
//! # Features
//!
//! The model uses 5 input features:
//! 1. `inserts_since_rebuild` — normalized insert count
//! 2. `deletes_since_rebuild` — normalized delete count
//! 3. `total_entries` — current graph size (log-scaled)
//! 4. `graph_density` — average edges per node / max edges
//! 5. `avg_recent_recall` — measured recall from recent queries
//!
//! The output is predicted recall loss (0.0 = perfect, 1.0 = useless).

use eml_core::EmlModel;
use serde::{Deserialize, Serialize};

/// Default rebuild threshold: rebuild when predicted recall drops > 5%.
const DEFAULT_REBUILD_THRESHOLD: f64 = 0.05;

/// Minimum training samples before the model can be trained.
const MIN_TRAINING_SAMPLES: usize = 50;

/// Observable graph statistics used as model inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Number of vectors inserted since last rebuild.
    pub inserts_since_rebuild: usize,
    /// Number of vectors deleted since last rebuild.
    pub deletes_since_rebuild: usize,
    /// Total number of entries currently in the graph.
    pub total_entries: usize,
    /// Graph density: average edges per node / max possible edges.
    /// Ranges from 0.0 (empty) to 1.0 (fully connected).
    pub graph_density: f64,
    /// Average recall measured from recent ground-truth queries.
    /// Ranges from 0.0 (no correct results) to 1.0 (perfect recall).
    pub avg_recent_recall: f64,
}

impl GraphStats {
    /// Convert graph stats into a normalized feature vector for the model.
    pub fn to_features(&self) -> Vec<f64> {
        // Normalize features to roughly [0, 1] range.
        let total = (self.total_entries as f64).max(1.0);
        vec![
            // Insert ratio: fraction of entries that are new since rebuild.
            (self.inserts_since_rebuild as f64 / total).min(2.0),
            // Delete ratio: fraction of entries deleted since rebuild.
            (self.deletes_since_rebuild as f64 / total).min(2.0),
            // Log-scaled total entries (normalized by 1M).
            (total.ln() / (1_000_000.0f64).ln()).min(2.0),
            // Graph density (already 0-1).
            self.graph_density.clamp(0.0, 1.0),
            // Recent recall (already 0-1).
            self.avg_recent_recall.clamp(0.0, 1.0),
        ]
    }
}

/// A training observation: graph stats at a point in time + the actual
/// recall measured.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RebuildObservation {
    stats: GraphStats,
    actual_recall: f64,
}

/// Predicts when HNSW index rebuild is needed based on graph statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebuildPredictor {
    /// The EML model: 5 inputs -> 1 output (predicted recall loss).
    model: EmlModel,
    /// Threshold for triggering a rebuild.
    threshold: f64,
    /// Accumulated training observations (skipped in serde).
    #[serde(skip)]
    observations: Vec<RebuildObservation>,
}

impl RebuildPredictor {
    /// Create a new untrained rebuild predictor.
    pub fn new() -> Self {
        Self::with_threshold(DEFAULT_REBUILD_THRESHOLD)
    }

    /// Create a new predictor with a custom rebuild threshold.
    ///
    /// The threshold is the maximum acceptable predicted recall loss
    /// before recommending a rebuild (e.g., 0.05 = 5% recall drop).
    pub fn with_threshold(threshold: f64) -> Self {
        // 5 input features, 1 output head.
        let model = EmlModel::new(4, 5, 1);
        Self {
            model,
            threshold: threshold.clamp(0.001, 0.5),
            observations: Vec::new(),
        }
    }

    /// Whether the predictor model has been trained.
    pub fn is_trained(&self) -> bool {
        self.model.is_trained()
    }

    /// Get the current rebuild threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Number of training observations accumulated.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Predict whether the index should be rebuilt.
    ///
    /// Returns `true` if the predicted recall loss exceeds the threshold.
    /// If the model is not yet trained, falls back to a simple heuristic
    /// based on the insert/delete ratio.
    pub fn should_rebuild(&self, stats: &GraphStats) -> bool {
        let predicted_loss = self.predict_recall_loss(stats);
        predicted_loss > self.threshold
    }

    /// Predict the recall loss for the given graph stats.
    ///
    /// Returns a value between 0.0 (no loss) and 1.0 (total loss).
    pub fn predict_recall_loss(&self, stats: &GraphStats) -> f64 {
        if self.model.is_trained() {
            let features = stats.to_features();
            self.model.predict_primary(&features).clamp(0.0, 1.0)
        } else {
            self.heuristic_loss(stats)
        }
    }

    /// Record an observation: graph stats at a point in time, paired with
    /// the actual recall measured at that time.
    ///
    /// # Arguments
    /// - `stats`: Current graph statistics.
    /// - `actual_recall`: Measured recall (0.0 to 1.0).
    pub fn record(&mut self, stats: &GraphStats, actual_recall: f64) {
        let recall = actual_recall.clamp(0.0, 1.0);
        let loss = 1.0 - recall;
        let features = stats.to_features();
        self.model.record(&features, &[Some(loss)]);
        self.observations.push(RebuildObservation {
            stats: stats.clone(),
            actual_recall: recall,
        });
    }

    /// Train the model from accumulated observations.
    ///
    /// Returns `true` if training converged.
    pub fn train(&mut self) -> bool {
        if self.observations.len() < MIN_TRAINING_SAMPLES {
            return false;
        }
        self.model.train()
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    /// Simple heuristic-based recall loss estimate (used before the
    /// model is trained).
    fn heuristic_loss(&self, stats: &GraphStats) -> f64 {
        let total = (stats.total_entries as f64).max(1.0);

        // Churn ratio: how much the graph has changed since rebuild.
        let churn = (stats.inserts_since_rebuild + stats.deletes_since_rebuild) as f64 / total;

        // Base loss from churn (roughly: 10% churn = 1% loss).
        let churn_loss = (churn * 0.1).min(0.5);

        // Density penalty: low density suggests fragmentation.
        let density_loss = if stats.graph_density < 0.3 {
            (0.3 - stats.graph_density) * 0.2
        } else {
            0.0
        };

        // Direct recall signal if available.
        let recall_loss = 1.0 - stats.avg_recent_recall;

        // Weighted combination.
        let combined = 0.3 * churn_loss + 0.2 * density_loss + 0.5 * recall_loss;
        combined.clamp(0.0, 1.0)
    }
}

impl Default for RebuildPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_stats() -> GraphStats {
        GraphStats {
            inserts_since_rebuild: 100,
            deletes_since_rebuild: 10,
            total_entries: 10_000,
            graph_density: 0.75,
            avg_recent_recall: 0.98,
        }
    }

    fn degraded_stats() -> GraphStats {
        GraphStats {
            inserts_since_rebuild: 50_000,
            deletes_since_rebuild: 30_000,
            total_entries: 10_000,
            graph_density: 0.15,
            avg_recent_recall: 0.60,
        }
    }

    #[test]
    fn new_predictor_defaults() {
        let p = RebuildPredictor::new();
        assert!(!p.is_trained());
        assert_eq!(p.observation_count(), 0);
        assert!((p.threshold() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn custom_threshold() {
        let p = RebuildPredictor::with_threshold(0.10);
        assert!((p.threshold() - 0.10).abs() < 1e-10);
    }

    #[test]
    fn heuristic_healthy_no_rebuild() {
        let p = RebuildPredictor::new();
        let stats = healthy_stats();
        // High recall, low churn: should NOT recommend rebuild.
        assert!(
            !p.should_rebuild(&stats),
            "Healthy graph should not need rebuild"
        );
    }

    #[test]
    fn heuristic_degraded_recommends_rebuild() {
        let p = RebuildPredictor::new();
        let stats = degraded_stats();
        // Low recall, high churn, low density: SHOULD recommend rebuild.
        assert!(
            p.should_rebuild(&stats),
            "Degraded graph should need rebuild"
        );
    }

    #[test]
    fn record_increments_count() {
        let mut p = RebuildPredictor::new();
        p.record(&healthy_stats(), 0.98);
        assert_eq!(p.observation_count(), 1);
    }

    #[test]
    fn train_insufficient_data_returns_false() {
        let mut p = RebuildPredictor::new();
        for _ in 0..10 {
            p.record(&healthy_stats(), 0.95);
        }
        assert!(!p.train());
    }

    #[test]
    fn train_with_sufficient_data() {
        let mut p = RebuildPredictor::new();

        // Record healthy observations (low loss).
        for i in 0..30 {
            let stats = GraphStats {
                inserts_since_rebuild: 100 + i * 10,
                deletes_since_rebuild: 5 + i,
                total_entries: 10_000,
                graph_density: 0.7 + (i as f64) * 0.001,
                avg_recent_recall: 0.95 + (i as f64) * 0.001,
            };
            p.record(&stats, 0.95 + (i as f64) * 0.001);
        }

        // Record degraded observations (high loss).
        for i in 0..30 {
            let stats = GraphStats {
                inserts_since_rebuild: 5000 + i * 500,
                deletes_since_rebuild: 3000 + i * 300,
                total_entries: 10_000,
                graph_density: 0.2 - (i as f64) * 0.005,
                avg_recent_recall: 0.6 - (i as f64) * 0.01,
            };
            p.record(&stats, 0.6 - (i as f64) * 0.01);
        }

        // May or may not converge, but should not panic.
        let _ = p.train();
        // Prediction should still be finite.
        let loss = p.predict_recall_loss(&healthy_stats());
        assert!(loss.is_finite());
        assert!(loss >= 0.0 && loss <= 1.0);
    }

    #[test]
    fn graph_stats_to_features_length() {
        let stats = healthy_stats();
        let features = stats.to_features();
        assert_eq!(features.len(), 5);
        for &f in &features {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn graph_stats_to_features_bounded() {
        let stats = degraded_stats();
        let features = stats.to_features();
        for &f in &features {
            assert!(f >= 0.0 && f <= 2.0, "Feature out of range: {}", f);
        }
    }

    #[test]
    fn predict_recall_loss_is_bounded() {
        let p = RebuildPredictor::new();
        let loss = p.predict_recall_loss(&healthy_stats());
        assert!(loss >= 0.0 && loss <= 1.0);

        let loss2 = p.predict_recall_loss(&degraded_stats());
        assert!(loss2 >= 0.0 && loss2 <= 1.0);
    }

    #[test]
    fn serialization_roundtrip() {
        let p = RebuildPredictor::with_threshold(0.08);
        let json = serde_json::to_string(&p).unwrap();
        let p2: RebuildPredictor = serde_json::from_str(&json).unwrap();
        assert!((p.threshold() - p2.threshold()).abs() < 1e-10);
        assert_eq!(p.is_trained(), p2.is_trained());
    }
}
