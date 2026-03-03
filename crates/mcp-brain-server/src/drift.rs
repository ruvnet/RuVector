//! Embedding drift detection and monitoring
//!
//! Uses ruvector-delta-core for precise delta computation between
//! consecutive embeddings.

use crate::types::DriftReport;
use ruvector_delta_core::{Delta, VectorDelta};
use std::collections::HashMap;

/// Monitor embedding drift across knowledge clusters
pub struct DriftMonitor {
    /// Historical centroids per domain
    centroids: HashMap<String, Vec<Vec<f32>>>,
    window_size: usize,
    cv_threshold: f64,
}

impl DriftMonitor {
    pub fn new() -> Self {
        Self {
            centroids: HashMap::new(),
            window_size: 50,
            cv_threshold: 0.5,
        }
    }

    /// Record a new embedding for a domain
    pub fn record(&mut self, domain: &str, embedding: &[f32]) {
        let history = self.centroids.entry(domain.to_string()).or_default();
        history.push(embedding.to_vec());
        if history.len() > self.window_size * 2 {
            history.drain(..self.window_size);
        }
    }

    /// Compute drift report for a domain
    pub fn compute_drift(&self, domain: Option<&str>) -> DriftReport {
        let domain_key = domain.unwrap_or("global");
        let history = match self.centroids.get(domain_key) {
            Some(h) if h.len() >= 2 => h,
            _ => {
                return DriftReport {
                    domain: domain.map(|s| s.to_string()),
                    coefficient_of_variation: 0.0,
                    is_drifting: false,
                    delta_sparsity: 1.0,
                    trend: "insufficient_data".to_string(),
                    suggested_action: "collect more data".to_string(),
                    window_size: 0,
                };
            }
        };

        // Compute distances between consecutive centroids using VectorDelta
        let mut distances = Vec::new();
        let mut identity_count = 0usize;
        for pair in history.windows(2) {
            let delta = VectorDelta::compute(&pair[0], &pair[1]);
            let dist = delta.l2_norm() as f64;
            if delta.is_identity() {
                identity_count += 1;
            }
            distances.push(dist);
        }

        let mean_dist: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance: f64 = distances
            .iter()
            .map(|d| (d - mean_dist).powi(2))
            .sum::<f64>()
            / distances.len() as f64;
        let std_dev = variance.sqrt();
        let cv = if mean_dist > 1e-10 {
            std_dev / mean_dist
        } else {
            0.0
        };

        let is_drifting = cv > self.cv_threshold;
        let recent_trend = if distances.len() >= 3 {
            let recent_avg: f64 = distances[distances.len() - 3..].iter().sum::<f64>() / 3.0;
            if recent_avg > mean_dist * 1.3 {
                "increasing"
            } else if recent_avg < mean_dist * 0.7 {
                "decreasing"
            } else {
                "stable"
            }
        } else {
            "unknown"
        };

        // Use identity delta count for sparsity instead of threshold-based
        let sparsity = identity_count as f64 / distances.len() as f64;

        let suggested = if is_drifting {
            "investigate recent contributions for potential poisoning".to_string()
        } else if sparsity > 0.7 {
            "knowledge is stagnant, encourage new contributions".to_string()
        } else {
            "healthy drift levels".to_string()
        };

        DriftReport {
            domain: domain.map(|s| s.to_string()),
            coefficient_of_variation: cv,
            is_drifting,
            delta_sparsity: sparsity,
            trend: recent_trend.to_string(),
            suggested_action: suggested,
            window_size: history.len(),
        }
    }

    /// Compute embedding delta norms between two embeddings
    pub fn compute_embedding_delta(&self, old: &[f32], new: &[f32]) -> (f32, f32) {
        let delta = VectorDelta::compute(&old.to_vec(), &new.to_vec());
        (delta.l2_norm(), delta.l1_norm())
    }
}

impl Default for DriftMonitor {
    fn default() -> Self {
        Self::new()
    }
}
