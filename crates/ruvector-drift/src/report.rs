use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Severity tier for a detected drift event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftSeverity {
    /// Drift score below threshold — memory distribution is stable.
    Stable,
    /// Score in [threshold, 2×threshold) — worth monitoring.
    Warning,
    /// Score ≥ 2×threshold — memory compaction or re-embedding recommended.
    Critical,
}

/// Complete report produced by a single `detect()` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Latest window id used in this report.
    pub window_id: u64,
    /// Normalised drift score in [0.0, …).
    pub drift_score: f32,
    /// Whether `drift_score` exceeds the configured threshold.
    pub is_drifted: bool,
    /// Severity classification.
    pub severity: DriftSeverity,
    /// Detector-specific sub-scores (centroid_l2, psi, coherence_delta, …).
    pub details: HashMap<String, f32>,
    /// Human-readable recommendation.
    pub recommendation: String,
}

impl DriftReport {
    pub(crate) fn stable(window_id: u64, score: f32) -> Self {
        Self {
            window_id,
            drift_score: score,
            is_drifted: false,
            severity: DriftSeverity::Stable,
            details: HashMap::new(),
            recommendation: "Memory distribution is stable.".to_string(),
        }
    }

    pub(crate) fn drifted(
        window_id: u64,
        score: f32,
        details: HashMap<String, f32>,
        critical: bool,
    ) -> Self {
        let severity = if critical {
            DriftSeverity::Critical
        } else {
            DriftSeverity::Warning
        };
        let recommendation = if critical {
            "Significant semantic drift detected. Consider re-embedding stale entries or triggering memory compaction.".to_string()
        } else {
            "Moderate drift detected. Monitor and consider selective memory refresh.".to_string()
        };
        Self {
            window_id,
            drift_score: score,
            is_drifted: true,
            severity,
            details,
            recommendation,
        }
    }

    /// No-op report when there are insufficient windows.
    pub(crate) fn insufficient(window_id: u64) -> Self {
        Self {
            window_id,
            drift_score: 0.0,
            is_drifted: false,
            severity: DriftSeverity::Stable,
            details: HashMap::new(),
            recommendation: "Insufficient windows — add at least two windows to detect drift."
                .to_string(),
        }
    }
}
