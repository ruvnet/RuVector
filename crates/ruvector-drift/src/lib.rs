//! Semantic drift detection for agent memory vector collections.
//!
//! Monitors the statistical distribution of stored embeddings over time to
//! detect when an agent's memory has drifted from its current usage pattern.
//! Supports three detection strategies with a common [`DriftDetector`] trait.
//!
//! # Variants
//!
//! | Name | Strategy | Sensitivity | Cost |
//! |------|----------|------------|------|
//! | [`CentroidDrift`] | L2 shift of window centroids | Low | O(N·D) |
//! | [`PsiDrift`] | Population Stability Index on cosine buckets | Medium | O(N·D + B) |
//! | [`CoherenceDrift`] | PSI + intra-window coherence score | High | O(N²·D per window) |
//!
//! # Quick start
//!
//! ```rust
//! use ruvector_drift::{CentroidDrift, DriftDetector, DriftConfig};
//!
//! let cfg = DriftConfig::default();
//! let mut det = CentroidDrift::new(cfg);
//!
//! // baseline window
//! let baseline: Vec<Vec<f32>> = (0..200).map(|i| vec![(i % 8) as f32 * 0.1; 32]).collect();
//! det.add_window(0, &baseline);
//!
//! // drifted window
//! let drifted: Vec<Vec<f32>> = (0..200).map(|i| vec![10.0 + (i % 8) as f32 * 0.1; 32]).collect();
//! det.add_window(1, &drifted);
//!
//! let report = det.detect();
//! assert!(report.is_drifted, "large centroid shift must be flagged");
//! ```

pub mod detectors;
pub mod report;
pub mod window;

pub use detectors::{CentroidDrift, CoherenceDrift, PsiDrift};
pub use report::{DriftReport, DriftSeverity};
pub use window::VectorWindow;

use serde::{Deserialize, Serialize};

/// Configuration shared across all detector variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Maximum L2 centroid shift before flagging drift (CentroidDrift).
    pub centroid_threshold: f32,
    /// PSI threshold — industry convention: >0.25 = significant drift (PsiDrift).
    pub psi_threshold: f32,
    /// Coherence weight in combined score (CoherenceDrift). [0.0, 1.0]
    pub coherence_weight: f32,
    /// Number of cosine-similarity buckets for PSI histogram.
    pub psi_buckets: usize,
    /// Minimum vectors per window to trigger detection.
    pub min_window_size: usize,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            centroid_threshold: 0.5,
            psi_threshold: 0.25,
            coherence_weight: 0.4,
            psi_buckets: 10,
            min_window_size: 10,
        }
    }
}

/// Core trait implemented by all three drift detector variants.
pub trait DriftDetector {
    /// Feed a new time window of vectors into the detector.
    ///
    /// `window_id` must be monotonically increasing.
    fn add_window(&mut self, window_id: u64, vectors: &[Vec<f32>]);

    /// Compute a drift report comparing the latest window to the baseline.
    ///
    /// Returns `None` if fewer than two windows have been added.
    fn detect(&self) -> DriftReport;

    /// Returns `true` when the latest window exceeds the configured threshold.
    fn is_drifted(&self) -> bool {
        self.detect().is_drifted
    }

    /// Name of the detector variant, for reporting.
    fn name(&self) -> &'static str;
}

// ── shared math helpers ────────────────────────────────────────────────────

pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

pub(crate) fn centroid(vecs: &[Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return Vec::new();
    }
    let d = vecs[0].len();
    let n = vecs.len() as f32;
    let mut c = vec![0.0f32; d];
    for v in vecs {
        for (ci, vi) in c.iter_mut().zip(v.iter()) {
            *ci += vi;
        }
    }
    c.iter_mut().for_each(|x| *x /= n);
    c
}

/// Cosine similarity in [-1, 1].
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Population Stability Index between two probability distributions.
///
/// PSI = Σ (P_i - Q_i) * ln(P_i / Q_i).
/// Interpretation: <0.10 stable, 0.10–0.25 monitor, >0.25 significant drift.
pub(crate) fn psi(p: &[f32], q: &[f32]) -> f32 {
    const EPS: f32 = 1e-6;
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            let pi = pi.max(EPS);
            let qi = qi.max(EPS);
            (pi - qi) * (pi / qi).ln()
        })
        .sum::<f32>()
        .abs()
}

/// Convert a slice of cosine similarity values into a normalised histogram.
pub(crate) fn cosine_histogram(values: &[f32], buckets: usize) -> Vec<f32> {
    // values are in [-1, 1]; bucket into `buckets` equal-width bins
    let mut hist = vec![0usize; buckets];
    for &v in values {
        let idx = ((v + 1.0) / 2.0 * buckets as f32) as usize;
        let idx = idx.min(buckets - 1);
        hist[idx] += 1;
    }
    let n = values.len().max(1) as f32;
    hist.iter().map(|&c| c as f32 / n).collect()
}

/// Mean cosine similarity of all pairs within a window (intra-coherence).
/// O(N²) — only called on small windows (≤ ~500 vectors in PoC).
pub(crate) fn mean_pairwise_cosine(vecs: &[Vec<f32>]) -> f32 {
    if vecs.len() < 2 {
        return 1.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for i in 0..vecs.len() {
        for j in (i + 1)..vecs.len() {
            sum += cosine(&vecs[i], &vecs[j]);
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        sum / count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_zero_for_identical() {
        let vecs: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0, 0.0, 0.0]).collect();
        let c = centroid(&vecs);
        assert!((c[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_unit_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine(&a, &b)).abs() < 1e-5, "orthogonal → 0");
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5, "same → 1");
    }

    #[test]
    fn psi_identical_distributions_near_zero() {
        let p = vec![0.1, 0.2, 0.3, 0.25, 0.15];
        let q = p.clone();
        let v = psi(&p, &q);
        assert!(v < 0.01, "identical → PSI ≈ 0, got {v}");
    }

    #[test]
    fn psi_very_different_distributions_high() {
        let p = vec![0.9, 0.05, 0.025, 0.025];
        let q = vec![0.025, 0.025, 0.05, 0.9];
        let v = psi(&p, &q);
        assert!(v > 0.25, "very different → PSI > 0.25, got {v}");
    }
}
