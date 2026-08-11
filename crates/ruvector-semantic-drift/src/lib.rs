//! Semantic drift detection for agent vector memory.
//!
//! Detects when the statistical distribution of vectors in an agent memory
//! collection has shifted significantly — triggering index maintenance,
//! memory compaction, or MCP drift notifications.
//!
//! Three detector variants with increasing sensitivity:
//! - [`WindowCentroidDetector`]: O(d) update, centroid-based
//! - [`ProjectionDriftDetector`]: O(k) update after projection, moment-based
//! - [`SentinelQueryDetector`]: O(s·n) refresh, result-overlap-based

pub mod centroid;
pub mod projection;
pub mod sentinel;

pub use centroid::WindowCentroidDetector;
pub use projection::ProjectionDriftDetector;
pub use sentinel::SentinelQueryDetector;

/// A discrete drift event emitted when a detector crosses its threshold.
#[derive(Debug, Clone)]
pub struct DriftEvent {
    /// Detector variant name.
    pub source: String,
    /// Normalized drift score in [0, 1]; threshold-crossing value.
    pub score: f32,
    /// Total vectors seen by this detector when the event fired.
    pub vectors_seen: u64,
    /// Recommended action for MCP consumers.
    pub action: DriftAction,
}

/// Suggested response to a [`DriftEvent`].
#[derive(Debug, Clone, PartialEq)]
pub enum DriftAction {
    /// Statistical drift is minor; log and observe.
    Observe,
    /// Significant shift; compact or re-cluster agent memory.
    Compact,
    /// Severe distribution change; rebuild the ANN index.
    Rebuild,
}

/// Core trait for all semantic drift detectors.
///
/// Implementations are expected to be cheap to update (called on every insert)
/// and cheap to query (called by MCP pollers at any frequency).
pub trait DriftDetector: Send + Sync {
    /// Feed one vector into the detector.
    fn update(&mut self, vector: &[f32]);

    /// Normalized drift score in [0, 1].
    ///
    /// 0.0 indicates no detectable drift from the reference window.
    /// Values above the detector's threshold trigger [`is_drifted`].
    fn drift_score(&self) -> f32;

    /// True when `drift_score()` exceeds the configured threshold.
    fn is_drifted(&self) -> bool;

    /// Human-readable variant name for benchmark tables and MCP tools.
    fn name(&self) -> &str;

    /// Approximate memory consumption in bytes.
    fn memory_bytes(&self) -> usize;

    /// Return a [`DriftEvent`] if drifted, otherwise `None`.
    fn poll_event(&self) -> Option<DriftEvent> {
        if !self.is_drifted() {
            return None;
        }
        let score = self.drift_score();
        let action = if score >= 0.75 {
            DriftAction::Rebuild
        } else if score >= 0.40 {
            DriftAction::Compact
        } else {
            DriftAction::Observe
        };
        Some(DriftEvent {
            source: self.name().to_owned(),
            score,
            vectors_seen: 0, // detectors may override via their own counter
            action,
        })
    }
}

/// Shared benchmark utilities: deterministic dataset generation.
pub mod dataset {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    /// Generate `n` independent Gaussian vectors with zero mean and unit variance.
    pub fn baseline(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
        (0..n)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

    /// Generate `n` drifted Gaussian vectors: mean shifted by `shift` in dimension 0,
    /// preserving unit variance. Models a sudden centroid displacement.
    pub fn drifted(n: usize, dim: usize, shift: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xdead_beef);
        let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
                v[0] += shift;
                v
            })
            .collect()
    }

    /// Cosine similarity of two equal-length f32 slices.
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na * nb)
    }

    /// Squared L2 distance between two equal-length f32 slices.
    pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // dim=128, n=500: expected IID normalised-L2 ≈ sqrt(2/500) ≈ 0.063.
    // Threshold 0.10 is ~1.6σ above mean — low FP, detects ≥3σ shifts.
    fn run_detector_acceptance(mut det: Box<dyn DriftDetector>, dim: usize) {
        let baseline = dataset::baseline(2_000, dim, 42);
        let drifted = dataset::drifted(2_000, dim, 5.0, 42);

        for v in &baseline {
            det.update(v);
        }
        assert!(
            !det.is_drifted(),
            "{}: false positive on baseline-only data",
            det.name()
        );

        for v in &drifted {
            det.update(v);
        }
        assert!(
            det.is_drifted(),
            "{}: failed to detect 5σ shift after 2000 drifted vectors",
            det.name()
        );
    }

    #[test]
    fn centroid_detects_shift() {
        // window=500, threshold=0.10; expected IID ≈ 0.063, 5σ ≈ 0.44.
        let det = WindowCentroidDetector::new(128, 500, 0.10);
        run_detector_acceptance(Box::new(det), 128);
    }

    #[test]
    fn projection_detects_shift() {
        // k=64 projections, window=500, threshold=0.12.
        let det = ProjectionDriftDetector::new(128, 64, 500, 0.12, 99);
        run_detector_acceptance(Box::new(det), 128);
    }

    #[test]
    fn sentinel_detects_shift() {
        // 10σ shift for sentinel KNN-distance detector.
        let mut det = SentinelQueryDetector::new(128, 8, 5, 300, 0.20, 77);
        let base = dataset::baseline(600, 128, 42);
        let drift = dataset::drifted(600, 128, 10.0, 42);
        for v in &base {
            det.update(v);
        }
        assert!(!det.is_drifted(), "sentinel: false positive on baseline");
        for v in &drift {
            det.update(v);
        }
        assert!(det.is_drifted(), "sentinel: failed to detect 10σ shift");
    }

    #[test]
    fn no_false_positive_stable_stream() {
        // dim=128, n=500 keeps expected normalised-L2 ≈ 0.063 well below 0.10.
        let mut det = Box::new(WindowCentroidDetector::new(128, 500, 0.10));
        let stable = dataset::baseline(4_000, 128, 123);
        for v in &stable {
            det.update(v);
        }
        assert!(
            !det.is_drifted(),
            "centroid: false positive on stable 4k stream"
        );
    }

    #[test]
    fn drift_event_shape() {
        // Large shift + low threshold to guarantee event fires.
        let mut det = ProjectionDriftDetector::new(64, 16, 100, 0.05, 7);
        let base = dataset::baseline(200, 64, 1);
        let drift = dataset::drifted(300, 64, 10.0, 1);
        for v in &base {
            det.update(v);
        }
        for v in &drift {
            det.update(v);
        }
        let ev = det.poll_event();
        assert!(ev.is_some(), "expected drift event after 10σ shift");
        let ev = ev.unwrap();
        assert!(!ev.source.is_empty());
        assert!(ev.score > 0.0);
        assert!(ev.action == DriftAction::Compact || ev.action == DriftAction::Rebuild);
    }
}
