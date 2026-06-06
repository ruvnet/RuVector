//! # ruvector-drift — Streaming Semantic Drift Detection for Agent Vector Memory
//!
//! Detects when the semantic distribution of an agent's vector memory has shifted —
//! enabling self-healing indexes, staleness eviction, and RAG safety guards.
//!
//! ## Variants
//!
//! | Variant          | Algorithm               | Memory      | Latency   |
//! |------------------|-------------------------|-------------|-----------|
//! | `MeanShiftDetector` | EMA mean distance     | O(D)        | O(D)      |
//! | `CusumDetector`     | CUSUM on projections  | O(D)        | O(D)      |
//! | `MmdRffDetector`    | MMD via RFF features  | O(D × R)    | O(D + R)  |
//!
//! All three implement [`DriftDetector`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cusum;
pub mod mean_shift;
pub mod mmd_rff;
pub mod stats;

pub use cusum::CusumDetector;
pub use mean_shift::MeanShiftDetector;
pub use mmd_rff::MmdRffDetector;
pub use stats::OnlineStats;

/// Core trait implemented by all drift detectors.
///
/// A detector ingests vectors one at a time via [`insert`], accumulates a
/// reference distribution during the warm-up phase, then continuously scores
/// divergence from that reference.  Callers gate on [`is_drifted`] and can
/// [`reset_reference`] when a controlled concept update occurs.
pub trait DriftDetector {
    /// Ingest one vector into the detector.
    fn insert(&mut self, vec: &[f32]);

    /// Scalar divergence from the reference distribution; 0.0 = no drift.
    fn drift_score(&self) -> f32;

    /// Whether the detector considers drift to have occurred.
    fn is_drifted(&self, threshold: f32) -> bool {
        self.drift_score() > threshold
    }

    /// Freeze the current distribution as the new reference baseline.
    fn reset_reference(&mut self);

    /// Number of vectors seen since last reset.
    fn count(&self) -> usize;

    /// Approximate heap bytes consumed by this detector.
    fn memory_bytes(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn gaussian(rng: &mut StdRng, dim: usize, mean: f64, std: f64) -> Vec<f32> {
        use std::f64::consts::PI;
        let mut out = Vec::with_capacity(dim);
        while out.len() < dim {
            let u1 = rng.gen::<f64>().max(1e-14);
            let u2 = rng.gen::<f64>();
            let r = (-2.0 * u1.ln()).sqrt() * std;
            let theta = 2.0 * PI * u2;
            out.push((mean + r * theta.cos()) as f32);
            if out.len() < dim {
                out.push((mean + r * theta.sin()) as f32);
            }
        }
        out.truncate(dim);
        out
    }

    fn run_detect<D: DriftDetector>(
        det: &mut D,
        rng: &mut StdRng,
        dim: usize,
        warm: usize,
        n_drift: usize,
        drift: f64,
        threshold: f32,
    ) -> Option<usize> {
        for _ in 0..warm {
            det.insert(&gaussian(rng, dim, 0.0, 1.0));
        }
        for i in 0..n_drift {
            det.insert(&gaussian(rng, dim, drift, 1.0));
            if det.is_drifted(threshold) {
                return Some(i + 1);
            }
        }
        None
    }

    #[test]
    fn mean_shift_detects_large_drift() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut det = MeanShiftDetector::new(64, 200, 0.05);
        let lag = run_detect(&mut det, &mut rng, 64, 200, 500, 3.0, 0.4);
        assert!(
            lag.is_some(),
            "MeanShift must detect drift=3.0 within 500 insertions"
        );
        assert!(
            lag.unwrap() <= 200,
            "detection lag must be ≤200; got {:?}",
            lag
        );
    }

    #[test]
    fn cusum_detects_moderate_drift() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut det = CusumDetector::new(64, 200, 1.0);
        let lag = run_detect(&mut det, &mut rng, 64, 200, 500, 2.0, 3.0);
        assert!(lag.is_some(), "CUSUM must detect drift=2.0");
        assert!(lag.unwrap() <= 300, "CUSUM lag must be ≤300; got {:?}", lag);
    }

    #[test]
    fn mmd_rff_detects_shift() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut det = MmdRffDetector::new(64, 128, 200, 1.0, 0.05, 42);
        let lag = run_detect(&mut det, &mut rng, 64, 200, 500, 2.5, 0.04);
        assert!(lag.is_some(), "MMD-RFF must detect drift=2.5");
    }

    #[test]
    fn mean_shift_drift_exceeds_nodrift_score() {
        // Verify drift signal >> natural noise floor.
        // EMA effective window n_eff = 1/alpha = 20. For D=64 iid N(0,1):
        //   expected no-drift L2 ≈ sqrt(D/n_eff) = sqrt(64/20) ≈ 1.79
        //   drift=4.0 per-dim pushes L2 far above that noise floor.
        let mut rng_nodrift = StdRng::seed_from_u64(4);
        let mut det_nodrift = MeanShiftDetector::new(64, 200, 0.05);
        for _ in 0..200 {
            det_nodrift.insert(&gaussian(&mut rng_nodrift, 64, 0.0, 1.0));
        }
        for _ in 0..100 {
            det_nodrift.insert(&gaussian(&mut rng_nodrift, 64, 0.0, 1.0));
        }
        let nodrift_score = det_nodrift.drift_score();

        let mut rng_drift = StdRng::seed_from_u64(4);
        let mut det_drift = MeanShiftDetector::new(64, 200, 0.05);
        for _ in 0..200 {
            det_drift.insert(&gaussian(&mut rng_drift, 64, 0.0, 1.0));
        }
        for _ in 0..100 {
            det_drift.insert(&gaussian(&mut rng_drift, 64, 4.0, 1.0));
        }
        let drift_score = det_drift.drift_score();

        // Drift score must be at least 3× larger than no-drift score.
        assert!(
            drift_score > nodrift_score * 3.0,
            "signal-to-noise too low: drift={drift_score:.2} nodrift={nodrift_score:.2}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut det = MeanShiftDetector::new(32, 100, 0.05);
        for _ in 0..100 {
            det.insert(&gaussian(&mut rng, 32, 0.0, 1.0));
        }
        for _ in 0..100 {
            det.insert(&gaussian(&mut rng, 32, 5.0, 1.0));
        }
        assert!(det.drift_score() > 0.1);
        det.reset_reference();
        assert_eq!(det.drift_score(), 0.0);
        assert_eq!(det.count(), 0);
    }

    #[test]
    fn memory_bytes_nonzero() {
        let det_ms = MeanShiftDetector::new(128, 100, 0.05);
        let det_cs = CusumDetector::new(128, 100, 1.0);
        let det_mmd = MmdRffDetector::new(128, 256, 100, 1.0, 0.05, 0);
        assert!(det_ms.memory_bytes() > 0);
        assert!(det_cs.memory_bytes() > 0);
        assert!(
            det_mmd.memory_bytes() >= 256 * 128 * 4,
            "RFF matrix should dominate"
        );
    }
}

/// Result of a single drift evaluation run.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Name of the detector variant.
    pub variant: String,
    /// Number of insertions in the reference phase.
    pub reference_count: usize,
    /// Number of insertions in the drift-observation phase.
    pub observation_count: usize,
    /// Drift score at end of reference phase (should be near 0).
    pub baseline_score: f32,
    /// Drift score when detection first triggered (0 if not triggered).
    pub trigger_score: f32,
    /// Insertions after drift injection until detection (None = not detected).
    pub detection_lag: Option<usize>,
    /// Final drift score after all observations.
    pub final_score: f32,
    /// Approximate memory used by the detector (bytes).
    pub memory_bytes: usize,
}
