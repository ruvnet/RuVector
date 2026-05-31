//! # ruvector-drift — Semantic Drift Detection for Agent Memory
//!
//! Detects when the distribution of query (or memory) vectors shifts over time.
//! Three detection strategies are provided behind a common [`DriftDetector`] trait:
//!
//! | Variant | Algorithm | Cost | Captures |
//! |---------|-----------|------|---------|
//! | [`CentroidDrift`] | L2 centroid shift | O(W·D) | mean shift |
//! | [`MmdDrift`] | MMD with RBF kernel | O(S²·D) | full distributional shift |
//! | [`FrechetDrift`] | Diagonal Fréchet distance | O(W·D) | mean + variance shift |
//!
//! where W = window size, S = subsample size, D = vector dimension.
//!
//! All detectors maintain a **reference window** (first full window observed) and a
//! **current window** (sliding, most recent W vectors).  The drift score is 0.0 when
//! the two windows are statistically identical and grows with distributional distance.
//!
//! ## Quick start
//! ```rust
//! use ruvector_drift::{DriftDetector, CentroidDrift};
//!
//! let mut det = CentroidDrift::new(64, 200, 1.5);
//! // Warm up on stable reference data
//! for _ in 0..200 {
//!     let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
//!     det.observe(&v);
//! }
//! assert!(!det.is_drifted());
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod centroid;
pub mod frechet;
pub mod mmd;
pub mod spectral;

pub use centroid::CentroidDrift;
pub use frechet::FrechetDrift;
pub use mmd::MmdDrift;
pub use spectral::{
    EvictionPlan, EvictionPolicy, LruEviction, MemoryEntry, RandomEviction, SpectralEviction,
};

/// Observation from a single [`DriftDetector::observe`] call.
#[derive(Debug, Clone)]
pub struct DriftObservation {
    /// Monotonically increasing count of vectors seen so far.
    pub observations: usize,
    /// Current drift score (0.0 = stable, higher = more drift).
    pub score: f64,
    /// Whether `score` exceeds the detector's configured threshold.
    pub is_drifted: bool,
}

/// Core trait shared by all drift detectors in this crate.
pub trait DriftDetector {
    /// Feed one new vector into the detector.
    fn observe(&mut self, vector: &[f32]) -> DriftObservation;

    /// Return the latest drift score without adding a new observation.
    fn score(&self) -> f64;

    /// Return `true` if the latest score exceeds the configured threshold.
    fn is_drifted(&self) -> bool;

    /// Human-readable detector name, used in benchmark output.
    fn name(&self) -> &str;

    /// Total vectors observed so far.
    fn observations(&self) -> usize;
}

/// Compute the squared L2 distance between two equal-length slices.
#[inline]
pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum()
}

/// Compute the centroid of a collection of vectors.
pub(crate) fn centroid(vecs: &[Vec<f32>], dim: usize) -> Vec<f64> {
    let n = vecs.len();
    if n == 0 {
        return vec![0.0; dim];
    }
    let mut c = vec![0.0f64; dim];
    for v in vecs {
        for (ci, &vi) in c.iter_mut().zip(v.iter()) {
            *ci += vi as f64;
        }
    }
    for ci in c.iter_mut() {
        *ci /= n as f64;
    }
    c
}
