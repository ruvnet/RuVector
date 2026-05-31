//! Semantic drift detection for agent memory and vector index health.
//!
//! Detects when the distribution of incoming vectors has shifted relative to a
//! reference window — a critical signal for long-running AI agents that need to
//! know when their memory is stale, when context has changed, or when the index
//! needs recompaction.
//!
//! Three variants are provided, each with different cost / accuracy tradeoffs:
//!
//! - [`centroid::CentroidDriftDetector`] — O(d) per observation, fast, detects
//!   mean shift but misses higher-order distributional changes.
//! - [`mmd::MmdDriftDetector`] — O(D·d) per observation, uses random Fourier
//!   features to approximate MMD, detects both mean and variance shifts.
//! - [`graph::GraphDriftDetector`] — O(n·k·d) per report, k-NN two-sample test,
//!   detects structural topology changes in the embedding neighborhood graph.

pub mod centroid;
pub mod graph;
pub mod mmd;

/// Score produced by a single observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftScore {
    /// Normalised drift magnitude ≥ 0.  Values > 1.0 typically indicate drift.
    pub score: f32,
    /// True when `score` exceeds the detector's configured threshold.
    pub alert: bool,
}

/// Summary produced by [`DriftDetector::report`].
#[derive(Debug, Clone)]
pub struct DriftReport {
    pub drift_detected: bool,
    /// Raw drift magnitude (same scale as [`DriftScore::score`]).
    pub magnitude: f32,
    /// Number of vectors in the current window.
    pub window_size: usize,
    /// Human-readable method name.
    pub method: &'static str,
}

/// Core abstraction for semantic drift detectors.
///
/// A detector maintains two windows:
/// - **Reference window** — established at construction or via [`DriftDetector::promote_current`].
/// - **Current window** — accumulates observations via [`DriftDetector::observe`].
///
/// Drift is measured as the statistical divergence between the two windows.
pub trait DriftDetector: Send + Sync {
    /// Record a new vector from the live distribution.
    ///
    /// Returns a per-observation drift score.  The score is incremental for
    /// centroid and MMD detectors; for graph it reflects the last full report.
    fn observe(&mut self, vec: &[f32]) -> DriftScore;

    /// Produce a full drift report from accumulated observations.
    fn report(&self) -> DriftReport;

    /// Clear the current window without touching the reference.
    fn reset_current(&mut self);

    /// Replace the reference window with the current window and clear current.
    ///
    /// Call this after the agent's context legitimately changes and the old
    /// reference is no longer meaningful.
    fn promote_current(&mut self);

    /// Return the dimensionality this detector was configured for.
    fn dims(&self) -> usize;

    /// Human-readable identifier for reporting.
    fn name(&self) -> &'static str;
}

/// Compute squared Euclidean distance between two equal-length slices.
#[inline]
pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Compute dot product.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
