//! Semantic drift detection for live vector indexes.
//!
//! When an embedding model is updated, data distribution shifts, or an
//! agent memory store accumulates off-distribution vectors, the search
//! quality of an ANN index degrades silently.  This crate provides three
//! lightweight statistical detectors that can run alongside any RuVector
//! index and trigger reindexing before recall drops below acceptable limits.
//!
//! # Variants
//!
//! | Variant            | Mechanism                          | State cost | Update cost |
//! |--------------------|------------------------------------|------------|-------------|
//! | `GlobalStats`      | Welford moment tracking            | O(D)       | O(D)        |
//! | `CentroidDrift`    | Online k-means centroid movement   | O(K·D)     | O(K·D)      |
//! | `NeighborhoodRecall`| Anchor k-NN recall regression    | O(A·n·D)   | O(n·D)      |
//!
//! # Quick start
//!
//! ```rust
//! use ruvector_drift_detect::{DriftDetector, GlobalStatsDriftDetector};
//!
//! let mut det = GlobalStatsDriftDetector::new(128);
//! for vec in generate_baseline() {
//!     det.observe(&vec);
//! }
//! det.snapshot();
//! for vec in generate_drifted() {
//!     det.observe(&vec);
//! }
//! println!("drift score: {:.4}", det.drift_score());
//! println!("drifted: {}", det.is_drifted(0.5));
//! # fn generate_baseline() -> Vec<Vec<f32>> { vec![] }
//! # fn generate_drifted() -> Vec<Vec<f32>> { vec![] }
//! ```

pub mod centroid_drift;
pub mod dataset;
pub mod global_stats;
pub mod neighborhood;

pub use centroid_drift::CentroidDriftDetector;
pub use global_stats::GlobalStatsDriftDetector;
pub use neighborhood::NeighborhoodDriftDetector;

/// Core trait for semantic drift detection over a vector index.
///
/// Implementors maintain a statistical snapshot of the distribution
/// observed before `snapshot()` is called and measure how much the
/// subsequently observed distribution has diverged.
pub trait DriftDetector {
    /// Feed one new vector into the online statistics.
    fn observe(&mut self, vec: &[f32]);

    /// Commit current statistics as the new baseline reference.
    /// Must be called at least once before `drift_score()` is meaningful.
    fn snapshot(&mut self);

    /// Return a drift score ≥ 0.0.  Higher = more divergence from baseline.
    /// The scale is variant-specific (see individual struct docs for thresholds).
    fn drift_score(&self) -> f64;

    /// Return true when `drift_score()` exceeds `threshold`.
    fn is_drifted(&self, threshold: f64) -> bool {
        self.drift_score() > threshold
    }

    /// Reset the baseline to the current statistics without erasing them.
    fn reset_baseline(&mut self);

    /// Number of vectors observed since the last `snapshot()`.
    fn post_snapshot_count(&self) -> usize;
}

/// Result returned by the benchmark binary for each detector variant.
#[derive(Debug, Clone)]
pub struct DriftReport {
    pub variant: &'static str,
    pub baseline_n: usize,
    pub post_snapshot_n: usize,
    pub drift_score: f64,
    pub is_drifted: bool,
    pub threshold: f64,
    /// Mean time per `observe()` call in nanoseconds.
    pub observe_ns_mean: u64,
    /// Time to compute `drift_score()` in nanoseconds.
    pub score_ns: u64,
}

impl DriftReport {
    pub fn print(&self) {
        println!(
            "  {:<22} drift={:.4}  drifted={:>5}  threshold={:.2}  \
             observe={:>5}ns/vec  score={:>6}ns  baseline_n={:>6}  post_n={:>6}",
            self.variant,
            self.drift_score,
            self.is_drifted,
            self.threshold,
            self.observe_ns_mean,
            self.score_ns,
            self.baseline_n,
            self.post_snapshot_n,
        );
    }
}
