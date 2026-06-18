//! # RVM Coherence Monitor
//!
//! Real-time coherence scoring and Phi computation for the RVM
//! microhypervisor, as specified in ADR-139. Coherence is the
//! first-class resource-allocation signal: partitions with higher
//! coherence receive more CPU time and memory grants.
//!
//! ## Coherence Pipeline
//!
//! ```text
//! Sensor data --> Phi computation --> Score update --> Scheduler feedback
//! ```
//!
//! ## Modules
//!
//! - [`graph`]: Fixed-size adjacency structure for partition communication topology.
//! - [`scoring`]: Coherence score computation (internal/total weight ratio).
//! - [`pressure`]: Cut pressure and split/merge signal computation.
//! - [`mincut`]: Budgeted approximate minimum cut (Stoer-Wagner heuristic).
//! - [`adaptive`]: Adaptive recomputation frequency based on CPU load.
//!
//! ## Optional Features
//!
//! - `sched`: Enables direct feedback to the coherence-weighted scheduler

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::manual_flatten,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::if_not_else,
    clippy::new_without_default,
    clippy::explicit_iter_loop,
    clippy::collapsible_else_if,
    clippy::double_must_use,
    clippy::result_large_err
)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod adaptive;
pub mod bridge;
pub mod engine;
pub mod fennel;
pub mod graph;
pub mod mincut;
pub mod pressure;
pub mod scoring;

use rvm_types::{CoherenceScore, PartitionId, PhiValue};

// Re-exports for convenience.
pub use adaptive::AdaptiveCoherenceEngine;
pub use bridge::{CoherenceBackend, MinCutBackend};
pub use engine::{
    CoherenceDecision, CoherenceEngine, DefaultCoherenceEngine, SplitPlan, SplitSide,
    CRITICAL_PRESSURE_BP, MAX_SPLIT_CONDUCTANCE_BP,
};
pub use fennel::{FennelPlacer, DEFAULT_ALPHA_MILLI};
pub use graph::{CoherenceGraph, GraphError, NeighborIter};
pub use mincut::{MinCutBridge, MinCutResult};
pub use pressure::{
    MergeSignal, PressureResult, MERGE_COHERENCE_THRESHOLD_BP, SPLIT_THRESHOLD_BP,
};
pub use scoring::{PartitionCoherenceResult, compute_coherence_score, recompute_all_scores};

#[cfg(feature = "ruvector")]
pub use engine::RuVectorCoherenceEngine;

/// A raw sensor reading fed into the coherence pipeline.
#[derive(Debug, Clone, Copy)]
pub struct SensorReading {
    /// The partition this reading is associated with.
    pub partition: PartitionId,
    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// The raw Phi value computed from integrated information.
    pub phi: PhiValue,
}

/// Exponential moving average (EMA) filter for coherence scores.
///
/// Uses fixed-point arithmetic to avoid floating-point in `no_std`.
/// The smoothing factor alpha is expressed in basis points (0..10000).
#[derive(Debug, Clone, Copy)]
pub struct EmaFilter {
    /// Current smoothed value in basis points.
    current_bp: u32,
    /// Smoothing factor in basis points (higher = more responsive).
    alpha_bp: u16,
    /// Whether the filter has been initialized.
    initialized: bool,
}

impl EmaFilter {
    /// Create a new EMA filter with the given smoothing factor.
    ///
    /// `alpha_bp` is in basis points: 1000 = 10%, 5000 = 50%.
    /// Values above 10_000 are clamped to 10_000 (100%).
    #[must_use]
    pub const fn new(alpha_bp: u16) -> Self {
        // Clamp alpha_bp to the valid basis-point range [0, 10_000].
        let clamped = if alpha_bp > 10_000 { 10_000 } else { alpha_bp };
        Self {
            current_bp: 0,
            alpha_bp: clamped,
            initialized: false,
        }
    }

    /// Feed a new sample into the filter and return the smoothed value.
    pub fn update(&mut self, sample_bp: u16) -> CoherenceScore {
        if !self.initialized {
            self.current_bp = sample_bp as u32;
            self.initialized = true;
        } else {
            // EMA: new = alpha * sample + (1 - alpha) * old
            // All in basis points (10000 = 1.0)
            let alpha = self.alpha_bp as u32;
            let one_minus_alpha = 10_000u32.saturating_sub(alpha);
            self.current_bp =
                (alpha * sample_bp as u32 + one_minus_alpha * self.current_bp) / 10_000;
        }

        let clamped = if self.current_bp > 10_000 {
            10_000u16
        } else {
            self.current_bp as u16
        };

        CoherenceScore::from_basis_points(clamped)
    }

    /// Return the current smoothed score without feeding a new sample.
    #[must_use]
    pub fn current(&self) -> CoherenceScore {
        let val = if self.current_bp > 10_000 {
            10_000u16
        } else {
            self.current_bp as u16
        };
        CoherenceScore::from_basis_points(val)
    }
}

/// Convert a raw Phi value to a coherence score in basis points.
///
/// This is a stub mapping. The real implementation will apply a
/// calibrated transfer function derived from IIT theory.
#[must_use]
pub fn phi_to_coherence_bp(phi: PhiValue) -> u16 {
    // Stub: linear mapping from Phi fixed-point to basis points,
    // clamped to [0, 10000].
    let raw = phi.as_fixed();
    if raw >= 10_000 { 10_000 } else { raw as u16 }
}
