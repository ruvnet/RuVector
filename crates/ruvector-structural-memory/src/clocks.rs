//! The three clocks under comparison. All are literal `emergent-time` types —
//! this module only fixes the [`StructuralMetric`] weights, it does not add
//! new clock math.

use emergent_time::{StructuralMetric, StructuralProperTime};

/// Candidate A: internal time = accumulated embedding movement only (`Δv`).
/// Cheapest structural clock: one L2 distance per step, no other signal
/// required.
pub fn structural_embedding_clock() -> StructuralProperTime {
    StructuralProperTime::new(StructuralMetric {
        w_embedding: 1.0,
        w_entropy: 0.0,
        w_graph: 0.0,
        w_coherence: 0.0,
        w_pred_error: 0.0,
        gate: 0.0,
    })
}

/// Candidate B: internal time = embedding movement (`Δv`) plus genuine
/// topic-uncertainty entropy (`ΔS`, see [`crate::scenario::build_snapshots`]).
/// `ΔG` and `ΔE` stay at zero weight: this harness has no honest graph or
/// prediction-error signal to feed them.
pub fn structural_full_clock() -> StructuralProperTime {
    StructuralProperTime::new(StructuralMetric {
        w_embedding: 1.0,
        w_entropy: 1.0,
        w_graph: 0.0,
        w_coherence: 0.0,
        w_pred_error: 0.0,
        gate: 0.0,
    })
}
