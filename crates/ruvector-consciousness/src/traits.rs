//! Trait hierarchy for consciousness computation engines.
//!
//! All Φ computation algorithms implement [`PhiEngine`]. Extension traits
//! provide causal emergence and quantum-collapse integration.

use crate::error::ConsciousnessError;
use crate::types::{ComputeBudget, EmergenceResult, PhiAlgorithm, PhiResult, TransitionMatrix};

/// Core trait for integrated information (Φ) computation.
///
/// Implementations must be thread-safe (`Send + Sync`) so they can be shared
/// across parallel pipelines.
pub trait PhiEngine: Send + Sync {
    /// Compute Φ for the given transition probability matrix.
    ///
    /// The `state` parameter specifies the current system state as an index
    /// into the TPM. If `None`, computes Φ over the stationary distribution.
    fn compute_phi(
        &self,
        tpm: &TransitionMatrix,
        state: Option<usize>,
        budget: &ComputeBudget,
    ) -> Result<PhiResult, ConsciousnessError>;

    /// Return the algorithm identifier.
    fn algorithm(&self) -> PhiAlgorithm;

    /// Estimate computational cost without performing the computation.
    fn estimate_cost(&self, n: usize) -> u64;
}

/// Extension trait for causal emergence computation.
pub trait EmergenceEngine: Send + Sync {
    /// Compute causal emergence for a system at multiple scales.
    ///
    /// Finds the coarse-graining of the micro-level TPM that maximizes
    /// effective information, then computes the emergence metric.
    fn compute_emergence(
        &self,
        tpm: &TransitionMatrix,
        budget: &ComputeBudget,
    ) -> Result<EmergenceResult, ConsciousnessError>;

    /// Compute effective information for a given TPM.
    fn effective_information(&self, tpm: &TransitionMatrix) -> Result<f64, ConsciousnessError>;
}

/// Trait for quantum-inspired consciousness collapse.
///
/// Integrates with `ruqu-exotic` quantum collapse search to model
/// consciousness as a measurement-like collapse from superposition
/// of possible partitions.
pub trait ConsciousnessCollapse: Send + Sync {
    /// Collapse the partition superposition to find the MIP.
    ///
    /// Instead of exhaustive enumeration, models partitions as amplitudes
    /// and uses Grover-like iterations biased by information loss to
    /// probabilistically find the minimum information partition.
    fn collapse_to_mip(
        &self,
        tpm: &TransitionMatrix,
        iterations: usize,
        seed: u64,
    ) -> Result<PhiResult, ConsciousnessError>;
}
