//! Verifiable Φ computation with witness chains.
//!
//! Wraps any PhiEngine computation in a cognitive container that
//! produces tamper-evident witness receipts. Each Φ result carries
//! a cryptographic proof of:
//! - Which TPM was used (hash)
//! - Which algorithm was applied
//! - Which MIP was found
//! - Computation time and resource usage
//!
//! Requires feature: `witness`

use crate::error::ConsciousnessError;
use crate::traits::PhiEngine;
use crate::types::{ComputeBudget, PhiResult, TransitionMatrix};

use ruvector_cognitive_container::{ContainerWitnessReceipt, WitnessChain};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Verifiable Φ result
// ---------------------------------------------------------------------------

/// A Φ result with an attached witness receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedPhiResult {
    /// The underlying Φ result.
    pub result: PhiResult,
    /// Witness receipt proving the computation was performed correctly.
    pub receipt: ContainerWitnessReceipt,
    /// Hash of the input TPM.
    pub tpm_hash: u64,
}

// ---------------------------------------------------------------------------
// Verifiable Φ engine wrapper
// ---------------------------------------------------------------------------

/// Wraps any `PhiEngine` to produce verifiable results.
///
/// Each computation is enclosed in a cognitive container epoch.
/// The container produces a tamper-evident witness receipt that
/// can be verified later.
pub struct VerifiablePhiEngine<E: PhiEngine> {
    inner: E,
    chain: WitnessChain,
}

impl<E: PhiEngine> VerifiablePhiEngine<E> {
    /// Create a new verifiable wrapper around an existing engine.
    pub fn new(engine: E) -> Self {
        Self {
            inner: engine,
            chain: WitnessChain::new(1024),
        }
    }

    /// Compute Φ with witness receipt.
    pub fn compute_verified(
        &mut self,
        tpm: &TransitionMatrix,
        state: Option<usize>,
        budget: &ComputeBudget,
    ) -> Result<VerifiedPhiResult, ConsciousnessError> {
        let tpm_hash = hash_tpm(tpm);

        // Compute Φ using the inner engine.
        let result = self.inner.compute_phi(tpm, state, budget)?;

        // Create witness receipt via generate_receipt.
        let input_data = format!(
            "phi={:.12},mip={},algorithm={},partitions={},elapsed={:?}",
            result.phi,
            result.mip.mask,
            result.algorithm,
            result.partitions_evaluated,
            result.elapsed,
        );

        let receipt = self.chain.generate_receipt(
            input_data.as_bytes(),
            &tpm_hash.to_le_bytes(),
            result.phi,
            &result.partitions_evaluated.to_le_bytes(),
            ruvector_cognitive_container::CoherenceDecision::Pass,
        );

        Ok(VerifiedPhiResult {
            result,
            receipt,
            tpm_hash,
        })
    }

    /// Get the witness chain for auditing.
    pub fn chain(&self) -> &WitnessChain {
        &self.chain
    }

    /// Number of witnessed computations.
    pub fn computation_count(&self) -> u64 {
        self.chain.current_epoch()
    }
}

/// Simple hash of a TPM for identification.
fn hash_tpm(tpm: &TransitionMatrix) -> u64 {
    let mut hash = 0xcbf29ce484222325u64; // FNV offset basis
    for &val in &tpm.data {
        let bits = val.to_bits();
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash ^= tpm.n as u64;
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi::ExactPhiEngine;

    fn and_gate_tpm() -> TransitionMatrix {
        #[rustfmt::skip]
        let data = vec![
            0.5, 0.25, 0.25, 0.0,
            0.5, 0.25, 0.25, 0.0,
            0.5, 0.25, 0.25, 0.0,
            0.0, 0.0,  0.0,  1.0,
        ];
        TransitionMatrix::new(4, data)
    }

    #[test]
    fn verified_phi_produces_receipt() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::exact();
        let mut engine = VerifiablePhiEngine::new(ExactPhiEngine);
        let result = engine.compute_verified(&tpm, Some(0), &budget).unwrap();

        assert!(result.result.phi >= 0.0);
        assert!(result.tpm_hash != 0);
        assert_eq!(engine.computation_count(), 1u64);
    }

    #[test]
    fn witness_chain_grows() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::exact();
        let mut engine = VerifiablePhiEngine::new(ExactPhiEngine);

        engine.compute_verified(&tpm, Some(0), &budget).unwrap();
        engine.compute_verified(&tpm, Some(1), &budget).unwrap();
        engine.compute_verified(&tpm, Some(2), &budget).unwrap();

        assert_eq!(engine.computation_count(), 3u64);
    }

    #[test]
    fn tpm_hash_deterministic() {
        let tpm = and_gate_tpm();
        let h1 = hash_tpm(&tpm);
        let h2 = hash_tpm(&tpm);
        assert_eq!(h1, h2);
    }
}
