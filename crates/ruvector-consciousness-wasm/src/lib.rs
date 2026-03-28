//! WASM bindings for ruvector-consciousness.
//!
//! Provides JavaScript-friendly APIs for:
//! - Φ (integrated information) computation
//! - Causal emergence analysis
//! - Quantum-inspired partition collapse
//!
//! ```javascript
//! import { WasmConsciousness } from 'ruvector-consciousness-wasm';
//!
//! const engine = new WasmConsciousness();
//! const result = engine.computePhi([0.5, 0.25, 0.25, 0.0, ...], 4, 0);
//! console.log('Φ =', result.phi);
//! ```

use wasm_bindgen::prelude::*;

use ruvector_consciousness::emergence::{CausalEmergenceEngine, effective_information};
use ruvector_consciousness::phi::{auto_compute_phi, ExactPhiEngine, SpectralPhiEngine, StochasticPhiEngine};
use ruvector_consciousness::geomip::GeoMipPhiEngine;
use ruvector_consciousness::collapse::QuantumCollapseEngine;
use ruvector_consciousness::rsvd_emergence::RsvdEmergenceEngine;
use ruvector_consciousness::traits::{PhiEngine, EmergenceEngine, ConsciousnessCollapse};
use ruvector_consciousness::types::{ComputeBudget, TransitionMatrix};

use serde::Serialize;
use std::time::Duration;

/// Initialize WASM module.
#[wasm_bindgen(start)]
pub fn init() {}

/// Get crate version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ---------------------------------------------------------------------------
// Result types for JS
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsPhiResult {
    phi: f64,
    mip_mask: u64,
    partitions_evaluated: u64,
    total_partitions: u64,
    algorithm: String,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct JsEmergenceResult {
    ei_micro: f64,
    ei_macro: f64,
    causal_emergence: f64,
    determinism: f64,
    degeneracy: f64,
    coarse_graining: Vec<usize>,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct JsRsvdEmergenceResult {
    singular_values: Vec<f64>,
    effective_rank: usize,
    spectral_entropy: f64,
    emergence_index: f64,
    reversibility: f64,
    elapsed_ms: f64,
}

// ---------------------------------------------------------------------------
// Main WASM API
// ---------------------------------------------------------------------------

/// Main consciousness computation engine for JavaScript.
#[wasm_bindgen]
pub struct WasmConsciousness {
    max_time_ms: f64,
    max_partitions: u64,
}

#[wasm_bindgen]
impl WasmConsciousness {
    /// Create a new engine with default settings.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            max_time_ms: 30000.0,
            max_partitions: 0,
        }
    }

    /// Set maximum computation time in milliseconds.
    #[wasm_bindgen(js_name = "setMaxTime")]
    pub fn set_max_time(&mut self, ms: f64) {
        self.max_time_ms = ms;
    }

    /// Set maximum partitions to evaluate (0 = unlimited).
    #[wasm_bindgen(js_name = "setMaxPartitions")]
    pub fn set_max_partitions(&mut self, max: u64) {
        self.max_partitions = max;
    }

    /// Compute Φ (integrated information) for a transition probability matrix.
    ///
    /// Auto-selects the best algorithm based on system size.
    ///
    /// @param tpm_data - Flat row-major TPM array
    /// @param n - Number of states
    /// @param state - Current state index
    #[wasm_bindgen(js_name = "computePhi")]
    pub fn compute_phi(
        &self,
        tpm_data: &[f64],
        n: usize,
        state: usize,
    ) -> Result<JsValue, JsError> {
        if tpm_data.len() != n * n {
            return Err(JsError::new(&format!(
                "TPM data length {} != n*n = {}",
                tpm_data.len(),
                n * n
            )));
        }

        let tpm = TransitionMatrix::new(n, tpm_data.to_vec());
        let budget = self.make_budget(1.0);

        let result = auto_compute_phi(&tpm, Some(state), &budget)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let js_result = JsPhiResult {
            phi: result.phi,
            mip_mask: result.mip.mask,
            partitions_evaluated: result.partitions_evaluated,
            total_partitions: result.total_partitions,
            algorithm: result.algorithm.to_string(),
            elapsed_ms: result.elapsed.as_secs_f64() * 1000.0,
        };

        serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Compute Φ using the exact algorithm (for small systems only, n ≤ 20).
    #[wasm_bindgen(js_name = "computePhiExact")]
    pub fn compute_phi_exact(
        &self,
        tpm_data: &[f64],
        n: usize,
        state: usize,
    ) -> Result<JsValue, JsError> {
        let tpm = TransitionMatrix::new(n, tpm_data.to_vec());
        let budget = self.make_budget(1.0);
        let result = ExactPhiEngine
            .compute_phi(&tpm, Some(state), &budget)
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.phi_to_js(&result)
    }

    /// Compute Φ using spectral approximation.
    #[wasm_bindgen(js_name = "computePhiSpectral")]
    pub fn compute_phi_spectral(
        &self,
        tpm_data: &[f64],
        n: usize,
        state: usize,
    ) -> Result<JsValue, JsError> {
        let tpm = TransitionMatrix::new(n, tpm_data.to_vec());
        let budget = self.make_budget(0.9);
        let result = SpectralPhiEngine::default()
            .compute_phi(&tpm, Some(state), &budget)
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.phi_to_js(&result)
    }

    /// Compute Φ using stochastic sampling.
    #[wasm_bindgen(js_name = "computePhiStochastic")]
    pub fn compute_phi_stochastic(
        &self,
        tpm_data: &[f64],
        n: usize,
        state: usize,
        samples: u64,
    ) -> Result<JsValue, JsError> {
        let tpm = TransitionMatrix::new(n, tpm_data.to_vec());
        let budget = self.make_budget(0.8);
        let result = StochasticPhiEngine::new(samples, 42)
            .compute_phi(&tpm, Some(state), &budget)
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.phi_to_js(&result)
    }

    /// Compute Φ using quantum-inspired collapse.
    #[wasm_bindgen(js_name = "computePhiCollapse")]
    pub fn compute_phi_collapse(
        &self,
        tpm_data: &[f64],
        n: usize,
        register_size: usize,
        iterations: usize,
    ) -> Result<JsValue, JsError> {
        let tpm = TransitionMatrix::new(n, tpm_data.to_vec());
        let engine = QuantumCollapseEngine::new(register_size);
        let result = engine
            .collapse_to_mip(&tpm, iterations, 42)
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.phi_to_js(&result)
    }

    /// Compute causal emergence for a TPM.
    #[wasm_bindgen(js_name = "computeEmergence")]
    pub fn compute_emergence(
        &self,
        tpm_data: &[f64],
        n: usize,
    ) -> Result<JsValue, JsError> {
        let tpm = TransitionMatrix::new(n, tpm_data.to_vec());
        let budget = self.make_budget(1.0);
        let engine = CausalEmergenceEngine::default();
        let result = engine
            .compute_emergence(&tpm, &budget)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let js_result = JsEmergenceResult {
            ei_micro: result.ei_micro,
            ei_macro: result.ei_macro,
            causal_emergence: result.causal_emergence,
            determinism: result.determinism,
            degeneracy: result.degeneracy,
            coarse_graining: result.coarse_graining,
            elapsed_ms: result.elapsed.as_secs_f64() * 1000.0,
        };

        serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Compute Φ using GeoMIP (hypercube BFS + automorphism pruning).
    ///
    /// 100-300x faster than exact for systems up to n=25.
    #[wasm_bindgen(js_name = "computePhiGeoMip")]
    pub fn compute_phi_geomip(
        &self,
        tpm_data: &[f64],
        n: usize,
        state: usize,
        prune: bool,
    ) -> Result<JsValue, JsError> {
        let tpm = TransitionMatrix::new(n, tpm_data.to_vec());
        let budget = self.make_budget(1.0);
        let engine = GeoMipPhiEngine::new(prune, 0);
        let result = engine
            .compute_phi(&tpm, Some(state), &budget)
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.phi_to_js(&result)
    }

    /// Compute SVD-based causal emergence metrics.
    ///
    /// Returns singular values, effective rank, spectral entropy,
    /// emergence index, and dynamical reversibility.
    #[wasm_bindgen(js_name = "computeRsvdEmergence")]
    pub fn compute_rsvd_emergence(
        &self,
        tpm_data: &[f64],
        n: usize,
        k: usize,
    ) -> Result<JsValue, JsError> {
        let tpm = TransitionMatrix::new(n, tpm_data.to_vec());
        let budget = self.make_budget(1.0);
        let engine = RsvdEmergenceEngine::new(k, 5, 42);
        let result = engine
            .compute(&tpm, &budget)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let js_result = JsRsvdEmergenceResult {
            singular_values: result.singular_values,
            effective_rank: result.effective_rank,
            spectral_entropy: result.spectral_entropy,
            emergence_index: result.emergence_index,
            reversibility: result.reversibility,
            elapsed_ms: result.elapsed.as_secs_f64() * 1000.0,
        };

        serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Compute effective information for a TPM.
    #[wasm_bindgen(js_name = "effectiveInformation")]
    pub fn effective_info(
        &self,
        tpm_data: &[f64],
        n: usize,
    ) -> Result<f64, JsError> {
        let tpm = TransitionMatrix::new(n, tpm_data.to_vec());
        effective_information(&tpm).map_err(|e| JsError::new(&e.to_string()))
    }

    fn make_budget(&self, approx_ratio: f64) -> ComputeBudget {
        ComputeBudget {
            max_time: Duration::from_secs_f64(self.max_time_ms / 1000.0),
            max_partitions: self.max_partitions,
            max_memory: 0,
            approximation_ratio: approx_ratio,
        }
    }

    fn phi_to_js(
        &self,
        result: &ruvector_consciousness::types::PhiResult,
    ) -> Result<JsValue, JsError> {
        let js_result = JsPhiResult {
            phi: result.phi,
            mip_mask: result.mip.mask,
            partitions_evaluated: result.partitions_evaluated,
            total_partitions: result.total_partitions,
            algorithm: result.algorithm.to_string(),
            elapsed_ms: result.elapsed.as_secs_f64() * 1000.0,
        };
        serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }
}
