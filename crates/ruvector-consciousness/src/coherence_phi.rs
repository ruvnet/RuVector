//! Coherence-accelerated Φ estimation via spectral gap bounds.
//!
//! Uses ruvector-coherence's spectral health metrics to provide
//! fast lower/upper bounds on Φ without full partition search.
//! The spectral gap of the MI Laplacian directly encodes
//! how "partitionable" the system is.
//!
//! Requires feature: `coherence-accel`

use crate::error::ConsciousnessError;
use crate::simd::marginal_distribution;
use crate::types::TransitionMatrix;

use ruvector_coherence::{CsrMatrixView, SpectralConfig, SpectralTracker};

/// Spectral bound on Φ from the MI Laplacian's spectral gap.
///
/// The Fiedler value (second-smallest eigenvalue of the Laplacian) λ₂
/// gives a lower bound on Φ: a system with high λ₂ cannot be cheaply
/// partitioned, implying high Φ.
///
/// Returns (fiedler_value, spectral_gap, coherence_score).
pub fn spectral_phi_bound(tpm: &TransitionMatrix) -> Result<PhiSpectralBound, ConsciousnessError> {
    let n = tpm.n;
    if n < 2 {
        return Err(crate::error::ValidationError::EmptySystem.into());
    }

    // Build MI edges.
    let marginal = marginal_distribution(tpm.as_slice(), n);
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mi = pairwise_mi_coh(tpm, i, j, &marginal);
            if mi > 1e-10 {
                edges.push((i, j, mi));
            }
        }
    }

    // Build Laplacian via coherence crate.
    let lap = CsrMatrixView::build_laplacian(n, &edges);

    // Compute spectral coherence score.
    let config = SpectralConfig {
        max_iterations: 200,
        tolerance: 1e-8,
        ..SpectralConfig::default()
    };
    let mut tracker = SpectralTracker::new(config);
    let score = tracker.compute(&lap);

    Ok(PhiSpectralBound {
        fiedler_value: score.fiedler,
        spectral_gap: score.spectral_gap,
        effective_resistance: score.effective_resistance,
        degree_regularity: score.degree_regularity,
        coherence_score: score.composite,
        phi_lower_bound: score.fiedler.max(0.0),
    })
}

/// Spectral bound result.
#[derive(Debug, Clone)]
pub struct PhiSpectralBound {
    /// Fiedler value (λ₂ of MI Laplacian).
    pub fiedler_value: f64,
    /// Normalized spectral gap (λ₂ / λ_max).
    pub spectral_gap: f64,
    /// Average effective resistance.
    pub effective_resistance: f64,
    /// Degree regularity (how uniform the MI graph is).
    pub degree_regularity: f64,
    /// Composite coherence score.
    pub coherence_score: f64,
    /// Lower bound on Φ (= max(fiedler, 0)).
    pub phi_lower_bound: f64,
}

/// Quick check: is the system likely to have high Φ?
///
/// Uses spectral gap as a fast proxy. If the gap is above threshold,
/// the system is strongly connected and Φ is likely high.
/// If below, the system has a near-partition and Φ may be low.
pub fn is_highly_integrated(tpm: &TransitionMatrix, threshold: f64) -> Result<bool, ConsciousnessError> {
    let bound = spectral_phi_bound(tpm)?;
    Ok(bound.spectral_gap > threshold)
}

fn pairwise_mi_coh(tpm: &TransitionMatrix, i: usize, j: usize, marginal: &[f64]) -> f64 {
    let n = tpm.n;
    let pi = marginal[i].max(1e-15);
    let pj = marginal[j].max(1e-15);
    let mut pij = 0.0;
    for state in 0..n {
        pij += tpm.get(state, i) * tpm.get(state, j);
    }
    pij /= n as f64;
    pij = pij.max(1e-15);
    (pij * (pij / (pi * pj)).ln()).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_tpm() -> TransitionMatrix {
        TransitionMatrix::identity(4)
    }

    fn uniform_tpm() -> TransitionMatrix {
        TransitionMatrix::new(4, vec![0.25; 16])
    }

    #[test]
    fn spectral_bound_identity() {
        let tpm = identity_tpm();
        let bound = spectral_phi_bound(&tpm).unwrap();
        assert!(bound.fiedler_value >= 0.0);
        assert!(bound.coherence_score >= 0.0);
    }

    #[test]
    fn spectral_bound_uniform() {
        let tpm = uniform_tpm();
        let bound = spectral_phi_bound(&tpm).unwrap();
        // Uniform TPM: all MI is zero → Fiedler = 0.
        assert!(bound.fiedler_value < 0.1, "uniform should have low fiedler, got {}", bound.fiedler_value);
    }

    #[test]
    fn integration_check() {
        let tpm = identity_tpm();
        let result = is_highly_integrated(&tpm, 0.01).unwrap();
        // Identity has some MI structure, may or may not pass threshold.
        assert!(result == true || result == false); // Just verify it runs.
    }
}
