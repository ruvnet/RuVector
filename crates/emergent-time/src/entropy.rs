//! Von Neumann / Shannon entropy helpers.
//!
//! Entropy is the monotone that several emergent-time constructions use as the
//! internal clock variable (the cold-atom toy universe in particular). All
//! logarithms are natural, so entropy is measured in nats.

use crate::complex_matrix::{hermitian_eigenvalues, CMatrix};
use crate::real_matrix::RealMatrix;

/// How negative an eigenvalue may be before we treat it as a genuine non-PSD
/// signal (rather than diagonalization round-off) in debug validation.
const NEG_TOL: f64 = -1e-9;

/// Debug-only sanity check that a spectrum is a valid probability distribution
/// (a density-matrix spectrum): it sums to ~1 and has no meaningfully-negative
/// eigenvalue. A failure here means a **non-PSD or non-normalized ρ** reached
/// the entropy routine — a real upstream bug, surfaced in dev only. No-op in
/// release builds, so it never alters production results.
#[inline]
fn debug_validate_spectrum(probs: &[f64]) {
    debug_assert!(
        probs.iter().all(|&p| p >= NEG_TOL),
        "entropy_from_spectrum: eigenvalue below {NEG_TOL:e} — ρ is not PSD: {probs:?}"
    );
    if !probs.is_empty() {
        let sum: f64 = probs.iter().sum();
        debug_assert!(
            (sum - 1.0).abs() < 1e-6,
            "entropy_from_spectrum: spectrum sums to {sum} (expected ~1) — ρ not normalized: {probs:?}"
        );
    }
}

/// Shannon / von Neumann entropy `S = -Σ p_k ln p_k` from a probability
/// spectrum (density-matrix eigenvalues).
///
/// Uses the standard von-Neumann clamp `if p > 0.0`: this skips exactly the
/// `0·ln0` term (the `lim_{p->0} p ln p = 0` convention) while keeping every
/// genuinely-positive probability, however small — so legitimate tiny
/// eigenvalues are *not* biased downward by an arbitrary epsilon. Round-off
/// negatives (`p <= 0`) contribute nothing.
///
/// In debug builds a [`debug_validate_spectrum`] check fires if the spectrum is
/// non-PSD or non-normalized, surfacing an upstream bad ρ rather than silently
/// masking it.
pub fn entropy_from_spectrum(probs: &[f64]) -> f64 {
    debug_validate_spectrum(probs);
    let mut s = 0.0;
    for &p in probs {
        if p > 0.0 {
            s -= p * p.ln();
        }
    }
    s
}

/// Von Neumann entropy of a real symmetric density matrix.
pub fn von_neumann_entropy_real(rho: &RealMatrix) -> f64 {
    let (eigs, _v) = rho.symmetric_eigen();
    entropy_from_spectrum(&eigs)
}

/// Von Neumann entropy of a complex Hermitian density matrix.
pub fn von_neumann_entropy_hermitian(rho: &CMatrix) -> f64 {
    entropy_from_spectrum(&hermitian_eigenvalues(rho))
}

/// Purity `Tr(ρ²) = Σ p_k²`. Equals 1 for a pure state, `1/d` for the maximally
/// mixed state of dimension `d`.
pub fn purity_from_spectrum(probs: &[f64]) -> f64 {
    probs.iter().map(|p| p * p).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_state_zero_entropy() {
        assert!(entropy_from_spectrum(&[1.0, 0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn maximally_mixed_is_ln_d() {
        let d = 4;
        let probs = vec![1.0 / d as f64; d];
        let s = entropy_from_spectrum(&probs);
        assert!((s - (d as f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn real_density_entropy() {
        let rho = RealMatrix::diag(&[0.5, 0.5]);
        assert!((von_neumann_entropy_real(&rho) - 2.0f64.ln()).abs() < 1e-10);
    }

    /// R1: a round-off-negative eigenvalue (`p = -1e-15`, well within `NEG_TOL`)
    /// contributes exactly 0 via the `p > 0` guard, so a `[0.5, 0.5, -1e-15]`
    /// spectrum still gives `ln 2`. The old `p > 1e-12` clamp would also skip it,
    /// but would additionally and wrongly bias any legitimate tiny-positive
    /// probability downward — which the new guard does not.
    #[test]
    fn roundoff_negative_eigenvalue_contributes_zero() {
        let s = entropy_from_spectrum(&[0.5, 0.5, -1e-15]);
        assert!((s - 2.0f64.ln()).abs() < 1e-12, "got {s}, expected ln2");
    }

    /// R1: a legitimately tiny but positive eigenvalue is *kept*, not silently
    /// dropped by an epsilon clamp. With p far below the old `1e-12` cutoff its
    /// `-p ln p` term is nonzero and must appear in the entropy.
    #[test]
    fn tiny_positive_probability_is_not_clamped_away() {
        let p = 1e-15;
        let s = entropy_from_spectrum(&[1.0 - p, p]);
        let expected = -(1.0 - p) * (1.0 - p).ln() - p * p.ln();
        assert!((s - expected).abs() < 1e-14, "got {s}, expected {expected}");
        assert!(s > 0.0, "tiny positive prob must contribute, got {s}");
    }

    /// R1: a clearly non-PSD spectrum (eigenvalue far below `NEG_TOL`) trips the
    /// debug validation in debug builds, surfacing an upstream non-PSD ρ.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "not PSD")]
    fn non_psd_spectrum_trips_debug_assert() {
        // Sums to 1 but has a genuinely-negative eigenvalue.
        let _ = entropy_from_spectrum(&[1.2, -0.2]);
    }

    /// R1: a non-normalized spectrum (does not sum to ~1) trips the debug
    /// validation in debug builds.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "not normalized")]
    fn non_normalized_spectrum_trips_debug_assert() {
        let _ = entropy_from_spectrum(&[0.5, 0.6]);
    }
}
