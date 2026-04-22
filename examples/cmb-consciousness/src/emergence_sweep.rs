//! Emergence sweep: find the natural resolution of CMB physics.
//!
//! By varying the number of multipole bins, we discover at which
//! granularity the causal structure is most deterministic. The peak
//! of effective information (EI) reveals the "natural resolution" --
//! the scale at which the CMB power spectrum encodes the most causal
//! information per degree of freedom.

use ruvector_consciousness::emergence::CausalEmergenceEngine;
use ruvector_consciousness::rsvd_emergence::RsvdEmergenceEngine;
use ruvector_consciousness::traits::EmergenceEngine;
use ruvector_consciousness::types::ComputeBudget;

use crate::data::{power_spectrum_to_tpm, PowerSpectrum};

/// Bin counts to sweep over -- from coarse to fine resolution.
const BIN_COUNTS: [usize; 11] = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64];

/// A single point in the emergence sweep.
pub struct SweepPoint {
    pub n_bins: usize,
    pub ei: f64,
    pub determinism: f64,
    pub degeneracy: f64,
    pub effective_rank: usize,
    pub spectral_entropy: f64,
    pub emergence_index: f64,
}

/// Results of the full emergence sweep.
pub struct EmergenceSweepResults {
    pub sweeps: Vec<SweepPoint>,
    pub peak_ei_bins: usize,
    pub peak_emergence_bins: usize,
}

/// Run the emergence sweep: vary bin count and track causal emergence metrics.
///
/// For each bin count, we build a TPM from the power spectrum and compute
/// both causal emergence (EI, determinism, degeneracy) and SVD emergence
/// (effective rank, spectral entropy, emergence index).
pub fn run_emergence_sweep(ps: &PowerSpectrum) -> EmergenceSweepResults {
    let budget = ComputeBudget::default();
    let alpha = 1.0;
    let mut sweeps = Vec::with_capacity(BIN_COUNTS.len());

    println!("  Sweeping bin counts: {:?}", BIN_COUNTS);
    println!();
    println!(
        "  {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Bins", "EI", "Determ", "Degen", "EffRank", "SpectH", "EmergIdx"
    );
    println!("  {}", "-".repeat(65));

    for &n_bins in &BIN_COUNTS {
        let tpm = power_spectrum_to_tpm(ps, n_bins, alpha);
        let ctpm = ruvector_consciousness::types::TransitionMatrix::new(tpm.size, tpm.data.clone());

        // Causal emergence
        let emergence_engine = CausalEmergenceEngine::new(n_bins.min(16));
        let (ei, determinism, degeneracy) = match emergence_engine.compute_emergence(&ctpm, &budget)
        {
            Ok(result) => (result.ei_micro, result.determinism, result.degeneracy),
            Err(_) => (0.0, 0.0, 0.0),
        };

        // SVD emergence
        let svd_engine = RsvdEmergenceEngine::default();
        let (effective_rank, spectral_entropy, emergence_index) =
            match svd_engine.compute(&ctpm, &budget) {
                Ok(result) => (
                    result.effective_rank,
                    result.spectral_entropy,
                    result.emergence_index,
                ),
                Err(_) => (0, 0.0, 0.0),
            };

        println!(
            "  {:>5} {:>8.4} {:>8.4} {:>8.4} {:>8} {:>8.4} {:>10.4}",
            n_bins, ei, determinism, degeneracy, effective_rank, spectral_entropy, emergence_index
        );

        sweeps.push(SweepPoint {
            n_bins,
            ei,
            determinism,
            degeneracy,
            effective_rank,
            spectral_entropy,
            emergence_index,
        });
    }

    // Find peak EI
    let peak_ei_bins = sweeps
        .iter()
        .max_by(|a, b| a.ei.partial_cmp(&b.ei).unwrap_or(std::cmp::Ordering::Equal))
        .map(|s| s.n_bins)
        .unwrap_or(16);

    // Find peak emergence index
    let peak_emergence_bins = sweeps
        .iter()
        .max_by(|a, b| {
            a.emergence_index
                .partial_cmp(&b.emergence_index)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.n_bins)
        .unwrap_or(16);

    // Print summary
    println!();
    println!("  === Emergence Sweep Summary ===");
    println!("  Peak EI at {} bins", peak_ei_bins);
    println!("  Peak emergence index at {} bins", peak_emergence_bins);

    // Physical interpretation
    println!();
    println!("  Physical interpretation:");
    println!("  The effective information (EI) measures how precisely the current");
    println!("  state of the power spectrum determines its future evolution.");
    println!();

    if peak_ei_bins <= 12 {
        println!(
            "  Peak EI at {} bins suggests the CMB causal structure is",
            peak_ei_bins
        );
        println!("  best captured at coarse resolution -- the broad acoustic peak");
        println!("  pattern (Sachs-Wolfe plateau, 3 acoustic peaks, damping tail)");
        println!("  contains most of the deterministic physics.");
    } else if peak_ei_bins <= 32 {
        println!(
            "  Peak EI at {} bins suggests intermediate resolution best",
            peak_ei_bins
        );
        println!("  captures the CMB causal structure -- fine enough to resolve");
        println!("  individual acoustic peaks but not so fine that noise dominates.");
    } else {
        println!(
            "  Peak EI at {} bins suggests fine-grained resolution captures",
            peak_ei_bins
        );
        println!("  additional causal structure, possibly from higher-order acoustic");
        println!("  oscillations or the Silk damping cutoff.");
    }

    println!();
    if peak_emergence_bins != peak_ei_bins {
        println!(
            "  The emergence index peaks at {} bins, different from the",
            peak_emergence_bins
        );
        println!("  EI peak. This indicates that the SVD spectrum (dynamical");
        println!("  reversibility) reveals different structure than the causal");
        println!("  information measure.");
    } else {
        println!("  Both EI and emergence index peak at the same resolution,");
        println!(
            "  confirming {} bins as the natural scale of CMB physics.",
            peak_ei_bins
        );
    }

    EmergenceSweepResults {
        sweeps,
        peak_ei_bins,
        peak_emergence_bins,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::download_power_spectrum;

    #[test]
    fn test_sweep_produces_results() {
        let ps = download_power_spectrum();
        let results = run_emergence_sweep(&ps);
        assert_eq!(results.sweeps.len(), BIN_COUNTS.len());
        assert!(results.peak_ei_bins >= 4);
        assert!(results.peak_emergence_bins >= 4);
    }

    #[test]
    fn test_sweep_points_have_valid_values() {
        let ps = download_power_spectrum();
        let results = run_emergence_sweep(&ps);
        for point in &results.sweeps {
            assert!(point.ei >= 0.0, "EI should be non-negative");
            assert!(
                point.determinism >= 0.0,
                "Determinism should be non-negative"
            );
            assert!(
                point.emergence_index >= 0.0,
                "Emergence index should be non-negative"
            );
            assert!(point.n_bins >= 4, "Bin count should be at least 4");
        }
    }
}
