//! Consciousness analysis pipeline: IIT Phi, causal emergence, null testing.

use ruvector_consciousness::emergence::CausalEmergenceEngine;
use ruvector_consciousness::phi::auto_compute_phi;
use ruvector_consciousness::rsvd_emergence::{RsvdEmergenceEngine, RsvdEmergenceResult};
use ruvector_consciousness::traits::EmergenceEngine;
use ruvector_consciousness::types::{
    ComputeBudget, EmergenceResult, PhiResult, TransitionMatrix as ConsciousnessTPM,
};

use crate::data::TransitionMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Full analysis results
pub struct AnalysisResults {
    pub full_phi: PhiResult,
    pub regional_phis: Vec<(String, PhiResult)>,
    pub phi_spectrum: Vec<(usize, usize, f64)>, // (start, end, phi)
    pub emergence: EmergenceResult,
    pub svd_emergence: RsvdEmergenceResult,
    pub null_phis: Vec<f64>,
    pub z_score: f64,
    pub p_value: f64,
}

/// Convert our TPM to the consciousness crate's format
fn to_consciousness_tpm(tpm: &TransitionMatrix) -> ConsciousnessTPM {
    ConsciousnessTPM::new(tpm.size, tpm.data.clone())
}

/// Extract a sub-TPM for a subset of bins
fn extract_sub_tpm(tpm: &TransitionMatrix, bins: &[usize]) -> ConsciousnessTPM {
    let n = bins.len();
    let mut sub = vec![0.0f64; n * n];
    for (si, &bi) in bins.iter().enumerate() {
        let row_sum: f64 = bins.iter().map(|&bj| tpm.data[bi * tpm.size + bj]).sum();
        for (sj, &bj) in bins.iter().enumerate() {
            sub[si * n + sj] = tpm.data[bi * tpm.size + bj] / row_sum.max(1e-30);
        }
    }
    ConsciousnessTPM::new(n, sub)
}

/// Run the complete analysis pipeline
pub fn run_analysis(
    tpm: &TransitionMatrix,
    ps: &crate::data::PowerSpectrum,
    n_bins: usize,
    alpha: f64,
    null_samples: usize,
) -> AnalysisResults {
    let budget = ComputeBudget::default();
    let ctpm = to_consciousness_tpm(tpm);

    // 1. Full system Phi
    println!("\n--- Computing IIT Phi (full system, n={}) ---", n_bins);
    let full_phi =
        auto_compute_phi(&ctpm, None, &budget).expect("Failed to compute Phi for full system");
    println!(
        "  Phi = {:.6}  (algorithm: {}, elapsed: {:?})",
        full_phi.phi, full_phi.algorithm, full_phi.elapsed
    );

    // 2. Regional sub-system Phi
    let mut regional_phis = Vec::new();
    let regions = define_regions(n_bins, &tpm.bin_centers);
    for (name, bins) in &regions {
        if bins.len() >= 2 {
            let sub = extract_sub_tpm(tpm, bins);
            match auto_compute_phi(&sub, None, &budget) {
                Ok(phi) => {
                    println!("  {} Phi = {:.6}  (bins {:?})", name, phi.phi, bins);
                    regional_phis.push((name.clone(), phi));
                }
                Err(e) => {
                    println!("  {} Phi computation failed: {}", name, e);
                }
            }
        }
    }

    // 3. Sliding window Phi spectrum
    let window = (n_bins / 4).max(3).min(8);
    let mut phi_spectrum = Vec::new();
    for start in 0..=(n_bins.saturating_sub(window)) {
        let bins: Vec<usize> = (start..start + window).collect();
        let sub = extract_sub_tpm(tpm, &bins);
        if let Ok(phi) = auto_compute_phi(&sub, None, &budget) {
            phi_spectrum.push((start, start + window, phi.phi));
        }
    }

    // 4. Causal emergence
    println!("\n--- Causal Emergence Analysis ---");
    let emergence_engine = CausalEmergenceEngine::new(n_bins.min(16));
    let emergence = emergence_engine
        .compute_emergence(&ctpm, &budget)
        .expect("Failed to compute causal emergence");
    println!(
        "  EI_micro = {:.4} bits, determinism = {:.4}, degeneracy = {:.4}",
        emergence.ei_micro, emergence.determinism, emergence.degeneracy
    );

    // 5. SVD emergence
    println!("\n--- SVD Emergence Analysis ---");
    let svd_engine = RsvdEmergenceEngine::default();
    let svd_emergence = svd_engine
        .compute(&ctpm, &budget)
        .expect("Failed to compute SVD emergence");
    println!(
        "  Effective rank = {}/{}, entropy = {:.4}, emergence = {:.4}",
        svd_emergence.effective_rank,
        n_bins,
        svd_emergence.spectral_entropy,
        svd_emergence.emergence_index
    );

    // 6. Null hypothesis testing
    println!(
        "\n--- Null Hypothesis Testing ({} GRF realizations) ---",
        null_samples
    );
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut null_phis = Vec::with_capacity(null_samples);
    for i in 0..null_samples {
        let null_tpm = crate::data::generate_null_tpm(ps, n_bins, alpha, &mut rng);
        let null_ctpm = to_consciousness_tpm(&null_tpm);
        if let Ok(null_phi) = auto_compute_phi(&null_ctpm, None, &budget) {
            null_phis.push(null_phi.phi);
        }
        if (i + 1) % 25 == 0 {
            print!("  [{}/{}] ", i + 1, null_samples);
        }
    }
    println!();

    // Compute statistics
    let null_mean = if null_phis.is_empty() {
        0.0
    } else {
        null_phis.iter().sum::<f64>() / null_phis.len() as f64
    };
    let null_std = if null_phis.len() > 1 {
        (null_phis
            .iter()
            .map(|&p| (p - null_mean).powi(2))
            .sum::<f64>()
            / (null_phis.len() as f64 - 1.0))
            .sqrt()
    } else {
        0.0
    };
    let z_score = if null_std > 1e-10 {
        (full_phi.phi - null_mean) / null_std
    } else {
        0.0
    };
    let p_value = if null_phis.is_empty() {
        1.0
    } else {
        null_phis.iter().filter(|&&p| p >= full_phi.phi).count() as f64 / null_phis.len() as f64
    };

    AnalysisResults {
        full_phi,
        regional_phis,
        phi_spectrum,
        emergence,
        svd_emergence,
        null_phis,
        z_score,
        p_value,
    }
}

/// Define physically meaningful regions based on CMB angular scales
fn define_regions(n_bins: usize, centers: &[f64]) -> Vec<(String, Vec<usize>)> {
    let mut regions = Vec::new();

    // Sachs-Wolfe plateau: l < 50
    let sw: Vec<usize> = centers
        .iter()
        .enumerate()
        .filter(|(_, &c)| c < 50.0)
        .map(|(i, _)| i)
        .collect();
    if !sw.is_empty() {
        regions.push(("Sachs-Wolfe".to_string(), sw));
    }

    // Acoustic peaks: l ~ 100-1000
    let acoustic: Vec<usize> = centers
        .iter()
        .enumerate()
        .filter(|(_, &c)| c >= 100.0 && c <= 1000.0)
        .map(|(i, _)| i)
        .collect();
    if !acoustic.is_empty() {
        regions.push(("Acoustic peaks".to_string(), acoustic));
    }

    // Damping tail: l > 1000
    let damping: Vec<usize> = centers
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 1000.0)
        .map(|(i, _)| i)
        .collect();
    if !damping.is_empty() {
        regions.push(("Damping tail".to_string(), damping));
    }

    // First acoustic peak region: l ~ 150-300
    let peak1: Vec<usize> = centers
        .iter()
        .enumerate()
        .filter(|(_, &c)| c >= 150.0 && c <= 300.0)
        .map(|(i, _)| i)
        .collect();
    if !peak1.is_empty() {
        regions.push(("1st peak".to_string(), peak1));
    }

    let _ = n_bins; // used for potential future expansion
    regions
}
