//! Consciousness analysis pipeline for gravitational wave backgrounds.
//!
//! Computes IIT Phi, causal emergence, and SVD emergence for each GW source
//! model, then runs null hypothesis testing against the SMBH merger baseline.

use ruvector_consciousness::emergence::CausalEmergenceEngine;
use ruvector_consciousness::phi::auto_compute_phi;
use ruvector_consciousness::rsvd_emergence::{RsvdEmergenceEngine, RsvdEmergenceResult};
use ruvector_consciousness::traits::EmergenceEngine;
use ruvector_consciousness::types::{
    ComputeBudget, EmergenceResult, PhiResult, TransitionMatrix as ConsciousnessTPM,
};

use crate::data::{GWSpectrum, TransitionMatrix};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Complete analysis results across all GW source models.
pub struct AnalysisResults {
    /// Phi for each source model: (model_name, phi_result).
    pub model_phis: Vec<(String, PhiResult)>,
    /// Causal emergence for each model.
    pub model_emergence: Vec<(String, EmergenceResult)>,
    /// SVD emergence for each model.
    pub model_svd: Vec<(String, RsvdEmergenceResult)>,
    /// Null distribution Phi values (SMBH baseline).
    pub null_phis: Vec<f64>,
    /// z-score of best exotic model relative to null.
    pub z_score: f64,
    /// p-value: fraction of null samples exceeding best exotic Phi.
    pub p_value: f64,
    /// Sliding-window Phi spectrum for the SMBH model.
    pub smbh_phi_spectrum: Vec<(usize, usize, f64)>,
}

/// Convert our TPM to the consciousness crate's format.
fn to_consciousness_tpm(tpm: &TransitionMatrix) -> ConsciousnessTPM {
    ConsciousnessTPM::new(tpm.size, tpm.data.clone())
}

/// Extract a sub-TPM for a subset of frequency bins.
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

/// Run the complete analysis pipeline across all GW source models.
pub fn run_analysis(
    tpms: &[(&str, TransitionMatrix)],
    spectra: &[(&str, GWSpectrum)],
    n_bins: usize,
    null_samples: usize,
) -> AnalysisResults {
    let budget = ComputeBudget::default();
    let mut model_phis = Vec::new();
    let mut model_emergence = Vec::new();
    let mut model_svd = Vec::new();

    // 1. Compute Phi for each source model
    println!("\n--- Computing IIT Phi for each GW source model ---");
    for (name, tpm) in tpms {
        let ctpm = to_consciousness_tpm(tpm);
        match auto_compute_phi(&ctpm, None, &budget) {
            Ok(phi) => {
                println!(
                    "  {:20} Phi = {:.6}  (algorithm: {}, elapsed: {:?})",
                    name, phi.phi, phi.algorithm, phi.elapsed
                );
                model_phis.push((name.to_string(), phi));
            }
            Err(e) => {
                println!("  {:20} Phi computation failed: {}", name, e);
            }
        }
    }

    // 2. Causal emergence for each model
    println!("\n--- Causal Emergence Analysis ---");
    let emergence_engine = CausalEmergenceEngine::new(n_bins.min(16));
    for (name, tpm) in tpms {
        let ctpm = to_consciousness_tpm(tpm);
        match emergence_engine.compute_emergence(&ctpm, &budget) {
            Ok(em) => {
                println!(
                    "  {:20} EI_micro = {:.4}, det = {:.4}, deg = {:.4}",
                    name, em.ei_micro, em.determinism, em.degeneracy
                );
                model_emergence.push((name.to_string(), em));
            }
            Err(e) => {
                println!("  {:20} Emergence failed: {}", name, e);
            }
        }
    }

    // 3. SVD emergence for each model
    println!("\n--- SVD Emergence Analysis ---");
    let svd_engine = RsvdEmergenceEngine::default();
    for (name, tpm) in tpms {
        let ctpm = to_consciousness_tpm(tpm);
        match svd_engine.compute(&ctpm, &budget) {
            Ok(svd) => {
                println!(
                    "  {:20} eff_rank = {}/{}, entropy = {:.4}, emergence = {:.4}",
                    name, svd.effective_rank, n_bins, svd.spectral_entropy, svd.emergence_index
                );
                model_svd.push((name.to_string(), svd));
            }
            Err(e) => {
                println!("  {:20} SVD emergence failed: {}", name, e);
            }
        }
    }

    // 4. Sliding window Phi spectrum for SMBH model
    let mut smbh_phi_spectrum = Vec::new();
    if let Some((_, smbh_tpm)) = tpms.iter().find(|(n, _)| *n == "smbh") {
        let window = (n_bins / 4).max(3).min(6);
        println!("\n--- Phi Spectrum (window={}) for SMBH model ---", window);
        for start in 0..=(n_bins.saturating_sub(window)) {
            let bins: Vec<usize> = (start..start + window).collect();
            let sub = extract_sub_tpm(smbh_tpm, &bins);
            if let Ok(phi) = auto_compute_phi(&sub, None, &budget) {
                smbh_phi_spectrum.push((start, start + window, phi.phi));
            }
        }
    }

    // 5. Null hypothesis testing
    // Generate null TPMs from SMBH model with noise, compare to best exotic Phi
    let best_exotic_phi = model_phis
        .iter()
        .filter(|(n, _)| n != "smbh")
        .map(|(_, p)| p.phi)
        .fold(0.0f64, f64::max);

    println!(
        "\n--- Null Hypothesis Testing ({} realizations) ---",
        null_samples
    );
    println!(
        "  Testing: best exotic Phi = {:.6} vs SMBH null",
        best_exotic_phi
    );

    let smbh_spec = spectra.iter().find(|(n, _)| *n == "smbh").map(|(_, s)| s);

    let mut null_phis = Vec::with_capacity(null_samples);

    if let Some(spec) = smbh_spec {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for i in 0..null_samples {
            let null_tpm = crate::data::generate_null_tpm(spec, n_bins, 1.0, &mut rng);
            let null_ctpm = to_consciousness_tpm(&null_tpm);
            if let Ok(null_phi) = auto_compute_phi(&null_ctpm, None, &budget) {
                null_phis.push(null_phi.phi);
            }
            if (i + 1) % 25 == 0 {
                print!("  [{}/{}] ", i + 1, null_samples);
            }
        }
        println!();
    }

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
        (best_exotic_phi - null_mean) / null_std
    } else {
        0.0
    };
    let p_value = if null_phis.is_empty() {
        1.0
    } else {
        null_phis.iter().filter(|&&p| p >= best_exotic_phi).count() as f64 / null_phis.len() as f64
    };

    AnalysisResults {
        model_phis,
        model_emergence,
        model_svd,
        null_phis,
        z_score,
        p_value,
        smbh_phi_spectrum,
    }
}
