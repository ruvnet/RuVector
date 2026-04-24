//! Consciousness analysis pipeline for climate teleconnections.

use ruvector_consciousness::emergence::CausalEmergenceEngine;
use ruvector_consciousness::phi::auto_compute_phi;
use ruvector_consciousness::rsvd_emergence::{RsvdEmergenceEngine, RsvdEmergenceResult};
use ruvector_consciousness::traits::EmergenceEngine;
use ruvector_consciousness::types::{
    ComputeBudget, EmergenceResult, PhiResult, TransitionMatrix as ConsciousnessTPM,
};

use crate::data::{self, ClimateCorrelations, TransitionMatrix};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Full analysis results for climate teleconnections.
pub struct AnalysisResults {
    /// Phi for the full 7-index neutral system.
    pub neutral_full_phi: PhiResult,
    /// Phi for the full 7-index El Nino active system.
    pub elnino_full_phi: PhiResult,
    /// Regional subset Phi values (neutral).
    pub neutral_regional_phis: Vec<(String, PhiResult)>,
    /// Regional subset Phi values (El Nino).
    pub elnino_regional_phis: Vec<(String, PhiResult)>,
    /// Causal emergence for neutral system.
    pub neutral_emergence: EmergenceResult,
    /// SVD emergence for neutral system.
    pub neutral_svd_emergence: RsvdEmergenceResult,
    /// Whether El Nino increases full-system Phi.
    pub elnino_increases_phi: bool,
    /// Whether the Pacific basin is the most integrated subsystem.
    pub pacific_most_integrated: bool,
    /// Monthly Phi values (seasonal cycle).
    pub monthly_phis: Vec<(String, f64)>,
    /// Null model Phi values.
    pub null_phis: Vec<f64>,
    /// Z-score of observed Phi vs null.
    pub z_score: f64,
    /// Empirical p-value.
    pub p_value: f64,
}

/// Convert our TPM to the consciousness crate's format.
fn to_consciousness_tpm(tpm: &TransitionMatrix) -> ConsciousnessTPM {
    ConsciousnessTPM::new(tpm.size, tpm.data.clone())
}

/// Run the complete analysis pipeline.
pub fn run_analysis(
    neutral_data: &ClimateCorrelations,
    neutral_tpm: &TransitionMatrix,
    _elnino_data: &ClimateCorrelations,
    elnino_tpm: &TransitionMatrix,
    null_samples: usize,
) -> AnalysisResults {
    let budget = ComputeBudget::default();

    // 1. Full system Phi -- neutral
    println!("\n--- Computing Phi: Neutral Climate System (7 indices) ---");
    let neutral_ctpm = to_consciousness_tpm(neutral_tpm);
    let neutral_full_phi = auto_compute_phi(&neutral_ctpm, None, &budget)
        .expect("Failed to compute Phi for neutral system");
    println!(
        "  Phi = {:.6}  (algorithm: {}, elapsed: {:?})",
        neutral_full_phi.phi, neutral_full_phi.algorithm, neutral_full_phi.elapsed
    );

    // 2. Full system Phi -- El Nino active
    println!("\n--- Computing Phi: El Nino Active (7 indices) ---");
    let elnino_ctpm = to_consciousness_tpm(elnino_tpm);
    let elnino_full_phi = auto_compute_phi(&elnino_ctpm, None, &budget)
        .expect("Failed to compute Phi for El Nino system");
    println!(
        "  Phi = {:.6}  (algorithm: {}, elapsed: {:?})",
        elnino_full_phi.phi, elnino_full_phi.algorithm, elnino_full_phi.elapsed
    );

    // 3. Regional Phi -- neutral
    println!("\n--- Computing Phi: Neutral Regional Subsets ---");
    let regions = data::all_regions();
    let mut neutral_regional_phis = Vec::new();
    for (name, indices) in &regions {
        if indices.len() >= 2 {
            let sub = data::extract_sub_tpm(neutral_tpm, indices);
            let sub_ctpm = to_consciousness_tpm(&sub);
            match auto_compute_phi(&sub_ctpm, None, &budget) {
                Ok(phi) => {
                    let idx_names: Vec<&str> =
                        indices.iter().map(|&i| data::INDEX_NAMES[i]).collect();
                    println!(
                        "  {} Phi = {:.6}  ({})",
                        name,
                        phi.phi,
                        idx_names.join(", ")
                    );
                    neutral_regional_phis.push((name.to_string(), phi));
                }
                Err(e) => {
                    println!("  {} Phi computation failed: {}", name, e);
                }
            }
        }
    }

    // 4. Regional Phi -- El Nino
    println!("\n--- Computing Phi: El Nino Regional Subsets ---");
    let mut elnino_regional_phis = Vec::new();
    for (name, indices) in &regions {
        if indices.len() >= 2 {
            let sub = data::extract_sub_tpm(elnino_tpm, indices);
            let sub_ctpm = to_consciousness_tpm(&sub);
            match auto_compute_phi(&sub_ctpm, None, &budget) {
                Ok(phi) => {
                    println!("  {} Phi = {:.6}", name, phi.phi);
                    elnino_regional_phis.push((name.to_string(), phi));
                }
                Err(e) => {
                    println!("  {} Phi computation failed: {}", name, e);
                }
            }
        }
    }

    // 5. Compare neutral vs El Nino
    let elnino_increases_phi = elnino_full_phi.phi > neutral_full_phi.phi;
    println!(
        "\n  El Nino Phi ({:.6}) {} Neutral Phi ({:.6})",
        elnino_full_phi.phi,
        if elnino_increases_phi { ">" } else { "<=" },
        neutral_full_phi.phi
    );

    // 6. Identify most integrated region
    let pacific_phi = neutral_regional_phis
        .iter()
        .find(|(n, _)| n == "Pacific")
        .map(|(_, p)| p.phi)
        .unwrap_or(0.0);
    let max_regional_phi = neutral_regional_phis
        .iter()
        .map(|(_, p)| p.phi)
        .fold(0.0f64, f64::max);
    let pacific_most_integrated =
        (pacific_phi - max_regional_phi).abs() < 1e-10 && pacific_phi > 0.0;

    // 7. Causal emergence
    println!("\n--- Causal Emergence Analysis (Neutral) ---");
    let emergence_engine = CausalEmergenceEngine::new(neutral_tpm.size.min(16));
    let neutral_emergence = emergence_engine
        .compute_emergence(&neutral_ctpm, &budget)
        .expect("Failed to compute causal emergence");
    println!(
        "  EI_micro = {:.4} bits, determinism = {:.4}, degeneracy = {:.4}",
        neutral_emergence.ei_micro, neutral_emergence.determinism, neutral_emergence.degeneracy
    );
    println!(
        "  Causal emergence = {:.4}, coarse-graining: {:?}",
        neutral_emergence.causal_emergence, neutral_emergence.coarse_graining
    );

    // 8. SVD emergence
    println!("\n--- SVD Emergence Analysis (Neutral) ---");
    let svd_engine = RsvdEmergenceEngine::default();
    let neutral_svd_emergence = svd_engine
        .compute(&neutral_ctpm, &budget)
        .expect("Failed to compute SVD emergence");
    println!(
        "  Effective rank = {}/{}, entropy = {:.4}, emergence = {:.4}",
        neutral_svd_emergence.effective_rank,
        neutral_tpm.size,
        neutral_svd_emergence.spectral_entropy,
        neutral_svd_emergence.emergence_index
    );

    // 9. Temporal analysis: monthly seasonal cycle
    println!("\n--- Temporal Analysis: Seasonal Phi Cycle ---");
    let monthly_tpms = data::generate_monthly_tpms(neutral_data);
    let mut monthly_phis = Vec::new();
    for (month, tpm) in &monthly_tpms {
        let ctpm = to_consciousness_tpm(tpm);
        match auto_compute_phi(&ctpm, None, &budget) {
            Ok(phi) => {
                println!("  {} Phi = {:.6}", month, phi.phi);
                monthly_phis.push((month.clone(), phi.phi));
            }
            Err(e) => {
                println!("  {} Phi failed: {}", month, e);
                monthly_phis.push((month.clone(), 0.0));
            }
        }
    }

    // 10. Null hypothesis testing
    println!(
        "\n--- Null Hypothesis Testing ({} shuffled correlations) ---",
        null_samples
    );
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut null_phis = Vec::with_capacity(null_samples);
    for i in 0..null_samples {
        let null_tpm = data::generate_null_tpm(neutral_data, &mut rng);
        let null_ctpm = to_consciousness_tpm(&null_tpm);
        if let Ok(null_phi) = auto_compute_phi(&null_ctpm, None, &budget) {
            null_phis.push(null_phi.phi);
        }
        if (i + 1) % 10 == 0 {
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
        (neutral_full_phi.phi - null_mean) / null_std
    } else {
        0.0
    };
    let p_value = if null_phis.is_empty() {
        1.0
    } else {
        null_phis
            .iter()
            .filter(|&&p| p >= neutral_full_phi.phi)
            .count() as f64
            / null_phis.len() as f64
    };

    AnalysisResults {
        neutral_full_phi,
        elnino_full_phi,
        neutral_regional_phis,
        elnino_regional_phis,
        neutral_emergence,
        neutral_svd_emergence,
        elnino_increases_phi,
        pacific_most_integrated,
        monthly_phis,
        null_phis,
        z_score,
        p_value,
    }
}
