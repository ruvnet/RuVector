//! Consciousness analysis pipeline for gene regulatory networks.

use ruvector_consciousness::emergence::CausalEmergenceEngine;
use ruvector_consciousness::phi::auto_compute_phi;
use ruvector_consciousness::rsvd_emergence::{RsvdEmergenceEngine, RsvdEmergenceResult};
use ruvector_consciousness::traits::EmergenceEngine;
use ruvector_consciousness::types::{
    ComputeBudget, EmergenceResult, PhiResult, TransitionMatrix as ConsciousnessTPM,
};

use crate::data::{self, GeneNetwork, TransitionMatrix};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Full analysis results for the gene regulatory network.
pub struct AnalysisResults {
    /// Phi for the full 16-gene normal network.
    pub normal_full_phi: PhiResult,
    /// Phi for the full 16-gene cancer network.
    pub cancer_full_phi: PhiResult,
    /// Phi for each 4-gene module in the normal network.
    pub normal_module_phis: Vec<(String, PhiResult)>,
    /// Phi for each 4-gene module in the cancer network.
    pub cancer_module_phis: Vec<(String, PhiResult)>,
    /// Causal emergence for normal network.
    pub normal_emergence: EmergenceResult,
    /// SVD emergence for normal network.
    pub normal_svd_emergence: RsvdEmergenceResult,
    /// Whether modules have higher Phi than the full network (expected: yes).
    pub modules_more_integrated: bool,
    /// Whether cancer rewiring increases cross-module Phi.
    pub cancer_higher_cross_phi: bool,
    /// Null model Phi values for statistical testing.
    pub null_phis: Vec<f64>,
    /// Z-score of observed Phi vs null distribution.
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
    normal_net: &GeneNetwork,
    normal_tpm: &TransitionMatrix,
    _cancer_net: &GeneNetwork,
    cancer_tpm: &TransitionMatrix,
    null_samples: usize,
) -> AnalysisResults {
    let budget = ComputeBudget::default();

    // 1. Full system Phi -- normal
    println!("\n--- Computing Phi: Normal Network (full 16-gene) ---");
    let normal_ctpm = to_consciousness_tpm(normal_tpm);
    let normal_full_phi = auto_compute_phi(&normal_ctpm, None, &budget)
        .expect("Failed to compute Phi for normal network");
    println!(
        "  Phi = {:.6}  (algorithm: {}, elapsed: {:?})",
        normal_full_phi.phi, normal_full_phi.algorithm, normal_full_phi.elapsed
    );

    // 2. Full system Phi -- cancer
    println!("\n--- Computing Phi: Cancer Network (full 16-gene) ---");
    let cancer_ctpm = to_consciousness_tpm(cancer_tpm);
    let cancer_full_phi = auto_compute_phi(&cancer_ctpm, None, &budget)
        .expect("Failed to compute Phi for cancer network");
    println!(
        "  Phi = {:.6}  (algorithm: {}, elapsed: {:?})",
        cancer_full_phi.phi, cancer_full_phi.algorithm, cancer_full_phi.elapsed
    );

    // 3. Module-level Phi -- normal
    println!("\n--- Computing Phi: Normal Network Modules ---");
    let modules = data::all_modules();
    let mut normal_module_phis = Vec::new();
    for (name, genes) in &modules {
        let sub = data::extract_sub_tpm(normal_tpm, genes);
        let sub_ctpm = to_consciousness_tpm(&sub);
        match auto_compute_phi(&sub_ctpm, None, &budget) {
            Ok(phi) => {
                println!("  {} Phi = {:.6}  (genes {:?})", name, phi.phi, genes);
                normal_module_phis.push((name.to_string(), phi));
            }
            Err(e) => {
                println!("  {} Phi computation failed: {}", name, e);
            }
        }
    }

    // 4. Module-level Phi -- cancer
    println!("\n--- Computing Phi: Cancer Network Modules ---");
    let mut cancer_module_phis = Vec::new();
    for (name, genes) in &modules {
        let sub = data::extract_sub_tpm(cancer_tpm, genes);
        let sub_ctpm = to_consciousness_tpm(&sub);
        match auto_compute_phi(&sub_ctpm, None, &budget) {
            Ok(phi) => {
                println!("  {} Phi = {:.6}  (genes {:?})", name, phi.phi, genes);
                cancer_module_phis.push((name.to_string(), phi));
            }
            Err(e) => {
                println!("  {} Phi computation failed: {}", name, e);
            }
        }
    }

    // 5. Compare: modules vs full network
    let avg_module_phi = if normal_module_phis.is_empty() {
        0.0
    } else {
        normal_module_phis.iter().map(|(_, p)| p.phi).sum::<f64>() / normal_module_phis.len() as f64
    };
    let modules_more_integrated = avg_module_phi > normal_full_phi.phi;
    println!(
        "\n  Avg module Phi ({:.6}) {} full network Phi ({:.6})",
        avg_module_phi,
        if modules_more_integrated { ">" } else { "<=" },
        normal_full_phi.phi
    );

    // 6. Compare: cancer vs normal cross-module integration
    let cancer_higher_cross_phi = cancer_full_phi.phi > normal_full_phi.phi;
    println!(
        "  Cancer Phi ({:.6}) {} Normal Phi ({:.6})",
        cancer_full_phi.phi,
        if cancer_higher_cross_phi { ">" } else { "<=" },
        normal_full_phi.phi
    );

    // 7. Causal emergence -- normal network
    println!("\n--- Causal Emergence Analysis (Normal) ---");
    let emergence_engine = CausalEmergenceEngine::new(normal_tpm.size.min(16));
    let normal_emergence = emergence_engine
        .compute_emergence(&normal_ctpm, &budget)
        .expect("Failed to compute causal emergence");
    println!(
        "  EI_micro = {:.4} bits, determinism = {:.4}, degeneracy = {:.4}",
        normal_emergence.ei_micro, normal_emergence.determinism, normal_emergence.degeneracy
    );
    println!(
        "  Causal emergence = {:.4}, coarse-graining: {:?}",
        normal_emergence.causal_emergence, normal_emergence.coarse_graining
    );

    // 8. SVD emergence
    println!("\n--- SVD Emergence Analysis (Normal) ---");
    let svd_engine = RsvdEmergenceEngine::default();
    let normal_svd_emergence = svd_engine
        .compute(&normal_ctpm, &budget)
        .expect("Failed to compute SVD emergence");
    println!(
        "  Effective rank = {}/{}, entropy = {:.4}, emergence = {:.4}",
        normal_svd_emergence.effective_rank,
        normal_tpm.size,
        normal_svd_emergence.spectral_entropy,
        normal_svd_emergence.emergence_index
    );

    // 9. Null hypothesis testing
    println!(
        "\n--- Null Hypothesis Testing ({} randomized networks) ---",
        null_samples
    );
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut null_phis = Vec::with_capacity(null_samples);
    for i in 0..null_samples {
        let null_tpm = data::generate_null_tpm(normal_net, &mut rng);
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
        (normal_full_phi.phi - null_mean) / null_std
    } else {
        0.0
    };
    let p_value = if null_phis.is_empty() {
        1.0
    } else {
        null_phis
            .iter()
            .filter(|&&p| p >= normal_full_phi.phi)
            .count() as f64
            / null_phis.len() as f64
    };

    AnalysisResults {
        normal_full_phi,
        cancer_full_phi,
        normal_module_phis,
        cancer_module_phis,
        normal_emergence,
        normal_svd_emergence,
        modules_more_integrated,
        cancer_higher_cross_phi,
        null_phis,
        z_score,
        p_value,
    }
}
