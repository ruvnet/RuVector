//! Consciousness analysis for ecosystem food webs.
//!
//! Computes IIT Phi for each ecosystem, measures resilience by removing
//! individual species, and performs causal emergence analysis.

use ruvector_consciousness::emergence::CausalEmergenceEngine;
use ruvector_consciousness::phi::auto_compute_phi;
use ruvector_consciousness::rsvd_emergence::RsvdEmergenceEngine;
use ruvector_consciousness::traits::EmergenceEngine;
use ruvector_consciousness::types::{
    ComputeBudget, EmergenceResult,
    TransitionMatrix as ConsciousnessTPM,
};
use ruvector_consciousness::rsvd_emergence::RsvdEmergenceResult;

use crate::data::Ecosystem;

/// Results for a single ecosystem analysis.
pub struct EcosystemResult {
    pub name: String,
    pub n_species: usize,
    pub full_phi: f64,
    pub algorithm: String,
    /// (species_index, species_name, phi_without, phi_contribution)
    pub species_contributions: Vec<(usize, String, f64, f64)>,
    pub emergence: EmergenceResult,
    pub svd_emergence: RsvdEmergenceResult,
    /// Trophic level colors for each species
    pub trophic_colors: Vec<String>,
    /// Species names
    pub species_names: Vec<String>,
}

/// Convert flat TPM data to consciousness crate format.
fn to_consciousness_tpm(data: &[f64], n: usize) -> ConsciousnessTPM {
    ConsciousnessTPM::new(n, data.to_vec())
}

/// Run the full analysis pipeline on all ecosystems.
pub fn run_ecosystem_analysis(ecosystems: &[Ecosystem]) -> Vec<EcosystemResult> {
    let budget = ComputeBudget::default();
    let mut results = Vec::with_capacity(ecosystems.len());

    for eco in ecosystems {
        println!("\n--- Analyzing: {} ({} species) ---", eco.name, eco.n());
        let n = eco.n();
        let ctpm = to_consciousness_tpm(&eco.tpm, n);

        // 1. Full system Phi
        let phi_result = auto_compute_phi(&ctpm, None, &budget)
            .expect("Failed to compute Phi");
        let full_phi = phi_result.phi;
        let algorithm = format!("{}", phi_result.algorithm);
        println!(
            "  Full Phi = {:.6}  (algorithm: {}, elapsed: {:?})",
            full_phi, algorithm, phi_result.elapsed
        );

        // 2. Species contribution analysis (resilience)
        println!("  Computing species contributions...");
        let mut contributions = Vec::with_capacity(n);
        for i in 0..n {
            let reduced_tpm = eco.tpm_without_species(i);
            let reduced_ctpm = to_consciousness_tpm(&reduced_tpm, n);
            let reduced_phi = match auto_compute_phi(&reduced_ctpm, None, &budget) {
                Ok(r) => r.phi,
                Err(_) => 0.0,
            };
            let contribution = full_phi - reduced_phi;
            contributions.push((
                i,
                eco.species[i].name.clone(),
                reduced_phi,
                contribution,
            ));
            println!(
                "    Remove {:20} -> Phi = {:.6}  (contribution: {:+.6})",
                eco.species[i].name, reduced_phi, contribution
            );
        }

        // Sort by contribution (highest first)
        contributions.sort_by(|a, b| {
            b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal)
        });

        // 3. Causal emergence
        println!("  Computing causal emergence...");
        let emergence_engine = CausalEmergenceEngine::new(n.min(16));
        let emergence = emergence_engine
            .compute_emergence(&ctpm, &budget)
            .expect("Failed to compute emergence");
        println!(
            "  EI_micro = {:.4}, determinism = {:.4}, degeneracy = {:.4}",
            emergence.ei_micro, emergence.determinism, emergence.degeneracy
        );
        println!(
            "  Causal emergence = {:.4} (EI_macro = {:.4})",
            emergence.causal_emergence, emergence.ei_macro
        );

        // 4. SVD emergence
        println!("  Computing SVD emergence...");
        let svd_engine = RsvdEmergenceEngine::default();
        let svd_emergence = svd_engine
            .compute(&ctpm, &budget)
            .expect("Failed to compute SVD emergence");
        println!(
            "  Effective rank = {}/{}, emergence index = {:.4}",
            svd_emergence.effective_rank, n, svd_emergence.emergence_index
        );

        let trophic_colors: Vec<String> = eco
            .species
            .iter()
            .map(|s| s.trophic_level.color().to_string())
            .collect();
        let species_names: Vec<String> = eco
            .species
            .iter()
            .map(|s| s.name.clone())
            .collect();

        results.push(EcosystemResult {
            name: eco.name.clone(),
            n_species: n,
            full_phi,
            algorithm,
            species_contributions: contributions,
            emergence,
            svd_emergence,
            trophic_colors,
            species_names,
        });
    }

    // Cross-ecosystem comparison
    println!("\n--- Cross-Ecosystem Comparison ---");
    for r in &results {
        let top = r
            .species_contributions
            .first()
            .map(|(_, name, _, c)| format!("{} ({:+.4})", name, c))
            .unwrap_or_default();
        println!(
            "  {:30} Phi = {:.6}  Top contributor: {}",
            r.name, r.full_phi, top
        );
    }

    results
}
