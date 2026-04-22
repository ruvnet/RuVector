//! Consciousness analysis for quantum circuits.
//!
//! Computes IIT Phi for each circuit's measurement TPM and compares
//! the Phi hierarchy with known entanglement measures.

use ruvector_consciousness::emergence::CausalEmergenceEngine;
use ruvector_consciousness::phi::auto_compute_phi;
use ruvector_consciousness::rsvd_emergence::RsvdEmergenceEngine;
use ruvector_consciousness::rsvd_emergence::RsvdEmergenceResult;
use ruvector_consciousness::traits::EmergenceEngine;
use ruvector_consciousness::types::{
    ComputeBudget, EmergenceResult, TransitionMatrix as ConsciousnessTPM,
};

use crate::data::QuantumCircuit;

/// Results for a single quantum circuit analysis.
pub struct CircuitResult {
    pub name: String,
    pub description: String,
    pub n_qubits: usize,
    pub tpm_size: usize,
    pub full_phi: f64,
    pub algorithm: String,
    pub emergence: EmergenceResult,
    pub svd_emergence: RsvdEmergenceResult,
}

/// Convert circuit TPM to consciousness crate format.
fn to_consciousness_tpm(tpm: &[f64], n: usize) -> ConsciousnessTPM {
    ConsciousnessTPM::new(n, tpm.to_vec())
}

/// Run analysis on all quantum circuits.
pub fn run_quantum_analysis(circuits: &[QuantumCircuit]) -> Vec<CircuitResult> {
    let budget = ComputeBudget::default();
    let mut results = Vec::with_capacity(circuits.len());

    for circuit in circuits {
        let dim = circuit.tpm_size();
        println!(
            "\n--- Analyzing: {} ({} qubits, {}x{} TPM) ---",
            circuit.name, circuit.n_qubits, dim, dim
        );

        let ctpm = to_consciousness_tpm(&circuit.tpm, dim);

        // 1. Compute Phi
        let phi_result = auto_compute_phi(&ctpm, None, &budget).expect("Failed to compute Phi");
        let full_phi = phi_result.phi;
        let algorithm = format!("{}", phi_result.algorithm);
        println!(
            "  Phi = {:.6}  (algorithm: {}, elapsed: {:?})",
            full_phi, algorithm, phi_result.elapsed
        );

        // 2. Causal emergence
        println!("  Computing causal emergence...");
        let emergence_engine = CausalEmergenceEngine::new(dim.min(16));
        let emergence = emergence_engine
            .compute_emergence(&ctpm, &budget)
            .expect("Failed to compute emergence");
        println!(
            "  EI_micro = {:.4}, causal_emergence = {:.4}",
            emergence.ei_micro, emergence.causal_emergence
        );

        // 3. SVD emergence
        println!("  Computing SVD emergence...");
        let svd_engine = RsvdEmergenceEngine::default();
        let svd_emergence = svd_engine
            .compute(&ctpm, &budget)
            .expect("Failed to compute SVD emergence");
        println!(
            "  Effective rank = {}/{}, emergence index = {:.4}",
            svd_emergence.effective_rank, dim, svd_emergence.emergence_index
        );

        results.push(CircuitResult {
            name: circuit.name.clone(),
            description: circuit.description.clone(),
            n_qubits: circuit.n_qubits,
            tpm_size: dim,
            full_phi,
            algorithm,
            emergence,
            svd_emergence,
        });
    }

    // Entanglement hierarchy comparison
    println!("\n--- Entanglement Hierarchy Comparison ---");
    println!("  Expected ordering: Product < W < Bell <= GHZ");
    println!("  Actual Phi values:");
    let mut sorted: Vec<&CircuitResult> = results.iter().collect();
    sorted.sort_by(|a, b| a.full_phi.partial_cmp(&b.full_phi).unwrap());
    for r in &sorted {
        println!("    {:25} Phi = {:.6}", r.name, r.full_phi);
    }

    // Check if ordering matches expectations
    let product_phi = results
        .iter()
        .find(|r| r.name == "Product State")
        .map(|r| r.full_phi)
        .unwrap_or(0.0);
    let w_phi = results
        .iter()
        .find(|r| r.name == "W State")
        .map(|r| r.full_phi)
        .unwrap_or(0.0);
    let bell_phi = results
        .iter()
        .find(|r| r.name == "Bell State")
        .map(|r| r.full_phi)
        .unwrap_or(0.0);
    let ghz_phi = results
        .iter()
        .find(|r| r.name == "GHZ State")
        .map(|r| r.full_phi)
        .unwrap_or(0.0);

    let order_ok = product_phi <= w_phi && w_phi <= bell_phi.max(ghz_phi);
    if order_ok {
        println!("\n  Phi ordering AGREES with entanglement hierarchy.");
    } else {
        println!("\n  Phi ordering DIFFERS from naive entanglement hierarchy.");
        println!("  This is expected: IIT Phi measures integrated information,");
        println!("  not entanglement per se. The two can diverge for certain states.");
    }

    // GHZ vs W emergence comparison
    println!("\n--- GHZ vs W: Emergence Structure ---");
    if let (Some(ghz), Some(w)) = (
        results.iter().find(|r| r.name == "GHZ State"),
        results.iter().find(|r| r.name == "W State"),
    ) {
        println!(
            "  GHZ: Phi={:.6}, emergence={:.4}, SVD rank={}/{}",
            ghz.full_phi,
            ghz.emergence.causal_emergence,
            ghz.svd_emergence.effective_rank,
            ghz.tpm_size
        );
        println!(
            "  W:   Phi={:.6}, emergence={:.4}, SVD rank={}/{}",
            w.full_phi, w.emergence.causal_emergence, w.svd_emergence.effective_rank, w.tpm_size
        );
        if ghz.emergence.causal_emergence > w.emergence.causal_emergence {
            println!("  GHZ shows MORE causal emergence than W.");
        } else {
            println!("  W shows MORE causal emergence than GHZ.");
        }
    }

    results
}
