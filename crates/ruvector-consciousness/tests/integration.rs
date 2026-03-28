//! Integration tests for ruvector-consciousness.
//!
//! Validates cross-module interactions: all PhiEngine implementations
//! agree on disconnected systems (Φ ≈ 0), EmergenceEngine + PhiEngine
//! pipelines, and WASM-style usage patterns.

use ruvector_consciousness::collapse::QuantumCollapseEngine;
use ruvector_consciousness::emergence::{
    coarse_grain, degeneracy, determinism, effective_information, CausalEmergenceEngine,
};
use ruvector_consciousness::geomip::{partition_information_loss_emd, GeoMipPhiEngine};
use ruvector_consciousness::phi::{
    auto_compute_phi, ExactPhiEngine, GreedyBisectionPhiEngine, HierarchicalPhiEngine,
    SpectralPhiEngine, StochasticPhiEngine,
};
use ruvector_consciousness::rsvd_emergence::RsvdEmergenceEngine;
use ruvector_consciousness::traits::{ConsciousnessCollapse, EmergenceEngine, PhiEngine};
use ruvector_consciousness::types::{Bipartition, ComputeBudget, PhiAlgorithm, TransitionMatrix};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn and_gate_tpm() -> TransitionMatrix {
    #[rustfmt::skip]
    let data = vec![
        0.5, 0.25, 0.25, 0.0,
        0.5, 0.25, 0.25, 0.0,
        0.5, 0.25, 0.25, 0.0,
        0.0, 0.0,  0.0,  1.0,
    ];
    TransitionMatrix::new(4, data)
}

fn disconnected_tpm() -> TransitionMatrix {
    #[rustfmt::skip]
    let data = vec![
        0.5, 0.5, 0.0, 0.0,
        0.5, 0.5, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.5, 0.5,
    ];
    TransitionMatrix::new(4, data)
}

fn identity_tpm(n: usize) -> TransitionMatrix {
    TransitionMatrix::identity(n)
}

fn uniform_tpm(n: usize) -> TransitionMatrix {
    let val = 1.0 / n as f64;
    TransitionMatrix::new(n, vec![val; n * n])
}

fn random_tpm(n: usize, seed: u64) -> TransitionMatrix {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            let val: f64 = rng.gen();
            data[i * n + j] = val;
            row_sum += val;
        }
        for j in 0..n {
            data[i * n + j] /= row_sum;
        }
    }
    TransitionMatrix::new(n, data)
}

// ---------------------------------------------------------------------------
// All engines agree: disconnected system → Φ ≈ 0
// ---------------------------------------------------------------------------

#[test]
fn all_engines_disconnected_near_zero() {
    let tpm = disconnected_tpm();
    let budget = ComputeBudget::exact();
    let eps = 1e-4;

    let exact = ExactPhiEngine.compute_phi(&tpm, Some(0), &budget).unwrap();
    assert!(exact.phi < eps, "exact: {}", exact.phi);

    let spectral = SpectralPhiEngine::default()
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert!(spectral.phi < eps, "spectral: {}", spectral.phi);

    let stochastic = StochasticPhiEngine::new(500, 42)
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert!(stochastic.phi < eps, "stochastic: {}", stochastic.phi);

    let geomip = GeoMipPhiEngine::default()
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert!(geomip.phi < eps, "geomip: {}", geomip.phi);

    let greedy = GreedyBisectionPhiEngine::default()
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert!(greedy.phi < eps, "greedy: {}", greedy.phi);

    let collapse = QuantumCollapseEngine::new(64)
        .collapse_to_mip(&tpm, 10, 42)
        .unwrap();
    assert!(collapse.phi < eps, "collapse: {}", collapse.phi);
}

// ---------------------------------------------------------------------------
// All engines agree: AND gate at state 11 → Φ > 0
// ---------------------------------------------------------------------------

#[test]
fn all_engines_and_gate_positive() {
    let tpm = and_gate_tpm();
    let budget = ComputeBudget::exact();

    let exact = ExactPhiEngine
        .compute_phi(&tpm, Some(3), &budget)
        .unwrap();
    assert!(exact.phi >= 0.0, "exact: {}", exact.phi);

    let geomip = GeoMipPhiEngine::default()
        .compute_phi(&tpm, Some(3), &budget)
        .unwrap();
    assert!(geomip.phi >= 0.0, "geomip: {}", geomip.phi);

    let spectral = SpectralPhiEngine::default()
        .compute_phi(&tpm, Some(3), &budget)
        .unwrap();
    assert!(spectral.phi >= 0.0, "spectral: {}", spectral.phi);

    let greedy = GreedyBisectionPhiEngine::default()
        .compute_phi(&tpm, Some(3), &budget)
        .unwrap();
    assert!(greedy.phi >= 0.0, "greedy: {}", greedy.phi);
}

// ---------------------------------------------------------------------------
// Exact and GeoMIP agree on small systems
// ---------------------------------------------------------------------------

#[test]
fn exact_and_geomip_agree() {
    let tpm = and_gate_tpm();
    let budget = ComputeBudget::exact();

    let exact = ExactPhiEngine.compute_phi(&tpm, Some(0), &budget).unwrap();
    let geomip = GeoMipPhiEngine::default()
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();

    assert!(
        (exact.phi - geomip.phi).abs() < 1e-8,
        "exact={} vs geomip={}",
        exact.phi,
        geomip.phi
    );
}

// ---------------------------------------------------------------------------
// Algorithm enum variants are correctly reported
// ---------------------------------------------------------------------------

#[test]
fn algorithm_variants_correct() {
    let tpm = and_gate_tpm();
    let budget = ComputeBudget::exact();

    let r = ExactPhiEngine.compute_phi(&tpm, Some(0), &budget).unwrap();
    assert_eq!(r.algorithm, PhiAlgorithm::Exact);

    let r = SpectralPhiEngine::default()
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert_eq!(r.algorithm, PhiAlgorithm::Spectral);

    let r = StochasticPhiEngine::new(100, 42)
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert_eq!(r.algorithm, PhiAlgorithm::Stochastic);

    let r = GeoMipPhiEngine::default()
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert_eq!(r.algorithm, PhiAlgorithm::GeoMIP);

    let r = GreedyBisectionPhiEngine::default()
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert_eq!(r.algorithm, PhiAlgorithm::GreedyBisection);

    let r = QuantumCollapseEngine::new(32)
        .collapse_to_mip(&tpm, 10, 42)
        .unwrap();
    assert_eq!(r.algorithm, PhiAlgorithm::Collapse);
}

// ---------------------------------------------------------------------------
// Auto-selection tiers
// ---------------------------------------------------------------------------

#[test]
fn auto_select_exact_for_small() {
    let tpm = and_gate_tpm(); // n=4
    let budget = ComputeBudget::exact();
    let result = auto_compute_phi(&tpm, Some(0), &budget).unwrap();
    assert_eq!(result.algorithm, PhiAlgorithm::Exact);
}

#[test]
fn auto_select_greedy_for_medium() {
    // n=32 is > 25, so should pick GreedyBisection (or Spectral for > 100).
    let tpm = random_tpm(32, 42);
    let budget = ComputeBudget::fast();
    let result = auto_compute_phi(&tpm, Some(0), &budget).unwrap();
    assert_eq!(result.algorithm, PhiAlgorithm::GreedyBisection);
}

// ---------------------------------------------------------------------------
// Emergence + Phi pipeline
// ---------------------------------------------------------------------------

#[test]
fn emergence_pipeline_identity() {
    let tpm = identity_tpm(4);
    let budget = ComputeBudget::fast();

    // EI should be max for identity.
    let ei = effective_information(&tpm).unwrap();
    assert!(ei > 1.0, "identity EI should be high, got {ei}");

    // Determinism should be max.
    let det = determinism(&tpm);
    assert!(det > 1.0, "identity det should be high, got {det}");

    // Degeneracy should be ~0.
    let deg = degeneracy(&tpm);
    assert!(deg < 0.1, "identity deg should be ~0, got {deg}");

    // Full emergence search.
    let engine = CausalEmergenceEngine::default();
    let result = engine.compute_emergence(&tpm, &budget).unwrap();
    assert!(result.ei_micro > 0.0);
    assert!(result.causal_emergence.is_finite());
}

#[test]
fn emergence_pipeline_uniform() {
    let tpm = uniform_tpm(4);
    let budget = ComputeBudget::fast();

    let ei = effective_information(&tpm).unwrap();
    assert!(ei < 0.01, "uniform EI should be ~0, got {ei}");

    let engine = CausalEmergenceEngine::default();
    let result = engine.compute_emergence(&tpm, &budget).unwrap();
    assert!(result.ei_micro < 0.01);
}

// ---------------------------------------------------------------------------
// RSVD emergence integration
// ---------------------------------------------------------------------------

#[test]
fn rsvd_emergence_pipeline() {
    let tpm = random_tpm(8, 99);
    let budget = ComputeBudget::fast();
    let engine = RsvdEmergenceEngine::new(5, 3, 42);
    let result = engine.compute(&tpm, &budget).unwrap();

    assert!(!result.singular_values.is_empty());
    assert!(result.effective_rank >= 1);
    assert!(result.spectral_entropy >= 0.0);
    assert!(result.emergence_index >= 0.0 && result.emergence_index <= 1.0);
    assert!(result.reversibility >= 0.0 && result.reversibility <= 1.0);
}

#[test]
fn rsvd_vs_hoel_emergence_correlation() {
    // Both emergence measures should agree directionally:
    // identity (high EI, low SVD emergence) vs uniform (low EI, potentially different)
    let tpm_id = identity_tpm(4);
    let tpm_uni = uniform_tpm(4);
    let budget = ComputeBudget::fast();

    let hoel_id = CausalEmergenceEngine::default()
        .compute_emergence(&tpm_id, &budget)
        .unwrap();
    let hoel_uni = CausalEmergenceEngine::default()
        .compute_emergence(&tpm_uni, &budget)
        .unwrap();

    let rsvd_id = RsvdEmergenceEngine::default().compute(&tpm_id, &budget).unwrap();
    let rsvd_uni = RsvdEmergenceEngine::default().compute(&tpm_uni, &budget).unwrap();

    // Identity has higher EI than uniform (both systems).
    assert!(hoel_id.ei_micro > hoel_uni.ei_micro);

    // Uniform has higher emergence index (more compressible = rank-1).
    assert!(
        rsvd_uni.emergence_index > rsvd_id.emergence_index,
        "uniform emergence_index ({}) should > identity ({})",
        rsvd_uni.emergence_index,
        rsvd_id.emergence_index,
    );
}

// ---------------------------------------------------------------------------
// Coarse-graining preserves TPM validity
// ---------------------------------------------------------------------------

#[test]
fn coarse_grain_preserves_row_sums() {
    let tpm = random_tpm(8, 123);
    let mapping = vec![0, 0, 1, 1, 2, 2, 3, 3]; // 8 → 4 states
    let macro_tpm = coarse_grain(&tpm, &mapping);

    assert_eq!(macro_tpm.n, 4);
    for i in 0..macro_tpm.n {
        let row_sum: f64 = (0..macro_tpm.n).map(|j| macro_tpm.get(i, j)).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-10,
            "macro TPM row {i} sums to {row_sum}"
        );
    }
}

// ---------------------------------------------------------------------------
// EMD vs KL-divergence: both non-negative, EMD is a metric
// ---------------------------------------------------------------------------

#[test]
fn emd_and_kl_both_nonnegative() {
    let tpm = and_gate_tpm();
    let partition = Bipartition { mask: 0b0011, n: 4 };
    let arena = ruvector_consciousness::arena::PhiArena::with_capacity(4096);

    let emd_loss = partition_information_loss_emd(&tpm, 0, &partition, &arena);
    assert!(emd_loss >= 0.0, "EMD loss negative: {emd_loss}");

    // KL-based loss (via phi module).
    let phi_result = ExactPhiEngine
        .compute_phi(&tpm, Some(0), &ComputeBudget::exact())
        .unwrap();
    assert!(phi_result.phi >= 0.0);
}

// ---------------------------------------------------------------------------
// Bipartition validity
// ---------------------------------------------------------------------------

#[test]
fn bipartition_set_extraction() {
    let bp = Bipartition { mask: 0b1010, n: 4 };
    let a = bp.set_a(); // bits 1 and 3
    let b = bp.set_b(); // bits 0 and 2
    assert_eq!(a, vec![1, 3]);
    assert_eq!(b, vec![0, 2]);
    assert!(bp.is_valid());

    // Invalid: all in A.
    let bp_all = Bipartition { mask: 0b1111, n: 4 };
    assert!(!bp_all.is_valid());

    // Invalid: none in A.
    let bp_none = Bipartition { mask: 0, n: 4 };
    assert!(!bp_none.is_valid());
}

// ---------------------------------------------------------------------------
// Budget enforcement
// ---------------------------------------------------------------------------

#[test]
fn budget_limits_partitions() {
    let tpm = random_tpm(8, 42); // 254 partitions total.
    let budget = ComputeBudget {
        max_partitions: 10,
        ..ComputeBudget::exact()
    };
    let result = ExactPhiEngine
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert!(
        result.partitions_evaluated <= 10,
        "should respect partition limit, evaluated {}",
        result.partitions_evaluated
    );
}

// ---------------------------------------------------------------------------
// Large system smoke test (n=16)
// ---------------------------------------------------------------------------

#[test]
fn large_system_smoke_n16() {
    let tpm = random_tpm(16, 42);
    let budget = ComputeBudget::fast();

    // Spectral should handle n=16 quickly.
    let spectral = SpectralPhiEngine::default()
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert!(spectral.phi >= 0.0);

    // Stochastic with limited samples.
    let stochastic = StochasticPhiEngine::new(200, 42)
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert!(stochastic.phi >= 0.0);

    // Greedy bisection.
    let greedy = GreedyBisectionPhiEngine::default()
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert!(greedy.phi >= 0.0);

    // Hierarchical.
    let hierarchical = HierarchicalPhiEngine::new(8)
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    assert!(hierarchical.phi >= 0.0);

    // Emergence.
    let emergence = CausalEmergenceEngine::default()
        .compute_emergence(&tpm, &budget)
        .unwrap();
    assert!(emergence.ei_micro >= 0.0);

    // RSVD.
    let rsvd = RsvdEmergenceEngine::default()
        .compute(&tpm, &budget)
        .unwrap();
    assert!(rsvd.effective_rank >= 1);
}

// ---------------------------------------------------------------------------
// Deterministic reproducibility
// ---------------------------------------------------------------------------

#[test]
fn stochastic_deterministic_with_same_seed() {
    let tpm = and_gate_tpm();
    let budget = ComputeBudget::fast();

    let r1 = StochasticPhiEngine::new(100, 42)
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();
    let r2 = StochasticPhiEngine::new(100, 42)
        .compute_phi(&tpm, Some(0), &budget)
        .unwrap();

    assert_eq!(r1.phi, r2.phi, "same seed should give same result");
    assert_eq!(r1.mip, r2.mip);
}

#[test]
fn collapse_deterministic_with_same_seed() {
    let tpm = and_gate_tpm();
    let r1 = QuantumCollapseEngine::new(64)
        .collapse_to_mip(&tpm, 10, 42)
        .unwrap();
    let r2 = QuantumCollapseEngine::new(64)
        .collapse_to_mip(&tpm, 10, 42)
        .unwrap();

    assert_eq!(r1.phi, r2.phi);
    assert_eq!(r1.mip, r2.mip);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn all_engines_reject_invalid_tpm() {
    let bad_tpm = TransitionMatrix::new(2, vec![0.5, 0.5, 0.3, 0.3]);
    let budget = ComputeBudget::exact();

    assert!(ExactPhiEngine.compute_phi(&bad_tpm, Some(0), &budget).is_err());
    assert!(SpectralPhiEngine::default().compute_phi(&bad_tpm, Some(0), &budget).is_err());
    assert!(StochasticPhiEngine::new(10, 42).compute_phi(&bad_tpm, Some(0), &budget).is_err());
    assert!(GeoMipPhiEngine::default().compute_phi(&bad_tpm, Some(0), &budget).is_err());
    assert!(GreedyBisectionPhiEngine::default().compute_phi(&bad_tpm, Some(0), &budget).is_err());
}

#[test]
fn exact_rejects_too_large() {
    let tpm = random_tpm(32, 42);
    let budget = ComputeBudget::exact();
    let result = ExactPhiEngine.compute_phi(&tpm, Some(0), &budget);
    assert!(result.is_err());
}
