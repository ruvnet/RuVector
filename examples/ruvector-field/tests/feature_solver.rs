//! Verifies the `solver` feature produces bounded, monotone coherence.
//!
//! Build/run with:
//!
//! ```text
//! cargo test --features solver --test feature_solver
//! ```

#![cfg(feature = "solver")]

use ruvector_field::model::Embedding;
use ruvector_field::scoring::coherence::solver_backend::{
    NeumannSolverBackend, SolverBackend,
};
use ruvector_field::scoring::local_coherence;

#[test]
fn coherence_bounded_in_unit_interval() {
    let center = Embedding::new(vec![1.0, 0.0, 0.0]);
    let n1 = Embedding::new(vec![0.9, 0.1, 0.0]);
    let n2 = Embedding::new(vec![0.8, 0.2, 0.0]);
    let coh = local_coherence(&center, &[(&n1, 0.9), (&n2, 0.8)], 4);
    assert!(coh > 0.0 && coh <= 1.0, "coh out of range: {}", coh);
}

#[test]
fn stronger_neighbors_monotone_increase() {
    let center = Embedding::new(vec![1.0, 0.0, 0.0]);
    let weak = Embedding::new(vec![0.2, 0.7, 0.0]);
    let strong = Embedding::new(vec![0.99, 0.01, 0.0]);
    let coh_weak = local_coherence(&center, &[(&weak, 0.3)], 4);
    let coh_strong = local_coherence(&center, &[(&strong, 0.9)], 4);
    assert!(
        coh_strong >= coh_weak,
        "expected monotonicity: weak={}, strong={}",
        coh_weak,
        coh_strong
    );
}

#[test]
fn backend_returns_nonnegative_resistance() {
    let backend = NeumannSolverBackend::default();
    let r = backend.mean_effective_resistance(&[0.5, 0.7, 0.9]);
    assert!(r >= 0.0 && r.is_finite());
}

#[test]
fn backend_matches_parallel_closed_form() {
    let backend = NeumannSolverBackend::default();
    // Parallel-combined R = 1 / sum(w_i) = 1/3 for three unit conductances.
    let r = backend.mean_effective_resistance(&[1.0, 1.0, 1.0]);
    assert!((r - (1.0 / 3.0)).abs() < 1e-2, "got {}", r);
}
