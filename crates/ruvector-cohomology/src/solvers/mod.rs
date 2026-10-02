//! Deterministic solver suite (issue #669 solver policy).
//!
//! Dominant-eigenvalue power iteration is **never** used to infer nullity or
//! small eigenvalues. Available paths:
//!
//! - [`cg`]: matrix-free conjugate gradient for the gauged PSD normal
//!   equations (min-norm solution from a zero start).
//! - [`minres`]: MINRES for symmetric indefinite / saddle-point systems.
//! - [`lobpcg`]: LOBPCG for the smallest eigenpairs (nullity, spectral gap).
//! - [`sparse_qr`]: column-pivoted rank-revealing Householder QR — the exact
//!   reference path for small and medium systems.
//! - [`admm`]: group-sparse ADMM for semantic repair.
//!
//! Every approximate result carries a [`ResidualCertificate`].

pub mod admm;
pub mod cg;
pub mod lobpcg;
pub mod minres;
pub mod sparse_qr;

use serde::{Deserialize, Serialize};

/// Configuration shared by the iterative solvers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Relative residual tolerance.
    pub tol: f64,
    /// Iteration cap.
    pub max_iter: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self { tol: 1e-12, max_iter: 10_000 }
    }
}

/// Certificate attached to every approximate solve (issue #669 requirement:
/// "residual certificates for every approximate result").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualCertificate {
    /// Final residual norm ‖Ax − b‖ (or method-specific analogue).
    pub residual_norm: f64,
    /// Norm of the right-hand side, for relative interpretation.
    pub rhs_norm: f64,
    /// Iterations consumed.
    pub iterations: usize,
    /// Whether the requested tolerance was met.
    pub converged: bool,
    /// The tolerance that was requested.
    pub tolerance: f64,
}

/// Committed deterministic pseudo-random stream (SplitMix64). Used for
/// eigensolver probes so that "randomized" paths are reproducible; the seed
/// is part of the witness-relevant solver configuration.
#[derive(Debug, Clone)]
pub struct CommittedSeed(pub u64);

impl CommittedSeed {
    /// Next raw value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `(-1, 1)`, deterministic.
    pub fn next_signed_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Dense dot product.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm.
#[inline]
pub fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `y += alpha * x`.
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// `x *= alpha`.
#[inline]
pub fn scale(alpha: f64, x: &mut [f64]) {
    for xi in x.iter_mut() {
        *xi *= alpha;
    }
}
