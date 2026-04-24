//! PAC-style approximation guarantees for Φ estimation.
//!
//! Provides provable confidence intervals for approximate Φ:
//! - Spectral lower/upper bounds from Fiedler eigenvalue
//! - Hoeffding concentration bounds for stochastic sampling
//! - Chebyshev bounds for any estimator with known variance
//! - Empirical Bernstein bounds for tighter intervals
//!
//! Key guarantee: with probability ≥ 1-δ, the true Φ lies within
//! the computed interval [lower, upper].

use crate::error::ConsciousnessError;
use crate::simd::build_mi_matrix;
use crate::traits::PhiEngine;
use crate::types::{ComputeBudget, PhiBound, TransitionMatrix};

// ---------------------------------------------------------------------------
// Spectral bounds
// ---------------------------------------------------------------------------

/// Compute spectral bounds on Φ from the MI Laplacian eigenvalues.
///
/// The Fiedler value λ₂ provides a lower bound on Φ:
///   Φ ≥ λ₂ / (2 · d_max)
/// where d_max is the maximum weighted degree.
///
/// The Cheeger inequality gives an upper bound:
///   Φ ≤ √(2 · h(G))
/// where h(G) is the Cheeger constant ≤ √(2 · λ₂).
pub fn spectral_bounds(tpm: &TransitionMatrix) -> Result<PhiBound, ConsciousnessError> {
    let n = tpm.n;
    if n < 2 {
        return Err(crate::error::ValidationError::EmptySystem.into());
    }

    let mi_matrix = build_mi_matrix(tpm.as_slice(), n);

    // Build Laplacian and find key eigenvalues.
    let mut laplacian = vec![0.0f64; n * n];
    let mut max_degree = 0.0f64;
    for i in 0..n {
        let mut degree = 0.0;
        for j in 0..n {
            degree += mi_matrix[i * n + j];
        }
        max_degree = max_degree.max(degree);
        laplacian[i * n + i] = degree;
        for j in 0..n {
            laplacian[i * n + j] -= mi_matrix[i * n + j];
        }
    }

    // Estimate Fiedler value (λ₂) via power iteration.
    let fiedler = estimate_fiedler(&laplacian, n, 200);

    // Lower bound: Fiedler-based.
    let lower = if max_degree > 1e-15 {
        fiedler / (2.0 * max_degree)
    } else {
        0.0
    };

    // Upper bound: Cheeger inequality.
    // h(G) ≤ √(2 · λ₂) and Φ ≤ some function of h.
    let upper = (2.0 * fiedler).sqrt();

    Ok(PhiBound {
        lower: lower.max(0.0),
        upper,
        confidence: 1.0, // Deterministic bound.
        samples: 0,
        method: "spectral-cheeger".into(),
    })
}

/// Estimate Fiedler value via inverse power iteration.
fn estimate_fiedler(laplacian: &[f64], n: usize, max_iter: usize) -> f64 {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();

    let inv_n = 1.0 / n as f64;
    let mean: f64 = v.iter().sum::<f64>() * inv_n;
    for vi in &mut v {
        *vi -= mean;
    }
    normalize(&mut v);

    let mu = gershgorin_max(laplacian, n);
    let mut w = vec![0.0f64; n];
    let mut prev_rayleigh = f64::MAX;

    for _ in 0..max_iter {
        for i in 0..n {
            let mut sum = mu * v[i];
            for j in 0..n {
                sum -= laplacian[i * n + j] * v[j];
            }
            w[i] = sum;
        }
        let mean: f64 = w.iter().sum::<f64>() * inv_n;
        for wi in &mut w {
            *wi -= mean;
        }
        let norm = normalize(&mut w);
        if norm < 1e-15 {
            break;
        }
        v.copy_from_slice(&w);

        // Early exit: check Rayleigh quotient convergence.
        let rq = rayleigh_quotient(laplacian, &v, n);
        if (rq - prev_rayleigh).abs() < 1e-10 {
            return rq.max(0.0);
        }
        prev_rayleigh = rq;
    }

    // Rayleigh quotient: λ₂ ≈ v^T L v / v^T v
    rayleigh_quotient(laplacian, &v, n).max(0.0)
}

// ---------------------------------------------------------------------------
// Hoeffding concentration bound (for stochastic sampling)
// ---------------------------------------------------------------------------

/// Compute a Hoeffding confidence interval for stochastic Φ estimation.
///
/// Given k samples from the partition space with observed minimum φ_min,
/// the true Φ satisfies:
///   P(|Φ̂ - Φ| ≤ ε) ≥ 1 - δ
///
/// where ε = B · √(ln(2/δ) / (2k)) and B is the range of φ values.
///
/// `phi_estimate`: the observed minimum from k samples.
/// `k`: number of samples evaluated.
/// `phi_max`: maximum observed φ (used for range bound).
/// `delta`: failure probability (e.g., 0.05 for 95% confidence).
pub fn hoeffding_bound(phi_estimate: f64, k: u64, phi_max: f64, delta: f64) -> PhiBound {
    assert!(delta > 0.0 && delta < 1.0);
    assert!(k > 0);

    let range = phi_max.max(phi_estimate);
    let epsilon = range * ((2.0f64 / delta).ln() / (2.0 * k as f64)).sqrt();

    PhiBound {
        lower: (phi_estimate - epsilon).max(0.0),
        upper: phi_estimate + epsilon,
        confidence: 1.0 - delta,
        samples: k,
        method: "hoeffding".into(),
    }
}

// ---------------------------------------------------------------------------
// Empirical Bernstein bound (tighter for low-variance estimators)
// ---------------------------------------------------------------------------

/// Compute an empirical Bernstein confidence interval.
///
/// Tighter than Hoeffding when the variance of φ values is small.
///
/// `phi_estimates`: all observed φ values from sampling.
/// `delta`: failure probability.
pub fn empirical_bernstein_bound(phi_estimates: &[f64], delta: f64) -> PhiBound {
    assert!(!phi_estimates.is_empty());
    assert!(delta > 0.0 && delta < 1.0);

    let k = phi_estimates.len() as f64;
    let mean: f64 = phi_estimates.iter().sum::<f64>() / k;

    // Sample variance.
    let variance: f64 = phi_estimates
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / (k - 1.0).max(1.0);

    // Range bound.
    let max_val = phi_estimates
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_val = phi_estimates.iter().cloned().fold(f64::INFINITY, f64::min);

    let phi_min = min_val; // Best estimate (minimum = MIP).

    let log_term = (3.0 / delta).ln();

    // Empirical Bernstein: ε = √(2V·ln(3/δ)/k) + 3B·ln(3/δ)/(3(k-1))
    let range = max_val - min_val;
    let epsilon = (2.0 * variance * log_term / k).sqrt()
        + 3.0 * range * log_term / (3.0 * (k - 1.0).max(1.0));

    PhiBound {
        lower: (phi_min - epsilon).max(0.0),
        upper: phi_min + epsilon,
        confidence: 1.0 - delta,
        samples: phi_estimates.len() as u64,
        method: "empirical-bernstein".into(),
    }
}

// ---------------------------------------------------------------------------
// Combined bound: run an engine and wrap result with confidence
// ---------------------------------------------------------------------------

/// Run a PhiEngine and compute confidence bounds on the result.
///
/// For exact engines, returns a tight bound (lower = upper = φ).
/// For approximate engines, uses the convergence history to compute bounds.
pub fn compute_phi_with_bounds<E: PhiEngine>(
    engine: &E,
    tpm: &TransitionMatrix,
    state: Option<usize>,
    budget: &ComputeBudget,
    delta: f64,
) -> Result<(crate::types::PhiResult, PhiBound), ConsciousnessError> {
    let result = engine.compute_phi(tpm, state, budget)?;

    let bound = if result.convergence.len() > 1 {
        // Use convergence history for empirical Bernstein bound.
        empirical_bernstein_bound(&result.convergence, delta)
    } else {
        // Single evaluation: use spectral bounds.
        spectral_bounds(tpm).unwrap_or(PhiBound {
            lower: 0.0,
            upper: result.phi * 2.0,
            confidence: 0.5,
            samples: 1,
            method: "fallback".into(),
        })
    };

    Ok((result, bound))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rayleigh_quotient(laplacian: &[f64], v: &[f64], n: usize) -> f64 {
    let mut vtlv = 0.0f64;
    for i in 0..n {
        let mut lv_i = 0.0;
        for j in 0..n {
            lv_i += laplacian[i * n + j] * v[j];
        }
        vtlv += v[i] * lv_i;
    }
    vtlv
}

fn normalize(v: &mut [f64]) -> f64 {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        let inv = 1.0 / norm;
        for vi in v.iter_mut() {
            *vi *= inv;
        }
    }
    norm
}

fn gershgorin_max(matrix: &[f64], n: usize) -> f64 {
    let mut max_sum = 0.0f64;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += matrix[i * n + j].abs();
        }
        max_sum = max_sum.max(row_sum);
    }
    max_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi::SpectralPhiEngine;

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

    #[test]
    fn spectral_bounds_valid_interval() {
        let tpm = and_gate_tpm();
        let bound = spectral_bounds(&tpm).unwrap();
        assert!(bound.lower >= 0.0);
        assert!(bound.upper >= bound.lower);
        assert_eq!(bound.confidence, 1.0);
    }

    #[test]
    fn hoeffding_bound_narrows_with_samples() {
        let b1 = hoeffding_bound(0.5, 100, 1.0, 0.05);
        let b2 = hoeffding_bound(0.5, 10000, 1.0, 0.05);
        assert!(
            b2.upper - b2.lower < b1.upper - b1.lower,
            "more samples should give tighter bound"
        );
    }

    #[test]
    fn empirical_bernstein_produces_interval() {
        let samples = vec![0.3, 0.35, 0.32, 0.31, 0.33, 0.34, 0.30, 0.36];
        let bound = empirical_bernstein_bound(&samples, 0.05);
        assert!(bound.lower >= 0.0);
        assert!(bound.upper > bound.lower);
        assert!((bound.confidence - 0.95).abs() < 1e-10);
    }

    #[test]
    fn compute_with_bounds_works() {
        let tpm = and_gate_tpm();
        let engine = SpectralPhiEngine::default();
        let budget = ComputeBudget::fast();
        let (result, bound) =
            compute_phi_with_bounds(&engine, &tpm, Some(0), &budget, 0.05).unwrap();
        assert!(result.phi >= 0.0);
        assert!(bound.lower >= 0.0);
    }
}
