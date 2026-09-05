//! MINRES for symmetric (possibly indefinite) systems.
//!
//! Used for saddle-point / indefinite gauge formulations where CG's positive
//! definiteness assumption fails (issue #669 solver policy, path 2).

use super::{norm2, ResidualCertificate, SolverConfig};

/// Solve `A x = b` with symmetric `apply_a` via MINRES (Paige–Saunders).
pub fn solve_symmetric(
    apply_a: impl Fn(&[f64], &mut [f64]),
    b: &[f64],
    cfg: &SolverConfig,
) -> (Vec<f64>, ResidualCertificate) {
    let n = b.len();
    let rhs_norm = norm2(b);
    let mut x = vec![0.0; n];
    if rhs_norm == 0.0 {
        return (
            x,
            ResidualCertificate {
                residual_norm: 0.0,
                rhs_norm,
                iterations: 0,
                converged: true,
                tolerance: cfg.tol,
            },
        );
    }

    // Lanczos vectors.
    let mut v_prev = vec![0.0; n];
    let mut v = b.to_vec();
    let mut beta = rhs_norm;
    for vi in v.iter_mut() {
        *vi /= beta;
    }

    // Givens rotation state.
    let (mut c_prev, mut s_prev) = (1.0f64, 0.0f64);
    let (mut c, mut s) = (1.0f64, 0.0f64);
    let mut w_prev = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut eta = beta;

    let mut av = vec![0.0; n];
    let mut residual_norm = rhs_norm;
    let mut iterations = 0;
    let threshold = cfg.tol * rhs_norm;

    for _ in 0..cfg.max_iter.min(4 * n.max(1)) {
        iterations += 1;
        apply_a(&v, &mut av);
        let alpha = super::dot(&v, &av);
        // Lanczos three-term recurrence.
        let mut v_next = av.clone();
        for i in 0..n {
            v_next[i] -= alpha * v[i] + beta * v_prev[i];
        }
        let beta_next = norm2(&v_next);
        if beta_next > 0.0 {
            for vi in v_next.iter_mut() {
                *vi /= beta_next;
            }
        }

        // Apply previous rotations to the new tridiagonal column.
        let delta = c * alpha - c_prev * s * beta;
        let gamma_bar = s * alpha + c_prev * c * beta;
        let epsilon = s_prev * beta;

        // New rotation to annihilate beta_next.
        let gamma = (delta * delta + beta_next * beta_next).sqrt();
        let gamma = if gamma == 0.0 { f64::MIN_POSITIVE } else { gamma };
        let c_next = delta / gamma;
        let s_next = beta_next / gamma;

        // Update solution direction.
        let mut w_next = vec![0.0; n];
        for i in 0..n {
            w_next[i] = (v[i] - gamma_bar * w[i] - epsilon * w_prev[i]) / gamma;
        }
        for i in 0..n {
            x[i] += c_next * eta * w_next[i];
        }
        residual_norm = (s_next * eta).abs();
        eta *= -s_next;

        if residual_norm <= threshold {
            break;
        }

        v_prev = std::mem::replace(&mut v, v_next);
        w_prev = std::mem::replace(&mut w, w_next);
        c_prev = c;
        s_prev = s;
        c = c_next;
        s = s_next;
        beta = beta_next;
        if beta == 0.0 {
            break;
        }
    }

    let converged = residual_norm <= threshold;
    (
        x,
        ResidualCertificate {
            residual_norm,
            rhs_norm,
            iterations,
            converged,
            tolerance: cfg.tol,
        },
    )
}
