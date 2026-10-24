//! Matrix-free conjugate gradient for gauged PSD systems.
//!
//! Solving `L x = δᵀ W b` with `L = δᵀ W δ` from `x₀ = 0` keeps every Krylov
//! iterate inside `range(L)`, so CG converges to the **minimum-norm**
//! (Moore–Penrose) solution without an explicit gauge constraint — this is
//! the [`crate::GaugePolicy::MinimumNorm`] gauge.

use super::{axpy, dot, norm2, ResidualCertificate, SolverConfig};

/// Solve `A x = b` for symmetric positive semidefinite `apply_a`, returning
/// the solution and a residual certificate.
///
/// `warm_start`, when provided and consistent, seeds the iteration (used by
/// the dynamic engine between flushes on an unchanged topology).
pub fn solve_psd(
    apply_a: impl Fn(&[f64], &mut [f64]),
    b: &[f64],
    warm_start: Option<&[f64]>,
    cfg: &SolverConfig,
) -> (Vec<f64>, ResidualCertificate) {
    let n = b.len();
    let rhs_norm = norm2(b);
    let mut x = match warm_start {
        Some(w) if w.len() == n => w.to_vec(),
        _ => vec![0.0; n],
    };
    let mut r = vec![0.0; n];
    if x.iter().any(|&v| v != 0.0) {
        apply_a(&x, &mut r);
        for (ri, bi) in r.iter_mut().zip(b.iter()) {
            *ri = bi - *ri;
        }
    } else {
        r.copy_from_slice(b);
    }

    let threshold = cfg.tol * rhs_norm.max(f64::MIN_POSITIVE);
    let mut rr = dot(&r, &r);
    if rr.sqrt() <= threshold || rhs_norm == 0.0 {
        let res = rr.sqrt();
        return (
            x,
            ResidualCertificate {
                residual_norm: res,
                rhs_norm,
                iterations: 0,
                converged: true,
                tolerance: cfg.tol,
            },
        );
    }

    let mut p = r.clone();
    let mut ap = vec![0.0; n];
    let mut iterations = 0;
    for _ in 0..cfg.max_iter.min(4 * n.max(1)) {
        iterations += 1;
        apply_a(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 || !pap.is_finite() {
            // Numerically at the boundary of the semidefinite cone (p in the
            // nullspace direction): the current iterate is the best gauged
            // answer this Krylov space offers.
            break;
        }
        let alpha = rr / pap;
        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);
        let rr_new = dot(&r, &r);
        if rr_new.sqrt() <= threshold {
            rr = rr_new;
            break;
        }
        let beta = rr_new / rr;
        rr = rr_new;
        for (pi, ri) in p.iter_mut().zip(r.iter()) {
            *pi = ri + beta * *pi;
        }
    }

    let residual_norm = rr.sqrt();
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
