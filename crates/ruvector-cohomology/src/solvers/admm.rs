//! Group-sparse ADMM for semantic repair (issue #669, Phase 3).
//!
//! Solves
//!
//! ```text
//! min_{x,q}  ½‖W^{1/2}(b − q − δx)‖²  +  λ Σ_e c_e ‖q_e‖₂
//! ```
//!
//! subject to `q_e = 0` on protected edges, via ADMM with splitting `z = q`.
//! The group ℓ₂ penalty removes or adjusts whole edge constraints, never
//! isolated scalar coordinates.

use super::cg::solve_psd;
use super::{norm2, SolverConfig};

/// The group-lasso repair problem, matrix-free over `δ`.
pub struct GroupLassoProblem<'a> {
    /// `y = δ x`.
    pub apply_delta: &'a dyn Fn(&[f64], &mut [f64]),
    /// `x = δᵀ y`.
    pub apply_delta_t: &'a dyn Fn(&[f64], &mut [f64]),
    /// Dimension of `C⁰`.
    pub n0: usize,
    /// Dimension of `C¹`.
    pub n1: usize,
    /// Per-edge `(offset, dim)` blocks inside `C¹`, in canonical order.
    pub edge_blocks: &'a [(usize, usize)],
    /// Per-edge positive weights `w_e`.
    pub weights: &'a [f64],
    /// Per-edge repair costs `c_e ≥ 0`.
    pub costs: &'a [f64],
    /// Protected edges: correction forced to zero.
    pub protected: &'a [bool],
    /// Observations `b ∈ C¹`.
    pub b: &'a [f64],
}

/// ADMM configuration.
#[derive(Debug, Clone)]
pub struct AdmmConfig {
    /// Group sparsity strength λ.
    pub lambda: f64,
    /// ADMM penalty ρ.
    pub rho: f64,
    /// Outer iteration cap.
    pub max_iter: usize,
    /// Primal/dual residual tolerance (absolute, scaled by √n1).
    pub tol: f64,
    /// Inner CG configuration for the x-step.
    pub inner: SolverConfig,
}

impl Default for AdmmConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            rho: 1.0,
            max_iter: 300,
            tol: 1e-8,
            inner: SolverConfig { tol: 1e-10, max_iter: 2000 },
        }
    }
}

/// ADMM output with residual certificates.
#[derive(Debug, Clone)]
pub struct AdmmResult {
    /// Sparse group correction `q` (exactly zero off-support).
    pub q: Vec<f64>,
    /// Final least-squares explanation `x` for the corrected observations.
    pub x: Vec<f64>,
    /// Objective value at `(x, q)`.
    pub objective: f64,
    /// Final primal residual ‖q − z‖.
    pub primal_residual: f64,
    /// Final dual residual ρ‖z − z_prev‖.
    pub dual_residual: f64,
    /// Outer iterations consumed.
    pub iterations: usize,
    /// Whether both residuals met the tolerance.
    pub converged: bool,
}

/// Evaluate the repair objective at `(x, q)`.
pub fn objective(p: &GroupLassoProblem, lambda: f64, x: &[f64], q: &[f64]) -> f64 {
    let mut dx = vec![0.0; p.n1];
    (p.apply_delta)(x, &mut dx);
    let mut fit = 0.0;
    let mut penalty = 0.0;
    for (e, &(off, dim)) in p.edge_blocks.iter().enumerate() {
        let mut block = 0.0;
        let mut qn = 0.0;
        for i in off..off + dim {
            let r = p.b[i] - q[i] - dx[i];
            block += r * r;
            qn += q[i] * q[i];
        }
        fit += p.weights[e] * block;
        penalty += p.costs[e] * qn.sqrt();
    }
    0.5 * fit + lambda * penalty
}

/// Solve the group-lasso repair problem with ADMM.
pub fn solve_group_lasso(p: &GroupLassoProblem, cfg: &AdmmConfig) -> AdmmResult {
    let n0 = p.n0;
    let n1 = p.n1;
    let mut x = vec![0.0; n0];
    let mut q = vec![0.0; n1];
    let mut z = vec![0.0; n1];
    let mut u = vec![0.0; n1];

    let apply_l = |v: &[f64], out: &mut [f64]| {
        let mut mid = vec![0.0; n1];
        (p.apply_delta)(v, &mut mid);
        for (e, &(off, dim)) in p.edge_blocks.iter().enumerate() {
            let w = p.weights[e];
            for m in &mut mid[off..off + dim] {
                *m *= w;
            }
        }
        (p.apply_delta_t)(&mid, out);
    };

    let mut dx = vec![0.0; n1];
    let mut rhs = vec![0.0; n0];
    let mut weighted = vec![0.0; n1];
    let scale = (n1 as f64).sqrt();
    let mut primal = f64::INFINITY;
    let mut dual = f64::INFINITY;
    let mut iterations = 0;

    for _ in 0..cfg.max_iter {
        iterations += 1;

        // (x, q)-step: two block-coordinate sweeps of the joint quadratic.
        for _ in 0..2 {
            // x given q: L x = δᵀ W (b − q).
            for i in 0..n1 {
                weighted[i] = p.b[i] - q[i];
            }
            for (e, &(off, dim)) in p.edge_blocks.iter().enumerate() {
                let w = p.weights[e];
                for wv in &mut weighted[off..off + dim] {
                    *wv *= w;
                }
            }
            (p.apply_delta_t)(&weighted, &mut rhs);
            let (x_new, _cert) = solve_psd(apply_l, &rhs, Some(&x), &cfg.inner);
            x = x_new;
            (p.apply_delta)(&x, &mut dx);

            // q given x: per-edge closed form of the ρ-augmented quadratic.
            for (e, &(off, dim)) in p.edge_blocks.iter().enumerate() {
                if p.protected[e] {
                    for i in off..off + dim {
                        q[i] = 0.0;
                    }
                    continue;
                }
                let w = p.weights[e];
                let denom = w + cfg.rho;
                for i in off..off + dim {
                    q[i] = (w * (p.b[i] - dx[i]) + cfg.rho * (z[i] - u[i])) / denom;
                }
            }
        }

        // z-step: block soft threshold of q + u with λ c_e / ρ.
        let z_prev = z.clone();
        for (e, &(off, dim)) in p.edge_blocks.iter().enumerate() {
            if p.protected[e] {
                for i in off..off + dim {
                    z[i] = 0.0;
                }
                continue;
            }
            let mut nrm = 0.0;
            for i in off..off + dim {
                let v = q[i] + u[i];
                nrm += v * v;
            }
            let nrm = nrm.sqrt();
            let threshold = cfg.lambda * p.costs[e] / cfg.rho;
            let shrink = if nrm > threshold { 1.0 - threshold / nrm } else { 0.0 };
            for i in off..off + dim {
                z[i] = shrink * (q[i] + u[i]);
            }
        }

        // Dual update.
        for i in 0..n1 {
            u[i] += q[i] - z[i];
        }

        primal = {
            let mut s = 0.0;
            for i in 0..n1 {
                s += (q[i] - z[i]).powi(2);
            }
            s.sqrt()
        };
        dual = {
            let mut s = 0.0;
            for i in 0..n1 {
                s += (z[i] - z_prev[i]).powi(2);
            }
            cfg.rho * s.sqrt()
        };
        if primal <= cfg.tol * scale && dual <= cfg.tol * scale {
            break;
        }
    }

    // Adopt the exactly-sparse z as the correction, then refit x once.
    q.copy_from_slice(&z);
    for i in 0..n1 {
        weighted[i] = p.b[i] - q[i];
    }
    for (e, &(off, dim)) in p.edge_blocks.iter().enumerate() {
        let w = p.weights[e];
        for wv in &mut weighted[off..off + dim] {
            *wv *= w;
        }
    }
    (p.apply_delta_t)(&weighted, &mut rhs);
    let (x_final, _) = solve_psd(apply_l, &rhs, Some(&x), &cfg.inner);
    x = x_final;

    let obj = objective(p, cfg.lambda, &x, &q);
    let converged = primal <= cfg.tol * scale && dual <= cfg.tol * scale;
    let _ = norm2(&dx);
    AdmmResult {
        q,
        x,
        objective: obj,
        primal_residual: primal,
        dual_residual: dual,
        iterations,
        converged,
    }
}
