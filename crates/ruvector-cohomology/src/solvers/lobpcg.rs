//! LOBPCG for the smallest eigenpairs of a PSD operator.
//!
//! Issue #669 solver policy, path 3: nullity and spectral-gap estimation for
//! the sheaf Laplacian must use a smallest-eigenpair method — dominant
//! power iteration is explicitly forbidden. The Rayleigh–Ritz step uses a
//! deterministic cyclic Jacobi symmetric eigensolver.

use super::{dot, norm2, CommittedSeed, ResidualCertificate, SolverConfig};
use crate::operator::DenseMatrix;

/// Result of a smallest-eigenpair computation.
#[derive(Debug, Clone)]
pub struct SmallestEigenpairs {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns (n × k), matching `eigenvalues`.
    pub eigenvectors: DenseMatrix,
    /// Residual certificate: `residual_norm` is the max over the block of
    /// ‖A v − λ v‖.
    pub certificate: ResidualCertificate,
}

/// Cyclic Jacobi eigensolver for a small dense symmetric matrix.
/// Returns eigenvalues ascending and eigenvectors as columns, with signs
/// canonicalized (first component of largest magnitude made positive).
pub fn jacobi_eigh(a: &DenseMatrix) -> (Vec<f64>, DenseMatrix) {
    let n = a.rows;
    debug_assert_eq!(a.rows, a.cols);
    let mut m = a.clone();
    let mut v = DenseMatrix::identity(n);
    let max_sweeps = 64;
    for _ in 0..max_sweeps {
        let mut off: f64 = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m.get(i, j).powi(2);
            }
        }
        if off.sqrt() < 1e-14 * (1.0 + m.frobenius_norm()) {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m.get(p, q);
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m.get(p, p);
                let aqq = m.get(q, q);
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m.get(k, p);
                    let mkq = m.get(k, q);
                    m.set(k, p, c * mkp - s * mkq);
                    m.set(k, q, s * mkp + c * mkq);
                }
                for k in 0..n {
                    let mpk = m.get(p, k);
                    let mqk = m.get(q, k);
                    m.set(p, k, c * mpk - s * mqk);
                    m.set(q, k, s * mpk + c * mqk);
                }
                for k in 0..n {
                    let vkp = v.get(k, p);
                    let vkq = v.get(k, q);
                    v.set(k, p, c * vkp - s * vkq);
                    v.set(k, q, s * vkp + c * vkq);
                }
            }
        }
    }
    // Sort ascending by eigenvalue, stable for determinism.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m.get(i, i).partial_cmp(&m.get(j, j)).unwrap().then(i.cmp(&j)));
    let mut evals = Vec::with_capacity(n);
    let mut evecs = DenseMatrix::zeros(n, n);
    for (c, &i) in order.iter().enumerate() {
        evals.push(m.get(i, i));
        // Sign canonicalization: largest-magnitude component positive.
        let mut best = 0usize;
        let mut best_abs = 0.0f64;
        for r in 0..n {
            let a = v.get(r, i).abs();
            if a > best_abs {
                best_abs = a;
                best = r;
            }
        }
        let sign = if v.get(best, i) < 0.0 { -1.0 } else { 1.0 };
        for r in 0..n {
            evecs.set(r, c, sign * v.get(r, i));
        }
    }
    (evals, evecs)
}

/// Orthonormalize the columns of `basis` in place with modified
/// Gram–Schmidt, dropping columns below `drop_tol`. Returns kept count.
fn orthonormalize(basis: &mut Vec<Vec<f64>>, drop_tol: f64) -> usize {
    let mut kept = 0usize;
    for i in 0..basis.len() {
        let (done, rest) = basis.split_at_mut(i);
        let col = &mut rest[0];
        for prev in done[..kept].iter() {
            let d = dot(prev, col);
            for (c, p) in col.iter_mut().zip(prev.iter()) {
                *c -= d * p;
            }
        }
        let n = norm2(col);
        if n > drop_tol {
            for c in col.iter_mut() {
                *c /= n;
            }
            if kept != i {
                basis.swap(kept, i);
            }
            kept += 1;
        }
    }
    basis.truncate(kept);
    kept
}

/// Compute the `k` smallest eigenpairs of a symmetric PSD operator via
/// LOBPCG with a committed seed for the starting block.
pub fn smallest_eigenpairs(
    apply_a: impl Fn(&[f64], &mut [f64]),
    n: usize,
    k: usize,
    seed: u64,
    cfg: &SolverConfig,
) -> SmallestEigenpairs {
    let k = k.min(n).max(1);
    let mut rng = CommittedSeed(seed);
    let mut x: Vec<Vec<f64>> = (0..k)
        .map(|_| (0..n).map(|_| rng.next_signed_unit()).collect())
        .collect();
    orthonormalize(&mut x, 1e-12);
    while x.len() < k {
        // Degenerate probe (tiny n): pad with unit vectors.
        let mut e = vec![0.0; n];
        e[x.len() % n] = 1.0;
        x.push(e);
        orthonormalize(&mut x, 1e-12);
    }

    let mut p: Vec<Vec<f64>> = Vec::new();
    let mut evals = vec![0.0; k];
    let mut max_res = f64::INFINITY;
    let mut iterations = 0;
    let scale_estimate = {
        // Cheap deterministic norm estimate for relative tolerance.
        let probe: Vec<f64> = (0..n).map(|_| rng.next_signed_unit()).collect();
        let nn = norm2(&probe).max(f64::MIN_POSITIVE);
        let mut out = vec![0.0; n];
        apply_a(&probe, &mut out);
        (norm2(&out) / nn).max(1.0)
    };

    for _ in 0..cfg.max_iter.min(500) {
        iterations += 1;
        // Residuals R = A X − X Θ with Rayleigh quotients Θ.
        let ax: Vec<Vec<f64>> = x
            .iter()
            .map(|xi| {
                let mut out = vec![0.0; n];
                apply_a(xi, &mut out);
                out
            })
            .collect();
        let mut residuals: Vec<Vec<f64>> = Vec::with_capacity(k);
        max_res = 0.0;
        for (i, (xi, axi)) in x.iter().zip(ax.iter()).enumerate() {
            let theta = dot(xi, axi);
            evals[i] = theta;
            let mut r = axi.clone();
            for (rj, xj) in r.iter_mut().zip(xi.iter()) {
                *rj -= theta * xj;
            }
            max_res = max_res.max(norm2(&r));
            residuals.push(r);
        }
        if max_res <= cfg.tol * scale_estimate {
            break;
        }

        // Subspace S = [X, R, P], orthonormalized.
        let mut s: Vec<Vec<f64>> = Vec::with_capacity(3 * k);
        s.extend(x.iter().cloned());
        s.extend(residuals);
        s.extend(p.iter().cloned());
        let kept = orthonormalize(&mut s, 1e-10);
        if kept == 0 {
            break;
        }

        // Rayleigh–Ritz on the small subspace.
        let a_s: Vec<Vec<f64>> = s
            .iter()
            .map(|si| {
                let mut out = vec![0.0; n];
                apply_a(si, &mut out);
                out
            })
            .collect();
        let mut g = DenseMatrix::zeros(kept, kept);
        for i in 0..kept {
            for j in i..kept {
                let v = dot(&s[i], &a_s[j]);
                g.set(i, j, v);
                g.set(j, i, v);
            }
        }
        let (_, vecs) = jacobi_eigh(&g);

        // New X from the k smallest Ritz vectors; P = new − old components.
        let mut x_new: Vec<Vec<f64>> = Vec::with_capacity(k);
        for c in 0..k.min(kept) {
            let mut xi = vec![0.0; n];
            for (row, sv) in s.iter().enumerate() {
                let coeff = vecs.get(row, c);
                if coeff != 0.0 {
                    super::axpy(coeff, sv, &mut xi);
                }
            }
            x_new.push(xi);
        }
        p = x_new
            .iter()
            .zip(x.iter())
            .map(|(new, old)| {
                let mut d = new.clone();
                for (di, oi) in d.iter_mut().zip(old.iter()) {
                    *di -= oi;
                }
                d
            })
            .collect();
        orthonormalize(&mut p, 1e-10);
        x = x_new;
        orthonormalize(&mut x, 1e-12);
    }

    let mut eigenvectors = DenseMatrix::zeros(n, x.len());
    for (c, xi) in x.iter().enumerate() {
        for (r, v) in xi.iter().enumerate() {
            eigenvectors.set(r, c, *v);
        }
    }
    let converged = max_res <= cfg.tol * scale_estimate;
    SmallestEigenpairs {
        eigenvalues: evals[..x.len().min(k)].to_vec(),
        eigenvectors,
        certificate: ResidualCertificate {
            residual_norm: max_res,
            rhs_norm: scale_estimate,
            iterations,
            converged,
            tolerance: cfg.tol,
        },
    }
}
