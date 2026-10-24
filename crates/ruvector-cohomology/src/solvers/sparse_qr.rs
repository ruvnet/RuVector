//! Column-pivoted (rank-revealing) Householder QR.
//!
//! The exact reference path (issue #669 solver policy, path 4): used by the
//! dense oracle to compute exact least-squares residuals, numerical rank,
//! and an orthonormal basis of `ker(Aᵀ)` (the harmonic/contradiction space).

use crate::operator::DenseMatrix;

/// Column-pivoted Householder QR factorization `A P = Q R`.
pub struct PivotedQr {
    m: usize,
    n: usize,
    /// Householder vectors below the diagonal, `R` on and above.
    qr: DenseMatrix,
    /// Householder coefficients.
    tau: Vec<f64>,
    /// Column permutation: factored column `j` is original column `perm[j]`.
    perm: Vec<usize>,
    rank: usize,
}

impl PivotedQr {
    /// Factor `A` with Businger–Golub column pivoting. `rank_tol_rel` is the
    /// relative diagonal threshold for the numerical rank decision.
    pub fn factor(a: &DenseMatrix, rank_tol_rel: f64) -> Self {
        let m = a.rows;
        let n = a.cols;
        let mut qr = a.clone();
        let kmax = m.min(n);
        let mut tau = vec![0.0; kmax];
        let mut perm: Vec<usize> = (0..n).collect();

        // Squared column norms for pivoting.
        let mut col_norms: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| qr.get(i, j).powi(2)).sum())
            .collect();
        let mut rank = 0;
        let mut first_diag = 0.0f64;

        for k in 0..kmax {
            // Pivot: largest remaining column norm; ties broken by lowest
            // index for determinism.
            let (pivot, &pnorm) = col_norms[k..]
                .iter()
                .enumerate()
                .fold((0usize, &f64::MIN), |best, (i, v)| if *v > *best.1 { (i, v) } else { best });
            let pivot = k + pivot;
            if pivot != k {
                for i in 0..m {
                    let t = qr.get(i, k);
                    qr.set(i, k, qr.get(i, pivot));
                    qr.set(i, pivot, t);
                }
                perm.swap(k, pivot);
                col_norms.swap(k, pivot);
            }
            let _ = pnorm;

            // Householder reflector on column k, rows k..m.
            let mut alpha = 0.0;
            for i in k..m {
                alpha += qr.get(i, k).powi(2);
            }
            let alpha = alpha.sqrt();
            let akk = qr.get(k, k);
            if k == 0 {
                first_diag = alpha;
            }
            if alpha <= rank_tol_rel * first_diag.max(f64::MIN_POSITIVE) {
                break; // Remaining block numerically zero: rank found.
            }
            rank += 1;
            let sign = if akk >= 0.0 { 1.0 } else { -1.0 };
            let v0 = akk + sign * alpha;
            // Normalize the reflector so v[k] = 1 implicitly.
            for i in (k + 1)..m {
                qr.set(i, k, qr.get(i, k) / v0);
            }
            let mut vtv = 1.0;
            for i in (k + 1)..m {
                vtv += qr.get(i, k).powi(2);
            }
            tau[k] = 2.0 / vtv;
            qr.set(k, k, -sign * alpha);

            // Apply reflector to the trailing columns.
            for j in (k + 1)..n {
                let mut dot = qr.get(k, j);
                for i in (k + 1)..m {
                    dot += qr.get(i, k) * qr.get(i, j);
                }
                let t = tau[k] * dot;
                qr.set(k, j, qr.get(k, j) - t);
                for i in (k + 1)..m {
                    qr.set(i, j, qr.get(i, j) - t * qr.get(i, k));
                }
                // Downdate the column norm.
                col_norms[j] = (col_norms[j] - qr.get(k, j).powi(2)).max(0.0);
            }
        }

        Self { m, n, qr, tau, perm, rank }
    }

    /// Numerical rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// `y ← Qᵀ y` (length m).
    pub fn apply_qt(&self, y: &mut [f64]) {
        for k in 0..self.rank {
            let mut dot = y[k];
            for i in (k + 1)..self.m {
                dot += self.qr.get(i, k) * y[i];
            }
            let t = self.tau[k] * dot;
            y[k] -= t;
            for i in (k + 1)..self.m {
                y[i] -= t * self.qr.get(i, k);
            }
        }
    }

    /// `y ← Q y` (length m).
    pub fn apply_q(&self, y: &mut [f64]) {
        for k in (0..self.rank).rev() {
            let mut dot = y[k];
            for i in (k + 1)..self.m {
                dot += self.qr.get(i, k) * y[i];
            }
            let t = self.tau[k] * dot;
            y[k] -= t;
            for i in (k + 1)..self.m {
                y[i] -= t * self.qr.get(i, k);
            }
        }
    }

    /// Exact least-squares solution of `min ‖Ax − b‖₂` (basic solution: free
    /// pivoted variables set to zero). The residual is exact regardless of
    /// which least-squares representative is returned.
    pub fn solve_least_squares(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), self.m);
        let mut qtb = b.to_vec();
        self.apply_qt(&mut qtb);
        // Back-substitute R[0..rank, 0..rank] z = (Qᵀb)[0..rank].
        let r = self.rank;
        let mut z = vec![0.0; r];
        for i in (0..r).rev() {
            let mut acc = qtb[i];
            for j in (i + 1)..r {
                acc -= self.qr.get(i, j) * z[j];
            }
            z[i] = acc / self.qr.get(i, i);
        }
        let mut x = vec![0.0; self.n];
        for (j, &zj) in z.iter().enumerate() {
            x[self.perm[j]] = zj;
        }
        x
    }

    /// Orthonormal basis of `ker(Aᵀ)` — the columns `Q e_i` for
    /// `i ∈ [rank, m)`. In syndrome coordinates (`A = W^{1/2}δ`) this is the
    /// harmonic/contradiction subspace `(im δ)^{⊥_W}` expressed in
    /// `W^{1/2}`-scaled coordinates.
    pub fn null_space_of_transpose(&self) -> DenseMatrix {
        let cols = self.m - self.rank;
        let mut basis = DenseMatrix::zeros(self.m, cols);
        let mut e = vec![0.0; self.m];
        for (c, i) in (self.rank..self.m).enumerate() {
            e.fill(0.0);
            e[i] = 1.0;
            self.apply_q(&mut e);
            for r in 0..self.m {
                basis.set(r, c, e[r]);
            }
        }
        basis
    }
}
