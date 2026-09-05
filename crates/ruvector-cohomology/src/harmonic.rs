//! Harmonic (contradiction) subspace extraction and canonicalization.
//!
//! The harmonic space is `(im δ)^{⊥_W} ⊂ C¹`, expressed in
//! `W^{1/2}`-scaled coordinates as `ker(Aᵀ)` for `A = W^{1/2}δ`. Its
//! coordinates give the compact contradiction witness. Degenerate subspaces
//! are canonicalized deterministically with QR against a hashed probe basis
//! (issue #669, §7 canonicalization rules).

use crate::block_sparse::BlockCoboundary;
use crate::operator::DenseMatrix;
use crate::solvers::lobpcg::smallest_eigenpairs;
use crate::solvers::sparse_qr::PivotedQr;
use crate::solvers::{CommittedSeed, ResidualCertificate, SolverConfig};

/// Canonical orthonormal basis of the harmonic subspace, in scaled
/// (`W^{1/2}`) edge coordinates.
pub struct HarmonicBasis {
    /// `dim_c1 × harmonic_dim` orthonormal columns.
    pub basis: DenseMatrix,
    /// Numerical rank of `W^{1/2}δ`.
    pub rank: usize,
}

impl HarmonicBasis {
    /// Exact dense-oracle path: rank-revealing QR of `A = W^{1/2}δ`, null
    /// space of `Aᵀ`, canonicalized against a probe derived from
    /// `operator_hash` so the basis is independent of factorization order.
    pub fn dense(
        op: &BlockCoboundary,
        weights: &[f64],
        rank_tol_rel: f64,
    ) -> Self {
        let a = op.assemble_scaled_dense(weights);
        let qr = PivotedQr::factor(&a, rank_tol_rel);
        let rank = qr.rank();
        let raw = qr.null_space_of_transpose();
        let basis = canonicalize_subspace(&raw, op.canonical_hash());
        Self { basis, rank }
    }

    /// Coordinates of a scaled edge field in this basis: `Bᵀ (W^{1/2} s)`.
    pub fn coordinates(&self, scaled_field: &[f64]) -> Vec<f64> {
        let mut coords = vec![0.0; self.basis.cols];
        for c in 0..self.basis.cols {
            let mut acc = 0.0;
            for r in 0..self.basis.rows {
                acc += self.basis.get(r, c) * scaled_field[r];
            }
            coords[c] = acc;
        }
        coords
    }
}

/// Canonicalize a subspace given *any* orthonormal basis of it: project a
/// deterministic hashed probe onto the subspace and re-orthonormalize with
/// modified Gram–Schmidt, then fix signs by the first component of
/// significant magnitude. The result depends only on the subspace and the
/// probe seed, not on the incoming basis or factorization order.
pub fn canonicalize_subspace(raw: &DenseMatrix, seed_hash: [u8; 32]) -> DenseMatrix {
    let m = raw.rows;
    let h = raw.cols;
    if h == 0 {
        return raw.clone();
    }
    let mut seed_bytes = [0u8; 8];
    seed_bytes.copy_from_slice(&seed_hash[..8]);
    let mut rng = CommittedSeed(u64::from_le_bytes(seed_bytes));

    // Probe columns projected into the subspace: C = B (Bᵀ P).
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(h);
    for _ in 0..h {
        let probe: Vec<f64> = (0..m).map(|_| rng.next_signed_unit()).collect();
        // coeff = Bᵀ p
        let mut coeff = vec![0.0; h];
        for c in 0..h {
            let mut acc = 0.0;
            for r in 0..m {
                acc += raw.get(r, c) * probe[r];
            }
            coeff[c] = acc;
        }
        // col = B coeff
        let mut col = vec![0.0; m];
        for c in 0..h {
            let cc = coeff[c];
            if cc != 0.0 {
                for (r, cv) in col.iter_mut().enumerate() {
                    *cv += raw.get(r, c) * cc;
                }
            }
        }
        cols.push(col);
    }

    // Modified Gram–Schmidt; a probe that degenerates is replaced by the
    // corresponding raw basis column to keep the dimension intact.
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(h);
    let mut fallback = 0usize;
    for mut col in cols {
        loop {
            for prev in &out {
                let d: f64 = prev.iter().zip(col.iter()).map(|(a, b)| a * b).sum();
                for (c, p) in col.iter_mut().zip(prev.iter()) {
                    *c -= d * p;
                }
            }
            let n: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
            if n > 1e-10 {
                for c in col.iter_mut() {
                    *c /= n;
                }
                break;
            }
            // Degenerate probe: fall back to the next raw column.
            if fallback >= h {
                break;
            }
            col = (0..m).map(|r| raw.get(r, fallback)).collect();
            fallback += 1;
        }
        out.push(col);
    }

    // Sign canonicalization: first component with |v| > tol made positive.
    let mut basis = DenseMatrix::zeros(m, h);
    for (c, col) in out.iter().enumerate() {
        let tol = 1e-12;
        let sign = col
            .iter()
            .find(|v| v.abs() > tol)
            .map(|v| if *v < 0.0 { -1.0 } else { 1.0 })
            .unwrap_or(1.0);
        for (r, v) in col.iter().enumerate() {
            basis.set(r, c, sign * v);
        }
    }
    basis
}

/// Spectral summary of the sheaf Laplacian obtained with LOBPCG — never
/// dominant power iteration (issue #669, implementation gap 3).
#[derive(Debug, Clone)]
pub struct SpectralSummary {
    /// Smallest computed eigenvalues, ascending.
    pub smallest_eigenvalues: Vec<f64>,
    /// Estimated nullity (eigenvalues below `nullity_tol`).
    pub nullity_estimate: usize,
    /// Smallest eigenvalue above the nullity tolerance, if any (spectral gap).
    pub spectral_gap: Option<f64>,
    /// Residual certificate from the eigensolver.
    pub certificate: ResidualCertificate,
}

/// Compute the spectral summary of `L = δᵀWδ` for `H⁰` analysis: the
/// nullity of `L` is `dim H⁰ = dim ker δ` (the space of global sections).
pub fn spectral_summary(
    op: &BlockCoboundary,
    weights: &[f64],
    k: usize,
    seed: u64,
    nullity_tol: f64,
    cfg: &SolverConfig,
) -> SpectralSummary {
    let n = op.dim_c0();
    let result = smallest_eigenpairs(
        |x, y| op.apply_weighted_laplacian(weights, x, y),
        n,
        k.min(n),
        seed,
        cfg,
    );
    let nullity = result
        .eigenvalues
        .iter()
        .filter(|&&ev| ev.abs() < nullity_tol)
        .count();
    let gap = result
        .eigenvalues
        .iter()
        .find(|&&ev| ev.abs() >= nullity_tol)
        .copied();
    SpectralSummary {
        smallest_eigenvalues: result.eigenvalues,
        nullity_estimate: nullity,
        spectral_gap: gap,
        certificate: result.certificate,
    }
}
