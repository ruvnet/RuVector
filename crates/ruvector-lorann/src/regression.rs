use nalgebra::{DMatrix, SVD};

use crate::error::{LorannError, Result};

/// Per-cluster reduced-rank regression model.
///
/// Stores the SVD factorisation of the cluster's document matrix:
///   X ≈ U_r Σ_r V_r^T   (X ∈ R^{m×d})
///
/// Score approximation for a query q:
///   X q ≈ A (B^T q)
/// where A = U_r Σ_r ∈ R^{m×r} and B = V_r ∈ R^{d×r}.
///
/// Query cost: r·d (compute B^T q) + m·r (expand via A) vs d·m for exact.
/// At r=32, d=128, m=200: 32×128 + 200×32 = 4096 + 6400 = 10496 vs 25600.
pub struct ClusterModel {
    /// A = U_r Σ_r, shape m×r, stored row-major.
    pub a: Vec<f32>,
    /// B = V_r, shape d×r, stored column-major (B^T is r×d, row-major).
    pub b_t: Vec<f32>,
    /// Number of documents in the cluster.
    pub m: usize,
    /// SVD rank used.
    pub rank: usize,
    /// Embedding dimension.
    pub dim: usize,
}

impl ClusterModel {
    /// Fit the per-cluster model by computing the truncated SVD of the
    /// doc matrix X ∈ R^{m×d}.
    pub fn fit(
        cluster_id: usize,
        docs: &[Vec<f32>],
        rank: usize,
    ) -> Result<Self> {
        let m = docs.len();
        let d = docs[0].len();
        let r = rank.min(m).min(d);

        if m < 2 {
            return Err(LorannError::ClusterTooSmall {
                id: cluster_id,
                size: m,
                min: 2,
                rank,
            });
        }

        // Build m × d f64 matrix (nalgebra SVD is more stable in f64)
        let flat: Vec<f64> = docs.iter().flat_map(|v| v.iter().map(|&x| x as f64)).collect();
        let matrix = DMatrix::from_row_slice(m, d, &flat);

        let svd = SVD::new(matrix, true, true);

        let u = svd.u.ok_or(LorannError::SvdFailed {
            cluster_id,
            rows: m,
            cols: d,
            rank: r,
        })?;
        let sigma = svd.singular_values;
        let vt = svd.v_t.ok_or(LorannError::SvdFailed {
            cluster_id,
            rows: m,
            cols: d,
            rank: r,
        })?;

        // A[i, j] = U[i, j] * sigma[j]  (m × r)
        let mut a = vec![0.0f32; m * r];
        for i in 0..m {
            for j in 0..r {
                a[i * r + j] = (u[(i, j)] * sigma[j]) as f32;
            }
        }

        // B^T[j, col] = V^T[j, col]  (r × d), so B = V ∈ R^{d×r}
        let mut b_t = vec![0.0f32; r * d];
        for j in 0..r {
            for col in 0..d {
                b_t[j * d + col] = vt[(j, col)] as f32;
            }
        }

        Ok(Self { a, b_t, m, rank: r, dim: d })
    }

    /// Approximate inner products of all m docs with `query`.
    ///
    /// Returns a Vec of length m with approximate scores (not distances).
    pub fn approximate_scores(&self, query: &[f32]) -> Vec<f32> {
        // Step 1: p = B^T q ∈ R^r  (r × d) · (d) = (r)
        let mut p = vec![0.0f32; self.rank];
        for j in 0..self.rank {
            let row_start = j * self.dim;
            let row = &self.b_t[row_start..row_start + self.dim];
            p[j] = row.iter().zip(query.iter()).map(|(b, q)| b * q).sum();
        }

        // Step 2: scores = A p ∈ R^m  (m × r) · (r) = (m)
        let mut scores = vec![0.0f32; self.m];
        for i in 0..self.m {
            let row_start = i * self.rank;
            let row = &self.a[row_start..row_start + self.rank];
            scores[i] = row.iter().zip(p.iter()).map(|(a, pp)| a * pp).sum();
        }

        scores
    }

    /// Bytes used by this model (A + B^T matrices only).
    pub fn memory_bytes(&self) -> usize {
        (self.a.len() + self.b_t.len()) * 4
    }
}
