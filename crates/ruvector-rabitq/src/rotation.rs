//! Random orthogonal rotation drawn from the Haar distribution via QR decomposition.
//!
//! We use a thin QR via Gram-Schmidt so we stay dependency-free (no nalgebra required
//! at runtime). For D ≤ 2048 this is fast enough to build once and cache.

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// A DxD random orthogonal matrix stored in row-major order.
///
/// Applying it to a vector: `apply(&matrix, v)` costs O(D²) — build once, amortise.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomRotation {
    /// Flattened row-major D×D matrix.
    pub matrix: Vec<f32>,
    pub dim: usize,
}

impl RandomRotation {
    /// Sample a Haar-uniform orthogonal matrix of size `dim × dim`.
    pub fn random(dim: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        // Fill a dim×dim matrix with N(0,1) entries.
        let mut m: Vec<Vec<f32>> = (0..dim)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                            as f32
                    })
                    .collect()
            })
            .collect();

        // Gram–Schmidt orthonormalisation (in-place).
        for i in 0..dim {
            // Subtract projections of all previous basis vectors.
            for j in 0..i {
                let dot: f32 = (0..dim).map(|k| m[i][k] * m[j][k]).sum();
                for k in 0..dim {
                    let v = m[j][k];
                    m[i][k] -= dot * v;
                }
            }
            // Normalise.
            let norm: f32 = m[i].iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                m[i].iter_mut().for_each(|x| *x /= norm);
            }
        }

        let matrix: Vec<f32> = m.into_iter().flatten().collect();
        Self { matrix, dim }
    }

    /// Apply the rotation: out = P · v  (length must equal dim).
    #[inline]
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        debug_assert_eq!(v.len(), self.dim);
        let d = self.dim;
        let mut out = vec![0.0f32; d];
        for (i, out_i) in out.iter_mut().enumerate() {
            let row = &self.matrix[i * d..(i + 1) * d];
            *out_i = row.iter().zip(v.iter()).map(|(&r, &x)| r * x).sum();
        }
        out
    }

    /// Memory usage in bytes.
    pub fn bytes(&self) -> usize {
        self.matrix.len() * 4
    }
}

/// Fast in-place L2 normalisation.
pub fn normalize_inplace(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full orthogonality check — every pair of rows must be orthonormal.
    /// Stricter than the shipped version at `f2dbb6efb` which only tested
    /// (row 0, row 1).
    #[test]
    fn orthogonality_all_pairs_d64() {
        check_orthonormal(64, 42, 1e-4);
    }

    #[test]
    fn orthogonality_all_pairs_d128() {
        check_orthonormal(128, 7, 1e-4);
    }

    /// At D=256 classical Gram-Schmidt accumulates enough f32 round-off
    /// that we widen the tolerance to 1e-3 — still tight enough for the
    /// estimator not to drift but surfaces that GS is not numerically
    /// stable at large D. Reminder to move to Householder / modified GS
    /// when we start shipping D ≥ 1024.
    #[test]
    fn orthogonality_all_pairs_d256() {
        check_orthonormal(256, 11, 1e-3);
    }

    fn check_orthonormal(dim: usize, seed: u64, tol: f32) {
        let rot = RandomRotation::random(dim, seed);
        let d = rot.dim;
        for i in 0..d {
            let ri = &rot.matrix[i * d..(i + 1) * d];
            // Unit norm.
            let ni: f32 = ri.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((ni - 1.0).abs() < tol, "row {i} norm = {ni}, D={d}");
            // Orthogonal to all later rows.
            for j in (i + 1)..d {
                let rj = &rot.matrix[j * d..(j + 1) * d];
                let dot: f32 = ri.iter().zip(rj.iter()).map(|(&a, &b)| a * b).sum();
                assert!(dot.abs() < tol, "rows {i},{j} dot={dot}, D={d}");
            }
        }
    }

    #[test]
    fn apply_preserves_norm() {
        let rot = RandomRotation::random(128, 7);
        let v: Vec<f32> = (0..128_u32).map(|i| (i as f32).sin()).collect();
        let rv = rot.apply(&v);
        let norm_in: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_out: f32 = rv.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm_in - norm_out).abs() / norm_in < 1e-3);
    }

    /// Determinism: same seed + same dim → bit-identical rotation matrix.
    #[test]
    fn seed_reproducibility() {
        let a = RandomRotation::random(64, 1234);
        let b = RandomRotation::random(64, 1234);
        assert_eq!(a.matrix, b.matrix);
    }
}
