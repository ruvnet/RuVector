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
                    .map(|_| <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng) as f32)
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

    #[test]
    fn orthogonality() {
        let rot = RandomRotation::random(64, 42);
        let d = rot.dim;
        // Each row should be unit length.
        for i in 0..d {
            let row = &rot.matrix[i * d..(i + 1) * d];
            let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "row {i} norm = {norm}");
        }
        // Dot product of distinct rows should be ≈ 0.
        let row0 = &rot.matrix[0..d];
        let row1 = &rot.matrix[d..2 * d];
        let dot: f32 = row0.iter().zip(row1.iter()).map(|(&a, &b)| a * b).sum();
        assert!(dot.abs() < 1e-3, "rows 0,1 not orthogonal: dot={dot}");
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
}
