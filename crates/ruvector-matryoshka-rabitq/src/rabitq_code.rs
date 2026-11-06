//! Minimal RaBitQ-style 1-bit quantization for the *coarse* stage of a
//! Matryoshka funnel (arXiv:2405.12497, Gao & Long, SIGMOD 2024).
//!
//! This is a small, self-contained reimplementation (not a dependency on
//! `ruvector-rabitq`) scoped to exactly what the cascade needs: a random
//! orthogonal rotation to decorrelate dimensions, a 1-bit sign code per
//! dimension, and the symmetric Charikar-style angular distance estimator
//! used to rank candidates before the full-dimension rerank stage.
//!
//! The estimator does not need to be exact — it only has to produce a
//! candidate *set* that is cheap to compute and likely to contain the true
//! top-k, because the final ranking always comes from an exact full-dim
//! rerank downstream.

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// A Haar-uniform random orthogonal `dim x dim` rotation, built once per
/// index and applied to every stored/query vector before binarisation.
pub struct RandomRotation {
    matrix: Vec<f32>,
    pub dim: usize,
}

impl RandomRotation {
    /// Sample a Haar-uniform orthogonal matrix via Gram-Schmidt on an i.i.d.
    /// Gaussian block. `O(dim^3)` to build, `O(dim^2)` to apply — fine at the
    /// small `coarse_dim` (32-64) this cascade runs the rotation at.
    pub fn random(dim: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
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

        for i in 0..dim {
            for j in 0..i {
                let dot: f32 = m[i].iter().zip(m[j].iter()).map(|(&a, &b)| a * b).sum();
                let (rows_before_i, rows_from_i) = m.split_at_mut(i);
                let row_i = &mut rows_from_i[0];
                let row_j = &rows_before_i[j];
                for (vi, &vj) in row_i.iter_mut().zip(row_j.iter()) {
                    *vi -= dot * vj;
                }
            }
            let norm: f32 = m[i].iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                m[i].iter_mut().for_each(|x| *x /= norm);
            }
        }

        Self {
            matrix: m.into_iter().flatten().collect(),
            dim,
        }
    }

    /// out = R * v
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        debug_assert_eq!(v.len(), self.dim);
        let d = self.dim;
        (0..d)
            .map(|i| {
                let row = &self.matrix[i * d..(i + 1) * d];
                row.iter().zip(v.iter()).map(|(&r, &x)| r * x).sum()
            })
            .collect()
    }

    /// Bytes of the one-time rotation matrix (amortised over all `n` stored
    /// vectors; reported separately in the benchmark's memory accounting).
    pub fn bytes(&self) -> usize {
        self.matrix.len() * 4
    }
}

/// A packed 1-bit-per-dimension code plus the pre-rotation L2 norm, which the
/// estimator needs to reconstruct an approximate squared-L2 distance.
#[derive(Clone)]
pub struct BinaryCode {
    words: Vec<u64>,
    norm: f32,
    dim: usize,
}

impl BinaryCode {
    /// `rotated` must already be `rotation.apply(&prefix)` for an
    /// L2-normalised prefix; `norm` is the pre-normalisation L2 norm of that
    /// prefix (needed to rescale the angular estimate back to squared-L2).
    pub fn encode(rotated: &[f32], norm: f32) -> Self {
        let dim = rotated.len();
        let n_words = dim.div_ceil(64);
        let mut words = vec![0u64; n_words];
        for (i, &v) in rotated.iter().enumerate() {
            if v >= 0.0 {
                words[i / 64] |= 1u64 << (63 - (i % 64));
            }
        }
        Self { words, norm, dim }
    }

    /// Padding-safe XNOR-popcount agreement count (0..=dim).
    #[inline]
    fn agreement(&self, other: &Self) -> u32 {
        let n_words = self.words.len();
        if n_words == 0 {
            return 0;
        }
        let mut sum = 0u32;
        for i in 0..n_words - 1 {
            sum += (!(self.words[i] ^ other.words[i])).count_ones();
        }
        let valid_bits = self.dim - 64 * (n_words - 1);
        let mask: u64 = if valid_bits == 64 {
            !0u64
        } else {
            !0u64 << (64 - valid_bits)
        };
        let last = !(self.words[n_words - 1] ^ other.words[n_words - 1]) & mask;
        sum + last.count_ones()
    }

    /// Estimated squared-L2 distance via the symmetric Charikar-style
    /// angular estimator: `E[agreement/dim] = 1 - theta/pi`, so
    /// `cos(theta) ~= cos(pi * (1 - agreement/dim))`, rescaled by the two
    /// stored norms into a squared-L2 estimate. Monotone in the true angle,
    /// which is all a candidate-ranking stage needs.
    #[inline]
    pub fn estimated_sq_distance(&self, query: &Self) -> f32 {
        use std::f32::consts::PI;
        let d = self.dim as f32;
        let agree = self.agreement(query) as f32;
        let est_cos = (PI * (1.0 - agree / d)).cos();
        let est_ip = self.norm * query.norm * est_cos;
        query.norm * query.norm + self.norm * self.norm - 2.0 * est_ip
    }

    /// Packed-words memory footprint of this one code.
    pub fn bytes(&self) -> usize {
        self.words.len() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_preserves_norm() {
        let rot = RandomRotation::random(32, 7);
        let v: Vec<f32> = (0..32u32).map(|i| (i as f32).sin()).collect();
        let rv = rot.apply(&v);
        let n_in: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let n_out: f32 = rv.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((n_in - n_out).abs() / n_in < 1e-3);
    }

    #[test]
    fn self_distance_is_near_zero() {
        let rot = RandomRotation::random(40, 1);
        let v: Vec<f32> = (0..40u32).map(|i| (i as f32 / 40.0).cos()).collect();
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = v.iter().map(|&x| x / norm).collect();
        let rotated = rot.apply(&unit);
        let code = BinaryCode::encode(&rotated, norm);
        assert!(code.estimated_sq_distance(&code).abs() < 1e-4);
    }

    #[test]
    fn closer_vector_estimates_closer() {
        use rand::{Rng as _, SeedableRng as _};
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let dim = 32;
        let rot = RandomRotation::random(dim, 5);

        let encode = |v: &[f32]| -> BinaryCode {
            let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let unit: Vec<f32> = v.iter().map(|&x| x / norm).collect();
            BinaryCode::encode(&rot.apply(&unit), norm)
        };

        let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let near: Vec<f32> = q.iter().map(|&x| x + rng.gen::<f32>() * 0.05).collect();
        let far: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

        let qc = encode(&q);
        let near_c = encode(&near);
        let far_c = encode(&far);
        assert!(near_c.estimated_sq_distance(&qc) < far_c.estimated_sq_distance(&qc));
    }

    #[test]
    fn non_64_aligned_dim_masks_padding() {
        // dim=40 -> 1 word, 24 padding bits that must not count as agreement.
        let a = BinaryCode::encode(&[1.0f32; 40], 1.0);
        let neg: Vec<f32> = vec![-1.0f32; 40];
        let b = BinaryCode::encode(&neg, 1.0);
        assert_eq!(
            a.agreement(&b),
            0,
            "opposite codes must have 0 agreement, padding leaking in"
        );
        assert_eq!(a.agreement(&a), 40);
    }
}
