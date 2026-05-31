//! Fixed Dimensional Encoding (FDE) for multi-vector sets.
//!
//! Algorithm (MUVERA, NeurIPS 2024):
//!   1. Sample R random unit vectors ("reps") from N(0,I_D).
//!   2. For each token vector v in a document/query:
//!      a. Find the rep r* with maximum inner product <v, r*>.
//!      b. Add v to the accumulator slot for r*.
//!   3. Concatenate all R accumulators → one vector of dimension R×D.
//!
//! Property: IP(FDE_q, FDE_d) ≈ MaxSim(Q, D) where MaxSim is the ColBERT
//! similarity ∑_{q∈Q} max_{d∈D} <q, d> / |Q|.

use crate::error::MuveraError;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Holds the random projection matrix (R × D) used to assign tokens to slots.
pub struct FdeEncoder {
    /// Row-major R×D matrix of random unit vectors.
    projections: Vec<f32>,
    pub num_reps: usize,
    pub orig_dim: usize,
    /// Output dimensionality: num_reps × orig_dim.
    pub fde_dim: usize,
}

impl FdeEncoder {
    /// Build a new encoder with `num_reps` random projections for `orig_dim`-dimensional tokens.
    pub fn new(num_reps: usize, orig_dim: usize, seed: u64) -> Result<Self, MuveraError> {
        if num_reps < 1 {
            return Err(MuveraError::InvalidNumReps { num_reps });
        }
        if orig_dim < 1 {
            return Err(MuveraError::InvalidDim { orig_dim });
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0_f32, 1.0).unwrap();
        let total = num_reps * orig_dim;
        let mut raw: Vec<f32> = (0..total).map(|_| normal.sample(&mut rng)).collect();

        // L2-normalize each row so inner product equals cosine similarity.
        for r in 0..num_reps {
            let start = r * orig_dim;
            let row = &mut raw[start..start + orig_dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-9 {
                row.iter_mut().for_each(|x| *x /= norm);
            }
        }
        Ok(Self {
            projections: raw,
            num_reps,
            orig_dim,
            fde_dim: num_reps * orig_dim,
        })
    }

    /// Return the index (0..num_reps) of the rep with highest IP with `vec`.
    #[inline]
    fn nearest_rep(&self, vec: &[f32]) -> usize {
        let mut best_rep = 0usize;
        let mut best_ip = f32::NEG_INFINITY;
        for r in 0..self.num_reps {
            let start = r * self.orig_dim;
            let row = &self.projections[start..start + self.orig_dim];
            let ip: f32 = row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            if ip > best_ip {
                best_ip = ip;
                best_rep = r;
            }
        }
        best_rep
    }

    /// Encode a set of token vectors into a single Fixed Dimensional Encoding.
    ///
    /// `token_vecs` — slice of token vectors, each of length `orig_dim`.
    /// Returns a vector of length `fde_dim` (= num_reps × orig_dim).
    pub fn encode(&self, token_vecs: &[Vec<f32>]) -> Vec<f32> {
        let mut accum = vec![0.0_f32; self.fde_dim];
        for v in token_vecs {
            let rep = self.nearest_rep(v);
            let start = rep * self.orig_dim;
            for (i, &x) in v.iter().enumerate() {
                accum[start + i] += x;
            }
        }
        accum
    }

    /// Encode and L2-normalize (useful for cosine-IP equivalence check).
    pub fn encode_normalized(&self, token_vecs: &[Vec<f32>]) -> Vec<f32> {
        let mut fde = self.encode(token_vecs);
        let norm: f32 = fde.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            fde.iter_mut().for_each(|x| *x /= norm);
        }
        fde
    }

    /// Exact MaxSim between two token sets (ground-truth for recall evaluation).
    /// MaxSim(Q, D) = (1/|Q|) ∑_{q∈Q} max_{d∈D} <q, d>
    pub fn max_sim(query_vecs: &[Vec<f32>], doc_vecs: &[Vec<f32>]) -> f32 {
        if query_vecs.is_empty() || doc_vecs.is_empty() {
            return 0.0;
        }
        let sum: f32 = query_vecs
            .iter()
            .map(|q| {
                doc_vecs
                    .iter()
                    .map(|d| q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>())
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum();
        sum / query_vecs.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fde_dim_is_correct() {
        let enc = FdeEncoder::new(8, 16, 42).unwrap();
        assert_eq!(enc.fde_dim, 128);
    }

    #[test]
    fn encode_returns_correct_length() {
        let enc = FdeEncoder::new(4, 8, 1).unwrap();
        let vecs = vec![vec![1.0_f32; 8], vec![-1.0_f32; 8]];
        let fde = enc.encode(&vecs);
        assert_eq!(fde.len(), 32);
    }

    #[test]
    fn projections_are_unit_length() {
        let enc = FdeEncoder::new(10, 16, 7).unwrap();
        for r in 0..enc.num_reps {
            let start = r * enc.orig_dim;
            let row = &enc.projections[start..start + enc.orig_dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "rep {r} norm={norm}");
        }
    }

    #[test]
    fn max_sim_self_is_positive() {
        let vecs = vec![vec![1.0_f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let s = FdeEncoder::max_sim(&vecs, &vecs);
        assert!(s > 0.9, "self MaxSim should be ~1.0, got {s}");
    }
}
