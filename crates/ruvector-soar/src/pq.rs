//! 8-bit Product Quantizer (PQ) with Asymmetric Distance Computation (ADC).
//!
//! Splits a D-dimensional vector into M subspaces of D/M dimensions. Each
//! subspace has an independent 256-centroid codebook trained via k-means.
//! Encodes each vector as M bytes. At query time, precomputes a lookup table
//! T[m][256] and scores candidates via table lookups in O(M) per candidate.

use crate::error::{Result, SoarError};
use crate::kmeans::l2_sq;
use rand::SeedableRng;
use rand::prelude::*;

pub const PQ_K: usize = 256; // 1 byte per subspace

/// Product Quantizer: M subspaces, 256 centroids each.
#[derive(Clone)]
pub struct ProductQuantizer {
    /// Number of subspaces
    pub m: usize,
    /// Dimensions per subspace
    pub dsub: usize,
    /// Total dimensions
    pub dim: usize,
    /// Codebooks: [m][PQ_K][dsub]
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    pub fn new(dim: usize, m: usize) -> Result<Self> {
        if dim % m != 0 {
            return Err(SoarError::InvalidConfig(format!(
                "dim ({dim}) must be divisible by m ({m})"
            )));
        }
        Ok(Self {
            m,
            dsub: dim / m,
            dim,
            codebooks: Vec::new(),
        })
    }

    /// Train codebooks by running k-means on each subspace independently.
    pub fn train(&mut self, vectors: &[Vec<f32>], max_iter: usize, seed: u64) -> Result<()> {
        if vectors.is_empty() {
            return Err(SoarError::Empty);
        }
        if vectors[0].len() != self.dim {
            return Err(SoarError::DimensionMismatch {
                expected: self.dim,
                actual: vectors[0].len(),
            });
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        self.codebooks = Vec::with_capacity(self.m);

        for sub in 0..self.m {
            let start = sub * self.dsub;
            let end = start + self.dsub;
            // Extract subspace vectors
            let sub_vecs: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();
            // Use up to PQ_K centroids (or fewer if dataset is small)
            let k = PQ_K.min(sub_vecs.len());
            let codebook = train_subspace_kmeans(&sub_vecs, k, max_iter, rng.gen());
            self.codebooks.push(codebook);
        }
        Ok(())
    }

    /// Encode a vector as M bytes (one code per subspace).
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        (0..self.m)
            .map(|sub| {
                let start = sub * self.dsub;
                let slice = &v[start..start + self.dsub];
                let cb = &self.codebooks[sub];
                cb.iter()
                    .enumerate()
                    .map(|(i, c)| (i, l2_sq(slice, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Precompute lookup table T[m][256] of squared distances from query subvectors
    /// to each codebook centroid. Used for fast ADC scoring.
    pub fn distance_table(&self, query: &[f32]) -> Vec<[f32; PQ_K]> {
        (0..self.m)
            .map(|sub| {
                let start = sub * self.dsub;
                let qsub = &query[start..start + self.dsub];
                let mut row = [0.0f32; PQ_K];
                let cb = &self.codebooks[sub];
                for (k, centroid) in cb.iter().enumerate().take(PQ_K) {
                    row[k] = l2_sq(qsub, centroid);
                }
                row
            })
            .collect()
    }

    /// Estimate L2^2 distance from query to encoded vector using precomputed table.
    #[inline]
    pub fn adc_distance(&self, code: &[u8], table: &[[f32; PQ_K]]) -> f32 {
        code.iter().zip(table.iter()).map(|(&c, row)| row[c as usize]).sum()
    }

    pub fn is_trained(&self) -> bool {
        !self.codebooks.is_empty()
    }
}

// ── simple k-means for a single subspace ─────────────────────────────────────

fn train_subspace_kmeans(
    sub_vecs: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n = sub_vecs.len();
    let dim = sub_vecs[0].len();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Random initialisation (fast; k-means++ adds ~2× build time for marginal gain in subspaces)
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);
    let mut centers: Vec<Vec<f32>> = idx.iter().take(k).map(|&i| sub_vecs[i].clone()).collect();

    let mut assignments = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        for (i, v) in sub_vecs.iter().enumerate() {
            let best = centers
                .iter()
                .enumerate()
                .map(|(j, c)| (j, l2_sq(v, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap();
            if best != assignments[i] {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in sub_vecs.iter().enumerate() {
            let c = assignments[i];
            for (d, x) in sums[c].iter_mut().zip(v.iter()) {
                *d += x;
            }
            counts[c] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    centers[j][d] = sums[j][d] / counts[j] as f32;
                }
            }
        }
    }
    centers
}
