//! Uniform-MSE Product Quantization. Baseline #2 — same code layout
//! as `AnisotropicPq` but trained with plain Lloyd's algorithm so the
//! only variable in our A/B is the loss function.

use crate::error::AvqError;
use crate::kmeans::kmeans_mse;
use crate::traits::{Encoder, Scorer};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// `m` subquantizers, each with `k` codewords (k <= 256).
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    pub(crate) dim: usize,
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) ds: usize, // sub-dim = dim / m
    /// Per-subspace centroids: `m` blocks of size `k * ds`.
    pub(crate) centroids: Vec<Vec<f32>>,
}

impl ProductQuantizer {
    pub fn fit(train: &[f32], dim: usize, m: usize, k: usize, seed: u64) -> Result<Self, AvqError> {
        if dim % m != 0 {
            return Err(AvqError::BadSubspace { dim, m });
        }
        if k == 0 || k > 256 {
            return Err(AvqError::BadK(k));
        }
        if train.is_empty() {
            return Err(AvqError::EmptyTrain);
        }
        let n = train.len() / dim;
        let ds = dim / m;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut centroids = Vec::with_capacity(m);
        for s in 0..m {
            // gather subspace slice
            let mut sub = Vec::with_capacity(n * ds);
            for i in 0..n {
                let off = i * dim + s * ds;
                sub.extend_from_slice(&train[off..off + ds]);
            }
            let (c, _) = kmeans_mse(&sub, n, ds, k, 25, &mut rng);
            centroids.push(c);
        }
        Ok(ProductQuantizer { dim, m, k, ds, centroids })
    }

    pub(crate) fn encode_row(&self, row: &[f32], code: &mut [u8]) {
        for s in 0..self.m {
            let sub = &row[s * self.ds..(s + 1) * self.ds];
            let cs = &self.centroids[s];
            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let cent = &cs[c * self.ds..(c + 1) * self.ds];
                let mut d = 0.0f32;
                for j in 0..self.ds {
                    let diff = sub[j] - cent[j];
                    d += diff * diff;
                }
                if d < best_d {
                    best_d = d;
                    best = c as u8;
                }
            }
            code[s] = best;
        }
    }

    /// Build the asymmetric distance lookup table for one query.
    /// Layout: `m` rows of `k` floats; entry `(s, c)` is `<query_s, centroid[s][c]>`.
    pub fn build_lut_ip(&self, query: &[f32]) -> Vec<f32> {
        let mut lut = vec![0.0f32; self.m * self.k];
        for s in 0..self.m {
            let qs = &query[s * self.ds..(s + 1) * self.ds];
            let cs = &self.centroids[s];
            for c in 0..self.k {
                let cent = &cs[c * self.ds..(c + 1) * self.ds];
                let mut acc = 0.0f32;
                for j in 0..self.ds {
                    acc += qs[j] * cent[j];
                }
                lut[s * self.k + c] = acc;
            }
        }
        lut
    }
}

impl Encoder for ProductQuantizer {
    fn code_size(&self) -> usize {
        self.m
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn encode(&self, xs: &[f32], codes: &mut [u8]) {
        let n = xs.len() / self.dim;
        for i in 0..n {
            self.encode_row(
                &xs[i * self.dim..(i + 1) * self.dim],
                &mut codes[i * self.m..(i + 1) * self.m],
            );
        }
    }
}

impl Scorer for ProductQuantizer {
    fn score_ip(&self, query: &[f32], codes: &[u8], out: &mut [f32]) {
        let lut = self.build_lut_ip(query);
        let n = codes.len() / self.m;
        for i in 0..n {
            let code = &codes[i * self.m..(i + 1) * self.m];
            let mut s = 0.0f32;
            for sub in 0..self.m {
                s += lut[sub * self.k + code[sub] as usize];
            }
            out[i] = s;
        }
    }
}
