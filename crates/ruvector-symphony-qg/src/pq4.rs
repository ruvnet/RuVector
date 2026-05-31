//! 4-bit Product Quantizer: trains M subspace codebooks (k=16 centroids each),
//! encodes float vectors to packed nibble codes, and builds per-query look-up tables.
//!
//! Layout: a vector of dimension D is split into M subvectors of dimension D/M.
//! Each subvector is assigned to its nearest centroid (0..15), stored as 4 bits.
//! Two consecutive codes are packed into one byte: low nibble = first, high = second.

use crate::error::SqgError;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const CENTROIDS_PER_SUBSPACE: usize = 16;

/// Trained 4-bit product quantizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pq4 {
    /// Number of subspaces.
    pub m: usize,
    /// Dimension of each subvector (dim / m).
    pub d_sub: usize,
    /// Flattened centroids: [m][16][d_sub], row-major.
    pub centroids: Vec<f32>,
}

impl Pq4 {
    /// Train on `vectors` (row-major, shape [n, dim]) with `m` subspaces.
    pub fn train(vectors: &[f32], dim: usize, m: usize, iters: usize) -> Result<Self, SqgError> {
        if dim % m != 0 {
            return Err(SqgError::Config(format!("dim {dim} not divisible by m={m}")));
        }
        let n = vectors.len() / dim;
        let d_sub = dim / m;
        let k = CENTROIDS_PER_SUBSPACE;
        let mut rng = rand::thread_rng();

        let mut all_centroids = vec![0f32; m * k * d_sub];

        for sub in 0..m {
            let off = sub * d_sub;
            // Collect this subspace's training data.
            let data: Vec<f32> = (0..n)
                .flat_map(|i| {
                    let base = i * dim + off;
                    vectors[base..base + d_sub].iter().copied()
                })
                .collect();

            // Seed centroids from random training samples.
            let mut centroids: Vec<f32> = (0..k)
                .flat_map(|_| {
                    let idx = rng.gen_range(0..n);
                    data[idx * d_sub..(idx + 1) * d_sub].iter().copied()
                })
                .collect();

            for _ in 0..iters {
                // Assignment step.
                let assign: Vec<usize> = (0..n)
                    .map(|i| {
                        let sv = &data[i * d_sub..(i + 1) * d_sub];
                        nearest_centroid(sv, &centroids, d_sub)
                    })
                    .collect();

                // Update step.
                let mut sums = vec![0f32; k * d_sub];
                let mut counts = vec![0usize; k];
                for (i, &c) in assign.iter().enumerate() {
                    let sv = &data[i * d_sub..(i + 1) * d_sub];
                    for (j, &v) in sv.iter().enumerate() {
                        sums[c * d_sub + j] += v;
                    }
                    counts[c] += 1;
                }
                for c in 0..k {
                    if counts[c] > 0 {
                        for j in 0..d_sub {
                            centroids[c * d_sub + j] = sums[c * d_sub + j] / counts[c] as f32;
                        }
                    }
                }
            }

            let base = sub * k * d_sub;
            all_centroids[base..base + k * d_sub].copy_from_slice(&centroids);
        }

        Ok(Self { m, d_sub, centroids: all_centroids })
    }

    /// Encode a single vector to packed nibble codes (m/2 bytes, rounded up).
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let k = CENTROIDS_PER_SUBSPACE;
        let byte_count = (self.m + 1) / 2;
        let mut codes = vec![0u8; byte_count];
        for sub in 0..self.m {
            let off = sub * self.d_sub;
            let sv = &vector[off..off + self.d_sub];
            let c_base = sub * k * self.d_sub;
            let centroids_sub = &self.centroids[c_base..c_base + k * self.d_sub];
            let code = nearest_centroid(sv, centroids_sub, self.d_sub) as u8;
            let byte_idx = sub / 2;
            if sub % 2 == 0 {
                codes[byte_idx] = code & 0x0F;
            } else {
                codes[byte_idx] |= (code & 0x0F) << 4;
            }
        }
        codes
    }

    /// Build a look-up table for `query`: for each subspace s, compute L2² to each centroid.
    /// Returns flat array [m * 16], values scaled to u8 (0..255).
    pub fn build_lut(&self, query: &[f32]) -> Vec<u8> {
        let k = CENTROIDS_PER_SUBSPACE;
        let mut lut = vec![0u8; self.m * k];
        let mut max_dist = f32::NEG_INFINITY;

        // Two-pass: collect raw distances, then quantize to u8.
        let mut raw = vec![0f32; self.m * k];
        for sub in 0..self.m {
            let off = sub * self.d_sub;
            let qsv = &query[off..off + self.d_sub];
            let c_base = sub * k * self.d_sub;
            for c in 0..k {
                let cent = &self.centroids[c_base + c * self.d_sub..c_base + (c + 1) * self.d_sub];
                let d: f32 = qsv.iter().zip(cent).map(|(a, b)| (a - b).powi(2)).sum();
                raw[sub * k + c] = d;
                if d > max_dist {
                    max_dist = d;
                }
            }
        }
        let scale = if max_dist > 0.0 { 255.0 / max_dist } else { 1.0 };
        for (i, &r) in raw.iter().enumerate() {
            lut[i] = (r * scale).min(255.0) as u8;
        }
        lut
    }

    /// Decode packed codes back to reconstructed float vector (for re-rank).
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let k = CENTROIDS_PER_SUBSPACE;
        let mut out = vec![0f32; self.m * self.d_sub];
        for sub in 0..self.m {
            let byte_idx = sub / 2;
            let nibble = if sub % 2 == 0 { codes[byte_idx] & 0x0F } else { (codes[byte_idx] >> 4) & 0x0F } as usize;
            let c_base = sub * k * self.d_sub + nibble * self.d_sub;
            let dst = sub * self.d_sub;
            out[dst..dst + self.d_sub].copy_from_slice(&self.centroids[c_base..c_base + self.d_sub]);
        }
        out
    }
}

fn nearest_centroid(sv: &[f32], centroids: &[f32], d_sub: usize) -> usize {
    let k = centroids.len() / d_sub;
    let mut best = 0;
    let mut best_d = f32::MAX;
    for c in 0..k {
        let base = c * d_sub;
        let d: f32 = sv.iter().zip(&centroids[base..base + d_sub]).map(|(a, b)| (a - b).powi(2)).sum();
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    fn random_vecs(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0f32, 1.0).unwrap();
        (0..n * d).map(|_| dist.sample(&mut rng)).collect()
    }

    #[test]
    fn train_and_encode_roundtrip() {
        let d = 16;
        let m = 4;
        let data = random_vecs(200, d, 42);
        let pq = Pq4::train(&data, d, m, 10).unwrap();
        assert_eq!(pq.m, m);
        assert_eq!(pq.d_sub, 4);

        let v = &data[..d];
        let codes = pq.encode(v);
        assert_eq!(codes.len(), (m + 1) / 2);
        let decoded = pq.decode(&codes);
        assert_eq!(decoded.len(), d);
    }

    #[test]
    fn lut_length() {
        let d = 32;
        let m = 8;
        let data = random_vecs(200, d, 7);
        let pq = Pq4::train(&data, d, m, 5).unwrap();
        let q = random_vecs(1, d, 99);
        let lut = pq.build_lut(&q);
        assert_eq!(lut.len(), m * CENTROIDS_PER_SUBSPACE);
    }

    #[test]
    fn decode_is_approximate() {
        let d = 32;
        let m = 8;
        let data = random_vecs(500, d, 1);
        let pq = Pq4::train(&data, d, m, 20).unwrap();
        let v = &data[..d];
        let codes = pq.encode(v);
        let decoded = pq.decode(&codes);
        // Decoded should be within some L2 error of original.
        let err: f32 = v.iter().zip(&decoded).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        // Loose bound — 4-bit PQ won't be exact but should be bounded.
        assert!(err < 10.0, "reconstruction error {err} too large");
    }
}
