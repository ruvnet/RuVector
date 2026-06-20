//! PQ codebook: M sub-spaces, K centroids each, trained with Lloyd's k-means.

use crate::l2_sq;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for product quantization.
#[derive(Debug, Clone)]
pub struct PqConfig {
    /// Number of sub-spaces.
    pub m: usize,
    /// Number of centroids per sub-space (must be ≤ 256 for u8 codes).
    pub k: usize,
    /// Number of Lloyd's iterations.
    pub iterations: usize,
    /// RNG seed (deterministic builds).
    pub seed: u64,
}

impl PqConfig {
    pub fn new(m: usize, k: usize) -> Self {
        assert!(k <= 256, "k must be ≤ 256 to fit in u8 code");
        Self {
            m,
            k,
            iterations: 25,
            seed: 42,
        }
    }

    /// Dimension of each sub-vector given total dimension `dim`.
    pub fn sub_dim(&self, dim: usize) -> usize {
        assert!(dim % self.m == 0, "dim must be divisible by m");
        dim / self.m
    }
}

/// Trained PQ codebook: `m` sub-codebooks, each with `k` centroids of `sub_dim` dims.
#[derive(Debug, Clone)]
pub struct PqCodebook {
    pub config: PqConfig,
    /// Flat layout: centroids[m][k][d] stored as centroids[m * k * sub_dim + c * sub_dim + d]
    pub centroids: Vec<f32>,
    /// Total vector dimension.
    pub dim: usize,
    /// Sub-vector dimension.
    pub sub_dim: usize,
}

impl PqCodebook {
    /// Train a PQ codebook from a slice of flat row-major vectors.
    /// `vectors`: n × dim flat layout.
    pub fn train(config: PqConfig, vectors: &[f32], dim: usize) -> Self {
        let n = vectors.len() / dim;
        assert!(n > 0, "need at least one vector to train");
        let sub_dim = config.sub_dim(dim);
        let m = config.m;
        let k = config.k;

        let mut centroids = vec![0.0f32; m * k * sub_dim];

        for sub in 0..m {
            let offset_start = sub * sub_dim;
            // Extract sub-vectors for this sub-space.
            let sub_vecs: Vec<&[f32]> = (0..n)
                .map(|i| &vectors[i * dim + offset_start..i * dim + offset_start + sub_dim])
                .collect();

            let c = train_kmeans(&sub_vecs, k, config.iterations, config.seed + sub as u64);

            let cent_offset = sub * k * sub_dim;
            centroids[cent_offset..cent_offset + k * sub_dim].copy_from_slice(&c);
        }

        Self {
            config,
            centroids,
            dim,
            sub_dim,
        }
    }

    /// Return centroid `c` in sub-space `sub` as a slice.
    #[inline]
    pub fn centroid(&self, sub: usize, c: usize) -> &[f32] {
        let start = sub * self.config.k * self.sub_dim + c * self.sub_dim;
        &self.centroids[start..start + self.sub_dim]
    }

    /// Find the nearest centroid index for a sub-vector in sub-space `sub`.
    #[inline]
    pub fn nearest_centroid(&self, sub: usize, sub_vec: &[f32]) -> u8 {
        let k = self.config.k;
        let mut best_c = 0usize;
        let mut best_d = f32::MAX;
        for c in 0..k {
            let d = l2_sq(sub_vec, self.centroid(sub, c));
            if d < best_d {
                best_d = d;
                best_c = c;
            }
        }
        best_c as u8
    }

    /// Build ADC lookup table for query `q`: m × k distances.
    /// `table[sub * k + c]` = L2² between q's sub-vector and centroid c.
    pub fn build_adc_table(&self, query: &[f32]) -> Vec<f32> {
        let m = self.config.m;
        let k = self.config.k;
        let mut table = vec![0.0f32; m * k];
        for sub in 0..m {
            let q_sub = &query[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            for c in 0..k {
                table[sub * k + c] = l2_sq(q_sub, self.centroid(sub, c));
            }
        }
        table
    }

    /// Compute ADC approximate distance for a PQ code using precomputed table.
    #[inline]
    pub fn adc_distance(&self, table: &[f32], code: &[u8]) -> f32 {
        let k = self.config.k;
        code.iter()
            .enumerate()
            .map(|(sub, &c)| table[sub * k + c as usize])
            .sum()
    }

    pub fn memory_bytes(&self) -> usize {
        self.centroids.len() * 4
    }
}

/// Lloyd's k-means on a set of sub-vectors. Returns flat centroid array (k × sub_dim).
fn train_kmeans(vecs: &[&[f32]], k: usize, iterations: usize, seed: u64) -> Vec<f32> {
    let n = vecs.len();
    let sub_dim = vecs[0].len();
    let mut rng = StdRng::seed_from_u64(seed);

    // Forgy initialization: pick k distinct random vectors.
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k.min(n) {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    let mut centroids: Vec<f32> = indices
        .iter()
        .take(k)
        .flat_map(|&idx| vecs[idx].iter().copied())
        .collect();
    // Pad to exactly k centroids if n < k.
    while centroids.len() < k * sub_dim {
        centroids.extend_from_slice(vecs[rng.gen_range(0..n)]);
    }
    centroids.truncate(k * sub_dim);

    let mut assignments = vec![0usize; n];

    for _ in 0..iterations {
        // Assignment step.
        for (i, v) in vecs.iter().enumerate() {
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for c in 0..k {
                let cent = &centroids[c * sub_dim..(c + 1) * sub_dim];
                let d = l2_sq(v, cent);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        // Update step.
        let mut sums = vec![0.0f32; k * sub_dim];
        let mut counts = vec![0usize; k];
        for (i, v) in vecs.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..sub_dim {
                sums[c * sub_dim + d] += v[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..sub_dim {
                    centroids[c * sub_dim + d] = sums[c * sub_dim + d] * inv;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_trains_without_panic() {
        let dim = 64usize;
        let n = 200usize;
        let config = PqConfig::new(8, 16);
        let vecs: Vec<f32> = (0..n * dim).map(|i| i as f32 * 0.01).collect();
        let cb = PqCodebook::train(config, &vecs, dim);
        assert_eq!(cb.centroids.len(), 8 * 16 * 8);
    }

    #[test]
    fn adc_table_sum_is_non_negative() {
        let dim = 64usize;
        let n = 100usize;
        let config = PqConfig::new(8, 16);
        let vecs: Vec<f32> = (0..n * dim).map(|i| (i as f32).sin()).collect();
        let cb = PqCodebook::train(config, &vecs, dim);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let table = cb.build_adc_table(&query);
        assert!(table.iter().all(|&v| v >= 0.0));
    }
}
