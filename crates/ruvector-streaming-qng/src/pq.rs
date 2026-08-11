//! Product Quantization (PQ) primitives shared by static and streaming variants.
//!
//! Layout: dims must be divisible by M (number of subspaces).
//! Each subspace has K centroids; codes stored as u8 (K ≤ 256).
//! Training: Lloyd's algorithm with TRAIN_ITERS iterations.

use crate::nearest_centroid;
use rand::SeedableRng;
use rand::seq::SliceRandom;

/// Number of subspaces.
pub const M: usize = 4;
/// Centroids per subspace (≤ 256 to fit u8).
pub const K: usize = 16;
/// K-means iterations during training.
pub const TRAIN_ITERS: usize = 20;

/// A trained PQ codebook.
#[derive(Clone)]
pub struct Codebook {
    pub m: usize,
    pub k: usize,
    pub ds: usize, // dims per subspace = total_dims / m
    /// centroids[subspace][centroid] = f32 slice of length ds
    pub centroids: Vec<Vec<Vec<f32>>>,
}

impl Codebook {
    /// Train a new codebook on `samples`. Panics if dims % m != 0.
    pub fn train(samples: &[Vec<f32>], m: usize, k: usize, seed: u64) -> Self {
        assert!(!samples.is_empty(), "cannot train on empty samples");
        let d = samples[0].len();
        assert_eq!(d % m, 0, "dims must be divisible by M (got d={d}, m={m})");
        let ds = d / m;

        let mut centroids = Vec::with_capacity(m);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        for sub in 0..m {
            let start = sub * ds;
            let end = start + ds;
            let slices: Vec<Vec<f32>> = samples
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            let k_eff = k.min(slices.len());
            // KMeans++ initialisation would be best; simple random sampling
            // works adequately when clusters are reasonably separated.
            let mut centers: Vec<Vec<f32>> =
                slices.choose_multiple(&mut rng, k_eff).cloned().collect();

            for _iter in 0..TRAIN_ITERS {
                let mut sums = vec![vec![0.0_f32; ds]; k_eff];
                let mut counts = vec![0usize; k_eff];
                for s in &slices {
                    let c = nearest_centroid(s, &centers);
                    for (a, b) in sums[c].iter_mut().zip(s.iter()) {
                        *a += b;
                    }
                    counts[c] += 1;
                }
                for c in 0..k_eff {
                    if counts[c] > 0 {
                        centers[c] =
                            sums[c].iter().map(|&s| s / counts[c] as f32).collect();
                    }
                }
            }
            centroids.push(centers);
        }

        Codebook { m, k: k.min(samples.len()), ds, centroids }
    }

    /// Encode a single vector into M u8 codes (one code per subspace).
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        (0..self.m)
            .map(|sub| {
                let start = sub * self.ds;
                let slice = &v[start..start + self.ds];
                nearest_centroid(slice, &self.centroids[sub]) as u8
            })
            .collect()
    }

    /// Build the ADC lookup table for a query.
    /// `table[sub][centroid_idx]` = sq_l2 from query subvector to that centroid.
    pub fn adc_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        (0..self.m)
            .map(|sub| {
                let start = sub * self.ds;
                let qsub = &query[start..start + self.ds];
                self.centroids[sub]
                    .iter()
                    .map(|c| crate::sq_l2(qsub, c))
                    .collect()
            })
            .collect()
    }

    /// Approximate distance via ADC lookup.
    #[inline]
    pub fn adc_dist(table: &[Vec<f32>], code: &[u8]) -> f32 {
        code.iter()
            .zip(table.iter())
            .map(|(&c, t)| t[c as usize])
            .sum()
    }

    /// One mini-batch k-means pass on `reservoir` to shift centroid positions.
    pub fn update_one_pass(&mut self, reservoir: &[Vec<f32>]) {
        if reservoir.is_empty() {
            return;
        }
        for sub in 0..self.m {
            let start = sub * self.ds;
            let end = start + self.ds;
            let slices: Vec<Vec<f32>> =
                reservoir.iter().map(|v| v[start..end].to_vec()).collect();

            let k_eff = self.centroids[sub].len();
            let mut sums = vec![vec![0.0_f32; self.ds]; k_eff];
            let mut counts = vec![0usize; k_eff];
            for s in &slices {
                let c = nearest_centroid(s, &self.centroids[sub]);
                for (a, b) in sums[c].iter_mut().zip(s.iter()) {
                    *a += b;
                }
                counts[c] += 1;
            }
            for c in 0..k_eff {
                if counts[c] > 0 {
                    // Exponential moving average: blend old centroid 30%, new mean 70%.
                    // This prevents over-eager churn on small reservoir updates.
                    let new_mean: Vec<f32> =
                        sums[c].iter().map(|&s| s / counts[c] as f32).collect();
                    for (old, new) in self.centroids[sub][c].iter_mut().zip(new_mean.iter()) {
                        *old = 0.30 * *old + 0.70 * new;
                    }
                }
            }
        }
    }
}
