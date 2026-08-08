//! AQ codebook: M sub-spaces, K centroids each.
//!
//! Two training modes:
//! - `Isotropic`: standard k-means on L2 (baseline).
//! - `Anisotropic { eta }`: ScaNN-style directional penalty — assignments
//!   minimise L2 + (eta-1) * (residual · x̂)², where x̂ = x/‖x‖.
//!   eta=1 recovers isotropic; eta≈2–4 is the practical range.
//!
//! ADC tables use inner product (not L2), matching cosine similarity search.

use crate::{dot, l2_norm, l2_sq};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub enum TrainMode {
    Isotropic,
    Anisotropic { eta: f32 },
}

#[derive(Debug, Clone)]
pub struct AqConfig {
    pub m: usize,
    pub k: usize,
    pub iterations: usize,
    pub seed: u64,
    pub mode: TrainMode,
}

impl AqConfig {
    pub fn isotropic(m: usize, k: usize) -> Self {
        assert!(k <= 256);
        Self {
            m,
            k,
            iterations: 30,
            seed: 42,
            mode: TrainMode::Isotropic,
        }
    }

    pub fn anisotropic(m: usize, k: usize, eta: f32) -> Self {
        assert!(k <= 256);
        assert!(eta >= 1.0, "eta must be >= 1.0");
        Self {
            m,
            k,
            iterations: 30,
            seed: 42,
            mode: TrainMode::Anisotropic { eta },
        }
    }

    pub fn sub_dim(&self, dim: usize) -> usize {
        assert!(dim % self.m == 0, "dim must be divisible by m");
        dim / self.m
    }
}

#[derive(Debug, Clone)]
pub struct AqCodebook {
    pub config: AqConfig,
    /// centroids[sub * k * sub_dim + c * sub_dim + d]
    pub centroids: Vec<f32>,
    pub dim: usize,
    pub sub_dim: usize,
}

impl AqCodebook {
    /// Train on L2-normalised vectors (flat row-major, n × dim).
    pub fn train(config: AqConfig, vectors: &[f32], dim: usize) -> Self {
        let n = vectors.len() / dim;
        assert!(n > 0);
        let sub_dim = config.sub_dim(dim);
        let m = config.m;
        let k = config.k;
        let eta = match config.mode {
            TrainMode::Isotropic => 1.0,
            TrainMode::Anisotropic { eta } => eta,
        };

        let mut centroids = vec![0.0f32; m * k * sub_dim];

        for sub in 0..m {
            let offset = sub * sub_dim;
            // sub-vectors for this partition
            let sub_vecs: Vec<Vec<f32>> = (0..n)
                .map(|i| vectors[i * dim + offset..i * dim + offset + sub_dim].to_vec())
                .collect();
            // full vectors needed for directional penalty
            let full_vecs: Vec<&[f32]> = (0..n).map(|i| &vectors[i * dim..(i + 1) * dim]).collect();

            let c = train_aq_kmeans(
                &sub_vecs,
                &full_vecs,
                sub,
                sub_dim,
                k,
                config.iterations,
                config.seed + sub as u64,
                eta,
            );
            let base = sub * k * sub_dim;
            centroids[base..base + k * sub_dim].copy_from_slice(&c);
        }

        Self {
            config,
            centroids,
            dim,
            sub_dim,
        }
    }

    #[inline]
    pub fn centroid(&self, sub: usize, c: usize) -> &[f32] {
        let start = sub * self.config.k * self.sub_dim + c * self.sub_dim;
        &self.centroids[start..start + self.sub_dim]
    }

    /// Assign sub-vector to best centroid using the configured loss.
    #[inline]
    pub fn assign(&self, sub: usize, sub_vec: &[f32], full_vec_norm: &[f32]) -> u8 {
        let eta = match self.config.mode {
            TrainMode::Isotropic => 1.0,
            TrainMode::Anisotropic { eta } => eta,
        };
        let k = self.config.k;
        let sub_norm = &full_vec_norm[sub * self.sub_dim..(sub + 1) * self.sub_dim];
        let mut best_c = 0usize;
        let mut best_loss = f32::MAX;
        for c in 0..k {
            let cent = self.centroid(sub, c);
            let loss = aq_loss(sub_vec, cent, sub_norm, eta);
            if loss < best_loss {
                best_loss = loss;
                best_c = c;
            }
        }
        best_c as u8
    }

    /// Nearest centroid by L2 only (for isotropic assignment at search time).
    #[inline]
    pub fn nearest_l2(&self, sub: usize, sub_vec: &[f32]) -> u8 {
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

    /// Build ADC table using inner product (for cosine similarity search).
    /// table[sub * k + c] = dot(q_sub, centroid_c)
    pub fn build_ip_adc_table(&self, query: &[f32]) -> Vec<f32> {
        let m = self.config.m;
        let k = self.config.k;
        let mut table = vec![0.0f32; m * k];
        for sub in 0..m {
            let q_sub = &query[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            for c in 0..k {
                table[sub * k + c] = dot(q_sub, self.centroid(sub, c));
            }
        }
        table
    }

    /// Approximate inner product score from PQ code + ADC table.
    #[inline]
    pub fn adc_score(&self, table: &[f32], code: &[u8]) -> f32 {
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

// Anisotropic loss: L2 + (eta-1) * (residual · sub_unit)²
// sub_unit = normalised sub-vector of full vector for this sub-space.
#[inline]
fn aq_loss(sub_vec: &[f32], centroid: &[f32], sub_unit: &[f32], eta: f32) -> f32 {
    let isotropic: f32 = sub_vec
        .iter()
        .zip(centroid)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    if eta == 1.0 {
        return isotropic;
    }
    let parallel: f32 = sub_vec
        .iter()
        .zip(centroid)
        .zip(sub_unit)
        .map(|((a, b), u)| (a - b) * u)
        .sum::<f32>();
    isotropic + (eta - 1.0) * parallel * parallel
}

/// Modified k-means using aq_loss for assignment.
/// sub_norms: unit sub-vectors extracted from each full vector for this sub-space.
fn train_aq_kmeans(
    sub_vecs: &[Vec<f32>],
    full_vecs: &[&[f32]],
    sub: usize,
    sub_dim: usize,
    k: usize,
    iterations: usize,
    seed: u64,
    eta: f32,
) -> Vec<f32> {
    let n = sub_vecs.len();
    let mut rng = StdRng::seed_from_u64(seed);

    // Pre-compute normalised sub-vectors for directional penalty.
    let sub_norms: Vec<Vec<f32>> = full_vecs
        .iter()
        .map(|fv| {
            let sv = &fv[sub * sub_dim..(sub + 1) * sub_dim];
            let norm = l2_norm(sv);
            sv.iter().map(|x| x / norm).collect()
        })
        .collect();

    // Forgy initialisation.
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k.min(n) {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    let mut centroids: Vec<f32> = indices
        .iter()
        .take(k)
        .flat_map(|&idx| sub_vecs[idx].iter().copied())
        .collect();
    while centroids.len() < k * sub_dim {
        centroids.extend_from_slice(&sub_vecs[rng.gen_range(0..n)]);
    }
    centroids.truncate(k * sub_dim);

    let mut assignments = vec![0usize; n];

    for _ in 0..iterations {
        // Assignment.
        for i in 0..n {
            let sv = &sub_vecs[i];
            let sn = &sub_norms[i];
            let mut best_c = 0usize;
            let mut best_loss = f32::MAX;
            for c in 0..k {
                let cent = &centroids[c * sub_dim..(c + 1) * sub_dim];
                let loss = aq_loss(sv, cent, sn, eta);
                if loss < best_loss {
                    best_loss = loss;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        // Update.
        let mut sums = vec![0.0f32; k * sub_dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..sub_dim {
                sums[c * sub_dim + d] += sub_vecs[i][d];
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

    fn synthetic_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let raw: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        // L2-normalise each row.
        let mut out = raw.clone();
        for i in 0..n {
            let row = &raw[i * dim..(i + 1) * dim];
            let norm = l2_norm(row);
            for d in 0..dim {
                out[i * dim + d] = row[d] / norm;
            }
        }
        out
    }

    #[test]
    fn isotropic_codebook_trains() {
        let vecs = synthetic_vectors(200, 64, 1);
        let cfg = AqConfig::isotropic(8, 16);
        let cb = AqCodebook::train(cfg, &vecs, 64);
        assert_eq!(cb.centroids.len(), 8 * 16 * 8);
    }

    #[test]
    fn anisotropic_codebook_trains() {
        let vecs = synthetic_vectors(200, 64, 2);
        let cfg = AqConfig::anisotropic(8, 16, 2.0);
        let cb = AqCodebook::train(cfg, &vecs, 64);
        assert_eq!(cb.centroids.len(), 8 * 16 * 8);
    }

    #[test]
    fn adc_table_is_finite() {
        let vecs = synthetic_vectors(100, 64, 3);
        let cfg = AqConfig::isotropic(8, 16);
        let cb = AqCodebook::train(cfg, &vecs, 64);
        let query: Vec<f32> = (0..64).map(|i| (i as f32).cos()).collect();
        let table = cb.build_ip_adc_table(&query);
        assert!(table.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn aq_loss_equals_l2_when_eta_is_one() {
        let sv = vec![1.0f32, 0.5, -0.3, 0.2];
        let cent = vec![0.9f32, 0.6, -0.2, 0.1];
        let sn = vec![0.5f32, 0.5, 0.5, 0.5];
        let iso = l2_sq(&sv, &cent);
        let aq = aq_loss(&sv, &cent, &sn, 1.0);
        assert!((iso - aq).abs() < 1e-6);
    }
}
