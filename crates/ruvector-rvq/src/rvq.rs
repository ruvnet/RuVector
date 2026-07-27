//! Residual Vector Quantizer: chains multiple codebooks so each stage
//! quantizes the residual left by the previous stage.
//!
//! Training follows the greedy forward algorithm:
//! 1. Train codebook 0 on the raw data.
//! 2. Compute residuals: r_i = v_i - centroid_0[encode_0(v_i)].
//! 3. Train codebook 1 on residuals.
//! 4. Repeat until `num_stages` codebooks are trained.
//!
//! Codebook dropout (arXiv:2306.06546 §3.2): during each stage's training
//! data construction, each previous-stage code is randomly zeroed (replaced
//! with its centroid set to the zero vector) with probability `dropout_prob`.
//! This prevents later codebooks from becoming dead and improves recall
//! variance on rare distributions.

use rand::SeedableRng;
use rand::Rng as _;

use crate::{
    codebook::{dot, l2_sq, l2_sq_self, Codebook},
    RvqConfig,
};

/// Trained RVQ encoder.  Contains `config.num_stages` codebooks.
#[derive(Debug, Clone)]
pub struct RvqEncoder {
    pub codebooks: Vec<Codebook>,
    pub config: RvqConfig,
}

impl RvqEncoder {
    /// Train a full RVQ encoder on `data`.
    pub fn train(config: RvqConfig, data: &[Vec<f32>]) -> Self {
        assert!(!data.is_empty(), "RVQ training requires data");
        assert!(config.codebook_size <= 256, "codebook_size must be ≤ 256 (fits u8)");

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut residuals: Vec<Vec<f32>> = data.to_vec();
        let mut codebooks = Vec::with_capacity(config.num_stages);

        for stage in 0..config.num_stages {
            // Apply codebook dropout: randomly zero residuals from previous stages
            // so this stage doesn't lean on a fixed prior pattern.
            let train_data: Vec<Vec<f32>> = if stage > 0 && config.dropout_prob > 0.0 {
                residuals
                    .iter()
                    .map(|r| {
                        if rng.gen::<f32>() < config.dropout_prob {
                            vec![0.0f32; config.dim]
                        } else {
                            r.clone()
                        }
                    })
                    .collect()
            } else {
                residuals.clone()
            };

            let cb = Codebook::train(
                &train_data,
                config.codebook_size,
                config.dim,
                config.train_iters,
                42 + stage as u64,
            );

            // Update residuals: subtract this stage's quantisation.
            residuals = residuals
                .iter()
                .map(|v| {
                    let c = cb.encode(v) as usize;
                    let centroid = cb.centroid(c);
                    v.iter().zip(centroid).map(|(a, b)| a - b).collect()
                })
                .collect();

            codebooks.push(cb);
        }

        RvqEncoder { codebooks, config }
    }

    /// Encode `v` into `num_stages` u8 codes.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut residual = v.to_vec();
        let mut codes = Vec::with_capacity(self.config.num_stages);
        for cb in &self.codebooks {
            let c = cb.encode(&residual);
            codes.push(c);
            let centroid = cb.centroid(c as usize);
            for (r, &ce) in residual.iter_mut().zip(centroid) {
                *r -= ce;
            }
        }
        codes
    }

    /// Reconstruct a vector from its codes (sum of stage centroids).
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.config.dim];
        for (cb, &c) in self.codebooks.iter().zip(codes) {
            let centroid = cb.centroid(c as usize);
            for (o, &ce) in out.iter_mut().zip(centroid) {
                *o += ce;
            }
        }
        out
    }

    /// Mean squared quantisation distortion across `data`.
    pub fn mean_distortion(&self, data: &[Vec<f32>]) -> f32 {
        let total: f32 = data
            .iter()
            .map(|v| {
                let codes = self.encode(v);
                let reconstructed = self.decode(&codes);
                l2_sq(v, &reconstructed)
            })
            .sum();
        total / data.len() as f32
    }

    /// Per-stage distortion reduction — useful for showing RVQ convergence.
    pub fn stage_distortions(&self, data: &[Vec<f32>]) -> Vec<f32> {
        let mut residuals: Vec<Vec<f32>> = data.to_vec();
        let mut out = Vec::with_capacity(self.codebooks.len());
        for cb in &self.codebooks {
            let dist: f32 = residuals
                .iter()
                .map(|v| {
                    let c = cb.encode(v) as usize;
                    l2_sq(v, cb.centroid(c))
                })
                .sum::<f32>()
                / data.len() as f32;
            out.push(dist);
            // Update residuals for the next stage.
            residuals = residuals
                .iter()
                .map(|v| {
                    let c = cb.encode(v) as usize;
                    let cen = cb.centroid(c);
                    v.iter().zip(cen).map(|(a, b)| a - b).collect()
                })
                .collect();
        }
        out
    }

    /// Build ADC lookup tables for a query vector (approximate L2 via inner products).
    ///
    /// Returns two tables of shape [num_stages][codebook_size]:
    /// - `inner[s][c]` = ⟨query, centroid_s[c]⟩
    /// - `norms[s][c]` = ‖centroid_s[c]‖²  (precomputed from codebook)
    ///
    /// Approximate distance to a DB code = ‖query‖² - 2·Σₛ inner[s][code_s] + Σₛ norms[s][code_s]
    pub fn adc_tables(&self, query: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let s = self.codebooks.len();
        let k = self.config.codebook_size;
        let mut inner = vec![vec![0.0f32; k]; s];
        let mut norms = vec![vec![0.0f32; k]; s];
        for (stage, cb) in self.codebooks.iter().enumerate() {
            for c in 0..k {
                let cen = cb.centroid(c);
                inner[stage][c] = dot(query, cen);
                norms[stage][c] = l2_sq_self(cen);
            }
        }
        (inner, norms)
    }

    /// Asymmetric distance via precomputed ADC tables.
    #[inline]
    pub fn adc_distance(
        query_norm_sq: f32,
        codes: &[u8],
        inner: &[Vec<f32>],
        norms: &[Vec<f32>],
    ) -> f32 {
        let mut dist = query_norm_sq;
        for (s, &c) in codes.iter().enumerate() {
            let c = c as usize;
            dist += norms[s][c] - 2.0 * inner[s][c];
        }
        dist.max(0.0) // numerical guard: ADC is approximate, can go slightly negative
    }
}

// ── Product quantizer (for fair comparison) ───────────────────────────────────

/// Standard flat Product Quantizer: splits the vector into `m` independent
/// sub-vectors of `dim/m` dimensions each and quantizes each independently.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    pub sub_codebooks: Vec<Codebook>,
    pub m: usize,         // number of subspaces
    pub sub_dim: usize,   // dim / m
    pub k: usize,         // centroids per subspace
}

impl ProductQuantizer {
    pub fn train(
        data: &[Vec<f32>],
        m: usize,
        k: usize,
        train_iters: usize,
        dim: usize,
    ) -> Self {
        assert_eq!(dim % m, 0, "dim must be divisible by m");
        let sub_dim = dim / m;
        let mut sub_codebooks = Vec::with_capacity(m);
        for sub in 0..m {
            let sub_data: Vec<Vec<f32>> = data
                .iter()
                .map(|v| v[sub * sub_dim..(sub + 1) * sub_dim].to_vec())
                .collect();
            sub_codebooks.push(Codebook::train(&sub_data, k, sub_dim, train_iters, 99 + sub as u64));
        }
        ProductQuantizer { sub_codebooks, m, sub_dim, k }
    }

    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        (0..self.m)
            .map(|sub| {
                let sv = &v[sub * self.sub_dim..(sub + 1) * self.sub_dim];
                self.sub_codebooks[sub].encode(sv)
            })
            .collect()
    }

    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        (0..self.m)
            .flat_map(|sub| {
                self.sub_codebooks[sub].centroid(codes[sub] as usize).to_vec()
            })
            .collect()
    }

    /// Build PQ ADC distance table: `table[sub][c]` = ‖q_sub - centroid_sub[c]‖²
    pub fn adc_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        (0..self.m)
            .map(|sub| {
                let q_sub = &query[sub * self.sub_dim..(sub + 1) * self.sub_dim];
                (0..self.k)
                    .map(|c| l2_sq(q_sub, self.sub_codebooks[sub].centroid(c)))
                    .collect()
            })
            .collect()
    }

    #[inline]
    pub fn adc_distance(codes: &[u8], table: &[Vec<f32>]) -> f32 {
        codes.iter().enumerate().map(|(sub, &c)| table[sub][c as usize]).sum()
    }
}
