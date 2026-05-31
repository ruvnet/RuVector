//! MUVERA Fixed Dimensional Encoding (FDE) encoder.
//!
//! Converts a set of token embeddings into a single fixed-length vector
//! by SimHash space partitioning and Rademacher random projection, approximating
//! Chamfer / MaxSim similarity with a formal ε-approximation guarantee.
//!
//! Reference: "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"
//!            arXiv:2405.19504 (NeurIPS 2024, Google Research)

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::error::MuveraError;

/// Configuration for the FDE encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdeConfig {
    /// Input embedding dimension (d).
    pub dim: usize,
    /// Number of SimHash buckets (B). Must be a power of two; k_sim = log2(B).
    pub buckets: usize,
    /// Output dimension per bucket after random projection (d_proj).
    pub d_proj: usize,
    /// Number of independent repetitions (R) concatenated in the final FDE.
    pub reps: usize,
}

impl FdeConfig {
    /// Total FDE output dimension: R × B × d_proj.
    #[inline]
    pub fn fde_dim(&self) -> usize {
        self.reps * self.buckets * self.d_proj
    }

    /// Number of SimHash hyperplanes per repetition: log2(B).
    #[inline]
    pub fn k_sim(&self) -> usize {
        debug_assert!(self.buckets.is_power_of_two());
        self.buckets.trailing_zeros() as usize
    }

    pub fn validate(&self) -> Result<(), MuveraError> {
        if self.dim == 0 {
            return Err(MuveraError::InvalidConfig("dim must be > 0".into()));
        }
        if !self.buckets.is_power_of_two() || self.buckets == 0 {
            return Err(MuveraError::InvalidConfig(
                "buckets must be a non-zero power of two".into(),
            ));
        }
        if self.d_proj == 0 {
            return Err(MuveraError::InvalidConfig("d_proj must be > 0".into()));
        }
        if self.reps == 0 {
            return Err(MuveraError::InvalidConfig("reps must be > 0".into()));
        }
        Ok(())
    }
}

/// Precomputed random state for one repetition.
struct RepState {
    /// k_sim Gaussian hyperplane normals, each of length `dim`.
    hyperplanes: Vec<Vec<f32>>,
    /// Rademacher projection matrix: d_proj rows × dim cols, entries ±1/√d_proj.
    projection: Vec<Vec<f32>>,
}

/// FDE encoder: converts multi-token sets into fixed-dimension single vectors.
pub struct FdeEncoder {
    pub config: FdeConfig,
    reps: Vec<RepState>,
}

impl FdeEncoder {
    /// Build encoder with the given config using `rng` for random initialisation.
    pub fn new<R: Rng>(config: FdeConfig, rng: &mut R) -> Result<Self, MuveraError> {
        config.validate()?;
        let k_sim = config.k_sim();
        let scale = (config.d_proj as f32).sqrt().recip();
        let normal = Normal::new(0.0f32, 1.0).unwrap();

        let reps = (0..config.reps)
            .map(|_| {
                let hyperplanes = (0..k_sim)
                    .map(|_| (0..config.dim).map(|_| normal.sample(rng)).collect())
                    .collect();
                let projection = (0..config.d_proj)
                    .map(|_| {
                        (0..config.dim)
                            .map(|_| if rng.gen::<bool>() { scale } else { -scale })
                            .collect()
                    })
                    .collect();
                RepState { hyperplanes, projection }
            })
            .collect();

        Ok(Self { config, reps })
    }

    /// Total FDE output dimension: R × B × d_proj.
    #[inline]
    pub fn fde_dim(&self) -> usize {
        self.config.fde_dim()
    }

    /// Encode a set of token embeddings into a single FDE vector of length `fde_dim()`.
    ///
    /// Algorithm (one repetition):
    ///   1. SimHash each token into bucket ∈ [0, B).
    ///   2. Accumulate per-bucket centroid sums; fill empty buckets with the token
    ///      nearest (by dot-product) to that bucket's hyperplane-defined center.
    ///   3. Project each centroid through the Rademacher matrix → d_proj values.
    ///   4. Concatenate all B blocks → B·d_proj values.
    /// Repeat R times and concatenate → R·B·d_proj = fde_dim.
    pub fn encode(&self, tokens: &[Vec<f32>]) -> Result<Vec<f32>, MuveraError> {
        if tokens.is_empty() {
            return Err(MuveraError::EmptyTokenSet);
        }
        for t in tokens {
            if t.len() != self.config.dim {
                return Err(MuveraError::DimensionMismatch {
                    expected: self.config.dim,
                    actual: t.len(),
                });
            }
        }

        let d = self.config.dim;
        let b = self.config.buckets;
        let dp = self.config.d_proj;
        let mut fde = vec![0.0f32; self.fde_dim()];

        for (r, rep) in self.reps.iter().enumerate() {
            let rep_offset = r * b * dp;

            // Step 1 & 2: accumulate centroid sums per bucket.
            let mut sums = vec![vec![0.0f32; d]; b];
            let mut counts = vec![0usize; b];

            for token in tokens {
                let bid = simhash(token, &rep.hyperplanes);
                for (s, &t) in sums[bid].iter_mut().zip(token.iter()) {
                    *s += t;
                }
                counts[bid] += 1;
            }

            // Fill empty buckets with the token nearest to that bucket's center.
            for bid in 0..b {
                if counts[bid] == 0 {
                    let center = bucket_center(bid, &rep.hyperplanes);
                    let best = tokens
                        .iter()
                        .max_by(|p, q| {
                            dot_raw(p, &center)
                                .partial_cmp(&dot_raw(q, &center))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap(); // tokens is non-empty
                    sums[bid] = best.clone();
                    counts[bid] = 1;
                }
            }

            // Step 3: project each centroid and write into fde.
            for bid in 0..b {
                let n = counts[bid] as f32;
                let block_start = rep_offset + bid * dp;
                for (p, proj_row) in rep.projection.iter().enumerate() {
                    let val: f32 = sums[bid]
                        .iter()
                        .zip(proj_row.iter())
                        .map(|(&s, &w)| (s / n) * w)
                        .sum();
                    fde[block_start + p] = val;
                }
            }
        }

        Ok(fde)
    }
}

/// SimHash: assign `token` to a bucket in [0, B) using k_sim Gaussian hyperplanes.
#[inline]
fn simhash(token: &[f32], hyperplanes: &[Vec<f32>]) -> usize {
    hyperplanes.iter().enumerate().fold(0usize, |acc, (i, hp)| {
        let dot: f32 = token.iter().zip(hp.iter()).map(|(a, b)| a * b).sum();
        if dot >= 0.0 { acc | (1 << i) } else { acc }
    })
}

/// Construct the "center direction" of bucket `bid` from its hyperplane normals.
/// The center is the vector sum of +g_i if bit i is set, −g_i otherwise.
fn bucket_center(bid: usize, hyperplanes: &[Vec<f32>]) -> Vec<f32> {
    let dim = hyperplanes[0].len();
    let mut c = vec![0.0f32; dim];
    for (i, hp) in hyperplanes.iter().enumerate() {
        let sign: f32 = if (bid >> i) & 1 == 1 { 1.0 } else { -1.0 };
        for (c_i, &h_i) in c.iter_mut().zip(hp.iter()) {
            *c_i += sign * h_i;
        }
    }
    c
}

/// Unchecked dot product (caller ensures equal lengths).
#[inline]
fn dot_raw(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn small_cfg() -> FdeConfig {
        FdeConfig { dim: 4, buckets: 4, d_proj: 4, reps: 2 }
    }

    fn make_encoder(cfg: FdeConfig) -> FdeEncoder {
        let mut rng = StdRng::seed_from_u64(1);
        FdeEncoder::new(cfg, &mut rng).unwrap()
    }

    #[test]
    fn fde_dim_correct() {
        let enc = make_encoder(FdeConfig { dim: 8, buckets: 8, d_proj: 4, reps: 3 });
        assert_eq!(enc.fde_dim(), 8 * 4 * 3);
    }

    #[test]
    fn encode_output_length() {
        let enc = make_encoder(small_cfg());
        let tokens = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let fde = enc.encode(&tokens).unwrap();
        assert_eq!(fde.len(), enc.fde_dim());
    }

    #[test]
    fn encode_empty_tokens_error() {
        let enc = make_encoder(small_cfg());
        assert!(enc.encode(&[]).is_err());
    }

    #[test]
    fn encode_dimension_mismatch_error() {
        let enc = make_encoder(small_cfg());
        let tokens = vec![vec![1.0, 0.0]]; // wrong dim
        assert!(enc.encode(&tokens).is_err());
    }

    #[test]
    fn encode_deterministic() {
        let enc = make_encoder(small_cfg());
        let tokens = vec![vec![0.5, 0.5, 0.5, 0.5]];
        let a = enc.encode(&tokens).unwrap();
        let b = enc.encode(&tokens).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn similar_sets_higher_score() {
        let enc = make_encoder(FdeConfig { dim: 8, buckets: 8, d_proj: 4, reps: 4 });
        let mut rng = StdRng::seed_from_u64(99);
        let t: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..8usize).map(|_| rng.gen::<f32>()).collect())
            .collect();
        // Slightly perturbed copy of t.
        let t_near: Vec<Vec<f32>> = t.iter().map(|v| {
            v.iter().map(|&x| x + rng.gen::<f32>() * 0.01).collect()
        }).collect();
        // Random unrelated set.
        let t_far: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..8usize).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let q = enc.encode(&t).unwrap();
        let near = enc.encode(&t_near).unwrap();
        let far = enc.encode(&t_far).unwrap();
        let score_near: f32 = q.iter().zip(near.iter()).map(|(a, b)| a * b).sum();
        let score_far: f32 = q.iter().zip(far.iter()).map(|(a, b)| a * b).sum();
        assert!(score_near > score_far, "near={score_near:.4} far={score_far:.4}");
    }

    #[test]
    fn invalid_config_rejected() {
        let cfg = FdeConfig { dim: 4, buckets: 3, d_proj: 4, reps: 1 }; // 3 not power-of-2
        let mut rng = StdRng::seed_from_u64(1);
        assert!(FdeEncoder::new(cfg, &mut rng).is_err());
    }
}
