//! Two-level (primary + residual) Locally-Adaptive Vector Quantization.
//!
//! After encoding `v` as LVQ-8, the reconstruction error
//! `r = v - decode(LVQ8(v))` is encoded with another independent LVQ-8 pass.
//! The full reconstruction is the sum of the two decoded levels.
//!
//! Compared to a single 16-bit quantizer, two-level 8+8 is friendlier for
//! reranking: the primary code alone already gives a useful (low-recall)
//! distance estimate which can be refined with the residual only on the
//! short-list of candidates. This is the SVS "LVQ-Bx8" recipe.

use serde::{Deserialize, Serialize};

use crate::error::LvqError;
use crate::quantize::{encode_one, Lvq8, Lvq8Stats};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Lvq8x8 {
    pub primary: Lvq8,
    /// Residual codes packed contiguously (same `dim`).
    pub residual_codes: Vec<u8>,
    pub residual_stats: Vec<Lvq8Stats>,
}

impl Lvq8x8 {
    pub fn new(dim: usize) -> Self {
        Self {
            primary: Lvq8::new(dim),
            residual_codes: Vec::new(),
            residual_stats: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.primary.dim
    }

    pub fn len(&self) -> usize {
        self.primary.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primary.is_empty()
    }

    pub fn byte_size(&self) -> usize {
        self.primary.byte_size()
            + self.residual_codes.len()
            + self.residual_stats.len() * std::mem::size_of::<Lvq8Stats>()
    }

    pub fn push(&mut self, v: &[f32]) -> Result<(), LvqError> {
        let i = self.primary.len();
        self.primary.push(v)?;
        let residual = self.primary.residual(i, v);
        let (rstats, rcode) = encode_one(&residual)?;
        self.residual_stats.push(rstats);
        self.residual_codes.extend_from_slice(&rcode);
        Ok(())
    }

    pub fn extend_from_flat(&mut self, flat: &[f32]) -> Result<(), LvqError> {
        let dim = self.primary.dim;
        if dim == 0 || flat.is_empty() {
            return Err(LvqError::Empty);
        }
        if flat.len() % dim != 0 {
            return Err(LvqError::DimMismatch {
                expected: dim,
                actual: flat.len() % dim,
            });
        }
        for chunk in flat.chunks_exact(dim) {
            self.push(chunk)?;
        }
        Ok(())
    }

    #[inline]
    pub fn residual_row(&self, i: usize) -> &[u8] {
        let dim = self.primary.dim;
        let off = i * dim;
        &self.residual_codes[off..off + dim]
    }

    pub fn decode(&self, i: usize) -> Vec<f32> {
        let dim = self.primary.dim;
        let p_stats = self.primary.stats_at(i);
        let p_row = self.primary.code_row(i);
        let r_stats = self.residual_stats[i];
        let r_row = self.residual_row(i);
        (0..dim)
            .map(|j| p_stats.decode_lane(p_row[j]) + r_stats.decode_lane(r_row[j]))
            .collect()
    }

    #[inline]
    pub fn primary_stats(&self, i: usize) -> Lvq8Stats {
        self.primary.stats_at(i)
    }

    #[inline]
    pub fn residual_stats_at(&self, i: usize) -> Lvq8Stats {
        self.residual_stats[i]
    }

    #[inline]
    pub fn primary_row(&self, i: usize) -> &[u8] {
        self.primary.code_row(i)
    }

    pub fn primary_only(&self) -> &Lvq8 {
        &self.primary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::{rngs::StdRng, Rng};

    #[test]
    fn two_level_strictly_better_than_one() {
        let mut rng = StdRng::seed_from_u64(7);
        let dim = 96;
        let mut sum_one = 0.0_f64;
        let mut sum_two = 0.0_f64;
        for _ in 0..32 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let mut q1 = Lvq8::new(dim);
            q1.push(&v).unwrap();
            let dec1 = q1.decode(0);
            let err1: f64 = v
                .iter()
                .zip(dec1.iter())
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum();

            let mut q2 = Lvq8x8::new(dim);
            q2.push(&v).unwrap();
            let dec2 = q2.decode(0);
            let err2: f64 = v
                .iter()
                .zip(dec2.iter())
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum();

            sum_one += err1;
            sum_two += err2;
        }
        assert!(
            sum_two < sum_one * 0.25,
            "two-level should reduce L2 error by >4x; got one={sum_one:.4} two={sum_two:.4}"
        );
    }
}
