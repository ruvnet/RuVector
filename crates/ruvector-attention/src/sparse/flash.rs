//! Flash attention - memory-efficient attention with tiled computation
//!
//! Memory: O(block_size) for attention matrix instead of O(n²)

use crate::error::{AttentionError, AttentionResult};
use crate::traits::Attention;
use crate::utils::{axpy, dot_f32};

/// Flash attention with block-wise computation
///
/// Computes attention in tiles to minimize memory usage while maintaining numerical stability.
pub struct FlashAttention {
    dim: usize,
    block_size: usize,
    scale: f32,
    causal: bool,
}

impl FlashAttention {
    /// Create new flash attention
    pub fn new(dim: usize, block_size: usize) -> Self {
        Self {
            dim,
            block_size,
            scale: 1.0 / (dim as f32).sqrt(),
            causal: false,
        }
    }

    /// Create with causal masking
    pub fn causal(dim: usize, block_size: usize) -> Self {
        Self {
            dim,
            block_size,
            scale: 1.0 / (dim as f32).sqrt(),
            causal: true,
        }
    }

    /// Compute attention scores for a block into `scores` (cleared first).
    fn compute_block_scores(
        &self,
        query: &[f32],
        keys: &[&[f32]],
        start_idx: usize,
        scores: &mut Vec<f32>,
    ) {
        scores.clear();
        scores.extend(keys.iter().enumerate().map(|(j, key)| {
            if self.causal && start_idx + j > 0 {
                // Simplified causal: assuming query is at position 0
                f32::NEG_INFINITY
            } else {
                dot_f32(query, key) * self.scale
            }
        }));
    }
}

impl Attention for FlashAttention {
    fn compute(
        &self,
        query: &[f32],
        keys: &[&[f32]],
        values: &[&[f32]],
    ) -> AttentionResult<Vec<f32>> {
        if keys.is_empty() {
            return Err(AttentionError::InvalidConfig("Empty keys".to_string()));
        }
        if keys.len() != values.len() {
            return Err(AttentionError::DimensionMismatch {
                expected: keys.len(),
                actual: values.len(),
            });
        }
        if query.len() != self.dim {
            return Err(AttentionError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }

        if self.block_size == 0 {
            return Err(AttentionError::InvalidConfig(
                "Block size must be greater than zero".to_string(),
            ));
        }
        let n = keys.len();
        let value_dim = values[0].len();
        if let Some(bad) = values.iter().find(|v| v.len() != value_dim) {
            return Err(AttentionError::DimensionMismatch {
                expected: value_dim,
                actual: bad.len(),
            });
        }

        // Online softmax with tiled computation
        let mut output = vec![0.0f32; value_dim];
        let mut max_so_far = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        // Process in blocks; one score buffer is reused across blocks.
        let mut block_scores = Vec::with_capacity(self.block_size.min(n));
        for block_start in (0..n).step_by(self.block_size) {
            let block_end = (block_start + self.block_size).min(n);

            // Compute attention scores for this block
            self.compute_block_scores(
                query,
                &keys[block_start..block_end],
                block_start,
                &mut block_scores,
            );

            // Find block maximum
            let block_max = block_scores
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);

            if !block_max.is_finite() {
                continue; // Skip fully masked blocks
            }

            // New maximum
            let new_max = max_so_far.max(block_max);

            // Rescale previous accumulations
            if max_so_far.is_finite() {
                let rescale = (max_so_far - new_max).exp();
                sum_exp *= rescale;
                output.iter_mut().for_each(|o| *o *= rescale);
            }

            // Add contribution from this block
            for (local_idx, &score) in block_scores.iter().enumerate() {
                if score.is_finite() {
                    let exp_score = (score - new_max).exp();
                    sum_exp += exp_score;

                    axpy(&mut output, exp_score, values[block_start + local_idx]);
                }
            }

            max_so_far = new_max;
        }

        // Final normalization
        if sum_exp > 1e-8 {
            output.iter_mut().for_each(|o| *o /= sum_exp);
        }

        Ok(output)
    }

    fn compute_with_mask(
        &self,
        query: &[f32],
        keys: &[&[f32]],
        values: &[&[f32]],
        mask: Option<&[bool]>,
    ) -> AttentionResult<Vec<f32>> {
        if let Some(m) = mask {
            if m.len() != keys.len() || keys.len() != values.len() {
                return Err(AttentionError::InvalidMask {
                    expected: format!("{} entries", keys.len()),
                    actual: format!("{} entries", m.len()),
                });
            }
            let filtered: Vec<(usize, bool)> = m
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, keep)| *keep)
                .collect();
            let filtered_keys: Vec<&[f32]> = filtered.iter().map(|(i, _)| keys[*i]).collect();
            let filtered_values: Vec<&[f32]> = filtered.iter().map(|(i, _)| values[*i]).collect();
            self.compute(query, &filtered_keys, &filtered_values)
        } else {
            self.compute(query, keys, values)
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::ScaledDotProductAttention;

    #[test]
    fn test_zero_block_size_is_an_error() {
        let attention = FlashAttention::new(4, 0);
        let query = vec![1.0_f32; 4];
        let key = vec![1.0_f32; 4];
        assert!(attention
            .compute(&query, &[key.as_slice()], &[key.as_slice()])
            .is_err());
    }

    #[test]
    fn test_ragged_values_are_an_error() {
        let attention = FlashAttention::new(4, 2);
        let query = vec![1.0_f32; 4];
        let key = vec![1.0_f32; 4];
        let short = vec![1.0_f32; 2];
        let long = vec![1.0_f32; 8];
        assert!(attention
            .compute(
                &query,
                &[key.as_slice(), key.as_slice()],
                &[short.as_slice(), long.as_slice()]
            )
            .is_err());
    }

    #[test]
    fn test_flash_attention() {
        let attention = FlashAttention::new(64, 16);

        let query = vec![0.5; 64];
        let keys: Vec<Vec<f32>> = (0..256).map(|_| vec![0.3; 64]).collect();
        let values: Vec<Vec<f32>> = (0..256).map(|_| vec![1.0; 64]).collect();

        let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

        let result = attention.compute(&query, &keys_refs, &values_refs).unwrap();
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_flash_matches_standard() {
        let dim = 32;
        let flash = FlashAttention::new(dim, 8);
        let standard = ScaledDotProductAttention::new(dim);

        let query = vec![0.5; dim];
        let keys: Vec<Vec<f32>> = (0..16).map(|i| vec![(i as f32) * 0.1; dim]).collect();
        let values: Vec<Vec<f32>> = (0..16).map(|i| vec![(i as f32) * 0.2; dim]).collect();

        let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

        let flash_result = flash.compute(&query, &keys_refs, &values_refs).unwrap();
        let standard_result = standard.compute(&query, &keys_refs, &values_refs).unwrap();

        // Results should be approximately equal
        for (f, s) in flash_result.iter().zip(standard_result.iter()) {
            assert!((f - s).abs() < 1e-4, "Flash: {}, Standard: {}", f, s);
        }
    }

    #[test]
    fn test_causal_flash() {
        let attention = FlashAttention::causal(32, 8);

        let query = vec![1.0; 32];
        let keys = vec![vec![0.5; 32]; 20];
        let values = vec![vec![1.0; 32]; 20];

        let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

        let result = attention.compute(&query, &keys_refs, &values_refs).unwrap();
        assert_eq!(result.len(), 32);
    }
}
