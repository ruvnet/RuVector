//! Multi-head attention implementation.
//!
//! Implements parallel attention heads for diverse representation learning.

use crate::{
    error::{AttentionError, AttentionResult},
    traits::Attention,
};

use super::scaled_dot_product::ScaledDotProductAttention;

/// Multi-head attention mechanism.
///
/// Splits the input into multiple heads, applies attention in parallel,
/// and concatenates the results. This allows the model to attend to
/// different representation subspaces simultaneously.
pub struct MultiHeadAttention {
    dim: usize,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Creates a new multi-head attention mechanism.
    ///
    /// # Arguments
    ///
    /// * `dim` - The embedding dimension
    /// * `num_heads` - Number of attention heads
    ///
    /// # Panics
    ///
    /// Panics if `num_heads` is zero or `dim` is not divisible by `num_heads`.
    /// Use [`MultiHeadAttention::try_new`] for untrusted configuration.
    pub fn new(dim: usize, num_heads: usize) -> Self {
        assert!(num_heads > 0, "Number of heads must be greater than zero");
        assert!(
            dim % num_heads == 0,
            "Dimension {} must be divisible by number of heads {}",
            dim,
            num_heads
        );

        Self {
            dim,
            num_heads,
            head_dim: dim / num_heads,
        }
    }

    /// Creates a new multi-head attention mechanism, returning an error instead
    /// of panicking when the configuration is invalid.
    pub fn try_new(dim: usize, num_heads: usize) -> AttentionResult<Self> {
        if dim == 0 {
            return Err(AttentionError::InvalidConfig(
                "Dimension must be greater than zero".to_string(),
            ));
        }
        if num_heads == 0 || dim % num_heads != 0 {
            return Err(AttentionError::InvalidHeadCount { dim, num_heads });
        }
        Ok(Self::new(dim, num_heads))
    }
}

impl Attention for MultiHeadAttention {
    fn compute(
        &self,
        query: &[f32],
        keys: &[&[f32]],
        values: &[&[f32]],
    ) -> AttentionResult<Vec<f32>> {
        if query.len() != self.dim {
            return Err(AttentionError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if keys.len() != values.len() {
            return Err(AttentionError::DimensionMismatch {
                expected: keys.len(),
                actual: values.len(),
            });
        }
        for vector in keys.iter().chain(values.iter()) {
            if vector.len() != self.dim {
                return Err(AttentionError::DimensionMismatch {
                    expected: self.dim,
                    actual: vector.len(),
                });
            }
        }

        // Run each head on borrowed sub-slices: no per-key copies.
        let hd = self.head_dim;
        let head_attn = ScaledDotProductAttention::new(hd);
        let mut output = Vec::with_capacity(self.dim);
        let mut head_keys: Vec<&[f32]> = Vec::with_capacity(keys.len());
        let mut head_values: Vec<&[f32]> = Vec::with_capacity(values.len());
        for h in 0..self.num_heads {
            let range = h * hd..(h + 1) * hd;
            head_keys.clear();
            head_keys.extend(keys.iter().map(|k| &k[range.clone()]));
            head_values.clear();
            head_values.extend(values.iter().map(|v| &v[range.clone()]));
            let head_out = head_attn.compute(&query[range], &head_keys, &head_values)?;
            output.extend_from_slice(&head_out);
        }
        Ok(output)
    }

    fn compute_with_mask(
        &self,
        query: &[f32],
        keys: &[&[f32]],
        values: &[&[f32]],
        _mask: Option<&[bool]>,
    ) -> AttentionResult<Vec<f32>> {
        // For simplicity, delegate to compute (mask handling can be added per-head)
        self.compute(query, keys, values)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head() {
        let attn = MultiHeadAttention::new(8, 2);
        let query = vec![1.0_f32; 8];
        let key1 = vec![0.5_f32; 8];
        let key2 = vec![0.3_f32; 8];
        let val1 = vec![1.0_f32; 8];
        let val2 = vec![2.0_f32; 8];
        let keys = vec![key1.as_slice(), key2.as_slice()];
        let values = vec![val1.as_slice(), val2.as_slice()];

        let result = attn.compute(&query, &keys, &values).unwrap();
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_short_keys_or_values_are_rejected() {
        let attn = MultiHeadAttention::new(8, 2);
        let query = vec![1.0_f32; 8];
        let full = vec![0.5_f32; 8];
        let short = vec![0.5_f32; 3];
        assert!(attn
            .compute(&query, &[short.as_slice()], &[full.as_slice()])
            .is_err());
        assert!(attn
            .compute(&query, &[full.as_slice()], &[short.as_slice()])
            .is_err());
        assert!(attn
            .compute(
                &query,
                &[full.as_slice(), full.as_slice()],
                &[full.as_slice()]
            )
            .is_err());
    }

    #[test]
    fn test_try_new_rejects_bad_heads() {
        assert!(MultiHeadAttention::try_new(8, 0).is_err());
        assert!(MultiHeadAttention::try_new(10, 3).is_err());
        assert!(MultiHeadAttention::try_new(0, 1).is_err());
        assert!(MultiHeadAttention::try_new(8, 2).is_ok());
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn test_invalid_heads() {
        MultiHeadAttention::new(10, 3);
    }
}
