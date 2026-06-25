//! Ternary Sparse Feed-Forward Network.
//!
//! Drop-in replacement for [`SparseFfn`] that stores W1 and W2 as ternary
//! `{-1, 0, +1}` i8 weights instead of f32. This gives **two independent
//! axes of sparsity**:
//!
//! 1. **Activation sparsity** (caller-supplied via `active_neurons`):
//!    only the neurons selected by the predictor are visited at all.
//! 2. **Weight sparsity** (inherent in ternary quantization):
//!    zero weights inside each active neuron's dot product are skipped,
//!    saving multiplications proportional to the zero-fraction.
//!
//! At BitNet b1.58 realistic sparsity (>=50% zeros in W1) combined with
//! 70% neuron activation sparsity, active multiply-accumulate operations
//! drop to ~15% of the equivalent dense f32 baseline.
//!
//! # Example
//!
//! ```rust,ignore
//! use ruvector_sparse_inference::sparse::TernarySparseFfn;
//! use ruvector_sparse_inference::config::ActivationType;
//!
//! // Quantize f32 weights at mean-absolute-value threshold (BitNet b1.58 recipe)
//! let threshold = w1_weights.iter().map(|w| w.abs()).sum::<f32>() / w1_weights.len() as f32;
//! let ffn = TernarySparseFfn::from_f32(
//!     512, 2048, 512,
//!     &w1_weights, &w2_weights,
//!     threshold,
//!     None, None,
//!     ActivationType::Silu,
//! )?;
//!
//! // Sparse forward (active_neurons from LowRankPredictor)
//! let output = ffn.forward_sparse(&input, &active_neurons)?;
//! println!("W1 sparsity: {:.1}%", ffn.w1_sparsity() * 100.0);
//! ```
//!
//! [`SparseFfn`]: super::ffn::SparseFfn

use serde::{Deserialize, Serialize};

use crate::config::ActivationType;
use crate::error::{InferenceError, Result};
use crate::ops::LinearBitNet;

/// Sparse FFN whose weight matrices are ternary-quantized.
///
/// Weights are stored as `i8` in `{-1, 0, +1}`. Zero-weight MACs are skipped
/// in the inner loop. Biases remain `f32` for full precision.
///
/// # Layout
///
/// - `w1`: `[hidden_dim x in_features]` row-major.
/// - `w2`: `[out_features x hidden_dim]` row-major.
///
/// See the [module docs](self) for the dual-sparsity story.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernarySparseFfn {
    w1: LinearBitNet,
    w2: LinearBitNet,
    b1: Vec<f32>,
    b2: Vec<f32>,
    activation: ActivationType,
}

impl TernarySparseFfn {
    /// Build from raw f32 weight slices.
    ///
    /// `w1_weights` must be `hidden_dim * in_features` values in row-major order.
    /// `w2_weights` must be `out_features * hidden_dim` values in row-major order.
    ///
    /// `threshold` controls ternary quantization: `|w| <= threshold` -> 0, else +-1.
    /// The BitNet b1.58 paper uses `mean(|W|)` as threshold.
    pub fn from_f32(
        in_features: usize,
        hidden_dim: usize,
        out_features: usize,
        w1_weights: &[f32],
        w2_weights: &[f32],
        threshold: f32,
        b1: Option<Vec<f32>>,
        b2: Option<Vec<f32>>,
        activation: ActivationType,
    ) -> Result<Self> {
        if w1_weights.len() != hidden_dim * in_features {
            return Err(InferenceError::Failed(format!(
                "w1 length mismatch: expected {}, got {}",
                hidden_dim * in_features,
                w1_weights.len()
            ))
            .into());
        }
        if w2_weights.len() != out_features * hidden_dim {
            return Err(InferenceError::Failed(format!(
                "w2 length mismatch: expected {}, got {}",
                out_features * hidden_dim,
                w2_weights.len()
            ))
            .into());
        }

        let b1 = b1.unwrap_or_else(|| vec![0.0; hidden_dim]);
        let b2 = b2.unwrap_or_else(|| vec![0.0; out_features]);

        if b1.len() != hidden_dim {
            return Err(InferenceError::Failed(format!(
                "b1 length mismatch: expected {}, got {}",
                hidden_dim,
                b1.len()
            ))
            .into());
        }
        if b2.len() != out_features {
            return Err(InferenceError::Failed(format!(
                "b2 length mismatch: expected {}, got {}",
                out_features,
                b2.len()
            ))
            .into());
        }

        let w1 = LinearBitNet::from_f32(hidden_dim, in_features, w1_weights, threshold, None);
        let w2 = LinearBitNet::from_f32(out_features, hidden_dim, w2_weights, threshold, None);

        Ok(Self {
            w1,
            w2,
            b1,
            b2,
            activation,
        })
    }

    /// Input dimension.
    pub fn input_dim(&self) -> usize {
        self.w1.in_features
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.w1.out_features
    }

    /// Output dimension.
    pub fn output_dim(&self) -> usize {
        self.w2.out_features
    }

    /// Fraction of W1 weights that are zero (0.0-1.0).
    pub fn w1_sparsity(&self) -> f32 {
        self.w1.sparsity()
    }

    /// Fraction of W2 weights that are zero (0.0-1.0).
    pub fn w2_sparsity(&self) -> f32 {
        self.w2.sparsity()
    }

    /// Sparse forward pass -- only `active_neurons` contribute.
    ///
    /// For each active neuron `n`:
    /// 1. Ternary dot `w1[n] * input` (zero weights skipped)
    /// 2. Activation applied to the scalar result
    /// 3. `hidden[n] * w2_col[n]` accumulated into output (zero weights skipped)
    pub fn forward_sparse(&self, input: &[f32], active_neurons: &[usize]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim() {
            return Err(InferenceError::InputDimensionMismatch {
                expected: self.input_dim(),
                actual: input.len(),
            }
            .into());
        }
        if active_neurons.is_empty() {
            return Err(InferenceError::NoActiveNeurons.into());
        }

        let hidden_dim = self.hidden_dim();
        let in_dim = self.input_dim();
        let out_dim = self.output_dim();

        // Step 1: sparse W1 -- one ternary dot per active neuron
        let mut hidden: Vec<f32> = Vec::with_capacity(active_neurons.len());
        for &n in active_neurons {
            if n >= hidden_dim {
                return Err(
                    InferenceError::Failed(format!("neuron index {} >= hidden_dim {}", n, hidden_dim))
                        .into(),
                );
            }
            let row = &self.w1.weight[n * in_dim..(n + 1) * in_dim];
            let mut acc = self.b1[n];
            for (j, &w) in row.iter().enumerate() {
                if w != 0 {
                    acc += w as f32 * input[j];
                }
            }
            hidden.push(acc);
        }

        // Step 2: activation
        apply_activation(&mut hidden, self.activation);

        // Step 3: sparse W2 accumulation
        let mut output: Vec<f32> = self.b2.clone();
        for (i, &n) in active_neurons.iter().enumerate() {
            let row = &self.w2.weight[n * out_dim..(n + 1) * out_dim];
            let h_val = hidden[i];
            for (k, &w) in row.iter().enumerate() {
                if w != 0 {
                    output[k] += w as f32 * h_val;
                }
            }
        }

        Ok(output)
    }

    /// Dense forward pass (all neurons, all weights; used for validation and baseline).
    pub fn forward_dense(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim() {
            return Err(InferenceError::InputDimensionMismatch {
                expected: self.input_dim(),
                actual: input.len(),
            }
            .into());
        }

        // W1 * input + b1 (zero-skip still applies inside LinearBitNet::forward)
        let mut hidden = self.w1.forward(input);
        for (h, &b) in hidden.iter_mut().zip(self.b1.iter()) {
            *h += b;
        }

        apply_activation(&mut hidden, self.activation);

        // W2 * hidden + b2
        let mut output = self.w2.forward(&hidden);
        for (o, &b) in output.iter_mut().zip(self.b2.iter()) {
            *o += b;
        }

        Ok(output)
    }
}

fn apply_activation(v: &mut Vec<f32>, act: ActivationType) {
    use std::f32::consts::PI;
    match act {
        ActivationType::Relu => {
            for x in v.iter_mut() {
                *x = x.max(0.0);
            }
        }
        ActivationType::Gelu => {
            for x in v.iter_mut() {
                *x = 0.5 * *x * (1.0 + ((2.0 / PI).sqrt() * (*x + 0.044715 * x.powi(3))).tanh());
            }
        }
        ActivationType::Silu | ActivationType::Swish => {
            for x in v.iter_mut() {
                *x = *x / (1.0 + (-*x).exp());
            }
        }
        ActivationType::Identity => {}
    }
}

impl super::FeedForward for TernarySparseFfn {
    fn forward_sparse(
        &self,
        input: &[f32],
        active_neurons: &[usize],
    ) -> crate::error::Result<Vec<f32>> {
        TernarySparseFfn::forward_sparse(self, input, active_neurons)
    }

    fn forward_dense(&self, input: &[f32]) -> crate::error::Result<Vec<f32>> {
        TernarySparseFfn::forward_dense(self, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ffn(in_f: usize, hidden: usize, out_f: usize) -> TernarySparseFfn {
        // Alternating +1/-1 so sparsity=0 but output is deterministic
        let w1: Vec<f32> = (0..hidden * in_f)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let w2: Vec<f32> = (0..out_f * hidden)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        TernarySparseFfn::from_f32(
            in_f, hidden, out_f, &w1, &w2, 0.5, None, None, ActivationType::Relu,
        )
        .unwrap()
    }

    #[test]
    fn test_dimensions() {
        let ffn = make_ffn(32, 128, 32);
        assert_eq!(ffn.input_dim(), 32);
        assert_eq!(ffn.hidden_dim(), 128);
        assert_eq!(ffn.output_dim(), 32);
    }

    #[test]
    fn test_sparse_output_shape() {
        let ffn = make_ffn(16, 64, 16);
        let input = vec![1.0f32; 16];
        let active: Vec<usize> = (0..32).collect();
        let out = ffn.forward_sparse(&input, &active).unwrap();
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_dense_output_shape() {
        let ffn = make_ffn(16, 64, 16);
        let input = vec![1.0f32; 16];
        let out = ffn.forward_dense(&input).unwrap();
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_all_neurons_sparse_matches_dense() {
        let ffn = make_ffn(8, 32, 8);
        let input = vec![0.5f32; 8];
        let all: Vec<usize> = (0..32).collect();

        let sparse = ffn.forward_sparse(&input, &all).unwrap();
        let dense = ffn.forward_dense(&input).unwrap();

        for (s, d) in sparse.iter().zip(dense.iter()) {
            assert!(
                (s - d).abs() < 1e-4,
                "sparse={s:.6} dense={d:.6} -- sparse/dense disagreement"
            );
        }
    }

    #[test]
    fn test_empty_active_neurons_errors() {
        let ffn = make_ffn(8, 32, 8);
        let input = vec![1.0f32; 8];
        assert!(ffn.forward_sparse(&input, &[]).is_err());
    }

    #[test]
    fn test_invalid_neuron_index_errors() {
        let ffn = make_ffn(8, 32, 8);
        let input = vec![1.0f32; 8];
        assert!(ffn.forward_sparse(&input, &[999]).is_err());
    }

    #[test]
    fn test_sparsity_accessors() {
        // threshold=2.0 -> all +-1.0 weights quantize to 0
        let w = vec![1.0f32; 16];
        let ffn = TernarySparseFfn::from_f32(
            4, 4, 4, &w, &w, 2.0, None, None, ActivationType::Relu,
        )
        .unwrap();
        assert!((ffn.w1_sparsity() - 1.0).abs() < 1e-5);
        assert!((ffn.w2_sparsity() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let result = TernarySparseFfn::from_f32(
            4,
            8,
            4,
            &vec![1.0f32; 10], // should be 32
            &vec![1.0f32; 32],
            0.5,
            None,
            None,
            ActivationType::Relu,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_silu_activation() {
        let w1: Vec<f32> = vec![1.0; 4 * 4];
        let w2: Vec<f32> = vec![1.0; 4 * 4];
        let ffn = TernarySparseFfn::from_f32(
            4, 4, 4, &w1, &w2, 0.5, None, None, ActivationType::Silu,
        )
        .unwrap();
        let input = vec![1.0f32; 4];
        let all = vec![0, 1, 2, 3];
        let out = ffn.forward_sparse(&input, &all).unwrap();
        assert_eq!(out.len(), 4);
    }
}
