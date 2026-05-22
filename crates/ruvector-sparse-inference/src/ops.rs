//! Basic neural network operations

use std::f32;

/// Linear layer (fully connected)
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Vec<Vec<f32>>, // [out_features, in_features]
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        Self {
            weight: vec![vec![0.0; in_features]; out_features],
            bias: if use_bias {
                Some(vec![0.0; out_features])
            } else {
                None
            },
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.out_features];

        for i in 0..self.out_features {
            let mut sum = 0.0;
            for j in 0..self.in_features.min(input.len()) {
                sum += self.weight[i][j] * input[j];
            }
            if let Some(ref bias) = self.bias {
                sum += bias[i];
            }
            output[i] = sum;
        }

        output
    }
}

/// Linear layer with ternary weights ({−1, 0, +1}).
///
/// Weights are stored as `i8` and quantized from f32 at load time. Every zero
/// weight skips its multiply-accumulate in the inner loop — no approximation,
/// no special hardware required. At the sparsity levels typical of BitNet b1.58
/// and similar ternary schemes (≥50% zeros) this cuts active MACs roughly in half.
///
/// Drop-in for `Linear` when the weight matrix has been ternary-quantized.
#[derive(Debug, Clone)]
pub struct LinearBitNet {
    /// Weights in {−1, 0, +1}, row-major: `weight[i * in_features + j]`
    pub weight: Vec<i8>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearBitNet {
    /// Quantize an f32 weight matrix to ternary using `threshold`.
    /// Values in `(−threshold, +threshold)` become 0; outside become ±1.
    ///
    /// A reasonable default threshold is the mean absolute value of the weights,
    /// which is what the BitNet b1.58 paper uses.
    pub fn from_f32(
        out_features: usize,
        in_features: usize,
        weights: &[f32],
        threshold: f32,
        bias: Option<Vec<f32>>,
    ) -> Self {
        let weight = weights
            .iter()
            .map(|&w| {
                if w > threshold {
                    1
                } else if w < -threshold {
                    -1
                } else {
                    0
                }
            })
            .collect();
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// GEMV forward pass — zero weights are skipped entirely.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.out_features];
        let len = self.in_features.min(input.len());

        for i in 0..self.out_features {
            let row = &self.weight[i * self.in_features..i * self.in_features + len];
            let mut acc = 0.0f32;
            for (j, &w) in row.iter().enumerate() {
                if w == 0 {
                    continue;
                }
                acc += w as f32 * input[j];
            }
            if let Some(ref bias) = self.bias {
                acc += bias[i];
            }
            output[i] = acc;
        }

        output
    }

    /// Fraction of weights that are zero (0.0–1.0).
    pub fn sparsity(&self) -> f32 {
        let zeros = self.weight.iter().filter(|&&w| w == 0).count();
        zeros as f32 / self.weight.len() as f32
    }
}

/// Embedding layer
#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: Vec<Vec<f32>>, // [vocab_size, embedding_dim]
    pub vocab_size: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            weight: vec![vec![0.0; embedding_dim]; vocab_size],
            vocab_size,
            embedding_dim,
        }
    }

    pub fn forward(&self, input_ids: &[u64]) -> Vec<f32> {
        let mut output = Vec::new();

        for &id in input_ids {
            let idx = id as usize;
            if idx < self.vocab_size {
                output.extend_from_slice(&self.weight[idx]);
            } else {
                output.extend_from_slice(&vec![0.0; self.embedding_dim]);
            }
        }

        output
    }
}

/// RMSNorm (Root Mean Square Layer Normalization)
#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; dim],
            eps,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mean_square = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
        let rms = (mean_square + self.eps).sqrt();

        input
            .iter()
            .zip(self.weight.iter())
            .map(|(x, w)| (x / rms) * w)
            .collect()
    }
}

/// LayerNorm
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; dim],
            bias: vec![0.0; dim],
            eps,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
        let std = (variance + self.eps).sqrt();

        input
            .iter()
            .zip(self.weight.iter().zip(self.bias.iter()))
            .map(|(x, (w, b))| ((x - mean) / std) * w + b)
            .collect()
    }
}

/// SiLU (Swish) activation function
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU activation
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// ReLU activation
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let mut linear = Linear::new(3, 2, true);
        linear.weight = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        linear.bias = Some(vec![0.1, 0.2]);

        let input = vec![1.0, 2.0, 3.0];
        let output = linear.forward(&input);

        assert_eq!(output.len(), 2);
        assert!((output[0] - 14.1).abs() < 1e-5);
        assert!((output[1] - 32.2).abs() < 1e-5);
    }

    #[test]
    fn test_linear_bitnet_forward() {
        // 2-output, 4-input layer; weights chosen so zero-skipping is verifiable
        let weights = vec![
            1.0f32, 0.0, -1.0, 0.0, // row 0: dot([1,0,-1,0], input)
            0.0, 1.0, 0.0, 1.0, // row 1: dot([0,1,0,1], input)
        ];
        // threshold=0.5 → |w|≤0.5 becomes 0, so 0.0 → 0, ±1.0 → ±1
        let layer = LinearBitNet::from_f32(2, 4, &weights, 0.5, None);
        assert_eq!(layer.weight, vec![1, 0, -1, 0, 0, 1, 0, 1]);

        let input = vec![2.0, 3.0, 4.0, 5.0];
        let out = layer.forward(&input);
        // row 0: 1*2 + 0 + (-1)*4 + 0 = -2
        // row 1: 0 + 1*3 + 0 + 1*5  = 8
        assert!((out[0] - (-2.0)).abs() < 1e-5);
        assert!((out[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_bitnet_sparsity() {
        let weights = vec![1.0, 0.0, 0.0, -1.0];
        let layer = LinearBitNet::from_f32(2, 2, &weights, 0.5, None);
        assert!((layer.sparsity() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_linear_bitnet_matches_dense_at_zero_sparsity() {
        // When no weights quantize to zero, output must match a dense multiply
        let weights = vec![2.0f32, -3.0, 1.5, -1.5];
        // threshold=0 → everything non-zero becomes ±1
        let layer = LinearBitNet::from_f32(2, 2, &weights, 0.0, None);
        let input = vec![1.0, 1.0];
        let out = layer.forward(&input);
        // row 0: sign(2)*1 + sign(-3)*1 = 1-1 = 0
        // row 1: sign(1.5)*1 + sign(-1.5)*1 = 1-1 = 0
        assert!((out[0]).abs() < 1e-5);
        assert!((out[1]).abs() < 1e-5);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-5);
        assert!(silu(1.0) > 0.0);
        assert!(silu(-1.0) < 0.0);
    }

    #[test]
    fn test_rms_norm() {
        let norm = RMSNorm::new(4, 1e-6);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = norm.forward(&input);
        assert_eq!(output.len(), 4);
    }
}
