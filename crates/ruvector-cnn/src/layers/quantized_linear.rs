//! Quantized Linear (Fully Connected) Layer
//!
//! INT8 quantized linear layer with:
//! - GEMM-based forward pass
//! - Fused bias and requantization
//! - Per-channel or per-tensor quantization

use crate::{CnnError, CnnResult, Tensor};

use super::Linear;

/// Quantized Linear Layer
///
/// Performs matrix multiplication in INT8:
/// output = (input @ weights^T + bias) * scale
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Quantized weights: [out_features, in_features] in i8
    weights_q: Vec<i8>,

    /// Per-output-feature weight scales (per-channel quantization)
    weight_scales: Vec<f32>,

    /// Bias in i32 accumulator space
    bias_q: Vec<i32>,

    /// Original FP32 bias
    bias_f32: Vec<f32>,

    /// Layer dimensions
    in_features: usize,
    out_features: usize,
}

impl QuantizedLinear {
    /// Create from FP32 Linear layer
    ///
    /// # Arguments
    /// * `linear` - FP32 linear layer to quantize
    /// * `input_scale` - Expected input activation scale
    pub fn from_fp32(linear: &Linear, input_scale: f32) -> Self {
        let in_features = linear.in_features();
        let out_features = linear.out_features();
        let weights = linear.weight();

        // Compute per-output-feature weight scales
        let mut weight_scales = vec![0.0f32; out_features];

        for of in 0..out_features {
            let mut max_abs = 0.0f32;
            for if_ in 0..in_features {
                let idx = of * in_features + if_;
                max_abs = max_abs.max(weights[idx].abs());
            }
            weight_scales[of] = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        }

        // Quantize weights
        let mut weights_q = vec![0i8; weights.len()];
        for of in 0..out_features {
            let scale = weight_scales[of];
            for if_ in 0..in_features {
                let idx = of * in_features + if_;
                let w_q = (weights[idx] / scale).round().clamp(-127.0, 127.0) as i8;
                weights_q[idx] = w_q;
            }
        }

        // Pre-compute bias in i32 accumulator space
        let bias_f32 = linear
            .bias()
            .map(|b| b.to_vec())
            .unwrap_or_else(|| vec![0.0; out_features]);
        let mut bias_q = vec![0i32; out_features];

        for of in 0..out_features {
            let combined_scale = input_scale * weight_scales[of];
            bias_q[of] = if combined_scale > 0.0 {
                (bias_f32[of] / combined_scale).round() as i32
            } else {
                0
            };
        }

        Self {
            weights_q,
            weight_scales,
            bias_q,
            bias_f32,
            in_features,
            out_features,
        }
    }

    /// Forward pass with INT8 computation
    ///
    /// # Arguments
    /// * `input` - Quantized u8 input tensor [batch, in_features]
    /// * `batch_size` - Batch size
    /// * `input_scale` - Input quantization scale
    /// * `input_zero_point` - Input quantization zero point
    pub fn forward_int8(
        &self,
        input: &[u8],
        batch_size: usize,
        input_scale: f32,
        input_zero_point: u8,
    ) -> CnnResult<Tensor> {
        if input.len() != batch_size * self.in_features {
            return Err(CnnError::invalid_shape(
                format!("input size {}", batch_size * self.in_features),
                format!("size {}", input.len()),
            ));
        }

        let mut output_i32 = vec![0i32; batch_size * self.out_features];

        // Pre-compute weight sums for zero-point correction
        let mut weight_sums = vec![0i32; self.out_features];
        for of in 0..self.out_features {
            let mut sum = 0i32;
            for if_ in 0..self.in_features {
                sum += self.weights_q[of * self.in_features + if_] as i32;
            }
            weight_sums[of] = sum;
        }

        // GEMM: output = input @ weights^T + bias
        for b in 0..batch_size {
            for of in 0..self.out_features {
                // Initialize with bias and zero-point correction
                let mut acc = self.bias_q[of] - (input_zero_point as i32) * weight_sums[of];

                // Dot product
                for if_ in 0..self.in_features {
                    let input_val = input[b * self.in_features + if_] as i32;
                    let weight_val = self.weights_q[of * self.in_features + if_] as i32;
                    acc += input_val * weight_val;
                }

                output_i32[b * self.out_features + of] = acc;
            }
        }

        // Dequantize to f32
        let output_f32 = self.dequantize_output(&output_i32, input_scale);

        Tensor::from_data(output_f32, &[batch_size, self.out_features])
    }

    /// Dequantize i32 accumulator to f32
    fn dequantize_output(&self, acc: &[i32], input_scale: f32) -> Vec<f32> {
        let mut output = vec![0.0f32; acc.len()];

        for (i, &val) in acc.iter().enumerate() {
            let of = i % self.out_features;
            let scale = input_scale * self.weight_scales[of];
            output[i] = val as f32 * scale;
        }

        output
    }

    /// Get input features
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_linear_creation() {
        let linear = Linear::new(128, 64, true).unwrap();
        let qlinear = QuantizedLinear::from_fp32(&linear, 0.01);

        assert_eq!(qlinear.in_features(), 128);
        assert_eq!(qlinear.out_features(), 64);
    }

    #[test]
    fn test_quantized_linear_forward() {
        let linear = Linear::new(32, 16, true).unwrap();
        let qlinear = QuantizedLinear::from_fp32(&linear, 0.01);

        let batch_size = 4;
        let input = vec![128u8; batch_size * 32];

        let output = qlinear.forward_int8(&input, batch_size, 0.01, 128).unwrap();

        assert_eq!(output.shape(), &[batch_size, 16]);
    }

    #[test]
    fn test_quantized_linear_zero_point_correction() {
        let linear = Linear::new(8, 4, true).unwrap();
        let qlinear = QuantizedLinear::from_fp32(&linear, 0.01);

        // Test with non-zero zero-point
        let input = vec![200u8; 1 * 8];
        let output = qlinear.forward_int8(&input, 1, 0.01, 128).unwrap();

        assert_eq!(output.shape(), &[1, 4]);
    }
}
