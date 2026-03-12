//! INT8 quantization module for ADR-091
//!
//! This module provides INT8 quantization primitives for testing.
//! Full implementation will be added in subsequent phases.

pub mod kernels;

use serde::{Deserialize, Serialize};

/// Quantization parameters for INT8 conversion
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantParams {
    /// Scale factor for quantization
    pub scale: f32,
    /// Zero point offset
    pub zero_point: i8,
}

impl QuantParams {
    /// Compute quantization parameters from a tensor
    pub fn from_tensor(tensor: &[f32]) -> Self {
        if tensor.is_empty() {
            return Self {
                scale: 1.0,
                zero_point: 0,
            };
        }

        let min_val = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Handle all zeros or constant tensors
        if (max_val - min_val).abs() < 1e-10 {
            return Self {
                scale: 1.0,
                zero_point: 0,
            };
        }

        // Compute scale and zero_point for asymmetric quantization
        // Map [min_val, max_val] to [-128, 127]
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-128.0 - min_val / scale).round() as i8;

        Self { scale, zero_point }
    }
}

/// Quantize FP32 tensor to INT8
pub fn quantize_tensor(fp32: &[f32], params: &QuantParams) -> Vec<i8> {
    fp32.iter()
        .map(|&x| {
            let quantized = (x / params.scale + params.zero_point as f32).round();
            quantized.clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Dequantize INT8 tensor to FP32
pub fn dequantize_tensor(int8: &[i8], params: &QuantParams) -> Vec<f32> {
    int8.iter()
        .map(|&x| (x as f32 - params.zero_point as f32) * params.scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_params_from_tensor() {
        let tensor = vec![-1.0, 0.0, 1.0, 2.0];
        let params = QuantParams::from_tensor(&tensor);

        assert!(params.scale > 0.0);
        assert!(params.scale.is_finite());
        assert!(params.zero_point >= -128 && params.zero_point <= 127);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let fp32 = vec![0.5, -0.3, 0.8, -0.1];
        let params = QuantParams::from_tensor(&fp32);
        let int8 = quantize_tensor(&fp32, &params);
        let dequant = dequantize_tensor(&int8, &params);

        for (orig, recovered) in fp32.iter().zip(dequant.iter()) {
            let error = (orig - recovered).abs();
            assert!(error < 0.1, "Roundtrip error too large: {}", error);
        }
    }

    #[test]
    fn test_empty_tensor() {
        let empty: Vec<f32> = vec![];
        let params = QuantParams::from_tensor(&empty);
        assert_eq!(params.scale, 1.0);
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn test_constant_tensor() {
        let constant = vec![0.5; 100];
        let params = QuantParams::from_tensor(&constant);
        assert_eq!(params.scale, 1.0);
        assert_eq!(params.zero_point, 0);
    }
}
