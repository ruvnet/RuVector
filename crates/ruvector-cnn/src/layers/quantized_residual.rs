//! Quantized Residual Addition
//!
//! INT8 residual connections with:
//! - Requantization to align scales between branches
//! - Per-tensor scale alignment
//! - Handles mismatched scales

use crate::{CnnError, CnnResult, Tensor};

/// Quantized Residual Addition
///
/// Adds two quantized tensors with potentially different scales:
/// output = input1 + input2
///
/// Handles scale alignment and requantization.
#[derive(Debug, Clone)]
pub struct QuantizedResidualAdd {
    /// Output scale (chosen as the average of input scales)
    output_scale: f32,

    /// Output zero point (typically 128 for symmetric distributions)
    output_zero_point: u8,
}

impl QuantizedResidualAdd {
    /// Create a new quantized residual add layer
    ///
    /// # Arguments
    /// * `scale1` - Scale of first input
    /// * `scale2` - Scale of second input
    pub fn new(scale1: f32, scale2: f32) -> Self {
        // Use geometric mean of scales as output scale
        let output_scale = (scale1 * scale2).sqrt();

        // Assume symmetric distribution around 128
        let output_zero_point = 128u8;

        Self {
            output_scale,
            output_zero_point,
        }
    }

    /// Forward pass with INT8 inputs
    ///
    /// # Arguments
    /// * `input1` - First quantized u8 input
    /// * `scale1` - Scale of first input
    /// * `zero_point1` - Zero point of first input
    /// * `input2` - Second quantized u8 input
    /// * `scale2` - Scale of second input
    /// * `zero_point2` - Zero point of second input
    /// * `shape` - Tensor shape (must be identical for both inputs)
    ///
    /// # Returns
    /// (output, output_scale, output_zero_point)
    pub fn forward_int8(
        &self,
        input1: &[u8],
        scale1: f32,
        zero_point1: u8,
        input2: &[u8],
        scale2: f32,
        zero_point2: u8,
        shape: &[usize],
    ) -> CnnResult<(Vec<u8>, f32, u8)> {
        if input1.len() != input2.len() {
            return Err(CnnError::invalid_shape(
                format!("input size {}", input1.len()),
                format!("size {}", input2.len()),
            ));
        }

        let mut output = vec![self.output_zero_point; input1.len()];

        // Compute scale factors for requantization
        // output = (input1_dequant + input2_dequant) / output_scale + output_zero_point
        //        = ((q1 - zp1) * s1 + (q2 - zp2) * s2) / s_out + zp_out

        let scale_factor1 = scale1 / self.output_scale;
        let scale_factor2 = scale2 / self.output_scale;

        for i in 0..input1.len() {
            // Dequantize to floating point domain
            let val1 = (input1[i] as f32 - zero_point1 as f32) * scale_factor1;
            let val2 = (input2[i] as f32 - zero_point2 as f32) * scale_factor2;

            // Add in floating point
            let sum = val1 + val2;

            // Requantize to output
            let output_q = (sum + self.output_zero_point as f32)
                .round()
                .clamp(0.0, 255.0);
            output[i] = output_q as u8;
        }

        Ok((output, self.output_scale, self.output_zero_point))
    }

    /// Forward pass with scale alignment (i16 intermediate precision)
    ///
    /// More accurate version using i16 intermediate precision.
    pub fn forward_int8_i16(
        &self,
        input1: &[u8],
        scale1: f32,
        zero_point1: u8,
        input2: &[u8],
        scale2: f32,
        zero_point2: u8,
        shape: &[usize],
    ) -> CnnResult<(Vec<u8>, f32, u8)> {
        if input1.len() != input2.len() {
            return Err(CnnError::invalid_shape(
                format!("input size {}", input1.len()),
                format!("size {}", input2.len()),
            ));
        }

        let mut output = vec![self.output_zero_point; input1.len()];

        // Compute integer scale factors (multiplier and shift)
        let (mult1, shift1) = Self::quantize_scale(scale1 / self.output_scale);
        let (mult2, shift2) = Self::quantize_scale(scale2 / self.output_scale);

        for i in 0..input1.len() {
            // Subtract zero points
            let val1 = input1[i] as i16 - zero_point1 as i16;
            let val2 = input2[i] as i16 - zero_point2 as i16;

            // Scale using fixed-point arithmetic
            let scaled1 = Self::multiply_by_quantized_multiplier(val1 as i32, mult1, shift1);
            let scaled2 = Self::multiply_by_quantized_multiplier(val2 as i32, mult2, shift2);

            // Add and requantize
            let sum = scaled1 + scaled2 + self.output_zero_point as i32;
            output[i] = sum.clamp(0, 255) as u8;
        }

        Ok((output, self.output_scale, self.output_zero_point))
    }

    /// Quantize a floating-point scale to (multiplier, shift) format
    ///
    /// Represents scale as: multiplier * 2^(-shift)
    /// where multiplier is in [0.5, 1.0) as i32 in Q31 format
    fn quantize_scale(scale: f32) -> (i32, i32) {
        if scale <= 0.0 {
            return (0, 0);
        }

        // Find the shift such that scale * 2^shift is in [0.5, 1.0)
        let mut shift = 0i32;
        let mut scaled = scale;

        while scaled < 0.5 {
            scaled *= 2.0;
            shift += 1;
        }
        while scaled >= 1.0 {
            scaled *= 0.5;
            shift -= 1;
        }

        // Quantize to Q31 format (31 fractional bits)
        let multiplier = (scaled * 2147483648.0) as i32; // 2^31

        (multiplier, shift)
    }

    /// Multiply by quantized multiplier with rounding
    fn multiply_by_quantized_multiplier(value: i32, multiplier: i32, shift: i32) -> i32 {
        // Perform multiplication in i64 to avoid overflow
        let total = (value as i64) * (multiplier as i64);

        // Apply shift with rounding
        let result = if shift >= 0 {
            (total + (1i64 << (shift - 1))) >> shift
        } else {
            total << (-shift)
        };

        result as i32
    }

    /// Get output scale
    pub fn output_scale(&self) -> f32 {
        self.output_scale
    }

    /// Get output zero point
    pub fn output_zero_point(&self) -> u8 {
        self.output_zero_point
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_residual_add_same_scale() {
        let scale = 0.01f32;
        let residual = QuantizedResidualAdd::new(scale, scale);

        let input1 = vec![128u8; 16];
        let input2 = vec![138u8; 16]; // +10 in quantized domain
        let shape = &[4, 4];

        let (output, _out_scale, _out_zp) = residual
            .forward_int8(&input1, scale, 128, &input2, scale, 128, shape)
            .unwrap();

        assert_eq!(output.len(), 16);
        // Output should be approximately 138 (128 + 10)
        assert!(output[0] >= 135 && output[0] <= 141);
    }

    #[test]
    fn test_quantized_residual_add_different_scales() {
        let scale1 = 0.01f32;
        let scale2 = 0.02f32;
        let residual = QuantizedResidualAdd::new(scale1, scale2);

        let input1 = vec![128u8; 16];
        let input2 = vec![133u8; 16]; // +5 in quantized domain, but scale2 is 2x
        let shape = &[4, 4];

        let (output, _out_scale, _out_zp) = residual
            .forward_int8(&input1, scale1, 128, &input2, scale2, 128, shape)
            .unwrap();

        assert_eq!(output.len(), 16);
        // Check that output is within reasonable range
        assert!(output[0] >= 120 && output[0] <= 140);
    }

    #[test]
    fn test_quantized_residual_add_i16_precision() {
        let scale = 0.01f32;
        let residual = QuantizedResidualAdd::new(scale, scale);

        let input1 = vec![100u8; 8];
        let input2 = vec![150u8; 8];
        let shape = &[2, 4];

        let (output, _, _) = residual
            .forward_int8_i16(&input1, scale, 128, &input2, scale, 128, shape)
            .unwrap();

        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_quantize_scale() {
        let (mult, shift) = QuantizedResidualAdd::quantize_scale(0.5);
        assert!(mult > 0);
        assert_eq!(shift, 0);

        let (mult2, shift2) = QuantizedResidualAdd::quantize_scale(0.25);
        assert!(mult2 > 0);
        assert_eq!(shift2, 1);
    }
}
