//! Quantized Depthwise Convolution Layer
//!
//! INT8 quantized depthwise convolution with:
//! - Separate kernel for channel-wise operations
//! - Efficient memory layout (channel-first processing)
//! - Per-channel quantization

use crate::{CnnError, CnnResult, Tensor};

use super::{Conv2d, Layer};

/// Quantized Depthwise Convolution Layer
///
/// Performs depthwise separable convolution in INT8:
/// - Each input channel is convolved with a single kernel
/// - No cross-channel mixing (unlike standard convolution)
/// - Memory efficient for mobile architectures
#[derive(Debug, Clone)]
pub struct QuantizedDepthwiseConv2d {
    /// Quantized weights: [channels, kh, kw] in i8
    weights_q: Vec<i8>,

    /// Per-channel weight scales
    weight_scales: Vec<f32>,

    /// Bias in i32 accumulator space
    bias_q: Vec<i32>,

    /// Original FP32 bias
    bias_f32: Vec<f32>,

    /// Layer configuration
    channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl QuantizedDepthwiseConv2d {
    /// Create from FP32 depthwise convolution
    ///
    /// # Arguments
    /// * `channels` - Number of input/output channels
    /// * `kernel_size` - Kernel size (assumed square)
    /// * `weights` - FP32 weights [channels, kh, kw]
    /// * `bias` - Optional FP32 bias [channels]
    /// * `stride` - Stride
    /// * `padding` - Padding
    /// * `input_scale` - Expected input activation scale
    pub fn from_fp32(
        channels: usize,
        kernel_size: usize,
        weights: &[f32],
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        input_scale: f32,
    ) -> Self {
        // Compute per-channel weight scales
        let mut weight_scales = vec![0.0f32; channels];

        for c in 0..channels {
            let mut max_abs = 0.0f32;
            for kh in 0..kernel_size {
                for kw in 0..kernel_size {
                    let idx = c * kernel_size * kernel_size + kh * kernel_size + kw;
                    max_abs = max_abs.max(weights[idx].abs());
                }
            }
            weight_scales[c] = if max_abs > 0.0 {
                max_abs / 127.0
            } else {
                1.0
            };
        }

        // Quantize weights
        let mut weights_q = vec![0i8; weights.len()];
        for c in 0..channels {
            let scale = weight_scales[c];
            for kh in 0..kernel_size {
                for kw in 0..kernel_size {
                    let idx = c * kernel_size * kernel_size + kh * kernel_size + kw;
                    let w_q = (weights[idx] / scale).round().clamp(-127.0, 127.0) as i8;
                    weights_q[idx] = w_q;
                }
            }
        }

        // Pre-compute bias in i32 accumulator space
        let bias_f32 = bias.map(|b| b.to_vec()).unwrap_or_else(|| vec![0.0; channels]);
        let mut bias_q = vec![0i32; channels];

        for c in 0..channels {
            let combined_scale = input_scale * weight_scales[c];
            bias_q[c] = if combined_scale > 0.0 {
                (bias_f32[c] / combined_scale).round() as i32
            } else {
                0
            };
        }

        Self {
            weights_q,
            weight_scales,
            bias_q,
            bias_f32,
            channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass with INT8 computation
    ///
    /// # Arguments
    /// * `input` - Quantized u8 input tensor (NHWC layout)
    /// * `input_shape` - Input shape [N, H, W, C]
    /// * `input_scale` - Input quantization scale
    /// * `input_zero_point` - Input quantization zero point
    pub fn forward_int8(
        &self,
        input: &[u8],
        input_shape: &[usize],
        input_scale: f32,
        input_zero_point: u8,
    ) -> CnnResult<Tensor> {
        if input_shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D input (NHWC)",
                format!("{}D", input_shape.len())
            ));
        }

        let batch = input_shape[0];
        let in_h = input_shape[1];
        let in_w = input_shape[2];
        let in_c = input_shape[3];

        if in_c != self.channels {
            return Err(CnnError::invalid_shape(
                format!("{} channels", self.channels),
                format!("{} channels", in_c)
            ));
        }

        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output_i32 = vec![0i32; batch * out_h * out_w * self.channels];

        // Process each batch
        for b in 0..batch {
            let batch_in_size = in_h * in_w * in_c;
            let batch_out_size = out_h * out_w * self.channels;

            let input_slice = &input[b * batch_in_size..(b + 1) * batch_in_size];
            let output_slice = &mut output_i32[b * batch_out_size..(b + 1) * batch_out_size];

            self.depthwise_conv_int8_scalar(
                input_slice,
                input_zero_point as i32,
                output_slice,
                in_h, in_w, out_h, out_w,
            );
        }

        // Dequantize to f32
        let output_f32 = self.dequantize_output(&output_i32, input_scale);

        Tensor::from_data(
            output_f32,
            &[batch, out_h, out_w, self.channels],
        )
    }

    /// Scalar depthwise convolution implementation
    fn depthwise_conv_int8_scalar(
        &self,
        input: &[u8],
        input_zero_point: i32,
        output: &mut [i32],
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let ks = self.kernel_size;

        // Pre-compute weight sums per channel
        let mut weight_sums = vec![0i32; self.channels];
        for c in 0..self.channels {
            let mut sum = 0i32;
            for kh in 0..ks {
                for kw in 0..ks {
                    let idx = c * ks * ks + kh * ks + kw;
                    sum += self.weights_q[idx] as i32;
                }
            }
            weight_sums[c] = sum;
        }

        // Depthwise convolution: each channel processed independently
        for oh in 0..out_h {
            for ow in 0..out_w {
                for c in 0..self.channels {
                    // Initialize with bias and zero-point correction
                    let mut acc = self.bias_q[c] - input_zero_point * weight_sums[c];

                    // Convolve over kernel
                    for kh in 0..ks {
                        for kw in 0..ks {
                            let ih = (oh * self.stride + kh) as isize - self.padding as isize;
                            let iw = (ow * self.stride + kw) as isize - self.padding as isize;

                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let ih = ih as usize;
                                let iw = iw as usize;

                                let input_idx = (ih * in_w + iw) * self.channels + c;
                                let weight_idx = c * ks * ks + kh * ks + kw;

                                acc += (input[input_idx] as i32) * (self.weights_q[weight_idx] as i32);
                            }
                        }
                    }

                    output[(oh * out_w + ow) * self.channels + c] = acc;
                }
            }
        }
    }

    /// Dequantize i32 accumulator to f32
    fn dequantize_output(&self, acc: &[i32], input_scale: f32) -> Vec<f32> {
        let mut output = vec![0.0f32; acc.len()];

        for (i, &val) in acc.iter().enumerate() {
            let c = i % self.channels;
            let scale = input_scale * self.weight_scales[c];
            output[i] = val as f32 * scale;
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_depthwise_conv2d_creation() {
        let channels = 32;
        let kernel_size = 3;
        let weights = vec![0.1f32; channels * kernel_size * kernel_size];
        let bias_vec = vec![0.0f32; channels];

        let qconv = QuantizedDepthwiseConv2d::from_fp32(
            channels,
            kernel_size,
            &weights,
            Some(&bias_vec),
            1,
            1,
            0.01,
        );

        assert_eq!(qconv.channels, 32);
        assert_eq!(qconv.kernel_size, 3);
    }

    #[test]
    fn test_quantized_depthwise_conv2d_forward() {
        let channels = 16;
        let kernel_size = 3;
        let weights = vec![0.1f32; channels * kernel_size * kernel_size];

        let qconv = QuantizedDepthwiseConv2d::from_fp32(
            channels,
            kernel_size,
            &weights,
            None,
            1,
            1,
            0.01,
        );

        let input = vec![128u8; 1 * 8 * 8 * channels];
        let input_shape = &[1, 8, 8, channels];

        let output = qconv.forward_int8(&input, input_shape, 0.01, 128).unwrap();

        assert_eq!(output.shape(), &[1, 8, 8, channels]);
    }
}
