//! Quantized Pooling Layers
//!
//! INT8 quantized pooling operations:
//! - QuantizedMaxPool2d: operates in INT8 domain (no scale change)
//! - QuantizedAvgPool2d: uses i16 intermediate precision for accumulation

use crate::{CnnError, CnnResult, Tensor};

/// Quantized Max Pooling 2D
///
/// Operates directly in INT8 domain without scale changes.
/// Output has the same scale and zero-point as input.
#[derive(Debug, Clone)]
pub struct QuantizedMaxPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl QuantizedMaxPool2d {
    /// Create a new quantized max pooling layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass with INT8 input
    ///
    /// # Arguments
    /// * `input` - Quantized u8 input tensor (NHWC layout)
    /// * `input_shape` - Input shape [N, H, W, C]
    /// * `scale` - Input/output scale (unchanged)
    /// * `zero_point` - Input/output zero point (unchanged)
    pub fn forward_int8(
        &self,
        input: &[u8],
        input_shape: &[usize],
        scale: f32,
        zero_point: u8,
    ) -> CnnResult<(Vec<u8>, Vec<usize>, f32, u8)> {
        if input_shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D input (NHWC)",
                format!("{}D", input_shape.len()),
            ));
        }

        let batch = input_shape[0];
        let in_h = input_shape[1];
        let in_w = input_shape[2];
        let channels = input_shape[3];

        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output = vec![zero_point; batch * out_h * out_w * channels];

        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for c in 0..channels {
                        let mut max_val = zero_point;

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = (oh * self.stride + kh) as isize - self.padding as isize;
                                let iw = (ow * self.stride + kw) as isize - self.padding as isize;

                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let input_idx = ((b * in_h + ih) * in_w + iw) * channels + c;
                                    max_val = max_val.max(input[input_idx]);
                                }
                            }
                        }

                        let output_idx = ((b * out_h + oh) * out_w + ow) * channels + c;
                        output[output_idx] = max_val;
                    }
                }
            }
        }

        Ok((
            output,
            vec![batch, out_h, out_w, channels],
            scale,
            zero_point,
        ))
    }
}

/// Quantized Average Pooling 2D
///
/// Uses i16 intermediate precision to accumulate sums before division.
/// Output may have different scale than input due to averaging.
#[derive(Debug, Clone)]
pub struct QuantizedAvgPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl QuantizedAvgPool2d {
    /// Create a new quantized average pooling layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass with INT8 input
    ///
    /// # Arguments
    /// * `input` - Quantized u8 input tensor (NHWC layout)
    /// * `input_shape` - Input shape [N, H, W, C]
    /// * `input_scale` - Input scale
    /// * `input_zero_point` - Input zero point
    ///
    /// # Returns
    /// (output, output_shape, output_scale, output_zero_point)
    pub fn forward_int8(
        &self,
        input: &[u8],
        input_shape: &[usize],
        input_scale: f32,
        input_zero_point: u8,
    ) -> CnnResult<(Vec<u8>, Vec<usize>, f32, u8)> {
        if input_shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D input (NHWC)",
                format!("{}D", input_shape.len()),
            ));
        }

        let batch = input_shape[0];
        let in_h = input_shape[1];
        let in_w = input_shape[2];
        let channels = input_shape[3];

        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // Use i16 for accumulation to avoid overflow
        let mut output_i16 = vec![0i16; batch * out_h * out_w * channels];

        let kernel_area = self.kernel_size * self.kernel_size;

        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for c in 0..channels {
                        let mut sum = 0i16;
                        let mut count = 0;

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = (oh * self.stride + kh) as isize - self.padding as isize;
                                let iw = (ow * self.stride + kw) as isize - self.padding as isize;

                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let input_idx = ((b * in_h + ih) * in_w + iw) * channels + c;
                                    sum += input[input_idx] as i16;
                                    count += 1;
                                }
                            }
                        }

                        // Compute average
                        let avg = if count > 0 {
                            (sum + count / 2) / count // Rounding division
                        } else {
                            input_zero_point as i16
                        };

                        let output_idx = ((b * out_h + oh) * out_w + ow) * channels + c;
                        output_i16[output_idx] = avg;
                    }
                }
            }
        }

        // Convert i16 back to u8
        let output: Vec<u8> = output_i16.iter().map(|&v| v.clamp(0, 255) as u8).collect();

        // Output scale remains the same as input for average pooling
        Ok((
            output,
            vec![batch, out_h, out_w, channels],
            input_scale,
            input_zero_point,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_maxpool2d() {
        let pool = QuantizedMaxPool2d::new(2, 2, 0);

        let input = vec![
            100, 150, 200, 255, 120, 180, 210, 230, 110, 140, 190, 240, 130, 160, 220, 250,
        ];
        let input_shape = &[1, 4, 4, 1];

        let (output, output_shape, scale, _zp) =
            pool.forward_int8(&input, input_shape, 0.01, 0).unwrap();

        assert_eq!(output_shape, vec![1, 2, 2, 1]);
        assert_eq!(scale, 0.01);

        // Check that max values are selected
        assert!(output[0] >= 100);
    }

    #[test]
    fn test_quantized_avgpool2d() {
        let pool = QuantizedAvgPool2d::new(2, 2, 0);

        let input = vec![
            100, 100, 200, 200, 100, 100, 200, 200, 100, 100, 200, 200, 100, 100, 200, 200,
        ];
        let input_shape = &[1, 4, 4, 1];

        let (output, output_shape, scale, _zp) =
            pool.forward_int8(&input, input_shape, 0.01, 0).unwrap();

        assert_eq!(output_shape, vec![1, 2, 2, 1]);
        assert_eq!(scale, 0.01);

        // Check approximate averages
        assert!(output[0] >= 95 && output[0] <= 105); // ~100
        assert!(output[1] >= 195 && output[1] <= 205); // ~200
    }

    #[test]
    fn test_quantized_maxpool2d_with_padding() {
        let pool = QuantizedMaxPool2d::new(3, 1, 1);

        let input = vec![100u8; 1 * 4 * 4 * 1];
        let input_shape = &[1, 4, 4, 1];

        let (_output, output_shape, _, _) =
            pool.forward_int8(&input, input_shape, 0.01, 50).unwrap();

        assert_eq!(output_shape, vec![1, 4, 4, 1]);
    }
}
