//! Pooling Layers
//!
//! SIMD-optimized pooling operations:
//! - GlobalAvgPool: Global average pooling over spatial dimensions
//! - MaxPool2d: Max pooling with configurable kernel and stride
//! - AvgPool2d: Average pooling with configurable kernel and stride

use crate::{simd, CnnError, CnnResult, Tensor};

use super::Layer;

/// Alias for GlobalAvgPool (for API compatibility)
pub type GlobalAvgPool2d = GlobalAvgPool;

/// Global Average Pooling
///
/// Reduces spatial dimensions to 1x1 by averaging over all spatial positions.
/// Commonly used before the final fully-connected layer in CNNs.
///
/// Input: [batch, height, width, channels]
/// Output: [batch, 1, 1, channels]
#[derive(Debug, Clone, Default)]
pub struct GlobalAvgPool;

impl GlobalAvgPool {
    /// Create a new global average pooling layer
    pub fn new() -> Self {
        Self
    }
}

impl Layer for GlobalAvgPool {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D tensor (NHWC)",
                format!("{}D tensor", shape.len()),
            ));
        }

        let batch = shape[0];
        let h = shape[1];
        let w = shape[2];
        let c = shape[3];

        let out_shape = vec![batch, 1, 1, c];
        let mut output = Tensor::zeros(&out_shape);

        let batch_in_size = h * w * c;

        for b in 0..batch {
            let input_slice = &input.data()[b * batch_in_size..(b + 1) * batch_in_size];
            let output_slice = &mut output.data_mut()[b * c..(b + 1) * c];

            simd::global_avg_pool_simd(input_slice, output_slice, h, w, c);
        }

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "GlobalAvgPool"
    }
}

/// 2D Max Pooling
///
/// Performs max pooling over spatial dimensions with configurable kernel size,
/// stride, and padding.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    /// Kernel size (height and width)
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
}

impl MaxPool2d {
    /// Create a new max pooling layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Create a max pooling layer with stride equal to kernel size
    pub fn with_kernel(kernel_size: usize) -> Self {
        Self::new(kernel_size, kernel_size, 0)
    }

    /// Get the output shape for a given input shape
    pub fn output_shape(&self, input_shape: &[usize]) -> CnnResult<Vec<usize>> {
        if input_shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D tensor (NHWC)",
                format!("{}D tensor", input_shape.len()),
            ));
        }

        let batch = input_shape[0];
        let in_h = input_shape[1];
        let in_w = input_shape[2];
        let c = input_shape[3];

        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        Ok(vec![batch, out_h, out_w, c])
    }
}

impl Layer for MaxPool2d {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D tensor (NHWC)",
                format!("{}D tensor", shape.len()),
            ));
        }

        let batch = shape[0];
        let h = shape[1];
        let w = shape[2];
        let c = shape[3];

        let out_shape = self.output_shape(shape)?;
        let out_h = out_shape[1];
        let out_w = out_shape[2];

        let mut output = Tensor::zeros(&out_shape);

        let batch_in_size = h * w * c;
        let batch_out_size = out_h * out_w * c;

        for b in 0..batch {
            let input_slice = &input.data()[b * batch_in_size..(b + 1) * batch_in_size];
            let output_slice = &mut output.data_mut()[b * batch_out_size..(b + 1) * batch_out_size];

            if self.kernel_size == 2 && self.padding == 0 {
                simd::max_pool_2x2_simd(input_slice, output_slice, h, w, c, self.stride);
            } else {
                simd::scalar::max_pool_scalar(
                    input_slice,
                    output_slice,
                    h,
                    w,
                    c,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                );
            }
        }

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "MaxPool2d"
    }
}

/// 2D Average Pooling
///
/// Performs average pooling over spatial dimensions with configurable kernel size,
/// stride, and padding.
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    /// Kernel size (height and width)
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
}

impl AvgPool2d {
    /// Create a new average pooling layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Create an average pooling layer with stride equal to kernel size
    pub fn with_kernel(kernel_size: usize) -> Self {
        Self::new(kernel_size, kernel_size, 0)
    }

    /// Get the output shape for a given input shape
    pub fn output_shape(&self, input_shape: &[usize]) -> CnnResult<Vec<usize>> {
        if input_shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D tensor (NHWC)",
                format!("{}D tensor", input_shape.len()),
            ));
        }

        let batch = input_shape[0];
        let in_h = input_shape[1];
        let in_w = input_shape[2];
        let c = input_shape[3];

        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        Ok(vec![batch, out_h, out_w, c])
    }
}

impl Layer for AvgPool2d {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D tensor (NHWC)",
                format!("{}D tensor", shape.len()),
            ));
        }

        let batch = shape[0];
        let h = shape[1];
        let w = shape[2];
        let c = shape[3];

        let out_shape = self.output_shape(shape)?;
        let out_h = out_shape[1];
        let out_w = out_shape[2];

        let mut output = Tensor::zeros(&out_shape);

        let batch_in_size = h * w * c;
        let batch_out_size = out_h * out_w * c;

        for b in 0..batch {
            let input_slice = &input.data()[b * batch_in_size..(b + 1) * batch_in_size];
            let output_slice = &mut output.data_mut()[b * batch_out_size..(b + 1) * batch_out_size];

            if self.kernel_size == 2 && self.padding == 0 {
                simd::scalar::avg_pool_2x2_scalar(input_slice, output_slice, h, w, c, self.stride);
            } else {
                simd::scalar::avg_pool_scalar(
                    input_slice,
                    output_slice,
                    h,
                    w,
                    c,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                );
            }
        }

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "AvgPool2d"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_avg_pool() {
        let pool = GlobalAvgPool::new();
        let input = Tensor::ones(&[2, 4, 4, 8]);
        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 1, 1, 8]);

        // All ones averaged = 1
        for &val in output.data() {
            assert!((val - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_global_avg_pool_values() {
        let pool = GlobalAvgPool::new();

        // Create input with known values: channel 0 = 1, channel 1 = 2
        let mut data = vec![0.0; 2 * 2 * 2];
        for i in 0..4 {
            data[i * 2] = 1.0; // channel 0
            data[i * 2 + 1] = 2.0; // channel 1
        }
        let input = Tensor::from_data(data, &[1, 2, 2, 2]).unwrap();

        let output = pool.forward(&input).unwrap();

        assert!((output.data()[0] - 1.0).abs() < 0.001);
        assert!((output.data()[1] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_max_pool2d() {
        let pool = MaxPool2d::new(2, 2, 0);
        let input = Tensor::ones(&[1, 8, 8, 4]);
        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 4, 4, 4]);
    }

    #[test]
    fn test_max_pool2d_values() {
        let pool = MaxPool2d::new(2, 2, 0);

        // 2x2 input, 1 channel: [[1, 2], [3, 4]]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(data, &[1, 2, 2, 1]).unwrap();

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 1, 1, 1]);
        assert_eq!(output.data()[0], 4.0);
    }

    #[test]
    fn test_max_pool2d_output_shape() {
        let pool = MaxPool2d::new(2, 2, 0);
        let shape = pool.output_shape(&[1, 224, 224, 64]).unwrap();
        assert_eq!(shape, vec![1, 112, 112, 64]);
    }

    #[test]
    fn test_avg_pool2d() {
        let pool = AvgPool2d::new(2, 2, 0);
        let input = Tensor::ones(&[1, 8, 8, 4]);
        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 4, 4, 4]);
    }

    #[test]
    fn test_avg_pool2d_values() {
        let pool = AvgPool2d::new(2, 2, 0);

        // 2x2 input, 1 channel: [[1, 2], [3, 4]]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(data, &[1, 2, 2, 1]).unwrap();

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 1, 1, 1]);
        assert!((output.data()[0] - 2.5).abs() < 0.001); // (1+2+3+4)/4 = 2.5
    }

    #[test]
    fn test_max_pool_with_stride1() {
        let pool = MaxPool2d::new(2, 1, 0);
        let shape = pool.output_shape(&[1, 4, 4, 1]).unwrap();
        assert_eq!(shape, vec![1, 3, 3, 1]);
    }
}
