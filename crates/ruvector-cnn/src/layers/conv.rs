//! Convolutional Layers
//!
//! SIMD-optimized 2D convolution implementations:
//! - Conv2d: Standard 2D convolution
//! - DepthwiseSeparableConv: MobileNet-style efficient convolution

use crate::{simd, CnnError, CnnResult, Tensor};

use super::{Layer, TensorShape};

/// 2D Convolution Layer
///
/// Performs 2D convolution on NHWC tensors with configurable:
/// - Kernel size
/// - Stride
/// - Padding
/// - Groups (for depthwise and grouped convolutions)
///
/// Kernel layout: [out_channels, kernel_h, kernel_w, in_channels] (OHWI)
#[derive(Debug, Clone)]
pub struct Conv2d {
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Kernel size (height and width)
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
    /// Groups for grouped/depthwise convolution
    groups: usize,
    /// Kernel weights: [out_c, kh, kw, in_c/groups]
    weights: Vec<f32>,
    /// Bias: [out_c]
    bias: Option<Vec<f32>>,
}

/// Builder for Conv2d layer
#[derive(Debug, Clone)]
pub struct Conv2dBuilder {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    bias: bool,
}

impl Conv2dBuilder {
    /// Create a new builder
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: true,
        }
    }

    /// Set stride
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Set groups for grouped convolution
    pub fn groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Set whether to use bias
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Build the Conv2d layer
    pub fn build(self) -> CnnResult<Conv2d> {
        if self.in_channels % self.groups != 0 {
            return Err(CnnError::InvalidParameter(
                format!("in_channels {} must be divisible by groups {}", self.in_channels, self.groups)
            ));
        }
        if self.out_channels % self.groups != 0 {
            return Err(CnnError::InvalidParameter(
                format!("out_channels {} must be divisible by groups {}", self.out_channels, self.groups)
            ));
        }

        let in_channels_per_group = self.in_channels / self.groups;
        let num_weights = self.out_channels * self.kernel_size * self.kernel_size * in_channels_per_group;

        // Xavier/Glorot initialization
        let fan_in = in_channels_per_group * self.kernel_size * self.kernel_size;
        let fan_out = (self.out_channels / self.groups) * self.kernel_size * self.kernel_size;
        let std_dev = (2.0 / (fan_in + fan_out) as f32).sqrt();

        let weights: Vec<f32> = (0..num_weights)
            .map(|i| {
                let x = ((i * 1103515245 + 12345) % (1 << 31)) as f32 / (1u32 << 31) as f32;
                (x * 2.0 - 1.0) * std_dev
            })
            .collect();

        let bias = if self.bias {
            Some(vec![0.0; self.out_channels])
        } else {
            None
        };

        Ok(Conv2d {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            groups: self.groups,
            weights,
            bias,
        })
    }
}

impl Conv2d {
    /// Create a new Conv2d layer with Xavier initialization
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let num_weights = out_channels * kernel_size * kernel_size * in_channels;

        // Xavier/Glorot initialization
        let fan_in = in_channels * kernel_size * kernel_size;
        let fan_out = out_channels * kernel_size * kernel_size;
        let std_dev = (2.0 / (fan_in + fan_out) as f32).sqrt();

        // Simple pseudo-random initialization (for deterministic tests)
        let weights: Vec<f32> = (0..num_weights)
            .map(|i| {
                let x = ((i * 1103515245 + 12345) % (1 << 31)) as f32 / (1u32 << 31) as f32;
                (x * 2.0 - 1.0) * std_dev
            })
            .collect();

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups: 1,
            weights,
            bias: None,
        }
    }

    /// Create a Conv2d builder
    pub fn builder(in_channels: usize, out_channels: usize, kernel_size: usize) -> Conv2dBuilder {
        Conv2dBuilder::new(in_channels, out_channels, kernel_size)
    }

    /// Create Conv2d with bias
    pub fn with_bias(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let mut conv = Self::new(in_channels, out_channels, kernel_size, stride, padding);
        conv.bias = Some(vec![0.0; out_channels]);
        conv
    }

    /// Get the output shape for a TensorShape input (NCHW format)
    pub fn output_shape_nchw(&self, input_shape: &TensorShape) -> TensorShape {
        let out_h = (input_shape.h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (input_shape.w + 2 * self.padding - self.kernel_size) / self.stride + 1;
        TensorShape::new(input_shape.n, self.out_channels, out_h, out_w)
    }

    /// Set the weights directly
    pub fn set_weights(&mut self, weights: Vec<f32>) -> CnnResult<()> {
        let expected = self.out_channels * self.kernel_size * self.kernel_size * self.in_channels;
        if weights.len() != expected {
            return Err(CnnError::invalid_shape(
                format!("{} weights", expected),
                format!("{} weights", weights.len()),
            ));
        }
        self.weights = weights;
        Ok(())
    }

    /// Set the bias
    pub fn set_bias(&mut self, bias: Vec<f32>) -> CnnResult<()> {
        if bias.len() != self.out_channels {
            return Err(CnnError::invalid_shape(
                format!("{} bias values", self.out_channels),
                format!("{} bias values", bias.len()),
            ));
        }
        self.bias = Some(bias);
        Ok(())
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

        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        Ok(vec![batch, out_h, out_w, self.out_channels])
    }

    /// Get weights reference
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get bias reference
    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Get the kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get the stride
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get the padding
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// Get the number of output channels
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get the number of input channels
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get the number of groups
    pub fn groups(&self) -> usize {
        self.groups
    }
}

impl Layer for Conv2d {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D tensor (NHWC)",
                format!("{}D tensor", shape.len()),
            ));
        }

        let in_channels = shape[3];
        if in_channels != self.in_channels {
            return Err(CnnError::invalid_shape(
                format!("{} input channels", self.in_channels),
                format!("{} input channels", in_channels),
            ));
        }

        let batch = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];

        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let out_shape = vec![batch, out_h, out_w, self.out_channels];
        let mut output = Tensor::zeros(&out_shape);

        // Process each batch
        let batch_in_size = in_h * in_w * in_channels;
        let batch_out_size = out_h * out_w * self.out_channels;

        for b in 0..batch {
            let input_slice = &input.data()[b * batch_in_size..(b + 1) * batch_in_size];
            let output_slice = &mut output.data_mut()[b * batch_out_size..(b + 1) * batch_out_size];

            if self.kernel_size == 3 && self.groups == 1 {
                // Standard 3x3 convolution (non-grouped)
                simd::conv_3x3_simd(
                    input_slice,
                    &self.weights,
                    output_slice,
                    in_h,
                    in_w,
                    self.in_channels,
                    self.out_channels,
                    self.stride,
                    self.padding,
                );
            } else if self.kernel_size == 3 && self.groups == self.in_channels && self.in_channels == self.out_channels {
                // Depthwise 3x3 convolution (groups == in_channels == out_channels)
                simd::depthwise_conv_3x3_simd(
                    input_slice,
                    &self.weights,
                    output_slice,
                    in_h,
                    in_w,
                    self.in_channels,
                    self.stride,
                    self.padding,
                );
            } else {
                // Fallback to generic convolution for other cases
                self.conv_generic(input_slice, output_slice, in_h, in_w, out_h, out_w);
            }
        }

        // Add bias if present
        if let Some(bias) = &self.bias {
            for val in output.data_mut().chunks_mut(self.out_channels) {
                for (i, v) in val.iter_mut().enumerate() {
                    *v += bias[i];
                }
            }
        }

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "Conv2d"
    }

    fn num_params(&self) -> usize {
        let weight_params =
            self.out_channels * self.kernel_size * self.kernel_size * self.in_channels;
        let bias_params = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        weight_params + bias_params
    }
}

impl Conv2d {
    /// Generic convolution for arbitrary kernel sizes
    fn conv_generic(
        &self,
        input: &[f32],
        output: &mut [f32],
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let ks = self.kernel_size;
        let in_channels_per_group = self.in_channels / self.groups;
        let out_channels_per_group = self.out_channels / self.groups;

        for oh in 0..out_h {
            for ow in 0..out_w {
                for g in 0..self.groups {
                    let in_c_start = g * in_channels_per_group;
                    let out_c_start = g * out_channels_per_group;

                    for oc_local in 0..out_channels_per_group {
                        let oc = out_c_start + oc_local;
                        let mut sum = 0.0f32;

                        for kh in 0..ks {
                            for kw in 0..ks {
                                let ih = (oh * self.stride + kh) as isize - self.padding as isize;
                                let iw = (ow * self.stride + kw) as isize - self.padding as isize;

                                if ih >= 0
                                    && ih < in_h as isize
                                    && iw >= 0
                                    && iw < in_w as isize
                                {
                                    let ih = ih as usize;
                                    let iw = iw as usize;

                                    for ic_local in 0..in_channels_per_group {
                                        let ic = in_c_start + ic_local;
                                        let input_idx =
                                            ih * in_w * self.in_channels + iw * self.in_channels + ic;
                                        // Kernel layout: [out_c, kh, kw, in_c_per_group]
                                        let kernel_idx = oc * ks * ks * in_channels_per_group
                                            + kh * ks * in_channels_per_group
                                            + kw * in_channels_per_group
                                            + ic_local;
                                        sum += input[input_idx] * self.weights[kernel_idx];
                                    }
                                }
                            }
                        }

                        output[oh * out_w * self.out_channels + ow * self.out_channels + oc] = sum;
                    }
                }
            }
        }
    }
}

/// Depthwise Separable Convolution
///
/// Efficient convolution used in MobileNet architectures:
/// 1. Depthwise convolution: one filter per input channel
/// 2. Pointwise convolution: 1x1 conv to mix channels
///
/// Reduces parameters from O(K^2 * C_in * C_out) to O(K^2 * C_in + C_in * C_out)
#[derive(Debug, Clone)]
pub struct DepthwiseSeparableConv {
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Depthwise kernel size
    kernel_size: usize,
    /// Stride for depthwise conv
    stride: usize,
    /// Padding for depthwise conv
    padding: usize,
    /// Depthwise weights: [in_channels, kernel_h, kernel_w]
    depthwise_weights: Vec<f32>,
    /// Pointwise weights: [out_channels, in_channels]
    pointwise_weights: Vec<f32>,
}

impl DepthwiseSeparableConv {
    /// Create a new depthwise separable convolution
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let dw_size = in_channels * kernel_size * kernel_size;
        let pw_size = out_channels * in_channels;

        // Initialize with small random values
        let depthwise_weights: Vec<f32> = (0..dw_size)
            .map(|i| {
                let x = ((i * 1103515245 + 12345) % (1 << 31)) as f32 / (1u32 << 31) as f32;
                (x * 2.0 - 1.0) * 0.1
            })
            .collect();

        let pointwise_weights: Vec<f32> = (0..pw_size)
            .map(|i| {
                let x = ((i * 1103515245 + 54321) % (1 << 31)) as f32 / (1u32 << 31) as f32;
                (x * 2.0 - 1.0) * 0.1
            })
            .collect();

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            depthwise_weights,
            pointwise_weights,
        }
    }

    /// Set depthwise weights
    pub fn set_depthwise_weights(&mut self, weights: Vec<f32>) -> CnnResult<()> {
        let expected = self.in_channels * self.kernel_size * self.kernel_size;
        if weights.len() != expected {
            return Err(CnnError::invalid_shape(
                format!("{} depthwise weights", expected),
                format!("{} weights", weights.len()),
            ));
        }
        self.depthwise_weights = weights;
        Ok(())
    }

    /// Set pointwise weights
    pub fn set_pointwise_weights(&mut self, weights: Vec<f32>) -> CnnResult<()> {
        let expected = self.out_channels * self.in_channels;
        if weights.len() != expected {
            return Err(CnnError::invalid_shape(
                format!("{} pointwise weights", expected),
                format!("{} weights", weights.len()),
            ));
        }
        self.pointwise_weights = weights;
        Ok(())
    }
}

impl Layer for DepthwiseSeparableConv {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D tensor (NHWC)",
                format!("{}D tensor", shape.len()),
            ));
        }

        let in_channels = shape[3];
        if in_channels != self.in_channels {
            return Err(CnnError::invalid_shape(
                format!("{} input channels", self.in_channels),
                format!("{} input channels", in_channels),
            ));
        }

        let batch = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];

        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // Step 1: Depthwise convolution
        let dw_shape = vec![batch, out_h, out_w, self.in_channels];
        let mut dw_output = Tensor::zeros(&dw_shape);

        let batch_in_size = in_h * in_w * self.in_channels;
        let batch_dw_size = out_h * out_w * self.in_channels;

        for b in 0..batch {
            let input_slice = &input.data()[b * batch_in_size..(b + 1) * batch_in_size];
            let output_slice = &mut dw_output.data_mut()[b * batch_dw_size..(b + 1) * batch_dw_size];

            if self.kernel_size == 3 {
                simd::depthwise_conv_3x3_simd(
                    input_slice,
                    &self.depthwise_weights,
                    output_slice,
                    in_h,
                    in_w,
                    self.in_channels,
                    self.stride,
                    self.padding,
                );
            } else {
                self.depthwise_generic(input_slice, output_slice, in_h, in_w, out_h, out_w);
            }
        }

        // Step 2: Pointwise (1x1) convolution
        let pw_shape = vec![batch, out_h, out_w, self.out_channels];
        let mut output = Tensor::zeros(&pw_shape);

        let batch_pw_size = out_h * out_w * self.out_channels;

        for b in 0..batch {
            let dw_slice = &dw_output.data()[b * batch_dw_size..(b + 1) * batch_dw_size];
            let output_slice = &mut output.data_mut()[b * batch_pw_size..(b + 1) * batch_pw_size];

            simd::scalar::conv_1x1_scalar(
                dw_slice,
                &self.pointwise_weights,
                output_slice,
                out_h,
                out_w,
                self.in_channels,
                self.out_channels,
            );
        }

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "DepthwiseSeparableConv"
    }

    fn num_params(&self) -> usize {
        let dw_params = self.in_channels * self.kernel_size * self.kernel_size;
        let pw_params = self.out_channels * self.in_channels;
        dw_params + pw_params
    }
}

impl DepthwiseSeparableConv {
    /// Generic depthwise convolution for arbitrary kernel sizes
    fn depthwise_generic(
        &self,
        input: &[f32],
        output: &mut [f32],
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let ks = self.kernel_size;

        for oh in 0..out_h {
            for ow in 0..out_w {
                for ch in 0..self.in_channels {
                    let mut sum = 0.0f32;

                    for kh in 0..ks {
                        for kw in 0..ks {
                            let ih = (oh * self.stride + kh) as isize - self.padding as isize;
                            let iw = (ow * self.stride + kw) as isize - self.padding as isize;

                            if ih >= 0
                                && ih < in_h as isize
                                && iw >= 0
                                && iw < in_w as isize
                            {
                                let ih = ih as usize;
                                let iw = iw as usize;

                                let input_idx =
                                    ih * in_w * self.in_channels + iw * self.in_channels + ch;
                                let kernel_idx = ch * ks * ks + kh * ks + kw;
                                sum += input[input_idx] * self.depthwise_weights[kernel_idx];
                            }
                        }
                    }

                    output[oh * out_w * self.in_channels + ow * self.in_channels + ch] = sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_creation() {
        let conv = Conv2d::new(3, 64, 3, 1, 1);
        assert_eq!(conv.num_params(), 3 * 64 * 3 * 3);
    }

    #[test]
    fn test_conv2d_output_shape() {
        let conv = Conv2d::new(3, 64, 3, 1, 1);
        let shape = conv.output_shape(&[1, 224, 224, 3]).unwrap();
        assert_eq!(shape, vec![1, 224, 224, 64]);
    }

    #[test]
    fn test_conv2d_output_shape_stride2() {
        let conv = Conv2d::new(3, 64, 3, 2, 1);
        let shape = conv.output_shape(&[1, 224, 224, 3]).unwrap();
        assert_eq!(shape, vec![1, 112, 112, 64]);
    }

    #[test]
    fn test_conv2d_forward() {
        let conv = Conv2d::new(3, 16, 3, 1, 1);
        let input = Tensor::ones(&[1, 8, 8, 3]);
        let output = conv.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 8, 8, 16]);
    }

    #[test]
    fn test_depthwise_separable_conv() {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1);
        let input = Tensor::ones(&[1, 8, 8, 16]);
        let output = conv.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 8, 8, 32]);
    }

    #[test]
    fn test_depthwise_separable_conv_params() {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1);

        // depthwise: 16 * 3 * 3 = 144
        // pointwise: 32 * 16 = 512
        // total: 656
        assert_eq!(conv.num_params(), 144 + 512);

        // Compare to regular conv: 32 * 3 * 3 * 16 = 4608
        // Savings: 4608 / 656 = 7x fewer params
    }
}
