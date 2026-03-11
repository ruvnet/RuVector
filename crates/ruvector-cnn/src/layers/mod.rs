//! Neural Network Layers
//!
//! This module provides standard CNN layers with SIMD-optimized implementations:
//! - **Conv2d**: 2D convolution with configurable kernel, stride, padding
//! - **DepthwiseSeparableConv**: MobileNet-style efficient convolutions
//! - **BatchNorm**: Batch normalization with learned parameters
//! - **Activations**: ReLU, ReLU6, Swish, HardSwish, Sigmoid
//! - **Pooling**: GlobalAvgPool, MaxPool2d, AvgPool2d
//! - **Linear**: Fully connected layer

pub mod activation;
pub mod batchnorm;
pub mod conv;
pub mod linear;
pub mod pooling;

pub use activation::{Activation, ActivationType, HardSwish, ReLU, ReLU6, Sigmoid, Swish};
pub use batchnorm::{BatchNorm, BatchNorm2d};
pub use conv::{Conv2d, DepthwiseSeparableConv};
pub use linear::Linear;
pub use pooling::{AvgPool2d, GlobalAvgPool, GlobalAvgPool2d, MaxPool2d};

use crate::{CnnResult, Tensor};

/// Tensor shape for 4D tensors (N, C, H, W).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// Batch size
    pub n: usize,
    /// Number of channels
    pub c: usize,
    /// Height
    pub h: usize,
    /// Width
    pub w: usize,
}

impl TensorShape {
    /// Creates a new tensor shape.
    pub fn new(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self { n, c, h, w }
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.n * self.c * self.h * self.w
    }

    /// Returns the spatial size (H * W).
    pub fn spatial_size(&self) -> usize {
        self.h * self.w
    }

    /// Returns the channel size (C * H * W).
    pub fn channel_size(&self) -> usize {
        self.c * self.h * self.w
    }
}

impl std::fmt::Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.n, self.c, self.h, self.w)
    }
}

/// Computes output size for convolution or pooling.
pub fn conv_output_size(input: usize, kernel: usize, stride: usize, padding: usize, dilation: usize) -> usize {
    let effective_kernel = dilation * (kernel - 1) + 1;
    (input + 2 * padding - effective_kernel) / stride + 1
}

/// Trait for all neural network layers
pub trait Layer {
    /// Perform the forward pass
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor>;

    /// Get the layer name
    fn name(&self) -> &'static str;

    /// Get the number of parameters
    fn num_params(&self) -> usize {
        0
    }
}

// =============================================================================
// Standalone layer functions (for backward compatibility with old backbone code)
// =============================================================================

/// 3x3 convolution function (standalone, for backward compatibility)
///
/// Input layout: NCHW flattened as [N * C * H * W]
/// Output layout: NCHW flattened
pub fn conv2d_3x3(
    input: &[f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let out_h = height; // Same padding assumed
    let out_w = width;
    let mut output = vec![0.0f32; out_h * out_w * out_channels];

    crate::simd::scalar::conv_3x3_scalar(
        input,
        weights,
        &mut output,
        height,
        width,
        in_channels,
        out_channels,
        1, // stride
        1, // padding for same output size
    );

    output
}

/// Batch normalization function (standalone, for backward compatibility)
///
/// y = gamma * (x - mean) / sqrt(var + epsilon) + beta
pub fn batch_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    let channels = gamma.len();

    crate::simd::batch_norm_simd(
        input,
        &mut output,
        gamma,
        beta,
        mean,
        var,
        epsilon,
        channels,
    );

    output
}

/// HardSwish activation function (standalone, for backward compatibility)
///
/// hard_swish(x) = x * relu6(x + 3) / 6
pub fn hard_swish(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    crate::simd::scalar::hard_swish_scalar(input, &mut output);
    output
}

/// ReLU activation function (standalone, for backward compatibility)
pub fn relu(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    crate::simd::relu_simd(input, &mut output);
    output
}

/// ReLU6 activation function (standalone, for backward compatibility)
pub fn relu6(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    crate::simd::relu6_simd(input, &mut output);
    output
}

/// Global average pooling function (standalone, for backward compatibility)
///
/// Assumes NCHW layout, pools over H*W dimensions
pub fn global_avg_pool(input: &[f32], channels: usize) -> Vec<f32> {
    let spatial_size = input.len() / channels;
    let mut output = vec![0.0f32; channels];

    // Sum over spatial dimensions
    for i in 0..input.len() {
        let c = i % channels;
        output[c] += input[i];
    }

    // Average
    let inv_spatial = 1.0 / spatial_size as f32;
    for o in output.iter_mut() {
        *o *= inv_spatial;
    }

    output
}

// Re-export Conv2dBuilder from conv module
pub use conv::Conv2dBuilder;

// =============================================================================
// Activation helper methods
// =============================================================================

impl Activation {
    /// Creates a ReLU activation.
    pub fn relu() -> Self {
        Self::new(ActivationType::ReLU)
    }

    /// Creates a ReLU6 activation.
    pub fn relu6() -> Self {
        Self::new(ActivationType::ReLU6)
    }

    /// Creates a HardSwish activation.
    pub fn hard_swish() -> Self {
        Self::new(ActivationType::HardSwish)
    }

    /// Creates a HardSigmoid activation (using Sigmoid as approximation).
    pub fn hard_sigmoid() -> Self {
        Self::new(ActivationType::Sigmoid)
    }

    /// Creates an identity (no-op) activation.
    pub fn identity() -> Self {
        Self::new(ActivationType::Identity)
    }
}
