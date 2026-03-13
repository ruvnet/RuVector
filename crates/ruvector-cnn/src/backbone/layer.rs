//! Backbone-specific Layer trait
//!
//! This module provides a Layer trait that works with raw f32 slices
//! and TensorShape, which is more suitable for composing backbone modules.

use crate::error::CnnResult;
use crate::layers::{
    conv_output_size, Activation, ActivationType, BatchNorm, Conv2d, GlobalAvgPool, Linear,
    TensorShape,
};
use crate::Tensor;

/// Trait for backbone layers that operate on raw f32 slices with TensorShape.
///
/// This is used by backbone modules (ConvBNActivation, InvertedResidual, etc.)
/// that compose multiple layers and work with raw data in NCHW format.
pub trait Layer: Send + Sync {
    /// Perform the forward pass
    fn forward(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>>;

    /// Get the output shape for a given input shape
    fn output_shape(&self, input_shape: &TensorShape) -> TensorShape;

    /// Get the number of parameters
    fn num_params(&self) -> usize {
        0
    }
}

// =============================================================================
// Layer implementations for standard types
// =============================================================================

impl Layer for Conv2d {
    fn forward(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        // Convert NCHW input to NHWC for the underlying Conv2d
        let nhwc_input = nchw_to_nhwc(input, input_shape);
        let nhwc_shape = [input_shape.n, input_shape.h, input_shape.w, input_shape.c];

        let tensor = Tensor::from_data(nhwc_input, &nhwc_shape)?;
        let output_tensor = crate::layers::Layer::forward(self, &tensor)?;

        // Convert output back to NCHW
        let out_shape = output_tensor.shape();
        let out_tensor_shape =
            TensorShape::new(out_shape[0], out_shape[3], out_shape[1], out_shape[2]);
        Ok(nhwc_to_nchw(output_tensor.data(), &out_tensor_shape))
    }

    fn output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        let out_h = conv_output_size(
            input_shape.h,
            self.kernel_size(),
            self.stride(),
            self.padding(),
            1,
        );
        let out_w = conv_output_size(
            input_shape.w,
            self.kernel_size(),
            self.stride(),
            self.padding(),
            1,
        );
        TensorShape::new(input_shape.n, self.out_channels(), out_h, out_w)
    }

    fn num_params(&self) -> usize {
        crate::layers::Layer::num_params(self)
    }
}

impl Layer for BatchNorm {
    fn forward(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        // Convert NCHW input to NHWC for BatchNorm
        let nhwc_input = nchw_to_nhwc(input, input_shape);
        let nhwc_shape = [input_shape.n, input_shape.h, input_shape.w, input_shape.c];

        let tensor = Tensor::from_data(nhwc_input, &nhwc_shape)?;
        let output_tensor = crate::layers::Layer::forward(self, &tensor)?;

        // Convert back to NCHW
        Ok(nhwc_to_nchw(output_tensor.data(), input_shape))
    }

    fn output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        *input_shape
    }

    fn num_params(&self) -> usize {
        crate::layers::Layer::num_params(self)
    }
}

impl Layer for Activation {
    fn forward(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        // Activation is element-wise, no layout conversion needed
        let mut output = input.to_vec();
        self.apply_inplace(&mut output);
        Ok(output)
    }

    fn output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        *input_shape
    }

    fn num_params(&self) -> usize {
        0
    }
}

impl Layer for GlobalAvgPool {
    fn forward(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        // Global average pooling: [N, C, H, W] -> [N, C, 1, 1]
        let batch_size = input_shape.n;
        let channels = input_shape.c;
        let spatial_size = input_shape.spatial_size();

        let mut output = vec![0.0; batch_size * channels];

        for n in 0..batch_size {
            for c in 0..channels {
                let mut sum = 0.0;
                for s in 0..spatial_size {
                    let idx = n * channels * spatial_size + c * spatial_size + s;
                    sum += input[idx];
                }
                output[n * channels + c] = sum / spatial_size as f32;
            }
        }

        Ok(output)
    }

    fn output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        TensorShape::new(input_shape.n, input_shape.c, 1, 1)
    }

    fn num_params(&self) -> usize {
        0
    }
}

impl Layer for Linear {
    fn forward(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        // Linear expects flattened [batch, features] input
        self.forward_batch(input, input_shape.n)
    }

    fn output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        TensorShape::new(input_shape.n, self.out_features(), 1, 1)
    }

    fn num_params(&self) -> usize {
        let weight_params = self.out_features() * self.in_features();
        let bias_params = if self.bias().is_some() {
            self.out_features()
        } else {
            0
        };
        weight_params + bias_params
    }
}

// =============================================================================
// Helper functions for tensor layout conversion
// =============================================================================

/// Convert NCHW layout to NHWC layout
fn nchw_to_nhwc(input: &[f32], shape: &TensorShape) -> Vec<f32> {
    let n = shape.n;
    let c = shape.c;
    let h = shape.h;
    let w = shape.w;

    let mut output = vec![0.0; input.len()];

    for batch in 0..n {
        for channel in 0..c {
            for row in 0..h {
                for col in 0..w {
                    let nchw_idx = batch * c * h * w + channel * h * w + row * w + col;
                    let nhwc_idx = batch * h * w * c + row * w * c + col * c + channel;
                    output[nhwc_idx] = input[nchw_idx];
                }
            }
        }
    }

    output
}

/// Convert NHWC layout to NCHW layout
fn nhwc_to_nchw(input: &[f32], shape: &TensorShape) -> Vec<f32> {
    let n = shape.n;
    let c = shape.c;
    let h = shape.h;
    let w = shape.w;

    let mut output = vec![0.0; input.len()];

    for batch in 0..n {
        for row in 0..h {
            for col in 0..w {
                for channel in 0..c {
                    let nhwc_idx = batch * h * w * c + row * w * c + col * c + channel;
                    let nchw_idx = batch * c * h * w + channel * h * w + row * w + col;
                    output[nchw_idx] = input[nhwc_idx];
                }
            }
        }
    }

    output
}
