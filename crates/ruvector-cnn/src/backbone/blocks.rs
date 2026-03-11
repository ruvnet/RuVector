//! MobileNet-V3 building blocks.
//!
//! This module implements the core building blocks used in MobileNet-V3:
//! - ConvBNActivation: Conv2d + BatchNorm + Activation
//! - SqueezeExcitation: Channel attention mechanism
//! - InvertedResidual: The main building block with optional SE

use crate::error::CnnResult;
use crate::layers::{
    Activation, ActivationType, BatchNorm2d, Conv2d, GlobalAvgPool2d, Linear, TensorShape,
};
use super::layer::Layer;

/// Convolution + BatchNorm + Activation block.
///
/// A standard building block that combines:
/// 1. 2D Convolution
/// 2. Batch Normalization
/// 3. Activation function
#[derive(Clone, Debug)]
pub struct ConvBNActivation {
    /// Convolution layer
    conv: Conv2d,
    /// Batch normalization layer
    bn: BatchNorm2d,
    /// Activation function
    activation: Activation,
}

impl ConvBNActivation {
    /// Creates a new ConvBNActivation block.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        activation: ActivationType,
    ) -> CnnResult<Self> {
        let conv = Conv2d::builder(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .groups(groups)
            .bias(false) // No bias when using BatchNorm
            .build()?;

        let bn = BatchNorm2d::new(out_channels);
        let activation = Activation::new(activation);

        Ok(Self {
            conv,
            bn,
            activation,
        })
    }

    /// Creates a standard 3x3 convolution block.
    pub fn conv3x3(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        activation: ActivationType,
    ) -> CnnResult<Self> {
        Self::new(in_channels, out_channels, 3, stride, 1, 1, activation)
    }

    /// Creates a pointwise (1x1) convolution block.
    pub fn pointwise(
        in_channels: usize,
        out_channels: usize,
        activation: ActivationType,
    ) -> CnnResult<Self> {
        Self::new(in_channels, out_channels, 1, 1, 0, 1, activation)
    }

    /// Creates a depthwise convolution block.
    pub fn depthwise(
        channels: usize,
        kernel_size: usize,
        stride: usize,
        activation: ActivationType,
    ) -> CnnResult<Self> {
        let padding = kernel_size / 2;
        Self::new(channels, channels, kernel_size, stride, padding, channels, activation)
    }

    /// Returns a reference to the convolution layer.
    pub fn conv(&self) -> &Conv2d {
        &self.conv
    }

    /// Returns a mutable reference to the convolution layer.
    pub fn conv_mut(&mut self) -> &mut Conv2d {
        &mut self.conv
    }

    /// Returns a reference to the batch normalization layer.
    pub fn bn(&self) -> &BatchNorm2d {
        &self.bn
    }

    /// Returns a mutable reference to the batch normalization layer.
    pub fn bn_mut(&mut self) -> &mut BatchNorm2d {
        &mut self.bn
    }
}

impl Layer for ConvBNActivation {
    fn forward(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        // Conv -> BN -> Activation
        // Use UFCS to call the backbone Layer trait methods, not the inherent methods
        let conv_shape = Layer::output_shape(&self.conv, input_shape);
        let conv_out = Layer::forward(&self.conv, input, input_shape)?;

        let bn_out = Layer::forward(&self.bn, &conv_out, &conv_shape)?;
        let act_out = Layer::forward(&self.activation, &bn_out, &conv_shape)?;

        Ok(act_out)
    }

    fn output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        Layer::output_shape(&self.conv, input_shape)
    }

    fn num_params(&self) -> usize {
        Layer::num_params(&self.conv) + Layer::num_params(&self.bn)
    }
}

/// Squeeze-and-Excitation block for channel attention.
///
/// Applies channel-wise attention by:
/// 1. Global average pooling to squeeze spatial dimensions
/// 2. FC layer to reduce channels
/// 3. ReLU activation
/// 4. FC layer to restore channels
/// 5. HardSigmoid activation for gating
/// 6. Scale input by the computed attention weights
#[derive(Clone, Debug)]
pub struct SqueezeExcitation {
    /// Number of input/output channels
    channels: usize,
    /// Squeeze ratio (typically 4)
    squeeze_channels: usize,
    /// Global average pooling
    pool: GlobalAvgPool2d,
    /// First FC layer (squeeze)
    fc1: Linear,
    /// Second FC layer (excite)
    fc2: Linear,
    /// Activation after fc1
    relu: Activation,
    /// Activation after fc2 (gating)
    hard_sigmoid: Activation,
}

impl SqueezeExcitation {
    /// Creates a new Squeeze-and-Excitation block.
    ///
    /// # Arguments
    /// * `channels` - Number of input/output channels
    /// * `squeeze_channels` - Number of channels after squeeze (typically channels/4)
    pub fn new(channels: usize, squeeze_channels: usize) -> CnnResult<Self> {
        let pool = GlobalAvgPool2d::new();
        let fc1 = Linear::new(channels, squeeze_channels, true)?;
        let fc2 = Linear::new(squeeze_channels, channels, true)?;
        let relu = Activation::relu();
        let hard_sigmoid = Activation::hard_sigmoid();

        Ok(Self {
            channels,
            squeeze_channels,
            pool,
            fc1,
            fc2,
            relu,
            hard_sigmoid,
        })
    }

    /// Creates an SE block with default squeeze ratio of 4.
    pub fn with_default_ratio(channels: usize) -> CnnResult<Self> {
        let squeeze_channels = (channels / 4).max(1);
        Self::new(channels, squeeze_channels)
    }

    /// Returns the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the squeeze channels.
    pub fn squeeze_channels(&self) -> usize {
        self.squeeze_channels
    }

    /// Returns a reference to fc1.
    pub fn fc1(&self) -> &Linear {
        &self.fc1
    }

    /// Returns a mutable reference to fc1.
    pub fn fc1_mut(&mut self) -> &mut Linear {
        &mut self.fc1
    }

    /// Returns a reference to fc2.
    pub fn fc2(&self) -> &Linear {
        &self.fc2
    }

    /// Returns a mutable reference to fc2.
    pub fn fc2_mut(&mut self) -> &mut Linear {
        &mut self.fc2
    }

    /// Forward pass computing attention weights and scaling input.
    pub fn forward_scale(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        let batch_size = input_shape.n;
        let spatial_size = input_shape.spatial_size();

        // 1. Global average pooling: [N, C, H, W] -> [N, C]
        // Use UFCS to call the backbone Layer trait methods
        let _pooled_shape = Layer::output_shape(&self.pool, input_shape);
        let pooled = Layer::forward(&self.pool, input, input_shape)?;

        // 2. FC1 + ReLU: [N, C] -> [N, squeeze_C]
        let fc1_shape = TensorShape::new(batch_size, self.channels, 1, 1);
        let fc1_out = Layer::forward(&self.fc1, &pooled, &fc1_shape)?;
        let fc1_out_shape = TensorShape::new(batch_size, self.squeeze_channels, 1, 1);
        let relu_out = Layer::forward(&self.relu, &fc1_out, &fc1_out_shape)?;

        // 3. FC2 + HardSigmoid: [N, squeeze_C] -> [N, C]
        let fc2_out = Layer::forward(&self.fc2, &relu_out, &fc1_out_shape)?;
        let fc2_out_shape = TensorShape::new(batch_size, self.channels, 1, 1);
        let attention = Layer::forward(&self.hard_sigmoid, &fc2_out, &fc2_out_shape)?;

        // 4. Scale input by attention weights
        let mut output = vec![0.0; input.len()];

        for n in 0..batch_size {
            for c in 0..self.channels {
                let scale = attention[n * self.channels + c];
                let channel_offset = (n * self.channels + c) * spatial_size;

                for i in 0..spatial_size {
                    output[channel_offset + i] = input[channel_offset + i] * scale;
                }
            }
        }

        Ok(output)
    }
}

impl Layer for SqueezeExcitation {
    fn forward(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        self.forward_scale(input, input_shape)
    }

    fn output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        *input_shape
    }

    fn num_params(&self) -> usize {
        Layer::num_params(&self.fc1) + Layer::num_params(&self.fc2)
    }
}

/// Configuration for an InvertedResidual block.
#[derive(Clone, Debug)]
pub struct InvertedResidualConfig {
    /// Input channels
    pub in_channels: usize,
    /// Expanded channels (after expansion)
    pub expanded_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size for depthwise conv
    pub kernel_size: usize,
    /// Stride for depthwise conv
    pub stride: usize,
    /// Whether to use Squeeze-and-Excitation
    pub use_se: bool,
    /// Activation type for non-linear layers
    pub activation: ActivationType,
}

impl InvertedResidualConfig {
    /// Creates a new InvertedResidualConfig.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        expanded_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        use_se: bool,
        activation: ActivationType,
    ) -> Self {
        Self {
            in_channels,
            expanded_channels,
            out_channels,
            kernel_size,
            stride,
            use_se,
            activation,
        }
    }
}

/// Inverted Residual block (MBConv) - the main building block of MobileNet-V3.
///
/// Architecture:
/// 1. Expansion: 1x1 conv to expand channels (skipped if expansion ratio = 1)
/// 2. Depthwise: NxN depthwise conv
/// 3. SE: Optional Squeeze-and-Excitation
/// 4. Projection: 1x1 conv to project to output channels
/// 5. Residual: Add input if stride=1 and in_channels=out_channels
#[derive(Clone, Debug)]
pub struct InvertedResidual {
    /// Configuration
    config: InvertedResidualConfig,
    /// Expansion layer (None if expansion ratio = 1)
    expand: Option<ConvBNActivation>,
    /// Depthwise convolution
    depthwise: ConvBNActivation,
    /// Squeeze-and-Excitation block (optional)
    se: Option<SqueezeExcitation>,
    /// Projection layer (linear, no activation)
    project: ConvBNActivation,
    /// Whether to use residual connection
    use_residual: bool,
}

impl InvertedResidual {
    /// Creates a new InvertedResidual block from configuration.
    pub fn new(config: InvertedResidualConfig) -> CnnResult<Self> {
        // Determine if we need expansion
        let expand = if config.expanded_channels != config.in_channels {
            Some(ConvBNActivation::pointwise(
                config.in_channels,
                config.expanded_channels,
                config.activation,
            )?)
        } else {
            None
        };

        // Depthwise convolution
        let depthwise = ConvBNActivation::depthwise(
            config.expanded_channels,
            config.kernel_size,
            config.stride,
            config.activation,
        )?;

        // Optional SE
        let se = if config.use_se {
            let se_channels = (config.expanded_channels / 4).max(1);
            Some(SqueezeExcitation::new(config.expanded_channels, se_channels)?)
        } else {
            None
        };

        // Projection (linear, no activation)
        let project = ConvBNActivation::pointwise(
            config.expanded_channels,
            config.out_channels,
            ActivationType::Identity,
        )?;

        // Use residual connection if stride=1 and channels match
        let use_residual = config.stride == 1 && config.in_channels == config.out_channels;

        Ok(Self {
            config,
            expand,
            depthwise,
            se,
            project,
            use_residual,
        })
    }

    /// Creates an InvertedResidual with the given parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        in_channels: usize,
        expanded_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        use_se: bool,
        activation: ActivationType,
    ) -> CnnResult<Self> {
        let config = InvertedResidualConfig::new(
            in_channels,
            expanded_channels,
            out_channels,
            kernel_size,
            stride,
            use_se,
            activation,
        );
        Self::new(config)
    }

    /// Returns the configuration.
    pub fn config(&self) -> &InvertedResidualConfig {
        &self.config
    }

    /// Returns whether residual connection is used.
    pub fn use_residual(&self) -> bool {
        self.use_residual
    }

    /// Returns a reference to the expansion layer.
    pub fn expand(&self) -> Option<&ConvBNActivation> {
        self.expand.as_ref()
    }

    /// Returns a mutable reference to the expansion layer.
    pub fn expand_mut(&mut self) -> Option<&mut ConvBNActivation> {
        self.expand.as_mut()
    }

    /// Returns a reference to the depthwise layer.
    pub fn depthwise(&self) -> &ConvBNActivation {
        &self.depthwise
    }

    /// Returns a mutable reference to the depthwise layer.
    pub fn depthwise_mut(&mut self) -> &mut ConvBNActivation {
        &mut self.depthwise
    }

    /// Returns a reference to the SE block.
    pub fn se(&self) -> Option<&SqueezeExcitation> {
        self.se.as_ref()
    }

    /// Returns a mutable reference to the SE block.
    pub fn se_mut(&mut self) -> Option<&mut SqueezeExcitation> {
        self.se.as_mut()
    }

    /// Returns a reference to the projection layer.
    pub fn project(&self) -> &ConvBNActivation {
        &self.project
    }

    /// Returns a mutable reference to the projection layer.
    pub fn project_mut(&mut self) -> &mut ConvBNActivation {
        &mut self.project
    }
}

impl Layer for InvertedResidual {
    fn forward(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        let mut x = input.to_vec();
        let mut shape = *input_shape;

        // 1. Expansion
        if let Some(ref expand) = self.expand {
            let new_shape = expand.output_shape(&shape);
            x = expand.forward(&x, &shape)?;
            shape = new_shape;
        }

        // 2. Depthwise
        let dw_shape = self.depthwise.output_shape(&shape);
        x = self.depthwise.forward(&x, &shape)?;
        shape = dw_shape;

        // 3. Squeeze-and-Excitation
        if let Some(ref se) = self.se {
            x = se.forward(&x, &shape)?;
        }

        // 4. Projection
        let proj_shape = self.project.output_shape(&shape);
        x = self.project.forward(&x, &shape)?;

        // 5. Residual connection
        if self.use_residual {
            for (i, val) in x.iter_mut().enumerate() {
                *val += input[i];
            }
        }

        Ok(x)
    }

    fn output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        let mut shape = *input_shape;

        if let Some(ref expand) = self.expand {
            shape = expand.output_shape(&shape);
        }

        shape = self.depthwise.output_shape(&shape);
        shape = self.project.output_shape(&shape);

        shape
    }

    fn num_params(&self) -> usize {
        let expand_params = self.expand.as_ref().map_or(0, |e| e.num_params());
        let dw_params = self.depthwise.num_params();
        let se_params = self.se.as_ref().map_or(0, |s| s.num_params());
        let proj_params = self.project.num_params();

        expand_params + dw_params + se_params + proj_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_bn_activation_shape() {
        let block = ConvBNActivation::conv3x3(3, 16, 2, ActivationType::HardSwish).unwrap();
        let input_shape = TensorShape::new(1, 3, 224, 224);
        let output_shape = block.output_shape(&input_shape);

        assert_eq!(output_shape.n, 1);
        assert_eq!(output_shape.c, 16);
        assert_eq!(output_shape.h, 112);
        assert_eq!(output_shape.w, 112);
    }

    #[test]
    fn test_squeeze_excitation() {
        let se = SqueezeExcitation::new(64, 16).unwrap();
        assert_eq!(se.channels(), 64);
        assert_eq!(se.squeeze_channels(), 16);
    }

    #[test]
    fn test_se_output_shape() {
        let se = SqueezeExcitation::new(64, 16).unwrap();
        let input_shape = TensorShape::new(1, 64, 7, 7);
        let output_shape = se.output_shape(&input_shape);

        // SE should preserve shape
        assert_eq!(output_shape, input_shape);
    }

    #[test]
    fn test_inverted_residual_no_expansion() {
        let block = InvertedResidual::create(
            16, 16, 16, // in == exp == out (no expansion)
            3, 1, false, ActivationType::ReLU,
        )
        .unwrap();

        assert!(block.expand().is_none());
        assert!(block.use_residual());
    }

    #[test]
    fn test_inverted_residual_with_expansion() {
        let block = InvertedResidual::create(
            16, 64, 24, // expansion ratio 4
            3, 1, true, ActivationType::HardSwish,
        )
        .unwrap();

        assert!(block.expand().is_some());
        assert!(block.se().is_some());
        assert!(!block.use_residual()); // in != out
    }

    #[test]
    fn test_inverted_residual_output_shape() {
        let block = InvertedResidual::create(
            16, 64, 24,
            3, 2, // stride 2
            true, ActivationType::HardSwish,
        )
        .unwrap();

        let input_shape = TensorShape::new(1, 16, 56, 56);
        let output_shape = block.output_shape(&input_shape);

        assert_eq!(output_shape.n, 1);
        assert_eq!(output_shape.c, 24);
        assert_eq!(output_shape.h, 28);
        assert_eq!(output_shape.w, 28);
    }

    #[test]
    fn test_inverted_residual_params() {
        let block = InvertedResidual::create(
            16, 64, 24,
            3, 1,
            true, ActivationType::HardSwish,
        )
        .unwrap();

        // Should have params from: expand, depthwise, SE, project
        assert!(block.num_params() > 0);
    }
}
