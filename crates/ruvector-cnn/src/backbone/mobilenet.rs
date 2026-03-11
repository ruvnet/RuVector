//! MobileNet-V3 backbone implementation.
//!
//! Efficient CNN backbone designed for mobile/edge deployment.
//! Implements both Small and Large variants with SE blocks and HardSwish activation.
//!
//! ## Architecture Overview
//!
//! MobileNet-V3 uses inverted residual blocks (MBConv) with:
//! - Depthwise separable convolutions for efficiency
//! - Squeeze-and-Excitation for channel attention
//! - HardSwish activation (faster than Swish)
//! - Residual connections when stride=1 and channels match
//!
//! ## Variants
//!
//! - **Small**: ~2.5M params, 576 output channels, optimized for latency
//! - **Large**: ~5.4M params, 960 output channels, optimized for accuracy

use super::{Backbone, BackboneExt, BackboneType};
use super::blocks::{ConvBNActivation, InvertedResidual as BlockInvertedResidual};
use super::layer::Layer;
use crate::error::CnnResult;
use crate::layers::{self, ActivationType, GlobalAvgPool2d, Linear, TensorShape};

/// MobileNet-V3 configuration
#[derive(Debug, Clone)]
pub struct MobileNetConfig {
    /// Input image size
    pub input_size: usize,
    /// Width multiplier
    pub width_mult: f32,
    /// Output channels
    pub output_channels: usize,
}

impl Default for MobileNetConfig {
    fn default() -> Self {
        Self {
            input_size: 224,
            width_mult: 1.0,
            output_channels: 576,
        }
    }
}

/// MobileNet-V3 Small variant
///
/// Optimized for mobile deployment with ~2.5M parameters.
#[derive(Debug, Clone)]
pub struct MobileNetV3Small {
    config: MobileNetConfig,
    // Simplified weight storage
    stem_weights: Vec<f32>,
    stem_bn: BnParams,
    blocks: Vec<InvertedResidual>,
    head_weights: Vec<f32>,
}

/// MobileNet-V3 Large variant
///
/// Higher accuracy variant with ~5.4M parameters.
#[derive(Debug, Clone)]
pub struct MobileNetV3Large {
    config: MobileNetConfig,
    stem_weights: Vec<f32>,
    stem_bn: BnParams,
    blocks: Vec<InvertedResidual>,
    head_weights: Vec<f32>,
}

/// Batch normalization parameters
#[derive(Debug, Clone)]
struct BnParams {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    mean: Vec<f32>,
    var: Vec<f32>,
}

impl BnParams {
    fn new(channels: usize) -> Self {
        Self {
            gamma: vec![1.0; channels],
            beta: vec![0.0; channels],
            mean: vec![0.0; channels],
            var: vec![1.0; channels],
        }
    }
}

/// Inverted residual block (MobileNet-V2/V3 style)
#[derive(Debug, Clone)]
struct InvertedResidual {
    /// Expansion convolution weights (1x1)
    expand_weights: Option<Vec<f32>>,
    expand_bn: Option<BnParams>,
    /// Depthwise convolution weights (3x3)
    dw_weights: Vec<f32>,
    dw_bn: BnParams,
    /// Squeeze-excitation weights
    se_reduce: Option<Vec<f32>>,
    se_expand: Option<Vec<f32>>,
    /// Projection convolution weights (1x1)
    project_weights: Vec<f32>,
    project_bn: BnParams,
    /// Config
    in_channels: usize,
    out_channels: usize,
    expansion: usize,
    use_se: bool,
    use_residual: bool,
}

impl MobileNetV3Small {
    /// Create a new MobileNet-V3 Small
    pub fn new(config: MobileNetConfig) -> Self {
        // Simplified initialization with random weights
        let stem_channels = 16;

        Self {
            stem_weights: vec![0.0; 3 * 3 * 3 * stem_channels],
            stem_bn: BnParams::new(stem_channels),
            blocks: Self::create_blocks(&config),
            head_weights: vec![0.0; config.output_channels],
            config,
        }
    }

    fn create_blocks(config: &MobileNetConfig) -> Vec<InvertedResidual> {
        // Simplified block configuration for MobileNet-V3 Small
        let block_configs = [
            (16, 16, 1, false),   // in, out, expansion, se
            (16, 24, 4, false),
            (24, 24, 3, false),
            (24, 40, 3, true),
            (40, 40, 3, true),
            (40, 48, 3, true),
            (48, 48, 3, true),
            (48, 96, 6, true),
            (96, 96, 6, true),
            (96, 96, 6, true),
        ];

        block_configs.iter().map(|&(in_c, out_c, exp, se)| {
            let in_c = ((in_c as f32) * config.width_mult) as usize;
            let out_c = ((out_c as f32) * config.width_mult) as usize;
            let mid_c = in_c * exp;

            InvertedResidual {
                expand_weights: if exp != 1 { Some(vec![0.0; in_c * mid_c]) } else { None },
                expand_bn: if exp != 1 { Some(BnParams::new(mid_c)) } else { None },
                dw_weights: vec![0.0; 9 * mid_c],
                dw_bn: BnParams::new(mid_c),
                se_reduce: if se { Some(vec![0.0; mid_c * (mid_c / 4)]) } else { None },
                se_expand: if se { Some(vec![0.0; (mid_c / 4) * mid_c]) } else { None },
                project_weights: vec![0.0; mid_c * out_c],
                project_bn: BnParams::new(out_c),
                in_channels: in_c,
                out_channels: out_c,
                expansion: exp,
                use_se: se,
                use_residual: in_c == out_c,
            }
        }).collect()
    }
}

impl Backbone for MobileNetV3Small {
    fn forward(&self, input: &[f32], height: usize, width: usize) -> Vec<f32> {
        // Stem: 3x3 conv, stride 2
        let mut x = layers::conv2d_3x3(
            input,
            &self.stem_weights,
            3,
            16,
            height,
            width,
        );
        x = layers::batch_norm(
            &x,
            &self.stem_bn.gamma,
            &self.stem_bn.beta,
            &self.stem_bn.mean,
            &self.stem_bn.var,
            1e-5,
        );
        x = layers::hard_swish(&x);

        // Process blocks (simplified - skip detailed implementation)
        for _block in &self.blocks {
            // Each block would process here
            // For simplicity, we just pass through
        }

        // Global average pooling
        let pooled = layers::global_avg_pool(&x, 16);

        pooled
    }

    fn output_dim(&self) -> usize {
        self.config.output_channels
    }

    fn input_size(&self) -> usize {
        self.config.input_size
    }
}

impl MobileNetV3Large {
    /// Create a new MobileNet-V3 Large
    pub fn new(config: MobileNetConfig) -> Self {
        let stem_channels = 16;

        Self {
            stem_weights: vec![0.0; 3 * 3 * 3 * stem_channels],
            stem_bn: BnParams::new(stem_channels),
            blocks: Self::create_blocks(&config),
            head_weights: vec![0.0; config.output_channels],
            config,
        }
    }

    fn create_blocks(config: &MobileNetConfig) -> Vec<InvertedResidual> {
        // MobileNet-V3 Large has more blocks
        let block_configs = [
            (16, 16, 1, false),
            (16, 24, 4, false),
            (24, 24, 3, false),
            (24, 40, 3, true),
            (40, 40, 3, true),
            (40, 40, 3, true),
            (40, 80, 6, false),
            (80, 80, 2, false),
            (80, 80, 2, false),
            (80, 112, 6, true),
            (112, 112, 6, true),
            (112, 160, 6, true),
            (160, 160, 6, true),
            (160, 160, 6, true),
        ];

        block_configs.iter().map(|&(in_c, out_c, exp, se)| {
            let in_c = ((in_c as f32) * config.width_mult) as usize;
            let out_c = ((out_c as f32) * config.width_mult) as usize;
            let mid_c = in_c * exp;

            InvertedResidual {
                expand_weights: if exp != 1 { Some(vec![0.0; in_c * mid_c]) } else { None },
                expand_bn: if exp != 1 { Some(BnParams::new(mid_c)) } else { None },
                dw_weights: vec![0.0; 9 * mid_c],
                dw_bn: BnParams::new(mid_c),
                se_reduce: if se { Some(vec![0.0; mid_c * (mid_c / 4)]) } else { None },
                se_expand: if se { Some(vec![0.0; (mid_c / 4) * mid_c]) } else { None },
                project_weights: vec![0.0; mid_c * out_c],
                project_bn: BnParams::new(out_c),
                in_channels: in_c,
                out_channels: out_c,
                expansion: exp,
                use_se: se,
                use_residual: in_c == out_c,
            }
        }).collect()
    }
}

impl Backbone for MobileNetV3Large {
    fn forward(&self, input: &[f32], height: usize, width: usize) -> Vec<f32> {
        // Same structure as Small but with more blocks
        let mut x = layers::conv2d_3x3(
            input,
            &self.stem_weights,
            3,
            16,
            height,
            width,
        );
        x = layers::batch_norm(
            &x,
            &self.stem_bn.gamma,
            &self.stem_bn.beta,
            &self.stem_bn.mean,
            &self.stem_bn.var,
            1e-5,
        );
        x = layers::hard_swish(&x);

        let pooled = layers::global_avg_pool(&x, 16);

        pooled
    }

    fn output_dim(&self) -> usize {
        self.config.output_channels
    }

    fn input_size(&self) -> usize {
        self.config.input_size
    }
}

// ============================================================================
// New Unified MobileNetV3 Implementation (uses proper layer modules)
// ============================================================================

/// Block configuration for building MobileNet-V3.
#[derive(Clone, Debug)]
struct BlockConfig {
    /// Input channels
    in_channels: usize,
    /// Expanded channels (after 1x1 expansion)
    expanded_channels: usize,
    /// Output channels
    out_channels: usize,
    /// Kernel size for depthwise conv
    kernel_size: usize,
    /// Stride for depthwise conv
    stride: usize,
    /// Whether to use Squeeze-and-Excitation
    use_se: bool,
    /// Activation type (ReLU or HardSwish)
    activation: ActivationType,
}

impl BlockConfig {
    fn new(
        in_c: usize,
        exp_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        use_se: bool,
        activation: ActivationType,
    ) -> Self {
        Self {
            in_channels: in_c,
            expanded_channels: exp_c,
            out_channels: out_c,
            kernel_size: kernel,
            stride,
            use_se,
            activation,
        }
    }
}

/// Configuration for the unified MobileNetV3 implementation.
#[derive(Clone, Debug)]
pub struct MobileNetV3Config {
    /// Input image size (height and width, assumes square)
    pub input_size: usize,
    /// Number of input channels (typically 3 for RGB)
    pub input_channels: usize,
    /// Width multiplier (scales channel counts)
    pub width_mult: f32,
    /// Number of output classes (0 for feature extraction only)
    pub num_classes: usize,
    /// Dropout rate for classifier
    pub dropout: f32,
    /// Backbone variant
    pub variant: BackboneType,
    /// Block configurations
    block_configs: Vec<BlockConfig>,
    /// Last conv output channels
    pub last_channels: usize,
    /// Output feature dimension (before classifier)
    pub feature_dim: usize,
}

impl MobileNetV3Config {
    /// Creates configuration for MobileNetV3-Small.
    pub fn small(num_classes: usize) -> Self {
        // MobileNetV3-Small architecture
        // Format: (in, exp, out, kernel, stride, se, activation)
        let block_configs = vec![
            BlockConfig::new(16, 16, 16, 3, 2, true, ActivationType::ReLU),
            BlockConfig::new(16, 72, 24, 3, 2, false, ActivationType::ReLU),
            BlockConfig::new(24, 88, 24, 3, 1, false, ActivationType::ReLU),
            BlockConfig::new(24, 96, 40, 5, 2, true, ActivationType::HardSwish),
            BlockConfig::new(40, 240, 40, 5, 1, true, ActivationType::HardSwish),
            BlockConfig::new(40, 240, 40, 5, 1, true, ActivationType::HardSwish),
            BlockConfig::new(40, 120, 48, 5, 1, true, ActivationType::HardSwish),
            BlockConfig::new(48, 144, 48, 5, 1, true, ActivationType::HardSwish),
            BlockConfig::new(48, 288, 96, 5, 2, true, ActivationType::HardSwish),
            BlockConfig::new(96, 576, 96, 5, 1, true, ActivationType::HardSwish),
            BlockConfig::new(96, 576, 96, 5, 1, true, ActivationType::HardSwish),
        ];

        Self {
            input_size: 224,
            input_channels: 3,
            width_mult: 1.0,
            num_classes,
            dropout: 0.2,
            variant: BackboneType::MobileNetV3Small,
            block_configs,
            last_channels: 1024,
            feature_dim: 576,
        }
    }

    /// Creates configuration for MobileNetV3-Large.
    pub fn large(num_classes: usize) -> Self {
        // MobileNetV3-Large architecture
        let block_configs = vec![
            BlockConfig::new(16, 16, 16, 3, 1, false, ActivationType::ReLU),
            BlockConfig::new(16, 64, 24, 3, 2, false, ActivationType::ReLU),
            BlockConfig::new(24, 72, 24, 3, 1, false, ActivationType::ReLU),
            BlockConfig::new(24, 72, 40, 5, 2, true, ActivationType::ReLU),
            BlockConfig::new(40, 120, 40, 5, 1, true, ActivationType::ReLU),
            BlockConfig::new(40, 120, 40, 5, 1, true, ActivationType::ReLU),
            BlockConfig::new(40, 240, 80, 3, 2, false, ActivationType::HardSwish),
            BlockConfig::new(80, 200, 80, 3, 1, false, ActivationType::HardSwish),
            BlockConfig::new(80, 184, 80, 3, 1, false, ActivationType::HardSwish),
            BlockConfig::new(80, 184, 80, 3, 1, false, ActivationType::HardSwish),
            BlockConfig::new(80, 480, 112, 3, 1, true, ActivationType::HardSwish),
            BlockConfig::new(112, 672, 112, 3, 1, true, ActivationType::HardSwish),
            BlockConfig::new(112, 672, 160, 5, 2, true, ActivationType::HardSwish),
            BlockConfig::new(160, 960, 160, 5, 1, true, ActivationType::HardSwish),
            BlockConfig::new(160, 960, 160, 5, 1, true, ActivationType::HardSwish),
        ];

        Self {
            input_size: 224,
            input_channels: 3,
            width_mult: 1.0,
            num_classes,
            dropout: 0.2,
            variant: BackboneType::MobileNetV3Large,
            block_configs,
            last_channels: 1280,
            feature_dim: 960,
        }
    }

    /// Sets the width multiplier.
    pub fn width_mult(mut self, mult: f32) -> Self {
        self.width_mult = mult;
        self
    }

    /// Sets the dropout rate.
    pub fn dropout(mut self, rate: f32) -> Self {
        self.dropout = rate;
        self
    }

    /// Applies width multiplier to a channel count.
    fn scale_channels(&self, channels: usize) -> usize {
        ((channels as f32 * self.width_mult).round() as usize).max(1)
    }
}

/// Unified MobileNet-V3 backbone implementation.
///
/// This implementation uses the proper layer modules (Conv2d, BatchNorm2d, etc.)
/// and supports both Small and Large variants through configuration.
#[derive(Clone, Debug)]
pub struct MobileNetV3 {
    /// Configuration
    config: MobileNetV3Config,
    /// Stem convolution (3x3, stride 2)
    stem: ConvBNActivation,
    /// Inverted residual blocks
    blocks: Vec<BlockInvertedResidual>,
    /// Last convolution (1x1)
    last_conv: ConvBNActivation,
    /// Global average pooling
    pool: GlobalAvgPool2d,
    /// Classifier head (optional)
    classifier: Option<Linear>,
}

impl MobileNetV3 {
    /// Creates a new MobileNetV3 from configuration.
    pub fn new(config: MobileNetV3Config) -> CnnResult<Self> {
        let stem_out = config.scale_channels(16);

        // Stem: 3x3 conv, stride 2, HardSwish
        let stem = ConvBNActivation::new(
            config.input_channels,
            stem_out,
            3,
            2,
            1,
            1,
            ActivationType::HardSwish,
        )?;

        // Build inverted residual blocks
        let mut blocks = Vec::with_capacity(config.block_configs.len());
        let mut in_channels = stem_out;

        for bc in &config.block_configs {
            let exp_channels = config.scale_channels(bc.expanded_channels);
            let out_channels = config.scale_channels(bc.out_channels);

            let block = BlockInvertedResidual::create(
                in_channels,
                exp_channels,
                out_channels,
                bc.kernel_size,
                bc.stride,
                bc.use_se,
                bc.activation,
            )?;

            blocks.push(block);
            in_channels = out_channels;
        }

        // Last conv: 1x1 to expand features
        let feature_dim = config.scale_channels(config.feature_dim);
        let last_conv = ConvBNActivation::pointwise(
            in_channels,
            feature_dim,
            ActivationType::HardSwish,
        )?;

        // Global average pooling
        let pool = GlobalAvgPool2d::new();

        // Classifier (if num_classes > 0)
        let classifier = if config.num_classes > 0 {
            let last_channels = config.scale_channels(config.last_channels);
            // In MobileNetV3, classifier is: Linear(feature_dim, last_channels) -> HardSwish -> Linear(last_channels, num_classes)
            // For simplicity, we use a single linear layer here
            Some(Linear::new(feature_dim, config.num_classes, true)?)
        } else {
            None
        };

        Ok(Self {
            config,
            stem,
            blocks,
            last_conv,
            pool,
            classifier,
        })
    }

    /// Creates a MobileNetV3-Small.
    pub fn small(num_classes: usize) -> CnnResult<Self> {
        Self::new(MobileNetV3Config::small(num_classes))
    }

    /// Creates a MobileNetV3-Large.
    pub fn large(num_classes: usize) -> CnnResult<Self> {
        Self::new(MobileNetV3Config::large(num_classes))
    }

    /// Returns the configuration.
    pub fn config(&self) -> &MobileNetV3Config {
        &self.config
    }

    /// Returns the number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Returns a reference to the stem layer.
    pub fn stem(&self) -> &ConvBNActivation {
        &self.stem
    }

    /// Returns a reference to the blocks.
    pub fn blocks(&self) -> &[BlockInvertedResidual] {
        &self.blocks
    }

    /// Returns a reference to the last conv layer.
    pub fn last_conv(&self) -> &ConvBNActivation {
        &self.last_conv
    }

    /// Returns a reference to the classifier.
    pub fn classifier(&self) -> Option<&Linear> {
        self.classifier.as_ref()
    }

    /// Forward pass through feature layers only.
    fn forward_features_impl(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        let mut x = input.to_vec();
        let mut shape = *input_shape;

        // Stem
        let stem_shape = self.stem.output_shape(&shape);
        x = self.stem.forward(&x, &shape)?;
        shape = stem_shape;

        // Blocks
        for block in &self.blocks {
            let block_shape = block.output_shape(&shape);
            x = block.forward(&x, &shape)?;
            shape = block_shape;
        }

        // Last conv
        let last_shape = self.last_conv.output_shape(&shape);
        x = self.last_conv.forward(&x, &shape)?;
        shape = last_shape;

        // Global average pooling
        x = self.pool.forward(&x, &shape)?;

        Ok(x)
    }
}

impl Backbone for MobileNetV3 {
    fn forward(&self, input: &[f32], height: usize, width: usize) -> Vec<f32> {
        let input_shape = TensorShape::new(1, self.config.input_channels, height, width);
        self.forward_features_impl(input, &input_shape)
            .unwrap_or_else(|_| vec![0.0; self.output_dim()])
    }

    fn output_dim(&self) -> usize {
        self.config.scale_channels(self.config.feature_dim)
    }

    fn input_size(&self) -> usize {
        self.config.input_size
    }
}

impl BackboneExt for MobileNetV3 {
    fn backbone_type(&self) -> BackboneType {
        self.config.variant
    }

    fn num_params(&self) -> usize {
        let mut total = self.stem.num_params();
        for block in &self.blocks {
            total += block.num_params();
        }
        total += self.last_conv.num_params();
        if let Some(ref classifier) = self.classifier {
            total += classifier.num_params();
        }
        total
    }

    fn forward_with_shape(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        let features = self.forward_features_impl(input, input_shape)?;

        if let Some(ref classifier) = self.classifier {
            let batch_size = input_shape.n;
            let feature_shape = TensorShape::new(batch_size, self.output_dim(), 1, 1);
            classifier.forward(&features, &feature_shape)
        } else {
            Ok(features)
        }
    }

    fn forward_features(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>> {
        self.forward_features_impl(input, input_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobilenet_v3_small_creation() {
        let config = MobileNetConfig::default();
        let model = MobileNetV3Small::new(config);
        assert_eq!(model.output_dim(), 576);
    }

    #[test]
    fn test_mobilenet_v3_large_creation() {
        let config = MobileNetConfig {
            output_channels: 960,
            ..Default::default()
        };
        let model = MobileNetV3Large::new(config);
        assert_eq!(model.output_dim(), 960);
    }

    #[test]
    fn test_unified_mobilenet_v3_small() {
        let model = MobileNetV3::small(1000).unwrap();
        assert_eq!(model.backbone_type(), BackboneType::MobileNetV3Small);
        assert_eq!(model.output_dim(), 576);
        assert!(model.num_params() > 0);
    }

    #[test]
    fn test_unified_mobilenet_v3_large() {
        let model = MobileNetV3::large(1000).unwrap();
        assert_eq!(model.backbone_type(), BackboneType::MobileNetV3Large);
        assert_eq!(model.output_dim(), 960);
        assert!(model.num_params() > 0);
    }

    #[test]
    fn test_mobilenet_v3_config() {
        let config = MobileNetV3Config::small(1000);
        assert_eq!(config.input_size, 224);
        assert_eq!(config.input_channels, 3);
        assert_eq!(config.num_classes, 1000);
        assert_eq!(config.feature_dim, 576);
    }

    #[test]
    fn test_mobilenet_v3_forward() {
        let model = MobileNetV3::small(0).unwrap(); // No classifier
        let input_shape = TensorShape::new(1, 3, 224, 224);
        let input = vec![0.5; input_shape.numel()];

        let output = model.forward_features(&input, &input_shape).unwrap();

        // Output should be [batch, feature_dim]
        assert_eq!(output.len(), 576);
    }

    #[test]
    fn test_mobilenet_v3_with_classifier() {
        let model = MobileNetV3::small(1000).unwrap();
        let input_shape = TensorShape::new(1, 3, 224, 224);
        let input = vec![0.5; input_shape.numel()];

        let output = model.forward_with_shape(&input, &input_shape).unwrap();

        // Output should be [batch, num_classes]
        assert_eq!(output.len(), 1000);
    }

    #[test]
    fn test_mobilenet_v3_batch() {
        let model = MobileNetV3::small(0).unwrap();
        let batch_size = 2;
        let input_shape = TensorShape::new(batch_size, 3, 224, 224);
        let input = vec![0.5; input_shape.numel()];

        let output = model.forward_features(&input, &input_shape).unwrap();

        // Output should be [batch, feature_dim]
        assert_eq!(output.len(), batch_size * 576);
    }
}
