//! Configuration types for CNN layers and feature extractors.
//!
//! This module provides configuration structs and builders for various
//! CNN components including convolutions, pooling, normalization, and backbones.

use serde::{Deserialize, Serialize};

use crate::error::{CnnError, CnnResult};

/// Configuration for 2D convolution layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvConfig {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels (filters)
    pub out_channels: usize,
    /// Kernel size (assumes square kernels)
    pub kernel_size: usize,
    /// Stride for the convolution
    pub stride: usize,
    /// Padding applied to input
    pub padding: usize,
    /// Dilation factor
    pub dilation: usize,
    /// Number of groups for grouped convolution
    pub groups: usize,
    /// Whether to include bias term
    pub bias: bool,
}

impl ConvConfig {
    /// Creates a new builder for ConvConfig.
    pub fn builder() -> ConvConfigBuilder {
        ConvConfigBuilder::default()
    }

    /// Validates the configuration.
    pub fn validate(&self) -> CnnResult<()> {
        if self.in_channels == 0 {
            return Err(CnnError::InvalidConfig(
                "in_channels must be greater than 0".to_string(),
            ));
        }

        if self.out_channels == 0 {
            return Err(CnnError::InvalidConfig(
                "out_channels must be greater than 0".to_string(),
            ));
        }

        if self.kernel_size == 0 {
            return Err(CnnError::InvalidConfig(
                "kernel_size must be greater than 0".to_string(),
            ));
        }

        if self.stride == 0 {
            return Err(CnnError::InvalidConfig(
                "stride must be greater than 0".to_string(),
            ));
        }

        if self.dilation == 0 {
            return Err(CnnError::InvalidConfig(
                "dilation must be greater than 0".to_string(),
            ));
        }

        if self.groups == 0 {
            return Err(CnnError::InvalidConfig(
                "groups must be greater than 0".to_string(),
            ));
        }

        if self.in_channels % self.groups != 0 {
            return Err(CnnError::InvalidConfig(format!(
                "in_channels ({}) must be divisible by groups ({})",
                self.in_channels, self.groups
            )));
        }

        if self.out_channels % self.groups != 0 {
            return Err(CnnError::InvalidConfig(format!(
                "out_channels ({}) must be divisible by groups ({})",
                self.out_channels, self.groups
            )));
        }

        Ok(())
    }

    /// Computes the output spatial size for given input size.
    #[inline]
    pub fn output_size(&self, input_size: usize) -> usize {
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1;
        (input_size + 2 * self.padding - effective_kernel) / self.stride + 1
    }
}

/// Builder for ConvConfig.
#[derive(Default)]
pub struct ConvConfigBuilder {
    in_channels: Option<usize>,
    out_channels: Option<usize>,
    kernel_size: Option<usize>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
}

impl ConvConfigBuilder {
    /// Sets the number of input channels.
    pub fn in_channels(mut self, in_channels: usize) -> Self {
        self.in_channels = Some(in_channels);
        self
    }

    /// Sets the number of output channels.
    pub fn out_channels(mut self, out_channels: usize) -> Self {
        self.out_channels = Some(out_channels);
        self
    }

    /// Sets the kernel size.
    pub fn kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = Some(kernel_size);
        self
    }

    /// Sets the stride.
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Sets the padding.
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the dilation.
    pub fn dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Sets the number of groups.
    pub fn groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Sets whether to include bias.
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Builds the ConvConfig.
    pub fn build(self) -> CnnResult<ConvConfig> {
        let config = ConvConfig {
            in_channels: self.in_channels.ok_or_else(|| {
                CnnError::InvalidConfig("in_channels must be specified".to_string())
            })?,
            out_channels: self.out_channels.ok_or_else(|| {
                CnnError::InvalidConfig("out_channels must be specified".to_string())
            })?,
            kernel_size: self.kernel_size.ok_or_else(|| {
                CnnError::InvalidConfig("kernel_size must be specified".to_string())
            })?,
            stride: if self.stride == 0 { 1 } else { self.stride },
            padding: self.padding,
            dilation: if self.dilation == 0 { 1 } else { self.dilation },
            groups: if self.groups == 0 { 1 } else { self.groups },
            bias: self.bias,
        };

        config.validate()?;
        Ok(config)
    }
}

/// Configuration for pooling layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Kernel size for pooling
    pub kernel_size: usize,
    /// Stride for pooling
    pub stride: usize,
    /// Padding applied to input
    pub padding: usize,
    /// Whether to ceil the output size (vs floor)
    pub ceil_mode: bool,
}

impl PoolConfig {
    /// Creates a new builder for PoolConfig.
    pub fn builder() -> PoolConfigBuilder {
        PoolConfigBuilder::default()
    }

    /// Validates the configuration.
    pub fn validate(&self) -> CnnResult<()> {
        if self.kernel_size == 0 {
            return Err(CnnError::InvalidConfig(
                "kernel_size must be greater than 0".to_string(),
            ));
        }

        if self.stride == 0 {
            return Err(CnnError::InvalidConfig(
                "stride must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Computes the output spatial size for given input size.
    #[inline]
    pub fn output_size(&self, input_size: usize) -> usize {
        let numerator = input_size + 2 * self.padding - self.kernel_size;
        if self.ceil_mode {
            (numerator + self.stride - 1) / self.stride + 1
        } else {
            numerator / self.stride + 1
        }
    }
}

/// Builder for PoolConfig.
#[derive(Default)]
pub struct PoolConfigBuilder {
    kernel_size: Option<usize>,
    stride: Option<usize>,
    padding: usize,
    ceil_mode: bool,
}

impl PoolConfigBuilder {
    /// Sets the kernel size.
    pub fn kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = Some(kernel_size);
        self
    }

    /// Sets the stride.
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = Some(stride);
        self
    }

    /// Sets the padding.
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Sets whether to use ceil mode.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.ceil_mode = ceil_mode;
        self
    }

    /// Builds the PoolConfig.
    pub fn build(self) -> CnnResult<PoolConfig> {
        let kernel_size = self.kernel_size.ok_or_else(|| {
            CnnError::InvalidConfig("kernel_size must be specified".to_string())
        })?;

        let config = PoolConfig {
            kernel_size,
            stride: self.stride.unwrap_or(kernel_size),
            padding: self.padding,
            ceil_mode: self.ceil_mode,
        };

        config.validate()?;
        Ok(config)
    }
}

/// Configuration for normalization layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormConfig {
    /// Number of features to normalize
    pub num_features: usize,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Momentum for running statistics
    pub momentum: f32,
    /// Whether to use learnable affine parameters
    pub affine: bool,
    /// Whether to track running mean/variance
    pub track_running_stats: bool,
}

impl NormConfig {
    /// Creates a new builder for NormConfig.
    pub fn builder() -> NormConfigBuilder {
        NormConfigBuilder::default()
    }

    /// Validates the configuration.
    pub fn validate(&self) -> CnnResult<()> {
        if self.num_features == 0 {
            return Err(CnnError::InvalidConfig(
                "num_features must be greater than 0".to_string(),
            ));
        }

        if self.eps <= 0.0 || !self.eps.is_finite() {
            return Err(CnnError::InvalidConfig(
                "eps must be positive and finite".to_string(),
            ));
        }

        if self.momentum < 0.0 || self.momentum > 1.0 {
            return Err(CnnError::InvalidConfig(
                "momentum must be in range [0.0, 1.0]".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for NormConfig.
#[derive(Default)]
pub struct NormConfigBuilder {
    num_features: Option<usize>,
    eps: f32,
    momentum: f32,
    affine: bool,
    track_running_stats: bool,
}

impl NormConfigBuilder {
    /// Sets the number of features.
    pub fn num_features(mut self, num_features: usize) -> Self {
        self.num_features = Some(num_features);
        self
    }

    /// Sets the epsilon.
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the momentum.
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets whether to use affine parameters.
    pub fn affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// Sets whether to track running statistics.
    pub fn track_running_stats(mut self, track: bool) -> Self {
        self.track_running_stats = track;
        self
    }

    /// Builds the NormConfig.
    pub fn build(self) -> CnnResult<NormConfig> {
        let config = NormConfig {
            num_features: self.num_features.ok_or_else(|| {
                CnnError::InvalidConfig("num_features must be specified".to_string())
            })?,
            eps: if self.eps == 0.0 { 1e-5 } else { self.eps },
            momentum: if self.momentum == 0.0 { 0.1 } else { self.momentum },
            affine: self.affine,
            track_running_stats: self.track_running_stats,
        };

        config.validate()?;
        Ok(config)
    }
}

/// Configuration for backbone feature extractors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackboneConfig {
    /// Input image channels (typically 3 for RGB)
    pub in_channels: usize,
    /// Output feature dimension
    pub out_features: usize,
    /// Base channel width
    pub base_channels: usize,
    /// Number of blocks/stages
    pub num_stages: usize,
    /// Whether to use global average pooling at the end
    pub global_pool: bool,
    /// Dropout rate
    pub dropout: f32,
}

impl BackboneConfig {
    /// Creates a new builder for BackboneConfig.
    pub fn builder() -> BackboneConfigBuilder {
        BackboneConfigBuilder::default()
    }

    /// Validates the configuration.
    pub fn validate(&self) -> CnnResult<()> {
        if self.in_channels == 0 {
            return Err(CnnError::InvalidConfig(
                "in_channels must be greater than 0".to_string(),
            ));
        }

        if self.out_features == 0 {
            return Err(CnnError::InvalidConfig(
                "out_features must be greater than 0".to_string(),
            ));
        }

        if self.base_channels == 0 {
            return Err(CnnError::InvalidConfig(
                "base_channels must be greater than 0".to_string(),
            ));
        }

        if self.num_stages == 0 {
            return Err(CnnError::InvalidConfig(
                "num_stages must be greater than 0".to_string(),
            ));
        }

        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(CnnError::InvalidConfig(
                "dropout must be in range [0.0, 1.0]".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for BackboneConfig.
#[derive(Default)]
pub struct BackboneConfigBuilder {
    in_channels: usize,
    out_features: usize,
    base_channels: usize,
    num_stages: usize,
    global_pool: bool,
    dropout: f32,
}

impl BackboneConfigBuilder {
    /// Sets the input channels.
    pub fn in_channels(mut self, in_channels: usize) -> Self {
        self.in_channels = in_channels;
        self
    }

    /// Sets the output features.
    pub fn out_features(mut self, out_features: usize) -> Self {
        self.out_features = out_features;
        self
    }

    /// Sets the base channels.
    pub fn base_channels(mut self, base_channels: usize) -> Self {
        self.base_channels = base_channels;
        self
    }

    /// Sets the number of stages.
    pub fn num_stages(mut self, num_stages: usize) -> Self {
        self.num_stages = num_stages;
        self
    }

    /// Sets whether to use global pooling.
    pub fn global_pool(mut self, global_pool: bool) -> Self {
        self.global_pool = global_pool;
        self
    }

    /// Sets the dropout rate.
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Builds the BackboneConfig.
    pub fn build(self) -> CnnResult<BackboneConfig> {
        let config = BackboneConfig {
            in_channels: if self.in_channels == 0 { 3 } else { self.in_channels },
            out_features: if self.out_features == 0 { 512 } else { self.out_features },
            base_channels: if self.base_channels == 0 { 64 } else { self.base_channels },
            num_stages: if self.num_stages == 0 { 4 } else { self.num_stages },
            global_pool: self.global_pool,
            dropout: self.dropout,
        };

        config.validate()?;
        Ok(config)
    }
}

/// Configuration for contrastive learning projectors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectorConfig {
    /// Input feature dimension
    pub in_features: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Output embedding dimension
    pub out_features: usize,
    /// Number of MLP layers
    pub num_layers: usize,
    /// Whether to use batch normalization
    pub use_bn: bool,
}

impl ProjectorConfig {
    /// Creates a new builder for ProjectorConfig.
    pub fn builder() -> ProjectorConfigBuilder {
        ProjectorConfigBuilder::default()
    }

    /// Validates the configuration.
    pub fn validate(&self) -> CnnResult<()> {
        if self.in_features == 0 {
            return Err(CnnError::InvalidConfig(
                "in_features must be greater than 0".to_string(),
            ));
        }

        if self.hidden_dim == 0 {
            return Err(CnnError::InvalidConfig(
                "hidden_dim must be greater than 0".to_string(),
            ));
        }

        if self.out_features == 0 {
            return Err(CnnError::InvalidConfig(
                "out_features must be greater than 0".to_string(),
            ));
        }

        if self.num_layers == 0 {
            return Err(CnnError::InvalidConfig(
                "num_layers must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for ProjectorConfig.
#[derive(Default)]
pub struct ProjectorConfigBuilder {
    in_features: Option<usize>,
    hidden_dim: usize,
    out_features: usize,
    num_layers: usize,
    use_bn: bool,
}

impl ProjectorConfigBuilder {
    /// Sets the input features.
    pub fn in_features(mut self, in_features: usize) -> Self {
        self.in_features = Some(in_features);
        self
    }

    /// Sets the hidden dimension.
    pub fn hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = hidden_dim;
        self
    }

    /// Sets the output features.
    pub fn out_features(mut self, out_features: usize) -> Self {
        self.out_features = out_features;
        self
    }

    /// Sets the number of layers.
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Sets whether to use batch normalization.
    pub fn use_bn(mut self, use_bn: bool) -> Self {
        self.use_bn = use_bn;
        self
    }

    /// Builds the ProjectorConfig.
    pub fn build(self) -> CnnResult<ProjectorConfig> {
        let in_features = self.in_features.ok_or_else(|| {
            CnnError::InvalidConfig("in_features must be specified".to_string())
        })?;

        let config = ProjectorConfig {
            in_features,
            hidden_dim: if self.hidden_dim == 0 { in_features * 2 } else { self.hidden_dim },
            out_features: if self.out_features == 0 { 128 } else { self.out_features },
            num_layers: if self.num_layers == 0 { 2 } else { self.num_layers },
            use_bn: self.use_bn,
        };

        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_config_builder() {
        let config = ConvConfig::builder()
            .in_channels(3)
            .out_channels(64)
            .kernel_size(3)
            .stride(1)
            .padding(1)
            .build()
            .unwrap();

        assert_eq!(config.in_channels, 3);
        assert_eq!(config.out_channels, 64);
        assert_eq!(config.kernel_size, 3);
        assert_eq!(config.stride, 1);
        assert_eq!(config.padding, 1);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.groups, 1);
    }

    #[test]
    fn test_conv_output_size() {
        let config = ConvConfig::builder()
            .in_channels(3)
            .out_channels(64)
            .kernel_size(3)
            .stride(1)
            .padding(1)
            .build()
            .unwrap();

        // Same padding: output size = input size
        assert_eq!(config.output_size(224), 224);
    }

    #[test]
    fn test_conv_validation_grouped() {
        // Valid grouped convolution
        let result = ConvConfig::builder()
            .in_channels(64)
            .out_channels(128)
            .kernel_size(3)
            .groups(32)
            .build();
        assert!(result.is_ok());

        // Invalid: in_channels not divisible by groups
        let result = ConvConfig::builder()
            .in_channels(64)
            .out_channels(128)
            .kernel_size(3)
            .groups(3)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_config_builder() {
        let config = PoolConfig::builder()
            .kernel_size(2)
            .stride(2)
            .build()
            .unwrap();

        assert_eq!(config.kernel_size, 2);
        assert_eq!(config.stride, 2);
        assert_eq!(config.padding, 0);
    }

    #[test]
    fn test_pool_output_size() {
        let config = PoolConfig::builder()
            .kernel_size(2)
            .stride(2)
            .build()
            .unwrap();

        assert_eq!(config.output_size(224), 112);
    }

    #[test]
    fn test_norm_config_builder() {
        let config = NormConfig::builder()
            .num_features(64)
            .eps(1e-5)
            .momentum(0.1)
            .affine(true)
            .track_running_stats(true)
            .build()
            .unwrap();

        assert_eq!(config.num_features, 64);
        assert_eq!(config.eps, 1e-5);
        assert!(config.affine);
    }

    #[test]
    fn test_backbone_config_builder() {
        let config = BackboneConfig::builder()
            .in_channels(3)
            .out_features(512)
            .base_channels(64)
            .num_stages(4)
            .global_pool(true)
            .build()
            .unwrap();

        assert_eq!(config.in_channels, 3);
        assert_eq!(config.out_features, 512);
        assert_eq!(config.num_stages, 4);
        assert!(config.global_pool);
    }

    #[test]
    fn test_projector_config_builder() {
        let config = ProjectorConfig::builder()
            .in_features(512)
            .hidden_dim(1024)
            .out_features(128)
            .num_layers(2)
            .use_bn(true)
            .build()
            .unwrap();

        assert_eq!(config.in_features, 512);
        assert_eq!(config.hidden_dim, 1024);
        assert_eq!(config.out_features, 128);
        assert!(config.use_bn);
    }
}
