//! CNN Backbone implementations.
//!
//! This module provides various backbone architectures for image feature extraction:
//! - MobileNet-V3 (Small and Large variants)
//!
//! Backbones are typically used as feature extractors in tasks like:
//! - Image classification
//! - Object detection
//! - Image embedding/retrieval
//! - Contrastive learning

mod blocks;
mod layer;
mod mobilenet;

pub use layer::Layer;

pub use blocks::{ConvBNActivation, InvertedResidual, InvertedResidualConfig, SqueezeExcitation};
pub use mobilenet::{MobileNetV3, MobileNetV3Config};

// Keep backward compatibility with old API
pub use mobilenet::{MobileNetConfig, MobileNetV3Large, MobileNetV3Small};

use crate::error::CnnResult;
use crate::layers::TensorShape;

/// Types of supported backbone architectures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BackboneType {
    /// MobileNet-V3 Small (576 output channels, ~2.5M params)
    MobileNetV3Small,
    /// MobileNet-V3 Large (960 output channels, ~5.4M params)
    MobileNetV3Large,
}

impl BackboneType {
    /// Returns the output feature dimension for this backbone.
    pub fn output_dim(&self) -> usize {
        match self {
            BackboneType::MobileNetV3Small => 576,
            BackboneType::MobileNetV3Large => 960,
        }
    }

    /// Returns a human-readable name for this backbone.
    pub fn name(&self) -> &'static str {
        match self {
            BackboneType::MobileNetV3Small => "MobileNetV3-Small",
            BackboneType::MobileNetV3Large => "MobileNetV3-Large",
        }
    }

    /// Returns the expected input image size (height, width).
    pub fn input_size(&self) -> (usize, usize) {
        (224, 224)
    }

    /// Returns the expected number of input channels.
    pub fn input_channels(&self) -> usize {
        3
    }
}

impl std::fmt::Display for BackboneType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for backbone networks that extract features from images.
pub trait Backbone: Send + Sync {
    /// Forward pass returning feature maps.
    ///
    /// # Arguments
    /// * `input` - Input tensor in NCHW format, flattened
    /// * `height` - Input height
    /// * `width` - Input width
    ///
    /// # Returns
    /// Feature vector
    fn forward(&self, input: &[f32], height: usize, width: usize) -> Vec<f32>;

    /// Returns the output feature dimension.
    fn output_dim(&self) -> usize;

    /// Returns the expected input size.
    fn input_size(&self) -> usize;
}

/// Extended backbone trait with TensorShape support.
pub trait BackboneExt: Backbone {
    /// Returns the backbone type.
    fn backbone_type(&self) -> BackboneType;

    /// Returns the number of trainable parameters.
    fn num_params(&self) -> usize;

    /// Forward pass through the entire backbone including classifier.
    ///
    /// # Arguments
    /// * `input` - Input tensor in NCHW format, flattened
    /// * `input_shape` - Shape of the input tensor
    ///
    /// # Returns
    /// Output logits of shape [batch, num_classes]
    fn forward_with_shape(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>>;

    /// Forward pass through feature extraction layers only (no classifier).
    ///
    /// # Arguments
    /// * `input` - Input tensor in NCHW format, flattened
    /// * `input_shape` - Shape of the input tensor
    ///
    /// # Returns
    /// Feature tensor of shape [batch, output_dim]
    fn forward_features(&self, input: &[f32], input_shape: &TensorShape) -> CnnResult<Vec<f32>>;

    /// Returns the output shape for the feature extractor.
    fn feature_output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        TensorShape {
            n: input_shape.n,
            c: self.output_dim(),
            h: 1,
            w: 1,
        }
    }
}

/// Creates a backbone network of the specified type.
///
/// # Arguments
/// * `backbone_type` - Type of backbone to create
/// * `num_classes` - Number of output classes for classifier (use 0 for feature extraction only)
///
/// # Returns
/// A boxed backbone implementation
pub fn create_backbone(
    backbone_type: BackboneType,
    num_classes: usize,
) -> CnnResult<Box<dyn BackboneExt>> {
    match backbone_type {
        BackboneType::MobileNetV3Small => {
            let config = MobileNetV3Config::small(num_classes);
            Ok(Box::new(MobileNetV3::new(config)?))
        }
        BackboneType::MobileNetV3Large => {
            let config = MobileNetV3Config::large(num_classes);
            Ok(Box::new(MobileNetV3::new(config)?))
        }
    }
}

/// Creates a MobileNetV3-Small backbone.
pub fn mobilenet_v3_small(num_classes: usize) -> CnnResult<MobileNetV3> {
    let config = MobileNetV3Config::small(num_classes);
    MobileNetV3::new(config)
}

/// Creates a MobileNetV3-Large backbone.
pub fn mobilenet_v3_large(num_classes: usize) -> CnnResult<MobileNetV3> {
    let config = MobileNetV3Config::large(num_classes);
    MobileNetV3::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backbone_type_output_dim() {
        assert_eq!(BackboneType::MobileNetV3Small.output_dim(), 576);
        assert_eq!(BackboneType::MobileNetV3Large.output_dim(), 960);
    }

    #[test]
    fn test_backbone_type_name() {
        assert_eq!(BackboneType::MobileNetV3Small.name(), "MobileNetV3-Small");
        assert_eq!(BackboneType::MobileNetV3Large.name(), "MobileNetV3-Large");
    }

    #[test]
    fn test_create_backbone_small() {
        let backbone = create_backbone(BackboneType::MobileNetV3Small, 1000).unwrap();
        assert_eq!(backbone.backbone_type(), BackboneType::MobileNetV3Small);
        assert_eq!(backbone.output_dim(), 576);
    }

    #[test]
    fn test_create_backbone_large() {
        let backbone = create_backbone(BackboneType::MobileNetV3Large, 1000).unwrap();
        assert_eq!(backbone.backbone_type(), BackboneType::MobileNetV3Large);
        assert_eq!(backbone.output_dim(), 960);
    }

    #[test]
    fn test_backward_compat_small() {
        let config = MobileNetConfig::default();
        let model = MobileNetV3Small::new(config);
        assert_eq!(model.output_dim(), 576);
    }

    #[test]
    fn test_backward_compat_large() {
        let config = MobileNetConfig {
            output_channels: 960,
            ..Default::default()
        };
        let model = MobileNetV3Large::new(config);
        assert_eq!(model.output_dim(), 960);
    }
}
