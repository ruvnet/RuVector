//! Model configuration for the Geometric Context Transformer.

use serde::{Deserialize, Serialize};

/// Hyperparameters of the LingBot-Map Geometric Context Transformer.
///
/// Field defaults follow the upstream demo configuration (518x378 input,
/// streaming inference). Load real values from the checkpoint's `config.json`
/// via [`ModelConfig::from_file`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hidden / model dimension.
    #[serde(default = "default_hidden")]
    pub hidden_dim: usize,
    /// Number of transformer layers.
    #[serde(default = "default_layers")]
    pub num_layers: usize,
    /// Number of attention heads.
    #[serde(default = "default_heads")]
    pub num_heads: usize,
    /// Feed-forward inner dimension.
    #[serde(default = "default_ffn")]
    pub ffn_dim: usize,
    /// Input image height in pixels.
    #[serde(default = "default_height")]
    pub image_height: usize,
    /// Input image width in pixels.
    #[serde(default = "default_width")]
    pub image_width: usize,
    /// Patch size for tokenization.
    #[serde(default = "default_patch")]
    pub patch_size: usize,
    /// Dimensionality of the keyframe feature exported to the memory layer.
    #[serde(default = "default_feature_dim")]
    pub keyframe_feature_dim: usize,
    /// Top-K anchors retrieved per frame for drift correction.
    #[serde(default = "default_topk")]
    pub anchor_top_k: usize,
}

fn default_hidden() -> usize {
    1024
}
fn default_layers() -> usize {
    24
}
fn default_heads() -> usize {
    16
}
fn default_ffn() -> usize {
    4096
}
fn default_height() -> usize {
    378
}
fn default_width() -> usize {
    518
}
fn default_patch() -> usize {
    14
}
fn default_feature_dim() -> usize {
    512
}
fn default_topk() -> usize {
    16
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: default_hidden(),
            num_layers: default_layers(),
            num_heads: default_heads(),
            ffn_dim: default_ffn(),
            image_height: default_height(),
            image_width: default_width(),
            patch_size: default_patch(),
            keyframe_feature_dim: default_feature_dim(),
            anchor_top_k: default_topk(),
        }
    }
}

impl ModelConfig {
    /// Load from a `config.json`-style file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, super::TensorError> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Number of patch tokens per frame.
    pub fn num_patches(&self) -> usize {
        (self.image_height / self.patch_size) * (self.image_width / self.patch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = ModelConfig::default();
        assert!(c.hidden_dim > 0 && c.num_heads > 0);
        assert_eq!(c.hidden_dim % c.num_heads, 0);
    }

    #[test]
    fn patch_count() {
        let c = ModelConfig::default();
        assert_eq!(c.num_patches(), (378 / 14) * (518 / 14));
    }

    #[test]
    fn partial_json_uses_defaults() {
        let c: ModelConfig = serde_json::from_str(r#"{"hidden_dim": 768}"#).unwrap();
        assert_eq!(c.hidden_dim, 768);
        assert_eq!(c.num_layers, default_layers());
    }
}
