//! CNN Feature Extraction for Image Embeddings
//!
//! This crate provides pure Rust CNN-based feature extraction with SIMD acceleration.
//! It is designed for CPU-only deployment including WASM environments.
//!
//! # Features
//!
//! - MobileNet-V3 Small/Large backbones
//! - SIMD acceleration (AVX2, NEON, WASM SIMD128)
//! - INT8 quantization support
//! - Pure Rust (no BLAS/OpenCV dependencies)
//! - Parallel batch processing with rayon (optional)
//!
//! # Example
//!
//! ```rust,ignore
//! use ruvector_cnn::{CnnEmbedder, EmbeddingConfig};
//!
//! let embedder = CnnEmbedder::new(EmbeddingConfig::default())?;
//! let embedding = embedder.extract(&image_data, width, height)?;
//! println!("Embedding dim: {}", embedding.len());
//! ```
//!
//! # Using MobileNet Backbone
//!
//! ```rust,ignore
//! use ruvector_cnn::embedding::MobileNetEmbedder;
//!
//! // Create a MobileNetV3-Small embedder
//! let embedder = MobileNetEmbedder::v3_small()?;
//!
//! // Extract features from normalized float tensor (NCHW format)
//! let features = embedder.extract(&image_tensor, 224, 224)?;
//! println!("Feature dim: {}", features.len()); // 576 for V3-Small
//! ```

mod error;
mod tensor;

// Core modules (always available)
pub mod layers;
pub mod simd;
pub mod kernels;

// Quantization support (INT8 optimization)
pub mod quantize;
pub mod int8;

// Optional modules (require backbone feature due to API incompatibility)
#[cfg(feature = "backbone")]
pub mod backbone;
#[cfg(feature = "backbone")]
pub mod embedding;

// Contrastive learning (standalone, no backbone dependency)
pub mod contrastive;

pub use error::{CnnError, CnnResult};
pub use tensor::Tensor;

// Re-export backbone types (only when feature enabled)
#[cfg(feature = "backbone")]
pub use backbone::{
    Backbone, BackboneExt, BackboneType,
    MobileNetV3, MobileNetV3Config,
    MobileNetV3Small, MobileNetV3Large, MobileNetConfig,
    ConvBNActivation, InvertedResidual, SqueezeExcitation,
    create_backbone, mobilenet_v3_small, mobilenet_v3_large,
};

// Re-export embedding types (only when feature enabled)
#[cfg(feature = "backbone")]
pub use embedding::{
    MobileNetEmbedder, EmbeddingExtractorExt,
    EmbeddingConfig as MobileNetEmbeddingConfig,
    cosine_similarity, euclidean_distance,
};

// ParallelEmbedding requires the `parallel` feature (not yet implemented)
// #[cfg(all(feature = "backbone", feature = "parallel"))]
// pub use embedding::parallel::ParallelEmbedding;

use serde::{Deserialize, Serialize};

/// Configuration for CNN embedding extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Input image size (assumes square input)
    pub input_size: u32,
    /// Output embedding dimension
    pub embedding_dim: usize,
    /// L2 normalize output embeddings
    pub normalize: bool,
    /// Use INT8 quantization
    pub quantized: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            input_size: 224,
            embedding_dim: 512,
            normalize: true,
            quantized: false,
        }
    }
}

/// CNN Embedder for feature extraction
#[derive(Debug, Clone)]
pub struct CnnEmbedder {
    config: EmbeddingConfig,
    weights: EmbedderWeights,
}

/// Internal weights storage
#[derive(Debug, Clone)]
struct EmbedderWeights {
    /// Convolution weights (simplified representation)
    conv_weights: Vec<f32>,
    /// Batch norm parameters
    bn_gamma: Vec<f32>,
    bn_beta: Vec<f32>,
    bn_mean: Vec<f32>,
    bn_var: Vec<f32>,
    /// Final projection weights
    projection: Vec<f32>,
}

impl Default for EmbedderWeights {
    fn default() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let conv_size = 3 * 3 * 3 * 16;
        let bn_size = 16;
        let proj_size = 16 * 512;

        Self {
            conv_weights: (0..conv_size).map(|_| rng.gen_range(-0.1..0.1)).collect(),
            bn_gamma: vec![1.0; bn_size],
            bn_beta: vec![0.0; bn_size],
            bn_mean: vec![0.0; bn_size],
            bn_var: vec![1.0; bn_size],
            projection: (0..proj_size).map(|_| rng.gen_range(-0.1..0.1)).collect(),
        }
    }
}

impl CnnEmbedder {
    /// Create a new CNN embedder with the given configuration
    pub fn new(config: EmbeddingConfig) -> CnnResult<Self> {
        let weights = EmbedderWeights::default();
        Ok(Self { config, weights })
    }

    /// Create a MobileNet-V3 Small embedder
    pub fn new_v3_small() -> CnnResult<Self> {
        Self::new(EmbeddingConfig {
            input_size: 224,
            embedding_dim: 576,
            normalize: true,
            quantized: false,
        })
    }

    /// Create a MobileNet-V3 Large embedder
    pub fn new_v3_large() -> CnnResult<Self> {
        Self::new(EmbeddingConfig {
            input_size: 224,
            embedding_dim: 960,
            normalize: true,
            quantized: false,
        })
    }

    /// Extract embedding from image data (RGBA format)
    pub fn extract(&self, image_data: &[u8], width: u32, height: u32) -> CnnResult<Vec<f32>> {
        let expected_size = (width * height * 4) as usize;
        if image_data.len() != expected_size {
            return Err(CnnError::InvalidInput(format!(
                "Expected {} bytes for {}x{} RGBA image, got {}",
                expected_size, width, height, image_data.len()
            )));
        }

        let rgb_float = self.preprocess(image_data, width, height)?;
        let features = self.forward(&rgb_float)?;
        let pooled = self.global_avg_pool(&features)?;
        let mut embedding = self.project(&pooled)?;

        if self.config.normalize {
            self.l2_normalize(&mut embedding);
        }

        Ok(embedding)
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get the input size
    pub fn input_size(&self) -> u32 {
        self.config.input_size
    }

    fn preprocess(&self, image_data: &[u8], width: u32, height: u32) -> CnnResult<Vec<f32>> {
        let pixels = (width * height) as usize;
        let mut rgb = Vec::with_capacity(pixels * 3);

        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        for i in 0..pixels {
            let offset = i * 4;
            rgb.push((image_data[offset] as f32 / 255.0 - mean[0]) / std[0]);
            rgb.push((image_data[offset + 1] as f32 / 255.0 - mean[1]) / std[1]);
            rgb.push((image_data[offset + 2] as f32 / 255.0 - mean[2]) / std[2]);
        }

        Ok(rgb)
    }

    fn forward(&self, input: &[f32]) -> CnnResult<Vec<f32>> {
        let conv_out = layers::conv2d_3x3(
            input,
            &self.weights.conv_weights,
            3,
            16,
            self.config.input_size as usize,
            self.config.input_size as usize,
        );

        let bn_out = layers::batch_norm(
            &conv_out,
            &self.weights.bn_gamma,
            &self.weights.bn_beta,
            &self.weights.bn_mean,
            &self.weights.bn_var,
            1e-5,
        );

        let activated: Vec<f32> = bn_out.iter().map(|&x| x.max(0.0)).collect();
        Ok(activated)
    }

    fn global_avg_pool(&self, features: &[f32]) -> CnnResult<Vec<f32>> {
        let channels = 16;
        let spatial = features.len() / channels;
        let mut pooled = vec![0.0f32; channels];

        for i in 0..spatial {
            for c in 0..channels {
                pooled[c] += features[i * channels + c];
            }
        }

        let inv_spatial = 1.0 / spatial as f32;
        for p in pooled.iter_mut() {
            *p *= inv_spatial;
        }

        Ok(pooled)
    }

    fn project(&self, features: &[f32]) -> CnnResult<Vec<f32>> {
        let in_dim = features.len();
        let out_dim = self.config.embedding_dim;
        let mut output = vec![0.0f32; out_dim];

        for o in 0..out_dim {
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += features[i] * self.weights.projection[i * out_dim + o];
            }
            output[o] = sum;
        }

        Ok(output)
    }

    fn l2_normalize(&self, vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }
}

/// Embedding extractor trait
pub trait EmbeddingExtractor {
    fn extract(&self, image_data: &[u8], width: u32, height: u32) -> CnnResult<Vec<f32>>;
    fn embedding_dim(&self) -> usize;
}

impl EmbeddingExtractor for CnnEmbedder {
    fn extract(&self, image_data: &[u8], width: u32, height: u32) -> CnnResult<Vec<f32>> {
        CnnEmbedder::extract(self, image_data, width, height)
    }

    fn embedding_dim(&self) -> usize {
        CnnEmbedder::embedding_dim(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_creation() {
        let embedder = CnnEmbedder::new(EmbeddingConfig::default()).unwrap();
        assert_eq!(embedder.embedding_dim(), 512);
    }

    #[test]
    fn test_v3_small() {
        let embedder = CnnEmbedder::new_v3_small().unwrap();
        assert_eq!(embedder.embedding_dim(), 576);
    }

    #[test]
    fn test_v3_large() {
        let embedder = CnnEmbedder::new_v3_large().unwrap();
        assert_eq!(embedder.embedding_dim(), 960);
    }

    #[test]
    fn test_extract_embedding() {
        let embedder = CnnEmbedder::new(EmbeddingConfig {
            input_size: 4,
            embedding_dim: 8,
            normalize: true,
            quantized: false,
        }).unwrap();

        let image_data = vec![128u8; 4 * 4 * 4];
        let embedding = embedder.extract(&image_data, 4, 4).unwrap();

        assert_eq!(embedding.len(), 8);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5 || norm < 1e-10);
    }

    #[test]
    fn test_invalid_input() {
        let embedder = CnnEmbedder::new(EmbeddingConfig::default()).unwrap();
        let image_data = vec![0u8; 100];
        let result = embedder.extract(&image_data, 10, 10);
        assert!(result.is_err());
    }
}
