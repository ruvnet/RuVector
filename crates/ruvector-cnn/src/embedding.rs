//! Embedding extraction for image retrieval and contrastive learning.
//!
//! This module provides high-level APIs for extracting embeddings from images
//! using CNN backbones like MobileNet-V3.
//!
//! ## Features
//!
//! - Single image embedding extraction
//! - Batch processing with optional parallelism (via rayon)
//! - L2 normalization for similarity search
//! - Support for different backbone architectures
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvector_cnn::embedding::{MobileNetEmbedder, EmbeddingExtractor};
//!
//! let embedder = MobileNetEmbedder::v3_small()?;
//! let embedding = embedder.extract(&image_tensor)?;
//! println!("Embedding dim: {}", embedding.len());
//! ```

use crate::backbone::{Backbone, BackboneExt, BackboneType, MobileNetV3, MobileNetV3Config};
use crate::error::{CnnError, CnnResult};
use crate::layers::TensorShape;

/// Trait for embedding extractors.
///
/// Any type that can extract embeddings from images implements this trait.
pub trait EmbeddingExtractorExt: Send + Sync {
    /// Extracts an embedding from a single image.
    ///
    /// # Arguments
    /// * `image` - Image tensor in NCHW format, flattened as [C * H * W]
    /// * `height` - Image height
    /// * `width` - Image width
    ///
    /// # Returns
    /// Embedding vector
    fn extract(&self, image: &[f32], height: usize, width: usize) -> CnnResult<Vec<f32>>;

    /// Extracts an embedding from a single image with TensorShape.
    fn extract_with_shape(&self, image: &[f32], shape: &TensorShape) -> CnnResult<Vec<f32>>;

    /// Extracts embeddings from a batch of images.
    ///
    /// # Arguments
    /// * `images` - Batch of image tensors in NCHW format
    /// * `batch_size` - Number of images in the batch
    /// * `height` - Image height
    /// * `width` - Image width
    ///
    /// # Returns
    /// Vector of embeddings, one per image
    fn extract_batch(
        &self,
        images: &[f32],
        batch_size: usize,
        height: usize,
        width: usize,
    ) -> CnnResult<Vec<Vec<f32>>>;

    /// Returns the embedding dimension.
    fn embedding_dim(&self) -> usize;

    /// Returns whether embeddings are L2 normalized.
    fn is_normalized(&self) -> bool;
}

/// Configuration for embedding extraction.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmbeddingConfig {
    /// Backbone type to use
    pub backbone_type: BackboneType,
    /// L2 normalize output embeddings
    pub normalize: bool,
    /// Expected input image size
    pub input_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backbone_type: BackboneType::MobileNetV3Small,
            normalize: true,
            input_size: 224,
        }
    }
}

impl EmbeddingConfig {
    /// Creates config for MobileNetV3-Small.
    pub fn mobilenet_v3_small() -> Self {
        Self {
            backbone_type: BackboneType::MobileNetV3Small,
            normalize: true,
            input_size: 224,
        }
    }

    /// Creates config for MobileNetV3-Large.
    pub fn mobilenet_v3_large() -> Self {
        Self {
            backbone_type: BackboneType::MobileNetV3Large,
            normalize: true,
            input_size: 224,
        }
    }

    /// Sets whether to normalize embeddings.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// MobileNet-based embedding extractor.
///
/// Extracts embeddings from images using MobileNet-V3 backbone.
#[derive(Clone, Debug)]
pub struct MobileNetEmbedder {
    /// The backbone network
    backbone: MobileNetV3,
    /// Whether to L2 normalize embeddings
    normalize: bool,
    /// Input image size
    input_size: usize,
}

impl MobileNetEmbedder {
    /// Creates a new MobileNetEmbedder with the given configuration.
    pub fn new(config: EmbeddingConfig) -> CnnResult<Self> {
        let backbone_config = match config.backbone_type {
            BackboneType::MobileNetV3Small => MobileNetV3Config::small(0), // No classifier
            BackboneType::MobileNetV3Large => MobileNetV3Config::large(0),
        };

        let backbone = MobileNetV3::new(backbone_config)?;

        Ok(Self {
            backbone,
            normalize: config.normalize,
            input_size: config.input_size,
        })
    }

    /// Creates a MobileNetV3-Small embedder.
    pub fn v3_small() -> CnnResult<Self> {
        Self::new(EmbeddingConfig::mobilenet_v3_small())
    }

    /// Creates a MobileNetV3-Large embedder.
    pub fn v3_large() -> CnnResult<Self> {
        Self::new(EmbeddingConfig::mobilenet_v3_large())
    }

    /// Creates an embedder without normalization.
    pub fn without_normalization(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Returns a reference to the backbone.
    pub fn backbone(&self) -> &MobileNetV3 {
        &self.backbone
    }

    /// Returns the expected input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// L2 normalizes a vector in place.
    fn l2_normalize_inplace(&self, vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// L2 normalizes a vector, returning a new vector.
    fn l2_normalize(&self, vec: &[f32]) -> Vec<f32> {
        let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            vec.iter().map(|&x| x / norm).collect()
        } else {
            vec.to_vec()
        }
    }
}

impl EmbeddingExtractorExt for MobileNetEmbedder {
    fn extract(&self, image: &[f32], height: usize, width: usize) -> CnnResult<Vec<f32>> {
        let shape = TensorShape::new(1, 3, height, width);
        self.extract_with_shape(image, &shape)
    }

    fn extract_with_shape(&self, image: &[f32], shape: &TensorShape) -> CnnResult<Vec<f32>> {
        if image.len() != shape.numel() {
            return Err(CnnError::DimensionMismatch(format!(
                "Image has {} elements, expected {} for shape {}",
                image.len(),
                shape.numel(),
                shape
            )));
        }

        let mut embedding = self.backbone.forward_features(image, shape)?;

        if self.normalize {
            self.l2_normalize_inplace(&mut embedding);
        }

        Ok(embedding)
    }

    fn extract_batch(
        &self,
        images: &[f32],
        batch_size: usize,
        height: usize,
        width: usize,
    ) -> CnnResult<Vec<Vec<f32>>> {
        let image_size = 3 * height * width;

        if images.len() != batch_size * image_size {
            return Err(CnnError::DimensionMismatch(format!(
                "Images have {} elements, expected {} for batch of {} images",
                images.len(),
                batch_size * image_size,
                batch_size
            )));
        }

        // Extract embeddings for each image
        let embeddings: CnnResult<Vec<Vec<f32>>> = (0..batch_size)
            .map(|i| {
                let start = i * image_size;
                let end = start + image_size;
                let image = &images[start..end];
                self.extract(image, height, width)
            })
            .collect();

        embeddings
    }

    fn embedding_dim(&self) -> usize {
        self.backbone.output_dim()
    }

    fn is_normalized(&self) -> bool {
        self.normalize
    }
}

/// Parallel batch extraction using rayon.
#[cfg(feature = "parallel")]
pub mod parallel {
    use super::*;
    use rayon::prelude::*;

    /// Extension trait for parallel embedding extraction.
    pub trait ParallelEmbedding: EmbeddingExtractorExt {
        /// Extracts embeddings from a batch of images in parallel.
        ///
        /// Uses rayon for parallel processing across CPU cores.
        fn extract_batch_parallel(
            &self,
            images: &[f32],
            batch_size: usize,
            height: usize,
            width: usize,
        ) -> CnnResult<Vec<Vec<f32>>>;

        /// Extracts embeddings from a list of image tensors in parallel.
        fn extract_many_parallel(
            &self,
            images: &[&[f32]],
            height: usize,
            width: usize,
        ) -> CnnResult<Vec<Vec<f32>>>;
    }

    impl<T: EmbeddingExtractorExt + Sync> ParallelEmbedding for T {
        fn extract_batch_parallel(
            &self,
            images: &[f32],
            batch_size: usize,
            height: usize,
            width: usize,
        ) -> CnnResult<Vec<Vec<f32>>> {
            let image_size = 3 * height * width;

            if images.len() != batch_size * image_size {
                return Err(CnnError::DimensionMismatch(format!(
                    "Images have {} elements, expected {} for batch of {} images",
                    images.len(),
                    batch_size * image_size,
                    batch_size
                )));
            }

            let results: Vec<CnnResult<Vec<f32>>> = (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let start = i * image_size;
                    let end = start + image_size;
                    let image = &images[start..end];
                    self.extract(image, height, width)
                })
                .collect();

            // Collect results, propagating any errors
            results.into_iter().collect()
        }

        fn extract_many_parallel(
            &self,
            images: &[&[f32]],
            height: usize,
            width: usize,
        ) -> CnnResult<Vec<Vec<f32>>> {
            let results: Vec<CnnResult<Vec<f32>>> = images
                .par_iter()
                .map(|image| self.extract(image, height, width))
                .collect();

            results.into_iter().collect()
        }
    }
}

/// Computes cosine similarity between two embeddings.
///
/// Assumes embeddings are L2 normalized (then cosine = dot product).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Computes Euclidean distance between two embeddings.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_creation() {
        let embedder = MobileNetEmbedder::v3_small().unwrap();
        assert_eq!(embedder.embedding_dim(), 576);
        assert!(embedder.is_normalized());
    }

    #[test]
    fn test_embedder_v3_large() {
        let embedder = MobileNetEmbedder::v3_large().unwrap();
        assert_eq!(embedder.embedding_dim(), 960);
    }

    #[test]
    fn test_embedder_config() {
        let config = EmbeddingConfig::mobilenet_v3_small().normalize(false);
        let embedder = MobileNetEmbedder::new(config).unwrap();
        assert!(!embedder.is_normalized());
    }

    #[test]
    fn test_extract_embedding() {
        let embedder = MobileNetEmbedder::v3_small().unwrap();
        let image = vec![0.5f32; 3 * 224 * 224];

        let embedding = embedder.extract(&image, 224, 224).unwrap();

        assert_eq!(embedding.len(), 576);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5 || norm < 1e-10);
    }

    #[test]
    fn test_extract_batch() {
        let embedder = MobileNetEmbedder::v3_small().unwrap();
        let batch_size = 2;
        let images = vec![0.5f32; batch_size * 3 * 224 * 224];

        let embeddings = embedder
            .extract_batch(&images, batch_size, 224, 224)
            .unwrap();

        assert_eq!(embeddings.len(), batch_size);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), 576);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_without_normalization() {
        let embedder = MobileNetEmbedder::v3_small()
            .unwrap()
            .without_normalization();
        assert!(!embedder.is_normalized());
    }
}
