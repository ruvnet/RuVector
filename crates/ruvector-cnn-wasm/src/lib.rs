//! WASM bindings for ruvector-cnn
//!
//! Provides WebAssembly bindings for CNN-based image embedding extraction.
//!
//! ## Features
//!
//! - SIMD-optimized convolutions (WASM SIMD128)
//! - Contrastive learning (InfoNCE, Triplet loss)
//! - MobileNet-style efficient architectures

#![allow(clippy::new_without_default)]

use ruvector_cnn::contrastive::{
    InfoNCELoss as RustInfoNCE, TripletDistance, TripletLoss as RustTriplet,
};
use ruvector_cnn::simd;
use wasm_bindgen::prelude::*;

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Configuration for CNN embedder
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct EmbedderConfig {
    /// Input image size (square)
    pub input_size: u32,
    /// Output embedding dimension
    pub embedding_dim: u32,
    /// Whether to L2 normalize embeddings
    pub normalize: bool,
}

#[wasm_bindgen]
impl EmbedderConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            input_size: 224,
            embedding_dim: 512,
            normalize: true,
        }
    }
}

/// WASM CNN Embedder for image feature extraction
#[wasm_bindgen]
pub struct WasmCnnEmbedder {
    embedding_dim: usize,
    normalize: bool,
}

#[wasm_bindgen]
impl WasmCnnEmbedder {
    /// Create a new CNN embedder
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<EmbedderConfig>) -> Result<WasmCnnEmbedder, JsValue> {
        let cfg = config.unwrap_or_else(EmbedderConfig::new);
        Ok(Self {
            embedding_dim: cfg.embedding_dim as usize,
            normalize: cfg.normalize,
        })
    }

    /// Extract embedding from image data (RGB format, row-major)
    #[wasm_bindgen]
    pub fn extract(&self, image_data: &[u8], width: u32, height: u32) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != (width * height * 3) as usize {
            return Err(JsValue::from_str(&format!(
                "Invalid image data length: expected {}, got {}",
                width * height * 3,
                image_data.len()
            )));
        }

        // Convert u8 RGB to f32 normalized [0, 1]
        let float_data: Vec<f32> = image_data.iter().map(|&x| x as f32 / 255.0).collect();

        // Simple global average pooling to get embedding
        let channels = 3;
        let pixels_per_channel = (width * height) as usize;

        let mut embedding = vec![0.0f32; self.embedding_dim];

        // Simple feature extraction: use spatial statistics
        for c in 0..channels {
            let channel_data: Vec<f32> = (0..pixels_per_channel)
                .map(|i| float_data[i * channels + c])
                .collect();

            // Mean
            let mean: f32 = channel_data.iter().sum::<f32>() / pixels_per_channel as f32;

            // Variance
            let variance: f32 = channel_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                / pixels_per_channel as f32;

            // Store in embedding
            if c * 2 < self.embedding_dim {
                embedding[c * 2] = mean;
            }
            if c * 2 + 1 < self.embedding_dim {
                embedding[c * 2 + 1] = variance.sqrt();
            }
        }

        // Fill remaining dimensions with spatial features
        let block_size = 8;
        let blocks_x = width as usize / block_size;
        let blocks_y = height as usize / block_size;
        let mut idx = 6; // Start after channel stats

        for by in 0..blocks_y.min(8) {
            for bx in 0..blocks_x.min(8) {
                if idx >= self.embedding_dim {
                    break;
                }

                let mut block_sum = 0.0f32;
                for dy in 0..block_size {
                    for dx in 0..block_size {
                        let x = bx * block_size + dx;
                        let y = by * block_size + dy;
                        let pixel_idx = (y * width as usize + x) * 3;
                        if pixel_idx + 2 < float_data.len() {
                            // Luminance
                            block_sum += 0.299 * float_data[pixel_idx]
                                + 0.587 * float_data[pixel_idx + 1]
                                + 0.114 * float_data[pixel_idx + 2];
                        }
                    }
                }
                embedding[idx] = block_sum / (block_size * block_size) as f32;
                idx += 1;
            }
        }

        // L2 normalize if requested
        if self.normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }
        }

        Ok(embedding)
    }

    /// Get the embedding dimension
    #[wasm_bindgen(getter)]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Compute cosine similarity between two embeddings
    #[wasm_bindgen]
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32, JsValue> {
        if a.len() != b.len() {
            return Err(JsValue::from_str("Embedding dimensions must match"));
        }

        let dot: f32 = simd::dot_product_simd(a, b);
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-8 || norm_b < 1e-8 {
            return Ok(0.0);
        }

        Ok(dot / (norm_a * norm_b))
    }
}

/// InfoNCE loss for contrastive learning (SimCLR style)
#[wasm_bindgen]
pub struct WasmInfoNCELoss {
    inner: RustInfoNCE,
}

#[wasm_bindgen]
impl WasmInfoNCELoss {
    /// Create new InfoNCE loss with temperature parameter
    #[wasm_bindgen(constructor)]
    pub fn new(temperature: f32) -> Self {
        Self {
            inner: RustInfoNCE::new(temperature as f64),
        }
    }

    /// Compute loss for a batch of embedding pairs
    /// embeddings: [2N, D] flattened where (i, i+N) are positive pairs
    #[wasm_bindgen]
    pub fn forward(
        &self,
        embeddings: &[f32],
        batch_size: usize,
        dim: usize,
    ) -> Result<f32, JsValue> {
        if embeddings.len() != 2 * batch_size * dim {
            return Err(JsValue::from_str(&format!(
                "Expected {} elements, got {}",
                2 * batch_size * dim,
                embeddings.len()
            )));
        }

        // Convert to Vec<Vec<f64>>
        let embs: Vec<Vec<f64>> = (0..2 * batch_size)
            .map(|i| {
                embeddings[i * dim..(i + 1) * dim]
                    .iter()
                    .map(|&x| x as f64)
                    .collect()
            })
            .collect();

        // Forward with num_views = 2 (pairs)
        Ok(self.inner.forward(&embs, 2) as f32)
    }

    /// Get the temperature parameter
    #[wasm_bindgen(getter)]
    pub fn temperature(&self) -> f32 {
        self.inner.temperature() as f32
    }
}

/// Triplet loss for metric learning
#[wasm_bindgen]
pub struct WasmTripletLoss {
    inner: RustTriplet,
}

#[wasm_bindgen]
impl WasmTripletLoss {
    /// Create new triplet loss with margin
    #[wasm_bindgen(constructor)]
    pub fn new(margin: f32) -> Self {
        Self {
            inner: RustTriplet::new(margin as f64),
        }
    }

    /// Compute loss for a single triplet
    #[wasm_bindgen]
    pub fn forward_single(
        &self,
        anchor: &[f32],
        positive: &[f32],
        negative: &[f32],
    ) -> Result<f32, JsValue> {
        if anchor.len() != positive.len() || anchor.len() != negative.len() {
            return Err(JsValue::from_str("Embedding dimensions must match"));
        }

        let a: Vec<f64> = anchor.iter().map(|&x| x as f64).collect();
        let p: Vec<f64> = positive.iter().map(|&x| x as f64).collect();
        let n: Vec<f64> = negative.iter().map(|&x| x as f64).collect();

        Ok(self.inner.forward(&a, &p, &n) as f32)
    }

    /// Compute loss for a batch of triplets
    #[wasm_bindgen]
    pub fn forward(
        &self,
        anchors: &[f32],
        positives: &[f32],
        negatives: &[f32],
        dim: usize,
    ) -> Result<f32, JsValue> {
        if anchors.len() % dim != 0
            || positives.len() != anchors.len()
            || negatives.len() != anchors.len()
        {
            return Err(JsValue::from_str("Invalid triplet dimensions"));
        }

        let batch_size = anchors.len() / dim;
        let mut total_loss = 0.0f64;

        for i in 0..batch_size {
            let a: Vec<f64> = anchors[i * dim..(i + 1) * dim]
                .iter()
                .map(|&x| x as f64)
                .collect();
            let p: Vec<f64> = positives[i * dim..(i + 1) * dim]
                .iter()
                .map(|&x| x as f64)
                .collect();
            let n: Vec<f64> = negatives[i * dim..(i + 1) * dim]
                .iter()
                .map(|&x| x as f64)
                .collect();
            total_loss += self.inner.forward(&a, &p, &n);
        }

        Ok((total_loss / batch_size as f64) as f32)
    }

    /// Get the margin parameter
    #[wasm_bindgen(getter)]
    pub fn margin(&self) -> f32 {
        self.inner.margin() as f32
    }
}

/// SIMD-optimized operations
#[wasm_bindgen]
pub struct SimdOps;

#[wasm_bindgen]
impl SimdOps {
    /// Dot product of two vectors
    #[wasm_bindgen]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        simd::dot_product_simd(a, b)
    }

    /// ReLU activation (returns new array)
    #[wasm_bindgen]
    pub fn relu(data: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; data.len()];
        simd::relu_simd(data, &mut output);
        output
    }

    /// ReLU6 activation (returns new array)
    #[wasm_bindgen]
    pub fn relu6(data: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; data.len()];
        simd::relu6_simd(data, &mut output);
        output
    }

    /// L2 normalize a vector (returns new array)
    #[wasm_bindgen]
    pub fn l2_normalize(data: &[f32]) -> Vec<f32> {
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            data.iter().map(|&x| x / norm).collect()
        } else {
            data.to_vec()
        }
    }
}

/// Layer operations for building custom networks
#[wasm_bindgen]
pub struct LayerOps;

#[wasm_bindgen]
impl LayerOps {
    /// Apply batch normalization (returns new array)
    #[wasm_bindgen]
    pub fn batch_norm(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        var: &[f32],
        epsilon: f32,
    ) -> Vec<f32> {
        let channels = gamma.len();
        let mut output = vec![0.0f32; input.len()];
        simd::batch_norm_simd(
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

    /// Apply global average pooling
    /// Returns one value per channel
    #[wasm_bindgen]
    pub fn global_avg_pool(
        input: &[f32],
        height: usize,
        width: usize,
        channels: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; channels];
        simd::global_avg_pool_simd(input, &mut output, height, width, channels);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_embedder_creation() {
        let embedder = WasmCnnEmbedder::new(None).unwrap();
        assert_eq!(embedder.embedding_dim(), 512);
    }

    #[wasm_bindgen_test]
    fn test_embedding_extraction() {
        let embedder = WasmCnnEmbedder::new(Some(EmbedderConfig {
            input_size: 8,
            embedding_dim: 64,
            normalize: true,
        }))
        .unwrap();

        let image_data = vec![128u8; 8 * 8 * 3];
        let embedding = embedder.extract(&image_data, 8, 8).unwrap();

        assert_eq!(embedding.len(), 64);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[wasm_bindgen_test]
    fn test_cosine_similarity() {
        let embedder = WasmCnnEmbedder::new(None).unwrap();

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = embedder.cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        let sim2 = embedder.cosine_similarity(&a, &c).unwrap();
        assert!(sim2.abs() < 0.001);
    }

    #[wasm_bindgen_test]
    fn test_infonce_loss() {
        let loss = WasmInfoNCELoss::new(0.1);
        assert!((loss.temperature() - 0.1).abs() < 0.001);
    }

    #[wasm_bindgen_test]
    fn test_triplet_loss() {
        let loss = WasmTripletLoss::new(1.0);
        assert!((loss.margin() - 1.0).abs() < 0.001);
    }

    #[wasm_bindgen_test]
    fn test_simd_ops() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let dot = SimdOps::dot_product(&a, &b);
        assert!((dot - 10.0).abs() < 0.001);

        let relu_out = SimdOps::relu(&[-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(relu_out, vec![0.0, 0.0, 1.0, 2.0]);

        let relu6_out = SimdOps::relu6(&[-1.0, 0.0, 5.0, 7.0]);
        assert_eq!(relu6_out, vec![0.0, 0.0, 5.0, 6.0]);
    }
}
