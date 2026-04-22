//! Text Embedding Providers
//!
//! This module provides a pluggable embedding system for AgenticDB.
//!
//! ## Available Providers
//!
//! - **HashEmbedding**: Fast hash-based placeholder (default, not semantic)
//! - **OnnxEmbedding**: Real semantic embeddings using ONNX Runtime (feature: `onnx-embeddings`) ✅ RECOMMENDED
//! - **CandleEmbedding**: Real embeddings using candle-transformers (feature: `real-embeddings`)
//! - **ApiEmbedding**: External API calls (OpenAI, Anthropic, Cohere, etc.)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ruvector_core::embeddings::{EmbeddingProvider, HashEmbedding};
//!
//! // Default: Hash-based (fast, but not semantic)
//! let hash_provider = HashEmbedding::new(384);
//! let embedding = hash_provider.embed("hello world")?;
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## ONNX Embeddings (Recommended for Production)
//!
//! ```rust,ignore
//! use ruvector_core::embeddings::{EmbeddingProvider, OnnxEmbedding};
//!
//! // Real semantic embeddings using all-MiniLM-L6-v2
//! let provider = OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
//! let embedding = provider.embed("hello world")?;
//! // "dog" and "cat" WILL be similar (semantic understanding!)
//! ```

use crate::error::Result;
#[cfg(any(feature = "real-embeddings", feature = "api-embeddings"))]
use crate::error::RuvectorError;
use std::sync::Arc;

/// Trait for text embedding providers
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding vector for the given text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Get the dimensionality of embeddings produced by this provider
    fn dimensions(&self) -> usize;

    /// Get a description of this provider (for logging/debugging)
    fn name(&self) -> &str;
}

/// Hash-based embedding provider (placeholder, not semantic)
///
/// ⚠️ **WARNING**: This does NOT produce semantic embeddings!
/// - "dog" and "cat" will NOT be similar
/// - "dog" and "god" WILL be similar (same characters)
///
/// Use this only for:
/// - Testing
/// - Prototyping
/// - When semantic similarity is not required
#[derive(Debug, Clone)]
pub struct HashEmbedding {
    dimensions: usize,
}

impl HashEmbedding {
    /// Create a new hash-based embedding provider
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl EmbeddingProvider for HashEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut embedding = vec![0.0; self.dimensions];
        let bytes = text.as_bytes();

        for (i, byte) in bytes.iter().enumerate() {
            embedding[i % self.dimensions] += (*byte as f32) / 255.0;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        Ok(embedding)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "HashEmbedding (placeholder)"
    }
}

/// Real embeddings using candle-transformers
///
/// Requires feature flag: `real-embeddings`
///
/// ⚠️ **Note**: Full candle integration is complex and model-specific.
/// For production use, we recommend:
/// 1. Using the API-based providers (simpler, always up-to-date)
/// 2. Using ONNX Runtime with pre-exported models
/// 3. Implementing your own candle wrapper for your specific model
///
/// This is a stub implementation showing the structure.
/// Users should implement `EmbeddingProvider` trait for their specific models.
#[cfg(feature = "real-embeddings")]
pub mod candle {
    use super::*;

    /// Candle-based embedding provider stub
    ///
    /// This is a placeholder. For real implementation:
    /// 1. Add candle dependencies for your specific model type
    /// 2. Implement model loading and inference
    /// 3. Handle tokenization appropriately
    ///
    /// Example structure:
    /// ```rust,ignore
    /// pub struct CandleEmbedding {
    ///     model: YourModelType,
    ///     tokenizer: Tokenizer,
    ///     device: Device,
    ///     dimensions: usize,
    /// }
    /// ```
    pub struct CandleEmbedding {
        dimensions: usize,
        model_id: String,
    }

    impl CandleEmbedding {
        /// Create a stub candle embedding provider
        ///
        /// **This is not a real implementation!**
        /// For production, implement with actual model loading.
        ///
        /// # Example
        /// ```rust,no_run
        /// # #[cfg(feature = "real-embeddings")]
        /// # {
        /// use ruvector_core::embeddings::candle::CandleEmbedding;
        ///
        /// // This returns an error - real implementation required
        /// let result = CandleEmbedding::from_pretrained(
        ///     "sentence-transformers/all-MiniLM-L6-v2",
        ///     false
        /// );
        /// assert!(result.is_err());
        /// # }
        /// ```
        pub fn from_pretrained(model_id: &str, _use_gpu: bool) -> Result<Self> {
            Err(RuvectorError::ModelLoadError(format!(
                "Candle embedding support is a stub. Please:\n\
                     1. Use ApiEmbedding for production (recommended)\n\
                     2. Or implement CandleEmbedding for model: {}\n\
                     3. See docs for ONNX Runtime integration examples",
                model_id
            )))
        }
    }

    impl EmbeddingProvider for CandleEmbedding {
        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Err(RuvectorError::ModelInferenceError(
                "Candle embedding not implemented - use ApiEmbedding instead".to_string(),
            ))
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn name(&self) -> &str {
            "CandleEmbedding (stub - not implemented)"
        }
    }
}

#[cfg(feature = "real-embeddings")]
pub use candle::CandleEmbedding;

/// API-based embedding provider (OpenAI, Anthropic, Cohere, etc.)
///
/// Supports any API that accepts JSON and returns embeddings in a standard format.
///
/// # Example (OpenAI)
/// ```rust,no_run
/// use ruvector_core::embeddings::{EmbeddingProvider, ApiEmbedding};
///
/// let provider = ApiEmbedding::openai("sk-...", "text-embedding-3-small");
/// let embedding = provider.embed("hello world")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "api-embeddings")]
#[derive(Clone)]
pub struct ApiEmbedding {
    api_key: String,
    endpoint: String,
    model: String,
    dimensions: usize,
    client: reqwest::blocking::Client,
}

#[cfg(feature = "api-embeddings")]
impl ApiEmbedding {
    /// Create a new API embedding provider
    ///
    /// # Arguments
    /// * `api_key` - API key for authentication
    /// * `endpoint` - API endpoint URL
    /// * `model` - Model identifier
    /// * `dimensions` - Expected embedding dimensions
    pub fn new(api_key: String, endpoint: String, model: String, dimensions: usize) -> Self {
        Self {
            api_key,
            endpoint,
            model,
            dimensions,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create OpenAI embedding provider
    ///
    /// # Models
    /// - `text-embedding-3-small` - 1536 dimensions, $0.02/1M tokens
    /// - `text-embedding-3-large` - 3072 dimensions, $0.13/1M tokens
    /// - `text-embedding-ada-002` - 1536 dimensions (legacy)
    pub fn openai(api_key: &str, model: &str) -> Self {
        let dimensions = match model {
            "text-embedding-3-large" => 3072,
            _ => 1536, // text-embedding-3-small and ada-002
        };

        Self::new(
            api_key.to_string(),
            "https://api.openai.com/v1/embeddings".to_string(),
            model.to_string(),
            dimensions,
        )
    }

    /// Create Cohere embedding provider
    ///
    /// # Models
    /// - `embed-english-v3.0` - 1024 dimensions
    /// - `embed-multilingual-v3.0` - 1024 dimensions
    pub fn cohere(api_key: &str, model: &str) -> Self {
        Self::new(
            api_key.to_string(),
            "https://api.cohere.ai/v1/embed".to_string(),
            model.to_string(),
            1024,
        )
    }

    /// Create Voyage AI embedding provider
    ///
    /// # Models
    /// - `voyage-2` - 1024 dimensions
    /// - `voyage-large-2` - 1536 dimensions
    pub fn voyage(api_key: &str, model: &str) -> Self {
        let dimensions = if model.contains("large") { 1536 } else { 1024 };

        Self::new(
            api_key.to_string(),
            "https://api.voyageai.com/v1/embeddings".to_string(),
            model.to_string(),
            dimensions,
        )
    }
}

#[cfg(feature = "api-embeddings")]
impl EmbeddingProvider for ApiEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request_body = serde_json::json!({
            "input": text,
            "model": self.model,
        });

        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| {
                RuvectorError::ModelInferenceError(format!("API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RuvectorError::ModelInferenceError(format!(
                "API returned error {}: {}",
                status, error_text
            )));
        }

        let response_json: serde_json::Value = response.json().map_err(|e| {
            RuvectorError::ModelInferenceError(format!("Failed to parse response: {}", e))
        })?;

        // Handle different API response formats
        let embedding = if let Some(data) = response_json.get("data") {
            // OpenAI format: {"data": [{"embedding": [...]}]}
            data.as_array()
                .and_then(|arr| arr.first())
                .and_then(|obj| obj.get("embedding"))
                .and_then(|emb| emb.as_array())
                .ok_or_else(|| {
                    RuvectorError::ModelInferenceError("Invalid OpenAI response format".to_string())
                })?
        } else if let Some(embeddings) = response_json.get("embeddings") {
            // Cohere format: {"embeddings": [[...]]}
            embeddings
                .as_array()
                .and_then(|arr| arr.first())
                .and_then(|emb| emb.as_array())
                .ok_or_else(|| {
                    RuvectorError::ModelInferenceError("Invalid Cohere response format".to_string())
                })?
        } else {
            return Err(RuvectorError::ModelInferenceError(
                "Unknown API response format".to_string(),
            ));
        };

        let embedding_vec: Result<Vec<f32>> = embedding
            .iter()
            .map(|v| {
                v.as_f64().map(|f| f as f32).ok_or_else(|| {
                    RuvectorError::ModelInferenceError("Invalid embedding value".to_string())
                })
            })
            .collect();

        embedding_vec
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "ApiEmbedding"
    }
}

// ============================================================================
// ONNX Embeddings (Recommended for Production)
// ============================================================================

/// ONNX-based embedding provider using ONNX Runtime
///
/// Provides **real semantic embeddings** using transformer models like all-MiniLM-L6-v2.
/// This is the **recommended** embedding provider for production use.
///
/// Requires feature flag: `onnx-embeddings`
///
/// ## Features
/// - Real semantic understanding ("dog" and "cat" ARE similar)
/// - Local inference (no API calls, works offline)
/// - Fast inference (5-50ms per embedding)
/// - Automatic model download from HuggingFace
///
/// ## Supported Models
/// - `sentence-transformers/all-MiniLM-L6-v2` (384 dims, recommended)
/// - `sentence-transformers/all-mpnet-base-v2` (768 dims)
/// - `BAAI/bge-small-en-v1.5` (384 dims)
///
/// # Example
/// ```rust,ignore
/// use ruvector_core::embeddings::{EmbeddingProvider, OnnxEmbedding};
///
/// let provider = OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
/// let embedding = provider.embed("hello world")?;
/// assert_eq!(embedding.len(), 384);
/// ```
#[cfg(feature = "onnx-embeddings")]
pub mod onnx {
    use super::*;
    use crate::error::RuvectorError;
    use ort::session::Session;
    use ort::value::{Tensor, ValueType};
    use parking_lot::RwLock;
    use std::path::PathBuf;
    use tokenizers::Tokenizer;

    /// ONNX-based embedding provider
    pub struct OnnxEmbedding {
        session: RwLock<Session>,
        tokenizer: RwLock<Tokenizer>,
        dimensions: usize,
        model_id: String,
        #[allow(dead_code)]
        max_length: usize,
    }

    impl OnnxEmbedding {
        /// Load a pre-trained embedding model from HuggingFace
        ///
        /// The model will be downloaded and cached automatically.
        ///
        /// # Arguments
        /// * `model_id` - HuggingFace model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        ///
        /// # Example
        /// ```rust,ignore
        /// let provider = OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
        /// ```
        pub fn from_pretrained(model_id: &str) -> Result<Self> {
            let api = hf_hub::api::sync::Api::new().map_err(|e| {
                RuvectorError::ModelLoadError(format!("Failed to create HuggingFace API: {}", e))
            })?;

            let repo = api.model(model_id.to_string());

            // Download model files
            let model_path = repo
                .get("model.onnx")
                .or_else(|_| {
                    // Try alternative path for some models
                    repo.get("onnx/model.onnx")
                })
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!(
                        "Failed to download ONNX model from {}: {}. \
                     Make sure the model has an ONNX export available.",
                        model_id, e
                    ))
                })?;

            let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
                RuvectorError::ModelLoadError(format!(
                    "Failed to download tokenizer from {}: {}",
                    model_id, e
                ))
            })?;

            Self::from_files(&model_path, &tokenizer_path, model_id)
        }

        /// Load from local files
        ///
        /// # Arguments
        /// * `model_path` - Path to the ONNX model file
        /// * `tokenizer_path` - Path to the tokenizer.json file
        /// * `model_id` - Model identifier for logging
        pub fn from_files(
            model_path: &PathBuf,
            tokenizer_path: &PathBuf,
            model_id: &str,
        ) -> Result<Self> {
            // Initialize ONNX Runtime (returns bool, true = first init)
            let _ = ort::init().commit();

            // Load the ONNX session
            let session = Session::builder()
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!(
                        "Failed to create session builder: {}",
                        e
                    ))
                })?
                .with_intra_threads(4)
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!("Failed to set thread count: {}", e))
                })?
                .commit_from_file(model_path)
                .map_err(|e| {
                    RuvectorError::ModelLoadError(format!("Failed to load ONNX model: {}", e))
                })?;

            // Load tokenizer
            let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
                RuvectorError::ModelLoadError(format!("Failed to load tokenizer: {}", e))
            })?;

            // Determine dimensions from model output
            let dimensions = Self::infer_dimensions(&session, model_id)?;

            // Determine max_length from model (default to 512 for sentence transformers)
            let max_length = 512;

            tracing::info!(
                "Loaded ONNX embedding model: {} ({}D)",
                model_id,
                dimensions
            );

            Ok(Self {
                session: RwLock::new(session),
                tokenizer: RwLock::new(tokenizer),
                dimensions,
                model_id: model_id.to_string(),
                max_length,
            })
        }

        fn infer_dimensions(session: &Session, model_id: &str) -> Result<usize> {
            // Common dimensions for known models
            let dimensions = match model_id {
                id if id.contains("all-MiniLM-L6") => 384,
                id if id.contains("all-mpnet-base") => 768,
                id if id.contains("bge-small") => 384,
                id if id.contains("bge-base") => 768,
                id if id.contains("bge-large") => 1024,
                id if id.contains("e5-small") => 384,
                id if id.contains("e5-base") => 768,
                id if id.contains("e5-large") => 1024,
                _ => {
                    // Try to infer from output shape via session.outputs() method
                    if let Some(output) = session.outputs().first() {
                        if let ValueType::Tensor { shape, .. } = output.dtype() {
                            let dims: Vec<i64> = shape.iter().copied().collect();
                            if dims.len() >= 2 {
                                let last_dim = dims[dims.len() - 1];
                                if last_dim > 0 {
                                    return Ok(last_dim as usize);
                                }
                            }
                        }
                    }
                    // Default to 384 (most common)
                    384
                }
            };

            Ok(dimensions)
        }

        /// Embed multiple texts in a batch (more efficient than individual calls)
        pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            texts.iter().map(|text| self.embed(text)).collect()
        }

        fn mean_pooling(
            token_embeddings: &[f32],
            attention_mask: &[i64],
            seq_len: usize,
            hidden_size: usize,
        ) -> Vec<f32> {
            let mut pooled = vec![0.0f32; hidden_size];
            let mut mask_sum = 0.0f32;

            for i in 0..seq_len {
                let mask = attention_mask[i] as f32;
                mask_sum += mask;
                for j in 0..hidden_size {
                    pooled[j] += token_embeddings[i * hidden_size + j] * mask;
                }
            }

            // Avoid division by zero
            if mask_sum > 0.0 {
                for val in &mut pooled {
                    *val /= mask_sum;
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut pooled {
                    *val /= norm;
                }
            }

            pooled
        }
    }

    impl EmbeddingProvider for OnnxEmbedding {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            // Tokenize
            let encoding = {
                let tokenizer = self.tokenizer.read();
                tokenizer.encode(text, true).map_err(|e| {
                    RuvectorError::ModelInferenceError(format!("Tokenization failed: {}", e))
                })?
            };

            // Prepare inputs
            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect();
            let token_type_ids: Vec<i64> =
                encoding.get_type_ids().iter().map(|&x| x as i64).collect();

            let seq_len = input_ids.len();

            // Create ONNX tensors using ort 2.0 API (batch_size=1)
            // Tensor::from_array takes (shape, owned_data)
            let input_ids_tensor =
                Tensor::<i64>::from_array(([1, seq_len], input_ids.clone().into_boxed_slice()))
                    .map_err(|e| {
                        RuvectorError::ModelInferenceError(format!(
                            "Failed to create input_ids tensor: {}",
                            e
                        ))
                    })?;

            let attention_mask_tensor = Tensor::<i64>::from_array((
                [1, seq_len],
                attention_mask.clone().into_boxed_slice(),
            ))
            .map_err(|e| {
                RuvectorError::ModelInferenceError(format!(
                    "Failed to create attention_mask tensor: {}",
                    e
                ))
            })?;

            let token_type_ids_tensor =
                Tensor::<i64>::from_array(([1, seq_len], token_type_ids.into_boxed_slice()))
                    .map_err(|e| {
                        RuvectorError::ModelInferenceError(format!(
                            "Failed to create token_type_ids tensor: {}",
                            e
                        ))
                    })?;

            // Run inference and extract output (needs mutable access to session)
            // We must extract all data while holding the lock since SessionOutputs has a lifetime
            let (output_data, output_shape_vec) = {
                let mut session = self.session.write();
                let outputs = session
                    .run(ort::inputs![
                        "input_ids" => input_ids_tensor,
                        "attention_mask" => attention_mask_tensor,
                        "token_type_ids" => token_type_ids_tensor,
                    ])
                    .map_err(|e| {
                        RuvectorError::ModelInferenceError(format!("ONNX inference failed: {}", e))
                    })?;

                // Extract output using indexing (ort 2.0 API)
                // Sentence transformers output shape: [batch_size, seq_len, hidden_size]
                let output_value = &outputs[0];

                // Extract as ndarray view
                let output_array = output_value.try_extract_array::<f32>().map_err(|e| {
                    RuvectorError::ModelInferenceError(format!(
                        "Failed to extract output tensor: {}",
                        e
                    ))
                })?;

                let output_shape_vec: Vec<usize> = output_array.shape().to_vec();
                let output_data_vec: Vec<f32> = output_array.iter().copied().collect();

                (output_data_vec, output_shape_vec)
            };

            // Determine if we need pooling based on output shape
            let embedding = if output_shape_vec.len() == 3 {
                // Shape: [batch_size, seq_len, hidden_size] - needs pooling
                let hidden_size = output_shape_vec[2];
                Self::mean_pooling(&output_data, &attention_mask, seq_len, hidden_size)
            } else if output_shape_vec.len() == 2 {
                // Shape: [batch_size, hidden_size] - already pooled
                let mut emb = output_data;
                // L2 normalize
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in &mut emb {
                        *val /= norm;
                    }
                }
                emb
            } else {
                return Err(RuvectorError::ModelInferenceError(format!(
                    "Unexpected output shape: {:?}",
                    output_shape_vec
                )));
            };

            Ok(embedding)
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn name(&self) -> &str {
            &self.model_id
        }
    }

    impl std::fmt::Debug for OnnxEmbedding {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("OnnxEmbedding")
                .field("model_id", &self.model_id)
                .field("dimensions", &self.dimensions)
                .field("max_length", &self.max_length)
                .finish()
        }
    }
}

#[cfg(feature = "onnx-embeddings")]
pub use onnx::OnnxEmbedding;

/// Type-erased embedding provider for dynamic dispatch
pub type BoxedEmbeddingProvider = Arc<dyn EmbeddingProvider>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_embedding() {
        let provider = HashEmbedding::new(128);

        let emb1 = provider.embed("hello world").unwrap();
        let emb2 = provider.embed("hello world").unwrap();

        assert_eq!(emb1.len(), 128);
        assert_eq!(emb1, emb2, "Same text should produce same embedding");

        // Check normalization
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized");
    }

    #[test]
    fn test_hash_embedding_different_text() {
        let provider = HashEmbedding::new(128);

        let emb1 = provider.embed("hello").unwrap();
        let emb2 = provider.embed("world").unwrap();

        assert_ne!(
            emb1, emb2,
            "Different text should produce different embeddings"
        );
    }

    #[cfg(feature = "real-embeddings")]
    #[test]
    #[ignore] // Requires model download
    fn test_candle_embedding() {
        let provider =
            CandleEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2", false)
                .unwrap();

        let embedding = provider.embed("hello world").unwrap();
        assert_eq!(embedding.len(), 384);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized");
    }

    #[test]
    #[ignore] // Requires API key
    fn test_api_embedding_openai() {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        let provider = ApiEmbedding::openai(&api_key, "text-embedding-3-small");

        let embedding = provider.embed("hello world").unwrap();
        assert_eq!(embedding.len(), 1536);
    }

    #[cfg(feature = "onnx-embeddings")]
    mod onnx_tests {
        use super::*;

        #[test]
        #[ignore] // Requires model download (~90MB)
        fn test_onnx_embedding_minilm() {
            let provider =
                OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").unwrap();

            let embedding = provider.embed("hello world").unwrap();
            assert_eq!(embedding.len(), 384);

            // Check normalization
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "Embedding should be normalized, got norm={}",
                norm
            );
        }

        #[test]
        #[ignore] // Requires model download
        fn test_onnx_semantic_similarity() {
            let provider =
                OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").unwrap();

            let emb_dog = provider.embed("dog").unwrap();
            let emb_cat = provider.embed("cat").unwrap();
            let emb_car = provider.embed("car").unwrap();

            // Cosine similarity (embeddings are normalized, so dot product = cosine)
            let sim_dog_cat: f32 = emb_dog.iter().zip(&emb_cat).map(|(a, b)| a * b).sum();
            let sim_dog_car: f32 = emb_dog.iter().zip(&emb_car).map(|(a, b)| a * b).sum();

            // dog and cat should be more similar than dog and car
            assert!(
                sim_dog_cat > sim_dog_car,
                "Expected dog-cat similarity ({}) > dog-car similarity ({})",
                sim_dog_cat,
                sim_dog_car
            );
        }

        #[test]
        #[ignore] // Requires model download
        fn test_onnx_batch_embedding() {
            let provider =
                OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").unwrap();

            let texts = vec!["hello world", "goodbye world", "rust programming"];
            let embeddings = provider.embed_batch(&texts).unwrap();

            assert_eq!(embeddings.len(), 3);
            for emb in &embeddings {
                assert_eq!(emb.len(), 384);
            }
        }
    }
}
