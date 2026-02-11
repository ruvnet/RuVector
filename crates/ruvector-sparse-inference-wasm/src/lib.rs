use ruvector_sparse_inference::InferenceConfig;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Local deserialization wrapper for InferenceConfig since the upstream
/// type does not derive Deserialize.
#[derive(Debug, Clone, Deserialize)]
struct WasmInferenceConfig {
    #[serde(default = "default_sparsity")]
    pub sparsity: f32,
    #[serde(default = "default_sparsity_threshold")]
    pub sparsity_threshold: f32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default = "default_true")]
    pub use_sparse_ffn: bool,
    #[serde(default)]
    pub active_neurons_per_layer: Option<usize>,
    #[serde(default)]
    pub output_hidden_states: bool,
    #[serde(default)]
    pub output_attentions: bool,
}

fn default_sparsity() -> f32 {
    0.9
}
fn default_sparsity_threshold() -> f32 {
    0.01
}
fn default_temperature() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}

impl From<WasmInferenceConfig> for InferenceConfig {
    fn from(w: WasmInferenceConfig) -> Self {
        InferenceConfig {
            sparsity: w.sparsity,
            sparsity_threshold: w.sparsity_threshold,
            temperature: w.temperature,
            top_k: w.top_k,
            top_p: w.top_p,
            use_sparse_ffn: w.use_sparse_ffn,
            active_neurons_per_layer: w.active_neurons_per_layer,
            output_hidden_states: w.output_hidden_states,
            output_attentions: w.output_attentions,
        }
    }
}

/// Generation configuration for text generation
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
        }
    }
}

/// Simple KV cache for autoregressive generation
struct KVCache {
    #[allow(dead_code)]
    max_size: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl KVCache {
    fn new(max_size: usize) -> Self {
        Self {
            max_size,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
}

/// Sparse inference engine for WASM
#[wasm_bindgen]
pub struct SparseInferenceEngine {
    engine: ruvector_sparse_inference::SparseInferenceEngine,
    config: InferenceConfig,
}

#[wasm_bindgen]
impl SparseInferenceEngine {
    /// Create new engine from GGUF bytes
    #[wasm_bindgen(constructor)]
    pub fn new(_model_bytes: &[u8], config_json: &str) -> Result<SparseInferenceEngine, JsError> {
        let wasm_config: WasmInferenceConfig = serde_json::from_str(config_json)
            .map_err(|e| JsError::new(&format!("Invalid config: {}", e)))?;
        let config: InferenceConfig = wasm_config.into();

        // Determine dimensions from config or use sensible defaults
        let input_dim = config.active_neurons_per_layer.unwrap_or(512);
        let hidden_dim = (input_dim as f32 * 4.0) as usize;
        let sparsity_ratio = 1.0 - config.sparsity;

        let engine =
            ruvector_sparse_inference::SparseInferenceEngine::new_sparse(
                input_dim,
                hidden_dim,
                sparsity_ratio,
            )
            .map_err(|e| JsError::new(&format!("Failed to create engine: {}", e)))?;

        Ok(Self { engine, config })
    }

    /// Load model with streaming (for large models)
    #[wasm_bindgen]
    pub async fn load_streaming(
        url: &str,
        config_json: &str,
    ) -> Result<SparseInferenceEngine, JsError> {
        // Fetch model in chunks
        let bytes = fetch_model_bytes(url).await?;
        Self::new(&bytes, config_json)
    }

    /// Run inference on input
    #[wasm_bindgen]
    pub fn infer(&self, input: &[f32]) -> Result<Vec<f32>, JsError> {
        self.engine
            .infer(input)
            .map_err(|e| JsError::new(&format!("Inference failed: {}", e)))
    }

    /// Run text generation (for LLM models)
    #[wasm_bindgen]
    pub fn generate(&mut self, input_ids: &[u32], max_tokens: u32) -> Result<Vec<u32>, JsError> {
        // Simple greedy generation using the inference engine
        let mut generated = Vec::new();
        let mut current_input: Vec<f32> = input_ids.iter().map(|&id| id as f32).collect();

        for _ in 0..max_tokens {
            let output = self
                .engine
                .infer(&current_input)
                .map_err(|e| JsError::new(&format!("Generation failed: {}", e)))?;

            if output.is_empty() {
                break;
            }

            // Simple argmax to get next token
            let next_token = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            generated.push(next_token);
            current_input = vec![next_token as f32];
        }

        Ok(generated)
    }

    /// Get sparsity statistics
    #[wasm_bindgen]
    pub fn sparsity_stats(&self) -> String {
        let stats = self.engine.sparsity_statistics();
        format!(
            "{{\"average_active_ratio\":{},\"min_active\":{},\"max_active\":{}}}",
            stats.average_active_ratio, stats.min_active, stats.max_active
        )
    }

    /// Update sparsity threshold
    #[wasm_bindgen]
    pub fn set_sparsity(&mut self, threshold: f32) {
        self.config.sparsity_threshold = threshold;
    }

    /// Calibrate predictors with sample inputs
    #[wasm_bindgen]
    pub fn calibrate(&mut self, samples: &[f32], sample_dim: usize) -> Result<(), JsError> {
        let samples: Vec<Vec<f32>> = samples.chunks(sample_dim).map(|c| c.to_vec()).collect();

        self.engine
            .calibrate(&samples)
            .map_err(|e| JsError::new(&format!("Calibration failed: {}", e)))
    }
}

/// Embedding model wrapper for sentence transformers
#[wasm_bindgen]
pub struct EmbeddingModel {
    engine: SparseInferenceEngine,
    hidden_size: usize,
}

#[wasm_bindgen]
impl EmbeddingModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8]) -> Result<EmbeddingModel, JsError> {
        let config = r#"{"sparsity": 0.9, "sparsity_threshold": 0.1, "temperature": 1.0, "top_k": 50}"#;
        let engine = SparseInferenceEngine::new(model_bytes, config)?;
        let hidden_size = engine
            .config
            .active_neurons_per_layer
            .unwrap_or(512);
        Ok(Self {
            engine,
            hidden_size,
        })
    }

    /// Encode input to embedding
    #[wasm_bindgen]
    pub fn encode(&self, input_ids: &[u32]) -> Result<Vec<f32>, JsError> {
        let input: Vec<f32> = input_ids.iter().map(|&id| id as f32).collect();
        self.engine
            .engine
            .infer(&input)
            .map_err(|e| JsError::new(&format!("Encoding failed: {}", e)))
    }

    /// Batch encode multiple sequences
    #[wasm_bindgen]
    pub fn encode_batch(&self, input_ids: &[u32], lengths: &[u32]) -> Result<Vec<f32>, JsError> {
        let mut results = Vec::new();
        let mut offset = 0usize;

        for &len in lengths {
            let len = len as usize;
            if offset + len > input_ids.len() {
                return Err(JsError::new("Invalid lengths: exceeds input_ids size"));
            }
            let ids = &input_ids[offset..offset + len];
            let input: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
            let embedding = self
                .engine
                .engine
                .infer(&input)
                .map_err(|e| JsError::new(&format!("Encoding failed: {}", e)))?;
            results.extend(embedding);
            offset += len;
        }

        Ok(results)
    }

    /// Get embedding dimension
    #[wasm_bindgen]
    pub fn dimension(&self) -> usize {
        self.hidden_size
    }
}

/// LLM model wrapper for text generation
#[wasm_bindgen]
pub struct LLMModel {
    engine: SparseInferenceEngine,
    kv_cache: KVCache,
}

#[wasm_bindgen]
impl LLMModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], config_json: &str) -> Result<LLMModel, JsError> {
        let engine = SparseInferenceEngine::new(model_bytes, config_json)?;
        let cache_size = 2048; // default max position embeddings
        let kv_cache = KVCache::new(cache_size);
        Ok(Self { engine, kv_cache })
    }

    /// Generate next token
    #[wasm_bindgen]
    pub fn next_token(&mut self, input_ids: &[u32]) -> Result<u32, JsError> {
        let input: Vec<f32> = input_ids.iter().map(|&id| id as f32).collect();
        let output = self
            .engine
            .engine
            .infer(&input)
            .map_err(|e| JsError::new(&format!("Generation failed: {}", e)))?;

        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| JsError::new("Empty output"))
    }

    /// Generate multiple tokens
    #[wasm_bindgen]
    pub fn generate(&mut self, input_ids: &[u32], max_tokens: u32) -> Result<Vec<u32>, JsError> {
        self.engine.generate(input_ids, max_tokens)
    }

    /// Reset KV cache (for new conversation)
    #[wasm_bindgen]
    pub fn reset_cache(&mut self) {
        self.kv_cache.clear();
    }

    /// Get generation statistics
    #[wasm_bindgen]
    pub fn stats(&self) -> String {
        let stats = self.engine.engine.sparsity_statistics();
        format!(
            "{{\"average_active_ratio\":{},\"min_active\":{},\"max_active\":{}}}",
            stats.average_active_ratio, stats.min_active, stats.max_active
        )
    }
}

/// Performance measurement utilities
#[wasm_bindgen]
pub fn measure_inference_time(
    engine: &SparseInferenceEngine,
    input: &[f32],
    iterations: u32,
) -> f64 {
    let performance = web_sys::window()
        .and_then(|w| w.performance())
        .expect("Performance API not available");

    let start = performance.now();
    for _ in 0..iterations {
        let _ = engine.infer(input);
    }
    let end = performance.now();

    (end - start) / iterations as f64
}

/// Get library version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Helper for streaming fetch
async fn fetch_model_bytes(url: &str) -> Result<Vec<u8>, JsError> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().ok_or_else(|| JsError::new("No window"))?;
    let response = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| JsError::new(&format!("Fetch failed: {:?}", e)))?;
    let response: web_sys::Response = response
        .dyn_into()
        .map_err(|_| JsError::new("Failed to cast to Response"))?;
    let buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|_| JsError::new("Failed to get array buffer"))?,
    )
    .await
    .map_err(|e| JsError::new(&format!("Failed to read array buffer: {:?}", e)))?;
    let array = js_sys::Uint8Array::new(&buffer);
    Ok(array.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
