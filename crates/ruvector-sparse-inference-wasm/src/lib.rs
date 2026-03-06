use ruvector_sparse_inference::{
    model::{GgufParser, ModelMetadata, ModelRunner, SparseModel},
    InferenceConfig, LowRankPredictor, SparsityConfig,
};
use wasm_bindgen::prelude::*;

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Sparse inference engine for WASM
#[wasm_bindgen]
pub struct SparseInferenceEngine {
    model: SparseModel,
    config: InferenceConfig,
}

#[wasm_bindgen]
impl SparseInferenceEngine {
    /// Create new engine from GGUF bytes
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], config_json: &str) -> Result<SparseInferenceEngine, JsError> {
        let config: InferenceConfig = serde_json::from_str(config_json)
            .map_err(|e| JsError::new(&format!("Invalid config: {}", e)))?;

        let model = GgufParser::parse(model_bytes)
            .map_err(|e| JsError::new(&format!("Failed to parse model: {}", e)))?;

        Ok(Self { model, config })
    }

    /// Load model with streaming (for large models)
    #[wasm_bindgen]
    pub async fn load_streaming(
        url: &str,
        config_json: &str,
    ) -> Result<SparseInferenceEngine, JsError> {
        let bytes = fetch_model_bytes(url).await?;
        Self::new(&bytes, config_json)
    }

    /// Run inference on input
    #[wasm_bindgen]
    pub fn infer(&self, input: &[f32]) -> Result<Vec<f32>, JsError> {
        self.model
            .forward_embedding(input, &self.config)
            .map_err(|e| JsError::new(&format!("Inference failed: {}", e)))
    }

    /// Get model metadata as JSON
    #[wasm_bindgen]
    pub fn metadata(&self) -> String {
        serde_json::to_string(&self.model.metadata()).unwrap_or_default()
    }

    /// Get sparsity statistics
    #[wasm_bindgen]
    pub fn sparsity_stats(&self) -> String {
        let stats = self.model.sparsity_statistics();
        serde_json::to_string(&stats).unwrap_or_default()
    }

    /// Calibrate with sample inputs
    #[wasm_bindgen]
    pub fn calibrate(&mut self, samples: &[f32], sample_dim: usize) -> Result<(), JsError> {
        let samples: Vec<Vec<f32>> = samples.chunks(sample_dim).map(|c| c.to_vec()).collect();
        self.model
            .calibrate(&samples)
            .map_err(|e| JsError::new(&format!("Calibration failed: {}", e)))
    }
}

/// Embedding model wrapper for sentence transformers
#[wasm_bindgen]
pub struct EmbeddingModel {
    engine: SparseInferenceEngine,
}

#[wasm_bindgen]
impl EmbeddingModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8]) -> Result<EmbeddingModel, JsError> {
        let config =
            r#"{"sparsity": {"enabled": true, "threshold": 0.1}, "temperature": 1.0, "top_k": 50}"#;
        let engine = SparseInferenceEngine::new(model_bytes, config)?;
        Ok(Self { engine })
    }

    /// Encode text to embedding (requires tokenizer)
    #[wasm_bindgen]
    pub fn encode(&self, input_ids: &[u32]) -> Result<Vec<f32>, JsError> {
        self.engine
            .model
            .encode(input_ids)
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
            let embedding = self
                .engine
                .model
                .encode(ids)
                .map_err(|e| JsError::new(&format!("Encoding failed: {}", e)))?;
            results.extend(embedding);
            offset += len;
        }
        Ok(results)
    }

    /// Get embedding dimension
    #[wasm_bindgen]
    pub fn dimension(&self) -> usize {
        self.engine.model.metadata().hidden_size
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
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().ok_or_else(|| JsError::new("No window"))?;
    let response = JsFuture::from(window.fetch_with_str(url)).await?;
    let response: web_sys::Response = response
        .dyn_into()
        .map_err(|_| JsError::new("Failed to cast to Response"))?;
    let buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|_| JsError::new("Failed to get array buffer"))?,
    )
    .await?;
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
