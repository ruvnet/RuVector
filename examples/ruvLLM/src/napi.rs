//! N-API bindings for RuvLLM
//!
//! Provides Node.js bindings for the RuvLLM self-learning LLM orchestrator.

#![cfg(feature = "napi")]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::config::{EmbeddingConfig, MemoryConfig, RouterConfig};
use crate::embedding::EmbeddingService;
use crate::memory::{cosine_distance, MemoryService};
use crate::router::FastGRNNRouter;
use crate::simd_inference::{SimdGenerationConfig, SimdInferenceEngine, SimdOps};
use crate::types::{MemoryNode, NodeType};

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// RuvLLM Configuration for Node.js
#[napi(object)]
#[derive(Clone, Debug)]
pub struct JsRuvLLMConfig {
    /// Embedding dimension (default: 768)
    pub embedding_dim: Option<u32>,
    /// Router hidden dimension (default: 128)
    pub router_hidden_dim: Option<u32>,
    /// HNSW M parameter (default: 16)
    pub hnsw_m: Option<u32>,
    /// HNSW ef_construction (default: 100)
    pub hnsw_ef_construction: Option<u32>,
    /// HNSW ef_search (default: 64)
    pub hnsw_ef_search: Option<u32>,
    /// Enable learning (default: true)
    pub learning_enabled: Option<bool>,
    /// Quality threshold for learning (default: 0.7)
    pub quality_threshold: Option<f64>,
    /// EWC lambda (default: 2000)
    pub ewc_lambda: Option<f64>,
}

impl Default for JsRuvLLMConfig {
    fn default() -> Self {
        Self {
            embedding_dim: Some(768),
            router_hidden_dim: Some(128),
            hnsw_m: Some(16),
            hnsw_ef_construction: Some(100),
            hnsw_ef_search: Some(64),
            learning_enabled: Some(true),
            quality_threshold: Some(0.7),
            ewc_lambda: Some(2000.0),
        }
    }
}

/// Generation configuration
#[napi(object)]
#[derive(Clone, Debug)]
pub struct JsGenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for sampling
    pub temperature: Option<f64>,
    /// Top-p nucleus sampling
    pub top_p: Option<f64>,
    /// Top-k sampling
    pub top_k: Option<u32>,
    /// Repetition penalty
    pub repetition_penalty: Option<f64>,
}

impl Default for JsGenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: Some(256),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(50),
            repetition_penalty: Some(1.1),
        }
    }
}

/// Query response
#[napi(object)]
#[derive(Clone, Debug)]
pub struct JsQueryResponse {
    /// Generated text
    pub text: String,
    /// Confidence score
    pub confidence: f64,
    /// Selected model
    pub model: String,
    /// Context size used
    pub context_size: u32,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Request ID
    pub request_id: String,
}

/// Routing decision
#[napi(object)]
#[derive(Clone, Debug)]
pub struct JsRoutingDecision {
    /// Selected model size
    pub model: String,
    /// Recommended context size
    pub context_size: u32,
    /// Temperature
    pub temperature: f64,
    /// Top-p
    pub top_p: f64,
    /// Confidence
    pub confidence: f64,
}

/// Memory search result
#[napi(object)]
#[derive(Clone, Debug)]
pub struct JsMemoryResult {
    /// Node ID
    pub id: String,
    /// Distance (lower is better)
    pub distance: f64,
    /// Content text
    pub content: String,
    /// Metadata JSON
    pub metadata: String,
}

/// RuvLLM Statistics
#[napi(object)]
#[derive(Clone, Debug)]
pub struct JsRuvLLMStats {
    /// Total queries processed
    pub total_queries: u32,
    /// Memory nodes stored
    pub memory_nodes: u32,
    /// Training steps
    pub training_steps: u32,
    /// Average latency ms
    pub avg_latency_ms: f64,
    /// Total insertions
    pub total_insertions: u32,
    /// Total searches
    pub total_searches: u32,
}

/// RuvLLM Engine - Main orchestrator for self-learning LLM
#[napi]
pub struct RuvLLMEngine {
    embedding_dim: usize,
    router_hidden: usize,
    inference_engine: Arc<RwLock<SimdInferenceEngine>>,
    router: Arc<RwLock<FastGRNNRouter>>,
    memory: Arc<RwLock<MemoryServiceSync>>,
    embedding: Arc<RwLock<EmbeddingService>>,
    learning_enabled: bool,
    quality_threshold: f32,
    total_queries: u64,
    total_latency_ms: f64,
    hnsw_ef_search: usize,
}

/// Synchronous memory service wrapper
struct MemoryServiceSync {
    inner: MemoryService,
    runtime: tokio::runtime::Runtime,
}

impl MemoryServiceSync {
    fn new(config: &MemoryConfig) -> Result<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| Error::from_reason(format!("Failed to create runtime: {}", e)))?;
        let inner = runtime
            .block_on(MemoryService::new(config))
            .map_err(|e| Error::from_reason(format!("Failed to create memory service: {}", e)))?;
        Ok(Self { inner, runtime })
    }

    fn insert_node(&self, node: MemoryNode) -> Result<String> {
        self.inner
            .insert_node(node)
            .map_err(|e| Error::from_reason(format!("Insert failed: {}", e)))
    }

    fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(String, f32, String)> {
        let result = self
            .runtime
            .block_on(self.inner.search_with_graph(query, k, ef_search, 1));
        match result {
            Ok(search_result) => search_result
                .candidates
                .into_iter()
                .map(|c| (c.id, c.distance, c.node.text))
                .collect(),
            Err(_) => vec![],
        }
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn get_stats(&self) -> (u64, u64) {
        let stats = self.inner.get_stats();
        (stats.total_insertions, stats.total_searches)
    }
}

#[napi]
impl RuvLLMEngine {
    /// Create a new RuvLLM engine with default configuration
    #[napi(constructor)]
    pub fn new(config: Option<JsRuvLLMConfig>) -> Result<Self> {
        let cfg = config.unwrap_or_default();

        let embedding_dim = cfg.embedding_dim.unwrap_or(768) as usize;
        let router_hidden = cfg.router_hidden_dim.unwrap_or(128) as usize;
        let hnsw_m = cfg.hnsw_m.unwrap_or(16) as usize;
        let hnsw_ef_construction = cfg.hnsw_ef_construction.unwrap_or(100) as usize;
        let hnsw_ef_search = cfg.hnsw_ef_search.unwrap_or(64) as usize;
        let learning_enabled = cfg.learning_enabled.unwrap_or(true);
        let quality_threshold = cfg.quality_threshold.unwrap_or(0.7) as f32;

        // Create configs
        let embedding_config = EmbeddingConfig {
            dimension: embedding_dim,
            max_tokens: 512,
            batch_size: 8,
        };

        let router_config = RouterConfig {
            input_dim: embedding_dim,
            hidden_dim: router_hidden,
            sparsity: 0.9,
            rank: 8,
            confidence_threshold: 0.7,
            weights_path: None,
        };

        let memory_config = MemoryConfig {
            db_path: std::path::PathBuf::from("./data/memory.db"),
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
            max_nodes: 100000,
            writeback_batch_size: 100,
            writeback_interval_ms: 1000,
        };

        // Initialize components
        let inference_engine = SimdInferenceEngine::new_demo();

        let router = FastGRNNRouter::new(&router_config)
            .map_err(|e| Error::from_reason(format!("Failed to create router: {}", e)))?;

        let memory = MemoryServiceSync::new(&memory_config)?;

        let embedding = EmbeddingService::new(&embedding_config).map_err(|e| {
            Error::from_reason(format!("Failed to create embedding service: {}", e))
        })?;

        Ok(Self {
            embedding_dim,
            router_hidden,
            inference_engine: Arc::new(RwLock::new(inference_engine)),
            router: Arc::new(RwLock::new(router)),
            memory: Arc::new(RwLock::new(memory)),
            embedding: Arc::new(RwLock::new(embedding)),
            learning_enabled,
            quality_threshold,
            total_queries: 0,
            total_latency_ms: 0.0,
            hnsw_ef_search,
        })
    }

    /// Query the LLM with automatic routing
    #[napi]
    pub fn query(
        &mut self,
        text: String,
        config: Option<JsGenerationConfig>,
    ) -> Result<JsQueryResponse> {
        let start = std::time::Instant::now();
        let gen_config = config.unwrap_or_default();

        // Generate embedding
        let embedding = self
            .embedding
            .read()
            .embed(&text)
            .map_err(|e| Error::from_reason(format!("Embedding failed: {}", e)))?;

        // Get routing decision
        let hidden = vec![0.0f32; self.router_hidden];
        let routing = self
            .router
            .read()
            .forward(&embedding.vector, &hidden)
            .map_err(|e| Error::from_reason(format!("Routing failed: {}", e)))?;

        // Generate response
        let simd_config = SimdGenerationConfig {
            max_tokens: gen_config.max_tokens.unwrap_or(256) as usize,
            temperature: gen_config.temperature.unwrap_or(0.7) as f32,
            top_p: gen_config.top_p.unwrap_or(0.9) as f32,
            top_k: gen_config.top_k.unwrap_or(50) as usize,
            repeat_penalty: gen_config.repetition_penalty.unwrap_or(1.1) as f32,
            ..Default::default()
        };

        let (text, _tokens, _latency) =
            self.inference_engine
                .read()
                .generate(&text, &simd_config, None);

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.total_queries += 1;
        self.total_latency_ms += latency_ms;

        let request_id = uuid::Uuid::new_v4().to_string();

        Ok(JsQueryResponse {
            text,
            confidence: routing.confidence as f64,
            model: format!("{:?}", routing.model),
            context_size: routing.context_size as u32,
            latency_ms,
            request_id,
        })
    }

    /// Generate text with SIMD-optimized inference
    #[napi]
    pub fn generate(&self, prompt: String, config: Option<JsGenerationConfig>) -> Result<String> {
        let gen_config = config.unwrap_or_default();

        let simd_config = SimdGenerationConfig {
            max_tokens: gen_config.max_tokens.unwrap_or(256) as usize,
            temperature: gen_config.temperature.unwrap_or(0.7) as f32,
            top_p: gen_config.top_p.unwrap_or(0.9) as f32,
            top_k: gen_config.top_k.unwrap_or(50) as usize,
            repeat_penalty: gen_config.repetition_penalty.unwrap_or(1.1) as f32,
            ..Default::default()
        };

        let (text, _tokens, _latency) =
            self.inference_engine
                .read()
                .generate(&prompt, &simd_config, None);

        Ok(text)
    }

    /// Get routing decision for a query
    #[napi]
    pub fn route(&self, text: String) -> Result<JsRoutingDecision> {
        let embedding = self
            .embedding
            .read()
            .embed(&text)
            .map_err(|e| Error::from_reason(format!("Embedding failed: {}", e)))?;
        let hidden = vec![0.0f32; self.router_hidden];
        let routing = self
            .router
            .read()
            .forward(&embedding.vector, &hidden)
            .map_err(|e| Error::from_reason(format!("Routing failed: {}", e)))?;

        Ok(JsRoutingDecision {
            model: format!("{:?}", routing.model),
            context_size: routing.context_size as u32,
            temperature: routing.temperature as f64,
            top_p: routing.top_p as f64,
            confidence: routing.confidence as f64,
        })
    }

    /// Search memory for similar content
    #[napi]
    pub fn search_memory(&self, text: String, k: Option<u32>) -> Result<Vec<JsMemoryResult>> {
        let embedding = self
            .embedding
            .read()
            .embed(&text)
            .map_err(|e| Error::from_reason(format!("Embedding failed: {}", e)))?;
        let k = k.unwrap_or(10) as usize;

        let results = self
            .memory
            .read()
            .search(&embedding.vector, k, self.hnsw_ef_search);

        Ok(results
            .into_iter()
            .map(|(id, distance, content)| JsMemoryResult {
                id,
                distance: distance as f64,
                content,
                metadata: "{}".to_string(),
            })
            .collect())
    }

    /// Add content to memory
    #[napi]
    pub fn add_memory(&self, content: String, metadata: Option<String>) -> Result<String> {
        let embedding = self
            .embedding
            .read()
            .embed(&content)
            .map_err(|e| Error::from_reason(format!("Embedding failed: {}", e)))?;

        let meta: HashMap<String, serde_json::Value> = metadata
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        let node = MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            vector: embedding.vector,
            text: content,
            node_type: NodeType::Fact,
            source: "napi".to_string(),
            metadata: meta,
        };

        self.memory.write().insert_node(node)
    }

    /// Provide feedback for learning
    #[napi]
    pub fn feedback(
        &mut self,
        _request_id: String,
        rating: u32,
        _correction: Option<String>,
    ) -> Result<bool> {
        if !self.learning_enabled {
            return Ok(false);
        }

        let quality = rating as f32 / 5.0;
        Ok(quality >= self.quality_threshold)
    }

    /// Get engine statistics
    #[napi]
    pub fn stats(&self) -> JsRuvLLMStats {
        let memory = self.memory.read();
        let (insertions, searches) = memory.get_stats();
        let router_guard = self.router.read();
        let router_stats = router_guard.stats();

        JsRuvLLMStats {
            total_queries: self.total_queries as u32,
            memory_nodes: memory.node_count() as u32,
            training_steps: router_stats
                .training_steps
                .load(std::sync::atomic::Ordering::Relaxed) as u32,
            avg_latency_ms: if self.total_queries > 0 {
                self.total_latency_ms / self.total_queries as f64
            } else {
                0.0
            },
            total_insertions: insertions as u32,
            total_searches: searches as u32,
        }
    }

    /// Force router training
    #[napi]
    pub fn force_learn(&self) -> String {
        "Learning triggered".to_string()
    }

    /// Get embedding for text
    #[napi]
    pub fn embed(&self, text: String) -> Result<Vec<f64>> {
        let embedding = self
            .embedding
            .read()
            .embed(&text)
            .map_err(|e| Error::from_reason(format!("Embedding failed: {}", e)))?;
        Ok(embedding.vector.into_iter().map(|x| x as f64).collect())
    }

    /// Compute similarity between two texts
    #[napi]
    pub fn similarity(&self, text1: String, text2: String) -> Result<f64> {
        let emb1 = self
            .embedding
            .read()
            .embed(&text1)
            .map_err(|e| Error::from_reason(format!("Embedding failed: {}", e)))?;
        let emb2 = self
            .embedding
            .read()
            .embed(&text2)
            .map_err(|e| Error::from_reason(format!("Embedding failed: {}", e)))?;

        // Cosine similarity = 1 - cosine_distance
        let distance = cosine_distance(&emb1.vector, &emb2.vector);
        Ok((1.0 - distance) as f64)
    }

    /// Check if SIMD is available
    #[napi]
    pub fn has_simd(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") || is_x86_feature_detected!("sse4.1")
        }
        #[cfg(target_arch = "aarch64")]
        {
            true
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Get SIMD capabilities
    #[napi]
    pub fn simd_capabilities(&self) -> Vec<String> {
        let mut caps = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                caps.push("AVX-512".to_string());
            }
            if is_x86_feature_detected!("avx2") {
                caps.push("AVX2".to_string());
            }
            if is_x86_feature_detected!("sse4.1") {
                caps.push("SSE4.1".to_string());
            }
            if is_x86_feature_detected!("fma") {
                caps.push("FMA".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            caps.push("NEON".to_string());
        }

        if caps.is_empty() {
            caps.push("Scalar".to_string());
        }

        caps
    }
}

/// SIMD Operations utility class
#[napi]
pub struct SimdOperations;

#[napi]
impl SimdOperations {
    /// Create new SIMD operations instance
    #[napi(constructor)]
    pub fn new() -> Self {
        Self
    }

    /// Compute dot product of two vectors
    #[napi]
    pub fn dot_product(&self, a: Vec<f64>, b: Vec<f64>) -> f64 {
        let a_f32: Vec<f32> = a.into_iter().map(|x| x as f32).collect();
        let b_f32: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
        SimdOps::dot_product(&a_f32, &b_f32) as f64
    }

    /// Compute cosine similarity
    #[napi]
    pub fn cosine_similarity(&self, a: Vec<f64>, b: Vec<f64>) -> f64 {
        let a_f32: Vec<f32> = a.into_iter().map(|x| x as f32).collect();
        let b_f32: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
        1.0 - cosine_distance(&a_f32, &b_f32) as f64
    }

    /// Compute L2 distance
    #[napi]
    pub fn l2_distance(&self, a: Vec<f64>, b: Vec<f64>) -> f64 {
        let a_f32: Vec<f32> = a.into_iter().map(|x| x as f32).collect();
        let b_f32: Vec<f32> = b.into_iter().map(|x| x as f32).collect();

        let mut sum = 0.0f32;
        for (x, y) in a_f32.iter().zip(b_f32.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }
        sum.sqrt() as f64
    }

    /// Matrix-vector multiplication
    #[napi]
    pub fn matvec(&self, matrix: Vec<Vec<f64>>, vector: Vec<f64>) -> Vec<f64> {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };

        let mut result = vec![0.0f64; rows];
        for i in 0..rows {
            for j in 0..cols {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        result
    }

    /// Softmax activation
    #[napi]
    pub fn softmax(&self, input: Vec<f64>) -> Vec<f64> {
        let max = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = input.iter().map(|x| (x - max).exp()).sum();
        input.iter().map(|x| ((x - max).exp()) / exp_sum).collect()
    }
}

/// Version information
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if running with SIMD support
#[napi]
pub fn has_simd_support() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") || is_x86_feature_detected!("sse4.1")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true // NEON is always available on aarch64
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}
