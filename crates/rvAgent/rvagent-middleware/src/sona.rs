//! SONA Adaptive Learning Middleware (ADR-103 B5, ADR-108)
//!
//! Integrates SONA (Self-Optimizing Neural Architecture) into the rvAgent middleware pipeline
//! for runtime-adaptive learning with three loops:
//!
//! - **Loop A (Instant)**: Record trajectories in `wrap_model_call` via lock-free TrajectoryBuffer
//! - **Loop B (Background)**: Periodic ReasoningBank pattern extraction (hourly)
//! - **Loop C (Deep)**: Session-end consolidation with EWC++ to prevent catastrophic forgetting
//!
//! # Example
//!
//! ```rust,ignore
//! use rvagent_middleware::sona::{SonaMiddleware, SonaMiddlewareConfig};
//!
//! let config = SonaMiddlewareConfig::default();
//! let middleware = SonaMiddleware::new(config);
//!
//! // Add to pipeline
//! pipeline.push(Box::new(middleware));
//! ```

#[cfg(feature = "sona")]
use ruvector_sona::{
    EwcConfig, EwcPlusPlus, PatternConfig, ReasoningBank, SonaConfig, SonaEngine,
    TrajectoryBuffer, TrajectoryBuilder, TrajectoryIdGen,
};

use crate::{
    AgentState, AgentStateUpdate, AsyncModelHandler, Middleware, ModelHandler,
    ModelRequest, ModelResponse, Role, RunnableConfig, Runtime,
};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
#[allow(unused_imports)]
use tracing::{debug, info, trace};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for SONA middleware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonaMiddlewareConfig {
    /// Whether SONA learning is enabled.
    pub enabled: bool,

    /// Hidden dimension for embeddings.
    pub hidden_dim: usize,

    /// Embedding dimension (usually same as hidden_dim).
    pub embedding_dim: usize,

    /// Micro-LoRA rank (1-2 recommended for instant learning).
    pub micro_lora_rank: usize,

    /// Base-LoRA rank for background learning.
    pub base_lora_rank: usize,

    /// Trajectory buffer capacity (lock-free queue).
    pub trajectory_buffer_capacity: usize,

    /// Quality threshold for learning (0.0-1.0).
    pub quality_threshold: f32,

    /// Background learning interval in seconds.
    pub background_interval_secs: u64,

    /// Number of pattern clusters for K-means++.
    pub pattern_clusters: usize,

    /// EWC lambda for catastrophic forgetting prevention.
    pub ewc_lambda: f32,

    /// Maximum tasks to remember in EWC++.
    pub ewc_max_tasks: usize,

    /// Enable trajectory recording.
    pub record_trajectories: bool,

    /// Enable pattern search for context augmentation.
    pub enable_pattern_search: bool,

    /// Number of patterns to retrieve for context.
    pub pattern_search_k: usize,
}

impl Default for SonaMiddlewareConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hidden_dim: 256,
            embedding_dim: 256,
            micro_lora_rank: 2,
            base_lora_rank: 8,
            trajectory_buffer_capacity: 1024,
            quality_threshold: 0.3,
            background_interval_secs: 3600, // 1 hour
            pattern_clusters: 100,
            ewc_lambda: 2000.0,
            ewc_max_tasks: 10,
            record_trajectories: true,
            enable_pattern_search: true,
            pattern_search_k: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Trajectory recording state
// ---------------------------------------------------------------------------

/// Simple embedding generator using hash-based projection.
/// For production, replace with actual embedding model.
#[allow(dead_code)]
fn generate_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut embedding = vec![0.0f32; dim];

    // Simple hash-based embedding (deterministic)
    for (i, word) in text.split_whitespace().enumerate() {
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        let hash = hasher.finish();

        // Distribute hash across embedding dimensions
        for j in 0..dim {
            let idx = (j + i * 7) % dim;
            let val = ((hash >> (j % 64)) & 0xFF) as f32 / 255.0;
            embedding[idx] += val * 0.1;
        }
    }

    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for e in &mut embedding {
            *e /= norm;
        }
    }

    embedding
}

/// Quality estimator based on response characteristics.
#[allow(dead_code)]
fn estimate_quality(_request: &ModelRequest, response: &ModelResponse) -> f32 {
    let mut quality = 0.5f32;

    // Longer responses often indicate more thorough answers
    let response_len = response.message.content.len();
    if response_len > 100 {
        quality += 0.1;
    }
    if response_len > 500 {
        quality += 0.1;
    }

    // Tool calls indicate structured work
    if !response.tool_calls.is_empty() {
        quality += 0.15;
    }

    // Check for error indicators
    let content_lower = response.message.content.to_lowercase();
    if content_lower.contains("error") || content_lower.contains("failed") {
        quality -= 0.2;
    }
    if content_lower.contains("sorry") || content_lower.contains("cannot") {
        quality -= 0.1;
    }

    // Success indicators
    if content_lower.contains("success") || content_lower.contains("completed") {
        quality += 0.1;
    }

    // Clamp to valid range
    quality.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// SONA Middleware State
// ---------------------------------------------------------------------------

/// Internal state for SONA middleware.
struct SonaState {
    /// SONA engine (when feature enabled).
    #[cfg(feature = "sona")]
    engine: SonaEngine,

    /// Trajectory buffer for lock-free recording.
    #[cfg(feature = "sona")]
    buffer: TrajectoryBuffer,

    /// Trajectory ID generator.
    #[cfg(feature = "sona")]
    id_gen: TrajectoryIdGen,

    /// EWC++ for catastrophic forgetting prevention.
    #[cfg(feature = "sona")]
    ewc: EwcPlusPlus,

    /// ReasoningBank for pattern extraction.
    #[cfg(feature = "sona")]
    reasoning_bank: ReasoningBank,

    /// Last background learning time.
    last_background_run: Instant,

    /// Total trajectories recorded.
    trajectories_recorded: AtomicU64,

    /// Total patterns extracted.
    patterns_extracted: AtomicU64,

    /// Background learning count.
    background_runs: AtomicU64,

    /// Consolidation count.
    consolidations: AtomicU64,

    /// Configuration.
    config: SonaMiddlewareConfig,
}

impl SonaState {
    fn new(config: SonaMiddlewareConfig) -> Self {
        #[cfg(feature = "sona")]
        let engine = {
            let sona_config = SonaConfig {
                hidden_dim: config.hidden_dim,
                embedding_dim: config.embedding_dim,
                micro_lora_rank: config.micro_lora_rank,
                base_lora_rank: config.base_lora_rank,
                trajectory_capacity: config.trajectory_buffer_capacity,
                quality_threshold: config.quality_threshold,
                ..Default::default()
            };
            SonaEngine::with_config(sona_config)
        };

        #[cfg(feature = "sona")]
        let buffer = TrajectoryBuffer::new(config.trajectory_buffer_capacity);

        #[cfg(feature = "sona")]
        let id_gen = TrajectoryIdGen::new();

        #[cfg(feature = "sona")]
        let ewc = {
            let ewc_config = EwcConfig {
                param_count: config.hidden_dim * config.micro_lora_rank,
                max_tasks: config.ewc_max_tasks,
                initial_lambda: config.ewc_lambda,
                ..Default::default()
            };
            EwcPlusPlus::new(ewc_config)
        };

        #[cfg(feature = "sona")]
        let reasoning_bank = {
            let pattern_config = PatternConfig {
                k_clusters: config.pattern_clusters,
                embedding_dim: config.embedding_dim,
                quality_threshold: config.quality_threshold,
                ..Default::default()
            };
            ReasoningBank::new(pattern_config)
        };

        Self {
            #[cfg(feature = "sona")]
            engine,
            #[cfg(feature = "sona")]
            buffer,
            #[cfg(feature = "sona")]
            id_gen,
            #[cfg(feature = "sona")]
            ewc,
            #[cfg(feature = "sona")]
            reasoning_bank,
            last_background_run: Instant::now(),
            trajectories_recorded: AtomicU64::new(0),
            patterns_extracted: AtomicU64::new(0),
            background_runs: AtomicU64::new(0),
            consolidations: AtomicU64::new(0),
            config,
        }
    }

    /// Record a trajectory from a model call.
    #[cfg(feature = "sona")]
    fn record_trajectory(
        &self,
        request: &ModelRequest,
        response: &ModelResponse,
        latency: Duration,
    ) {
        if !self.config.record_trajectories {
            return;
        }

        // Generate embedding from request
        let query_text = request
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::User))
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let query_embedding = generate_embedding(&query_text, self.config.embedding_dim);

        // Build trajectory
        let id = self.id_gen.next();
        let mut builder = TrajectoryBuilder::new(id, query_embedding);

        // Add response as a step
        let response_embedding =
            generate_embedding(&response.message.content, self.config.embedding_dim);
        let quality = estimate_quality(request, response);

        builder.add_step(response_embedding, vec![], quality);

        // Set model route if available
        if let Some(sys) = &request.system_message {
            if sys.contains("claude") {
                builder.set_model_route("claude");
            } else if sys.contains("gpt") {
                builder.set_model_route("openai");
            } else if sys.contains("gemini") {
                builder.set_model_route("google");
            }
        }

        // Build and record
        let trajectory = builder.build_with_latency(quality, latency.as_micros() as u64);

        if self.buffer.record(trajectory.clone()) {
            self.trajectories_recorded.fetch_add(1, Ordering::Relaxed);
            trace!(
                "Recorded trajectory {} with quality {:.2}",
                id,
                quality
            );

            // Also submit to engine for instant learning
            self.engine.submit_trajectory(trajectory);
        } else {
            debug!("Trajectory buffer full, dropped trajectory {}", id);
        }
    }

    /// Record trajectory (no-op when SONA feature is disabled).
    #[cfg(not(feature = "sona"))]
    fn record_trajectory(
        &self,
        _request: &ModelRequest,
        _response: &ModelResponse,
        _latency: Duration,
    ) {
        // No-op when SONA is disabled
    }

    /// Check if background learning is due.
    fn should_run_background(&self) -> bool {
        self.last_background_run.elapsed().as_secs() >= self.config.background_interval_secs
    }

    /// Run background learning (Loop B).
    #[cfg(feature = "sona")]
    fn run_background_learning(&mut self) -> usize {
        let start = Instant::now();

        // Drain trajectories from buffer
        let trajectories = self.buffer.drain();
        let trajectory_count = trajectories.len();

        if trajectory_count == 0 {
            return 0;
        }

        // Add to reasoning bank for pattern extraction
        for trajectory in &trajectories {
            self.reasoning_bank.add_trajectory(trajectory);
        }

        // Extract patterns using K-means++
        let patterns = self.reasoning_bank.extract_patterns();
        let pattern_count = patterns.len();

        // Update Fisher information in EWC++ for important parameters
        if pattern_count > 0 {
            // Compute pseudo-gradients from pattern centroids
            for pattern in &patterns {
                if pattern.centroid.len() >= self.config.hidden_dim * self.config.micro_lora_rank {
                    let gradients: Vec<f32> = pattern
                        .centroid
                        .iter()
                        .take(self.config.hidden_dim * self.config.micro_lora_rank)
                        .map(|&x| x * pattern.avg_quality)
                        .collect();
                    self.ewc.update_fisher(&gradients);
                }
            }
        }

        // Run engine tick
        if let Some(msg) = self.engine.tick() {
            debug!("SONA engine tick: {}", msg);
        }

        // Update metrics
        self.patterns_extracted
            .fetch_add(pattern_count as u64, Ordering::Relaxed);
        self.background_runs.fetch_add(1, Ordering::Relaxed);
        self.last_background_run = Instant::now();

        info!(
            "Background learning: {} trajectories -> {} patterns in {:?}",
            trajectory_count,
            pattern_count,
            start.elapsed()
        );

        pattern_count
    }

    /// Run background learning (no-op when SONA feature is disabled).
    #[cfg(not(feature = "sona"))]
    fn run_background_learning(&mut self) -> usize {
        self.last_background_run = Instant::now();
        0
    }

    /// Run consolidation (Loop C) - session-end learning.
    #[cfg(feature = "sona")]
    fn consolidate(&mut self) {
        let start = Instant::now();

        // First run any pending background learning
        self.run_background_learning();

        // Detect task boundary in EWC++
        let gradients: Vec<f32> = self
            .reasoning_bank
            .get_all_patterns()
            .iter()
            .flat_map(|p| p.centroid.iter().copied())
            .take(self.config.hidden_dim * self.config.micro_lora_rank)
            .collect();

        if !gradients.is_empty() && self.ewc.detect_task_boundary(&gradients) {
            info!("Task boundary detected, starting new EWC++ task");
            self.ewc.start_new_task();
        }

        // Prune low-quality patterns
        self.reasoning_bank.prune_patterns(
            self.config.quality_threshold,
            0,      // min accesses
            86400,  // max age (24 hours)
        );

        // Consolidate similar patterns
        self.reasoning_bank.consolidate(0.95);

        // Consolidate EWC++ tasks if we have too many
        if self.ewc.task_count() > self.config.ewc_max_tasks / 2 {
            self.ewc.consolidate_all_tasks();
        }

        // Flush instant loop updates
        self.engine.flush();

        // Force a learning cycle
        let result = self.engine.force_learn();
        debug!("Consolidation forced learning: {}", result);

        self.consolidations.fetch_add(1, Ordering::Relaxed);

        info!(
            "Session consolidation complete in {:?} (tasks: {}, patterns: {})",
            start.elapsed(),
            self.ewc.task_count(),
            self.reasoning_bank.pattern_count()
        );
    }

    /// Run consolidation (no-op when SONA feature is disabled).
    #[cfg(not(feature = "sona"))]
    fn consolidate(&mut self) {
        self.consolidations.fetch_add(1, Ordering::Relaxed);
    }

    /// Find similar patterns for context augmentation.
    #[cfg(feature = "sona")]
    fn find_similar_patterns(&self, query: &str) -> Vec<String> {
        if !self.config.enable_pattern_search {
            return vec![];
        }

        let query_embedding = generate_embedding(query, self.config.embedding_dim);
        let patterns = self
            .reasoning_bank
            .find_similar(&query_embedding, self.config.pattern_search_k);

        patterns
            .iter()
            .map(|p| {
                format!(
                    "[Pattern {} (quality: {:.2}, type: {:?})]",
                    p.id, p.avg_quality, p.pattern_type
                )
            })
            .collect()
    }

    /// Find similar patterns (returns empty when SONA feature is disabled).
    #[cfg(not(feature = "sona"))]
    fn find_similar_patterns(&self, _query: &str) -> Vec<String> {
        vec![]
    }

    /// Get statistics.
    fn stats(&self) -> SonaStats {
        SonaStats {
            trajectories_recorded: self.trajectories_recorded.load(Ordering::Relaxed),
            patterns_extracted: self.patterns_extracted.load(Ordering::Relaxed),
            background_runs: self.background_runs.load(Ordering::Relaxed),
            consolidations: self.consolidations.load(Ordering::Relaxed),
            #[cfg(feature = "sona")]
            buffer_len: self.buffer.len(),
            #[cfg(not(feature = "sona"))]
            buffer_len: 0,
            #[cfg(feature = "sona")]
            buffer_success_rate: self.buffer.success_rate(),
            #[cfg(not(feature = "sona"))]
            buffer_success_rate: 1.0,
            #[cfg(feature = "sona")]
            ewc_task_count: self.ewc.task_count(),
            #[cfg(not(feature = "sona"))]
            ewc_task_count: 0,
            #[cfg(feature = "sona")]
            pattern_count: self.reasoning_bank.pattern_count(),
            #[cfg(not(feature = "sona"))]
            pattern_count: 0,
        }
    }
}

/// SONA middleware statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonaStats {
    pub trajectories_recorded: u64,
    pub patterns_extracted: u64,
    pub background_runs: u64,
    pub consolidations: u64,
    pub buffer_len: usize,
    pub buffer_success_rate: f64,
    pub ewc_task_count: usize,
    pub pattern_count: usize,
}

// ---------------------------------------------------------------------------
// SONA Middleware
// ---------------------------------------------------------------------------

/// SONA Adaptive Learning Middleware.
///
/// Implements ADR-103 B5 with three learning loops:
/// - **Instant (Loop A)**: Records trajectories during `wrap_model_call`
/// - **Background (Loop B)**: Periodic pattern extraction
/// - **Deep (Loop C)**: Session consolidation with EWC++
pub struct SonaMiddleware {
    state: Arc<RwLock<SonaState>>,
    enabled: AtomicBool,
}

impl SonaMiddleware {
    /// Create a new SONA middleware with default configuration.
    pub fn new(config: SonaMiddlewareConfig) -> Self {
        let enabled = config.enabled;
        Self {
            state: Arc::new(RwLock::new(SonaState::new(config))),
            enabled: AtomicBool::new(enabled),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SonaMiddlewareConfig::default())
    }

    /// Enable or disable the middleware.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    /// Check if enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Get statistics.
    pub fn stats(&self) -> SonaStats {
        self.state.read().stats()
    }

    /// Force background learning cycle.
    pub fn force_background_learning(&self) -> usize {
        self.state.write().run_background_learning()
    }

    /// Force consolidation (session end).
    pub fn consolidate(&self) {
        self.state.write().consolidate();
    }

    /// Find similar patterns for a query.
    pub fn find_patterns(&self, query: &str) -> Vec<String> {
        self.state.read().find_similar_patterns(query)
    }
}

#[async_trait]
impl Middleware for SonaMiddleware {
    fn name(&self) -> &str {
        "sona"
    }

    fn before_agent(
        &self,
        state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if !self.is_enabled() {
            return None;
        }

        // Check if background learning is due
        {
            let mut sona_state = self.state.write();
            if sona_state.should_run_background() {
                sona_state.run_background_learning();
            }
        }

        // Find similar patterns for context augmentation
        let last_user_message = state
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, Role::User));

        if let Some(msg) = last_user_message {
            let patterns = self.state.read().find_similar_patterns(&msg.content);

            if !patterns.is_empty() {
                // Store patterns in extensions for potential use
                let mut extensions = std::collections::HashMap::new();
                extensions.insert(
                    "sona_patterns".to_string(),
                    serde_json::json!(patterns),
                );

                return Some(AgentStateUpdate {
                    messages: None,
                    todos: None,
                    extensions,
                });
            }
        }

        None
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        if !self.is_enabled() {
            return handler.call(request);
        }

        let start = Instant::now();

        // Call the underlying handler
        let response = handler.call(request.clone());

        // Record trajectory (Loop A - Instant Learning)
        let latency = start.elapsed();
        self.state.read().record_trajectory(&request, &response, latency);

        response
    }

    async fn awrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn AsyncModelHandler,
    ) -> ModelResponse {
        if !self.is_enabled() {
            return handler.call(request).await;
        }

        let start = Instant::now();

        // Call the underlying handler
        let response = handler.call(request.clone()).await;

        // Record trajectory (Loop A - Instant Learning)
        let latency = start.elapsed();
        self.state.read().record_trajectory(&request, &response, latency);

        response
    }
}

impl std::fmt::Debug for SonaMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SonaMiddleware")
            .field("enabled", &self.is_enabled())
            .field("stats", &self.stats())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    #[test]
    fn test_config_default() {
        let config = SonaMiddlewareConfig::default();
        assert!(config.enabled);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.micro_lora_rank, 2);
        assert_eq!(config.trajectory_buffer_capacity, 1024);
    }

    #[test]
    fn test_middleware_creation() {
        let middleware = SonaMiddleware::default_config();
        assert!(middleware.is_enabled());
        assert_eq!(middleware.name(), "sona");
    }

    #[test]
    fn test_enable_disable() {
        let middleware = SonaMiddleware::default_config();
        assert!(middleware.is_enabled());

        middleware.set_enabled(false);
        assert!(!middleware.is_enabled());

        middleware.set_enabled(true);
        assert!(middleware.is_enabled());
    }

    #[test]
    fn test_generate_embedding() {
        let text = "Hello world this is a test";
        let embedding = generate_embedding(text, 64);

        assert_eq!(embedding.len(), 64);

        // Should be normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        // Deterministic
        let embedding2 = generate_embedding(text, 64);
        assert_eq!(embedding, embedding2);
    }

    #[test]
    fn test_estimate_quality() {
        let request = ModelRequest::new(vec![Message::user("test")]);

        // Short response
        let short_response = ModelResponse::text("ok");
        let quality1 = estimate_quality(&request, &short_response);
        assert!(quality1 >= 0.0 && quality1 <= 1.0);

        // Long response
        let long_response = ModelResponse::text("This is a much longer response that contains more detailed information and should score higher for quality estimation purposes. ".repeat(10));
        let quality2 = estimate_quality(&request, &long_response);
        assert!(quality2 > quality1);

        // Error response
        let error_response = ModelResponse::text("Sorry, I cannot help with that. An error occurred.");
        let quality3 = estimate_quality(&request, &error_response);
        assert!(quality3 < quality1);
    }

    #[test]
    fn test_stats() {
        let middleware = SonaMiddleware::default_config();
        let stats = middleware.stats();

        assert_eq!(stats.trajectories_recorded, 0);
        assert_eq!(stats.patterns_extracted, 0);
        assert_eq!(stats.background_runs, 0);
        assert_eq!(stats.consolidations, 0);
    }

    #[test]
    fn test_find_patterns_disabled() {
        let config = SonaMiddlewareConfig {
            enable_pattern_search: false,
            ..Default::default()
        };
        let middleware = SonaMiddleware::new(config);

        let patterns = middleware.find_patterns("test query");
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_force_consolidation() {
        let middleware = SonaMiddleware::default_config();

        // Should not panic
        middleware.consolidate();

        let stats = middleware.stats();
        assert_eq!(stats.consolidations, 1);
    }

    struct TestHandler;
    impl ModelHandler for TestHandler {
        fn call(&self, _request: ModelRequest) -> ModelResponse {
            ModelResponse::text("Test response with some content for quality estimation")
        }
    }

    #[test]
    fn test_wrap_model_call() {
        let middleware = SonaMiddleware::default_config();
        let handler = TestHandler;
        let request = ModelRequest::new(vec![Message::user("Hello")]);

        let response = middleware.wrap_model_call(request, &handler);

        assert!(response.message.content.contains("Test response"));

        #[cfg(feature = "sona")]
        {
            let stats = middleware.stats();
            assert_eq!(stats.trajectories_recorded, 1);
        }
    }

    #[test]
    fn test_wrap_model_call_disabled() {
        let middleware = SonaMiddleware::default_config();
        middleware.set_enabled(false);

        let handler = TestHandler;
        let request = ModelRequest::new(vec![Message::user("Hello")]);

        let response = middleware.wrap_model_call(request, &handler);

        assert!(response.message.content.contains("Test response"));

        // No recording when disabled
        let stats = middleware.stats();
        assert_eq!(stats.trajectories_recorded, 0);
    }
}
