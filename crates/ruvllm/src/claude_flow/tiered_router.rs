//! Tiered Router — orchestration layer for multi-backend inference.
//!
//! Routes tasks to the optimal backend (Local/Ollama/Claude) based on
//! complexity analysis, with automatic fallback when a backend is unavailable.
//!
//! ## Fallback Chain
//!
//! ```text
//! Local fails? → Try Ollama → Ollama fails? → Try Claude → All fail? → Error
//! ```
//!
//! ## Knowledge Distillation
//!
//! Every time a higher-tier backend (Ollama or Claude) serves a request,
//! a [`DistillationEvent`] is emitted through the registered [`DistillationSink`].
//! This lets the local model learn from escalated responses:
//!
//! ```text
//! ┌─────────────────┐    fallback/escalation    ┌──────────────────┐
//! │  Local (Candle)  │ ───────────────────────> │  Ollama / Claude │
//! └────────┬────────┘                           └────────┬─────────┘
//!          │                                             │
//!          │  <── DistillationEvent (prompt, response) ──┘
//!          │
//!          ▼
//!   SONA instant loop / MicroLoRA adapt / ReasoningBank record
//! ```
//!
//! Over time the local model absorbs patterns from higher tiers,
//! reducing escalation rate and cost.
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::claude_flow::tiered_router::{TieredRouter, TieredRouterConfig};
//! use ruvllm::backends::{UnifiedInferenceBackend, UnifiedRequest, OllamaAdapter, ClaudeAdapter};
//! use ruvllm::claude_flow::model_router::InferenceTier;
//!
//! let config = TieredRouterConfig::default();
//! let mut router = TieredRouter::new(config);
//!
//! // Register backends
//! router.register(InferenceTier::Ollama, Arc::new(OllamaAdapter::with_defaults()?));
//! router.register(InferenceTier::CloudClaude, Arc::new(ClaudeAdapter::from_env(ClaudeModel::Opus)?));
//!
//! // Route and generate — picks tier automatically, emits distillation events
//! let response = router.route_and_generate("fix typo in README").await?;
//! println!("Backend: {}, Response: {}", response.backend_name, response.text);
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::backends::unified_backend::{
    UnifiedInferenceBackend, UnifiedRequest, UnifiedResponse, UnifiedStreamToken,
};
use crate::error::{Result, RuvLLMError};

use super::model_router::{InferenceTier, TaskComplexityAnalyzer};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the tiered router
#[derive(Debug, Clone)]
pub struct TieredRouterConfig {
    /// Complexity threshold below which tasks go to local inference
    pub local_threshold: f32,
    /// Complexity threshold above which tasks go to Claude API
    pub claude_threshold: f32,
    /// Token limit for local inference
    pub local_token_limit: usize,
    /// Token limit for Ollama
    pub ollama_token_limit: usize,
    /// Whether to enable automatic fallback
    pub enable_fallback: bool,
    /// Maximum fallback attempts
    pub max_fallback_attempts: usize,
    /// Enable knowledge distillation from higher-tier responses
    pub enable_distillation: bool,
    /// Minimum quality score to distill (skip low-quality responses)
    pub distillation_quality_threshold: f32,
}

impl Default for TieredRouterConfig {
    fn default() -> Self {
        Self {
            local_threshold: 0.35,
            claude_threshold: 0.70,
            local_token_limit: 500,
            ollama_token_limit: 2000,
            enable_fallback: true,
            max_fallback_attempts: 3,
            enable_distillation: true,
            distillation_quality_threshold: 0.5,
        }
    }
}

// ============================================================================
// Knowledge Distillation
// ============================================================================

/// Event emitted when a higher-tier backend serves a request.
///
/// Contains the full (prompt, response) pair plus routing metadata so the
/// local model can learn from escalated responses.
#[derive(Debug, Clone)]
pub struct DistillationEvent {
    /// The original task/prompt text
    pub prompt: String,
    /// The generated response text from the higher-tier backend
    pub response: String,
    /// The tier that was originally recommended (often Local)
    pub recommended_tier: InferenceTier,
    /// The tier that actually served the request
    pub actual_tier: InferenceTier,
    /// Whether this was a fallback (recommended tier failed) vs. direct routing
    pub was_fallback: bool,
    /// Complexity score from the analyzer
    pub complexity: f32,
    /// Response quality estimate (output_tokens / latency as a proxy, 0.0-1.0)
    pub quality_estimate: f32,
    /// Input token count from the response
    pub input_tokens: usize,
    /// Output token count from the response
    pub output_tokens: usize,
    /// Total latency in milliseconds
    pub latency_ms: u64,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Trait for receiving distillation events from the tiered router.
///
/// Implement this to wire escalation data into your learning pipeline
/// (SONA, MicroLoRA, ReasoningBank, etc.). The sink is called asynchronously
/// after each successful higher-tier response.
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::claude_flow::tiered_router::{DistillationSink, DistillationEvent};
/// use ruvllm::sona::{SonaIntegration, Trajectory};
///
/// struct SonaDistillationSink {
///     sona: Arc<SonaIntegration>,
/// }
///
/// #[async_trait::async_trait]
/// impl DistillationSink for SonaDistillationSink {
///     async fn on_distillation(&self, event: DistillationEvent) {
///         // Convert event to SONA trajectory and record
///         let trajectory = Trajectory { ... };
///         let _ = self.sona.record_trajectory(trajectory);
///     }
/// }
/// ```
#[async_trait::async_trait]
pub trait DistillationSink: Send + Sync {
    /// Called when a higher-tier response is available for distillation.
    ///
    /// This is fire-and-forget — errors are logged but don't fail the request.
    async fn on_distillation(&self, event: DistillationEvent);
}

/// A no-op sink that just logs distillation events.
pub struct LoggingDistillationSink;

#[async_trait::async_trait]
impl DistillationSink for LoggingDistillationSink {
    async fn on_distillation(&self, event: DistillationEvent) {
        tracing::info!(
            recommended = ?event.recommended_tier,
            actual = ?event.actual_tier,
            fallback = event.was_fallback,
            complexity = event.complexity,
            quality = event.quality_estimate,
            output_tokens = event.output_tokens,
            latency_ms = event.latency_ms,
            "Distillation event: {} -> {} (prompt: {}...)",
            event.recommended_tier.name(),
            event.actual_tier.name(),
            &event.prompt[..event.prompt.len().min(60)],
        );
    }
}

/// A sink that collects events in-memory for testing or batch processing.
pub struct BufferedDistillationSink {
    events: Arc<tokio::sync::Mutex<Vec<DistillationEvent>>>,
}

impl BufferedDistillationSink {
    /// Create a new buffered sink
    pub fn new() -> Self {
        Self {
            events: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        }
    }

    /// Drain all buffered events
    pub async fn drain(&self) -> Vec<DistillationEvent> {
        let mut events = self.events.lock().await;
        std::mem::take(&mut *events)
    }

    /// Get the number of buffered events
    pub async fn len(&self) -> usize {
        self.events.lock().await.len()
    }
}

#[async_trait::async_trait]
impl DistillationSink for BufferedDistillationSink {
    async fn on_distillation(&self, event: DistillationEvent) {
        self.events.lock().await.push(event);
    }
}

// ============================================================================
// Routing History
// ============================================================================

/// Record of a routing decision and its outcome
#[derive(Debug, Clone)]
pub struct RoutingRecord {
    /// Task description (truncated)
    pub task_summary: String,
    /// Tier that was selected
    pub selected_tier: InferenceTier,
    /// Tier that actually served the request (may differ due to fallback)
    pub actual_tier: InferenceTier,
    /// Whether a fallback occurred
    pub fallback_used: bool,
    /// Number of fallback attempts
    pub fallback_count: usize,
    /// Total latency in milliseconds
    pub total_ms: u64,
    /// Whether the request succeeded
    pub success: bool,
    /// Complexity score
    pub complexity: f32,
    /// Whether a distillation event was emitted
    pub distilled: bool,
    /// Timestamp
    pub timestamp: Instant,
}

/// Statistics for the tiered router
#[derive(Debug, Clone, Default)]
pub struct TieredRouterStats {
    /// Total requests routed
    pub total_requests: u64,
    /// Requests by tier
    pub requests_by_tier: HashMap<InferenceTier, u64>,
    /// Fallback count
    pub fallback_count: u64,
    /// Success count
    pub success_count: u64,
    /// Distillation events emitted
    pub distillation_count: u64,
    /// Average latency per tier (ms)
    pub avg_latency_by_tier: HashMap<InferenceTier, f64>,
}

// ============================================================================
// Tiered Router
// ============================================================================

/// Orchestrates inference across Local, Ollama, and Claude backends.
///
/// Uses the existing `TaskComplexityAnalyzer` to score task complexity,
/// maps that to an `InferenceTier`, and dispatches to the registered
/// backend. If the chosen backend fails, falls back through the chain.
///
/// When a request is served by a tier higher than Local, a
/// [`DistillationEvent`] is emitted through the registered
/// [`DistillationSink`] so the local model can learn from the response.
pub struct TieredRouter {
    /// Configuration
    config: TieredRouterConfig,
    /// Registered backends by tier
    backends: HashMap<InferenceTier, Arc<dyn UnifiedInferenceBackend>>,
    /// Complexity analyzer (reused from model_router)
    analyzer: TaskComplexityAnalyzer,
    /// Optional distillation sink for knowledge transfer
    distillation_sink: Option<Arc<dyn DistillationSink>>,
    /// Routing history for learning
    history: Vec<RoutingRecord>,
    /// Request counter
    request_count: u64,
}

impl TieredRouter {
    /// Create a new tiered router with the given configuration
    pub fn new(config: TieredRouterConfig) -> Self {
        Self {
            config,
            backends: HashMap::new(),
            analyzer: TaskComplexityAnalyzer::new(),
            distillation_sink: None,
            history: Vec::new(),
            request_count: 0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(TieredRouterConfig::default())
    }

    /// Register a backend for a specific tier
    pub fn register(&mut self, tier: InferenceTier, backend: Arc<dyn UnifiedInferenceBackend>) {
        self.backends.insert(tier, backend);
    }

    /// Remove a backend for a specific tier
    pub fn unregister(&mut self, tier: InferenceTier) {
        self.backends.remove(&tier);
    }

    /// Set the distillation sink for knowledge transfer from higher tiers.
    ///
    /// When set, every successful response from Ollama or Claude will emit
    /// a [`DistillationEvent`] through this sink, allowing the local model
    /// to learn from escalated responses.
    pub fn set_distillation_sink(&mut self, sink: Arc<dyn DistillationSink>) {
        self.distillation_sink = Some(sink);
    }

    /// Remove the distillation sink
    pub fn clear_distillation_sink(&mut self) {
        self.distillation_sink = None;
    }

    /// Check which tiers have registered backends
    pub fn registered_tiers(&self) -> Vec<InferenceTier> {
        self.backends.keys().copied().collect()
    }

    /// Check if a specific tier is registered and available
    pub async fn is_tier_available(&self, tier: InferenceTier) -> bool {
        match self.backends.get(&tier) {
            Some(backend) => backend.is_available().await,
            None => false,
        }
    }

    /// Analyze task complexity and determine the target tier.
    ///
    /// This is a pure routing decision without executing the request.
    pub fn analyze_tier(&mut self, task: &str) -> (InferenceTier, f32) {
        let score = self.analyzer.analyze(task);
        let tier = score.recommended_inference_tier();
        (tier, score.overall)
    }

    /// Route a task to the best available backend and generate a response.
    ///
    /// Analyzes complexity, picks a tier, and executes with fallback.
    /// Emits a distillation event if served by a tier above Local.
    pub async fn route_and_generate(&mut self, task: &str) -> Result<UnifiedResponse> {
        let request = UnifiedRequest::new(task);
        self.route_and_generate_request(task, &request).await
    }

    /// Route with a custom UnifiedRequest (allows setting temperature, max_tokens, etc.)
    pub async fn route_and_generate_request(
        &mut self,
        task: &str,
        request: &UnifiedRequest,
    ) -> Result<UnifiedResponse> {
        self.request_count += 1;
        let start = Instant::now();

        // Analyze complexity and pick tier
        let score = self.analyzer.analyze(task);
        let initial_tier = score.recommended_inference_tier();
        let complexity = score.overall;

        // Try the selected tier, with fallback chain
        let mut current_tier = initial_tier;
        let mut fallback_count = 0;
        let mut last_error = None;

        loop {
            // Check if we have a backend for this tier
            if let Some(backend) = self.backends.get(&current_tier) {
                // Check availability
                if backend.is_available().await {
                    match backend.generate(request).await {
                        Ok(response) => {
                            let total_ms = start.elapsed().as_millis() as u64;
                            let was_fallback = fallback_count > 0;

                            // Emit distillation event if served by higher tier
                            let distilled = self
                                .maybe_distill(
                                    task,
                                    &response,
                                    initial_tier,
                                    current_tier,
                                    was_fallback,
                                    complexity,
                                    total_ms,
                                )
                                .await;

                            self.record_routing(
                                task,
                                initial_tier,
                                current_tier,
                                fallback_count,
                                total_ms,
                                true,
                                complexity,
                                distilled,
                            );
                            return Ok(response);
                        }
                        Err(e) => {
                            tracing::warn!(
                                tier = ?current_tier,
                                error = %e,
                                "Backend failed, attempting fallback"
                            );
                            last_error = Some(e);
                        }
                    }
                } else {
                    tracing::info!(tier = ?current_tier, "Backend not available, skipping");
                }
            }

            // Fallback
            if !self.config.enable_fallback
                || fallback_count >= self.config.max_fallback_attempts
            {
                break;
            }

            match current_tier.fallback() {
                Some(next_tier) => {
                    tracing::info!(
                        from = ?current_tier,
                        to = ?next_tier,
                        "Falling back to next tier"
                    );
                    current_tier = next_tier;
                    fallback_count += 1;
                }
                None => break, // No more fallback options
            }
        }

        // Record failure
        self.record_routing(
            task,
            initial_tier,
            current_tier,
            fallback_count,
            start.elapsed().as_millis() as u64,
            false,
            complexity,
            false,
        );

        // Return the last error, or a generic one
        match last_error {
            Some(e) => Err(e),
            None => Err(RuvLLMError::Backend(format!(
                "No available backend for tier {:?} (registered: {:?})",
                initial_tier,
                self.registered_tiers()
            ))),
        }
    }

    /// Route and generate with streaming output.
    ///
    /// Returns a channel receiver that yields tokens from whichever backend
    /// handles the request.
    ///
    /// Note: distillation for streaming requires the caller to collect
    /// the full response and call [`record_stream_distillation`] manually,
    /// since we can't buffer the full stream without breaking the streaming contract.
    pub async fn route_and_stream(
        &mut self,
        task: &str,
        request: &UnifiedRequest,
    ) -> Result<(
        tokio::sync::mpsc::Receiver<Result<UnifiedStreamToken>>,
        StreamDistillationContext,
    )> {
        self.request_count += 1;
        let start = Instant::now();

        let score = self.analyzer.analyze(task);
        let initial_tier = score.recommended_inference_tier();
        let complexity = score.overall;

        let mut current_tier = initial_tier;
        let mut fallback_count = 0;
        let mut last_error = None;

        loop {
            if let Some(backend) = self.backends.get(&current_tier) {
                if backend.is_available().await {
                    match backend.generate_stream(request).await {
                        Ok(rx) => {
                            self.record_routing(
                                task,
                                initial_tier,
                                current_tier,
                                fallback_count,
                                start.elapsed().as_millis() as u64,
                                true,
                                complexity,
                                false, // distillation happens after stream completes
                            );

                            let ctx = StreamDistillationContext {
                                prompt: task.to_string(),
                                recommended_tier: initial_tier,
                                actual_tier: current_tier,
                                was_fallback: fallback_count > 0,
                                complexity,
                                start_time: start,
                            };

                            return Ok((rx, ctx));
                        }
                        Err(e) => {
                            tracing::warn!(
                                tier = ?current_tier,
                                error = %e,
                                "Stream backend failed, attempting fallback"
                            );
                            last_error = Some(e);
                        }
                    }
                }
            }

            if !self.config.enable_fallback
                || fallback_count >= self.config.max_fallback_attempts
            {
                break;
            }

            match current_tier.fallback() {
                Some(next_tier) => {
                    current_tier = next_tier;
                    fallback_count += 1;
                }
                None => break,
            }
        }

        self.record_routing(
            task,
            initial_tier,
            current_tier,
            fallback_count,
            start.elapsed().as_millis() as u64,
            false,
            complexity,
            false,
        );

        match last_error {
            Some(e) => Err(e),
            None => Err(RuvLLMError::Backend(format!(
                "No available streaming backend for tier {:?}",
                initial_tier,
            ))),
        }
    }

    /// Complete distillation for a streaming response.
    ///
    /// Call this after collecting the full streamed response text.
    /// Emits a [`DistillationEvent`] if the response came from a higher tier.
    pub async fn record_stream_distillation(
        &self,
        ctx: &StreamDistillationContext,
        full_response: &str,
        output_tokens: usize,
    ) {
        if !self.should_distill(ctx.actual_tier) {
            return;
        }

        let total_ms = ctx.start_time.elapsed().as_millis() as u64;
        let quality = estimate_quality(output_tokens, total_ms);

        if quality < self.config.distillation_quality_threshold {
            return;
        }

        let event = DistillationEvent {
            prompt: ctx.prompt.clone(),
            response: full_response.to_string(),
            recommended_tier: ctx.recommended_tier,
            actual_tier: ctx.actual_tier,
            was_fallback: ctx.was_fallback,
            complexity: ctx.complexity,
            quality_estimate: quality,
            input_tokens: ctx.prompt.len() / 4,
            output_tokens,
            latency_ms: total_ms,
            timestamp: std::time::SystemTime::now(),
        };

        if let Some(sink) = &self.distillation_sink {
            sink.on_distillation(event).await;
        }
    }

    /// Record the outcome of a completed request for feedback learning.
    ///
    /// Call this after evaluating the quality of a response to improve
    /// future routing decisions.
    pub fn record_outcome(&mut self, success: bool) {
        if let Some(record) = self.history.last_mut() {
            record.success = success;
        }
        // Forward feedback to the complexity analyzer
        self.analyzer.record_feedback(
            self.history
                .last()
                .map(|r| r.complexity)
                .unwrap_or(0.5),
            if success { 1.0 } else { 0.0 },
            super::claude_integration::ClaudeModel::Sonnet,
        );
    }

    /// Get routing statistics
    pub fn stats(&self) -> TieredRouterStats {
        let mut requests_by_tier: HashMap<InferenceTier, u64> = HashMap::new();
        let mut latency_sums: HashMap<InferenceTier, (f64, u64)> = HashMap::new();
        let mut fallback_count = 0u64;
        let mut success_count = 0u64;
        let mut distillation_count = 0u64;

        for record in &self.history {
            *requests_by_tier.entry(record.actual_tier).or_insert(0) += 1;
            let entry = latency_sums
                .entry(record.actual_tier)
                .or_insert((0.0, 0));
            entry.0 += record.total_ms as f64;
            entry.1 += 1;
            if record.fallback_used {
                fallback_count += 1;
            }
            if record.success {
                success_count += 1;
            }
            if record.distilled {
                distillation_count += 1;
            }
        }

        let avg_latency_by_tier = latency_sums
            .into_iter()
            .map(|(tier, (sum, count))| (tier, sum / count as f64))
            .collect();

        TieredRouterStats {
            total_requests: self.request_count,
            requests_by_tier,
            fallback_count,
            success_count,
            distillation_count,
            avg_latency_by_tier,
        }
    }

    /// Get routing history
    pub fn history(&self) -> &[RoutingRecord] {
        &self.history
    }

    /// Clear routing history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get the configuration
    pub fn config(&self) -> &TieredRouterConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: TieredRouterConfig) {
        self.config = config;
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Check if we should emit a distillation event for this tier
    fn should_distill(&self, actual_tier: InferenceTier) -> bool {
        self.config.enable_distillation
            && self.distillation_sink.is_some()
            && actual_tier != InferenceTier::Local
    }

    /// Emit a distillation event if the response came from a higher tier.
    /// Returns true if an event was emitted.
    async fn maybe_distill(
        &self,
        prompt: &str,
        response: &UnifiedResponse,
        recommended_tier: InferenceTier,
        actual_tier: InferenceTier,
        was_fallback: bool,
        complexity: f32,
        total_ms: u64,
    ) -> bool {
        if !self.should_distill(actual_tier) {
            return false;
        }

        let quality = estimate_quality(response.output_tokens, total_ms);

        if quality < self.config.distillation_quality_threshold {
            tracing::debug!(
                quality = quality,
                threshold = self.config.distillation_quality_threshold,
                "Skipping distillation: quality below threshold"
            );
            return false;
        }

        let event = DistillationEvent {
            prompt: prompt.to_string(),
            response: response.text.clone(),
            recommended_tier,
            actual_tier,
            was_fallback,
            complexity,
            quality_estimate: quality,
            input_tokens: response.input_tokens,
            output_tokens: response.output_tokens,
            latency_ms: total_ms,
            timestamp: std::time::SystemTime::now(),
        };

        if let Some(sink) = &self.distillation_sink {
            sink.on_distillation(event).await;
        }

        true
    }

    /// Record a routing event in history
    fn record_routing(
        &mut self,
        task: &str,
        selected_tier: InferenceTier,
        actual_tier: InferenceTier,
        fallback_count: usize,
        total_ms: u64,
        success: bool,
        complexity: f32,
        distilled: bool,
    ) {
        let summary = if task.len() > 100 {
            format!("{}...", &task[..97])
        } else {
            task.to_string()
        };

        self.history.push(RoutingRecord {
            task_summary: summary,
            selected_tier,
            actual_tier,
            fallback_used: fallback_count > 0,
            fallback_count,
            total_ms,
            success,
            complexity,
            distilled,
            timestamp: Instant::now(),
        });

        // Bound history size
        if self.history.len() > 10_000 {
            self.history.drain(..5_000);
        }
    }
}

/// Context for deferred distillation of streaming responses.
///
/// Returned by [`TieredRouter::route_and_stream`]. After collecting
/// the full streamed response, pass this to
/// [`TieredRouter::record_stream_distillation`].
#[derive(Debug, Clone)]
pub struct StreamDistillationContext {
    /// Original prompt
    pub prompt: String,
    /// Recommended tier from complexity analysis
    pub recommended_tier: InferenceTier,
    /// Tier that actually served the stream
    pub actual_tier: InferenceTier,
    /// Whether fallback was used
    pub was_fallback: bool,
    /// Complexity score
    pub complexity: f32,
    /// When the stream started (for latency calculation)
    pub start_time: Instant,
}

/// Estimate response quality from output metrics.
///
/// Uses output_tokens / latency as a proxy: a backend that produces
/// substantial output quickly is likely producing quality content.
/// Returns a value in [0.0, 1.0].
fn estimate_quality(output_tokens: usize, latency_ms: u64) -> f32 {
    if output_tokens == 0 || latency_ms == 0 {
        return 0.0;
    }
    // Tokens per second as quality proxy (capped at 1.0)
    // 50+ tok/s = 1.0, 0 tok/s = 0.0
    let tps = output_tokens as f64 / (latency_ms as f64 / 1000.0);
    (tps / 50.0).min(1.0) as f32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock backend for testing
    struct MockBackend {
        name: String,
        available: bool,
        response_text: String,
        should_fail: bool,
    }

    impl MockBackend {
        fn new(name: &str, available: bool, response_text: &str) -> Self {
            Self {
                name: name.to_string(),
                available,
                response_text: response_text.to_string(),
                should_fail: false,
            }
        }

        fn failing(name: &str) -> Self {
            Self {
                name: name.to_string(),
                available: true,
                response_text: String::new(),
                should_fail: true,
            }
        }
    }

    #[async_trait::async_trait]
    impl UnifiedInferenceBackend for MockBackend {
        fn name(&self) -> &str {
            &self.name
        }

        async fn is_available(&self) -> bool {
            self.available
        }

        async fn generate(&self, _request: &UnifiedRequest) -> Result<UnifiedResponse> {
            if self.should_fail {
                return Err(RuvLLMError::Backend("Mock failure".to_string()));
            }
            Ok(UnifiedResponse {
                text: self.response_text.clone(),
                input_tokens: 10,
                output_tokens: 20,
                ttft_ms: 50,
                total_ms: 100,
                backend_name: self.name.clone(),
            })
        }
    }

    #[test]
    fn test_tiered_router_config_defaults() {
        let config = TieredRouterConfig::default();
        assert_eq!(config.local_threshold, 0.35);
        assert_eq!(config.claude_threshold, 0.70);
        assert!(config.enable_fallback);
        assert!(config.enable_distillation);
        assert_eq!(config.distillation_quality_threshold, 0.5);
    }

    #[test]
    fn test_tier_analysis() {
        let mut router = TieredRouter::with_defaults();

        let (tier, _) = router.analyze_tier("fix typo");
        assert_eq!(tier, InferenceTier::Local);

        let (tier, _) = router.analyze_tier(
            "Design a distributed architecture for concurrent microservices with \
             security audit for vulnerabilities, cryptography, and performance optimization",
        );
        assert_eq!(tier, InferenceTier::CloudClaude);
    }

    #[tokio::test]
    async fn test_route_to_available_backend() {
        let mut router = TieredRouter::with_defaults();

        // Register only Ollama tier
        router.register(
            InferenceTier::Ollama,
            Arc::new(MockBackend::new("ollama", true, "hello from ollama")),
        );

        // Even though "fix typo" would route to Local, it should fallback to Ollama
        let result = router.route_and_generate("fix typo").await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.text, "hello from ollama");
    }

    #[tokio::test]
    async fn test_fallback_chain() {
        let mut router = TieredRouter::with_defaults();

        // Local fails, Ollama not registered, Claude succeeds
        router.register(
            InferenceTier::Local,
            Arc::new(MockBackend::failing("local")),
        );
        router.register(
            InferenceTier::CloudClaude,
            Arc::new(MockBackend::new("claude", true, "hello from claude")),
        );

        let result = router.route_and_generate("fix typo").await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.text, "hello from claude");

        // Verify fallback was recorded
        let stats = router.stats();
        assert!(stats.fallback_count > 0);
    }

    #[tokio::test]
    async fn test_all_backends_unavailable() {
        let mut router = TieredRouter::with_defaults();

        router.register(
            InferenceTier::Local,
            Arc::new(MockBackend::new("local", false, "")),
        );
        router.register(
            InferenceTier::Ollama,
            Arc::new(MockBackend::new("ollama", false, "")),
        );
        router.register(
            InferenceTier::CloudClaude,
            Arc::new(MockBackend::new("claude", false, "")),
        );

        let result = router.route_and_generate("fix typo").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_no_fallback_when_disabled() {
        let config = TieredRouterConfig {
            enable_fallback: false,
            ..Default::default()
        };
        let mut router = TieredRouter::new(config);

        router.register(
            InferenceTier::Local,
            Arc::new(MockBackend::failing("local")),
        );
        router.register(
            InferenceTier::CloudClaude,
            Arc::new(MockBackend::new("claude", true, "from claude")),
        );

        // Should fail because fallback is disabled
        let result = router.route_and_generate("fix typo").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_direct_tier_routing() {
        let mut router = TieredRouter::with_defaults();

        router.register(
            InferenceTier::Local,
            Arc::new(MockBackend::new("local", true, "local response")),
        );
        router.register(
            InferenceTier::Ollama,
            Arc::new(MockBackend::new("ollama", true, "ollama response")),
        );
        router.register(
            InferenceTier::CloudClaude,
            Arc::new(MockBackend::new("claude", true, "claude response")),
        );

        // Simple task → Local
        let resp = router.route_and_generate("fix typo").await.unwrap();
        assert_eq!(resp.backend_name, "local");

        // Complex task → CloudClaude
        let resp = router
            .route_and_generate(
                "Design a distributed architecture for concurrent microservices with \
                 security audit for vulnerabilities, cryptography, performance optimization, \
                 machine learning pipeline, and scalability planning",
            )
            .await
            .unwrap();
        assert_eq!(resp.backend_name, "claude");
    }

    #[tokio::test]
    async fn test_distillation_fires_on_escalation() {
        let config = TieredRouterConfig {
            distillation_quality_threshold: 0.0, // mock completes in ~0ms
            ..Default::default()
        };
        let mut router = TieredRouter::new(config);
        let sink = Arc::new(BufferedDistillationSink::new());
        router.set_distillation_sink(sink.clone());

        // Register only Ollama (no Local), so simple tasks escalate
        router.register(
            InferenceTier::Ollama,
            Arc::new(MockBackend::new("ollama", true, "ollama handled it")),
        );

        // Simple task that would go to Local, but falls back to Ollama
        let _ = router.route_and_generate("fix typo").await.unwrap();

        // Should have emitted a distillation event
        let events = sink.drain().await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].actual_tier, InferenceTier::Ollama);
        assert_eq!(events[0].recommended_tier, InferenceTier::Local);
        assert!(events[0].was_fallback);
        assert_eq!(events[0].prompt, "fix typo");
        assert_eq!(events[0].response, "ollama handled it");
    }

    #[tokio::test]
    async fn test_no_distillation_for_local() {
        let mut router = TieredRouter::with_defaults();
        let sink = Arc::new(BufferedDistillationSink::new());
        router.set_distillation_sink(sink.clone());

        router.register(
            InferenceTier::Local,
            Arc::new(MockBackend::new("local", true, "local did it")),
        );

        // Simple task served by Local — no distillation needed
        let _ = router.route_and_generate("fix typo").await.unwrap();

        let events = sink.drain().await;
        assert_eq!(events.len(), 0);
    }

    #[tokio::test]
    async fn test_distillation_disabled() {
        let config = TieredRouterConfig {
            enable_distillation: false,
            ..Default::default()
        };
        let mut router = TieredRouter::new(config);
        let sink = Arc::new(BufferedDistillationSink::new());
        router.set_distillation_sink(sink.clone());

        router.register(
            InferenceTier::Ollama,
            Arc::new(MockBackend::new("ollama", true, "response")),
        );

        let _ = router.route_and_generate("fix typo").await.unwrap();

        // Distillation is disabled, so no events
        let events = sink.drain().await;
        assert_eq!(events.len(), 0);
    }

    #[tokio::test]
    async fn test_distillation_stats() {
        let config = TieredRouterConfig {
            distillation_quality_threshold: 0.0, // mock completes in ~0ms
            ..Default::default()
        };
        let mut router = TieredRouter::new(config);
        let sink = Arc::new(BufferedDistillationSink::new());
        router.set_distillation_sink(sink.clone());

        router.register(
            InferenceTier::Ollama,
            Arc::new(MockBackend::new("ollama", true, "response")),
        );

        let _ = router.route_and_generate("fix typo").await.unwrap();
        let _ = router.route_and_generate("another typo fix").await.unwrap();

        let stats = router.stats();
        assert_eq!(stats.distillation_count, 2);
    }

    #[test]
    fn test_routing_stats() {
        let router = TieredRouter::with_defaults();
        let stats = router.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.fallback_count, 0);
        assert_eq!(stats.distillation_count, 0);
    }

    #[test]
    fn test_estimate_quality() {
        // 0 tokens = 0 quality
        assert_eq!(estimate_quality(0, 100), 0.0);
        // 0 latency = 0 quality
        assert_eq!(estimate_quality(100, 0), 0.0);
        // 50 tokens in 1 second = 50 tps = 1.0
        assert_eq!(estimate_quality(50, 1000), 1.0);
        // 25 tokens in 1 second = 25 tps = 0.5
        assert_eq!(estimate_quality(25, 1000), 0.5);
        // Very fast = capped at 1.0
        assert_eq!(estimate_quality(1000, 100), 1.0);
    }
}
