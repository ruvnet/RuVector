//! # RuvLLM - LLM Serving Runtime with Ruvector Integration
//!
//! RuvLLM is an edge-focused LLM serving runtime designed for portable, high-performance
//! inference across heterogeneous hardware. It integrates with Ruvector for intelligent
//! memory capabilities, enabling continuous self-improvement through SONA learning.
//!
//! ## Architecture
//!
//! RuvLLM uses Ruvector as a unified memory layer with three distinct roles:
//!
//! - **Policy Memory Store**: Learned thresholds and parameters for runtime decisions
//! - **Session State Index**: Multi-turn conversation state with KV cache references
//! - **Witness Log Index**: Audit logging with semantic search capabilities
//!
//! ## Key Components
//!
//! - [`PagedAttention`]: Memory-efficient attention mechanism with page tables
//! - [`TwoTierKvCache`]: FP16 tail + quantized store for optimal memory/quality tradeoff
//! - [`AdapterManager`]: LoRA adapter loading and hot-swapping
//! - [`SessionManager`]: Session lifecycle and state management
//! - [`PolicyStore`]: Ruvector-backed policy storage with semantic search
//! - [`WitnessLog`]: Audit logging with HNSW-indexed semantic search
//! - [`SonaIntegration`]: Three-tier learning loop integration
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::{RuvLLMConfig, RuvLLMEngine};
//!
//! // Create engine with default configuration
//! let config = RuvLLMConfig::default();
//! let engine = RuvLLMEngine::new(config)?;
//!
//! // Create a session
//! let session = engine.create_session("user-123")?;
//!
//! // Process a request
//! let response = engine.process(&session, "Hello, world!")?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod adapter_manager;
pub mod autodetect;
pub mod backends;
pub mod capabilities;
pub mod claude_flow;
pub mod context;
pub mod error;
pub mod evaluation;
pub mod gguf;
pub mod hub;
pub mod kernels;
pub mod kv_cache;
pub mod lora;
pub mod memory_pool;
#[cfg(all(target_os = "macos", feature = "metal-compute"))]
pub mod metal;
pub mod models;
pub mod optimization;
pub mod paged_attention;
pub mod policy_store;
pub mod quality;
pub mod quantize;
pub mod reasoning_bank;
pub mod reflection;
pub mod ruvector_integration;
pub mod serving;
pub mod session;
pub mod session_index;
pub mod sona;
pub mod speculative;
pub mod tokenizer;
pub mod training;
pub mod types;
pub mod witness_log;

// Test modules
#[cfg(test)]
mod tests;

// Re-exports
pub use adapter_manager::{AdapterManager, LoraAdapter, AdapterConfig};
pub use autodetect::{
    SystemCapabilities, Platform, Architecture, CpuFeatures,
    GpuCapabilities, GpuBackend, CoreInfo, ComputeBackend,
    InferenceConfig,
};
pub use lora::{
    MicroLoRA, MicroLoraConfig, TargetModule, AdaptFeedback,
    AdapterRegistry, AdapterPool, AdapterComposer, CompositionStrategy,
    TrainingPipeline, TrainingConfig, EwcRegularizer, LearningRateSchedule,
};
pub use backends::{
    create_backend, DeviceType, DType, GenerateParams, GeneratedToken, LlmBackend,
    ModelArchitecture, ModelConfig, ModelInfo, Quantization, SharedBackend, SpecialTokens,
    StreamEvent, TokenStream, Tokenizer,
};
#[cfg(feature = "candle")]
pub use backends::CandleBackend;
#[cfg(feature = "async-runtime")]
pub use backends::{AsyncTokenStream, LlmBackendAsync};
pub use error::{RuvLLMError, Result};
pub use kv_cache::{
    TwoTierKvCache, KvCacheConfig, CacheTier, CacheQuantization, KvCacheStats,
    PooledKvCache, PooledKvBlock, PooledKvCacheStats,
};
pub use memory_pool::{
    InferenceArena, ArenaStats,
    BufferPool, BufferSize, PooledBuffer, BufferPoolStats,
    ScratchSpaceManager, ScratchSpace, ScratchStats,
    MemoryManager, MemoryManagerConfig, MemoryManagerStats,
    CACHE_LINE_SIZE, DEFAULT_ALIGNMENT,
};
pub use paged_attention::{PagedAttention, PagedAttentionConfig, PageTable, PageBlock};
pub use policy_store::{PolicyStore, PolicyEntry, PolicyType, QuantizationPolicy, RouterPolicy};
pub use session::{SessionManager, Session, SessionConfig};
pub use session_index::{SessionIndex, SessionState, KvCacheReference};
pub use sona::{SonaIntegration, SonaConfig, LearningLoop};
pub use claude_flow::{
    ClaudeFlowAgent, ClaudeFlowTask,
    AgentRouter, AgentType, RoutingDecision as AgentRoutingDecision,
    TaskClassifier, TaskType, ClassificationResult,
    FlowOptimizer, OptimizationConfig, OptimizationResult,
    // HNSW semantic router (150x faster pattern search)
    HnswRouter, HnswRouterConfig, HnswRouterStats, HnswRoutingResult,
    HnswDistanceMetric, TaskPattern, HybridRouter,
    // Claude API Integration (NEW)
    ClaudeModel, MessageRole, ContentBlock, Message, ClaudeRequest, ClaudeResponse, UsageStats,
    StreamToken, StreamEvent as ClaudeStreamEvent, QualityMonitor, ResponseStreamer, StreamStats,
    ContextWindow, ContextManager,
    AgentState, AgentContext, WorkflowStep, WorkflowResult, StepResult,
    AgentCoordinator, CoordinatorStats,
    CostEstimator, LatencyTracker, LatencySample, LatencyStats as ClaudeLatencyStats,
    // Model Router (NEW) - Intelligent routing to Haiku/Sonnet/Opus
    ComplexityFactors, ComplexityWeights, ComplexityScore,
    TaskComplexityAnalyzer, AnalyzerStats as ModelAnalyzerStats,
    SelectionCriteria, ModelRoutingDecision, ModelSelector, SelectorStats,
    ModelRouter,
    // Hooks Integration (NEW v2.3) - Unified Claude Flow hooks interface
    HooksIntegration, HooksConfig,
    PreTaskInput, PreTaskResult, PostTaskInput, PostTaskResult,
    PreEditInput, PreEditResult, PostEditInput, PostEditResult,
    SessionState as HooksSessionState, SessionEndResult, SessionMetrics,
    PatternMatch, QualityAssessment, LearningMetrics,
};
pub use optimization::{
    InferenceMetrics, MetricsCollector, MetricsSnapshot, MovingAverage, LatencyHistogram,
    RealtimeOptimizer, RealtimeConfig, BatchSizeStrategy, KvCachePressurePolicy,
    TokenBudgetAllocation, SpeculativeConfig, OptimizationDecision,
    SonaLlm, SonaLlmConfig, TrainingSample, AdaptationResult, LearningLoopStats,
    ConsolidationStrategy, OptimizationTrigger,
};
pub use tokenizer::{
    RuvTokenizer, ChatMessage, ChatTemplate, Role, TokenizerSpecialTokens,
    StreamingDecodeBuffer,
};
pub use speculative::{
    SpeculativeDecoder, SpeculativeConfig as SpeculativeDecodingConfig,
    SpeculativeStats, AtomicSpeculativeStats, VerificationResult,
    SpeculationTree, TreeNode,
    softmax, log_softmax, sample_from_probs, top_k_filter, top_p_filter,
};
pub use types::*;
pub use witness_log::{WitnessLog, WitnessEntry, LatencyBreakdown, RoutingDecision, AsyncWriteConfig, WitnessLogStats};
pub use gguf::{
    GgufFile, GgufModelLoader, GgufHeader, GgufValue, GgufQuantType,
    TensorInfo, QuantizedTensor, ModelConfig as GgufModelConfig,
    // New GGUF loading types
    GgufLoader, LoadConfig, LoadProgress, LoadedWeights, LoadedTensor,
    TensorCategory, TensorNameMapper, StreamingLoader,
    ModelInitializer, ModelWeights, LayerWeights, WeightTensor, QuantizedWeight,
    ProgressModelBuilder,
};
pub use hub::{
    // Download
    ModelDownloader, DownloadConfig, DownloadProgress, DownloadError, ChecksumVerifier,
    // Upload
    ModelUploader, UploadConfig, UploadProgress, UploadError, ModelMetadata,
    // Registry
    RuvLtraRegistry, ModelInfo as HubModelInfo, ModelSize, QuantizationLevel,
    HardwareRequirements, get_model_info,
    // Model Card
    ModelCard, ModelCardBuilder, TaskType as HubTaskType, Framework, License, DatasetInfo, MetricResult,
    // Progress
    ProgressBar, ProgressIndicator, ProgressStyle, ProgressCallback, MultiProgress,
    // Common
    HubError, default_cache_dir, get_hf_token,
};
pub use serving::{
    // Request types
    InferenceRequest, RequestId, Priority, RequestState, RunningRequest,
    CompletedRequest, FinishReason, TokenOutput,
    // Batch types
    BatchedRequest, BatchStats, ScheduledBatch, IterationPlan, PrefillTask, DecodeTask, TokenBudget,
    // KV cache management
    KvCacheManager, KvCachePoolConfig, KvCacheAllocation, KvCacheManagerStats,
    // Scheduler
    ContinuousBatchScheduler, IterationScheduler, SchedulerConfig, SchedulerStats,
    RequestQueue, PreemptionMode, PriorityPolicy,
    // Engine
    ServingEngine, ServingEngineConfig, ServingMetrics, GenerationResult,
};
pub use quantize::{
    // Core quantizer
    RuvltraQuantizer, QuantConfig, TargetFormat,
    // Quantization functions
    quantize_ruvltra_q4, quantize_ruvltra_q5, quantize_ruvltra_q8, dequantize_for_ane,
    // Memory estimation
    estimate_memory_q4, estimate_memory_q5, estimate_memory_q8, MemoryEstimate,
    // Block types
    Q4KMBlock, Q5KMBlock, Q8Block,
    // Progress tracking
    QuantProgress, QuantStats,
};
pub use training::{
    // Claude task dataset
    ClaudeTaskDataset, ClaudeTaskExample, TaskCategory, TaskMetadata,
    ComplexityLevel, DomainType, DatasetConfig, AugmentationConfig,
    DatasetGenerator, DatasetStats,
    // GRPO optimizer for reinforcement learning
    GrpoConfig, GrpoOptimizer, GrpoSample, GrpoStats, GrpoUpdateResult,
    GrpoBatch, SampleGroup,
    // MCP tool training
    McpToolTrainer, McpTrainingConfig, ToolTrajectory, TrajectoryStep,
    TrajectoryBuilder, StepBuilder, TrajectoryMetadata,
    TrainingResult, TrainingStats, TrainingCheckpoint, EvaluationMetrics,
    // Tool calling dataset
    ToolCallDataset, ToolCallExample, ToolDatasetConfig, ToolDatasetStats,
    McpToolDef, ToolParam, ParamType, DifficultyLevel, DifficultyWeights,
    McpToolCategory,
};

// RuvLTRA model architecture exports
pub use models::{
    // Configuration
    RuvLtraConfig, AneOptimization, QuantizationType, MemoryLayout,
    // Model components
    RuvLtraModel, RuvLtraAttention, RuvLtraMLP, RuvLtraDecoderLayer,
    // Utilities
    RuvLtraModelInfo, AneDispatcher,
};

// Ruvector integration exports (unified entry point for all Ruvector capabilities)
pub use capabilities::{
    RuvectorCapabilities, HNSW_AVAILABLE, ATTENTION_AVAILABLE, GRAPH_AVAILABLE,
    GNN_AVAILABLE, SONA_AVAILABLE, SIMD_AVAILABLE, PARALLEL_AVAILABLE,
    gate_feature, gate_feature_or,
};
pub use ruvector_integration::{
    // Main integration
    RuvectorIntegration, IntegrationConfig, IntegrationStats,
    // Unified index
    UnifiedIndex, VectorMetadata, IndexStats, SearchResultWithMetadata,
    // Intelligence layer
    IntelligenceLayer, IntelligentRoutingDecision, IntelligenceLayerStats,
};

// Quality scoring exports
pub use quality::{
    // Core metrics
    QualityMetrics, QualityWeights, QualityDimension, QualitySummary, TrendDirection,
    // Scoring engine
    QualityScoringEngine, ScoringConfig, ScoringContext, QualityHistory,
    ComparisonResult, TrendAnalysis, ImprovementRecommendation,
    // Coherence validation
    CoherenceValidator, CoherenceConfig, SemanticConsistencyResult,
    ContradictionResult, CoherenceViolation, LogicalFlowResult,
    // Diversity analysis
    DiversityAnalyzer, DiversityConfig, DiversityResult,
    DiversificationSuggestion, ModeCollapseResult,
    // Schema validators
    SchemaValidator, JsonSchemaValidator, TypeValidator, RangeValidator,
    FormatValidator, CombinedValidator, ValidationResult, ValidationError,
    ValidationCombinator,
};

// Context management exports (intelligent pruning and semantic memory)
pub use context::{
    // Agentic memory
    AgenticMemory, AgenticMemoryConfig, MemoryType,
    // Working memory
    WorkingMemory, WorkingMemoryConfig, TaskContext, ScratchpadEntry, AttentionWeights,
    // Episodic memory
    EpisodicMemory, EpisodicMemoryConfig, Episode, EpisodeMetadata,
    EpisodeTrajectory, CompressedEpisode,
    // Context manager
    IntelligentContextManager, ContextManagerConfig, PreparedContext,
    PriorityScorer, ContextElement, ElementPriority,
    // Semantic cache
    SemanticToolCache, SemanticCacheConfig, CachedToolResult, CacheStats,
    // Claude Flow bridge
    ClaudeFlowMemoryBridge, ClaudeFlowBridgeConfig, SyncResult,
};

// Self-Reflection architecture exports (error recovery and self-correction)
pub use reflection::{
    // Reflective agent wrapper
    ReflectiveAgent, ReflectionStrategy, ReflectionConfig, RetryConfig,
    ExecutionContext, ExecutionResult, Reflection, PreviousAttempt,
    BaseAgent, ReflectiveAgentStats,
    // Confidence-based revision (IoE pattern)
    ConfidenceChecker, ConfidenceConfig, ConfidenceLevel, WeakPoint, RevisionResult,
    ConfidenceCheckRecord, ConfidenceFactorWeights, WeaknessType,
    // Error pattern learning
    ErrorPatternLearner, ErrorPatternLearnerConfig, ErrorPattern, ErrorCluster,
    RecoveryStrategy, RecoverySuggestion, ErrorCategory, RecoveryOutcome,
    SimilarError, ErrorLearnerStats,
    // Multi-perspective critique
    Perspective, CorrectnessChecker, CompletenessChecker, ConsistencyChecker,
    CritiqueResult, CritiqueIssue, IssueCategory, UnifiedCritique, PerspectiveConfig,
};

// ReasoningBank exports (learning from Claude trajectories)
pub use reasoning_bank::{
    // Main ReasoningBank
    ReasoningBank, ReasoningBankConfig, ReasoningBankStats,
    // Trajectory recording (aliased to avoid conflict with training::TrajectoryStep)
    Trajectory as ReasoningTrajectory,
    TrajectoryStep as ReasoningTrajectoryStep,
    TrajectoryRecorder, TrajectoryId, StepOutcome,
    // Pattern storage with HNSW
    PatternStore, PatternStoreConfig, Pattern, PatternCategory, PatternSearchResult, PatternStats,
    // Verdict system (aliased to avoid conflict with claude_flow::reasoning_bank::Verdict)
    Verdict as ReasoningVerdict,
    RootCause, VerdictAnalyzer, FailurePattern as VerdictFailurePattern,
    RecoveryStrategy as VerdictRecoveryStrategy,
    // EWC++ consolidation
    PatternConsolidator, ConsolidationConfig, FisherInformation, ImportanceScore,
    // Memory distillation
    MemoryDistiller, DistillationConfig, CompressedTrajectory, KeyLesson,
};

// Metal GPU acceleration exports (macOS only)
#[cfg(all(target_os = "macos", feature = "metal-compute"))]
pub use metal::{
    MetalContext, MetalConfig, MetalPipelines, MetalBuffer, MetalBufferPool,
    AttentionParams, GemmParams, NormParams, RopeParams,
    is_metal_available, get_device_info, MetalDeviceInfo,
    tile_sizes, shader_source,
};

/// RuvLLM engine configuration.
///
/// This configuration struct controls all aspects of the RuvLLM engine,
/// including storage paths, attention mechanisms, KV cache settings,
/// session management, and SONA learning parameters.
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::{RuvLLMConfig, PagedAttentionConfig, KvCacheConfig};
///
/// let config = RuvLLMConfig {
///     storage_path: "/var/ruvllm".to_string(),
///     max_sessions: 500,
///     embedding_dim: 1024,
///     ..Default::default()
/// };
/// ```
///
/// # Performance Tuning
///
/// | Parameter | Default | High Throughput | Low Latency |
/// |-----------|---------|-----------------|-------------|
/// | `max_sessions` | 1000 | 2000 | 500 |
/// | `embedding_dim` | 768 | 1024 | 512 |
#[derive(Debug, Clone)]
pub struct RuvLLMConfig {
    /// Path to Ruvector storage
    pub storage_path: String,
    /// Paged attention configuration
    pub paged_attention: PagedAttentionConfig,
    /// KV cache configuration
    pub kv_cache: KvCacheConfig,
    /// Session configuration
    pub session: SessionConfig,
    /// SONA learning configuration
    pub sona: SonaConfig,
    /// Maximum concurrent sessions
    pub max_sessions: usize,
    /// Embedding dimension for semantic search
    pub embedding_dim: usize,
}

impl Default for RuvLLMConfig {
    fn default() -> Self {
        Self {
            storage_path: ".ruvllm".to_string(),
            paged_attention: PagedAttentionConfig::default(),
            kv_cache: KvCacheConfig::default(),
            session: SessionConfig::default(),
            sona: SonaConfig::default(),
            max_sessions: 1000,
            embedding_dim: 768,
        }
    }
}

/// Main RuvLLM engine for LLM inference with intelligent memory.
///
/// The `RuvLLMEngine` is the primary entry point for RuvLLM, providing:
///
/// - **Session Management**: Create and manage user sessions with state persistence
/// - **Policy Storage**: Ruvector-backed semantic search for runtime policies
/// - **Adapter Management**: Hot-swapping LoRA adapters for task-specific tuning
/// - **Witness Logging**: Audit trail with HNSW-indexed semantic search
/// - **SONA Learning**: Three-tier continuous learning integration
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::{RuvLLMEngine, RuvLLMConfig};
///
/// // Create engine with configuration
/// let config = RuvLLMConfig::default();
/// let engine = RuvLLMEngine::new(config)?;
///
/// // Create a session for a user
/// let session = engine.create_session(Some("user-123"))?;
///
/// // Search for relevant policies
/// let embedding = compute_embedding("code completion task");
/// let policies = engine.search_policies(&embedding, 5)?;
///
/// // Record audit entry
/// let entry = WitnessEntry::new("completion", latency, routing);
/// engine.record_witness(entry)?;
/// ```
///
/// # Architecture
///
/// ```text
/// +-------------------+     +-------------------+
/// | RuvLLMEngine      |---->| PolicyStore       |
/// |                   |     | (Ruvector)        |
/// |                   |     +-------------------+
/// |                   |
/// |                   |---->| SessionIndex      |
/// |                   |     | (Ruvector)        |
/// |                   |     +-------------------+
/// |                   |
/// |                   |---->| WitnessLog        |
/// |                   |     | (HNSW search)     |
/// +-------------------+     +-------------------+
/// ```
pub struct RuvLLMEngine {
    /// Configuration
    config: RuvLLMConfig,
    /// Policy store backed by Ruvector
    policy_store: PolicyStore,
    /// Session manager
    session_manager: SessionManager,
    /// Session index backed by Ruvector
    session_index: SessionIndex,
    /// Adapter manager
    adapter_manager: AdapterManager,
    /// Witness log for audit
    witness_log: WitnessLog,
    /// SONA learning integration
    sona: SonaIntegration,
}

impl RuvLLMEngine {
    /// Create a new RuvLLM engine with the given configuration.
    ///
    /// This initializes all subsystems including:
    /// - Policy store for learned thresholds
    /// - Session index for conversation state
    /// - Witness log for audit trails
    /// - SONA integration for learning loops
    ///
    /// # Arguments
    ///
    /// * `config` - Engine configuration
    ///
    /// # Errors
    ///
    /// Returns an error if storage paths cannot be created or initialized.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvllm::{RuvLLMEngine, RuvLLMConfig};
    ///
    /// let engine = RuvLLMEngine::new(RuvLLMConfig::default())?;
    /// ```
    pub fn new(config: RuvLLMConfig) -> Result<Self> {
        let storage_path = &config.storage_path;

        let policy_store = PolicyStore::new(
            &format!("{}/policies", storage_path),
            config.embedding_dim,
        )?;

        let session_index = SessionIndex::new(
            &format!("{}/sessions", storage_path),
            config.embedding_dim,
        )?;

        let witness_log = WitnessLog::new(
            &format!("{}/witness", storage_path),
            config.embedding_dim,
        )?;

        let session_manager = SessionManager::new(config.session.clone());
        let adapter_manager = AdapterManager::new();
        let sona = SonaIntegration::new(config.sona.clone());

        Ok(Self {
            config,
            policy_store,
            session_manager,
            session_index,
            adapter_manager,
            witness_log,
            sona,
        })
    }

    /// Create a new session for a user.
    ///
    /// Sessions track conversation state, KV cache references, and enable
    /// multi-turn interactions. Each session is automatically indexed in
    /// Ruvector for semantic retrieval.
    ///
    /// # Arguments
    ///
    /// * `user_id` - Optional user identifier for session tracking
    ///
    /// # Returns
    ///
    /// A new `Session` instance with a unique ID.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Anonymous session
    /// let session = engine.create_session(None)?;
    ///
    /// // User-identified session
    /// let session = engine.create_session(Some("user-123"))?;
    /// println!("Session ID: {}", session.id());
    /// ```
    pub fn create_session(&self, user_id: Option<&str>) -> Result<Session> {
        let session = self.session_manager.create_session(user_id)?;

        // Index the session in Ruvector
        let state = SessionState::from_session(&session);
        self.session_index.store(&state)?;

        Ok(session)
    }

    /// Get session by ID
    pub fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        self.session_manager.get_session(session_id)
    }

    /// Search for policies matching the given context embedding.
    ///
    /// Uses HNSW-indexed semantic search to find relevant policies
    /// (quantization settings, routing rules, etc.) based on the
    /// current request context.
    ///
    /// # Arguments
    ///
    /// * `context_embedding` - Vector embedding of the current context
    /// * `limit` - Maximum number of policies to return
    ///
    /// # Returns
    ///
    /// Vector of matching `PolicyEntry` items, sorted by relevance.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let context = compute_embedding("code completion for Python");
    /// let policies = engine.search_policies(&context, 5)?;
    ///
    /// for policy in policies {
    ///     println!("Policy: {:?}, score: {}", policy.policy_type, policy.score);
    /// }
    /// ```
    pub fn search_policies(&self, context_embedding: &[f32], limit: usize) -> Result<Vec<PolicyEntry>> {
        self.policy_store.search(context_embedding, limit)
    }

    /// Record a witness entry for audit logging.
    ///
    /// Witness entries provide an audit trail of inference decisions,
    /// including latency breakdowns, routing decisions, and quality scores.
    /// All entries are HNSW-indexed for semantic search.
    ///
    /// # Arguments
    ///
    /// * `entry` - The witness entry to record
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvllm::{WitnessEntry, LatencyBreakdown, RoutingDecision};
    ///
    /// let entry = WitnessEntry {
    ///     session_id: session.id().to_string(),
    ///     request_type: "completion".to_string(),
    ///     latency: LatencyBreakdown {
    ///         prefill_ms: 45.0,
    ///         decode_ms: 120.0,
    ///         total_ms: 165.0,
    ///     },
    ///     routing: RoutingDecision::default(),
    ///     ..Default::default()
    /// };
    ///
    /// engine.record_witness(entry)?;
    /// ```
    pub fn record_witness(&self, entry: WitnessEntry) -> Result<()> {
        self.witness_log.record(entry)
    }

    /// Search witness logs semantically
    pub fn search_witness(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<WitnessEntry>> {
        self.witness_log.search(query_embedding, limit)
    }

    /// Get the SONA integration for learning
    pub fn sona(&self) -> &SonaIntegration {
        &self.sona
    }

    /// Get the adapter manager
    pub fn adapters(&self) -> &AdapterManager {
        &self.adapter_manager
    }

    /// Get the policy store
    pub fn policies(&self) -> &PolicyStore {
        &self.policy_store
    }
}

