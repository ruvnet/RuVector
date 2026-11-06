//! Simplified NAPI-RS bindings for Node.js
//! Enable with feature flag: `napi`
//!
//! This version uses a simpler API that doesn't expose TrajectoryBuilder to JS.
//!
//! A Rust panic cannot unwind into Node: it aborts the whole process. Inputs
//! that the core asserts on are validated here and thrown as JS errors.

#![cfg(feature = "napi")]

use napi::bindgen_prelude::AsyncTask;
use napi::{Env, Error, Result, Task};
use napi_derive::napi;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::inflight::InflightStore;
use crate::{
    LearnedPattern, SonaConfig, SonaEngine as RustSonaEngine,
    TrajectoryBuilder as RustTrajectoryBuilder,
};

/// Most trajectories one engine keeps open. Beyond this, beginning a
/// trajectory discards the oldest unfinished one (it is never submitted), so
/// JS that begins trajectories without ending them cannot grow memory
/// without bound.
pub const MAX_INFLIGHT_TRAJECTORIES: usize = 10_000;

/// Most steps recorded per trajectory; further steps are ignored.
pub const MAX_STEPS_PER_TRAJECTORY: usize = 4_096;

/// Most context ids recorded per trajectory; further ids are ignored.
pub const MAX_CONTEXTS_PER_TRAJECTORY: usize = 1_024;

/// Largest accepted hidden / embedding dimension.
pub const MAX_DIM: u32 = 1 << 16;
/// Largest accepted base-LoRA rank.
pub const MAX_RANK: u32 = 1024;
/// Largest number of base-LoRA weights a config may allocate (256 MiB of f32).
pub const MAX_BASE_LORA_ELEMENTS: u64 = 1 << 26;
/// Largest accepted pattern cluster count.
pub const MAX_PATTERN_CLUSTERS: u32 = 1 << 16;
/// Largest accepted trajectory buffer capacity.
pub const MAX_TRAJECTORY_CAPACITY: u32 = 1 << 20;

fn check_range(what: &str, value: u32, min: u32, max: u32) -> Result<usize> {
    if (min..=max).contains(&value) {
        Ok(value as usize)
    } else {
        Err(Error::from_reason(format!(
            "{what} must be between {min} and {max}, got {value}"
        )))
    }
}

fn check_len(what: &str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(Error::from_reason(format!(
            "{what} length {got} does not match hiddenDim {expected}"
        )))
    }
}

/// Node.js SONA Engine wrapper
#[napi]
pub struct SonaEngine {
    /// Shared so `forceLearnAsync` can run the learning cycle off the JS thread
    inner: Arc<RustSonaEngine>,
    /// Open trajectories of this engine (ids are per engine, so one engine
    /// cannot end another's trajectory, and they are freed with the engine).
    /// Bounded by `MAX_INFLIGHT_TRAJECTORIES`.
    builders: Mutex<InflightStore<RustTrajectoryBuilder>>,
}

impl SonaEngine {
    fn from_inner(inner: RustSonaEngine) -> Self {
        Self {
            inner: Arc::new(inner),
            builders: Mutex::new(InflightStore::new(MAX_INFLIGHT_TRAJECTORIES)),
        }
    }

    fn builders(&self) -> MutexGuard<'_, InflightStore<RustTrajectoryBuilder>> {
        self.builders.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[napi]
impl SonaEngine {
    /// Create a new SONA engine with default configuration
    /// @param hidden_dim - Hidden dimension size (e.g., 256, 512)
    #[napi(constructor, catch_unwind)]
    pub fn new(hidden_dim: u32) -> Result<Self> {
        let dim = check_range("hiddenDim", hidden_dim, 1, MAX_DIM)?;
        Ok(Self::from_inner(RustSonaEngine::new(dim)))
    }

    /// Create with custom configuration
    /// @param config - Custom SONA configuration object
    #[napi(factory, catch_unwind)]
    pub fn with_config(config: JsSonaConfig) -> Result<Self> {
        let micro_lora_rank = config.micro_lora_rank.unwrap_or(1);
        if !(1..=2).contains(&micro_lora_rank) {
            return Err(Error::from_reason(format!(
                "microLoraRank must be 1 or 2, got {micro_lora_rank}"
            )));
        }
        check_range("hiddenDim", config.hidden_dim, 1, MAX_DIM)?;
        check_range(
            "embeddingDim",
            config.embedding_dim.unwrap_or(config.hidden_dim),
            1,
            MAX_DIM,
        )?;
        check_range(
            "baseLoraRank",
            config.base_lora_rank.unwrap_or(8),
            0,
            MAX_RANK,
        )?;
        // BaseLoRA keeps 12 layers of [hidden_dim x rank] down/up projections.
        let base_elems =
            u64::from(config.hidden_dim) * u64::from(config.base_lora_rank.unwrap_or(8)) * 24;
        if base_elems > MAX_BASE_LORA_ELEMENTS {
            return Err(Error::from_reason(format!(
                "hiddenDim x baseLoraRank is too large ({base_elems} weights, max {MAX_BASE_LORA_ELEMENTS})"
            )));
        }
        check_range(
            "patternClusters",
            config.pattern_clusters.unwrap_or(50),
            1,
            MAX_PATTERN_CLUSTERS,
        )?;
        check_range(
            "trajectoryCapacity",
            config.trajectory_capacity.unwrap_or(10000),
            1,
            MAX_TRAJECTORY_CAPACITY,
        )?;
        let rust_config = SonaConfig {
            hidden_dim: config.hidden_dim as usize,
            embedding_dim: config.embedding_dim.unwrap_or(config.hidden_dim) as usize,
            micro_lora_rank: micro_lora_rank as usize,
            base_lora_rank: config.base_lora_rank.unwrap_or(8) as usize,
            micro_lora_lr: config.micro_lora_lr.unwrap_or(0.001) as f32,
            base_lora_lr: config.base_lora_lr.unwrap_or(0.0001) as f32,
            ewc_lambda: config.ewc_lambda.unwrap_or(1000.0) as f32,
            pattern_clusters: config.pattern_clusters.unwrap_or(50) as usize,
            trajectory_capacity: config.trajectory_capacity.unwrap_or(10000) as usize,
            background_interval_ms: config.background_interval_ms.unwrap_or(3600000) as u64,
            quality_threshold: config.quality_threshold.unwrap_or(0.5) as f32,
            enable_simd: config.enable_simd.unwrap_or(true),
        };
        Ok(Self::from_inner(RustSonaEngine::with_config(rust_config)))
    }

    /// Start a new trajectory recording
    /// @param query_embedding - Query embedding vector (Float64Array)
    /// @returns Trajectory ID for adding steps
    ///
    /// At most 10000 trajectories stay open per engine: beginning another
    /// discards the oldest unfinished one, whose ID then acts like an unknown
    /// ID. IDs are unique among the open trajectories, including after the
    /// u32 counter wraps.
    #[napi(catch_unwind)]
    pub fn begin_trajectory(&self, query_embedding: Vec<f64>) -> u32 {
        let embedding: Vec<f32> = query_embedding.iter().map(|&x| x as f32).collect();
        let builder = self.inner.begin_trajectory(embedding);
        let (id, _evicted) = self.builders().insert(builder);
        id
    }

    /// Add a step to trajectory
    /// @param trajectory_id - Trajectory ID from beginTrajectory
    /// @param activations - Layer activations (Float64Array)
    /// @param attention_weights - Attention weights (Float64Array)
    /// @param reward - Reward signal for this step
    ///
    /// Steps beyond 4096 per trajectory are ignored.
    #[napi(catch_unwind)]
    pub fn add_trajectory_step(
        &self,
        trajectory_id: u32,
        activations: Vec<f64>,
        attention_weights: Vec<f64>,
        reward: f64,
    ) {
        let mut builders = self.builders();
        if let Some(builder) = builders
            .get_mut(trajectory_id)
            .filter(|b| b.step_count() < MAX_STEPS_PER_TRAJECTORY)
        {
            let act: Vec<f32> = activations.iter().map(|&x| x as f32).collect();
            let att: Vec<f32> = attention_weights.iter().map(|&x| x as f32).collect();
            builder.add_step(act, att, reward as f32);
        }
    }

    /// Set model route for trajectory
    /// @param trajectory_id - Trajectory ID
    /// @param route - Model route identifier
    #[napi(catch_unwind)]
    pub fn set_trajectory_route(&self, trajectory_id: u32, route: String) {
        let mut builders = self.builders();
        if let Some(builder) = builders.get_mut(trajectory_id) {
            builder.set_model_route(&route);
        }
    }

    /// Add context to trajectory
    /// @param trajectory_id - Trajectory ID
    /// @param context_id - Context identifier
    ///
    /// Context ids beyond 1024 per trajectory are ignored.
    #[napi(catch_unwind)]
    pub fn add_trajectory_context(&self, trajectory_id: u32, context_id: String) {
        let mut builders = self.builders();
        if let Some(builder) = builders
            .get_mut(trajectory_id)
            .filter(|b| b.context_count() < MAX_CONTEXTS_PER_TRAJECTORY)
        {
            builder.add_context(&context_id);
        }
    }

    /// Complete a trajectory and submit for learning
    /// @param trajectory_id - Trajectory ID
    /// @param quality - Final quality score [0.0, 1.0]
    #[napi(catch_unwind)]
    pub fn end_trajectory(&self, trajectory_id: u32, quality: f64) {
        let mut builders = self.builders();
        if let Some(builder) = builders.remove(trajectory_id) {
            let trajectory = builder.build(quality as f32);
            self.inner.submit_trajectory(trajectory);
        }
    }

    /// Apply micro-LoRA transformation to input
    /// @param input - Input vector (Float64Array)
    /// @returns Transformed output vector
    #[napi(catch_unwind)]
    pub fn apply_micro_lora(&self, input: Vec<f64>) -> Result<Vec<f64>> {
        check_len("input", input.len(), self.inner.config().hidden_dim)?;
        let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
        // The Rust LoRA forward pass has residual semantics (it adds the
        // learned delta into `output`), so `output` must start as a clone
        // of `input` — not zeros — or callers only ever see the delta.
        let mut output = input_f32.clone();
        self.inner.apply_micro_lora(&input_f32, &mut output);
        Ok(output.iter().map(|&x| x as f64).collect())
    }

    /// Apply base-LoRA transformation to layer output
    /// @param layer_idx - Layer index
    /// @param input - Input vector (Float64Array)
    /// @returns Transformed output vector
    #[napi(catch_unwind)]
    pub fn apply_base_lora(&self, layer_idx: u32, input: Vec<f64>) -> Result<Vec<f64>> {
        check_len("input", input.len(), self.inner.config().hidden_dim)?;
        let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
        // Same residual semantics as apply_micro_lora — seed with input.
        let mut output = input_f32.clone();
        self.inner
            .apply_base_lora(layer_idx as usize, &input_f32, &mut output);
        Ok(output.iter().map(|&x| x as f64).collect())
    }

    /// Run background learning cycle if due
    /// @returns Optional status message if cycle was executed
    #[napi(catch_unwind)]
    pub fn tick(&self) -> Option<String> {
        self.inner.tick()
    }

    /// Force background learning cycle immediately
    /// @returns Status message with learning results
    #[napi(catch_unwind)]
    pub fn force_learn(&self) -> String {
        self.inner.force_learn()
    }

    /// Force background learning cycle on a worker thread, so a large
    /// trajectory buffer does not block the event loop
    /// @returns Promise of the status message with learning results
    #[napi(catch_unwind)]
    pub fn force_learn_async(&self) -> AsyncTask<ForceLearnTask> {
        AsyncTask::new(ForceLearnTask {
            engine: Arc::clone(&self.inner),
        })
    }

    /// Flush instant loop updates
    #[napi(catch_unwind)]
    pub fn flush(&self) {
        self.inner.flush();
    }

    /// Find similar learned patterns to query
    /// @param query_embedding - Query embedding vector
    /// @param k - Number of patterns to return
    /// @returns Array of learned patterns
    #[napi(catch_unwind)]
    pub fn find_patterns(&self, query_embedding: Vec<f64>, k: u32) -> Vec<JsLearnedPattern> {
        let query: Vec<f32> = query_embedding.iter().map(|&x| x as f32).collect();
        self.inner
            .find_patterns(&query, k as usize)
            .into_iter()
            .map(JsLearnedPattern::from)
            .collect()
    }

    /// Get engine statistics as JSON string
    /// @returns Statistics object as JSON string
    #[napi(catch_unwind)]
    pub fn get_stats(&self) -> String {
        serde_json::to_string(&self.inner.stats())
            .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }

    /// Enable or disable the engine
    /// @param enabled - Whether to enable the engine
    /// @throws If a forceLearnAsync call is still running
    #[napi(catch_unwind)]
    pub fn set_enabled(&mut self, enabled: bool) -> Result<()> {
        let inner = Arc::get_mut(&mut self.inner).ok_or_else(|| {
            Error::from_reason("cannot change enabled while forceLearnAsync is running")
        })?;
        inner.set_enabled(enabled);
        Ok(())
    }

    /// Check if engine is enabled
    /// @returns Whether the engine is enabled
    #[napi(catch_unwind)]
    pub fn is_enabled(&self) -> bool {
        self.inner.is_enabled()
    }
}

/// Background learning cycle run by `forceLearnAsync`
pub struct ForceLearnTask {
    engine: Arc<RustSonaEngine>,
}

impl Task for ForceLearnTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<Self::Output> {
        Ok(self.engine.force_learn())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

/// SONA configuration for Node.js
#[napi(object)]
pub struct JsSonaConfig {
    /// Hidden dimension size
    pub hidden_dim: u32,
    /// Embedding dimension (defaults to hidden_dim)
    pub embedding_dim: Option<u32>,
    /// Micro-LoRA rank (1-2, default: 1)
    pub micro_lora_rank: Option<u32>,
    /// Base LoRA rank (default: 8)
    pub base_lora_rank: Option<u32>,
    /// Micro-LoRA learning rate (default: 0.001)
    pub micro_lora_lr: Option<f64>,
    /// Base LoRA learning rate (default: 0.0001)
    pub base_lora_lr: Option<f64>,
    /// EWC lambda regularization (default: 1000.0)
    pub ewc_lambda: Option<f64>,
    /// Number of pattern clusters (default: 50)
    pub pattern_clusters: Option<u32>,
    /// Trajectory buffer capacity (default: 10000)
    pub trajectory_capacity: Option<u32>,
    /// Background learning interval in ms (default: 3600000 = 1 hour)
    pub background_interval_ms: Option<i64>,
    /// Quality threshold for learning (default: 0.5)
    pub quality_threshold: Option<f64>,
    /// Enable SIMD optimizations (default: true)
    pub enable_simd: Option<bool>,
}

/// Learned pattern for Node.js
#[napi(object)]
pub struct JsLearnedPattern {
    /// Pattern identifier
    pub id: String,
    /// Cluster centroid embedding
    pub centroid: Vec<f64>,
    /// Number of trajectories in cluster
    pub cluster_size: u32,
    /// Total weight of trajectories
    pub total_weight: f64,
    /// Average quality of member trajectories
    pub avg_quality: f64,
    /// Creation timestamp (Unix seconds)
    pub created_at: String,
    /// Last access timestamp (Unix seconds)
    pub last_accessed: String,
    /// Total access count
    pub access_count: u32,
    /// Pattern type
    pub pattern_type: String,
}

impl From<LearnedPattern> for JsLearnedPattern {
    fn from(pattern: LearnedPattern) -> Self {
        Self {
            id: pattern.id.to_string(),
            centroid: pattern.centroid.iter().map(|&x| x as f64).collect(),
            cluster_size: pattern.cluster_size as u32,
            total_weight: pattern.total_weight as f64,
            avg_quality: pattern.avg_quality as f64,
            created_at: pattern.created_at.to_string(),
            last_accessed: pattern.last_accessed.to_string(),
            access_count: pattern.access_count,
            pattern_type: format!("{:?}", pattern.pattern_type),
        }
    }
}
