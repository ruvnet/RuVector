//! RuvLLM Evaluation Harness
//!
//! Three-layer evaluation framework:
//! 1. **Correctness**: Does the patch actually work?
//! 2. **Diff Quality**: Does it behave like a senior engineer?
//! 3. **Systems Economics**: Is it worth running at scale?
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruvllm::evaluation::{EvaluationHarness, EvalConfig, AblationMode};
//!
//! let config = EvalConfig {
//!     task_count: 100,
//!     seeds: vec![42, 123, 456],
//!     ablation_modes: vec![
//!         AblationMode::Baseline,
//!         AblationMode::RetrievalOnly,
//!         AblationMode::AdaptersOnly,
//!         AblationMode::RetrievalPlusAdapters,
//!         AblationMode::Full, // retrieval + adapters + SONA
//!     ],
//!     ..Default::default()
//! };
//!
//! let harness = EvaluationHarness::new(config);
//! let report = harness.run_evaluation(&tasks).await?;
//! println!("{}", report.to_leaderboard());
//! ```

mod correctness;
mod diff_quality;
mod economics;
mod harness;
mod metrics;
mod real_harness;
mod report;
pub mod swe_bench;

pub use correctness::{
    CorrectnessMetrics, TaskResult, TestSuiteResult, VerificationLevel,
};
pub use diff_quality::{
    DiffQualityMetrics, DiffAnalyzer, EditLocality, Minimality, ReviewBurden,
};
pub use economics::{
    EconomicsMetrics, LatencyDistribution, CostTracker, StabilityMetrics,
};
pub use harness::{
    EvaluationHarness, EvalConfig, AblationMode, EvalTask, EvalRun, EvalReport, ModeMetrics,
};
pub use metrics::{
    MetricCollector, MetricSnapshot, AggregatedMetrics,
};
pub use report::{
    LeaderboardEntry, AblationComparison,
};
pub use real_harness::{
    RealEvaluationHarness, RealInferenceConfig,
};
