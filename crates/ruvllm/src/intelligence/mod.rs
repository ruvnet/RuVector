//! External Intelligence Providers for SONA Learning (ADR-029)
//!
//! This module defines the [`IntelligenceProvider`] trait — the extension point
//! that lets external systems feed quality signals into ruvllm's learning loops
//! without modifying ruvllm core.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use ruvllm::intelligence::{FileSignalProvider, IntelligenceProvider};
//! use std::path::PathBuf;
//!
//! // File-based (non-Rust systems write JSON, Rust reads it)
//! let provider = FileSignalProvider::new(PathBuf::from("signals.json"));
//! let signals = provider.load_signals()?;
//!
//! // Custom provider
//! struct CiPipelineProvider;
//! impl IntelligenceProvider for CiPipelineProvider {
//!     fn name(&self) -> &str { "ci-pipeline" }
//!     fn load_signals(&self) -> ruvllm::Result<Vec<QualitySignal>> {
//!         Ok(vec![])
//!     }
//! }
//! ```

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::RuvLLMError;

// ============================================================================
// Signal types
// ============================================================================

/// A quality signal from an external system.
///
/// Represents one completed task with quality assessment data that can feed
/// into SONA trajectories, the embedding classifier, and model router
/// calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySignal {
    /// Unique identifier for this signal.
    pub id: String,
    /// Human-readable task description (used for embedding generation).
    pub task_description: String,
    /// Execution outcome: `"success"`, `"partial_success"`, or `"failure"`.
    pub outcome: String,
    /// Composite quality score (0.0–1.0).
    pub quality_score: f32,
    /// Optional human verdict: `"approved"`, `"rejected"`, or `None`.
    #[serde(default)]
    pub human_verdict: Option<String>,
    /// Optional structured quality factors for detailed analysis.
    #[serde(default)]
    pub quality_factors: Option<QualityFactors>,
    /// ISO 8601 timestamp of task completion.
    pub completed_at: String,
}

/// Granular quality factor breakdown.
///
/// Not all providers will have all factors.  Fields default to `None`,
/// meaning "not assessed" (distinct from `Some(0.0)`, which means
/// "assessed as zero").
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityFactors {
    pub acceptance_criteria_met: Option<f32>,
    pub tests_passing: Option<f32>,
    pub no_regressions: Option<f32>,
    pub lint_clean: Option<f32>,
    pub type_check_clean: Option<f32>,
    pub follows_patterns: Option<f32>,
    pub context_relevance: Option<f32>,
    pub reasoning_coherence: Option<f32>,
    pub execution_efficiency: Option<f32>,
}

/// Quality weight overrides from a provider.
///
/// Weights influence how the intelligence loader computes composite quality
/// for that provider's signals.  They must sum to approximately 1.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityWeights {
    pub task_completion: f32,
    pub code_quality: f32,
    pub process: f32,
}

impl Default for QualityWeights {
    fn default() -> Self {
        Self {
            task_completion: 0.5,
            code_quality: 0.3,
            process: 0.2,
        }
    }
}

// ============================================================================
// Provider trait
// ============================================================================

/// Extension point for external systems that supply quality signals to ruvllm.
///
/// Implementations are registered with an intelligence loader and called
/// during signal ingestion.  The loader handles mapping signals to SONA
/// trajectories, classifier entries, and router calibration data.
///
/// # Design
///
/// Follows the same trait pattern as [`LlmBackend`](crate::backends::LlmBackend)
/// and [`Tokenizer`](crate::backends::Tokenizer) — a trait object behind
/// `Box<dyn IntelligenceProvider>`.
pub trait IntelligenceProvider: Send + Sync {
    /// Provider identity, used in logging and diagnostics.
    fn name(&self) -> &str;

    /// Load quality signals from this provider's data source.
    ///
    /// Called once per ingestion cycle.  Returns an empty `Vec` when no
    /// signals are available (this is normal, not an error).
    fn load_signals(&self) -> Result<Vec<QualitySignal>>;

    /// Optionally provide quality weight overrides.
    ///
    /// If `Some`, these weights are used when computing composite quality
    /// for this provider's signals.  If `None`, default weights apply.
    fn quality_weights(&self) -> Option<QualityWeights> {
        None
    }
}

// ============================================================================
// Built-in file provider
// ============================================================================

/// Reads quality signals from a JSON file.
///
/// This is the default provider for systems that write a signal file to
/// `.claude/intelligence/data/`.  Non-Rust integrations (TypeScript, Python,
/// etc.) typically use this path.
///
/// The expected JSON format is either:
/// - A JSON array of [`QualitySignal`] objects, or
/// - A JSON object with a `"signals"` key containing such an array.
pub struct FileSignalProvider {
    path: PathBuf,
}

impl FileSignalProvider {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl IntelligenceProvider for FileSignalProvider {
    fn name(&self) -> &str {
        "file-signals"
    }

    fn load_signals(&self) -> Result<Vec<QualitySignal>> {
        if !self.path.exists() {
            return Ok(vec![]); // No file = no signals, not an error
        }
        parse_signal_file(&self.path)
    }

    fn quality_weights(&self) -> Option<QualityWeights> {
        let config_path = self
            .path
            .parent()
            .unwrap_or(Path::new("."))
            .join("quality-weights.json");
        read_weight_config(&config_path).ok()
    }
}

// ============================================================================
// Intelligence loader (provider registry)
// ============================================================================

/// Registry that collects signals from multiple [`IntelligenceProvider`]s.
///
/// ```rust,ignore
/// let mut loader = IntelligenceProviderLoader::new();
/// loader.register(Box::new(FileSignalProvider::new(path)));
/// let (signals, stats) = loader.load_all();
/// ```
pub struct IntelligenceProviderLoader {
    providers: Vec<Box<dyn IntelligenceProvider>>,
}

/// Statistics from a `load_all()` run.
#[derive(Debug, Clone, Default)]
pub struct ProviderLoadStats {
    /// Total signals successfully loaded across all providers.
    pub total_signals: usize,
    /// Per-provider signal counts (name, count).
    pub per_provider: Vec<(String, usize)>,
    /// Providers that errored (name, error message).
    pub errors: Vec<(String, String)>,
}

impl IntelligenceProviderLoader {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Register an external intelligence provider.
    ///
    /// Providers are called in registration order during [`load_all()`].
    pub fn register(&mut self, provider: Box<dyn IntelligenceProvider>) {
        self.providers.push(provider);
    }

    /// Number of registered providers.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Load signals from all registered providers.
    ///
    /// Provider failures are non-fatal — they are recorded in
    /// [`ProviderLoadStats::errors`] and the remaining providers continue.
    pub fn load_all(&self) -> (Vec<QualitySignal>, ProviderLoadStats) {
        let mut all_signals = Vec::new();
        let mut stats = ProviderLoadStats::default();

        for provider in &self.providers {
            match provider.load_signals() {
                Ok(signals) => {
                    let count = signals.len();
                    stats
                        .per_provider
                        .push((provider.name().to_string(), count));
                    stats.total_signals += count;
                    all_signals.extend(signals);
                    tracing::info!("Provider '{}': loaded {} signals", provider.name(), count);
                }
                Err(e) => {
                    let msg = e.to_string();
                    tracing::warn!("Provider '{}' failed: {}", provider.name(), msg);
                    stats
                        .errors
                        .push((provider.name().to_string(), msg));
                }
            }
        }

        (all_signals, stats)
    }
}

impl Default for IntelligenceProviderLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// File parsing helpers
// ============================================================================

/// Wrapper for JSON files that may be a bare array or `{ "signals": [...] }`.
#[derive(Deserialize)]
#[serde(untagged)]
enum SignalFileFormat {
    Array(Vec<QualitySignal>),
    Wrapped { signals: Vec<QualitySignal> },
}

fn parse_signal_file(path: &Path) -> Result<Vec<QualitySignal>> {
    let data = std::fs::read_to_string(path).map_err(|e| {
        RuvLLMError::Config(format!("Failed to read signal file {}: {}", path.display(), e))
    })?;
    let parsed: SignalFileFormat = serde_json::from_str(&data).map_err(|e| {
        RuvLLMError::Config(format!(
            "Failed to parse signal file {}: {}",
            path.display(),
            e
        ))
    })?;
    Ok(match parsed {
        SignalFileFormat::Array(v) => v,
        SignalFileFormat::Wrapped { signals } => signals,
    })
}

fn read_weight_config(path: &Path) -> Result<QualityWeights> {
    let data = std::fs::read_to_string(path).map_err(|e| {
        RuvLLMError::Config(format!(
            "Failed to read weight config {}: {}",
            path.display(),
            e
        ))
    })?;
    serde_json::from_str(&data).map_err(|e| {
        RuvLLMError::Config(format!(
            "Failed to parse weight config {}: {}",
            path.display(),
            e
        ))
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_file_provider_missing_file_returns_empty() {
        let provider = FileSignalProvider::new(PathBuf::from("/nonexistent/signals.json"));
        let signals = provider.load_signals().unwrap();
        assert!(signals.is_empty());
    }

    #[test]
    fn test_file_provider_parses_array_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("signals.json");
        let mut f = std::fs::File::create(&path).unwrap();
        write!(
            f,
            r#"[{{"id":"s1","task_description":"test","outcome":"success","quality_score":0.9,"completed_at":"2026-01-01T00:00:00Z"}}]"#
        )
        .unwrap();

        let provider = FileSignalProvider::new(path);
        let signals = provider.load_signals().unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].id, "s1");
        assert!((signals[0].quality_score - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_file_provider_parses_wrapped_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("signals.json");
        let mut f = std::fs::File::create(&path).unwrap();
        write!(
            f,
            r#"{{"signals":[{{"id":"s2","task_description":"lint","outcome":"failure","quality_score":0.2,"completed_at":"2026-01-01T00:00:00Z"}}]}}"#
        )
        .unwrap();

        let provider = FileSignalProvider::new(path);
        let signals = provider.load_signals().unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].outcome, "failure");
    }

    #[test]
    fn test_quality_weights_default() {
        let w = QualityWeights::default();
        let sum = w.task_completion + w.code_quality + w.process;
        assert!((sum - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_loader_no_providers() {
        let loader = IntelligenceProviderLoader::new();
        let (signals, stats) = loader.load_all();
        assert!(signals.is_empty());
        assert_eq!(stats.total_signals, 0);
        assert!(stats.errors.is_empty());
    }

    #[test]
    fn test_loader_multiple_providers() {
        struct MockProvider {
            name: &'static str,
            count: usize,
        }
        impl IntelligenceProvider for MockProvider {
            fn name(&self) -> &str {
                self.name
            }
            fn load_signals(&self) -> Result<Vec<QualitySignal>> {
                Ok((0..self.count)
                    .map(|i| QualitySignal {
                        id: format!("{}-{}", self.name, i),
                        task_description: "test".into(),
                        outcome: "success".into(),
                        quality_score: 0.8,
                        human_verdict: None,
                        quality_factors: None,
                        completed_at: "2026-01-01T00:00:00Z".into(),
                    })
                    .collect())
            }
        }

        let mut loader = IntelligenceProviderLoader::new();
        loader.register(Box::new(MockProvider {
            name: "ci",
            count: 3,
        }));
        loader.register(Box::new(MockProvider {
            name: "review",
            count: 2,
        }));

        let (signals, stats) = loader.load_all();
        assert_eq!(signals.len(), 5);
        assert_eq!(stats.total_signals, 5);
        assert_eq!(stats.per_provider.len(), 2);
    }

    #[test]
    fn test_loader_tolerates_provider_failure() {
        struct FailProvider;
        impl IntelligenceProvider for FailProvider {
            fn name(&self) -> &str {
                "broken"
            }
            fn load_signals(&self) -> Result<Vec<QualitySignal>> {
                Err(RuvLLMError::Config("boom".into()))
            }
        }

        struct OkProvider;
        impl IntelligenceProvider for OkProvider {
            fn name(&self) -> &str {
                "ok"
            }
            fn load_signals(&self) -> Result<Vec<QualitySignal>> {
                Ok(vec![QualitySignal {
                    id: "good".into(),
                    task_description: "test".into(),
                    outcome: "success".into(),
                    quality_score: 1.0,
                    human_verdict: None,
                    quality_factors: None,
                    completed_at: "2026-01-01T00:00:00Z".into(),
                }])
            }
        }

        let mut loader = IntelligenceProviderLoader::new();
        loader.register(Box::new(FailProvider));
        loader.register(Box::new(OkProvider));

        let (signals, stats) = loader.load_all();
        assert_eq!(signals.len(), 1);
        assert_eq!(stats.errors.len(), 1);
        assert_eq!(stats.errors[0].0, "broken");
    }
}
