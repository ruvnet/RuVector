//! Daily learning scheduler with triggers
//!
//! This module provides:
//! - DailyLearningScheduler for scheduled and triggered runs
//! - Trigger detection for opportunistic learning

use crate::discovery::{CodebaseScanner, DiscoveryLog, PatternAnalyzer};
use crate::goap::{GoapPlanner, LearningGoal, LearningWorldState};
use crate::integration::{GeminiGoapReasoner, PiRuvIoClient, SecretManager};
use crate::{CycleResult, consolidation::SonaConsolidator};
use anyhow::Result;
use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Daily learning scheduler
pub struct DailyLearningScheduler {
    /// Configuration
    config: SchedulerConfig,

    /// Current world state
    state: LearningWorldState,

    /// GOAP planner
    planner: GoapPlanner,

    /// Pattern analyzer
    analyzer: PatternAnalyzer,

    /// π.ruv.io client
    pi_client: PiRuvIoClient,

    /// Gemini reasoner
    gemini: Option<GeminiGoapReasoner>,

    /// Secret manager
    secrets: SecretManager,

    /// SONA consolidator
    consolidator: SonaConsolidator,

    /// Last cycle time
    last_cycle: Option<DateTime<Utc>>,

    /// Running state
    running: bool,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Primary run time (hour in UTC, e.g., 2 for 02:00)
    pub primary_run_hour: u32,

    /// Minimum hours between cycles
    pub min_cycle_interval_hours: u32,

    /// Maximum cycles per day
    pub max_daily_cycles: u32,

    /// Trigger configuration
    pub triggers: TriggerConfig,

    /// Quality configuration
    pub quality: QualityConfig,

    /// Scan configuration
    pub scan: ScanConfig,
}

/// Trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerConfig {
    /// Git commits threshold
    pub git_commit_threshold: usize,

    /// Idle period in minutes
    pub idle_period_minutes: u32,

    /// Trajectory buffer fill threshold (0.0-1.0)
    pub trajectory_buffer_threshold: f64,

    /// New patterns threshold
    pub new_patterns_threshold: usize,
}

/// Quality configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Minimum quality score to submit
    pub minimum_score: f64,

    /// Novelty weight in composite score
    pub novelty_weight: f64,

    /// Usefulness weight
    pub usefulness_weight: f64,

    /// Clarity weight
    pub clarity_weight: f64,

    /// Correctness weight
    pub correctness_weight: f64,

    /// Generalizability weight
    pub generalizability_weight: f64,
}

/// Scan configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanConfig {
    /// Root directory to scan
    pub root_directory: String,

    /// File extensions to include
    pub extensions: Vec<String>,

    /// Directories to exclude
    pub exclude_dirs: Vec<String>,

    /// Maximum file size in bytes
    pub max_file_size: u64,
}

/// Reason for triggering a cycle
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerReason {
    /// Scheduled run
    Scheduled,

    /// Manual request
    Manual,

    /// Git commit threshold reached
    GitCommits { count: usize },

    /// Idle period reached
    IdlePeriod { minutes: u32 },

    /// Trajectory buffer threshold
    TrajectoryBuffer { fill_ratio: f64 },

    /// New patterns available
    NewPatterns { count: usize },

    /// CI/CD completion
    CiCdCompletion,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            primary_run_hour: 2, // 02:00 UTC
            min_cycle_interval_hours: 6,
            max_daily_cycles: 4,
            triggers: TriggerConfig::default(),
            quality: QualityConfig::default(),
            scan: ScanConfig::default(),
        }
    }
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            git_commit_threshold: 10,
            idle_period_minutes: 30,
            trajectory_buffer_threshold: 0.8,
            new_patterns_threshold: 20,
        }
    }
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            minimum_score: 0.7,
            novelty_weight: 0.3,
            usefulness_weight: 0.25,
            clarity_weight: 0.15,
            correctness_weight: 0.2,
            generalizability_weight: 0.1,
        }
    }
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            root_directory: ".".to_string(),
            extensions: vec![
                "rs".to_string(),
                "ts".to_string(),
                "js".to_string(),
                "py".to_string(),
            ],
            exclude_dirs: vec![
                "node_modules".to_string(),
                "target".to_string(),
                ".git".to_string(),
            ],
            max_file_size: 1024 * 1024,
        }
    }
}

impl DailyLearningScheduler {
    /// Create a new scheduler
    pub async fn new(config: SchedulerConfig) -> Result<Self> {
        let mut secrets = SecretManager::default_project();

        // Try to initialize Gemini if API key is available
        let gemini = match secrets.get_gemini_api_key().await {
            Ok(key) => Some(GeminiGoapReasoner::new(key)),
            Err(e) => {
                tracing::warn!("Gemini not available: {}", e);
                None
            }
        };

        let mut state = LearningWorldState::default();
        state.gemini_available = gemini.is_some();

        Ok(Self {
            config,
            state,
            planner: GoapPlanner::new(),
            analyzer: PatternAnalyzer::new(),
            pi_client: PiRuvIoClient::default_client(),
            gemini,
            secrets,
            consolidator: SonaConsolidator::new(),
            last_cycle: None,
            running: false,
        })
    }

    /// Check if a cycle should trigger
    pub fn should_trigger(&self) -> Option<TriggerReason> {
        let now = Utc::now();

        // Check scheduled time
        if now.hour() == self.config.primary_run_hour && self.can_run_cycle() {
            return Some(TriggerReason::Scheduled);
        }

        // Check idle period
        if let Some(last) = self.last_cycle {
            let idle_minutes = (now - last).num_minutes() as u32;
            if idle_minutes >= self.config.triggers.idle_period_minutes && self.can_run_cycle() {
                return Some(TriggerReason::IdlePeriod {
                    minutes: idle_minutes,
                });
            }
        }

        // Check trajectory buffer (would need actual buffer state)
        // This is a placeholder - integrate with actual SONA state

        None
    }

    /// Check if we can run a cycle
    fn can_run_cycle(&self) -> bool {
        if let Some(last) = self.last_cycle {
            let hours_since = (Utc::now() - last).num_hours() as u32;
            hours_since >= self.config.min_cycle_interval_hours
        } else {
            true
        }
    }

    /// Run a learning cycle
    pub async fn run_cycle(&mut self) -> Result<CycleResult> {
        let started_at = Utc::now();
        let start_instant = Instant::now();
        let cycle_id = Uuid::new_v4();

        tracing::info!("Starting learning cycle {}", cycle_id);

        let mut errors = Vec::new();
        let mut discoveries: Vec<DiscoveryLog> = Vec::new();

        // Phase 1: Scan codebase
        self.state.scanning = true;
        let scan_result = self.scan_codebase().await;
        self.state.scanning = false;

        match scan_result {
            Ok(found) => {
                self.state.patterns_discovered = found.len();
                discoveries.extend(found);
                tracing::info!("Discovered {} patterns", discoveries.len());
            }
            Err(e) => {
                errors.push(format!("Scan failed: {}", e));
                tracing::error!("Scan error: {}", e);
            }
        }

        // Phase 2: Quality assessment (with Gemini if available)
        if self.gemini.is_some() && !discoveries.is_empty() {
            if let Err(e) = self.assess_quality(&mut discoveries).await {
                errors.push(format!("Quality assessment failed: {}", e));
            }
        }

        // Filter by quality threshold
        let quality_threshold = self.config.quality.minimum_score;
        let high_quality: Vec<_> = discoveries
            .iter()
            .filter(|d| d.quality.composite >= quality_threshold)
            .cloned()
            .collect();

        // Phase 3: Submit to π.ruv.io
        let mut submitted = 0;
        self.state.pi_ruv_io_connected = self.pi_client.check_connection().await;

        if self.state.pi_ruv_io_connected {
            for discovery in &high_quality {
                if let Err(e) = self.pi_client.submit(discovery).await {
                    errors.push(format!("Submission failed: {}", e));
                } else {
                    submitted += 1;
                }
            }
            self.state.patterns_pending_submission = high_quality.len() - submitted;
        }

        // Phase 4: Consolidation
        let consolidated = if self.state.needs_consolidation() {
            match self.consolidator.consolidate(&discoveries) {
                Ok(count) => {
                    self.state.consolidation_due = false;
                    count
                }
                Err(e) => {
                    errors.push(format!("Consolidation failed: {}", e));
                    0
                }
            }
        } else {
            0
        };

        // Update state
        self.last_cycle = Some(Utc::now());
        self.state.cycle_count += 1;

        let duration_ms = start_instant.elapsed().as_millis() as u64;

        let result = CycleResult {
            id: cycle_id,
            started_at,
            duration_ms,
            discoveries_found: discoveries.len(),
            discoveries_submitted: submitted,
            patterns_consolidated: consolidated,
            final_state: self.state.clone(),
            errors,
        };

        tracing::info!(
            "Cycle {} complete: {} discovered, {} submitted, {} consolidated in {}ms",
            cycle_id,
            result.discoveries_found,
            result.discoveries_submitted,
            result.patterns_consolidated,
            result.duration_ms
        );

        Ok(result)
    }

    /// Scan the codebase for patterns
    async fn scan_codebase(&self) -> Result<Vec<DiscoveryLog>> {
        let scanner = CodebaseScanner::new(&self.config.scan.root_directory)
            .with_extensions(self.config.scan.extensions.clone())
            .with_exclude_dirs(self.config.scan.exclude_dirs.clone())
            .with_max_file_size(self.config.scan.max_file_size);

        let files = scanner.scan().await?;

        let file_contents: Vec<(String, String)> = files
            .into_iter()
            .map(|f| (f.path.to_string_lossy().to_string(), f.content))
            .collect();

        Ok(self.analyzer.analyze_files(&file_contents))
    }

    /// Assess quality using Gemini
    async fn assess_quality(&self, discoveries: &mut [DiscoveryLog]) -> Result<()> {
        let gemini = match &self.gemini {
            Some(g) => g,
            None => return Ok(()),
        };

        for discovery in discoveries.iter_mut() {
            match gemini
                .assess_quality(
                    &discovery.title,
                    &discovery.description,
                    discovery.code_snippet.as_deref(),
                )
                .await
            {
                Ok(response) => {
                    discovery.quality.novelty = response.novelty;
                    discovery.quality.usefulness = response.usefulness;
                    discovery.quality.clarity = response.clarity;
                    discovery.quality.correctness = response.correctness;
                    discovery.quality.generalizability = response.generalizability;
                    discovery.quality.compute_composite();
                    discovery.quality.method = crate::discovery::AssessmentMethod::Gemini;
                }
                Err(e) => {
                    tracing::warn!("Quality assessment failed for {}: {}", discovery.title, e);
                }
            }
        }

        Ok(())
    }

    /// Start the scheduled loop (blocking)
    pub async fn start(&mut self) -> Result<()> {
        self.running = true;
        tracing::info!("Starting daily learning scheduler");

        while self.running {
            if let Some(reason) = self.should_trigger() {
                tracing::info!("Trigger detected: {:?}", reason);
                if let Err(e) = self.run_cycle().await {
                    tracing::error!("Cycle failed: {}", e);
                }
            }

            // Check every minute
            tokio::time::sleep(Duration::from_secs(60)).await;
        }

        Ok(())
    }

    /// Stop the scheduler
    pub async fn stop(&mut self) -> Result<()> {
        self.running = false;
        tracing::info!("Stopping daily learning scheduler");
        Ok(())
    }

    /// Get current state
    pub fn state(&self) -> &LearningWorldState {
        &self.state
    }

    /// Get last cycle time
    pub fn last_cycle(&self) -> Option<DateTime<Utc>> {
        self.last_cycle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert_eq!(config.primary_run_hour, 2);
        assert_eq!(config.max_daily_cycles, 4);
    }

    #[test]
    fn test_quality_config() {
        let config = QualityConfig::default();
        let total = config.novelty_weight
            + config.usefulness_weight
            + config.clarity_weight
            + config.correctness_weight
            + config.generalizability_weight;
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_trigger_reason_equality() {
        assert_eq!(TriggerReason::Scheduled, TriggerReason::Scheduled);
        assert_ne!(
            TriggerReason::GitCommits { count: 10 },
            TriggerReason::GitCommits { count: 20 }
        );
    }
}
