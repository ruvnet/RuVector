//! rvagent-learning: Daily Learning and Optimization Loop with GOAP Reasoning
//!
//! This crate implements ADR-115, providing:
//! - Goal-Oriented Action Planning (GOAP) for intelligent discovery
//! - Discovery logging with full provenance
//! - π.ruv.io integration for collective intelligence
//! - Gemini 2.5 Flash reasoning for quality assessment
//! - SONA integration for pattern consolidation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rvagent_learning::{DailyLearningLoop, SchedulerConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = SchedulerConfig::default();
//!     let mut loop_runner = DailyLearningLoop::new(config).await?;
//!     loop_runner.run_cycle().await?;
//!     Ok(())
//! }
//! ```

pub mod goap;
pub mod discovery;
pub mod integration;
pub mod scheduler;
pub mod consolidation;

// Re-exports
pub use goap::{GoapPlanner, GoapAction, LearningWorldState, LearningGoal};
pub use discovery::{DiscoveryLog, DiscoveryCategory, ToolUsage, ToolType};
pub use integration::{PiRuvIoClient, GeminiGoapReasoner};
pub use scheduler::{DailyLearningScheduler, SchedulerConfig};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Result of a daily learning cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleResult {
    /// Cycle unique identifier
    pub id: Uuid,

    /// When the cycle started
    pub started_at: DateTime<Utc>,

    /// Total duration
    pub duration_ms: u64,

    /// Number of discoveries found
    pub discoveries_found: usize,

    /// Number submitted to π.ruv.io
    pub discoveries_submitted: usize,

    /// Patterns consolidated to SONA
    pub patterns_consolidated: usize,

    /// World state after cycle
    pub final_state: LearningWorldState,

    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Main entry point for the daily learning loop
pub struct DailyLearningLoop {
    scheduler: DailyLearningScheduler,
}

impl DailyLearningLoop {
    /// Create a new learning loop with configuration
    pub async fn new(config: SchedulerConfig) -> Result<Self> {
        let scheduler = DailyLearningScheduler::new(config).await?;
        Ok(Self { scheduler })
    }

    /// Run a single learning cycle
    pub async fn run_cycle(&mut self) -> Result<CycleResult> {
        self.scheduler.run_cycle().await
    }

    /// Start the scheduled loop (blocking)
    pub async fn start(&mut self) -> Result<()> {
        self.scheduler.start().await
    }

    /// Stop the scheduled loop
    pub async fn stop(&mut self) -> Result<()> {
        self.scheduler.stop().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_result_serialization() {
        let result = CycleResult {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            duration_ms: 1000,
            discoveries_found: 5,
            discoveries_submitted: 3,
            patterns_consolidated: 2,
            final_state: LearningWorldState::default(),
            errors: vec![],
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("discoveries_found"));
    }
}
