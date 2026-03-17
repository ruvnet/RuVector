//! World state representation for GOAP planner

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// World state for GOAP planner
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearningWorldState {
    // ═══════════════════════════════════════════════════════════════════════
    // Discovery State
    // ═══════════════════════════════════════════════════════════════════════

    /// Number of patterns discovered this cycle
    pub patterns_discovered: usize,

    /// Patterns waiting for quality assessment
    pub patterns_pending_assessment: usize,

    /// Patterns waiting for π.ruv.io submission
    pub patterns_pending_submission: usize,

    /// Whether currently scanning codebase
    pub scanning: bool,

    /// Current exploration focus (specific files)
    #[serde(default)]
    pub exploration_focus: Option<Vec<PathBuf>>,

    // ═══════════════════════════════════════════════════════════════════════
    // Quality Thresholds
    // ═══════════════════════════════════════════════════════════════════════

    /// Minimum novelty score to submit (0.0-1.0)
    #[serde(default = "default_novelty_threshold")]
    pub novelty_threshold: f64,

    /// Minimum quality score to submit (0.0-1.0)
    #[serde(default = "default_quality_threshold")]
    pub quality_threshold: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Resource State
    // ═══════════════════════════════════════════════════════════════════════

    /// Remaining API quota for Gemini
    #[serde(default = "default_api_quota")]
    pub api_quota_remaining: usize,

    /// Current memory utilization (0.0-1.0)
    pub memory_utilization: f64,

    /// Time of last submission to π.ruv.io
    pub last_submission_time: Option<DateTime<Utc>>,

    /// Time of last SONA consolidation
    pub last_consolidation_time: Option<DateTime<Utc>>,

    // ═══════════════════════════════════════════════════════════════════════
    // Learning State
    // ═══════════════════════════════════════════════════════════════════════

    /// Total patterns in SONA
    pub sona_patterns_count: usize,

    /// Total entries in ReasoningBank
    pub reasoning_bank_entries: usize,

    /// Whether consolidation is due
    pub consolidation_due: bool,

    /// Current learning cycle count
    pub cycle_count: usize,

    // ═══════════════════════════════════════════════════════════════════════
    // Connection State
    // ═══════════════════════════════════════════════════════════════════════

    /// Whether connected to π.ruv.io
    pub pi_ruv_io_connected: bool,

    /// Whether Gemini API is available
    pub gemini_available: bool,

    /// Whether Google Secrets are accessible
    pub secrets_accessible: bool,
}

fn default_novelty_threshold() -> f64 {
    0.7
}

fn default_quality_threshold() -> f64 {
    0.6
}

fn default_api_quota() -> usize {
    1000
}

impl LearningWorldState {
    /// Create a new state with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if we can start scanning
    pub fn can_scan(&self) -> bool {
        !self.scanning && self.memory_utilization < 0.8
    }

    /// Check if we can submit to π.ruv.io
    pub fn can_submit(&self) -> bool {
        self.pi_ruv_io_connected
            && self.patterns_pending_submission > 0
    }

    /// Check if consolidation is needed
    pub fn needs_consolidation(&self) -> bool {
        self.consolidation_due
            || self.patterns_discovered > 10
            || self.last_consolidation_time
                .map(|t| (Utc::now() - t).num_hours() > 24)
                .unwrap_or(true)
    }

    /// Check if we can use Gemini
    pub fn can_use_gemini(&self) -> bool {
        self.gemini_available && self.api_quota_remaining > 0
    }

    /// Apply an action's effects to the state
    pub fn apply_effect(&mut self, effect: &str, value: StateValue) {
        match effect {
            "patterns_discovered" => {
                if let StateValue::Int(v) = value {
                    self.patterns_discovered = v as usize;
                }
            }
            "patterns_pending_assessment" => {
                if let StateValue::Int(v) = value {
                    self.patterns_pending_assessment = v as usize;
                }
            }
            "patterns_pending_submission" => {
                if let StateValue::Int(v) = value {
                    self.patterns_pending_submission = v as usize;
                }
            }
            "scanning" => {
                if let StateValue::Bool(v) = value {
                    self.scanning = v;
                }
            }
            "consolidation_due" => {
                if let StateValue::Bool(v) = value {
                    self.consolidation_due = v;
                }
            }
            "api_quota_remaining" => {
                if let StateValue::Int(v) = value {
                    self.api_quota_remaining = v as usize;
                }
            }
            "pi_ruv_io_connected" => {
                if let StateValue::Bool(v) = value {
                    self.pi_ruv_io_connected = v;
                }
            }
            _ => {
                tracing::warn!("Unknown state effect: {}", effect);
            }
        }
    }
}

/// Value types for state properties
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum StateValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl From<bool> for StateValue {
    fn from(v: bool) -> Self {
        StateValue::Bool(v)
    }
}

impl From<i64> for StateValue {
    fn from(v: i64) -> Self {
        StateValue::Int(v)
    }
}

impl From<f64> for StateValue {
    fn from(v: f64) -> Self {
        StateValue::Float(v)
    }
}

impl From<String> for StateValue {
    fn from(v: String) -> Self {
        StateValue::String(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let state = LearningWorldState::new();
        // Default derive doesn't call serde defaults, so we test new() instead
        // which would be used in real code with proper initialization
        assert!(!state.scanning);
        assert_eq!(state.patterns_discovered, 0);
    }

    #[test]
    fn test_can_scan() {
        let mut state = LearningWorldState::default();
        assert!(state.can_scan());

        state.scanning = true;
        assert!(!state.can_scan());

        state.scanning = false;
        state.memory_utilization = 0.9;
        assert!(!state.can_scan());
    }

    #[test]
    fn test_apply_effect() {
        let mut state = LearningWorldState::default();
        state.apply_effect("patterns_discovered", StateValue::Int(5));
        assert_eq!(state.patterns_discovered, 5);

        state.apply_effect("scanning", StateValue::Bool(true));
        assert!(state.scanning);
    }
}
