//! GOAP action definitions for the learning loop

use super::state::{LearningWorldState, StateValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A GOAP action with preconditions, effects, and cost
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoapAction {
    /// Action identifier
    pub name: String,

    /// Human-readable description
    pub description: String,

    /// Preconditions that must be true
    pub preconditions: HashMap<String, StateValue>,

    /// Effects applied when action completes
    pub effects: HashMap<String, StateValue>,

    /// Cost of executing this action
    pub cost: f64,

    /// How to execute the action
    pub executor: ActionExecutor,
}

/// How an action is executed
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ActionExecutor {
    /// Native Rust function
    Native {
        function: String,
    },

    /// rvagent tool call
    RvAgent {
        tool: String,
        params: serde_json::Value,
    },

    /// MCP tool call
    Mcp {
        server: String,
        tool: String,
        params: serde_json::Value,
    },

    /// Gemini reasoning
    Gemini {
        prompt_template: String,
    },

    /// Composite action (runs multiple sub-actions)
    Composite {
        actions: Vec<String>,
        parallel: bool,
    },
}

/// Result of executing an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    /// Whether the action succeeded
    pub success: bool,

    /// Updated state effects
    pub effects: HashMap<String, StateValue>,

    /// Any discovery made
    pub discovery: Option<crate::discovery::DiscoveryLog>,

    /// Execution duration in milliseconds
    pub duration_ms: u64,

    /// Error message if failed
    pub error: Option<String>,

    /// Raw output from executor
    pub output: serde_json::Value,
}

impl GoapAction {
    /// Check if preconditions are satisfied by current state
    pub fn preconditions_met(&self, state: &LearningWorldState) -> bool {
        for (key, expected) in &self.preconditions {
            let actual = state_value_for_key(state, key);
            if actual != *expected {
                return false;
            }
        }
        true
    }
}

/// Get state value for a given key
fn state_value_for_key(state: &LearningWorldState, key: &str) -> StateValue {
    match key {
        "scanning" => StateValue::Bool(state.scanning),
        "pi_ruv_io_connected" => StateValue::Bool(state.pi_ruv_io_connected),
        "gemini_available" => StateValue::Bool(state.gemini_available),
        "consolidation_due" => StateValue::Bool(state.consolidation_due),
        "patterns_discovered" => StateValue::Int(state.patterns_discovered as i64),
        "patterns_pending_assessment" => StateValue::Int(state.patterns_pending_assessment as i64),
        "patterns_pending_submission" => StateValue::Int(state.patterns_pending_submission as i64),
        "api_quota_remaining" => StateValue::Int(state.api_quota_remaining as i64),
        "memory_utilization" => StateValue::Float(state.memory_utilization),
        _ => StateValue::Bool(false), // Unknown keys default to false
    }
}

/// Registry of all available GOAP actions
pub struct ActionRegistry {
    actions: HashMap<String, GoapAction>,
}

impl ActionRegistry {
    /// Create registry with default actions
    pub fn new() -> Self {
        let mut actions = HashMap::new();

        // Scan codebase for patterns
        actions.insert("scan_codebase".to_string(), GoapAction {
            name: "scan_codebase".to_string(),
            description: "Scan codebase to discover patterns and optimizations".to_string(),
            preconditions: {
                let mut p = HashMap::new();
                p.insert("scanning".to_string(), StateValue::Bool(false));
                p
            },
            effects: {
                let mut e = HashMap::new();
                e.insert("scanning".to_string(), StateValue::Bool(true));
                e
            },
            cost: 10.0,
            executor: ActionExecutor::Native {
                function: "discovery::scanner::scan_codebase".to_string(),
            },
        });

        // Analyze discovered patterns
        actions.insert("analyze_patterns".to_string(), GoapAction {
            name: "analyze_patterns".to_string(),
            description: "Analyze raw patterns for quality and novelty".to_string(),
            preconditions: HashMap::new(), // patterns_pending > 0 checked at runtime
            effects: HashMap::new(), // +assessed_patterns
            cost: 5.0,
            executor: ActionExecutor::Native {
                function: "discovery::analyzer::analyze_patterns".to_string(),
            },
        });

        // Compute novelty score using Gemini
        actions.insert("compute_novelty".to_string(), GoapAction {
            name: "compute_novelty".to_string(),
            description: "Use Gemini to compute novelty score for a pattern".to_string(),
            preconditions: {
                let mut p = HashMap::new();
                p.insert("gemini_available".to_string(), StateValue::Bool(true));
                p
            },
            effects: HashMap::new(), // novelty_score computed
            cost: 3.0,
            executor: ActionExecutor::Gemini {
                prompt_template: include_str!("../prompts/novelty_assessment.txt").to_string(),
            },
        });

        // Log discovery with provenance
        actions.insert("log_discovery".to_string(), GoapAction {
            name: "log_discovery".to_string(),
            description: "Log a discovery with full method provenance".to_string(),
            preconditions: HashMap::new(), // novelty > threshold checked at runtime
            effects: HashMap::new(), // +logged_discovery
            cost: 2.0,
            executor: ActionExecutor::Native {
                function: "discovery::logger::log_discovery".to_string(),
            },
        });

        // Submit to π.ruv.io
        actions.insert("submit_to_pi".to_string(), GoapAction {
            name: "submit_to_pi".to_string(),
            description: "Submit logged discovery to π.ruv.io cloud brain".to_string(),
            preconditions: {
                let mut p = HashMap::new();
                p.insert("pi_ruv_io_connected".to_string(), StateValue::Bool(true));
                p
            },
            effects: HashMap::new(), // -pending, +submitted
            cost: 8.0,
            executor: ActionExecutor::Mcp {
                server: "pi-brain".to_string(),
                tool: "brain_share".to_string(),
                params: serde_json::json!({}),
            },
        });

        // Consolidate learning with SONA
        actions.insert("consolidate_sona".to_string(), GoapAction {
            name: "consolidate_sona".to_string(),
            description: "Consolidate learned patterns using SONA to prevent forgetting".to_string(),
            preconditions: {
                let mut p = HashMap::new();
                p.insert("consolidation_due".to_string(), StateValue::Bool(true));
                p
            },
            effects: {
                let mut e = HashMap::new();
                e.insert("consolidation_due".to_string(), StateValue::Bool(false));
                e
            },
            cost: 15.0,
            executor: ActionExecutor::Native {
                function: "consolidation::sona::consolidate".to_string(),
            },
        });

        // Refresh π.ruv.io connection
        actions.insert("refresh_connection".to_string(), GoapAction {
            name: "refresh_connection".to_string(),
            description: "Refresh connection to π.ruv.io".to_string(),
            preconditions: {
                let mut p = HashMap::new();
                p.insert("pi_ruv_io_connected".to_string(), StateValue::Bool(false));
                p
            },
            effects: {
                let mut e = HashMap::new();
                e.insert("pi_ruv_io_connected".to_string(), StateValue::Bool(true));
                e
            },
            cost: 5.0,
            executor: ActionExecutor::Native {
                function: "integration::pi_ruvio::refresh_connection".to_string(),
            },
        });

        // Reason with Gemini for planning
        actions.insert("reason_with_gemini".to_string(), GoapAction {
            name: "reason_with_gemini".to_string(),
            description: "Use Gemini 2.5 Flash for GOAP planning and reasoning".to_string(),
            preconditions: {
                let mut p = HashMap::new();
                p.insert("gemini_available".to_string(), StateValue::Bool(true));
                p
            },
            effects: HashMap::new(), // +reasoning_result
            cost: 20.0,
            executor: ActionExecutor::Gemini {
                prompt_template: include_str!("../prompts/goap_planning.txt").to_string(),
            },
        });

        Self { actions }
    }

    /// Get an action by name
    pub fn get(&self, name: &str) -> Option<&GoapAction> {
        self.actions.get(name)
    }

    /// List all action names
    pub fn list(&self) -> Vec<&str> {
        self.actions.keys().map(|s| s.as_str()).collect()
    }

    /// Get all actions
    pub fn all(&self) -> &HashMap<String, GoapAction> {
        &self.actions
    }
}

impl Default for ActionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_registry() {
        let registry = ActionRegistry::new();
        assert!(registry.get("scan_codebase").is_some());
        assert!(registry.get("submit_to_pi").is_some());
        assert!(registry.list().len() >= 8);
    }

    #[test]
    fn test_preconditions_check() {
        let registry = ActionRegistry::new();
        let action = registry.get("scan_codebase").unwrap();

        let mut state = LearningWorldState::default();
        assert!(action.preconditions_met(&state));

        state.scanning = true;
        assert!(!action.preconditions_met(&state));
    }
}
