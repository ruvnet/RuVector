//! Goal-Oriented Action Planning (GOAP) for intelligent discovery
//!
//! GOAP is a planning system that uses A* search to find optimal action sequences
//! to achieve goals based on current world state.

mod state;
mod actions;
mod planner;

pub use state::LearningWorldState;
pub use actions::{GoapAction, ActionExecutor, ActionResult};
pub use planner::GoapPlanner;

use serde::{Deserialize, Serialize};

/// Goals for the learning loop
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LearningGoal {
    /// Discover N new patterns
    DiscoverPatterns {
        target_count: usize,
    },

    /// Submit discoveries to π.ruv.io
    SubmitToCloudBrain {
        min_quality: f64,
    },

    /// Consolidate learned patterns (prevent forgetting)
    ConsolidateLearning,

    /// Optimize specific domain
    OptimizeDomain {
        domain: String,
    },

    /// Complete full daily learning cycle
    CompleteDailyCycle,

    /// Explore specific files for patterns
    ExploreFiles {
        paths: Vec<String>,
    },
}

/// A planned sequence of actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoapPlan {
    /// Ordered list of actions to execute
    pub actions: Vec<PlannedAction>,

    /// Total estimated cost
    pub estimated_cost: f64,

    /// Reasoning for this plan
    pub reasoning: String,

    /// Expected final state
    pub expected_state: LearningWorldState,
}

/// A single action in a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedAction {
    /// Action name
    pub action: String,

    /// Parameters for the action
    pub params: serde_json::Value,

    /// Actions that can run in parallel with this one
    pub parallel_with: Vec<String>,

    /// Estimated cost
    pub cost: f64,
}

impl GoapPlan {
    /// Create an empty plan
    pub fn empty() -> Self {
        Self {
            actions: vec![],
            estimated_cost: 0.0,
            reasoning: "No actions needed".to_string(),
            expected_state: LearningWorldState::default(),
        }
    }

    /// Check if the plan is empty
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}
