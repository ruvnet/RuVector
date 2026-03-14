//! rvAgent subagents — specification, compilation, orchestration, and result validation.
//!
//! This crate implements:
//! - `SubAgentSpec`: declarative subagent definition
//! - `CompiledSubAgent`: a spec compiled into a runnable graph
//! - `SubAgentResult`: outcome of a subagent execution
//! - `SubAgentOrchestrator`: spawn/parallel execution
//! - `SubAgentResultValidator`: security validation (ADR-103 C8)

pub mod builder;
pub mod orchestrator;
pub mod prompts;
pub mod validator;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// AgentState (simplified, JSON-based for cross-crate compatibility)
// ---------------------------------------------------------------------------

/// Agent state represented as a JSON map.
///
/// Matches `HashMap<String, serde_json::Value>` from ADR-097.
/// Future work (ADR-103 A1) will replace this with a typed struct.
pub type AgentState = HashMap<String, serde_json::Value>;

// ---------------------------------------------------------------------------
// RvAgentConfig
// ---------------------------------------------------------------------------

/// Minimal agent configuration passed to subagent compilation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvAgentConfig {
    /// Default model identifier (e.g. "anthropic:claude-sonnet-4-20250514").
    #[serde(default)]
    pub default_model: Option<String>,

    /// Tools available to the parent agent.
    #[serde(default)]
    pub tools: Vec<String>,

    /// Middleware names enabled on the parent agent.
    #[serde(default)]
    pub middleware: Vec<String>,

    /// Working directory for file operations.
    #[serde(default)]
    pub cwd: Option<String>,
}

impl Default for RvAgentConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            tools: Vec::new(),
            middleware: Vec::new(),
            cwd: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SubAgentSpec
// ---------------------------------------------------------------------------

/// Declarative specification for a subagent (not yet compiled).
///
/// Maps to Python `SubAgent(TypedDict)` from ADR-097.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentSpec {
    /// Unique name identifying this subagent type.
    pub name: String,

    /// Model identifier override (uses parent model if `None`).
    #[serde(default)]
    pub model: Option<String>,

    /// System prompt / instructions for this subagent.
    pub instructions: String,

    /// Tool names this subagent is allowed to use.
    #[serde(default)]
    pub tools: Vec<String>,

    /// Human-readable description for handoff messages.
    #[serde(default)]
    pub handoff_description: Option<String>,

    /// Whether this subagent can read files.
    #[serde(default = "default_true")]
    pub can_read: bool,

    /// Whether this subagent can write files.
    #[serde(default)]
    pub can_write: bool,

    /// Whether this subagent can execute shell commands.
    #[serde(default)]
    pub can_execute: bool,
}

fn default_true() -> bool {
    true
}

impl SubAgentSpec {
    /// Create a new spec with minimal required fields.
    pub fn new(name: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            model: None,
            instructions: instructions.into(),
            tools: Vec::new(),
            handoff_description: None,
            can_read: true,
            can_write: false,
            can_execute: false,
        }
    }

    /// Build a general-purpose subagent that mirrors the parent's tools.
    pub fn general_purpose() -> Self {
        Self {
            name: GENERAL_PURPOSE_NAME.to_string(),
            model: None,
            instructions: DEFAULT_SUBAGENT_PROMPT.to_string(),
            tools: Vec::new(), // inherits parent tools
            handoff_description: Some(GENERAL_PURPOSE_DESCRIPTION.to_string()),
            can_read: true,
            can_write: true,
            can_execute: true,
        }
    }
}

/// Name constant for the general-purpose subagent.
pub const GENERAL_PURPOSE_NAME: &str = "general-purpose";

/// Description for the general-purpose subagent.
pub const GENERAL_PURPOSE_DESCRIPTION: &str =
    "General-purpose agent for researching complex questions, searching for files \
     and content, and executing multi-step tasks. When you are searching for a keyword \
     or file and are not confident that you will find the right match in the first few \
     tries use this agent to perform the search for you. This agent has access to all \
     tools as the main agent.";

/// Default system prompt for subagents.
pub const DEFAULT_SUBAGENT_PROMPT: &str =
    "In order to complete the objective that the user asks of you, you have access \
     to a number of standard tools.";

// ---------------------------------------------------------------------------
// CompiledSubAgent
// ---------------------------------------------------------------------------

/// A subagent spec that has been compiled into a runnable form.
///
/// Contains the original spec plus the compiled graph and middleware pipeline.
#[derive(Debug, Clone)]
pub struct CompiledSubAgent {
    /// The original specification.
    pub spec: SubAgentSpec,

    /// Serialized graph representation (adjacency list of node names).
    pub graph: Vec<String>,

    /// Middleware names applied to this subagent (subset of parent's pipeline).
    pub middleware_pipeline: Vec<String>,

    /// Backend identifier used by this subagent.
    pub backend: String,
}

// ---------------------------------------------------------------------------
// SubAgentResult
// ---------------------------------------------------------------------------

/// The outcome of executing a subagent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentResult {
    /// Name of the subagent that produced this result.
    pub agent_name: String,

    /// The final message content returned by the subagent.
    pub result_message: String,

    /// Number of tool calls the subagent made during execution.
    pub tool_calls_count: usize,

    /// Wall-clock duration of the subagent execution.
    pub duration: Duration,
}

// ---------------------------------------------------------------------------
// State isolation constants (ADR-097)
// ---------------------------------------------------------------------------

/// Keys excluded when passing state to/from subagents.
///
/// These keys contain parent-specific data that must not leak into subagent
/// context (messages, todos, structured responses, etc.).
pub const EXCLUDED_STATE_KEYS: &[&str] = &[
    "messages",
    "remaining_steps",
    "task_completion",
    "todos",
    "structured_response",
    "skills_metadata",
    "memory_contents",
];

/// Prepare a filtered state for subagent invocation.
///
/// Strips excluded keys from the parent state, then injects a single
/// human message containing the task description.
pub fn prepare_subagent_state(parent_state: &AgentState, task_description: &str) -> AgentState {
    let mut state: AgentState = parent_state
        .iter()
        .filter(|(k, _)| !EXCLUDED_STATE_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    state.insert(
        "messages".to_string(),
        serde_json::json!([{"type": "human", "content": task_description}]),
    );

    state
}

/// Extract the final message from a subagent's result state.
pub fn extract_result_message(result_state: &AgentState) -> Option<String> {
    let messages = result_state.get("messages")?;
    let arr = messages.as_array()?;
    let last = arr.last()?;
    last.get("content").and_then(|c| c.as_str()).map(|s| s.trim_end().to_string())
}

/// Merge non-excluded state from subagent result back into parent state.
pub fn merge_subagent_state(parent: &mut AgentState, subagent_result: &AgentState) {
    for (k, v) in subagent_result {
        if !EXCLUDED_STATE_KEYS.contains(&k.as_str()) {
            parent.insert(k.clone(), v.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subagent_spec_new() {
        let spec = SubAgentSpec::new("test-agent", "Do the thing.");
        assert_eq!(spec.name, "test-agent");
        assert_eq!(spec.instructions, "Do the thing.");
        assert!(spec.can_read);
        assert!(!spec.can_write);
        assert!(!spec.can_execute);
        assert!(spec.tools.is_empty());
        assert!(spec.model.is_none());
    }

    #[test]
    fn test_general_purpose_spec() {
        let spec = SubAgentSpec::general_purpose();
        assert_eq!(spec.name, GENERAL_PURPOSE_NAME);
        assert!(spec.can_read);
        assert!(spec.can_write);
        assert!(spec.can_execute);
    }

    #[test]
    fn test_subagent_spec_serde_roundtrip() {
        let spec = SubAgentSpec {
            name: "researcher".into(),
            model: Some("anthropic:claude-sonnet-4-20250514".into()),
            instructions: "Research the topic.".into(),
            tools: vec!["grep".into(), "read_file".into()],
            handoff_description: Some("Researches topics".into()),
            can_read: true,
            can_write: false,
            can_execute: false,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: SubAgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "researcher");
        assert_eq!(back.tools.len(), 2);
    }

    #[test]
    fn test_state_isolation_prepare() {
        let mut parent = AgentState::new();
        parent.insert("messages".into(), serde_json::json!([{"type": "ai", "content": "secret"}]));
        parent.insert("remaining_steps".into(), serde_json::json!(5));
        parent.insert("task_completion".into(), serde_json::json!(false));
        parent.insert("custom_key".into(), serde_json::json!("visible"));
        parent.insert("todos".into(), serde_json::json!([]));

        let child = prepare_subagent_state(&parent, "Do X");

        // Parent messages must NOT leak
        let msgs = child.get("messages").unwrap().as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"], "Do X");
        assert_eq!(msgs[0]["type"], "human");

        // Excluded keys must not appear (except messages which is replaced)
        assert!(child.get("remaining_steps").is_none());
        assert!(child.get("task_completion").is_none());
        assert!(child.get("todos").is_none());

        // Non-excluded keys must pass through
        assert_eq!(child.get("custom_key").unwrap(), &serde_json::json!("visible"));
    }

    #[test]
    fn test_extract_result_message() {
        let mut state = AgentState::new();
        state.insert(
            "messages".into(),
            serde_json::json!([
                {"type": "human", "content": "do X"},
                {"type": "ai", "content": "Done with X.  "}
            ]),
        );
        let msg = extract_result_message(&state).unwrap();
        assert_eq!(msg, "Done with X.");
    }

    #[test]
    fn test_merge_subagent_state() {
        let mut parent = AgentState::new();
        parent.insert("messages".into(), serde_json::json!([]));
        parent.insert("existing".into(), serde_json::json!(1));

        let mut child_result = AgentState::new();
        child_result.insert("messages".into(), serde_json::json!([{"type": "ai", "content": "hi"}]));
        child_result.insert("new_key".into(), serde_json::json!("added"));
        child_result.insert("todos".into(), serde_json::json!(["leaked"]));

        merge_subagent_state(&mut parent, &child_result);

        // messages should NOT be overwritten (excluded)
        assert_eq!(parent.get("messages").unwrap(), &serde_json::json!([]));
        // todos should NOT leak
        assert!(parent.get("todos").is_none());
        // new non-excluded keys should merge
        assert_eq!(parent.get("new_key").unwrap(), &serde_json::json!("added"));
    }

    #[test]
    fn test_subagent_result_serde() {
        let result = SubAgentResult {
            agent_name: "coder".into(),
            result_message: "Fixed the bug.".into(),
            tool_calls_count: 3,
            duration: Duration::from_millis(1500),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: SubAgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.agent_name, "coder");
        assert_eq!(back.tool_calls_count, 3);
    }
}
