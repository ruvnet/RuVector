//! rvAgent subagents — specification, compilation, orchestration, and result validation.
//!
//! This crate implements:
//! - `SubAgentSpec`: declarative subagent definition
//! - `CompiledSubAgent`: a spec compiled into a runnable graph
//! - `SubAgentResult`: outcome of a subagent execution
//! - `SubAgentOrchestrator`: spawn/parallel execution
//! - `SubAgentResultValidator`: security validation (ADR-103 C8)

pub mod builder;
pub mod crdt_merge;
pub mod orchestrator;
pub mod prompts;
pub mod result_validator;
pub mod validator;

// Re-export result validator types
pub use result_validator::{
    SubAgentResultValidator, ValidationConfig, ValidationError, DEFAULT_MAX_RESPONSE_LENGTH,
};

// Re-export CRDT merge types
pub use crdt_merge::{merge_subagent_results, CrdtState, MergeError, VectorClock};

// Re-export orchestrator types
pub use orchestrator::{spawn_parallel, SpawnError, SubAgentOrchestrator};

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

use rvagent_core::messages::Message;

// ---------------------------------------------------------------------------
// AgentState — canonical typed state from rvagent-core (ADR-103 A1)
// ---------------------------------------------------------------------------

/// Agent state — the canonical typed `rvagent_core::state::AgentState`.
///
/// Replaces the former `HashMap<String, serde_json::Value>` alias (ADR-097)
/// per ADR-103 A1 / roadmap P0.1.
pub use rvagent_core::state::AgentState;

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
/// State-isolation semantics (ADR-097) on the typed state: parent
/// `messages`, `todos`, `memory_contents`, and `skills_metadata` are
/// excluded; `files` pass through (O(1) Arc clone); a single human message
/// containing the task description is injected.
pub fn prepare_subagent_state(parent_state: &AgentState, task_description: &str) -> AgentState {
    let mut state = AgentState::new();
    // Files are not in EXCLUDED_STATE_KEYS — they pass through to the child.
    state.files = Arc::clone(&parent_state.files);
    state.push_message(Message::human(task_description));
    state
}

/// Extract the final message from a subagent's result state.
pub fn extract_result_message(result_state: &AgentState) -> Option<String> {
    result_state
        .messages
        .last()
        .map(|m| m.content().trim_end().to_string())
}

/// Merge non-excluded state from subagent result back into parent state.
///
/// Only non-excluded state merges back: `files` (subagent wins on path
/// conflict). Parent `messages` and `todos` are never overwritten.
pub fn merge_subagent_state(parent: &mut AgentState, subagent_result: &AgentState) {
    for (path, data) in subagent_result.files.iter() {
        parent.set_file(path.clone(), data.clone());
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

    use rvagent_core::state::{FileData, TodoItem, TodoStatus};

    fn file(content: &str) -> FileData {
        FileData {
            content: content.into(),
            encoding: "utf-8".into(),
            modified_at: None,
        }
    }

    #[test]
    fn test_state_isolation_prepare() {
        let mut parent = AgentState::new();
        parent.push_message(Message::ai("secret"));
        parent.push_todo(TodoItem {
            content: "parent todo".into(),
            status: TodoStatus::Pending,
            active_form: String::new(),
        });
        parent.set_file("/src/main.rs", file("fn main() {}"));
        parent.memory_contents = Some(std::sync::Arc::new(
            [("AGENTS.md".to_string(), "secret memory".to_string())]
                .into_iter()
                .collect(),
        ));

        let child = prepare_subagent_state(&parent, "Do X");

        // Parent messages must NOT leak — child gets exactly one human message.
        assert_eq!(child.message_count(), 1);
        assert_eq!(child.messages[0].content(), "Do X");
        assert!(matches!(child.messages[0], Message::Human(_)));

        // Excluded state must not appear.
        assert!(child.todos.is_empty());
        assert!(child.memory_contents.is_none());
        assert!(child.skills_metadata.is_none());

        // Files pass through (non-excluded state).
        assert!(child.files.contains_key("/src/main.rs"));
    }

    #[test]
    fn test_extract_result_message() {
        let mut state = AgentState::new();
        state.push_message(Message::human("do X"));
        state.push_message(Message::ai("Done with X.  "));
        let msg = extract_result_message(&state).unwrap();
        assert_eq!(msg, "Done with X.");
    }

    #[test]
    fn test_merge_subagent_state() {
        let mut parent = AgentState::new();
        parent.push_message(Message::human("parent message"));
        parent.set_file("/existing.rs", file("existing"));

        let mut child_result = AgentState::new();
        child_result.push_message(Message::ai("hi"));
        child_result.push_todo(TodoItem {
            content: "leaked".into(),
            status: TodoStatus::Pending,
            active_form: String::new(),
        });
        child_result.set_file("/new.rs", file("added"));

        merge_subagent_state(&mut parent, &child_result);

        // messages should NOT be overwritten (excluded)
        assert_eq!(parent.message_count(), 1);
        assert_eq!(parent.messages[0].content(), "parent message");
        // todos should NOT leak
        assert!(parent.todos.is_empty());
        // new non-excluded state (files) should merge
        assert!(parent.files.contains_key("/new.rs"));
        assert!(parent.files.contains_key("/existing.rs"));
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
