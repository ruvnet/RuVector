//! Middleware request/response types built on the canonical `rvagent-core`
//! type system (P0.1 — unified types).
//!
//! `Message`, `ToolCall`, `AgentState`, `TodoItem`, `TodoStatus`,
//! `RunnableConfig`, and `ToolDefinition` are re-exported from
//! `rvagent_core`; this module only defines the middleware-specific
//! envelope types (`ModelRequest`, `ModelResponse`, `AgentStateUpdate`, …).

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

// Canonical types (single source of truth: rvagent-core).
pub use rvagent_core::config::RunnableConfig;
pub use rvagent_core::messages::{
    AiMessage, HumanMessage, Message, SystemMessage, ToolCall, ToolMessage,
};
pub use rvagent_core::models::ToolDefinition;
pub use rvagent_core::state::{AgentState, FileData, TodoItem, TodoStatus};

/// Cache control hint for prompt caching (Anthropic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    pub cache_type: String,
}

/// State update returned by `before_agent`. Merged into `AgentState`.
///
/// Extensions are stored in the typed extension slot of the core
/// `AgentState` as `serde_json::Value` entries.
#[derive(Debug, Clone, Default)]
pub struct AgentStateUpdate {
    pub messages: Option<Vec<Message>>,
    pub todos: Option<Vec<TodoItem>>,
    pub extensions: HashMap<String, serde_json::Value>,
}

impl AgentStateUpdate {
    /// Merge this update into an `AgentState`.
    pub fn apply_to(self, state: &mut AgentState) {
        if let Some(messages) = self.messages {
            state.messages = Arc::new(messages);
        }
        if let Some(todos) = self.todos {
            state.todos = Arc::new(todos);
        }
        for (k, v) in self.extensions {
            state.set_extension(k, v);
        }
    }
}

/// Read a JSON extension value stored on the core `AgentState`.
pub fn json_extension<'a>(state: &'a AgentState, key: &str) -> Option<&'a serde_json::Value> {
    state.get_extension::<serde_json::Value>(key)
}

/// Model request wrapping messages and configuration.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub system_message: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub cache_control: HashMap<String, CacheControl>,
    pub extensions: HashMap<String, serde_json::Value>,
}

impl ModelRequest {
    /// Create a new model request.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            system_message: None,
            messages,
            tools: vec![],
            cache_control: HashMap::new(),
            extensions: HashMap::new(),
        }
    }

    /// Return a copy with a different system message.
    pub fn with_system(mut self, system_message: Option<String>) -> Self {
        self.system_message = system_message;
        self
    }

    /// Return a copy with different messages.
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }
}

/// Model response from an LLM call.
#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub message: Message,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
}

impl ModelResponse {
    /// Create a simple text response (an AI message).
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            message: Message::ai(content),
            tool_calls: vec![],
            usage: None,
        }
    }

    /// Text content of the response message.
    pub fn content(&self) -> &str {
        self.message.content()
    }
}

/// Token usage information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_creation_tokens: u64,
}

/// Runtime context passed to middleware hooks.
pub struct Runtime {
    pub context: serde_json::Value,
    pub config: RunnableConfig,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            context: serde_json::Value::Null,
            config: RunnableConfig::default(),
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_request_with_system() {
        let req = ModelRequest::new(vec![Message::human("hi")]);
        assert!(req.system_message.is_none());
        let req2 = req.with_system(Some("system".into()));
        assert_eq!(req2.system_message, Some("system".into()));
    }

    #[test]
    fn test_model_response_text() {
        let resp = ModelResponse::text("hello");
        assert_eq!(resp.content(), "hello");
        assert!(matches!(resp.message, Message::Ai(_)));
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn test_runtime_default() {
        let rt = Runtime::default();
        assert_eq!(rt.context, serde_json::Value::Null);
    }

    #[test]
    fn test_agent_state_update_apply() {
        let mut state = AgentState::default();
        let mut update = AgentStateUpdate::default();
        update.messages = Some(vec![Message::human("hi")]);
        update.extensions.insert("k".into(), serde_json::json!("v"));
        update.apply_to(&mut state);
        assert_eq!(state.message_count(), 1);
        assert_eq!(json_extension(&state, "k"), Some(&serde_json::json!("v")));
    }
}
