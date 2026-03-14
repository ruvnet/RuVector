//! rvAgent WASM — browser and Node.js agent execution.
//!
//! Provides `WasmAgent`, a WASM-bindgen-exported agent that runs entirely
//! in the browser or Node.js. It uses an in-memory virtual filesystem
//! (`WasmStateBackend`) and delegates model calls to JavaScript via
//! `JsModelProvider`.

pub mod backends;
pub mod bridge;
pub mod tools;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use backends::WasmStateBackend;
use bridge::{to_js_value, BridgeMessage, JsModelProvider};
use tools::{TodoItem, ToolRequest, WasmToolExecutor};
#[cfg(test)]
use tools::TodoStatus;

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/// Crate version, kept in sync with Cargo.toml.
const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Agent configuration (WASM-specific, self-contained)
// ---------------------------------------------------------------------------

/// WASM agent configuration, parsed from JSON provided by the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmAgentConfig {
    /// Model identifier (e.g. "anthropic:claude-sonnet-4-20250514").
    #[serde(default = "default_model")]
    pub model: String,

    /// Optional agent name for identification.
    #[serde(default)]
    pub name: Option<String>,

    /// System instructions / base prompt.
    #[serde(default = "default_instructions")]
    pub instructions: String,

    /// Maximum conversation turns before auto-stop.
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
}

fn default_model() -> String {
    "anthropic:claude-sonnet-4-20250514".to_string()
}

fn default_instructions() -> String {
    "You are a helpful coding assistant running in a WASM sandbox.".to_string()
}

fn default_max_turns() -> u32 {
    50
}

impl Default for WasmAgentConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            name: None,
            instructions: default_instructions(),
            max_turns: default_max_turns(),
        }
    }
}

// ---------------------------------------------------------------------------
// Agent state
// ---------------------------------------------------------------------------

/// Serializable agent state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Conversation history.
    pub messages: Vec<BridgeMessage>,
    /// Current turn count.
    pub turn_count: u32,
    /// Whether the agent has been stopped.
    pub stopped: bool,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            turn_count: 0,
            stopped: false,
        }
    }
}

// ---------------------------------------------------------------------------
// WasmAgent — the main exported type
// ---------------------------------------------------------------------------

/// rvAgent WASM — browser and Node.js agent execution.
///
/// Create with `new WasmAgent(configJson)` from JavaScript.
#[wasm_bindgen]
pub struct WasmAgent {
    config: WasmAgentConfig,
    state: AgentState,
    backend: WasmStateBackend,
    todos: Vec<TodoItem>,
    model_provider: Option<JsModelProvider>,
}

#[wasm_bindgen]
impl WasmAgent {
    /// Create a new WasmAgent from a JSON configuration string.
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const agent = new WasmAgent('{"model": "anthropic:claude-sonnet-4-20250514"}');
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmAgent, JsValue> {
        let config: WasmAgentConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("invalid config: {}", e)))?;

        let mut state = AgentState::default();

        // Inject the system prompt as the first message.
        if !config.instructions.is_empty() {
            state
                .messages
                .push(BridgeMessage::system(&config.instructions));
        }

        Ok(Self {
            config,
            state,
            backend: WasmStateBackend::new(),
            todos: Vec::new(),
            model_provider: None,
        })
    }

    /// Attach a JavaScript model provider callback.
    ///
    /// The callback receives a JSON string of messages and must return
    /// a `Promise<string>` with the model response.
    pub fn set_model_provider(&mut self, callback: js_sys::Function) -> Result<(), JsValue> {
        self.model_provider = Some(JsModelProvider::new(callback)?);
        Ok(())
    }

    /// Send a prompt and get a response.
    ///
    /// If a model provider is set, the prompt is sent to the JS model.
    /// Otherwise, returns an echo response for testing.
    pub async fn prompt(&mut self, input: &str) -> Result<JsValue, JsValue> {
        if self.state.stopped {
            return Err(JsValue::from_str("agent is stopped"));
        }

        if self.state.turn_count >= self.config.max_turns {
            self.state.stopped = true;
            return Err(JsValue::from_str("max turns exceeded"));
        }

        // Add the user message.
        self.state.messages.push(BridgeMessage::user(input));
        self.state.turn_count += 1;

        let response_content = if let Some(ref provider) = self.model_provider {
            // Serialize messages and call the JS model provider.
            let messages_json = serde_json::to_string(&self.state.messages)
                .map_err(|e| JsValue::from_str(&format!("serialize error: {}", e)))?;
            provider.complete(&messages_json).await?
        } else {
            // No model provider — return an echo response for testing.
            format!("echo: {}", input)
        };

        // Add the assistant response.
        self.state
            .messages
            .push(BridgeMessage::assistant(&response_content));

        // Check if the response contains a tool call (JSON with "tool" field).
        if let Ok(tool_req) = serde_json::from_str::<ToolRequest>(&response_content) {
            let mut executor = WasmToolExecutor::new(&mut self.backend, &mut self.todos);
            let tool_result = executor.execute(&tool_req);
            let result_json = serde_json::to_string(&tool_result)
                .map_err(|e| JsValue::from_str(&format!("serialize error: {}", e)))?;
            return to_js_value(&serde_json::json!({
                "response": response_content,
                "tool_result": serde_json::from_str::<serde_json::Value>(&result_json)
                    .unwrap_or(serde_json::Value::Null),
            }));
        }

        to_js_value(&serde_json::json!({
            "response": response_content,
        }))
    }

    /// Get the current agent state as JSON.
    pub fn get_state(&self) -> Result<JsValue, JsValue> {
        to_js_value(&self.state)
    }

    /// Get the todo list as JSON.
    pub fn get_todos(&self) -> Result<JsValue, JsValue> {
        to_js_value(&self.todos)
    }

    /// Get the list of available tools.
    pub fn get_tools(&self) -> Result<JsValue, JsValue> {
        to_js_value(&tools::available_tools())
    }

    /// Execute a tool directly by passing a JSON tool request.
    pub fn execute_tool(&mut self, tool_json: &str) -> Result<JsValue, JsValue> {
        let request: ToolRequest = serde_json::from_str(tool_json)
            .map_err(|e| JsValue::from_str(&format!("invalid tool request: {}", e)))?;
        let mut executor = WasmToolExecutor::new(&mut self.backend, &mut self.todos);
        let result = executor.execute(&request);
        to_js_value(&result)
    }

    /// Reset the agent state, clearing messages and turn count.
    pub fn reset(&mut self) {
        self.state = AgentState::default();
        self.todos.clear();

        // Re-inject system prompt.
        if !self.config.instructions.is_empty() {
            self.state
                .messages
                .push(BridgeMessage::system(&self.config.instructions));
        }
    }

    /// Get the crate version.
    pub fn version() -> String {
        VERSION.to_string()
    }

    /// Get the agent name, if configured.
    pub fn name(&self) -> Option<String> {
        self.config.name.clone()
    }

    /// Get the configured model identifier.
    pub fn model(&self) -> String {
        self.config.model.clone()
    }

    /// Get the current turn count.
    pub fn turn_count(&self) -> u32 {
        self.state.turn_count
    }

    /// Check whether the agent is stopped.
    pub fn is_stopped(&self) -> bool {
        self.state.stopped
    }

    /// Get the number of files in the virtual filesystem.
    pub fn file_count(&self) -> usize {
        self.backend.file_count()
    }
}

// ---------------------------------------------------------------------------
// Unit tests (run with `cargo test`, no WASM runtime needed)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_agent_default() {
        let agent = WasmAgent::new("{}").unwrap();
        assert_eq!(agent.config.model, "anthropic:claude-sonnet-4-20250514");
        assert!(!agent.config.instructions.is_empty());
        assert_eq!(agent.state.turn_count, 0);
        assert!(!agent.state.stopped);
        // System message should be injected.
        assert_eq!(agent.state.messages.len(), 1);
        assert_eq!(agent.state.messages[0].role, "system");
    }

    #[test]
    fn test_create_agent_custom_config() {
        let config = r#"{
            "model": "openai:gpt-4o",
            "name": "test-agent",
            "instructions": "Be helpful.",
            "max_turns": 10
        }"#;
        let agent = WasmAgent::new(config).unwrap();
        assert_eq!(agent.config.model, "openai:gpt-4o");
        assert_eq!(agent.config.name.as_deref(), Some("test-agent"));
        assert_eq!(agent.config.instructions, "Be helpful.");
        assert_eq!(agent.config.max_turns, 10);
    }

    #[test]
    fn test_create_agent_invalid_json() {
        let result = WasmAgent::new("{bad json}");
        assert!(result.is_err());
    }

    #[test]
    fn test_version() {
        let v = WasmAgent::version();
        assert_eq!(v, "0.1.0");
    }

    #[test]
    fn test_reset() {
        let mut agent = WasmAgent::new("{}").unwrap();
        agent.state.turn_count = 5;
        agent.state.stopped = true;
        agent.state.messages.push(BridgeMessage::user("hello"));
        agent.todos.push(TodoItem {
            content: "task".into(),
            status: TodoStatus::Pending,
        });

        agent.reset();
        assert_eq!(agent.state.turn_count, 0);
        assert!(!agent.state.stopped);
        assert!(agent.todos.is_empty());
        // Should have system message re-injected.
        assert_eq!(agent.state.messages.len(), 1);
        assert_eq!(agent.state.messages[0].role, "system");
    }

    #[test]
    fn test_agent_name_and_model() {
        let agent = WasmAgent::new(r#"{"name": "coder", "model": "test:m"}"#).unwrap();
        assert_eq!(agent.name(), Some("coder".into()));
        assert_eq!(agent.model(), "test:m");
    }

    #[test]
    fn test_file_operations_via_agent() {
        let mut agent = WasmAgent::new("{}").unwrap();

        // Write a file via tool.
        let write_req = r#"{"tool": "write_file", "path": "test.rs", "content": "fn main() {}"}"#;
        let result = agent.execute_tool(write_req);
        assert!(result.is_ok());
        assert_eq!(agent.file_count(), 1);

        // Read it back.
        let read_req = r#"{"tool": "read_file", "path": "test.rs"}"#;
        let result = agent.execute_tool(read_req);
        assert!(result.is_ok());

        // Edit the file.
        let edit_req = r#"{
            "tool": "edit_file",
            "path": "test.rs",
            "old_string": "fn main()",
            "new_string": "fn main() -> i32"
        }"#;
        let result = agent.execute_tool(edit_req);
        assert!(result.is_ok());

        // Verify edit.
        let content = agent.backend.read_file("test.rs").unwrap();
        assert_eq!(content, "fn main() -> i32 {}");
    }

    #[test]
    fn test_execute_tool_invalid() {
        let mut agent = WasmAgent::new("{}").unwrap();
        let result = agent.execute_tool("{not valid}");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_defaults() {
        let cfg = WasmAgentConfig::default();
        assert_eq!(cfg.model, "anthropic:claude-sonnet-4-20250514");
        assert!(cfg.name.is_none());
        assert_eq!(cfg.max_turns, 50);
    }

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::default();
        assert!(state.messages.is_empty());
        assert_eq!(state.turn_count, 0);
        assert!(!state.stopped);
    }

    #[test]
    fn test_agent_state_serialization() {
        let mut state = AgentState::default();
        state.messages.push(BridgeMessage::user("hi"));
        state.turn_count = 1;

        let json = serde_json::to_string(&state).unwrap();
        let back: AgentState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.turn_count, 1);
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.messages[0].role, "user");
    }
}

// ---------------------------------------------------------------------------
// wasm-bindgen-test tests (run in browser/node wasm environment)
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_wasm_create_agent() {
        let agent = WasmAgent::new("{}").unwrap();
        assert_eq!(agent.turn_count(), 0);
        assert!(!agent.is_stopped());
    }

    #[wasm_bindgen_test]
    fn test_wasm_get_state() {
        let agent = WasmAgent::new("{}").unwrap();
        let state = agent.get_state();
        assert!(state.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_wasm_get_todos_empty() {
        let agent = WasmAgent::new("{}").unwrap();
        let todos = agent.get_todos();
        assert!(todos.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_wasm_version() {
        let v = WasmAgent::version();
        assert!(!v.is_empty());
    }

    #[wasm_bindgen_test]
    async fn test_wasm_prompt_echo() {
        let mut agent = WasmAgent::new("{}").unwrap();
        let result = agent.prompt("hello").await;
        assert!(result.is_ok());
        assert_eq!(agent.turn_count(), 1);
    }

    #[wasm_bindgen_test]
    fn test_wasm_file_ops() {
        let mut agent = WasmAgent::new("{}").unwrap();
        let write_req = r#"{"tool": "write_file", "path": "demo.txt", "content": "wasm works"}"#;
        agent.execute_tool(write_req).unwrap();
        assert_eq!(agent.file_count(), 1);

        let read_req = r#"{"tool": "read_file", "path": "demo.txt"}"#;
        let result = agent.execute_tool(read_req).unwrap();
        let output = js_sys::Reflect::get(&result, &JsValue::from_str("output")).unwrap();
        assert_eq!(output.as_string().unwrap(), "wasm works");
    }

    #[wasm_bindgen_test]
    fn test_wasm_reset() {
        let mut agent = WasmAgent::new("{}").unwrap();
        agent
            .execute_tool(r#"{"tool": "write_file", "path": "f.txt", "content": "x"}"#)
            .unwrap();
        agent.reset();
        assert_eq!(agent.turn_count(), 0);
        assert!(!agent.is_stopped());
    }

    #[wasm_bindgen_test]
    fn test_wasm_config_parsing() {
        let config = r#"{"model": "test:model", "max_turns": 5}"#;
        let agent = WasmAgent::new(config).unwrap();
        assert_eq!(agent.model(), "test:model");
    }
}
