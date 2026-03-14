//! Integration tests for rvAgent core.
//!
//! Tests the complete agent lifecycle: config creation, graph construction,
//! model invocation with mocks, and state management end-to-end.

use async_trait::async_trait;
use std::sync::Mutex;

use rvagent_core::config::RvAgentConfig;
use rvagent_core::error::{Result, RvAgentError};
use rvagent_core::graph::{AgentGraph, GraphConfig, ToolExecutor};
use rvagent_core::messages::{Message, ToolCall};
use rvagent_core::models::{ChatModel, Provider};
use rvagent_core::state::AgentState;

// ---------------------------------------------------------------------------
// Mock infrastructure
// ---------------------------------------------------------------------------

/// A mock ChatModel that returns a fixed sequence of responses.
struct MockModel {
    responses: Mutex<Vec<Message>>,
}

impl MockModel {
    fn new(responses: Vec<Message>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }
}

#[async_trait]
impl ChatModel for MockModel {
    async fn complete(&self, _messages: &[Message]) -> Result<Message> {
        let mut resps = self.responses.lock().unwrap();
        if resps.is_empty() {
            Ok(Message::ai("(no more responses)"))
        } else {
            Ok(resps.remove(0))
        }
    }

    async fn stream(&self, messages: &[Message]) -> Result<Vec<Message>> {
        let msg = self.complete(messages).await?;
        Ok(vec![msg])
    }
}

/// A mock ToolExecutor that echoes tool call name and args.
struct EchoToolExecutor;

#[async_trait]
impl ToolExecutor for EchoToolExecutor {
    async fn execute(&self, call: &ToolCall, _state: &AgentState) -> Result<String> {
        Ok(format!(
            "executed {} with args: {}",
            call.name,
            call.args
        ))
    }
}

/// A mock ToolExecutor that returns errors for specific tools.
struct FailingToolExecutor {
    fail_tool: String,
}

#[async_trait]
impl ToolExecutor for FailingToolExecutor {
    async fn execute(&self, call: &ToolCall, _state: &AgentState) -> Result<String> {
        if call.name == self.fail_tool {
            Err(RvAgentError::tool(format!("{} failed", call.name)))
        } else {
            Ok(format!("ok: {}", call.name))
        }
    }
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

/// Create config -> build graph -> run with mock model -> verify output.
#[tokio::test]
async fn test_agent_graph_basic_flow() {
    // 1. Create a config with defaults.
    let config = RvAgentConfig::default();
    assert_eq!(
        config.model,
        rvagent_core::config::DEFAULT_MODEL,
        "default model should be set"
    );
    assert!(config.security_policy.virtual_mode);

    // 2. Build a graph with a mock model that returns a simple response.
    let model = MockModel::new(vec![Message::ai("Hello! I can help you with that.")]);
    let executor = EchoToolExecutor;
    let graph = AgentGraph::new(model, executor);

    // 3. Run with initial state containing a system message and user message.
    let mut state = AgentState::with_system_message(&config.instructions);
    state.push_message(Message::human("What can you do?"));

    let result = graph.run(state).await.unwrap();

    // 4. Verify: should have system + human + ai = 3 messages.
    assert_eq!(result.message_count(), 3);
    assert_eq!(
        result.messages.last().unwrap().content(),
        "Hello! I can help you with that."
    );
    // No tool calls means the graph should have reached End.
    assert!(!result.messages.last().unwrap().has_tool_calls());
}

/// Agent loop: model returns tool_calls -> tools execute -> model sees results -> final answer.
#[tokio::test]
async fn test_agent_graph_with_tool_calls() {
    // Model first returns a tool call, then a final answer.
    let model = MockModel::new(vec![
        // First response: request to read a file.
        Message::ai_with_tools(
            "Let me read that file for you.",
            vec![ToolCall {
                id: "call_001".into(),
                name: "read_file".into(),
                args: serde_json::json!({"path": "/src/main.rs"}),
            }],
        ),
        // Second response: final answer after seeing tool result.
        Message::ai("The file contains a main function that initializes the app."),
    ]);
    let executor = EchoToolExecutor;
    let graph = AgentGraph::new(model, executor);

    let mut state = AgentState::with_system_message("You are helpful.");
    state.push_message(Message::human("Read /src/main.rs"));

    let result = graph.run(state).await.unwrap();

    // Expected messages:
    // 0: system
    // 1: human
    // 2: ai (with tool_calls)
    // 3: tool result
    // 4: ai (final answer)
    assert_eq!(result.message_count(), 5);

    // Verify tool result is present.
    let tool_msg = &result.messages[3];
    assert!(tool_msg.content().contains("executed read_file"));

    // Verify final answer.
    let final_msg = result.messages.last().unwrap();
    assert_eq!(
        final_msg.content(),
        "The file contains a main function that initializes the app."
    );
    assert!(!final_msg.has_tool_calls());
}

/// Agent loop with multiple tool calls in a single response.
#[tokio::test]
async fn test_agent_graph_with_parallel_tool_calls() {
    let model = MockModel::new(vec![
        // First response: two tool calls at once.
        Message::ai_with_tools(
            "Let me search for that.",
            vec![
                ToolCall {
                    id: "call_a".into(),
                    name: "grep".into(),
                    args: serde_json::json!({"pattern": "TODO"}),
                },
                ToolCall {
                    id: "call_b".into(),
                    name: "glob".into(),
                    args: serde_json::json!({"pattern": "**/*.rs"}),
                },
            ],
        ),
        // Second response: final answer.
        Message::ai("Found 3 TODOs across 5 Rust files."),
    ]);
    let executor = EchoToolExecutor;
    let config = GraphConfig {
        max_iterations: 10,
        parallel_tools: true,
    };
    let graph = AgentGraph::with_config(model, executor, config);

    let state = AgentState::with_system_message("sys");
    let result = graph.run(state).await.unwrap();

    // system + ai_with_tools + tool_a + tool_b + ai_final = 5
    assert_eq!(result.message_count(), 5);

    // Both tool results should be present.
    let tool_results: Vec<&str> = result
        .messages
        .iter()
        .filter(|m| matches!(m, Message::Tool(_)))
        .map(|m| m.content())
        .collect();
    assert_eq!(tool_results.len(), 2);
    assert!(tool_results.iter().any(|c| c.contains("grep")));
    assert!(tool_results.iter().any(|c| c.contains("glob")));
}

/// RvAgentConfig -> AgentGraph creation pipeline.
#[test]
fn test_config_to_graph_pipeline() {
    // 1. Create config from JSON (simulating file-based config loading).
    let json = r#"{
        "model": "openai:gpt-4o",
        "name": "test-agent",
        "middleware": [
            {"name": "filesystem", "settings": {}},
            {"name": "memory", "settings": {}}
        ],
        "tools": [
            {"name": "read_file", "settings": {}},
            {"name": "write_file", "settings": {}}
        ],
        "backend": {
            "backend_type": "local_shell",
            "cwd": "/tmp/project"
        },
        "security_policy": {
            "virtual_mode": true,
            "command_allowlist": ["ls", "cat"]
        }
    }"#;

    let config: RvAgentConfig = serde_json::from_str(json).unwrap();

    // 2. Verify config parsed correctly.
    assert_eq!(config.model, "openai:gpt-4o");
    assert_eq!(config.name.as_deref(), Some("test-agent"));
    assert_eq!(config.middleware.len(), 2);
    assert_eq!(config.tools.len(), 2);
    assert_eq!(config.backend.backend_type, "local_shell");
    assert!(config.security_policy.virtual_mode);
    assert_eq!(config.security_policy.command_allowlist.len(), 2);

    // 3. Resolve the model.
    let model_config = rvagent_core::models::resolve_model(&config.model);
    assert_eq!(model_config.provider, Provider::OpenAi);
    assert_eq!(model_config.model_id, "gpt-4o");

    // 4. Build a graph (using mocks for the model and tool executor).
    let model = MockModel::new(vec![Message::ai("ready")]);
    let executor = EchoToolExecutor;
    let graph = AgentGraph::new(model, executor);

    // 5. Verify the graph has the expected edges.
    let edges = graph.edges();
    assert_eq!(edges.len(), 4);
}

/// Tool execution failure propagates correctly through the graph.
#[tokio::test]
async fn test_agent_graph_tool_failure() {
    let model = MockModel::new(vec![
        Message::ai_with_tools(
            "",
            vec![ToolCall {
                id: "tc1".into(),
                name: "dangerous_tool".into(),
                args: serde_json::json!({}),
            }],
        ),
    ]);
    let executor = FailingToolExecutor {
        fail_tool: "dangerous_tool".into(),
    };
    let graph = AgentGraph::new(model, executor);

    let state = AgentState::new();
    let err = graph.run(state).await.unwrap_err();
    assert!(matches!(err, RvAgentError::Tool(_)));
    assert!(err.to_string().contains("dangerous_tool failed"));
}

/// State mutations during graph execution use copy-on-write correctly.
#[tokio::test]
async fn test_state_cow_during_graph_run() {
    let model = MockModel::new(vec![Message::ai("done")]);
    let executor = EchoToolExecutor;
    let graph = AgentGraph::new(model, executor);

    let state = AgentState::with_system_message("sys");
    let snapshot = state.clone();

    // Run the graph, which mutates state internally.
    let result = graph.run(state).await.unwrap();

    // Snapshot should be unaffected (COW semantics).
    assert_eq!(snapshot.message_count(), 1);
    assert!(result.message_count() > snapshot.message_count());
}

/// Config defaults are correct and complete.
#[test]
fn test_default_config_completeness() {
    let config = RvAgentConfig::default();

    // Model defaults to Anthropic.
    assert!(config.model.starts_with("anthropic:"));

    // Security policy defaults per ADR-103.
    assert!(config.security_policy.virtual_mode);
    assert!(!config.security_policy.trust_agents_md);
    assert_eq!(config.security_policy.max_response_length, 100 * 1024);
    assert!(config.security_policy.sensitive_env_patterns.len() >= 10);

    // Instructions should be the base prompt.
    assert!(config.instructions.contains("rvAgent"));

    // Backend defaults to local_shell.
    assert_eq!(config.backend.backend_type, "local_shell");
}

/// Max iterations prevents runaway agent loops.
#[tokio::test]
async fn test_max_iterations_terminates() {
    let infinite_tools: Vec<Message> = (0..200)
        .map(|i| {
            Message::ai_with_tools(
                "",
                vec![ToolCall {
                    id: format!("tc{}", i),
                    name: "loop".into(),
                    args: serde_json::json!({}),
                }],
            )
        })
        .collect();
    let model = MockModel::new(infinite_tools);
    let executor = EchoToolExecutor;
    let config = GraphConfig {
        max_iterations: 5,
        parallel_tools: false,
    };
    let graph = AgentGraph::with_config(model, executor, config);

    let state = AgentState::new();
    let err = graph.run(state).await.unwrap_err();
    assert!(matches!(err, RvAgentError::Timeout(_)));
    assert!(err.to_string().contains("5 iterations"));
}
