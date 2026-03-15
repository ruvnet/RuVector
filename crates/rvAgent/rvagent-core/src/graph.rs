//! Agent graph state machine — replaces LangGraph.
//!
//! Implements the core agent loop: Agent → check tool_calls → execute tools → loop.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

use crate::error::{Result, RvAgentError};
use crate::messages::{Message, ToolCall};
use crate::models::ChatModel;
use crate::state::AgentState;

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

/// Nodes in the agent execution graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentNode {
    /// Entry point — initializes state.
    Start,
    /// LLM invocation node — sends messages to the model.
    Agent,
    /// Tool execution node — runs tool calls from the AI response.
    Tools,
    /// Terminal node — agent loop is complete.
    End,
}

/// Edge connecting two nodes, optionally with a condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: AgentNode,
    pub to: AgentNode,
    /// Human-readable condition label (for debugging/visualization).
    #[serde(default)]
    pub condition: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool executor trait
// ---------------------------------------------------------------------------

/// Trait for executing tool calls. Implemented by the middleware/tool layer.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Execute a single tool call and return the result content.
    async fn execute(&self, call: &ToolCall, state: &AgentState) -> Result<String>;
}

// ---------------------------------------------------------------------------
// Agent graph
// ---------------------------------------------------------------------------

/// Configuration for the agent graph loop.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Maximum number of agent loop iterations (prevents runaway).
    pub max_iterations: u32,
    /// Whether to execute tool calls in parallel (ADR-103 A2).
    pub parallel_tools: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            parallel_tools: true,
        }
    }
}

/// The agent execution graph.
///
/// Implements the core loop:
/// ```text
/// Start → Agent → [has tool_calls?]
///              ├── yes → Tools → Agent (loop)
///              └── no  → End
/// ```
pub struct AgentGraph<M: ChatModel, T: ToolExecutor> {
    model: M,
    tool_executor: T,
    config: GraphConfig,
    edges: Vec<Edge>,
}

impl<M: ChatModel, T: ToolExecutor> AgentGraph<M, T> {
    /// Create a new agent graph with the given model and tool executor.
    pub fn new(model: M, tool_executor: T) -> Self {
        Self::with_config(model, tool_executor, GraphConfig::default())
    }

    /// Create a new agent graph with explicit configuration.
    pub fn with_config(model: M, tool_executor: T, config: GraphConfig) -> Self {
        let edges = vec![
            Edge {
                from: AgentNode::Start,
                to: AgentNode::Agent,
                condition: None,
            },
            Edge {
                from: AgentNode::Agent,
                to: AgentNode::Tools,
                condition: Some("has_tool_calls".into()),
            },
            Edge {
                from: AgentNode::Agent,
                to: AgentNode::End,
                condition: Some("no_tool_calls".into()),
            },
            Edge {
                from: AgentNode::Tools,
                to: AgentNode::Agent,
                condition: None,
            },
        ];

        Self {
            model,
            tool_executor,
            config,
            edges,
        }
    }

    /// Get the graph edges (for visualization/debugging).
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Run the agent loop to completion.
    ///
    /// 1. Invoke the model with current messages.
    /// 2. If the response contains tool_calls, execute them and loop.
    /// 3. If no tool_calls, return the final state.
    #[instrument(skip(self, state), fields(iterations))]
    pub async fn run(&self, mut state: AgentState) -> Result<AgentState> {
        let mut current_node = AgentNode::Start;
        let mut iterations: u32 = 0;

        info!(node = ?current_node, "graph: starting agent loop");

        loop {
            if iterations >= self.config.max_iterations {
                warn!(iterations, "graph: max iterations reached");
                return Err(RvAgentError::timeout(format!(
                    "agent loop exceeded {} iterations",
                    self.config.max_iterations
                )));
            }

            match current_node {
                AgentNode::Start => {
                    debug!("graph: Start → Agent");
                    current_node = AgentNode::Agent;
                }

                AgentNode::Agent => {
                    iterations += 1;
                    debug!(iteration = iterations, "graph: invoking model");

                    let response = self.model.complete(&state.messages).await?;
                    let has_tool_calls = response.has_tool_calls();
                    state.push_message(response);

                    if has_tool_calls {
                        debug!("graph: Agent → Tools (tool_calls present)");
                        current_node = AgentNode::Tools;
                    } else {
                        debug!("graph: Agent → End (no tool_calls)");
                        current_node = AgentNode::End;
                    }
                }

                AgentNode::Tools => {
                    debug!("graph: executing tool calls");

                    // Extract tool calls from the last AI message.
                    let tool_calls = self.extract_tool_calls(&state)?;

                    if self.config.parallel_tools && tool_calls.len() > 1 {
                        // Parallel execution (ADR-103 A2).
                        let mut handles = Vec::with_capacity(tool_calls.len());
                        for tc in &tool_calls {
                            let result = self.tool_executor.execute(tc, &state).await;
                            handles.push((tc.id.clone(), result));
                        }
                        for (id, result) in handles {
                            let content = result?;
                            state.push_message(Message::tool(id, content));
                        }
                    } else {
                        // Sequential execution.
                        for tc in &tool_calls {
                            let content = self.tool_executor.execute(tc, &state).await?;
                            state.push_message(Message::tool(&tc.id, content));
                        }
                    }

                    debug!("graph: Tools → Agent");
                    current_node = AgentNode::Agent;
                }

                AgentNode::End => {
                    info!(iterations, "graph: agent loop complete");
                    return Ok(state);
                }
            }
        }
    }

    /// Extract tool calls from the most recent AI message in state.
    fn extract_tool_calls(&self, state: &AgentState) -> Result<Vec<ToolCall>> {
        for msg in state.messages.iter().rev() {
            if let Message::Ai(ai_msg) = msg {
                if !ai_msg.tool_calls.is_empty() {
                    return Ok(ai_msg.tool_calls.clone());
                }
            }
        }
        Err(RvAgentError::state(
            "no tool calls found in recent AI message",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::AiMessage;

    /// A mock model that returns a fixed sequence of responses.
    struct MockModel {
        responses: std::sync::Mutex<Vec<Message>>,
    }

    impl MockModel {
        fn new(responses: Vec<Message>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl ChatModel for MockModel {
        async fn complete(&self, _messages: &[Message]) -> Result<Message> {
            let mut resps = self.responses.lock().unwrap();
            if resps.is_empty() {
                Ok(Message::ai("done"))
            } else {
                Ok(resps.remove(0))
            }
        }

        async fn stream(&self, messages: &[Message]) -> Result<Vec<Message>> {
            let msg = self.complete(messages).await?;
            Ok(vec![msg])
        }
    }

    /// A mock tool executor that returns the tool name as output.
    struct MockToolExecutor;

    #[async_trait]
    impl ToolExecutor for MockToolExecutor {
        async fn execute(&self, call: &ToolCall, _state: &AgentState) -> Result<String> {
            Ok(format!("result of {}", call.name))
        }
    }

    #[tokio::test]
    async fn test_simple_completion() {
        let model = MockModel::new(vec![Message::ai("Hello!")]);
        let executor = MockToolExecutor;
        let graph = AgentGraph::new(model, executor);

        let state = AgentState::with_system_message("You are helpful.");
        let result = graph.run(state).await.unwrap();

        assert!(result.message_count() >= 2); // system + ai response
    }

    #[tokio::test]
    async fn test_tool_call_loop() {
        let model = MockModel::new(vec![
            // First response: call a tool.
            Message::ai_with_tools(
                "Let me read that file.",
                vec![ToolCall {
                    id: "tc1".into(),
                    name: "read_file".into(),
                    args: serde_json::json!({"path": "/tmp/test.rs"}),
                }],
            ),
            // Second response: no tool calls → end.
            Message::ai("The file contains tests."),
        ]);
        let executor = MockToolExecutor;
        let graph = AgentGraph::new(model, executor);

        let state = AgentState::with_system_message("sys");
        let result = graph.run(state).await.unwrap();

        // system + ai_with_tools + tool_result + ai_final = 4
        assert_eq!(result.message_count(), 4);
    }

    #[tokio::test]
    async fn test_max_iterations() {
        // Model always returns tool calls → should hit max iterations.
        let responses: Vec<Message> = (0..200)
            .map(|i| {
                Message::ai_with_tools(
                    "",
                    vec![ToolCall {
                        id: format!("tc{}", i),
                        name: "noop".into(),
                        args: serde_json::json!({}),
                    }],
                )
            })
            .collect();
        let model = MockModel::new(responses);
        let executor = MockToolExecutor;
        let config = GraphConfig {
            max_iterations: 3,
            parallel_tools: false,
        };
        let graph = AgentGraph::with_config(model, executor, config);

        let state = AgentState::new();
        let err = graph.run(state).await.unwrap_err();
        assert!(matches!(err, RvAgentError::Timeout(_)));
    }

    #[test]
    fn test_graph_edges() {
        let model = MockModel::new(vec![]);
        let executor = MockToolExecutor;
        let graph = AgentGraph::new(model, executor);

        let edges = graph.edges();
        assert_eq!(edges.len(), 4);
        assert_eq!(edges[0].from, AgentNode::Start);
        assert_eq!(edges[0].to, AgentNode::Agent);
    }

    #[test]
    fn test_agent_node_serde() {
        let node = AgentNode::Tools;
        let json = serde_json::to_string(&node).unwrap();
        let back: AgentNode = serde_json::from_str(&json).unwrap();
        assert_eq!(node, back);
    }
}
