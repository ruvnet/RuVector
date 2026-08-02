//! Agent graph state machine — replaces LangGraph.
//!
//! Implements the core agent loop: Agent → check tool_calls → execute tools → loop.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

use crate::error::{Result, RvAgentError};
use crate::masking::{
    mask_observations, recall, recall_definition, recall_id_from_args, truncate_tool_result,
    MaskConfig, RECALL_TOOL,
};
use crate::messages::{Message, ToolCall};
use crate::models::{ChatModel, ToolDefinition};
use crate::parallel::parallel_execute_limited;
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

    /// Schemas for the tools this executor can dispatch.
    ///
    /// These are advertised to the model on every completion. The default is
    /// empty (pure-chat agents), but any executor that dispatches real tools
    /// MUST override this — otherwise the model can never call them.
    fn definitions(&self) -> Vec<ToolDefinition> {
        Vec::new()
    }
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
    /// Maximum tool calls in flight at once when `parallel_tools` is set.
    pub max_parallel_tools: usize,
    /// How many times an identical `(tool, args)` pair may repeat
    /// consecutively before the call is refused instead of executed.
    ///
    /// Set to 0 to disable loop detection entirely.
    pub loop_repeat_threshold: usize,
    /// Observation masking and tool-output caps (ADR-274).
    pub mask: MaskConfig,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            parallel_tools: true,
            max_parallel_tools: 8,
            loop_repeat_threshold: 3,
            mask: MaskConfig::default(),
        }
    }
}

/// Detects a stuck agent repeating the same tool call.
///
/// A stuck agent repeats one call forever; raising `max_iterations` only makes
/// that more expensive. Refusing the repeat and telling the model *why* is what
/// breaks the cycle.
///
/// Repeats are counted **consecutively**, not across a window. An agent that
/// re-runs `cargo test` between edits is doing legitimate work, and refusing
/// that would be worse than the loop it prevents — so only an unbroken run of
/// identical calls trips the detector.
///
/// Known limitation: alternating cycles (A, B, A, B, ...) are not detected.
/// `max_iterations` remains the backstop for those.
#[derive(Debug)]
struct LoopDetector {
    last: Option<u64>,
    consecutive: usize,
    threshold: usize,
}

impl LoopDetector {
    fn new(threshold: usize) -> Self {
        Self {
            last: None,
            consecutive: 0,
            threshold,
        }
    }

    fn enabled(&self) -> bool {
        self.threshold > 0
    }

    /// Fingerprint a call by name and arguments.
    ///
    /// Arguments are hashed via their JSON string, which is canonical for key
    /// ordering because `serde_json` maps are `BTreeMap` by default — so
    /// logically identical calls collide regardless of the order the model
    /// emitted the keys in.
    fn fingerprint(call: &ToolCall) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        call.name.hash(&mut hasher);
        call.args.to_string().hash(&mut hasher);
        hasher.finish()
    }

    /// Record a call and report whether it has now repeated too often.
    fn observe(&mut self, call: &ToolCall) -> bool {
        if !self.enabled() {
            return false;
        }
        let fp = Self::fingerprint(call);
        if self.last == Some(fp) {
            self.consecutive += 1;
        } else {
            self.last = Some(fp);
            self.consecutive = 1;
        }
        self.consecutive >= self.threshold
    }
}

/// The message substituted for a call that tripped loop detection.
///
/// Tool errors must be actionable rather than opaque — the model has to be
/// told what to do differently, or it repeats the call again.
fn loop_break_message(call: &ToolCall, threshold: usize) -> String {
    format!(
        "Tool execution refused: this exact call to '{}' with identical \
         arguments has already been made {} times without progress, so it was \
         not executed again. Repeating it will not produce a different result. \
         Change the arguments, use a different tool, or explain what is \
         blocking you.",
        call.name, threshold
    )
}

/// The agent execution graph.
///
/// Implements the core loop:
/// ```text
/// Start → Agent → [has tool_calls?]
///              ├── yes → Tools → Agent (loop)
///              └── no  → End
/// ```
pub struct AgentGraph<M: ChatModel, T: ToolExecutor + 'static> {
    model: M,
    // Arc so tool calls can be spawned onto the runtime for true parallel
    // execution (ADR-103 A2) — spawned futures must be 'static.
    tool_executor: Arc<T>,
    config: GraphConfig,
    edges: Vec<Edge>,
}

impl<M: ChatModel, T: ToolExecutor + 'static> AgentGraph<M, T> {
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
            tool_executor: Arc::new(tool_executor),
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
        // Tool schemas advertised to the model on every completion. Without
        // these the model can never emit a tool call.
        let mut tool_definitions = self.tool_executor.definitions();
        // Masked observations are useless without a way to dereference them,
        // so recall is advertised exactly when masking is active (ADR-274 §3.2).
        if self.config.mask.masking_enabled() {
            tool_definitions.push(recall_definition());
        }
        // Cumulative token usage across the loop, aggregated from per-message
        // usage metadata attached by provider backends.
        let mut total_input_tokens: u64 = 0;
        let mut total_output_tokens: u64 = 0;
        // Spans the whole run: a loop is only visible across iterations.
        let mut loop_detector = LoopDetector::new(self.config.loop_repeat_threshold);

        info!(node = ?current_node, tools = tool_definitions.len(), "graph: starting agent loop");

        loop {
            match current_node {
                AgentNode::Start => {
                    debug!("graph: Start → Agent");
                    current_node = AgentNode::Agent;
                }

                AgentNode::Agent => {
                    // The budget is checked here rather than at the top of the
                    // loop so that reaching End *within* the budget completes
                    // the run. Checking before every node discarded a finished
                    // run whose last allowed iteration produced the answer —
                    // with `max_iterations: 1`, every successful single-turn
                    // run failed.
                    if iterations >= self.config.max_iterations {
                        warn!(iterations, "graph: max iterations reached");
                        return Err(RvAgentError::timeout(format!(
                            "agent loop exceeded {} iterations",
                            self.config.max_iterations
                        )));
                    }
                    iterations += 1;
                    debug!(iteration = iterations, "graph: invoking model");

                    // The model sees a masked projection; `state.messages`
                    // stays the complete log so elided content remains
                    // addressable by tool_call_id (ADR-274).
                    let outbound = mask_observations(&state.messages, &self.config.mask);
                    let response = self.model.complete(&outbound, &tool_definitions).await?;
                    if let Some((input, output)) = usage_from_message(&response) {
                        total_input_tokens += input;
                        total_output_tokens += output;
                        debug!(
                            input_tokens = input,
                            output_tokens = output,
                            total_input_tokens,
                            total_output_tokens,
                            "graph: turn usage"
                        );
                    }
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

                    // Refuse calls that are repeating without progress. This
                    // happens before dispatch so a stuck agent stops burning
                    // tokens and side effects on the same call.
                    let mut looping: Vec<Option<String>> = Vec::with_capacity(tool_calls.len());
                    for tc in &tool_calls {
                        looping.push(if loop_detector.observe(tc) {
                            warn!(tool = %tc.name, "graph: loop detected, refusing repeated call");
                            Some(loop_break_message(tc, self.config.loop_repeat_threshold))
                        } else {
                            None
                        });
                    }
                    // Recall is served by the loop from the full log; it never
                    // reaches the executor, so a workspace tool cannot shadow it.
                    let mut handled: std::collections::HashMap<String, String> =
                        std::collections::HashMap::new();
                    for (tc, refused) in tool_calls.iter().zip(&looping) {
                        if refused.is_none() && tc.name == RECALL_TOOL {
                            let content = match recall_id_from_args(&tc.args) {
                                Ok(id) => recall(&state.messages, id),
                                Err(e) => e,
                            };
                            handled.insert(tc.id.clone(), content);
                        }
                    }

                    // Only calls that cleared loop detection and were not
                    // handled in-loop reach the executor.
                    let dispatch: Vec<ToolCall> = tool_calls
                        .iter()
                        .zip(&looping)
                        .filter(|(tc, refused)| refused.is_none() && !handled.contains_key(&tc.id))
                        .map(|(tc, _)| tc.clone())
                        .collect();

                    // Tool failures are fed back to the model as tool results
                    // rather than aborting the loop — the model must see the
                    // error to recover from it (execution alignment).
                    let mut executed: std::collections::HashMap<String, String> = handled;
                    executed.reserve(dispatch.len());
                    if self.config.parallel_tools && dispatch.len() > 1 {
                        // True parallel execution (ADR-103 A2): tasks are
                        // spawned onto the runtime with bounded concurrency;
                        // results are returned in input order.
                        let executor = Arc::clone(&self.tool_executor);
                        let exec_state = state.clone();
                        let results = parallel_execute_limited(
                            dispatch,
                            move |tc: ToolCall| {
                                let executor = Arc::clone(&executor);
                                let exec_state = exec_state.clone();
                                async move {
                                    let id = tc.id.clone();
                                    let name = tc.name.clone();
                                    // Run the tool in its own task so a panicking
                                    // tool surfaces as a tool error instead of
                                    // crashing the whole agent loop.
                                    let result = match tokio::spawn(async move {
                                        executor.execute(&tc, &exec_state).await
                                    })
                                    .await
                                    {
                                        Ok(res) => res,
                                        Err(join_err) => Err(RvAgentError::tool(format!(
                                            "tool '{name}' execution task failed: {join_err}"
                                        ))),
                                    };
                                    (id, name, result)
                                }
                            },
                            self.config.max_parallel_tools.max(1),
                        )
                        .await;
                        for (id, _name, result) in results {
                            executed.insert(
                                id,
                                truncate_tool_result(
                                    tool_result_content(result),
                                    self.config.mask.max_tool_result_bytes,
                                ),
                            );
                        }
                    } else {
                        // Sequential execution. Each call still runs in its own
                        // task, so a panicking tool becomes a tool error here
                        // exactly as it does on the parallel path — a panic on
                        // the single-call path would otherwise take down the
                        // whole agent loop.
                        for tc in &dispatch {
                            let executor = Arc::clone(&self.tool_executor);
                            let exec_state = state.clone();
                            let call = tc.clone();
                            let name = tc.name.clone();
                            let result = match tokio::spawn(async move {
                                executor.execute(&call, &exec_state).await
                            })
                            .await
                            {
                                Ok(res) => res,
                                Err(join_err) => Err(RvAgentError::tool(format!(
                                    "tool '{name}' execution task failed: {join_err}"
                                ))),
                            };
                            executed.insert(
                                tc.id.clone(),
                                truncate_tool_result(
                                    tool_result_content(result),
                                    self.config.mask.max_tool_result_bytes,
                                ),
                            );
                        }
                    }

                    // Emit one tool result per call, in the model's original
                    // call order, substituting the refusal for looping calls.
                    for (tc, refused) in tool_calls.iter().zip(looping) {
                        let content = match refused {
                            Some(msg) => msg,
                            None => executed.remove(&tc.id).unwrap_or_else(|| {
                                // Defensive: a dispatched call must always
                                // produce a result. Report rather than drop it,
                                // since a missing tool result desyncs the
                                // provider's tool_use/tool_result pairing.
                                format!(
                                    "Tool execution error: no result produced for '{}'",
                                    tc.name
                                )
                            }),
                        };
                        state.push_message(Message::tool_with_name(&tc.id, content, &tc.name));
                    }

                    debug!("graph: Tools → Agent");
                    current_node = AgentNode::Agent;
                }

                AgentNode::End => {
                    info!(
                        iterations,
                        total_input_tokens, total_output_tokens, "graph: agent loop complete"
                    );
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

/// Convert a tool execution result into tool-message content.
///
/// Errors become visible tool output instead of aborting the loop — feeding
/// the failure back to the model is the recovery path.
fn tool_result_content(result: Result<String>) -> String {
    match result {
        Ok(content) => content,
        Err(e) => format!("Tool execution error: {e}"),
    }
}

/// Extract `(input_tokens, output_tokens)` from a message's usage metadata,
/// as attached by provider backends under the `usage` key.
fn usage_from_message(msg: &Message) -> Option<(u64, u64)> {
    if let Message::Ai(ai) = msg {
        let usage = ai.metadata.get("usage")?;
        let input = usage.get("input_tokens").and_then(|v| v.as_u64())?;
        let output = usage.get("output_tokens").and_then(|v| v.as_u64())?;
        Some((input, output))
    } else {
        None
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
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> Result<Message> {
            let mut resps = self.responses.lock().unwrap();
            if resps.is_empty() {
                Ok(Message::ai("done"))
            } else {
                Ok(resps.remove(0))
            }
        }

        async fn stream(
            &self,
            messages: &[Message],
            tools: &[ToolDefinition],
        ) -> Result<Vec<Message>> {
            let msg = self.complete(messages, tools).await?;
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

    /// A tool executor that panics on one tool and succeeds on others.
    struct PanickyExecutor;

    #[async_trait]
    impl ToolExecutor for PanickyExecutor {
        async fn execute(&self, call: &ToolCall, _state: &AgentState) -> Result<String> {
            if call.name == "boom" {
                panic!("tool panicked");
            }
            Ok(format!("result of {}", call.name))
        }
    }

    #[tokio::test]
    async fn test_parallel_tool_panic_is_contained() {
        let model = MockModel::new(vec![
            Message::ai_with_tools(
                "",
                vec![
                    ToolCall {
                        id: "tc1".into(),
                        name: "boom".into(),
                        args: serde_json::json!({}),
                    },
                    ToolCall {
                        id: "tc2".into(),
                        name: "ok_tool".into(),
                        args: serde_json::json!({}),
                    },
                ],
            ),
            Message::ai("done"),
        ]);
        let graph = AgentGraph::new(model, PanickyExecutor);

        let state = AgentState::new();
        // Must not panic: the panicking tool becomes an error tool-result.
        let result = graph.run(state).await.unwrap();

        let contents: Vec<&str> = result
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::Tool(t) => Some(t.content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(contents.len(), 2);
        assert!(contents[0].contains("Tool execution error"));
        assert!(contents[1].contains("result of ok_tool"));
    }

    #[tokio::test]
    async fn test_single_tool_panic_is_contained() {
        // The single-call path runs sequentially even with parallel_tools on,
        // so it needs its own containment — a panic here used to abort the run.
        let model = MockModel::new(vec![
            Message::ai_with_tools(
                "",
                vec![ToolCall {
                    id: "tc1".into(),
                    name: "boom".into(),
                    args: serde_json::json!({}),
                }],
            ),
            Message::ai("done"),
        ]);
        let graph = AgentGraph::with_config(
            model,
            PanickyExecutor,
            GraphConfig {
                parallel_tools: false,
                ..GraphConfig::default()
            },
        );

        let result = graph.run(AgentState::new()).await.unwrap();

        let contents: Vec<&str> = result
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::Tool(t) => Some(t.content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(contents.len(), 1);
        assert!(contents[0].contains("Tool execution error"));
    }

    #[tokio::test]
    async fn test_max_iterations_of_one_allows_a_single_turn() {
        // A run that answers on its last allowed iteration has not exceeded the
        // budget; discarding it made max_iterations: 1 unusable.
        let model = MockModel::new(vec![Message::ai("Hello!")]);
        let graph = AgentGraph::with_config(
            model,
            MockToolExecutor,
            GraphConfig {
                max_iterations: 1,
                ..GraphConfig::default()
            },
        );

        let result = graph.run(AgentState::with_system_message("sys")).await;
        let state = result.expect("single-turn run within budget must succeed");
        assert!(matches!(state.messages.last(), Some(Message::Ai(_))));
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
            ..GraphConfig::default()
        };
        let graph = AgentGraph::with_config(model, executor, config);

        let state = AgentState::new();
        let err = graph.run(state).await.unwrap_err();
        assert!(matches!(err, RvAgentError::Timeout(_)));
    }

    /// Counts how many times the executor was actually invoked.
    struct CountingExecutor {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ToolExecutor for CountingExecutor {
        async fn execute(&self, call: &ToolCall, _state: &AgentState) -> Result<String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(format!("result of {}", call.name))
        }
    }

    fn repeated_call_model(n: usize, args: serde_json::Value) -> MockModel {
        let mut responses: Vec<Message> = (0..n)
            .map(|i| {
                Message::ai_with_tools(
                    "",
                    vec![ToolCall {
                        // Distinct ids, identical name+args — a real stuck loop
                        // looks exactly like this.
                        id: format!("tc{i}"),
                        name: "noop".into(),
                        args: args.clone(),
                    }],
                )
            })
            .collect();
        responses.push(Message::ai("done"));
        MockModel::new(responses)
    }

    #[tokio::test]
    async fn test_repeated_identical_call_is_refused() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let graph = AgentGraph::with_config(
            repeated_call_model(5, serde_json::json!({"x": 1})),
            CountingExecutor {
                calls: Arc::clone(&calls),
            },
            GraphConfig {
                max_iterations: 20,
                parallel_tools: false,
                loop_repeat_threshold: 3,
                ..GraphConfig::default()
            },
        );
        let result = graph.run(AgentState::new()).await.unwrap();

        // The 3rd identical call and everything after it must be refused, so
        // the executor sees exactly 2 invocations.
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "loop detection did not stop execution"
        );

        let tool_msgs: Vec<&str> = result
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::Tool(t) => Some(t.content.as_str()),
                _ => None,
            })
            .collect();
        // Every call still gets exactly one result — dropping one would desync
        // the provider's tool_use/tool_result pairing.
        assert_eq!(tool_msgs.len(), 5);
        assert!(tool_msgs[0].contains("result of noop"));
        assert!(tool_msgs[1].contains("result of noop"));
        for refused in &tool_msgs[2..] {
            assert!(
                refused.contains("Tool execution refused"),
                "expected refusal, got: {refused}"
            );
            // The refusal must tell the model what to do differently.
            assert!(refused.contains("Change the arguments"));
        }
    }

    #[tokio::test]
    async fn test_differing_args_are_not_treated_as_a_loop() {
        // Same tool, different arguments each time: legitimate work.
        let mut responses: Vec<Message> = (0..5)
            .map(|i| {
                Message::ai_with_tools(
                    "",
                    vec![ToolCall {
                        id: format!("tc{i}"),
                        name: "noop".into(),
                        args: serde_json::json!({ "x": i }),
                    }],
                )
            })
            .collect();
        responses.push(Message::ai("done"));

        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let graph = AgentGraph::with_config(
            MockModel::new(responses),
            CountingExecutor {
                calls: Arc::clone(&calls),
            },
            GraphConfig {
                max_iterations: 20,
                parallel_tools: false,
                ..GraphConfig::default()
            },
        );
        graph.run(AgentState::new()).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn test_interleaved_repeats_are_not_refused() {
        // The false positive that drove consecutive-only counting: re-running
        // the same check between edits is legitimate and must not be blocked.
        let check = ToolCall {
            id: "c".into(),
            name: "run_tests".into(),
            args: serde_json::json!({}),
        };
        let mut responses = Vec::new();
        for i in 0..4 {
            responses.push(Message::ai_with_tools(
                "",
                vec![ToolCall {
                    id: format!("e{i}"),
                    name: "edit".into(),
                    args: serde_json::json!({ "line": i }),
                }],
            ));
            responses.push(Message::ai_with_tools(
                "",
                vec![ToolCall {
                    id: format!("c{i}"),
                    ..check.clone()
                }],
            ));
        }
        responses.push(Message::ai("done"));

        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let graph = AgentGraph::with_config(
            MockModel::new(responses),
            CountingExecutor {
                calls: Arc::clone(&calls),
            },
            GraphConfig {
                max_iterations: 30,
                parallel_tools: false,
                ..GraphConfig::default()
            },
        );
        graph.run(AgentState::new()).await.unwrap();
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            8,
            "legitimate interleaved re-runs were refused"
        );
    }

    #[tokio::test]
    async fn test_loop_detection_can_be_disabled() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let graph = AgentGraph::with_config(
            repeated_call_model(4, serde_json::json!({})),
            CountingExecutor {
                calls: Arc::clone(&calls),
            },
            GraphConfig {
                max_iterations: 20,
                parallel_tools: false,
                loop_repeat_threshold: 0,
                ..GraphConfig::default()
            },
        );
        graph.run(AgentState::new()).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[test]
    fn test_fingerprint_ignores_call_id_and_key_order() {
        let a = ToolCall {
            id: "one".into(),
            name: "t".into(),
            args: serde_json::json!({"a": 1, "b": 2}),
        };
        let b = ToolCall {
            id: "two".into(),
            name: "t".into(),
            args: serde_json::json!({"b": 2, "a": 1}),
        };
        assert_eq!(LoopDetector::fingerprint(&a), LoopDetector::fingerprint(&b));
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
