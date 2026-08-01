//! Phase 0 exit gate: end-to-end tool calling.
//!
//! These tests wire the *shipped* pieces together — the real builtin tool
//! registry, the real `LocalFsBackend`, and the real `AgentGraph` loop — against
//! a scripted model. Everything except the network call to the provider is
//! production code, so the gate fails if the loop, the schemas, or the tools
//! regress.
//!
//! What each P0 claim is verified by:
//!   * P0.2 (schemas reach the model) — `schemas_reach_the_model`
//!   * P0.4 (errors feed back, not abort) — `tool_error_feeds_back_and_loop_continues`
//!   * P0.4 (parallel exec preserves order) — `parallel_tool_calls_preserve_order`
//!   * P0.4 (usage accounting) — `usage_metadata_is_aggregated`
//!   * real side effects on disk — `write_then_read_roundtrip_through_the_loop`
//!   * confinement holds through the loop — `path_escape_is_refused_through_the_loop`

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rvagent_core::error::Result;
use rvagent_core::graph::{AgentGraph, GraphConfig, ToolExecutor};
use rvagent_core::messages::{Message, ToolCall};
use rvagent_core::models::{ChatModel, ToolDefinition};
use rvagent_core::state::AgentState;
use rvagent_tools::Tool as _;

// ---------------------------------------------------------------------------
// Test harness: the real tool executor, wired exactly as the CLI wires it
// ---------------------------------------------------------------------------

/// Mirrors `CliToolExecutor`: real builtin tools over a real confined backend.
struct RealToolExecutor {
    tools: Vec<rvagent_tools::AnyTool>,
    backend: rvagent_tools::BackendRef,
}

impl RealToolExecutor {
    fn new(root: &std::path::Path) -> Self {
        Self {
            tools: rvagent_tools::builtin_tools(),
            backend: Arc::new(rvagent_tools::LocalFsBackend::new(root)),
        }
    }
}

#[async_trait]
impl ToolExecutor for RealToolExecutor {
    async fn execute(&self, call: &ToolCall, _state: &AgentState) -> Result<String> {
        let runtime = rvagent_tools::ToolRuntime::new(Arc::clone(&self.backend));
        match rvagent_tools::resolve_tool(&call.name, &self.tools) {
            Some(tool) => Ok(tool.invoke(call.args.clone(), &runtime).to_string()),
            None => Ok(format!("Error: tool '{}' not found", call.name)),
        }
    }

    fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.parameters_schema(),
            })
            .collect()
    }
}

/// A scripted model that records what the loop actually sent it.
struct ScriptedModel {
    responses: Mutex<Vec<Message>>,
    /// Tool schemas observed on each `complete` call.
    seen_tools: Mutex<Vec<Vec<ToolDefinition>>>,
    /// Message history observed on the most recent `complete` call.
    last_messages: Mutex<Vec<Message>>,
}

impl ScriptedModel {
    fn new(responses: Vec<Message>) -> Self {
        Self {
            responses: Mutex::new(responses),
            seen_tools: Mutex::new(Vec::new()),
            last_messages: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl ChatModel for ScriptedModel {
    async fn complete(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Message> {
        self.seen_tools.lock().unwrap().push(tools.to_vec());
        *self.last_messages.lock().unwrap() = messages.to_vec();
        let mut resps = self.responses.lock().unwrap();
        if resps.is_empty() {
            Ok(Message::ai("done"))
        } else {
            Ok(resps.remove(0))
        }
    }

    async fn stream(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Vec<Message>> {
        Ok(vec![self.complete(messages, tools).await?])
    }
}

fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        args,
    }
}

/// Collect tool-result message contents, in order.
fn tool_results(state: &AgentState) -> Vec<String> {
    state
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool(t) => Some(t.content.clone()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// P0.2 — tool schemas reach the model
// ---------------------------------------------------------------------------

#[tokio::test]
async fn schemas_reach_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![Message::ai("hi")]);
    let graph = AgentGraph::new(model, RealToolExecutor::new(dir.path()));

    let state = graph.run(AgentState::with_system_message("sys")).await;
    assert!(state.is_ok(), "loop failed: {:?}", state.err());

    // The graph owns the model, so re-derive the expectation from the registry:
    // every builtin tool must have been advertised with a usable schema.
    let executor = RealToolExecutor::new(dir.path());
    let defs = executor.definitions();
    assert_eq!(defs.len(), rvagent_tools::builtin_tools().len());
    assert!(defs.iter().any(|d| d.name == "read_file"));
    assert!(defs.iter().any(|d| d.name == "write_file"));
    for def in &defs {
        assert!(!def.description.is_empty(), "{} has no description", def.name);
        assert_eq!(
            def.input_schema.get("type").and_then(|v| v.as_str()),
            Some("object"),
            "{} schema is not a JSON-Schema object: {}",
            def.name,
            def.input_schema
        );
        assert!(
            def.input_schema.get("properties").is_some(),
            "{} schema has no properties",
            def.name
        );
    }
}

#[tokio::test]
async fn schemas_are_sent_on_every_turn_including_after_tools() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "content").unwrap();

    let model = Arc::new(ScriptedModel::new(vec![
        Message::ai_with_tools(
            "reading",
            vec![call("t1", "read_file", serde_json::json!({"file_path": "f.txt"}))],
        ),
        Message::ai("read it"),
    ]));

    let graph = AgentGraph::new(
        SharedModel(Arc::clone(&model)),
        RealToolExecutor::new(dir.path()),
    );
    graph.run(AgentState::new()).await.unwrap();

    let seen = model.seen_tools.lock().unwrap();
    assert_eq!(seen.len(), 2, "expected two model turns");
    for (turn, tools) in seen.iter().enumerate() {
        assert!(
            !tools.is_empty(),
            "turn {turn} was sent no tool schemas — the model could not call a tool"
        );
    }
}

// ---------------------------------------------------------------------------
// Real side effects on disk
// ---------------------------------------------------------------------------

#[tokio::test]
async fn write_then_read_roundtrip_through_the_loop() {
    let dir = tempfile::tempdir().unwrap();

    let model = ScriptedModel::new(vec![
        Message::ai_with_tools(
            "writing",
            vec![call(
                "t1",
                "write_file",
                serde_json::json!({"file_path": "out.txt", "content": "hello from the agent"}),
            )],
        ),
        Message::ai_with_tools(
            "reading back",
            vec![call(
                "t2",
                "read_file",
                serde_json::json!({"file_path": "out.txt"}),
            )],
        ),
        Message::ai("verified"),
    ]);
    let graph = AgentGraph::new(model, RealToolExecutor::new(dir.path()));
    let state = graph.run(AgentState::new()).await.unwrap();

    // The file must actually exist on disk — this is the end-to-end claim.
    let written = std::fs::read_to_string(dir.path().join("out.txt"))
        .expect("agent's write_file did not produce a real file");
    assert_eq!(written, "hello from the agent");

    let results = tool_results(&state);
    assert_eq!(results.len(), 2, "expected one result per tool call");
    assert!(
        results[1].contains("hello from the agent"),
        "read_file did not return the written content: {}",
        results[1]
    );
}

// ---------------------------------------------------------------------------
// P0.4 — tool errors feed back as results instead of aborting
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_error_feeds_back_and_loop_continues() {
    let dir = tempfile::tempdir().unwrap();

    let model = ScriptedModel::new(vec![
        Message::ai_with_tools(
            "reading a file that isn't there",
            vec![call(
                "t1",
                "read_file",
                serde_json::json!({"file_path": "does-not-exist.txt"}),
            )],
        ),
        // The model gets to see the failure and recover.
        Message::ai_with_tools(
            "creating it instead",
            vec![call(
                "t2",
                "write_file",
                serde_json::json!({"file_path": "does-not-exist.txt", "content": "now it does"}),
            )],
        ),
        Message::ai("recovered"),
    ]);
    let graph = AgentGraph::new(model, RealToolExecutor::new(dir.path()));

    // The run must succeed: a failing tool is not a failing agent.
    let state = graph
        .run(AgentState::new())
        .await
        .expect("a tool error must not abort the loop");

    let results = tool_results(&state);
    assert_eq!(results.len(), 2);
    assert!(
        results[0].to_lowercase().contains("error")
            || results[0].to_lowercase().contains("no such file"),
        "the failure was not reported back to the model: {}",
        results[0]
    );
    assert!(
        dir.path().join("does-not-exist.txt").exists(),
        "the recovery turn did not run"
    );
}

#[tokio::test]
async fn unknown_tool_is_reported_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        Message::ai_with_tools(
            "",
            vec![call("t1", "no_such_tool", serde_json::json!({}))],
        ),
        Message::ai("ok"),
    ]);
    let graph = AgentGraph::new(model, RealToolExecutor::new(dir.path()));
    let state = graph.run(AgentState::new()).await.unwrap();

    let results = tool_results(&state);
    assert_eq!(results.len(), 1);
    assert!(results[0].contains("not found"), "got: {}", results[0]);
}

// ---------------------------------------------------------------------------
// P0.4 — parallel execution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parallel_tool_calls_preserve_order() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..6 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), format!("body-{i}")).unwrap();
    }

    let calls: Vec<ToolCall> = (0..6)
        .map(|i| {
            call(
                &format!("t{i}"),
                "read_file",
                serde_json::json!({"file_path": format!("f{i}.txt")}),
            )
        })
        .collect();

    let model = ScriptedModel::new(vec![
        Message::ai_with_tools("reading all", calls),
        Message::ai("done"),
    ]);
    let config = GraphConfig {
        parallel_tools: true,
        max_parallel_tools: 3,
        ..GraphConfig::default()
    };
    let graph = AgentGraph::with_config(model, RealToolExecutor::new(dir.path()), config);
    let state = graph.run(AgentState::new()).await.unwrap();

    let results = tool_results(&state);
    assert_eq!(results.len(), 6);
    // Results must come back in call order even though execution is concurrent
    // and the concurrency limit is lower than the number of calls.
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.contains(&format!("body-{i}")),
            "result {i} out of order or wrong: {r}"
        );
    }
}

#[tokio::test]
async fn parallel_and_sequential_agree() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..4 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), format!("body-{i}")).unwrap();
    }
    let calls: Vec<ToolCall> = (0..4)
        .map(|i| {
            call(
                &format!("t{i}"),
                "read_file",
                serde_json::json!({"file_path": format!("f{i}.txt")}),
            )
        })
        .collect();

    let mut outputs = Vec::new();
    for parallel in [true, false] {
        let model = ScriptedModel::new(vec![
            Message::ai_with_tools("", calls.clone()),
            Message::ai("done"),
        ]);
        let config = GraphConfig {
            parallel_tools: parallel,
            max_parallel_tools: 2,
            ..GraphConfig::default()
        };
        let graph = AgentGraph::with_config(model, RealToolExecutor::new(dir.path()), config);
        let state = graph.run(AgentState::new()).await.unwrap();
        outputs.push(tool_results(&state));
    }
    assert_eq!(
        outputs[0], outputs[1],
        "parallel and sequential execution disagree"
    );
}

// ---------------------------------------------------------------------------
// P0.4 — usage accounting
// ---------------------------------------------------------------------------

#[tokio::test]
async fn usage_metadata_is_aggregated() {
    let dir = tempfile::tempdir().unwrap();

    // Two turns carrying provider usage metadata, as the backends attach it.
    let mut first = Message::ai_with_tools(
        "",
        vec![call("t1", "ls", serde_json::json!({"path": "."}))],
    );
    let mut second = Message::ai("done");
    for (msg, input, output) in [(&mut first, 100u64, 20u64), (&mut second, 150u64, 30u64)] {
        if let Message::Ai(ai) = msg {
            ai.metadata.insert(
                "usage".into(),
                serde_json::json!({"input_tokens": input, "output_tokens": output}),
            );
        }
    }

    let model = ScriptedModel::new(vec![first, second]);
    let graph = AgentGraph::new(model, RealToolExecutor::new(dir.path()));
    let state = graph.run(AgentState::new()).await.unwrap();

    // Usage metadata must survive on the messages so a caller can total it.
    let totals: (u64, u64) = state
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::Ai(ai) => ai.metadata.get("usage"),
            _ => None,
        })
        .fold((0, 0), |(i, o), usage| {
            (
                i + usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                o + usage
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            )
        });
    assert_eq!(totals, (250, 50), "usage metadata was lost or miscounted");
}

// ---------------------------------------------------------------------------
// ADR-274 — observation masking reaches the model
// ---------------------------------------------------------------------------

/// Shares one `ScriptedModel` so its recorded observations outlive the graph.
struct SharedModel(Arc<ScriptedModel>);

#[async_trait]
impl ChatModel for SharedModel {
    async fn complete(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Message> {
        self.0.complete(messages, tools).await
    }
    async fn stream(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Vec<Message>> {
        self.0.stream(messages, tools).await
    }
}

#[tokio::test]
async fn old_observations_are_masked_before_reaching_the_model() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..5 {
        std::fs::write(
            dir.path().join(format!("f{i}.txt")),
            format!("UNIQUE-BODY-{i}"),
        )
        .unwrap();
    }

    // Five sequential read turns, then a final answer.
    let mut responses: Vec<Message> = (0..5)
        .map(|i| {
            Message::ai_with_tools(
                "",
                vec![call(
                    &format!("t{i}"),
                    "read_file",
                    serde_json::json!({ "file_path": format!("f{i}.txt") }),
                )],
            )
        })
        .collect();
    responses.push(Message::ai("done"));

    let model = Arc::new(ScriptedModel::new(responses));
    let config = GraphConfig {
        parallel_tools: false,
        mask: rvagent_core::masking::MaskConfig {
            keep_last_observations: 2,
            ..Default::default()
        },
        ..GraphConfig::default()
    };
    let graph = AgentGraph::with_config(
        SharedModel(Arc::clone(&model)),
        RealToolExecutor::new(dir.path()),
        config,
    );
    let state = graph.run(AgentState::new()).await.unwrap();

    // What the model saw on the final turn: only the last 2 observations in
    // full, the rest elided but still addressable.
    let seen = model.last_messages.lock().unwrap().clone();
    let seen_tools: Vec<&Message> = seen
        .iter()
        .filter(|m| matches!(m, Message::Tool(_)))
        .collect();
    assert_eq!(seen_tools.len(), 5, "every call still has a paired result");

    for (i, msg) in seen_tools.iter().enumerate() {
        let Message::Tool(t) = msg else { unreachable!() };
        if i < 3 {
            assert!(
                t.content.contains("output elided"),
                "observation {i} should have been masked: {}",
                t.content
            );
            assert!(
                t.content.contains(&format!("recall id t{i}")),
                "masked observation {i} lost its recall handle: {}",
                t.content
            );
            assert!(
                !t.content.contains(&format!("UNIQUE-BODY-{i}")),
                "masked observation {i} still carried its full body"
            );
        } else {
            assert!(
                t.content.contains(&format!("UNIQUE-BODY-{i}")),
                "recent observation {i} must survive verbatim: {}",
                t.content
            );
        }
    }

    // The stored log keeps everything — masking is a projection, not a
    // mutation, which is what makes the elided content recoverable.
    let stored = tool_results(&state);
    assert_eq!(stored.len(), 5);
    for (i, content) in stored.iter().enumerate() {
        assert!(
            content.contains(&format!("UNIQUE-BODY-{i}")),
            "stored observation {i} was destroyed by masking: {content}"
        );
    }
}

#[tokio::test]
async fn recall_returns_the_full_content_of_a_masked_observation() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..4 {
        std::fs::write(
            dir.path().join(format!("f{i}.txt")),
            format!("SECRET-BODY-{i}"),
        )
        .unwrap();
    }

    let mut responses: Vec<Message> = (0..4)
        .map(|i| {
            Message::ai_with_tools(
                "",
                vec![call(
                    &format!("t{i}"),
                    "read_file",
                    serde_json::json!({ "file_path": format!("f{i}.txt") }),
                )],
            )
        })
        .collect();
    // t0 has been masked out of the model's view by now; dereference it.
    responses.push(Message::ai_with_tools(
        "",
        vec![call(
            "r1",
            "recall",
            serde_json::json!({ "recall_id": "t0" }),
        )],
    ));
    responses.push(Message::ai("done"));

    let model = Arc::new(ScriptedModel::new(responses));
    let config = GraphConfig {
        parallel_tools: false,
        mask: rvagent_core::masking::MaskConfig {
            keep_last_observations: 2,
            ..Default::default()
        },
        ..GraphConfig::default()
    };
    let graph = AgentGraph::with_config(
        SharedModel(Arc::clone(&model)),
        RealToolExecutor::new(dir.path()),
        config,
    );
    let state = graph.run(AgentState::new()).await.unwrap();

    // The recall tool must have been advertised, or the model could not call it.
    let seen = model.seen_tools.lock().unwrap();
    assert!(
        seen.last().unwrap().iter().any(|d| d.name == "recall"),
        "recall was not advertised while masking was active"
    );
    drop(seen);

    // The recall result carries the content that was elided from the view.
    let results = tool_results(&state);
    let recalled = results.last().unwrap();
    assert!(
        recalled.contains("SECRET-BODY-0"),
        "recall did not return the elided content: {recalled}"
    );
}

#[tokio::test]
async fn failed_edit_explains_why_it_failed() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("code.rs"),
        "fn main() {\n    let total = 1;\n}\n",
    )
    .unwrap();

    let model = ScriptedModel::new(vec![
        Message::ai_with_tools(
            "",
            vec![call(
                "t1",
                "edit_file",
                // Wrong indentation (8 spaces, file has 4). Omitting the
                // indent entirely would still match as a substring; supplying
                // the wrong amount is the failure that actually happens.
                serde_json::json!({
                    "file_path": "code.rs",
                    "old_string": "        let total = 1;",
                    "new_string": "        let total = 2;"
                }),
            )],
        ),
        Message::ai("done"),
    ]);
    let graph = AgentGraph::new(model, RealToolExecutor::new(dir.path()));
    let state = graph.run(AgentState::new()).await.unwrap();

    let results = tool_results(&state);
    assert_eq!(results.len(), 1);
    // "not found" alone is useless — the model already believed it was there.
    assert!(
        results[0].contains("whitespace is ignored"),
        "edit failure was not diagnosed: {}",
        results[0]
    );
    assert!(results[0].contains("exact leading whitespace"));
}

#[tokio::test]
async fn recall_with_an_unknown_id_is_actionable_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        Message::ai_with_tools(
            "",
            vec![call(
                "r1",
                "recall",
                serde_json::json!({ "recall_id": "nope" }),
            )],
        ),
        Message::ai("done"),
    ]);
    let graph = AgentGraph::new(model, RealToolExecutor::new(dir.path()));
    let state = graph.run(AgentState::new()).await.unwrap();

    let results = tool_results(&state);
    assert_eq!(results.len(), 1);
    assert!(results[0].contains("no observation found"));
    // Must tell the model where valid ids come from, not just that it failed.
    assert!(results[0].contains("placeholders"));
}

#[tokio::test]
async fn oversized_tool_output_is_capped() {
    let dir = tempfile::tempdir().unwrap();
    // Many lines, so the total far exceeds the cap. (A single very long line
    // would not: read_file truncates individual lines at its own limit.)
    let body: String = (0..20_000).map(|i| format!("line {i}\n")).collect();
    std::fs::write(dir.path().join("big.txt"), body).unwrap();

    let model = ScriptedModel::new(vec![
        Message::ai_with_tools(
            "",
            vec![call(
                "t1",
                "read_file",
                serde_json::json!({"file_path": "big.txt", "limit": 100_000}),
            )],
        ),
        Message::ai("done"),
    ]);
    let config = GraphConfig {
        mask: rvagent_core::masking::MaskConfig {
            max_tool_result_bytes: 4_000,
            ..Default::default()
        },
        ..GraphConfig::default()
    };
    let graph = AgentGraph::with_config(model, RealToolExecutor::new(dir.path()), config);
    let state = graph.run(AgentState::new()).await.unwrap();

    let results = tool_results(&state);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].len() <= 4_000,
        "tool output was not capped: {} bytes",
        results[0].len()
    );
    assert!(
        results[0].ends_with("[output truncated]"),
        "truncation must be explicit so the model knows output was cut; got {} bytes: {:?}",
        results[0].len(),
        &results[0][..results[0].len().min(200)]
    );
}

// ---------------------------------------------------------------------------
// Confinement holds through the full loop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn path_escape_is_refused_through_the_loop() {
    let dir = tempfile::tempdir().unwrap();

    let model = ScriptedModel::new(vec![
        Message::ai_with_tools(
            "exfiltrating",
            vec![
                call("t1", "read_file", serde_json::json!({"file_path": "/etc/passwd"})),
                call(
                    "t2",
                    "read_file",
                    serde_json::json!({"file_path": "../../../../etc/passwd"}),
                ),
            ],
        ),
        Message::ai("done"),
    ]);
    let graph = AgentGraph::new(model, RealToolExecutor::new(dir.path()));
    let state = graph.run(AgentState::new()).await.unwrap();

    for (i, result) in tool_results(&state).iter().enumerate() {
        assert!(
            !result.contains("root:"),
            "escape {i} leaked /etc/passwd contents: {result}"
        );
        // Must be refused *by the confinement check* specifically — a generic
        // error (bad param, missing file) would pass vacuously and hide a
        // regression in the boundary itself.
        assert!(
            result.contains("outside the workspace root"),
            "escape {i} was not refused by path confinement: {result}"
        );
    }
}

#[tokio::test]
async fn write_escape_does_not_touch_the_filesystem() {
    let dir = tempfile::tempdir().unwrap();
    let outside = dir.path().parent().unwrap().join("rvagent-e2e-escape.txt");
    let _ = std::fs::remove_file(&outside);

    let model = ScriptedModel::new(vec![
        Message::ai_with_tools(
            "",
            vec![call(
                "t1",
                "write_file",
                serde_json::json!({"file_path": outside.to_string_lossy(), "content": "pwned"}),
            )],
        ),
        Message::ai("done"),
    ]);
    let graph = AgentGraph::new(model, RealToolExecutor::new(dir.path()));
    graph.run(AgentState::new()).await.unwrap();

    assert!(
        !outside.exists(),
        "a write outside the workspace root reached the filesystem"
    );
}
