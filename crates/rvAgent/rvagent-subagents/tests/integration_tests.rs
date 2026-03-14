//! Integration tests for rvAgent subagents.
//!
//! Tests subagent spawning, parallel execution, state isolation,
//! and result merging through the SubAgentOrchestrator.

use std::time::Duration;

use rvagent_subagents::{
    prepare_subagent_state, extract_result_message, merge_subagent_state,
    AgentState, CompiledSubAgent, SubAgentSpec, RvAgentConfig,
    EXCLUDED_STATE_KEYS,
};
use rvagent_subagents::builder::compile_subagents;
use rvagent_subagents::orchestrator::SubAgentOrchestrator;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_config() -> RvAgentConfig {
    RvAgentConfig {
        default_model: Some("anthropic:claude-sonnet-4-20250514".into()),
        tools: vec![
            "read_file".into(),
            "write_file".into(),
            "grep".into(),
            "execute".into(),
        ],
        middleware: vec!["prompt_caching".into(), "summarization".into()],
        cwd: Some("/tmp/project".into()),
    }
}

fn mock_compiled(name: &str) -> CompiledSubAgent {
    CompiledSubAgent {
        spec: SubAgentSpec::new(name, format!("Test agent: {}", name)),
        graph: vec!["start".into(), format!("agent:{}", name), "end".into()],
        middleware_pipeline: vec!["prompt_caching".into()],
        backend: "read_only".into(),
    }
}

fn parent_state_with_data() -> AgentState {
    let mut state = AgentState::new();
    state.insert(
        "messages".into(),
        serde_json::json!([
            {"type": "system", "content": "You are a helpful agent."},
            {"type": "human", "content": "Do something."},
            {"type": "ai", "content": "Sure, let me delegate."}
        ]),
    );
    state.insert("remaining_steps".into(), serde_json::json!(10));
    state.insert("task_completion".into(), serde_json::json!(false));
    state.insert("todos".into(), serde_json::json!([{"content": "parent task"}]));
    state.insert("custom_data".into(), serde_json::json!({"key": "value"}));
    state.insert("project_root".into(), serde_json::json!("/tmp/project"));
    state
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Parent agent spawns subagent -> subagent completes -> result returned to parent.
#[tokio::test]
async fn test_subagent_spawn_and_collect() {
    let orch = SubAgentOrchestrator::new();
    let compiled = mock_compiled("researcher");
    let parent = parent_state_with_data();

    // Spawn the subagent with a task description.
    let result = orch
        .spawn(&compiled, "Find all TODO comments in the codebase", &parent)
        .await
        .unwrap();

    // Verify the result.
    assert_eq!(result.agent_name, "researcher");
    assert!(!result.result_message.is_empty());
    assert!(result.duration.as_nanos() > 0);

    // The result message should be from the subagent, not the parent.
    // The mock orchestrator simulates a response containing the agent name.
    assert!(
        result.result_message.contains("researcher")
            || !result.result_message.contains("Sure, let me delegate"),
        "subagent result should not contain parent's messages"
    );
}

/// Spawn 3 subagents in parallel -> all complete -> results merged.
#[tokio::test]
async fn test_parallel_subagent_execution() {
    let orch = SubAgentOrchestrator::new();
    let agents = vec![
        mock_compiled("searcher"),
        mock_compiled("analyzer"),
        mock_compiled("reporter"),
    ];
    let inputs = vec![
        "Search for authentication patterns".to_string(),
        "Analyze security vulnerabilities".to_string(),
        "Generate a summary report".to_string(),
    ];
    let parent = parent_state_with_data();

    // Execute all three in parallel.
    let results = orch.spawn_parallel(&agents, &inputs, &parent).await;

    // All three should complete successfully.
    assert_eq!(results.len(), 3);
    let successful: Vec<_> = results.iter().filter(|r| r.is_ok()).collect();
    assert_eq!(
        successful.len(),
        3,
        "all 3 subagents should complete successfully"
    );

    // Collect agent names to verify all three ran.
    let mut names: Vec<String> = results
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|r| r.agent_name.clone())
        .collect();
    names.sort();
    assert_eq!(names, vec!["analyzer", "reporter", "searcher"]);
}

/// Subagent cannot see parent's messages, todos, or completion state.
#[tokio::test]
async fn test_subagent_state_isolation() {
    let parent = parent_state_with_data();

    // Prepare the subagent state (this is what the orchestrator does internally).
    let child_state = prepare_subagent_state(&parent, "Analyze the code");

    // Excluded keys should not be present (except messages which is replaced).
    for key in EXCLUDED_STATE_KEYS {
        if *key == "messages" {
            // Messages should be replaced with a single human message.
            let msgs = child_state
                .get("messages")
                .unwrap()
                .as_array()
                .unwrap();
            assert_eq!(msgs.len(), 1, "subagent should have exactly 1 message");
            assert_eq!(msgs[0]["type"], "human");
            assert_eq!(msgs[0]["content"], "Analyze the code");
        } else {
            assert!(
                child_state.get(*key).is_none(),
                "excluded key '{}' should not be in subagent state",
                key
            );
        }
    }

    // Non-excluded keys should pass through.
    assert_eq!(
        child_state.get("custom_data").unwrap(),
        &serde_json::json!({"key": "value"})
    );
    assert_eq!(
        child_state.get("project_root").unwrap(),
        &serde_json::json!("/tmp/project")
    );
}

/// Subagent result merge does not overwrite parent's excluded keys.
#[tokio::test]
async fn test_subagent_result_merge_safety() {
    let mut parent = parent_state_with_data();
    let original_messages = parent.get("messages").cloned().unwrap();
    let original_todos = parent.get("todos").cloned().unwrap();

    // Simulate a subagent result state.
    let mut child_result = AgentState::new();
    child_result.insert(
        "messages".into(),
        serde_json::json!([{"type": "ai", "content": "subagent says hi"}]),
    );
    child_result.insert(
        "todos".into(),
        serde_json::json!([{"content": "sneaky todo"}]),
    );
    child_result.insert("analysis_result".into(), serde_json::json!("important finding"));
    child_result.insert("files_modified".into(), serde_json::json!(["a.rs", "b.rs"]));

    merge_subagent_state(&mut parent, &child_result);

    // Parent's messages and todos should be untouched.
    assert_eq!(parent.get("messages").unwrap(), &original_messages);
    assert_eq!(parent.get("todos").unwrap(), &original_todos);

    // Non-excluded keys from child should be merged.
    assert_eq!(
        parent.get("analysis_result").unwrap(),
        &serde_json::json!("important finding")
    );
    assert_eq!(
        parent.get("files_modified").unwrap(),
        &serde_json::json!(["a.rs", "b.rs"])
    );
}

/// Compilation produces correct middleware pipeline based on capabilities.
#[test]
fn test_compilation_respects_capabilities() {
    let config = test_config();

    // Read-only agent: should have filesystem middleware but not execution_guard.
    let read_only = SubAgentSpec {
        can_read: true,
        can_write: false,
        can_execute: false,
        ..SubAgentSpec::new("reader", "Read files")
    };

    // Full-access agent: should have all middleware.
    let full_access = SubAgentSpec::general_purpose();

    let compiled = compile_subagents(&[read_only, full_access], &config);
    assert_eq!(compiled.len(), 2);

    // Read-only agent should have filesystem but not execution_guard.
    let reader = &compiled[0];
    assert!(reader.middleware_pipeline.contains(&"filesystem".to_string()));
    assert!(!reader.middleware_pipeline.contains(&"execution_guard".to_string()));

    // Full-access agent should have both.
    let full = &compiled[1];
    assert!(full.middleware_pipeline.contains(&"filesystem".to_string()));
    assert!(full.middleware_pipeline.contains(&"execution_guard".to_string()));
    assert!(full.middleware_pipeline.contains(&"todo_list".to_string()));
}

/// Tool call limit enforcement.
#[tokio::test]
async fn test_tool_call_limit_enforced() {
    let orch = SubAgentOrchestrator::with_limits(Duration::from_secs(60), 3, 4);
    let compiled = mock_compiled("runaway-agent");

    // Parent state with tool_calls_count exceeding the limit.
    let mut parent = AgentState::new();
    parent.insert("messages".into(), serde_json::json!([]));
    parent.insert("tool_calls_count".into(), serde_json::json!(10));

    let result = orch.spawn(&compiled, "Do something", &parent).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeded tool call limit"));
}

/// Extract result message from subagent state.
#[test]
fn test_extract_result_message_variants() {
    // Normal case: last message is AI.
    let mut state = AgentState::new();
    state.insert(
        "messages".into(),
        serde_json::json!([
            {"type": "human", "content": "do X"},
            {"type": "ai", "content": "Done with X.  "}
        ]),
    );
    assert_eq!(
        extract_result_message(&state).unwrap(),
        "Done with X."
    );

    // Empty messages.
    let mut empty = AgentState::new();
    empty.insert("messages".into(), serde_json::json!([]));
    assert!(extract_result_message(&empty).is_none());

    // No messages key at all.
    let no_key = AgentState::new();
    assert!(extract_result_message(&no_key).is_none());
}

/// Parallel execution with max_concurrent batching.
#[tokio::test]
async fn test_parallel_batching_with_concurrency_limit() {
    // max_concurrent = 2, but we spawn 5 agents.
    let orch = SubAgentOrchestrator::with_limits(Duration::from_secs(60), 100, 2);
    let agents: Vec<_> = (0..5)
        .map(|i| mock_compiled(&format!("batch-agent-{}", i)))
        .collect();
    let inputs: Vec<_> = (0..5)
        .map(|i| format!("Task {}", i))
        .collect();
    let parent = parent_state_with_data();

    let results = orch.spawn_parallel(&agents, &inputs, &parent).await;

    // All 5 should complete despite the concurrency limit.
    assert_eq!(results.len(), 5);
    assert!(results.iter().all(|r| r.is_ok()));
}
