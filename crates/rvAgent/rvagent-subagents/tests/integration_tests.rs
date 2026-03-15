//! Integration tests for rvAgent subagents.

use std::collections::HashMap;

use rvagent_subagents::{
    prepare_subagent_state, extract_result_message, merge_subagent_state,
    AgentState, CompiledSubAgent, SubAgentSpec, RvAgentConfig,
    EXCLUDED_STATE_KEYS,
};
use rvagent_subagents::builder::compile_subagents;
use rvagent_subagents::orchestrator::{SubAgentOrchestrator, spawn_parallel};

fn test_config() -> RvAgentConfig {
    RvAgentConfig {
        default_model: Some("anthropic:claude-sonnet-4-20250514".into()),
        tools: vec!["read_file".into(), "write_file".into()],
        middleware: vec!["prompt_caching".into()],
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
    state.insert("messages".into(), serde_json::json!([
        {"type": "system", "content": "You are helpful."},
        {"type": "human", "content": "Do something."},
    ]));
    state.insert("remaining_steps".into(), serde_json::json!(10));
    state.insert("task_completion".into(), serde_json::json!({"status": "in_progress"}));
    state.insert("files".into(), serde_json::json!({"main.rs": "fn main() {}"}));
    state.insert("custom_data".into(), serde_json::json!("value"));
    state
}

#[test]
fn test_compile_subagent() {
    let config = test_config();
    let mut spec_read = SubAgentSpec::new("helper", "A helper agent");
    spec_read.can_read = true;

    let mut spec_write = SubAgentSpec::new("writer", "A writer agent");
    spec_write.can_read = true;
    spec_write.can_write = true;

    let compiled = compile_subagents(&[spec_read, spec_write], &config);
    assert_eq!(compiled.len(), 2);
    assert_eq!(compiled[0].spec.name, "helper");
    assert_eq!(compiled[1].spec.name, "writer");
    assert!(compiled[0].spec.can_read);
    assert!(!compiled[0].spec.can_write);
    assert!(compiled[1].spec.can_write);
}

#[test]
fn test_state_isolation() {
    let parent = parent_state_with_data();
    let child = prepare_subagent_state(&parent, "Do a subtask");

    // remaining_steps and task_completion should be excluded
    assert!(!child.contains_key("remaining_steps"), "remaining_steps leaked");
    assert!(!child.contains_key("task_completion"), "task_completion leaked");

    // messages is re-created with the task description, not the parent's messages
    let child_msgs = child.get("messages").unwrap().as_array().unwrap();
    assert_eq!(child_msgs.len(), 1);
    assert!(child_msgs[0]["content"].as_str().unwrap().contains("subtask"));

    // Non-excluded keys should be present
    assert!(child.contains_key("files"));
    assert!(child.contains_key("custom_data"));
}

#[test]
fn test_extract_result_message() {
    let mut state = AgentState::new();
    state.insert("messages".into(), serde_json::json!([
        {"type": "ai", "content": "Working..."},
        {"type": "ai", "content": "Done! Here is the result."}
    ]));

    let result = extract_result_message(&state);
    assert!(result.is_some());
    assert!(result.unwrap().contains("Done!"));
}

#[test]
fn test_merge_preserves_parent_messages() {
    let mut parent = parent_state_with_data();
    let parent_msgs = parent.get("messages").cloned();

    let mut child_result = AgentState::new();
    child_result.insert("messages".into(), serde_json::json!([{"type": "ai", "content": "child"}]));
    child_result.insert("new_key".into(), serde_json::json!("from child"));

    merge_subagent_state(&mut parent, &child_result);

    // Parent messages must not be overwritten
    assert_eq!(parent.get("messages"), parent_msgs.as_ref());
    // New keys from child should be merged
    assert_eq!(parent.get("new_key"), Some(&serde_json::json!("from child")));
}

#[test]
fn test_subagent_spawn_and_collect() {
    let agents = vec![mock_compiled("researcher")];
    let orch = SubAgentOrchestrator::new(agents);
    let parent = parent_state_with_data();

    let result = orch.spawn_sync("researcher", &parent, "Research topic X");
    assert!(result.is_ok());
    let r = result.unwrap();
    assert_eq!(r.agent_name, "researcher");
    assert!(r.result_message.contains("Research topic X"));
}

#[tokio::test]
async fn test_parallel_subagent_execution() {
    let agents = vec![
        mock_compiled("agent-a"),
        mock_compiled("agent-b"),
        mock_compiled("agent-c"),
    ];
    let orch = SubAgentOrchestrator::new(agents);
    let parent = parent_state_with_data();

    let tasks = vec![
        ("agent-a", &parent, "Task A"),
        ("agent-b", &parent, "Task B"),
        ("agent-c", &parent, "Task C"),
    ];

    let results = spawn_parallel(&orch, tasks).await;
    assert_eq!(results.len(), 3);
}

#[test]
fn test_compilation_respects_capabilities() {
    let config = test_config();
    let read_only = SubAgentSpec::new("reader", "Read only");
    let mut full = SubAgentSpec::new("full", "Full access");
    full.can_write = true;
    full.can_execute = true;

    let compiled = compile_subagents(&[read_only, full], &config);
    assert_eq!(compiled.len(), 2);
}

#[test]
fn test_extract_result_empty_messages() {
    let state = AgentState::new();
    assert!(extract_result_message(&state).is_none());

    let mut state2 = AgentState::new();
    state2.insert("messages".into(), serde_json::json!([]));
    assert!(extract_result_message(&state2).is_none());
}
