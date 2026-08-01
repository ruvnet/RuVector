//! Integration tests for rvAgent subagents.

use rvagent_core::messages::Message;
use rvagent_core::state::{FileData, TodoItem, TodoStatus};
use rvagent_subagents::builder::compile_subagents;
use rvagent_subagents::orchestrator::{spawn_parallel, SubAgentOrchestrator};
use rvagent_subagents::{
    extract_result_message, merge_subagent_state, prepare_subagent_state, AgentState,
    CompiledSubAgent, RvAgentConfig, SubAgentSpec,
};

fn file_data(content: &str) -> FileData {
    FileData {
        content: content.into(),
        encoding: "utf-8".into(),
        modified_at: None,
    }
}

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
    let mut state = AgentState::with_system_message("You are helpful.");
    state.push_message(Message::human("Do something."));
    state.push_todo(TodoItem {
        content: "parent task".into(),
        status: TodoStatus::InProgress,
        active_form: String::new(),
    });
    state.set_file("main.rs", file_data("fn main() {}"));
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

    // Parent todos should be excluded
    assert!(child.todos.is_empty(), "todos leaked");

    // messages is re-created with the task description, not the parent's messages
    assert_eq!(child.message_count(), 1);
    assert!(child.messages[0].content().contains("subtask"));

    // Non-excluded state (files) should be present
    assert!(child.files.contains_key("main.rs"));
}

#[test]
fn test_extract_result_message() {
    let mut state = AgentState::new();
    state.push_message(Message::ai("Working..."));
    state.push_message(Message::ai("Done! Here is the result."));

    let result = extract_result_message(&state);
    assert!(result.is_some());
    assert!(result.unwrap().contains("Done!"));
}

#[test]
fn test_merge_preserves_parent_messages() {
    let mut parent = parent_state_with_data();
    let parent_msg_count = parent.message_count();

    let mut child_result = AgentState::new();
    child_result.push_message(Message::ai("child"));
    child_result.set_file("child.rs", file_data("from child"));

    merge_subagent_state(&mut parent, &child_result);

    // Parent messages must not be overwritten
    assert_eq!(parent.message_count(), parent_msg_count);
    // New files from child should be merged
    assert!(parent.files.contains_key("child.rs"));
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
}
