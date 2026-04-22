//! Integration tests for AgentState (ADR-103 A1).
//!
//! Tests verify the typed AgentState with Arc-based shallow cloning,
//! extension map, serialization, and todo-item status transitions.

use std::sync::Arc;

use rvagent_core::messages::Message;
use rvagent_core::state::{AgentState, FileData, SkillMetadata, TodoItem, TodoStatus};

/// Cloning AgentState must be a shallow Arc clone (O(1)), not a deep copy.
#[test]
fn test_state_clone_is_shallow() {
    let mut state = AgentState::new();
    state.push_message(Message::human("hello"));
    state.push_message(Message::human("world"));

    let cloned = state.clone();

    // Both should point to the exact same Arc allocation.
    assert!(Arc::ptr_eq(&state.messages, &cloned.messages));
    assert!(Arc::ptr_eq(&state.todos, &cloned.todos));
    assert!(Arc::ptr_eq(&state.files, &cloned.files));

    // Extensions are NOT cloned (agent-local).
    state.set_extension("key", 42_u64);
    let cloned2 = state.clone();
    assert!(cloned2.get_extension::<u64>("key").is_none());
}

/// Default state should have empty collections and no memory/skills.
#[test]
fn test_state_default_values() {
    let state = AgentState::default();

    assert_eq!(state.message_count(), 0);
    assert!(state.messages.is_empty());
    assert!(state.todos.is_empty());
    assert!(state.files.is_empty());
    assert!(state.memory_contents.is_none());
    assert!(state.skills_metadata.is_none());
}

/// Merging sub-agent results should append messages, merge files
/// (child wins on conflict), and append todos.
#[test]
fn test_state_merge_subagent_results() {
    let mut parent = AgentState::new();
    parent.push_message(Message::system("parent sys"));
    parent.set_file(
        "existing.txt",
        FileData {
            content: "old content".into(),
            encoding: "utf-8".into(),
            modified_at: None,
        },
    );

    let mut child = AgentState::new();
    child.push_message(Message::ai("child response"));
    // Child overwrites existing file.
    child.set_file(
        "existing.txt",
        FileData {
            content: "new content".into(),
            encoding: "utf-8".into(),
            modified_at: Some("2026-03-14T12:00:00Z".into()),
        },
    );
    // Child adds a new file.
    child.set_file(
        "new_file.txt",
        FileData {
            content: "brand new".into(),
            encoding: "utf-8".into(),
            modified_at: None,
        },
    );
    child.push_todo(TodoItem {
        content: "child task".into(),
        status: TodoStatus::Completed,
        active_form: "Completing child task".into(),
    });

    parent.merge_subagent(&child);

    // Messages appended.
    assert_eq!(parent.message_count(), 2);
    assert_eq!(parent.messages[0].content(), "parent sys");
    assert_eq!(parent.messages[1].content(), "child response");

    // Child file wins on conflict.
    assert_eq!(parent.files["existing.txt"].content, "new content");

    // New file added.
    assert!(parent.files.contains_key("new_file.txt"));

    // Todos appended.
    assert_eq!(parent.todos.len(), 1);
    assert_eq!(parent.todos[0].content, "child task");
}

/// Extension map should allow inserting and retrieving typed values.
#[test]
fn test_state_extension_insert_retrieve() {
    let mut state = AgentState::new();

    state.set_extension("counter", 42_u64);
    state.set_extension("label", "test".to_string());

    assert_eq!(state.get_extension::<u64>("counter"), Some(&42));
    assert_eq!(
        state.get_extension::<String>("label"),
        Some(&"test".to_string())
    );

    // Wrong type should return None.
    assert_eq!(state.get_extension::<String>("counter"), None);

    // Missing key should return None.
    assert_eq!(state.get_extension::<u64>("missing"), None);
}

/// AgentState's serializable sub-types should survive a JSON round-trip.
#[test]
fn test_state_serialization_roundtrip() {
    // TodoItem round-trip.
    let todos = vec![
        TodoItem {
            content: "write tests".to_string(),
            status: TodoStatus::Pending,
            active_form: "Writing tests".to_string(),
        },
        TodoItem {
            content: "run ci".to_string(),
            status: TodoStatus::Completed,
            active_form: "Running CI".to_string(),
        },
    ];

    let todos_json = serde_json::to_string(&todos).unwrap();
    let todos_back: Vec<TodoItem> = serde_json::from_str(&todos_json).unwrap();
    assert_eq!(todos, todos_back);

    // FileData round-trip.
    let fd = FileData {
        content: "fn main() {}".into(),
        encoding: "utf-8".into(),
        modified_at: Some("2026-03-14T00:00:00Z".into()),
    };
    let fd_json = serde_json::to_string(&fd).unwrap();
    let fd_back: FileData = serde_json::from_str(&fd_json).unwrap();
    assert_eq!(fd.content, fd_back.content);
    assert_eq!(fd.encoding, fd_back.encoding);
    assert_eq!(fd.modified_at, fd_back.modified_at);

    // SkillMetadata round-trip.
    let sm = SkillMetadata {
        name: "deploy".into(),
        description: "Deploy the application".into(),
        parameters: serde_json::json!({"target": "production"}),
    };
    let sm_json = serde_json::to_string(&sm).unwrap();
    let sm_back: SkillMetadata = serde_json::from_str(&sm_json).unwrap();
    assert_eq!(sm, sm_back);
}

/// TodoItem status transitions: Pending -> InProgress -> Completed.
#[test]
fn test_todo_item_status_transitions() {
    let mut item = TodoItem {
        content: "implement feature".to_string(),
        status: TodoStatus::Pending,
        active_form: "Implementing feature".to_string(),
    };

    assert_eq!(item.status, TodoStatus::Pending);

    // Pending -> InProgress
    item.status = TodoStatus::InProgress;
    assert_eq!(item.status, TodoStatus::InProgress);

    // InProgress -> Completed
    item.status = TodoStatus::Completed;
    assert_eq!(item.status, TodoStatus::Completed);

    // Serialization of status values uses snake_case.
    let json = serde_json::to_string(&TodoStatus::InProgress).unwrap();
    assert_eq!(json, r#""in_progress""#);

    let back: TodoStatus = serde_json::from_str(r#""pending""#).unwrap();
    assert_eq!(back, TodoStatus::Pending);

    let back2: TodoStatus = serde_json::from_str(r#""completed""#).unwrap();
    assert_eq!(back2, TodoStatus::Completed);
}
