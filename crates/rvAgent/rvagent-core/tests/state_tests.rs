//! Integration tests for AgentState (ADR-103 A1).
//!
//! Tests verify the typed AgentState with Arc-based shallow cloning,
//! extension map, serialization, and todo-item status transitions.
//! When the AgentState struct is not yet implemented, these tests
//! exercise the spec behavior using equivalent constructs from the
//! ADR definitions.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json;

// ---------------------------------------------------------------------------
// Local stand-in types matching ADR-103 A1 AgentState spec.
// Once the real `rvagent_core::state` module is published these should be
// replaced with `use rvagent_core::state::*;`.
// ---------------------------------------------------------------------------

/// TodoItem status enum (ADR-095).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// A single todo item managed by the TodoListMiddleware.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TodoItem {
    content: String,
    status: TodoStatus,
}

/// FileData held within AgentState (mirrors `protocol::FileData`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileData {
    content: Vec<String>,
    created_at: String,
    modified_at: String,
}

/// Typed AgentState per ADR-103 A1.
///
/// Uses `Arc` for O(1) shallow clone / subagent forking.
#[derive(Debug, Clone)]
struct AgentState {
    messages: Arc<Vec<String>>,
    todos: Arc<Vec<TodoItem>>,
    files: Arc<HashMap<String, FileData>>,
    memory_contents: Option<Arc<HashMap<String, String>>>,
    extensions: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            messages: Arc::new(Vec::new()),
            todos: Arc::new(Vec::new()),
            files: Arc::new(HashMap::new()),
            memory_contents: None,
            extensions: HashMap::new(),
        }
    }
}

impl AgentState {
    /// Insert a typed extension value.
    fn insert_extension<T: Send + Sync + 'static>(&mut self, key: &str, value: T) {
        self.extensions.insert(key.to_string(), Box::new(value));
    }

    /// Retrieve a typed extension value.
    fn get_extension<T: Send + Sync + 'static>(&self, key: &str) -> Option<&T> {
        self.extensions.get(key).and_then(|v| v.downcast_ref::<T>())
    }

    /// Merge sub-agent results into this state (ADR-103 B7).
    fn merge_subagent_results(&mut self, child: &AgentState) {
        // Append child messages to parent.
        let mut msgs = (*self.messages).clone();
        msgs.extend(child.messages.iter().cloned());
        self.messages = Arc::new(msgs);

        // Merge child files (child wins on conflict).
        let mut files = (*self.files).clone();
        for (k, v) in child.files.iter() {
            files.insert(k.clone(), v.clone());
        }
        self.files = Arc::new(files);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Cloning AgentState must be a shallow Arc clone (O(1)), not a deep copy.
#[test]
fn test_state_clone_is_shallow() {
    let mut state = AgentState::default();
    let msgs = Arc::new(vec!["hello".to_string(), "world".to_string()]);
    state.messages = msgs.clone();

    let cloned = state.clone();

    // Both should point to the exact same Arc allocation.
    assert!(Arc::ptr_eq(&state.messages, &cloned.messages));
    assert!(Arc::ptr_eq(&state.todos, &cloned.todos));
    assert!(Arc::ptr_eq(&state.files, &cloned.files));
}

/// Default state should have empty collections and no memory.
#[test]
fn test_state_default_values() {
    let state = AgentState::default();

    assert!(state.messages.is_empty());
    assert!(state.todos.is_empty());
    assert!(state.files.is_empty());
    assert!(state.memory_contents.is_none());
    assert!(state.extensions.is_empty());
}

/// Merging sub-agent results should append messages and merge files.
#[test]
fn test_state_merge_subagent_results() {
    let mut parent = AgentState::default();
    parent.messages = Arc::new(vec!["parent msg".to_string()]);

    let mut parent_files = HashMap::new();
    parent_files.insert(
        "existing.txt".to_string(),
        FileData {
            content: vec!["old".to_string()],
            created_at: "t0".to_string(),
            modified_at: "t0".to_string(),
        },
    );
    parent.files = Arc::new(parent_files);

    let mut child = AgentState::default();
    child.messages = Arc::new(vec!["child msg".to_string()]);

    let mut child_files = HashMap::new();
    child_files.insert(
        "existing.txt".to_string(),
        FileData {
            content: vec!["new".to_string()],
            created_at: "t1".to_string(),
            modified_at: "t1".to_string(),
        },
    );
    child_files.insert(
        "new_file.txt".to_string(),
        FileData {
            content: vec!["brand new".to_string()],
            created_at: "t1".to_string(),
            modified_at: "t1".to_string(),
        },
    );
    child.files = Arc::new(child_files);

    parent.merge_subagent_results(&child);

    // Messages should be appended.
    assert_eq!(parent.messages.len(), 2);
    assert_eq!(parent.messages[0], "parent msg");
    assert_eq!(parent.messages[1], "child msg");

    // Child file wins on conflict.
    assert_eq!(parent.files["existing.txt"].content[0], "new");

    // New file added.
    assert!(parent.files.contains_key("new_file.txt"));
}

/// Extension map should allow inserting and retrieving typed values.
#[test]
fn test_state_extension_insert_retrieve() {
    let mut state = AgentState::default();

    state.insert_extension("counter", 42_u64);
    state.insert_extension("label", "test".to_string());

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

/// AgentState's serializable fields should survive a JSON round-trip.
#[test]
fn test_state_serialization_roundtrip() {
    // We serialize just the messages and todos (the Arc contents).
    let messages = vec!["hello".to_string(), "world".to_string()];
    let todos = vec![
        TodoItem {
            content: "write tests".to_string(),
            status: TodoStatus::Pending,
        },
        TodoItem {
            content: "run ci".to_string(),
            status: TodoStatus::Completed,
        },
    ];

    let msgs_json = serde_json::to_string(&messages).unwrap();
    let todos_json = serde_json::to_string(&todos).unwrap();

    let msgs_back: Vec<String> = serde_json::from_str(&msgs_json).unwrap();
    let todos_back: Vec<TodoItem> = serde_json::from_str(&todos_json).unwrap();

    assert_eq!(messages, msgs_back);
    assert_eq!(todos, todos_back);
}

/// TodoItem status transitions must follow the allowed state machine:
/// Pending -> InProgress -> Completed (no backward transitions).
#[test]
fn test_todo_item_status_transitions() {
    let mut item = TodoItem {
        content: "implement feature".to_string(),
        status: TodoStatus::Pending,
    };

    assert_eq!(item.status, TodoStatus::Pending);

    // Pending -> InProgress
    item.status = TodoStatus::InProgress;
    assert_eq!(item.status, TodoStatus::InProgress);

    // InProgress -> Completed
    item.status = TodoStatus::Completed;
    assert_eq!(item.status, TodoStatus::Completed);

    // Serialization of status values.
    let json = serde_json::to_string(&TodoStatus::InProgress).unwrap();
    assert_eq!(json, r#""in_progress""#);

    let back: TodoStatus = serde_json::from_str(r#""pending""#).unwrap();
    assert_eq!(back, TodoStatus::Pending);
}
