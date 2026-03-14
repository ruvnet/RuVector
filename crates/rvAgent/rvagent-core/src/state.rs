//! Typed `AgentState` — ADR-103 A1.
//!
//! Replaces `HashMap<String, serde_json::Value>` with a strongly-typed struct
//! using `Arc` for O(1) clone on subagent spawn.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::messages::Message;

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Status of a to-do item managed by the TodoList middleware.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// A to-do item (mirrors the task tracking structure).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TodoItem {
    /// Description of what needs to be done.
    pub content: String,
    /// Current status.
    pub status: TodoStatus,
    /// Present-continuous form shown during execution.
    #[serde(default)]
    pub active_form: String,
}

/// File data tracked in agent state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileData {
    /// File content (may be truncated for large files).
    pub content: String,
    /// Encoding (typically "utf-8").
    #[serde(default = "default_encoding")]
    pub encoding: String,
    /// Last modified timestamp.
    #[serde(default)]
    pub modified_at: Option<DateTime<Utc>>,
}

fn default_encoding() -> String {
    "utf-8".into()
}

/// Metadata for a discovered skill.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkillMetadata {
    /// Skill name (ASCII lowercase + digits + hyphens per ADR-103 C10).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Parameter definitions as JSON schema fragments.
    #[serde(default)]
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// AgentState
// ---------------------------------------------------------------------------

/// Typed agent state using `Arc` for O(1) clone (ADR-103 A1).
///
/// Core fields are wrapped in `Arc` so that cloning the entire state (e.g.
/// when spawning a subagent) is a constant-time reference count increment
/// rather than a deep copy.
///
/// The `extensions` map provides escape-hatch extensibility for middleware
/// that needs custom state not covered by the core fields.
pub struct AgentState {
    /// Conversation messages (system, human, ai, tool).
    pub messages: Arc<Vec<Message>>,

    /// Task tracking items.
    pub todos: Arc<Vec<TodoItem>>,

    /// Files read/written during the agent session.
    pub files: Arc<HashMap<String, FileData>>,

    /// Memory contents loaded from AGENTS.md / memory sources.
    pub memory_contents: Option<Arc<HashMap<String, String>>>,

    /// Skill metadata for progressive disclosure.
    pub skills_metadata: Option<Arc<Vec<SkillMetadata>>>,

    /// Extension slot for middleware-defined state.
    /// Keyed by a unique string identifier per middleware.
    extensions: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl AgentState {
    /// Create a new empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create state initialized with a system message.
    pub fn with_system_message(content: impl Into<String>) -> Self {
        Self {
            messages: Arc::new(vec![Message::system(content)]),
            ..Default::default()
        }
    }

    /// Return the number of messages.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Append a message, cloning the Arc only when needed (copy-on-write).
    pub fn push_message(&mut self, msg: Message) {
        Arc::make_mut(&mut self.messages).push(msg);
    }

    /// Append a to-do item.
    pub fn push_todo(&mut self, item: TodoItem) {
        Arc::make_mut(&mut self.todos).push(item);
    }

    /// Insert or update a file entry.
    pub fn set_file(&mut self, path: impl Into<String>, data: FileData) {
        Arc::make_mut(&mut self.files).insert(path.into(), data);
    }

    /// Get an extension value by key, downcasting to the expected type.
    pub fn get_extension<T: 'static + Send + Sync>(&self, key: &str) -> Option<&T> {
        self.extensions.get(key)?.downcast_ref()
    }

    /// Set an extension value.
    pub fn set_extension<T: 'static + Send + Sync>(&mut self, key: impl Into<String>, value: T) {
        self.extensions.insert(key.into(), Box::new(value));
    }

    /// Merge results from a subagent into this (parent) state.
    ///
    /// Strategy: append subagent messages, merge files (subagent wins on conflict),
    /// merge todos. Memory and skills are not merged (they are parent-owned).
    pub fn merge_subagent(&mut self, child: &AgentState) {
        // Append child messages.
        let parent_msgs = Arc::make_mut(&mut self.messages);
        parent_msgs.extend(child.messages.iter().cloned());

        // Merge files — child wins on conflict.
        if !child.files.is_empty() {
            let parent_files = Arc::make_mut(&mut self.files);
            for (path, data) in child.files.iter() {
                parent_files.insert(path.clone(), data.clone());
            }
        }

        // Append child todos.
        if !child.todos.is_empty() {
            let parent_todos = Arc::make_mut(&mut self.todos);
            parent_todos.extend(child.todos.iter().cloned());
        }
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            messages: Arc::new(Vec::new()),
            todos: Arc::new(Vec::new()),
            files: Arc::new(HashMap::new()),
            memory_contents: None,
            skills_metadata: None,
            extensions: HashMap::new(),
        }
    }
}

impl Clone for AgentState {
    /// Clone is O(1) for the Arc-wrapped fields.
    /// Extensions are not cloned (they are agent-local).
    fn clone(&self) -> Self {
        Self {
            messages: Arc::clone(&self.messages),
            todos: Arc::clone(&self.todos),
            files: Arc::clone(&self.files),
            memory_contents: self.memory_contents.as_ref().map(Arc::clone),
            skills_metadata: self.skills_metadata.as_ref().map(Arc::clone),
            extensions: HashMap::new(), // Extensions are not shared across clones.
        }
    }
}

impl fmt::Debug for AgentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentState")
            .field("messages", &self.messages.len())
            .field("todos", &self.todos.len())
            .field("files", &self.files.len())
            .field("memory_contents", &self.memory_contents.is_some())
            .field("skills_metadata", &self.skills_metadata.is_some())
            .field("extensions", &self.extensions.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_default_state() {
        let state = AgentState::default();
        assert_eq!(state.message_count(), 0);
        assert!(state.todos.is_empty());
        assert!(state.files.is_empty());
        assert!(state.memory_contents.is_none());
        assert!(state.skills_metadata.is_none());
    }

    #[test]
    fn test_with_system_message() {
        let state = AgentState::with_system_message("hello");
        assert_eq!(state.message_count(), 1);
        assert_eq!(state.messages[0].content(), "hello");
    }

    #[test]
    fn test_clone_is_o1_arc() {
        let mut state = AgentState::new();
        // Add some data.
        for i in 0..1000 {
            state.push_message(Message::human(format!("msg {}", i)));
        }
        // Clone should share the same Arc pointer.
        let cloned = state.clone();
        assert!(Arc::ptr_eq(&state.messages, &cloned.messages));
        assert!(Arc::ptr_eq(&state.todos, &cloned.todos));
        assert!(Arc::ptr_eq(&state.files, &cloned.files));
    }

    #[test]
    fn test_push_message_cow() {
        let mut state = AgentState::new();
        state.push_message(Message::human("first"));

        // Clone shares the arc.
        let snapshot = state.clone();
        assert!(Arc::ptr_eq(&state.messages, &snapshot.messages));

        // Mutating state triggers copy-on-write.
        state.push_message(Message::human("second"));
        assert!(!Arc::ptr_eq(&state.messages, &snapshot.messages));
        assert_eq!(state.message_count(), 2);
        assert_eq!(snapshot.message_count(), 1);
    }

    #[test]
    fn test_set_and_get_file() {
        let mut state = AgentState::new();
        state.set_file(
            "/tmp/test.rs",
            FileData {
                content: "fn main() {}".into(),
                encoding: "utf-8".into(),
                modified_at: None,
            },
        );
        assert!(state.files.contains_key("/tmp/test.rs"));
    }

    #[test]
    fn test_extensions() {
        let mut state = AgentState::new();
        state.set_extension("counter", 42u64);
        assert_eq!(state.get_extension::<u64>("counter"), Some(&42u64));
        assert_eq!(state.get_extension::<String>("counter"), None);
        assert_eq!(state.get_extension::<u64>("missing"), None);
    }

    #[test]
    fn test_extensions_not_cloned() {
        let mut state = AgentState::new();
        state.set_extension("key", "value".to_string());
        let cloned = state.clone();
        assert!(cloned.get_extension::<String>("key").is_none());
    }

    #[test]
    fn test_merge_subagent() {
        let mut parent = AgentState::new();
        parent.push_message(Message::system("parent sys"));

        let mut child = AgentState::new();
        child.push_message(Message::ai("child response"));
        child.set_file(
            "/tmp/new.rs",
            FileData {
                content: "// new".into(),
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
        assert_eq!(parent.message_count(), 2);
        assert!(parent.files.contains_key("/tmp/new.rs"));
        assert_eq!(parent.todos.len(), 1);
    }

    #[test]
    fn test_todo_item_serde() {
        let item = TodoItem {
            content: "write tests".into(),
            status: TodoStatus::InProgress,
            active_form: "Writing tests".into(),
        };
        let json = serde_json::to_string(&item).unwrap();
        let back: TodoItem = serde_json::from_str(&json).unwrap();
        assert_eq!(item, back);
    }

    #[test]
    fn test_file_data_serde() {
        let fd = FileData {
            content: "hello".into(),
            encoding: "utf-8".into(),
            modified_at: Some(Utc::now()),
        };
        let json = serde_json::to_string(&fd).unwrap();
        let back: FileData = serde_json::from_str(&json).unwrap();
        assert_eq!(fd.content, back.content);
    }

    #[test]
    fn test_skill_metadata_serde() {
        let sm = SkillMetadata {
            name: "deploy".into(),
            description: "Deploy the app".into(),
            parameters: serde_json::json!({"target": "string"}),
        };
        let json = serde_json::to_string(&sm).unwrap();
        let back: SkillMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(sm, back);
    }

    #[test]
    fn test_debug_format() {
        let state = AgentState::new();
        let dbg = format!("{:?}", state);
        assert!(dbg.contains("AgentState"));
        assert!(dbg.contains("messages"));
    }
}
