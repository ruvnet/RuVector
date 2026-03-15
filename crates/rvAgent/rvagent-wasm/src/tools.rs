//! WASM-compatible tool subset.
//!
//! These tools operate on the in-memory `WasmStateBackend` rather than
//! a real filesystem. Tools that require OS-level access (execute, glob, grep)
//! are intentionally omitted as they are unavailable in the browser sandbox.

use serde::{Deserialize, Serialize};

use crate::backends::WasmStateBackend;

// ---------------------------------------------------------------------------
// Tool request / response types
// ---------------------------------------------------------------------------

/// A tool invocation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case")]
pub enum ToolRequest {
    ReadFile {
        path: String,
    },
    WriteFile {
        path: String,
        content: String,
    },
    EditFile {
        path: String,
        old_string: String,
        new_string: String,
    },
    WriteTodos {
        todos: Vec<TodoItem>,
    },
    ListFiles,
}

/// A single todo item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub content: String,
    pub status: TodoStatus,
}

/// Todo item status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// Result from a tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the tool invocation succeeded.
    pub success: bool,
    /// Output content (file content, confirmation message, error description).
    pub output: String,
}

impl ToolResult {
    fn ok(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
        }
    }

    fn err(output: impl Into<String>) -> Self {
        Self {
            success: false,
            output: output.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool executor
// ---------------------------------------------------------------------------

/// Executes tools against a `WasmStateBackend` and a todo list.
pub struct WasmToolExecutor<'a> {
    backend: &'a mut WasmStateBackend,
    todos: &'a mut Vec<TodoItem>,
}

impl<'a> WasmToolExecutor<'a> {
    /// Create a new executor bound to the given backend and todo list.
    pub fn new(backend: &'a mut WasmStateBackend, todos: &'a mut Vec<TodoItem>) -> Self {
        Self { backend, todos }
    }

    /// Dispatch and execute a tool request, returning the result.
    pub fn execute(&mut self, request: &ToolRequest) -> ToolResult {
        match request {
            ToolRequest::ReadFile { path } => self.read_file(path),
            ToolRequest::WriteFile { path, content } => self.write_file(path, content),
            ToolRequest::EditFile {
                path,
                old_string,
                new_string,
            } => self.edit_file(path, old_string, new_string),
            ToolRequest::WriteTodos { todos } => self.write_todos(todos),
            ToolRequest::ListFiles => self.list_files(),
        }
    }

    /// Read a file from the virtual filesystem.
    fn read_file(&self, path: &str) -> ToolResult {
        match self.backend.read_file(path) {
            Ok(content) => ToolResult::ok(content),
            Err(e) => ToolResult::err(e.to_string()),
        }
    }

    /// Write a file to the virtual filesystem.
    fn write_file(&mut self, path: &str, content: &str) -> ToolResult {
        match self.backend.write_file(path, content) {
            Ok(()) => ToolResult::ok(format!("wrote {} bytes to {}", content.len(), path)),
            Err(e) => ToolResult::err(e.to_string()),
        }
    }

    /// Apply a string replacement edit to a file.
    fn edit_file(&mut self, path: &str, old: &str, new: &str) -> ToolResult {
        match self.backend.edit_file(path, old, new) {
            Ok(()) => ToolResult::ok(format!("edited {}", path)),
            Err(e) => ToolResult::err(e.to_string()),
        }
    }

    /// Replace the entire todo list.
    fn write_todos(&mut self, todos: &[TodoItem]) -> ToolResult {
        self.todos.clear();
        self.todos.extend(todos.iter().cloned());
        ToolResult::ok(format!("wrote {} todos", self.todos.len()))
    }

    /// List all files in the virtual filesystem.
    fn list_files(&self) -> ToolResult {
        let files = self.backend.list_files();
        match serde_json::to_string(&files) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::err(e.to_string()),
        }
    }
}

/// Parse a JSON string into a `ToolRequest`.
pub fn parse_tool_request(json: &str) -> Result<ToolRequest, String> {
    serde_json::from_str(json).map_err(|e| format!("invalid tool request: {}", e))
}

/// Serialize a `ToolResult` to JSON.
pub fn tool_result_to_json(result: &ToolResult) -> Result<String, String> {
    serde_json::to_string(result).map_err(|e| format!("serialization error: {}", e))
}

/// Returns the list of available tool names in this WASM environment.
pub fn available_tools() -> Vec<&'static str> {
    vec![
        "read_file",
        "write_file",
        "edit_file",
        "write_todos",
        "list_files",
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_executor() -> (WasmStateBackend, Vec<TodoItem>) {
        (WasmStateBackend::new(), Vec::new())
    }

    #[test]
    fn test_read_write_file() {
        let (mut backend, mut todos) = make_executor();
        let mut exec = WasmToolExecutor::new(&mut backend, &mut todos);

        let write_result = exec.execute(&ToolRequest::WriteFile {
            path: "hello.txt".into(),
            content: "world".into(),
        });
        assert!(write_result.success);

        let read_result = exec.execute(&ToolRequest::ReadFile {
            path: "hello.txt".into(),
        });
        assert!(read_result.success);
        assert_eq!(read_result.output, "world");
    }

    #[test]
    fn test_read_nonexistent() {
        let (mut backend, mut todos) = make_executor();
        let exec = WasmToolExecutor::new(&mut backend, &mut todos);
        let result = exec.read_file("nope.txt");
        assert!(!result.success);
        assert!(result.output.contains("not found"));
    }

    #[test]
    fn test_edit_file() {
        let (mut backend, mut todos) = make_executor();
        let mut exec = WasmToolExecutor::new(&mut backend, &mut todos);

        exec.execute(&ToolRequest::WriteFile {
            path: "code.rs".into(),
            content: "let x = 1;".into(),
        });

        let result = exec.execute(&ToolRequest::EditFile {
            path: "code.rs".into(),
            old_string: "let x = 1".into(),
            new_string: "let x = 42".into(),
        });
        assert!(result.success);

        let read = exec.execute(&ToolRequest::ReadFile {
            path: "code.rs".into(),
        });
        assert_eq!(read.output, "let x = 42;");
    }

    #[test]
    fn test_write_todos() {
        let (mut backend, mut todos) = make_executor();
        let mut exec = WasmToolExecutor::new(&mut backend, &mut todos);

        let items = vec![
            TodoItem {
                content: "implement feature".into(),
                status: TodoStatus::Pending,
            },
            TodoItem {
                content: "write tests".into(),
                status: TodoStatus::InProgress,
            },
        ];

        let result = exec.execute(&ToolRequest::WriteTodos { todos: items });
        assert!(result.success);
        assert!(result.output.contains("2 todos"));
    }

    #[test]
    fn test_list_files() {
        let (mut backend, mut todos) = make_executor();
        let mut exec = WasmToolExecutor::new(&mut backend, &mut todos);

        exec.execute(&ToolRequest::WriteFile {
            path: "a.txt".into(),
            content: "a".into(),
        });
        exec.execute(&ToolRequest::WriteFile {
            path: "b.txt".into(),
            content: "b".into(),
        });

        let result = exec.execute(&ToolRequest::ListFiles);
        assert!(result.success);
        assert!(result.output.contains("a.txt"));
        assert!(result.output.contains("b.txt"));
    }

    #[test]
    fn test_parse_tool_request() {
        let json = r#"{"tool": "read_file", "path": "test.rs"}"#;
        let req = parse_tool_request(json).unwrap();
        assert!(matches!(req, ToolRequest::ReadFile { path } if path == "test.rs"));
    }

    #[test]
    fn test_parse_tool_request_invalid() {
        let result = parse_tool_request("{bad json}");
        assert!(result.is_err());
    }

    #[test]
    fn test_available_tools() {
        let tools = available_tools();
        assert!(tools.contains(&"read_file"));
        assert!(tools.contains(&"write_file"));
        assert!(tools.contains(&"edit_file"));
        assert!(tools.contains(&"write_todos"));
        assert!(tools.contains(&"list_files"));
        // These should NOT be available in WASM:
        assert!(!tools.contains(&"execute"));
        assert!(!tools.contains(&"glob"));
        assert!(!tools.contains(&"grep"));
    }

    #[test]
    fn test_todo_status_serde() {
        let item = TodoItem {
            content: "task".into(),
            status: TodoStatus::Completed,
        };
        let json = serde_json::to_string(&item).unwrap();
        let back: TodoItem = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, TodoStatus::Completed);
    }
}
