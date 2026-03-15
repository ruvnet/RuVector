//! Integration tests for the `write_todos` tool.

use rvagent_tools::{
    Backend, BackendRef, ExecuteResponse, FileInfo, GrepMatch, StateUpdate,
    Tool, ToolResult, ToolRuntime, WriteTodosTool, WriteResult,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Mock backend (minimal — write_todos does not use filesystem ops)
// ---------------------------------------------------------------------------

struct TodoMockBackend;

impl Backend for TodoMockBackend {
    fn ls_info(&self, _: &str) -> Result<Vec<FileInfo>, String> {
        Ok(vec![])
    }
    fn read(&self, _: &str, _: usize, _: usize) -> Result<String, String> {
        Ok(String::new())
    }
    fn write(&self, _: &str, _: &str) -> WriteResult {
        WriteResult::default()
    }
    fn edit(&self, _: &str, _: &str, _: &str, _: bool) -> WriteResult {
        WriteResult::default()
    }
    fn glob_info(&self, _: &str, _: &str) -> Result<Vec<String>, String> {
        Ok(vec![])
    }
    fn grep_raw(
        &self,
        _: &str,
        _: Option<&str>,
        _: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        Ok(vec![])
    }
    fn execute(&self, _: &str, _: u32) -> Result<ExecuteResponse, String> {
        Ok(ExecuteResponse {
            output: String::new(),
            exit_code: 0,
        })
    }
}

fn todo_runtime() -> ToolRuntime {
    ToolRuntime::new(Arc::new(TodoMockBackend) as BackendRef)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_write_todo_list_with_multiple_items() {
    let runtime = todo_runtime();
    let result = WriteTodosTool.invoke(
        serde_json::json!({
            "todos": [
                {"content": "Build feature", "status": "completed", "activeForm": "Building feature"},
                {"content": "Write tests", "status": "in_progress", "activeForm": "Writing tests"},
                {"content": "Deploy", "status": "pending", "activeForm": "Deploying"}
            ]
        }),
        &runtime,
    );

    match result {
        ToolResult::Command(StateUpdate::Todos(todos)) => {
            assert_eq!(todos.len(), 3);
            assert_eq!(todos[0].content, "Build feature");
            assert_eq!(todos[0].status, "completed");
            assert_eq!(todos[1].content, "Write tests");
            assert_eq!(todos[1].status, "in_progress");
            assert_eq!(todos[2].content, "Deploy");
            assert_eq!(todos[2].status, "pending");
        }
        _ => panic!("expected Command(Todos) result"),
    }
}

#[test]
fn test_update_todo_status_from_pending_to_completed() {
    let runtime = todo_runtime();

    // Simulate updating a todo list where status changes
    let result = WriteTodosTool.invoke(
        serde_json::json!({
            "todos": [
                {"content": "Task A", "status": "completed", "activeForm": "Doing A"},
                {"content": "Task B", "status": "in_progress", "activeForm": "Doing B"}
            ]
        }),
        &runtime,
    );

    match result {
        ToolResult::Command(StateUpdate::Todos(todos)) => {
            assert_eq!(todos[0].status, "completed");
            assert_eq!(todos[1].status, "in_progress");
        }
        _ => panic!("expected Command(Todos) result"),
    }
}

#[test]
fn test_empty_todo_list_handling() {
    let runtime = todo_runtime();
    let result = WriteTodosTool.invoke(
        serde_json::json!({"todos": []}),
        &runtime,
    );

    match result {
        ToolResult::Command(StateUpdate::Todos(todos)) => {
            assert!(todos.is_empty(), "empty todo list should be accepted");
        }
        _ => panic!("expected Command(Todos) with empty list"),
    }
}

#[test]
fn test_write_todos_rejects_multiple_in_progress() {
    let runtime = todo_runtime();
    let result = WriteTodosTool.invoke(
        serde_json::json!({
            "todos": [
                {"content": "A", "status": "in_progress", "activeForm": "Doing A"},
                {"content": "B", "status": "in_progress", "activeForm": "Doing B"}
            ]
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("at most 1"),
                "should reject multiple in_progress, got: {}",
                s
            );
        }
        _ => panic!("expected error Text"),
    }
}

#[test]
fn test_write_todos_rejects_invalid_status() {
    let runtime = todo_runtime();
    let result = WriteTodosTool.invoke(
        serde_json::json!({
            "todos": [
                {"content": "Task", "status": "blocked", "activeForm": "Blocked"}
            ]
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("invalid status"),
                "should reject invalid status, got: {}",
                s
            );
        }
        _ => panic!("expected error Text"),
    }
}

#[test]
fn test_write_todos_missing_todos_field() {
    let runtime = todo_runtime();
    let result = WriteTodosTool.invoke(serde_json::json!({}), &runtime);

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("todos is required"),
                "should report missing todos, got: {}",
                s
            );
        }
        _ => panic!("expected error Text"),
    }
}

#[test]
fn test_write_todos_missing_active_form_field() {
    let runtime = todo_runtime();
    let result = WriteTodosTool.invoke(
        serde_json::json!({
            "todos": [
                {"content": "Task", "status": "pending"}
            ]
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("invalid todos format"),
                "should reject missing activeForm, got: {}",
                s
            );
        }
        _ => panic!("expected error Text"),
    }
}

#[test]
fn test_write_todos_preserves_active_form_serde_rename() {
    let runtime = todo_runtime();
    let result = WriteTodosTool.invoke(
        serde_json::json!({
            "todos": [
                {"content": "Test", "status": "pending", "activeForm": "Testing"}
            ]
        }),
        &runtime,
    );

    match result {
        ToolResult::Command(StateUpdate::Todos(todos)) => {
            assert_eq!(todos[0].active_form, "Testing");
        }
        _ => panic!("expected Command(Todos) result"),
    }
}

#[test]
fn test_write_todos_all_completed() {
    let runtime = todo_runtime();
    let result = WriteTodosTool.invoke(
        serde_json::json!({
            "todos": [
                {"content": "A", "status": "completed", "activeForm": "Doing A"},
                {"content": "B", "status": "completed", "activeForm": "Doing B"},
                {"content": "C", "status": "completed", "activeForm": "Doing C"}
            ]
        }),
        &runtime,
    );

    match result {
        ToolResult::Command(StateUpdate::Todos(todos)) => {
            assert_eq!(todos.len(), 3);
            assert!(todos.iter().all(|t| t.status == "completed"));
        }
        _ => panic!("expected Command(Todos) result"),
    }
}
