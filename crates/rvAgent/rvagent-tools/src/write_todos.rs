//! `write_todos` tool — manages todo list in agent state.

use crate::{StateUpdate, TodoItem, Tool, ToolResult, ToolRuntime};
use async_trait::async_trait;

/// Manages a structured todo list in agent state.
pub struct WriteTodosTool;

#[async_trait]
impl Tool for WriteTodosTool {
    fn name(&self) -> &str {
        "write_todos"
    }

    fn description(&self) -> &str {
        "Manage a structured todo list. Replaces the current todo list with the provided items."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "What needs to be done"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current status"
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Present continuous form (e.g. 'Running tests')"
                            }
                        },
                        "required": ["content", "status", "activeForm"]
                    },
                    "description": "The complete todo list"
                }
            },
            "required": ["todos"]
        })
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        let todos_value = match args.get("todos") {
            Some(v) => v.clone(),
            None => return ToolResult::Text("Error: todos is required".to_string()),
        };

        match serde_json::from_value::<Vec<TodoItem>>(todos_value) {
            Ok(todos) => {
                // Validate: at most one in_progress
                let in_progress_count = todos.iter().filter(|t| t.status == "in_progress").count();
                if in_progress_count > 1 {
                    return ToolResult::Text(format!(
                        "Error: at most 1 todo should be in_progress, found {}",
                        in_progress_count
                    ));
                }
                // Validate statuses
                for todo in &todos {
                    if !["pending", "in_progress", "completed"].contains(&todo.status.as_str()) {
                        return ToolResult::Text(format!(
                            "Error: invalid status '{}' for todo '{}'",
                            todo.status, todo.content
                        ));
                    }
                }
                let _ = runtime; // runtime available for future extensions
                ToolResult::Command(StateUpdate::Todos(todos))
            }
            Err(e) => ToolResult::Text(format!("Error: invalid todos format: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_common::*;

    #[test]
    fn test_write_todos_name() {
        assert_eq!(WriteTodosTool.name(), "write_todos");
    }

    #[test]
    fn test_write_todos_schema() {
        let schema = WriteTodosTool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("todos")));
    }

    #[test]
    fn test_write_todos_success() {
        let runtime = mock_runtime();
        let result = WriteTodosTool.invoke(
            serde_json::json!({
                "todos": [
                    {"content": "Fix bug", "status": "in_progress", "activeForm": "Fixing bug"},
                    {"content": "Write tests", "status": "pending", "activeForm": "Writing tests"}
                ]
            }),
            &runtime,
        );
        match result {
            ToolResult::Command(StateUpdate::Todos(todos)) => {
                assert_eq!(todos.len(), 2);
                assert_eq!(todos[0].content, "Fix bug");
                assert_eq!(todos[0].status, "in_progress");
                assert_eq!(todos[1].status, "pending");
            }
            _ => panic!("expected Command(Todos) result"),
        }
    }

    #[test]
    fn test_write_todos_empty_list() {
        let runtime = mock_runtime();
        let result = WriteTodosTool.invoke(serde_json::json!({"todos": []}), &runtime);
        match result {
            ToolResult::Command(StateUpdate::Todos(todos)) => {
                assert!(todos.is_empty());
            }
            _ => panic!("expected empty Todos"),
        }
    }

    #[test]
    fn test_write_todos_multiple_in_progress_error() {
        let runtime = mock_runtime();
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
            ToolResult::Text(s) => assert!(s.contains("at most 1")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_write_todos_invalid_status() {
        let runtime = mock_runtime();
        let result = WriteTodosTool.invoke(
            serde_json::json!({
                "todos": [
                    {"content": "A", "status": "invalid_status", "activeForm": "Doing A"}
                ]
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("invalid status")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_write_todos_missing_field() {
        let runtime = mock_runtime();
        let result = WriteTodosTool.invoke(
            serde_json::json!({
                "todos": [
                    {"content": "A", "status": "pending"}
                ]
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("invalid todos format")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_write_todos_missing_todos() {
        let runtime = mock_runtime();
        let result = WriteTodosTool.invoke(serde_json::json!({}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("todos is required")),
            _ => panic!("expected error"),
        }
    }
}
