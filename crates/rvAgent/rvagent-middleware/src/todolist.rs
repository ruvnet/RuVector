//! TodoListMiddleware — injects todo state into messages and provides write_todos tool.

use async_trait::async_trait;
use serde_json;

use crate::{
    AgentState, AgentStateUpdate, Middleware, RunnableConfig, Runtime, TodoItem, TodoStatus, Tool,
};

/// Middleware that manages a todo list in agent state.
///
/// - `before_agent`: injects current todo state into messages
/// - `tools()`: returns the `write_todos` tool
pub struct TodoListMiddleware;

impl TodoListMiddleware {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TodoListMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for TodoListMiddleware {
    fn name(&self) -> &str {
        "todolist"
    }

    fn before_agent(
        &self,
        state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if state.todos.is_empty() {
            return None;
        }

        // Format todo state for injection
        let todo_text = format_todos(&state.todos);

        // Store formatted todos in extensions for system prompt injection
        let mut update = AgentStateUpdate::default();
        update
            .extensions
            .insert("todo_context".into(), serde_json::Value::String(todo_text));
        Some(update)
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(WriteTodosTool)]
    }
}

/// Format todo items for display in conversation.
fn format_todos(todos: &[TodoItem]) -> String {
    let mut out = String::from("<todos>\n");
    for todo in todos {
        let status_str = match todo.status {
            TodoStatus::Pending => "pending",
            TodoStatus::InProgress => "in_progress",
            TodoStatus::Completed => "completed",
        };
        out.push_str(&format!(
            "  <todo id=\"{}\" status=\"{}\">{}</todo>\n",
            todo.id, status_str, todo.content
        ));
    }
    out.push_str("</todos>");
    out
}

/// Tool for writing/updating todo items.
struct WriteTodosTool;

impl Tool for WriteTodosTool {
    fn name(&self) -> &str {
        "write_todos"
    }

    fn description(&self) -> &str {
        "Create or update the todo list. Provide a complete list of todo items with id, content, and status (pending, in_progress, completed)."
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
                            "id": { "type": "string" },
                            "content": { "type": "string" },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            }
                        },
                        "required": ["id", "content", "status"]
                    }
                }
            },
            "required": ["todos"]
        })
    }

    fn invoke(&self, args: serde_json::Value) -> Result<String, String> {
        let todos = args
            .get("todos")
            .and_then(|v| v.as_array())
            .ok_or("missing 'todos' array")?;

        let count = todos.len();
        // Validate each item
        for item in todos {
            let _id = item
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or("each todo must have an 'id' string")?;
            let _content = item
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or("each todo must have a 'content' string")?;
            let status = item
                .get("status")
                .and_then(|v| v.as_str())
                .ok_or("each todo must have a 'status' string")?;
            match status {
                "pending" | "in_progress" | "completed" => {}
                other => return Err(format!("invalid status: {}", other)),
            }
        }

        Ok(format!("Updated {} todo items", count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_todos_empty() {
        let result = format_todos(&[]);
        assert_eq!(result, "<todos>\n</todos>");
    }

    #[test]
    fn test_format_todos() {
        let todos = vec![
            TodoItem {
                id: "1".into(),
                content: "Do something".into(),
                status: TodoStatus::Pending,
            },
            TodoItem {
                id: "2".into(),
                content: "Done".into(),
                status: TodoStatus::Completed,
            },
        ];
        let result = format_todos(&todos);
        assert!(result.contains("status=\"pending\""));
        assert!(result.contains("status=\"completed\""));
        assert!(result.contains("Do something"));
    }

    #[test]
    fn test_before_agent_empty_todos() {
        let mw = TodoListMiddleware::new();
        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &config).is_none());
    }

    #[test]
    fn test_before_agent_with_todos() {
        let mw = TodoListMiddleware::new();
        let mut state = AgentState::default();
        state.todos.push(TodoItem {
            id: "1".into(),
            content: "Test task".into(),
            status: TodoStatus::InProgress,
        });
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        let update = mw.before_agent(&state, &runtime, &config);
        assert!(update.is_some());
        let update = update.unwrap();
        assert!(update.extensions.contains_key("todo_context"));
    }

    #[test]
    fn test_write_todos_tool_name() {
        let tool = WriteTodosTool;
        assert_eq!(tool.name(), "write_todos");
    }

    #[test]
    fn test_write_todos_invoke_valid() {
        let tool = WriteTodosTool;
        let args = serde_json::json!({
            "todos": [
                {"id": "1", "content": "task 1", "status": "pending"},
                {"id": "2", "content": "task 2", "status": "completed"}
            ]
        });
        let result = tool.invoke(args);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("2 todo items"));
    }

    #[test]
    fn test_write_todos_invoke_invalid_status() {
        let tool = WriteTodosTool;
        let args = serde_json::json!({
            "todos": [{"id": "1", "content": "task", "status": "invalid"}]
        });
        let result = tool.invoke(args);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_todos_invoke_missing_field() {
        let tool = WriteTodosTool;
        let args = serde_json::json!({
            "todos": [{"id": "1"}]
        });
        let result = tool.invoke(args);
        assert!(result.is_err());
    }

    #[test]
    fn test_middleware_tools() {
        let mw = TodoListMiddleware::new();
        let tools = mw.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "write_todos");
    }

    #[test]
    fn test_middleware_name() {
        let mw = TodoListMiddleware::new();
        assert_eq!(mw.name(), "todolist");
    }
}
