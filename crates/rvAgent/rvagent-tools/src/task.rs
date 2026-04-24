//! `task` tool — spawns subagent tasks.

use crate::{Tool, ToolResult, ToolRuntime};
use async_trait::async_trait;

/// Spawns a subagent task with a description and prompt.
pub struct TaskTool;

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Spawn a subagent task. The task runs in a child agent with its own context."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short description of the task for tracking"
                },
                "prompt": {
                    "type": "string",
                    "description": "Full prompt/instructions for the subagent"
                }
            },
            "required": ["description", "prompt"]
        })
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        let description = match args.get("description").and_then(|v| v.as_str()) {
            Some(d) => d,
            None => return ToolResult::Text("Error: description is required".to_string()),
        };
        let prompt = match args.get("prompt").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::Text("Error: prompt is required".to_string()),
        };

        let task_id = runtime.tool_call_id.as_deref().unwrap_or("task_unknown");

        // In a real implementation, this would spawn a subagent via the orchestrator.
        // For now, return a confirmation with the task metadata.
        ToolResult::Text(format!(
            "Task spawned: id={}, description='{}', prompt_len={}",
            task_id,
            description,
            prompt.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_common::*;

    #[test]
    fn test_task_name() {
        assert_eq!(TaskTool.name(), "task");
    }

    #[test]
    fn test_task_schema() {
        let schema = TaskTool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("description")));
        assert!(required.contains(&serde_json::json!("prompt")));
    }

    #[test]
    fn test_task_invoke_success() {
        let mut runtime = mock_runtime();
        runtime.tool_call_id = Some("tc_42".to_string());
        let result = TaskTool.invoke(
            serde_json::json!({
                "description": "Run tests",
                "prompt": "Execute all unit tests and report results"
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("Task spawned"));
                assert!(s.contains("tc_42"));
                assert!(s.contains("Run tests"));
            }
            _ => panic!("expected Text result"),
        }
    }

    #[test]
    fn test_task_invoke_no_tool_call_id() {
        let runtime = mock_runtime();
        let result = TaskTool.invoke(
            serde_json::json!({
                "description": "Refactor module",
                "prompt": "Refactor the auth module"
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("task_unknown"));
            }
            _ => panic!("expected Text result"),
        }
    }

    #[test]
    fn test_task_missing_description() {
        let runtime = mock_runtime();
        let result = TaskTool.invoke(serde_json::json!({"prompt": "do stuff"}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("description is required")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_task_missing_prompt() {
        let runtime = mock_runtime();
        let result = TaskTool.invoke(serde_json::json!({"description": "task"}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("prompt is required")),
            _ => panic!("expected error"),
        }
    }
}
