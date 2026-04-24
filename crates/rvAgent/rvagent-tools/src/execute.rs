//! `execute` tool — shell command execution via backend.

use crate::{Tool, ToolResult, ToolRuntime, DEFAULT_EXECUTE_TIMEOUT};
use async_trait::async_trait;

/// Executes shell commands through the backend.
pub struct ExecuteTool;

#[async_trait]
impl Tool for ExecuteTool {
    fn name(&self) -> &str {
        "execute"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 120
                }
            },
            "required": ["command"]
        })
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        let command = match args.get("command").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::Text("Error: command is required".to_string()),
        };
        let timeout = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_EXECUTE_TIMEOUT as u64) as u32;

        match runtime.backend.execute(command, timeout) {
            Ok(response) => {
                let mut output = response.output;
                if response.exit_code != 0 {
                    output.push_str(&format!("\n[exit code: {}]", response.exit_code));
                }
                ToolResult::Text(output)
            }
            Err(e) => ToolResult::Text(format!("Error: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_common::*;

    #[test]
    fn test_execute_name() {
        assert_eq!(ExecuteTool.name(), "execute");
    }

    #[test]
    fn test_execute_schema() {
        let schema = ExecuteTool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("command")));
    }

    #[test]
    fn test_execute_invoke_success() {
        let runtime = mock_runtime();
        let result = ExecuteTool.invoke(serde_json::json!({"command": "echo hello"}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("mock output")),
            _ => panic!("expected Text result"),
        }
    }

    #[test]
    fn test_execute_missing_command() {
        let runtime = mock_runtime();
        let result = ExecuteTool.invoke(serde_json::json!({}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("command is required")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_execute_error() {
        let runtime = mock_runtime_with_error();
        let result = ExecuteTool.invoke(serde_json::json!({"command": "fail"}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("Error")),
            _ => panic!("expected error"),
        }
    }
}
