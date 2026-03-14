//! `ls` tool — lists directory contents with file metadata.

use crate::{Tool, ToolResult, ToolRuntime};
use async_trait::async_trait;

/// Lists directory contents formatted as a metadata table.
pub struct LsTool;

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List directory contents with file type, permissions, and size"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list",
                    "default": "/"
                }
            },
            "required": []
        })
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("/");

        match runtime.backend.ls_info(path) {
            Ok(infos) => {
                if infos.is_empty() {
                    return ToolResult::Text(format!("Directory '{}' is empty", path));
                }
                let mut output = String::with_capacity(infos.len() * 60);
                for info in &infos {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&format!(
                        "{}\t{}\t{}\t{}",
                        info.file_type, info.permissions, info.size, info.name
                    ));
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
    fn test_ls_name() {
        assert_eq!(LsTool.name(), "ls");
    }

    #[test]
    fn test_ls_description() {
        assert!(!LsTool.description().is_empty());
    }

    #[test]
    fn test_ls_schema() {
        let schema = LsTool.parameters_schema();
        assert!(schema["properties"]["path"].is_object());
    }

    #[test]
    fn test_ls_invoke_success() {
        let runtime = mock_runtime();
        let result = LsTool.invoke(serde_json::json!({"path": "/"}), &runtime);
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("test.txt"));
                assert!(s.contains("file"));
            }
            _ => panic!("expected Text result"),
        }
    }

    #[test]
    fn test_ls_invoke_default_path() {
        let runtime = mock_runtime();
        let result = LsTool.invoke(serde_json::json!({}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("test.txt")),
            _ => panic!("expected Text result"),
        }
    }

    #[test]
    fn test_ls_invoke_error() {
        let runtime = mock_runtime_with_error();
        let result = LsTool.invoke(serde_json::json!({"path": "/bad"}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("Error")),
            _ => panic!("expected Text error"),
        }
    }
}
