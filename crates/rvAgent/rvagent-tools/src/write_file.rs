//! `write_file` tool — creates or overwrites files.

use crate::{StateUpdate, Tool, ToolResult, ToolRuntime};
use async_trait::async_trait;

/// Creates a new file. Returns error if file exists (no force flag).
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Create a new file with the given content. Errors if the file already exists."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path for the file to create"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        })
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::Text("Error: file_path is required".to_string()),
        };
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::Text("Error: content is required".to_string()),
        };

        let result = runtime.backend.write(file_path, content);

        match result.error {
            Some(err) => ToolResult::Text(err),
            None => {
                if let Some(files_update) = result.files_update {
                    ToolResult::Command(StateUpdate::FilesUpdate(files_update))
                } else {
                    ToolResult::Text(format!("Successfully wrote to {}", file_path))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_common::*;

    #[test]
    fn test_write_file_name() {
        assert_eq!(WriteFileTool.name(), "write_file");
    }

    #[test]
    fn test_write_file_schema() {
        let schema = WriteFileTool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("file_path")));
        assert!(required.contains(&serde_json::json!("content")));
    }

    #[test]
    fn test_write_file_success() {
        let runtime = mock_runtime();
        let result = WriteFileTool.invoke(
            serde_json::json!({"file_path": "/new_file.txt", "content": "hello"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("Successfully wrote")),
            _ => panic!("expected success text"),
        }
    }

    #[test]
    fn test_write_file_already_exists() {
        let runtime = mock_runtime();
        let result = WriteFileTool.invoke(
            serde_json::json!({"file_path": "/test.txt", "content": "overwrite"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("already exists")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_write_file_missing_path() {
        let runtime = mock_runtime();
        let result = WriteFileTool.invoke(
            serde_json::json!({"content": "hello"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("file_path is required")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_write_file_missing_content() {
        let runtime = mock_runtime();
        let result = WriteFileTool.invoke(
            serde_json::json!({"file_path": "/foo.txt"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("content is required")),
            _ => panic!("expected error"),
        }
    }
}
