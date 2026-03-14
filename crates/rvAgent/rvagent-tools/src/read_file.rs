//! `read_file` tool — reads file content with line numbers.

use crate::{
    format_content_with_line_numbers, is_image_file, Tool, ToolResult, ToolRuntime,
    DEFAULT_READ_LIMIT, DEFAULT_READ_OFFSET, EMPTY_CONTENT_WARNING,
};
use async_trait::async_trait;

/// Reads a file with optional offset/limit and formats with line numbers.
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read file contents with line numbers. Supports offset and limit parameters."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based)",
                    "default": 0
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "default": 2000
                }
            },
            "required": ["file_path"]
        })
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::Text("Error: file_path is required".to_string()),
        };

        // Image files get a placeholder response.
        if is_image_file(file_path) {
            return ToolResult::Text(format!(
                "[Image file: {}. Image content cannot be displayed as text.]",
                file_path
            ));
        }

        let offset = args
            .get("offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_READ_OFFSET as u64) as usize;
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_READ_LIMIT as u64) as usize;

        match runtime.backend.read(file_path, offset, limit) {
            Ok(content) => {
                if content.is_empty() {
                    return ToolResult::Text(EMPTY_CONTENT_WARNING.to_string());
                }
                let start_line = offset + 1;
                let formatted = format_content_with_line_numbers(&content, start_line);
                ToolResult::Text(formatted)
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
    fn test_read_file_name() {
        assert_eq!(ReadFileTool.name(), "read_file");
    }

    #[test]
    fn test_read_file_schema() {
        let schema = ReadFileTool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("file_path")));
    }

    #[test]
    fn test_read_file_success() {
        let runtime = mock_runtime();
        let result = ReadFileTool.invoke(
            serde_json::json!({"file_path": "/test.txt"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("hello"));
                assert!(s.contains("world"));
                // Should have line numbers
                assert!(s.contains("1\t"));
            }
            _ => panic!("expected Text result"),
        }
    }

    #[test]
    fn test_read_file_with_offset() {
        let runtime = mock_runtime();
        let result = ReadFileTool.invoke(
            serde_json::json!({"file_path": "/test.txt", "offset": 1}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("world"));
                // Should start at line 2
                assert!(s.contains("2\t"));
            }
            _ => panic!("expected Text result"),
        }
    }

    #[test]
    fn test_read_file_not_found() {
        let runtime = mock_runtime();
        let result = ReadFileTool.invoke(
            serde_json::json!({"file_path": "/nonexistent.txt"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("Error")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_read_file_missing_path() {
        let runtime = mock_runtime();
        let result = ReadFileTool.invoke(serde_json::json!({}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("file_path is required")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_read_image_file() {
        let runtime = mock_runtime();
        let result = ReadFileTool.invoke(
            serde_json::json!({"file_path": "/photo.png"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("Image file")),
            _ => panic!("expected image message"),
        }
    }

    #[test]
    fn test_read_empty_file() {
        let runtime = mock_runtime_with_empty_file();
        let result = ReadFileTool.invoke(
            serde_json::json!({"file_path": "/empty.txt"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("empty contents")),
            _ => panic!("expected empty warning"),
        }
    }
}
