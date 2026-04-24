//! `edit_file` tool — exact string replacement in files.

use crate::{StateUpdate, Tool, ToolResult, ToolRuntime};
use async_trait::async_trait;

/// Performs exact string replacement in a file.
///
/// Returns an error if `old_string` is not unique (unless `replace_all` is true).
pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace exact string occurrences in a file. Fails if old_string \
         is not unique unless replace_all is true."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences instead of requiring uniqueness",
                    "default": false
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        })
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::Text("Error: file_path is required".to_string()),
        };
        let old_string = match args.get("old_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::Text("Error: old_string is required".to_string()),
        };
        let new_string = match args.get("new_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::Text("Error: new_string is required".to_string()),
        };
        let replace_all = args
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let result = runtime
            .backend
            .edit(file_path, old_string, new_string, replace_all);

        match result.error {
            Some(err) => ToolResult::Text(err),
            None => {
                if let Some(files_update) = result.files_update {
                    ToolResult::Command(StateUpdate::FilesUpdate(files_update))
                } else {
                    let occurrences = result.occurrences.unwrap_or(1);
                    ToolResult::Text(format!(
                        "Successfully edited {} ({} occurrence{})",
                        file_path,
                        occurrences,
                        if occurrences != 1 { "s" } else { "" }
                    ))
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
    fn test_edit_file_name() {
        assert_eq!(EditFileTool.name(), "edit_file");
    }

    #[test]
    fn test_edit_file_schema() {
        let schema = EditFileTool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("file_path")));
        assert!(required.contains(&serde_json::json!("old_string")));
        assert!(required.contains(&serde_json::json!("new_string")));
    }

    #[test]
    fn test_edit_file_unique_replacement() {
        let runtime = mock_runtime();
        let result = EditFileTool.invoke(
            serde_json::json!({
                "file_path": "/test.txt",
                "old_string": "hello",
                "new_string": "hi"
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("Successfully edited"));
                assert!(s.contains("1 occurrence"));
            }
            _ => panic!("expected success text"),
        }
    }

    #[test]
    fn test_edit_file_not_unique_error() {
        let runtime = mock_runtime();
        let result = EditFileTool.invoke(
            serde_json::json!({
                "file_path": "/multi.txt",
                "old_string": "aaa",
                "new_string": "zzz"
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("not unique"));
                assert!(s.contains("2 occurrences"));
            }
            _ => panic!("expected uniqueness error"),
        }
    }

    #[test]
    fn test_edit_file_replace_all() {
        let runtime = mock_runtime();
        let result = EditFileTool.invoke(
            serde_json::json!({
                "file_path": "/multi.txt",
                "old_string": "aaa",
                "new_string": "zzz",
                "replace_all": true
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("Successfully edited"));
                assert!(s.contains("2 occurrence"));
            }
            _ => panic!("expected success text"),
        }
    }

    #[test]
    fn test_edit_file_not_found() {
        let runtime = mock_runtime();
        let result = EditFileTool.invoke(
            serde_json::json!({
                "file_path": "/nonexistent.txt",
                "old_string": "x",
                "new_string": "y"
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("not found")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_edit_file_old_string_not_found() {
        let runtime = mock_runtime();
        let result = EditFileTool.invoke(
            serde_json::json!({
                "file_path": "/test.txt",
                "old_string": "nonexistent_string",
                "new_string": "y"
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("not found")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_edit_file_missing_params() {
        let runtime = mock_runtime();
        let result = EditFileTool.invoke(serde_json::json!({}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("required")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_edit_file_permission_denied() {
        let runtime = mock_runtime_with_error();
        let result = EditFileTool.invoke(
            serde_json::json!({
                "file_path": "/test.txt",
                "old_string": "x",
                "new_string": "y"
            }),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("Permission denied")),
            _ => panic!("expected error"),
        }
    }
}
