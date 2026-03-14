//! `glob` tool — file pattern matching.

use crate::{Tool, ToolResult, ToolRuntime};
use async_trait::async_trait;

/// Matches files by glob pattern and returns sorted paths.
pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Returns sorted file paths."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g. '**/*.rs', 'src/*.txt')"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in",
                    "default": "."
                }
            },
            "required": ["pattern"]
        })
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::Text("Error: pattern is required".to_string()),
        };
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        match runtime.backend.glob_info(pattern, path) {
            Ok(matches) => {
                if matches.is_empty() {
                    ToolResult::Text(format!("No files matching pattern '{}'", pattern))
                } else {
                    let count = matches.len();
                    let mut output = matches.join("\n");
                    output.push_str(&format!("\n\n({} files)", count));
                    ToolResult::Text(output)
                }
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
    fn test_glob_name() {
        assert_eq!(GlobTool.name(), "glob");
    }

    #[test]
    fn test_glob_schema() {
        let schema = GlobTool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("pattern")));
    }

    #[test]
    fn test_glob_invoke_success() {
        let runtime = mock_runtime();
        let result = GlobTool.invoke(
            serde_json::json!({"pattern": "*.txt"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("test.txt") || s.contains("multi.txt"));
                assert!(s.contains("files)"));
            }
            _ => panic!("expected Text result"),
        }
    }

    #[test]
    fn test_glob_no_matches() {
        let runtime = mock_runtime();
        let result = GlobTool.invoke(
            serde_json::json!({"pattern": "*.xyz_no_match"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("No files matching")),
            _ => panic!("expected no matches text"),
        }
    }

    #[test]
    fn test_glob_missing_pattern() {
        let runtime = mock_runtime();
        let result = GlobTool.invoke(serde_json::json!({}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("pattern is required")),
            _ => panic!("expected error"),
        }
    }
}
