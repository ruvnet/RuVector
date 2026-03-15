//! `grep` tool — literal text search (NOT regex, per ADR-103 C13).

use crate::{Tool, ToolResult, ToolRuntime};
use async_trait::async_trait;

/// Searches for literal text patterns in files.
///
/// Uses fixed-string/literal mode (`rg -F` equivalent) per ADR-103 C13.
/// Regex mode is intentionally NOT supported to prevent ReDoS.
pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for literal text in files. Returns matching lines with file path and line number."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Literal text pattern to search for (NOT regex)"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in"
                },
                "include": {
                    "type": "string",
                    "description": "Glob filter for files to include (e.g. '*.rs')"
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
        let path = args.get("path").and_then(|v| v.as_str());
        let include = args.get("include").and_then(|v| v.as_str());

        match runtime.backend.grep_raw(pattern, path, include) {
            Ok(matches) => {
                if matches.is_empty() {
                    return ToolResult::Text(format!(
                        "No matches found for '{}'",
                        pattern
                    ));
                }
                let mut output = String::with_capacity(matches.len() * 80);
                for m in &matches {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    // Format: file:line:text (same as ripgrep output)
                    output.push_str(&format!(
                        "{}:{}:{}",
                        m.file, m.line_number, m.text
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
    fn test_grep_name() {
        assert_eq!(GrepTool.name(), "grep");
    }

    #[test]
    fn test_grep_schema() {
        let schema = GrepTool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("pattern")));
    }

    #[test]
    fn test_grep_invoke_success() {
        let runtime = mock_runtime();
        let result = GrepTool.invoke(
            serde_json::json!({"pattern": "hello"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => {
                assert!(s.contains("hello"));
                assert!(s.contains("test.txt"));
                assert!(s.contains(":1:"));
            }
            _ => panic!("expected Text result"),
        }
    }

    #[test]
    fn test_grep_no_matches() {
        let runtime = mock_runtime();
        let result = GrepTool.invoke(
            serde_json::json!({"pattern": "nonexistent_xyz"}),
            &runtime,
        );
        match result {
            ToolResult::Text(s) => assert!(s.contains("No matches")),
            _ => panic!("expected no matches text"),
        }
    }

    #[test]
    fn test_grep_missing_pattern() {
        let runtime = mock_runtime();
        let result = GrepTool.invoke(serde_json::json!({}), &runtime);
        match result {
            ToolResult::Text(s) => assert!(s.contains("pattern is required")),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn test_grep_literal_not_regex() {
        assert!(GrepTool.description().contains("literal"));
    }
}
