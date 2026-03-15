//! Integration tests for the `grep` tool.

use rvagent_tools::{
    Backend, BackendRef, ExecuteResponse, FileInfo, GrepMatch, GrepTool,
    Tool, ToolResult, ToolRuntime, WriteResult,
};
use std::sync::Arc;

struct GrepMockBackend {
    matches: Vec<GrepMatch>,
}

impl Backend for GrepMockBackend {
    fn ls_info(&self, _: &str) -> Result<Vec<FileInfo>, String> { Ok(vec![]) }
    fn read(&self, _: &str, _: usize, _: usize) -> Result<String, String> { Ok(String::new()) }
    fn write(&self, _: &str, _: &str) -> WriteResult { WriteResult::default() }
    fn edit(&self, _: &str, _: &str, _: &str, _: bool) -> WriteResult { WriteResult::default() }
    fn glob_info(&self, _: &str, _: &str) -> Result<Vec<String>, String> { Ok(vec![]) }
    fn grep_raw(&self, pattern: &str, _path: Option<&str>, include: Option<&str>) -> Result<Vec<GrepMatch>, String> {
        let filtered: Vec<GrepMatch> = self.matches.iter()
            .filter(|m| m.text.contains(pattern))
            .filter(|m| {
                if let Some(inc) = include {
                    let ext = inc.trim_start_matches('*');
                    m.file.ends_with(ext)
                } else {
                    true
                }
            })
            .cloned()
            .collect();
        Ok(filtered)
    }
    fn execute(&self, _: &str, _: u32) -> Result<ExecuteResponse, String> {
        Ok(ExecuteResponse { output: String::new(), exit_code: 0 })
    }
}

fn grep_runtime() -> ToolRuntime {
    let backend = Arc::new(GrepMockBackend {
        matches: vec![
            GrepMatch { file: "src.rs".into(), line_number: 2, text: "    println!(\"hello\");".into() },
            GrepMatch { file: "notes.txt".into(), line_number: 1, text: "hello world notes".into() },
            GrepMatch { file: "code.rs".into(), line_number: 5, text: "let target = 42;".into() },
            GrepMatch { file: "notes.txt".into(), line_number: 3, text: "target reached".into() },
        ],
    }) as BackendRef;
    ToolRuntime::new(backend)
}

#[test]
fn test_grep_literal_match() {
    let runtime = grep_runtime();
    let result = GrepTool.invoke(serde_json::json!({"pattern": "println"}), &runtime);
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("println"), "should find 'println'");
            assert!(s.contains("src.rs"), "should reference the file");
            assert!(s.contains(":2:"), "match should be on line 2");
        }
        _ => panic!("expected Text result from grep"),
    }
}

#[test]
fn test_grep_no_results() {
    let runtime = grep_runtime();
    let result = GrepTool.invoke(
        serde_json::json!({"pattern": "nonexistent_pattern_xyz_123"}),
        &runtime,
    );
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("No matches"), "should report no matches, got: {}", s);
        }
        _ => panic!("expected Text from grep"),
    }
}

#[test]
fn test_grep_with_include_filter() {
    let runtime = grep_runtime();
    let result = GrepTool.invoke(
        serde_json::json!({"pattern": "target", "include": "*.rs"}),
        &runtime,
    );
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("code.rs"), "should find match in code.rs");
            assert!(!s.contains("notes.txt"),
                "should NOT include notes.txt due to include filter, got: {}", s);
        }
        _ => panic!("expected Text result from grep"),
    }
}
