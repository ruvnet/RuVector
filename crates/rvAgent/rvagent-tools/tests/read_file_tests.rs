//! Integration tests for the `read_file` tool.

use rvagent_tools::{
    Backend, BackendRef, ExecuteResponse, FileInfo, GrepMatch, ReadFileTool, Tool, ToolResult,
    ToolRuntime, WriteResult,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

struct ReadMockBackend {
    files: Mutex<HashMap<String, String>>,
}

impl ReadMockBackend {
    fn new(files: HashMap<String, String>) -> Self {
        Self {
            files: Mutex::new(files),
        }
    }
}

impl Backend for ReadMockBackend {
    fn ls_info(&self, _: &str) -> Result<Vec<FileInfo>, String> {
        Ok(vec![])
    }
    fn read(&self, path: &str, offset: usize, limit: usize) -> Result<String, String> {
        let files = self.files.lock().unwrap();
        match files.get(path) {
            Some(content) => {
                if content.is_empty() {
                    return Ok(String::new());
                }
                let lines: Vec<&str> = content.lines().collect();
                let start = offset.min(lines.len());
                let end = (start + limit).min(lines.len());
                Ok(lines[start..end].join("\n"))
            }
            None => Err(format!("File not found: {}", path)),
        }
    }
    fn write(&self, _: &str, _: &str) -> WriteResult {
        WriteResult::default()
    }
    fn edit(&self, _: &str, _: &str, _: &str, _: bool) -> WriteResult {
        WriteResult::default()
    }
    fn glob_info(&self, _: &str, _: &str) -> Result<Vec<String>, String> {
        Ok(vec![])
    }
    fn grep_raw(
        &self,
        _: &str,
        _: Option<&str>,
        _: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        Ok(vec![])
    }
    fn execute(&self, _: &str, _: u32) -> Result<ExecuteResponse, String> {
        Ok(ExecuteResponse {
            output: String::new(),
            exit_code: 0,
        })
    }
}

#[test]
fn test_read_full_file() {
    let mut files = HashMap::new();
    files.insert(
        "/sample.txt".into(),
        "line one\nline two\nline three".into(),
    );
    let runtime = ToolRuntime::new(Arc::new(ReadMockBackend::new(files)) as BackendRef);

    let result = ReadFileTool.invoke(serde_json::json!({"file_path": "/sample.txt"}), &runtime);
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("line one"), "should contain 'line one'");
            assert!(s.contains("line two"), "should contain 'line two'");
            assert!(s.contains("line three"), "should contain 'line three'");
            // Should have line numbers
            assert!(s.contains("1\t"), "should have line number 1");
        }
        _ => panic!("expected Text result from read_file"),
    }
}

#[test]
fn test_read_with_offset_limit() {
    let mut files = HashMap::new();
    let lines: Vec<String> = (1..=10).map(|i| format!("line {}", i)).collect();
    files.insert("/ten.txt".into(), lines.join("\n"));
    let runtime = ToolRuntime::new(Arc::new(ReadMockBackend::new(files)) as BackendRef);

    // offset=2, limit=3 should return lines 3, 4, 5
    let result = ReadFileTool.invoke(
        serde_json::json!({"file_path": "/ten.txt", "offset": 2, "limit": 3}),
        &runtime,
    );
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("line 3"), "should contain 'line 3'");
            assert!(s.contains("line 4"), "should contain 'line 4'");
            assert!(s.contains("line 5"), "should contain 'line 5'");
            assert!(!s.contains("line 1"), "should NOT contain 'line 1'");
            assert!(
                !s.contains("line 2\n"),
                "should NOT contain 'line 2' as its own line"
            );
        }
        _ => panic!("expected Text result from read_file"),
    }
}

#[test]
fn test_read_nonexistent_file() {
    let files = HashMap::new();
    let runtime = ToolRuntime::new(Arc::new(ReadMockBackend::new(files)) as BackendRef);

    let result = ReadFileTool.invoke(
        serde_json::json!({"file_path": "/does_not_exist.txt"}),
        &runtime,
    );
    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("Error") || s.contains("not found"),
                "nonexistent file should produce error, got: {}",
                s
            );
        }
        _ => panic!("expected Text error from read_file"),
    }
}
