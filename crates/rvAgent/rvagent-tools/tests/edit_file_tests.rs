//! Integration tests for the `edit_file` tool.

use rvagent_tools::{
    Backend, BackendRef, EditFileTool, ExecuteResponse, FileInfo, GrepMatch,
    Tool, ToolResult, ToolRuntime, WriteResult,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Backend that implements edit logic for testing.
struct EditMockBackend {
    files: Mutex<HashMap<String, String>>,
}

impl EditMockBackend {
    fn new(files: HashMap<String, String>) -> Self {
        Self { files: Mutex::new(files) }
    }
}

impl Backend for EditMockBackend {
    fn ls_info(&self, _: &str) -> Result<Vec<FileInfo>, String> { Ok(vec![]) }
    fn read(&self, _: &str, _: usize, _: usize) -> Result<String, String> { Ok(String::new()) }
    fn write(&self, _: &str, _: &str) -> WriteResult { WriteResult::default() }
    fn edit(&self, path: &str, old: &str, new: &str, replace_all: bool) -> WriteResult {
        let mut files = self.files.lock().unwrap();
        match files.get(path).cloned() {
            None => WriteResult {
                error: Some(format!("Error: file not found: {}", path)),
                ..Default::default()
            },
            Some(content) => {
                let count = content.matches(old).count();
                if count == 0 {
                    return WriteResult {
                        error: Some(format!("Error: old_string not found in {}", path)),
                        ..Default::default()
                    };
                }
                if count > 1 && !replace_all {
                    return WriteResult {
                        error: Some(format!(
                            "Error: old_string found {} times in {}. Use replace_all=true.",
                            count, path
                        )),
                        ..Default::default()
                    };
                }
                let new_content = if replace_all {
                    content.replace(old, new)
                } else {
                    content.replacen(old, new, 1)
                };
                files.insert(path.to_string(), new_content);
                WriteResult {
                    occurrences: Some(if replace_all { count } else { 1 }),
                    ..Default::default()
                }
            }
        }
    }
    fn glob_info(&self, _: &str, _: &str) -> Result<Vec<String>, String> { Ok(vec![]) }
    fn grep_raw(&self, _: &str, _: Option<&str>, _: Option<&str>) -> Result<Vec<GrepMatch>, String> { Ok(vec![]) }
    fn execute(&self, _: &str, _: u32) -> Result<ExecuteResponse, String> { Ok(ExecuteResponse { output: String::new(), exit_code: 0 }) }
}

#[test]
fn test_edit_unique_match() {
    let mut files = HashMap::new();
    files.insert("/code.rs".into(), "fn main() { println!(\"hello\"); }".into());
    let runtime = ToolRuntime::new(Arc::new(EditMockBackend::new(files)) as BackendRef);

    let result = EditFileTool.invoke(
        serde_json::json!({
            "file_path": "/code.rs",
            "old_string": "hello",
            "new_string": "world"
        }),
        &runtime,
    );
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("Successfully edited") || s.contains("1 occurrence"),
                "should report success, got: {}", s);
        }
        _ => panic!("expected Text result from edit_file"),
    }
}

#[test]
fn test_edit_non_unique_error() {
    let mut files = HashMap::new();
    files.insert("/dup.txt".into(), "foo bar\nbaz foo".into());
    let runtime = ToolRuntime::new(Arc::new(EditMockBackend::new(files)) as BackendRef);

    let result = EditFileTool.invoke(
        serde_json::json!({
            "file_path": "/dup.txt",
            "old_string": "foo",
            "new_string": "qux"
        }),
        &runtime,
    );
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("2 times") || s.contains("found 2") || s.contains("replace_all"),
                "should report non-unique, got: {}", s);
        }
        _ => panic!("expected Text error"),
    }
}

#[test]
fn test_edit_replace_all() {
    let mut files = HashMap::new();
    files.insert("/multi.txt".into(), "aaa bbb aaa ccc aaa".into());
    let runtime = ToolRuntime::new(Arc::new(EditMockBackend::new(files)) as BackendRef);

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
            assert!(s.contains("Successfully edited") || s.contains("3 occurrence"),
                "should report success with 3 occurrences, got: {}", s);
        }
        _ => panic!("expected Text result"),
    }
}

#[test]
fn test_edit_no_match() {
    let mut files = HashMap::new();
    files.insert("/stable.txt".into(), "nothing to change".into());
    let runtime = ToolRuntime::new(Arc::new(EditMockBackend::new(files)) as BackendRef);

    let result = EditFileTool.invoke(
        serde_json::json!({
            "file_path": "/stable.txt",
            "old_string": "nonexistent_pattern_xyz",
            "new_string": "replacement"
        }),
        &runtime,
    );
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("not found"), "should report not found, got: {}", s);
        }
        _ => panic!("expected Text error"),
    }
}
