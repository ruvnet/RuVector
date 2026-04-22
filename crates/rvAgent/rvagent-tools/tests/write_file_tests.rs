//! Integration tests for the `write_file` tool.

use rvagent_tools::{
    Backend, BackendRef, ExecuteResponse, FileInfo, GrepMatch, Tool, ToolResult, ToolRuntime,
    WriteFileTool, WriteResult,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Mock backend
// ---------------------------------------------------------------------------

struct WriteMockBackend {
    files: Mutex<HashMap<String, String>>,
}

impl WriteMockBackend {
    fn new(files: HashMap<String, String>) -> Self {
        Self {
            files: Mutex::new(files),
        }
    }

    fn empty() -> Self {
        Self::new(HashMap::new())
    }

    fn get_file(&self, path: &str) -> Option<String> {
        self.files.lock().unwrap().get(path).cloned()
    }
}

impl Backend for WriteMockBackend {
    fn ls_info(&self, _: &str) -> Result<Vec<FileInfo>, String> {
        Ok(vec![])
    }
    fn read(&self, _: &str, _: usize, _: usize) -> Result<String, String> {
        Ok(String::new())
    }
    fn write(&self, path: &str, content: &str) -> WriteResult {
        // Reject directory traversal attempts
        if path.contains("..") {
            return WriteResult {
                error: Some(format!(
                    "Error: invalid path (directory traversal): {}",
                    path
                )),
                ..Default::default()
            };
        }
        let mut files = self.files.lock().unwrap();
        if files.contains_key(path) {
            return WriteResult {
                error: Some(format!(
                    "Error: file {} already exists. Use force flag to overwrite.",
                    path
                )),
                ..Default::default()
            };
        }
        files.insert(path.to_string(), content.to_string());
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_write_new_file_to_temp_directory() {
    let backend = Arc::new(WriteMockBackend::empty());
    let runtime = ToolRuntime::new(backend.clone() as BackendRef);

    let result = WriteFileTool.invoke(
        serde_json::json!({
            "file_path": "/tmp/test_output.txt",
            "content": "hello world"
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("Successfully wrote"),
                "should report success, got: {}",
                s
            );
        }
        _ => panic!("expected Text result from write_file"),
    }

    // Verify content was stored
    let stored = backend.get_file("/tmp/test_output.txt");
    assert_eq!(stored, Some("hello world".to_string()));
}

#[test]
fn test_write_file_overwrite_existing_fails() {
    let mut files = HashMap::new();
    files.insert("/existing.txt".into(), "original content".into());
    let backend = Arc::new(WriteMockBackend::new(files));
    let runtime = ToolRuntime::new(backend.clone() as BackendRef);

    let result = WriteFileTool.invoke(
        serde_json::json!({
            "file_path": "/existing.txt",
            "content": "new content"
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("already exists"),
                "should report file exists error, got: {}",
                s
            );
        }
        _ => panic!("expected Text error from write_file"),
    }

    // Original content should be preserved
    let stored = backend.get_file("/existing.txt");
    assert_eq!(stored, Some("original content".to_string()));
}

#[test]
fn test_write_file_with_proper_content() {
    let backend = Arc::new(WriteMockBackend::empty());
    let runtime = ToolRuntime::new(backend.clone() as BackendRef);

    let content = "line 1\nline 2\nline 3\n";
    let result = WriteFileTool.invoke(
        serde_json::json!({
            "file_path": "/tmp/multiline.txt",
            "content": content
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => assert!(s.contains("Successfully wrote")),
        _ => panic!("expected Text result"),
    }

    let stored = backend.get_file("/tmp/multiline.txt");
    assert_eq!(stored, Some(content.to_string()));
}

#[test]
fn test_write_file_error_on_directory_traversal() {
    let backend = Arc::new(WriteMockBackend::empty());
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = WriteFileTool.invoke(
        serde_json::json!({
            "file_path": "/tmp/../etc/passwd",
            "content": "malicious"
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("Error") || s.contains("invalid path"),
                "should reject directory traversal, got: {}",
                s
            );
        }
        _ => panic!("expected Text error for directory traversal"),
    }
}

#[test]
fn test_write_file_empty_content() {
    let backend = Arc::new(WriteMockBackend::empty());
    let runtime = ToolRuntime::new(backend.clone() as BackendRef);

    let result = WriteFileTool.invoke(
        serde_json::json!({
            "file_path": "/tmp/empty.txt",
            "content": ""
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => assert!(s.contains("Successfully wrote")),
        _ => panic!("expected Text result"),
    }

    let stored = backend.get_file("/tmp/empty.txt");
    assert_eq!(stored, Some(String::new()));
}

#[test]
fn test_write_file_missing_file_path() {
    let backend = Arc::new(WriteMockBackend::empty());
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = WriteFileTool.invoke(serde_json::json!({"content": "hello"}), &runtime);

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("file_path is required"),
                "should report missing file_path, got: {}",
                s
            );
        }
        _ => panic!("expected Text error"),
    }
}

#[test]
fn test_write_file_missing_content() {
    let backend = Arc::new(WriteMockBackend::empty());
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = WriteFileTool.invoke(serde_json::json!({"file_path": "/tmp/test.txt"}), &runtime);

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("content is required"),
                "should report missing content, got: {}",
                s
            );
        }
        _ => panic!("expected Text error"),
    }
}

#[test]
fn test_write_file_special_characters_in_content() {
    let backend = Arc::new(WriteMockBackend::empty());
    let runtime = ToolRuntime::new(backend.clone() as BackendRef);

    let content = "special chars: \t\n\"quotes\" and 'single' and \\backslash\\";
    let result = WriteFileTool.invoke(
        serde_json::json!({
            "file_path": "/tmp/special.txt",
            "content": content
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => assert!(s.contains("Successfully wrote")),
        _ => panic!("expected Text result"),
    }

    let stored = backend.get_file("/tmp/special.txt");
    assert_eq!(stored, Some(content.to_string()));
}
