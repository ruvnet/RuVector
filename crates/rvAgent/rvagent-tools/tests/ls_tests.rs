//! Integration tests for the `ls` tool.

use rvagent_tools::{
    Backend, BackendRef, ExecuteResponse, FileInfo, GrepMatch, LsTool, Tool, ToolResult,
    ToolRuntime, WriteResult,
};
use std::sync::Arc;

struct LsMockBackend {
    entries: Vec<FileInfo>,
}

impl Backend for LsMockBackend {
    fn ls_info(&self, _path: &str) -> Result<Vec<FileInfo>, String> {
        Ok(self.entries.clone())
    }
    fn read(&self, _: &str, _: usize, _: usize) -> Result<String, String> {
        Ok(String::new())
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

struct ErrorLsBackend;

impl Backend for ErrorLsBackend {
    fn ls_info(&self, path: &str) -> Result<Vec<FileInfo>, String> {
        Err(format!("Error: path '{}' not found", path))
    }
    fn read(&self, _: &str, _: usize, _: usize) -> Result<String, String> {
        Err("n/a".into())
    }
    fn write(&self, _: &str, _: &str) -> WriteResult {
        WriteResult::default()
    }
    fn edit(&self, _: &str, _: &str, _: &str, _: bool) -> WriteResult {
        WriteResult::default()
    }
    fn glob_info(&self, _: &str, _: &str) -> Result<Vec<String>, String> {
        Err("n/a".into())
    }
    fn grep_raw(
        &self,
        _: &str,
        _: Option<&str>,
        _: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        Err("n/a".into())
    }
    fn execute(&self, _: &str, _: u32) -> Result<ExecuteResponse, String> {
        Err("n/a".into())
    }
}

#[test]
fn test_ls_directory_listing() {
    let backend = Arc::new(LsMockBackend {
        entries: vec![
            FileInfo {
                name: "file_a.txt".into(),
                file_type: "file".into(),
                permissions: "-rw-r--r--".into(),
                size: 5,
            },
            FileInfo {
                name: "file_b.rs".into(),
                file_type: "file".into(),
                permissions: "-rw-r--r--".into(),
                size: 12,
            },
            FileInfo {
                name: "subdir".into(),
                file_type: "dir".into(),
                permissions: "drwxr-xr-x".into(),
                size: 0,
            },
        ],
    }) as BackendRef;
    let runtime = ToolRuntime::new(backend);

    let result = LsTool.invoke(serde_json::json!({"path": "/test"}), &runtime);
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("file_a.txt"), "should list file_a.txt");
            assert!(s.contains("file_b.rs"), "should list file_b.rs");
            assert!(s.contains("subdir"), "should list subdir");
        }
        _ => panic!("expected Text result from ls"),
    }
}

#[test]
fn test_ls_nonexistent_path() {
    let runtime = ToolRuntime::new(Arc::new(ErrorLsBackend) as BackendRef);

    let result = LsTool.invoke(serde_json::json!({"path": "/nonexistent"}), &runtime);
    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("Error"), "should produce error, got: {}", s);
        }
        _ => panic!("expected Text error from ls"),
    }
}
