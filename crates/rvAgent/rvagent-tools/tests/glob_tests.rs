//! Integration tests for the `glob` tool.

use rvagent_tools::{
    Backend, BackendRef, ExecuteResponse, FileInfo, GlobTool, GrepMatch, Tool, ToolResult,
    ToolRuntime, WriteResult,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Mock backend
// ---------------------------------------------------------------------------

struct GlobMockBackend {
    files: Mutex<HashMap<String, String>>,
}

impl GlobMockBackend {
    fn new(file_paths: Vec<&str>) -> Self {
        let mut files = HashMap::new();
        for path in file_paths {
            files.insert(path.to_string(), String::new());
        }
        Self {
            files: Mutex::new(files),
        }
    }

    fn empty() -> Self {
        Self {
            files: Mutex::new(HashMap::new()),
        }
    }
}

impl Backend for GlobMockBackend {
    fn ls_info(&self, _: &str) -> Result<Vec<FileInfo>, String> {
        Ok(vec![])
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
    fn glob_info(&self, pattern: &str, _path: &str) -> Result<Vec<String>, String> {
        let files = self.files.lock().unwrap();
        let search = pattern.trim_start_matches('*').trim_end_matches('*');
        if search.is_empty() {
            // Wildcard-only pattern matches everything
            let mut matches: Vec<String> = files.keys().cloned().collect();
            matches.sort();
            return Ok(matches);
        }
        let mut matches: Vec<String> = files
            .keys()
            .filter(|k| k.contains(search))
            .cloned()
            .collect();
        matches.sort();
        Ok(matches)
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
fn test_glob_pattern_matching_files_in_directory() {
    let backend = Arc::new(GlobMockBackend::new(vec![
        "/src/main.rs",
        "/src/lib.rs",
        "/src/utils.rs",
        "/tests/test_main.rs",
        "/Cargo.toml",
    ]));
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = GlobTool.invoke(serde_json::json!({"pattern": "*.rs"}), &runtime);

    match result {
        ToolResult::Text(s) => {
            assert!(s.contains(".rs"), "should find .rs files, got: {}", s);
            assert!(s.contains("files)"), "should show file count");
        }
        _ => panic!("expected Text result"),
    }
}

#[test]
fn test_glob_matching_with_wildcards() {
    let backend = Arc::new(GlobMockBackend::new(vec![
        "/src/main.rs",
        "/src/lib.rs",
        "/docs/readme.md",
        "/docs/guide.md",
    ]));
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = GlobTool.invoke(serde_json::json!({"pattern": "*.md"}), &runtime);

    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("readme.md"), "should match readme.md");
            assert!(s.contains("guide.md"), "should match guide.md");
            assert!(!s.contains("main.rs"), "should not match .rs files");
        }
        _ => panic!("expected Text result"),
    }
}

#[test]
fn test_glob_empty_results() {
    let backend = Arc::new(GlobMockBackend::new(vec!["/src/main.rs", "/src/lib.rs"]));
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = GlobTool.invoke(
        serde_json::json!({"pattern": "*.xyz_nonexistent"}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("No files matching"),
                "should report no matches, got: {}",
                s
            );
        }
        _ => panic!("expected Text result for no matches"),
    }
}

#[test]
fn test_glob_nested_directory_matching() {
    let backend = Arc::new(GlobMockBackend::new(vec![
        "/project/src/main.rs",
        "/project/src/utils/helpers.rs",
        "/project/src/utils/math.rs",
        "/project/tests/integration.rs",
    ]));
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = GlobTool.invoke(serde_json::json!({"pattern": "*utils*"}), &runtime);

    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("utils"), "should match nested utils paths");
            assert!(s.contains("helpers.rs"), "should match helpers.rs");
            assert!(s.contains("math.rs"), "should match math.rs");
        }
        _ => panic!("expected Text result"),
    }
}

#[test]
fn test_glob_missing_pattern_parameter() {
    let backend = Arc::new(GlobMockBackend::empty());
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = GlobTool.invoke(serde_json::json!({}), &runtime);

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("pattern is required"),
                "should report missing pattern, got: {}",
                s
            );
        }
        _ => panic!("expected error Text"),
    }
}

#[test]
fn test_glob_with_explicit_path() {
    let backend = Arc::new(GlobMockBackend::new(vec![
        "/home/user/project/src/main.rs",
        "/home/user/project/src/lib.rs",
    ]));
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = GlobTool.invoke(
        serde_json::json!({"pattern": "*.rs", "path": "/home/user/project"}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains(".rs"),
                "should find .rs files with explicit path"
            );
        }
        _ => panic!("expected Text result"),
    }
}

#[test]
fn test_glob_empty_filesystem() {
    let backend = Arc::new(GlobMockBackend::empty());
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = GlobTool.invoke(serde_json::json!({"pattern": "*.rs"}), &runtime);

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("No files matching"),
                "empty fs should produce no matches, got: {}",
                s
            );
        }
        _ => panic!("expected Text result"),
    }
}

#[test]
fn test_glob_result_is_sorted() {
    let backend = Arc::new(GlobMockBackend::new(vec![
        "/c_file.txt",
        "/a_file.txt",
        "/b_file.txt",
    ]));
    let runtime = ToolRuntime::new(backend as BackendRef);

    let result = GlobTool.invoke(serde_json::json!({"pattern": "*.txt"}), &runtime);

    match result {
        ToolResult::Text(s) => {
            let lines: Vec<&str> = s.lines().collect();
            // The first three lines should be the files in sorted order
            assert!(lines.len() >= 3, "should have at least 3 lines");
            assert!(lines[0].contains("a_file"), "first should be a_file");
            assert!(lines[1].contains("b_file"), "second should be b_file");
            assert!(lines[2].contains("c_file"), "third should be c_file");
        }
        _ => panic!("expected Text result"),
    }
}
