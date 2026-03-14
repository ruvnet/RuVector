//! Integration tests for the `read_file` tool.

use rvagent_tools::{BuiltinTool, ToolResult, ToolRuntime};

#[test]
fn test_read_full_file() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    let content = "line one\nline two\nline three\n";
    std::fs::write(dir.path().join("sample.txt"), content).unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::ReadFile.invoke(
        serde_json::json!({"file_path": "sample.txt"}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            // Should contain all three lines with line numbers
            assert!(s.contains("line one"), "should contain 'line one'");
            assert!(s.contains("line two"), "should contain 'line two'");
            assert!(s.contains("line three"), "should contain 'line three'");
            // Line numbers: right-aligned 6-wide + tab
            assert!(s.contains("     1\t"), "should have line number 1");
            assert!(s.contains("     2\t"), "should have line number 2");
            assert!(s.contains("     3\t"), "should have line number 3");
        }
        _ => panic!("expected Text result from read_file"),
    }
}

#[test]
fn test_read_with_offset_limit() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    let lines: Vec<String> = (1..=10).map(|i| format!("line {}", i)).collect();
    std::fs::write(dir.path().join("ten_lines.txt"), lines.join("\n")).unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);

    // Read lines 3-5 (offset=2, limit=3)
    let result = BuiltinTool::ReadFile.invoke(
        serde_json::json!({"file_path": "ten_lines.txt", "offset": 2, "limit": 3}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            // Should contain lines 3, 4, 5
            assert!(s.contains("line 3"), "should contain 'line 3'");
            assert!(s.contains("line 4"), "should contain 'line 4'");
            assert!(s.contains("line 5"), "should contain 'line 5'");
            // Should NOT contain line 1 or line 2
            assert!(!s.contains("line 1"), "should not contain 'line 1'");
            assert!(!s.contains("line 2"), "should not contain 'line 2'");
            // Should indicate more lines remain
            assert!(s.contains("more lines"), "should indicate truncation");
        }
        _ => panic!("expected Text result from read_file"),
    }
}

#[test]
fn test_read_nonexistent_file() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::ReadFile.invoke(
        serde_json::json!({"file_path": "does_not_exist.txt"}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("Error"),
                "nonexistent file should produce an error, got: {}",
                s
            );
        }
        _ => panic!("expected Text error from read_file"),
    }
}
