//! Integration tests for the `grep` tool.

use rvagent_tools::{BuiltinTool, ToolResult, ToolRuntime};

#[test]
fn test_grep_literal_match() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    std::fs::write(dir.path().join("src.rs"), "fn main() {\n    println!(\"hello\");\n}\n").unwrap();
    std::fs::write(dir.path().join("notes.txt"), "no match here\n").unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::Grep.invoke(
        serde_json::json!({"pattern": "println"}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("println"), "should find 'println' in results");
            assert!(s.contains("src.rs"), "should reference the file containing the match");
            assert!(s.contains(":2:"), "match should be on line 2");
        }
        _ => panic!("expected Text result from grep"),
    }
}

#[test]
fn test_grep_no_results() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    std::fs::write(dir.path().join("empty_match.txt"), "alpha beta gamma\n").unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::Grep.invoke(
        serde_json::json!({"pattern": "nonexistent_pattern_xyz_123"}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("No results") || s.contains("No matches"),
                "should report no results, got: {}",
                s
            );
        }
        _ => panic!("expected Text from grep"),
    }
}

#[test]
fn test_grep_with_include_filter() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    // Both files contain "target", but include filter should restrict to *.rs
    std::fs::write(dir.path().join("code.rs"), "let target = 42;\n").unwrap();
    std::fs::write(dir.path().join("notes.txt"), "target reached\n").unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::Grep.invoke(
        serde_json::json!({
            "pattern": "target",
            "include": "*.rs"
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("code.rs"), "should find match in code.rs");
            assert!(
                !s.contains("notes.txt"),
                "should NOT include notes.txt due to include filter, got: {}",
                s
            );
        }
        _ => panic!("expected Text result from grep"),
    }
}
