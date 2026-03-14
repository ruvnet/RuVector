//! Integration tests for the `edit_file` tool.

use rvagent_tools::{BuiltinTool, ToolResult, ToolRuntime};

#[test]
fn test_edit_unique_match() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    std::fs::write(
        dir.path().join("code.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::EditFile.invoke(
        serde_json::json!({
            "file_path": "code.rs",
            "old_string": "hello",
            "new_string": "world"
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("Successfully edited"), "should report success, got: {}", s);
            assert!(s.contains("1 occurrence"), "should report 1 occurrence");
        }
        _ => panic!("expected Text result from edit_file"),
    }

    // Verify the file was actually changed
    let updated = std::fs::read_to_string(dir.path().join("code.rs")).unwrap();
    assert!(updated.contains("world"));
    assert!(!updated.contains("hello"));
}

#[test]
fn test_edit_non_unique_error() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    // File with "foo" appearing twice
    std::fs::write(
        dir.path().join("dup.txt"),
        "foo bar\nbaz foo\n",
    )
    .unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::EditFile.invoke(
        serde_json::json!({
            "file_path": "dup.txt",
            "old_string": "foo",
            "new_string": "qux"
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("2 times") || s.contains("found 2"),
                "should report non-unique match, got: {}",
                s
            );
        }
        _ => panic!("expected Text error from edit_file"),
    }

    // File should be unchanged
    let content = std::fs::read_to_string(dir.path().join("dup.txt")).unwrap();
    assert_eq!(content, "foo bar\nbaz foo\n");
}

#[test]
fn test_edit_replace_all() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    std::fs::write(
        dir.path().join("multi.txt"),
        "aaa bbb\nccc aaa\naaa\n",
    )
    .unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::EditFile.invoke(
        serde_json::json!({
            "file_path": "multi.txt",
            "old_string": "aaa",
            "new_string": "zzz",
            "replace_all": true
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("Successfully edited"), "should report success, got: {}", s);
            assert!(s.contains("3 occurrence"), "should report 3 occurrences");
        }
        _ => panic!("expected Text result from edit_file"),
    }

    let updated = std::fs::read_to_string(dir.path().join("multi.txt")).unwrap();
    assert!(!updated.contains("aaa"));
    assert_eq!(updated.matches("zzz").count(), 3);
}

#[test]
fn test_edit_no_match() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    std::fs::write(dir.path().join("stable.txt"), "nothing to change\n").unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::EditFile.invoke(
        serde_json::json!({
            "file_path": "stable.txt",
            "old_string": "nonexistent_pattern_xyz",
            "new_string": "replacement"
        }),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("not found"),
                "should report old_string not found, got: {}",
                s
            );
        }
        _ => panic!("expected Text error from edit_file"),
    }
}
