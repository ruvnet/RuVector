//! Integration tests for the `ls` tool.

use rvagent_tools::{BuiltinTool, ToolResult, ToolRuntime};

#[test]
fn test_ls_directory_listing() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    // Create files and a subdirectory
    std::fs::write(dir.path().join("file_a.txt"), "hello").unwrap();
    std::fs::write(dir.path().join("file_b.rs"), "fn main() {}").unwrap();
    std::fs::create_dir(dir.path().join("subdir")).unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::Ls.invoke(
        serde_json::json!({"path": "."}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("file_a.txt"), "should list file_a.txt");
            assert!(s.contains("file_b.rs"), "should list file_b.rs");
            assert!(s.contains("subdir/"), "subdirectory should have trailing slash");
        }
        _ => panic!("expected Text result from ls"),
    }
}

#[test]
fn test_ls_nonexistent_path() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::Ls.invoke(
        serde_json::json!({"path": "nonexistent_dir_xyz"}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("Error"),
                "nonexistent path should produce an error, got: {}",
                s
            );
        }
        _ => panic!("expected Text error from ls"),
    }
}
