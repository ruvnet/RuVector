//! Integration tests for the `execute` tool.

use rvagent_tools::{BuiltinTool, ToolResult, ToolRuntime};

#[test]
fn test_execute_echo() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);
    let result = BuiltinTool::Execute.invoke(
        serde_json::json!({"command": "echo hello_world"}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("hello_world"),
                "should capture echo output, got: {}",
                s
            );
        }
        _ => panic!("expected Text result from execute"),
    }
}

#[tokio::test]
async fn test_execute_timeout() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);

    // Use ainvoke with a very short timeout
    let result = BuiltinTool::Execute
        .ainvoke(
            serde_json::json!({"command": "sleep 30", "timeout": 1}),
            &runtime,
        )
        .await;

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("timed out"),
                "should report timeout, got: {}",
                s
            );
        }
        _ => panic!("expected Text timeout from execute"),
    }
}

#[test]
fn test_execute_exit_code() {
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);

    // Run a command that exits with non-zero
    let result = BuiltinTool::Execute.invoke(
        serde_json::json!({"command": "exit 42"}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("Exit code: 42") || s.contains("exit code: 42"),
                "should report exit code 42, got: {}",
                s
            );
        }
        _ => panic!("expected Text result from execute"),
    }
}
