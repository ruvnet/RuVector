//! Integration tests for the `execute` tool.

use rvagent_tools::{
    Backend, BackendRef, ExecuteResponse, ExecuteTool, FileInfo, GrepMatch, Tool, ToolResult,
    ToolRuntime, WriteResult,
};
use std::sync::Arc;

/// Mock backend that simulates command execution.
struct ExecMockBackend;

impl Backend for ExecMockBackend {
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
    fn execute(&self, command: &str, _timeout: u32) -> Result<ExecuteResponse, String> {
        if command.contains("echo hello_world") {
            Ok(ExecuteResponse {
                output: "hello_world\n".into(),
                exit_code: 0,
            })
        } else if command.contains("exit 42") {
            Ok(ExecuteResponse {
                output: String::new(),
                exit_code: 42,
            })
        } else if command.contains("sleep 30") {
            Err("command timed out after 1 seconds".into())
        } else {
            Ok(ExecuteResponse {
                output: format!("executed: {}", command),
                exit_code: 0,
            })
        }
    }
}

fn exec_runtime() -> ToolRuntime {
    ToolRuntime::new(Arc::new(ExecMockBackend) as BackendRef)
}

#[test]
fn test_execute_echo() {
    let runtime = exec_runtime();
    let result = ExecuteTool.invoke(serde_json::json!({"command": "echo hello_world"}), &runtime);

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

#[test]
fn test_execute_exit_code() {
    let runtime = exec_runtime();
    let result = ExecuteTool.invoke(serde_json::json!({"command": "exit 42"}), &runtime);

    match result {
        ToolResult::Text(s) => {
            assert!(
                s.contains("exit code: 42"),
                "should report exit code 42, got: {}",
                s
            );
        }
        _ => panic!("expected Text result from execute"),
    }
}

#[test]
fn test_execute_timeout() {
    let runtime = exec_runtime();
    let result = ExecuteTool.invoke(
        serde_json::json!({"command": "sleep 30", "timeout": 1}),
        &runtime,
    );

    match result {
        ToolResult::Text(s) => {
            assert!(s.contains("timed out"), "should report timeout, got: {}", s);
        }
        _ => panic!("expected Text timeout from execute"),
    }
}
