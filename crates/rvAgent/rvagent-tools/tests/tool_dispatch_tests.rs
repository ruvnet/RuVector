//! Integration tests for tool dispatch — BuiltinTool, AnyTool, parallel execution,
//! and ToolRuntime creation (ADR-103 A6, A2).

use rvagent_tools::{
    AnyTool, Backend, BackendRef, BuiltinTool, ExecuteResponse, FileInfo,
    GrepMatch, Tool, ToolCall, ToolResult, ToolRuntime, WriteResult,
    builtin_tools, execute_tools_parallel, resolve_builtin,
};
use async_trait::async_trait;
use std::sync::Arc;

/// Minimal mock backend for integration tests.
struct MockBackend;

impl Backend for MockBackend {
    fn ls_info(&self, _path: &str) -> Result<Vec<FileInfo>, String> {
        Ok(vec![FileInfo {
            name: "test.txt".into(),
            file_type: "file".into(),
            permissions: "-rw-r--r--".into(),
            size: 11,
        }])
    }
    fn read(&self, _path: &str, _offset: usize, _limit: usize) -> Result<String, String> {
        Ok("hello\nworld".into())
    }
    fn write(&self, _path: &str, _content: &str) -> WriteResult {
        WriteResult::default()
    }
    fn edit(&self, _path: &str, _old: &str, _new: &str, _all: bool) -> WriteResult {
        WriteResult { occurrences: Some(1), ..Default::default() }
    }
    fn glob_info(&self, _pattern: &str, _path: &str) -> Result<Vec<String>, String> {
        Ok(vec!["test.txt".into()])
    }
    fn grep_raw(&self, pattern: &str, _path: Option<&str>, _include: Option<&str>) -> Result<Vec<GrepMatch>, String> {
        Ok(vec![GrepMatch { file: "test.txt".into(), line_number: 1, text: format!("line with {}", pattern) }])
    }
    fn execute(&self, cmd: &str, _timeout: u32) -> Result<ExecuteResponse, String> {
        Ok(ExecuteResponse { output: format!("executed: {}", cmd), exit_code: 0 })
    }
}

fn mock_runtime() -> ToolRuntime {
    ToolRuntime::new(Arc::new(MockBackend) as BackendRef)
}

// ---------------------------------------------------------------------------
// test_builtin_tool_enum_dispatch
// ---------------------------------------------------------------------------

#[test]
fn test_builtin_tool_enum_dispatch() {
    let variants: Vec<(&str, BuiltinTool)> = vec![
        ("ls", resolve_builtin("ls").unwrap()),
        ("read_file", resolve_builtin("read_file").unwrap()),
        ("write_file", resolve_builtin("write_file").unwrap()),
        ("edit_file", resolve_builtin("edit_file").unwrap()),
        ("glob", resolve_builtin("glob").unwrap()),
        ("grep", resolve_builtin("grep").unwrap()),
        ("execute", resolve_builtin("execute").unwrap()),
        ("write_todos", resolve_builtin("write_todos").unwrap()),
        ("task", resolve_builtin("task").unwrap()),
    ];

    for (expected_name, variant) in &variants {
        assert_eq!(variant.name(), *expected_name);
        // Each variant should produce a non-empty description and valid schema
        assert!(!variant.description().is_empty());
        assert!(variant.parameters_schema().is_object());
    }

    // Unknown name returns None
    assert!(resolve_builtin("nonexistent_tool").is_none());
}

// ---------------------------------------------------------------------------
// test_any_tool_builtin_vs_dynamic
// ---------------------------------------------------------------------------

/// A minimal dynamic tool for testing AnyTool::Dynamic.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "echoes input" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    fn invoke(&self, args: serde_json::Value, _runtime: &ToolRuntime) -> ToolResult {
        let msg = args.get("message").and_then(|v| v.as_str()).unwrap_or("(empty)");
        ToolResult::Text(format!("echo: {}", msg))
    }
}

#[test]
fn test_any_tool_builtin_vs_dynamic() {
    let runtime = mock_runtime();

    // Builtin path
    let builtin = AnyTool::Builtin(resolve_builtin("ls").unwrap());
    assert_eq!(builtin.name(), "ls");
    let result = builtin.invoke(serde_json::json!({"path": "/"}), &runtime);
    match result {
        ToolResult::Text(s) => assert!(s.contains("test.txt")),
        _ => panic!("expected Text from ls"),
    }

    // Dynamic path
    let dynamic = AnyTool::Dynamic(Box::new(EchoTool));
    assert_eq!(dynamic.name(), "echo");
    let result = dynamic.invoke(serde_json::json!({"message": "hello"}), &runtime);
    match result {
        ToolResult::Text(s) => assert_eq!(s, "echo: hello"),
        _ => panic!("expected Text from EchoTool"),
    }
}

// ---------------------------------------------------------------------------
// test_parallel_tool_execution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_parallel_tool_execution() {
    let runtime = mock_runtime();
    let tools = builtin_tools();

    let calls = vec![
        ToolCall { id: "c1".into(), name: "ls".into(), args: serde_json::json!({"path": "/"}) },
        ToolCall { id: "c2".into(), name: "grep".into(), args: serde_json::json!({"pattern": "hello"}) },
    ];

    let results = execute_tools_parallel(&calls, &tools, &runtime).await;
    assert_eq!(results.len(), 2);

    // ls result
    match &results[0] {
        ToolResult::Text(s) => assert!(s.contains("test.txt")),
        _ => panic!("expected Text from ls"),
    }
    // grep result
    match &results[1] {
        ToolResult::Text(s) => assert!(s.contains("hello")),
        _ => panic!("expected Text from grep"),
    }
}

// ---------------------------------------------------------------------------
// test_tool_runtime_creation
// ---------------------------------------------------------------------------

#[test]
fn test_tool_runtime_creation() {
    let runtime = mock_runtime();
    assert!(runtime.context.is_null());
    assert!(runtime.tool_call_id.is_none());
    assert!(runtime.stream_writer.is_none());
    assert!(runtime.store.is_none());
}
