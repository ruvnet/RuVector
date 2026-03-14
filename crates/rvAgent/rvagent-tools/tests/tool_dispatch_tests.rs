//! Integration tests for tool dispatch — BuiltinTool, AnyTool, parallel execution,
//! and ToolRuntime creation (ADR-103 A6, A2).

use rvagent_tools::{
    AnyTool, BuiltinTool, Tool, ToolResult, ToolRuntime,
    execute_tool_calls_parallel, resolve_builtin,
};
use async_trait::async_trait;

// ---------------------------------------------------------------------------
// test_builtin_tool_enum_dispatch
// ---------------------------------------------------------------------------

#[test]
fn test_builtin_tool_enum_dispatch() {
    // Each BuiltinTool variant must return the correct canonical name and
    // produce a ToolResult without panicking.
    let variants = vec![
        (BuiltinTool::Ls, "ls"),
        (BuiltinTool::ReadFile, "read_file"),
        (BuiltinTool::WriteFile, "write_file"),
        (BuiltinTool::EditFile, "edit_file"),
        (BuiltinTool::Glob, "glob"),
        (BuiltinTool::Grep, "grep"),
        (BuiltinTool::Execute, "execute"),
        (BuiltinTool::WriteTodos, "write_todos"),
        (BuiltinTool::Task, "task"),
    ];

    for (variant, expected_name) in &variants {
        assert_eq!(
            variant.tool_name(),
            *expected_name,
            "BuiltinTool::{:?} should have name '{}'",
            variant,
            expected_name,
        );
    }

    // resolve_builtin round-trips every name
    for (variant, name) in &variants {
        let resolved = resolve_builtin(name);
        assert!(
            resolved.is_some(),
            "resolve_builtin should find '{}'",
            name
        );
        assert_eq!(resolved.unwrap().tool_name(), *name);
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
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "echoes input"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    fn invoke(&self, args: serde_json::Value, _runtime: &ToolRuntime) -> ToolResult {
        let msg = args
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("(empty)");
        ToolResult::Text(format!("echo: {}", msg))
    }
}

#[test]
fn test_any_tool_builtin_vs_dynamic() {
    let runtime = ToolRuntime::new();

    // Builtin path
    let builtin = AnyTool::Builtin(BuiltinTool::WriteTodos);
    assert_eq!(builtin.tool_name(), "write_todos");
    let result = builtin.invoke(serde_json::json!({}), &runtime);
    match result {
        ToolResult::Text(s) => assert!(s.contains("stub")),
        _ => panic!("expected Text from WriteTodos stub"),
    }

    // Dynamic path
    let dynamic = AnyTool::Dynamic(Box::new(EchoTool));
    assert_eq!(dynamic.tool_name(), "echo");
    let result = dynamic.invoke(
        serde_json::json!({"message": "hello"}),
        &runtime,
    );
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
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap().to_string();

    // Create a couple of files so ls and grep have something to work with.
    std::fs::write(dir.path().join("a.txt"), "alpha\nbeta\n").unwrap();
    std::fs::write(dir.path().join("b.txt"), "gamma\ndelta\n").unwrap();

    let runtime = ToolRuntime::new().with_cwd(&dir_path);

    let tools: Vec<AnyTool> = vec![
        AnyTool::Builtin(BuiltinTool::Ls),
        AnyTool::Builtin(BuiltinTool::Grep),
    ];

    let calls = vec![
        (0, serde_json::json!({"path": "."})),       // ls
        (1, serde_json::json!({"pattern": "alpha"})), // grep
    ];

    let results = execute_tool_calls_parallel(&tools, calls, &runtime).await;

    // Both should complete.
    assert_eq!(results.len(), 2);

    // Results sorted by index.
    assert_eq!(results[0].0, 0); // ls
    assert_eq!(results[1].0, 1); // grep

    // ls should list files
    match &results[0].1 {
        ToolResult::Text(s) => assert!(s.contains("a.txt")),
        _ => panic!("expected Text from ls"),
    }

    // grep should find "alpha"
    match &results[1].1 {
        ToolResult::Text(s) => assert!(s.contains("alpha")),
        _ => panic!("expected Text from grep"),
    }
}

// ---------------------------------------------------------------------------
// test_tool_runtime_creation
// ---------------------------------------------------------------------------

#[test]
fn test_tool_runtime_creation() {
    // Default runtime
    let rt = ToolRuntime::new();
    assert!(rt.cwd.is_none());
    assert!(rt.tool_call_id.is_none());
    assert_eq!(rt.context, serde_json::Value::Null);

    // With cwd
    let rt2 = ToolRuntime::new().with_cwd("/tmp/project");
    assert_eq!(rt2.cwd.as_deref(), Some("/tmp/project"));

    // Default trait
    let rt3 = ToolRuntime::default();
    assert!(rt3.cwd.is_none());
}
