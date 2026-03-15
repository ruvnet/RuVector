//! Stress and property-based tests for rvagent-mcp.
//! Tests registry scaling, concurrent access patterns, protocol serde,
//! and edge cases for the MCP system.

use std::sync::Arc;

use rvagent_mcp::protocol::*;
use rvagent_mcp::registry::*;
use rvagent_mcp::{McpError, McpToolRegistry};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple handler that returns "ok" for any arguments.
struct OkHandler;

#[async_trait::async_trait]
impl McpToolHandler for OkHandler {
    async fn execute(&self, _arguments: serde_json::Value) -> rvagent_mcp::Result<ToolCallResult> {
        Ok(ToolCallResult {
            content: vec![Content::text("ok")],
            is_error: false,
        })
    }
}

fn make_tool(name: &str) -> McpToolDefinition {
    McpToolDefinition {
        name: name.into(),
        description: format!("{} tool", name),
        input_schema: serde_json::json!({"type": "object", "properties": {}}),
        handler: Arc::new(OkHandler),
    }
}

fn make_tool_with_schema(name: &str, schema: serde_json::Value) -> McpToolDefinition {
    McpToolDefinition {
        name: name.into(),
        description: format!("{} tool", name),
        input_schema: schema,
        handler: Arc::new(OkHandler),
    }
}

// ---------------------------------------------------------------------------
// Stress: Registry scaling
// ---------------------------------------------------------------------------

/// Stress test: Register 500 tools in a single registry.
#[test]
fn stress_registry_500_tools() {
    let reg = McpToolRegistry::new();
    for i in 0..500 {
        let name = format!("tool-{}", i);
        reg.register_tool(make_tool(&name)).unwrap();
    }
    assert_eq!(reg.len(), 500);

    // Lookup should still work for all tools
    for i in 0..500 {
        let name = format!("tool-{}", i);
        assert!(reg.get_tool(&name).is_some(), "Missing tool: {}", name);
    }

    // List should be sorted
    let tools = reg.list_tools();
    assert_eq!(tools.len(), 500);
    for w in tools.windows(2) {
        assert!(w[0].name <= w[1].name, "Not sorted: {} > {}", w[0].name, w[1].name);
    }
}

/// Stress test: Rapid register/unregister churn.
#[test]
fn stress_registry_churn() {
    let reg = McpToolRegistry::new();

    // Add 100 tools
    for i in 0..100 {
        reg.register_tool(make_tool(&format!("churn-{}", i))).unwrap();
    }
    assert_eq!(reg.len(), 100);

    // Remove every other tool
    for i in (0..100).step_by(2) {
        reg.unregister_tool(&format!("churn-{}", i)).unwrap();
    }
    assert_eq!(reg.len(), 50);

    // Remaining tools should be the odd-numbered ones
    for i in (1..100).step_by(2) {
        assert!(reg.get_tool(&format!("churn-{}", i)).is_some());
    }

    // Removed tools should be gone
    for i in (0..100).step_by(2) {
        assert!(reg.get_tool(&format!("churn-{}", i)).is_none());
    }
}

/// Stress test: Re-register after unregister cycle.
#[test]
fn stress_registry_re_register() {
    let reg = McpToolRegistry::new();
    for cycle in 0..10 {
        let name = format!("cycle-tool-{}", cycle);
        reg.register_tool(make_tool(&name)).unwrap();
        reg.unregister_tool(&name).unwrap();
        // Re-register same name
        reg.register_tool(make_tool(&name)).unwrap();
    }
    assert_eq!(reg.len(), 10);
}

// ---------------------------------------------------------------------------
// Stress: Protocol serde throughput
// ---------------------------------------------------------------------------

/// Stress: Serialize/deserialize 1000 JsonRpcRequests.
#[test]
fn stress_jsonrpc_request_serde_throughput() {
    let base = JsonRpcRequest::new(1, "tools/call")
        .with_params(serde_json::json!({"name": "read_file", "arguments": {"file_path": "/test.txt"}}));

    for i in 0..1000 {
        let mut req = base.clone();
        req.id = serde_json::json!(i);
        let json = serde_json::to_string(&req).unwrap();
        let back: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.method, "tools/call");
        assert_eq!(back.jsonrpc, "2.0");
    }
}

/// Stress: Serialize/deserialize 1000 JsonRpcResponses.
#[test]
fn stress_jsonrpc_response_serde_throughput() {
    for i in 0..1000 {
        let resp = JsonRpcResponse::success(
            serde_json::json!(i),
            serde_json::json!({"content": [{"type": "text", "text": "result"}]}),
        );
        let json = serde_json::to_string(&resp).unwrap();
        let back: JsonRpcResponse = serde_json::from_str(&json).unwrap();
        assert!(back.result.is_some());
        assert!(back.error.is_none());
    }
}

/// Stress: McpTool serde roundtrip at scale.
#[test]
fn stress_mcp_tool_serde_roundtrip() {
    for i in 0..500 {
        let tool = McpTool {
            name: format!("tool-{}", i),
            description: format!("Tool number {} for stress testing", i),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"},
                    "arg2": {"type": "integer"}
                },
                "required": ["arg1"]
            }),
        };
        let json = serde_json::to_string(&tool).unwrap();
        let back: McpTool = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, tool.name);
        assert_eq!(back.description, tool.description);
        assert!(back.input_schema.is_object());
    }
}

/// Stress: Content variant serde roundtrip at scale.
#[test]
fn stress_content_serde_roundtrip() {
    for i in 0..500 {
        let content = match i % 2 {
            0 => Content::text(format!("text-content-{}", i)),
            _ => Content::image(format!("base64data{}==", i), "image/png"),
        };
        let json = serde_json::to_string(&content).unwrap();
        let back: Content = serde_json::from_str(&json).unwrap();
        match (&content, &back) {
            (Content::Text { text: a }, Content::Text { text: b }) => assert_eq!(a, b),
            (Content::Image { data: a, .. }, Content::Image { data: b, .. }) => assert_eq!(a, b),
            _ => panic!("Mismatched content types at iteration {}", i),
        }
    }
}

// ---------------------------------------------------------------------------
// Stress: Async tool execution
// ---------------------------------------------------------------------------

/// Stress: Call tools many times via the registry.
#[tokio::test]
async fn stress_registry_call_tool_throughput() {
    let reg = McpToolRegistry::new();
    for i in 0..20 {
        reg.register_tool(make_tool(&format!("fast-{}", i))).unwrap();
    }

    for round in 0..50 {
        let tool_name = format!("fast-{}", round % 20);
        let result = reg.call_tool(&tool_name, serde_json::json!({})).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
    }
}

/// Stress: Register builtins and call them many times.
#[tokio::test]
async fn stress_builtins_repeated_calls() {
    let reg = McpToolRegistry::new();
    register_builtins(&reg, serde_json::json!({"tools": true, "resources": false})).unwrap();

    for _ in 0..100 {
        let ping_result = reg.call_tool("ping", serde_json::Value::Null).await.unwrap();
        assert!(!ping_result.is_error);

        let echo_result = reg
            .call_tool("echo", serde_json::json!({"text": "hello"}))
            .await
            .unwrap();
        match &echo_result.content[0] {
            Content::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("expected text content"),
        }
    }
}

// ---------------------------------------------------------------------------
// Property: Validation consistency
// ---------------------------------------------------------------------------

/// Property: validate_args is consistent for all registered tools.
#[test]
fn property_validate_args_consistency() {
    let reg = McpToolRegistry::new();
    let schemas = vec![
        ("obj-tool", serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]})),
        ("no-req", serde_json::json!({"type": "object", "properties": {"b": {"type": "number"}}})),
        ("empty-obj", serde_json::json!({"type": "object", "properties": {}})),
    ];

    for (name, schema) in &schemas {
        reg.register_tool(make_tool_with_schema(name, schema.clone())).unwrap();
    }

    // Tool with required field: empty object fails
    assert!(reg.validate_args("obj-tool", &serde_json::json!({})).is_err());
    // Tool with required field: present field passes
    assert!(reg.validate_args("obj-tool", &serde_json::json!({"a": "val"})).is_ok());
    // Tool without required fields: empty object passes
    assert!(reg.validate_args("no-req", &serde_json::json!({})).is_ok());
    // Tool with empty properties: passes
    assert!(reg.validate_args("empty-obj", &serde_json::json!({})).is_ok());
    // Non-object arg against object schema: fails
    assert!(reg.validate_args("obj-tool", &serde_json::json!("string")).is_err());
}

/// Property: All McpMethod variants roundtrip through as_str/from_str.
#[test]
fn property_mcp_method_roundtrip_all() {
    let methods = [
        McpMethod::Initialize,
        McpMethod::ToolsList,
        McpMethod::ToolsCall,
        McpMethod::ResourcesList,
        McpMethod::ResourcesRead,
        McpMethod::ResourcesTemplatesList,
        McpMethod::PromptsList,
        McpMethod::PromptsGet,
        McpMethod::Ping,
    ];

    for method in &methods {
        let s = method.as_str();
        let parsed = McpMethod::from_str(s);
        assert_eq!(parsed.as_ref(), Some(method), "Failed roundtrip for {:?}", method);
    }
}

/// Property: All JsonRpcError factory methods produce correct codes.
#[test]
fn property_jsonrpc_error_codes_valid() {
    let cases = vec![
        (JsonRpcError::parse_error("x"), -32700),
        (JsonRpcError::invalid_request("x"), -32600),
        (JsonRpcError::method_not_found("x"), -32601),
        (JsonRpcError::invalid_params("x"), -32602),
        (JsonRpcError::internal_error("x"), -32603),
    ];
    for (err, expected_code) in &cases {
        assert_eq!(err.code, *expected_code);
        // Serde roundtrip
        let json = serde_json::to_string(&err).unwrap();
        let back: JsonRpcError = serde_json::from_str(&json).unwrap();
        assert_eq!(back.code, *expected_code);
        assert_eq!(back.message, "x");
    }
}

/// Property: ServerCapabilities default roundtrips correctly.
#[test]
fn property_server_capabilities_default_roundtrip() {
    let caps = ServerCapabilities::default();
    let json = serde_json::to_string(&caps).unwrap();
    let back: ServerCapabilities = serde_json::from_str(&json).unwrap();
    assert!(back.tools.is_none());
    assert!(back.resources.is_none());
    assert!(back.prompts.is_none());
}

/// Property: McpToolDefinition clone preserves all fields.
#[test]
fn property_tool_definition_clone_preserves_fields() {
    for i in 0..50 {
        let original = make_tool_with_schema(
            &format!("clone-test-{}", i),
            serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"]
            }),
        );
        let cloned = original.clone();
        assert_eq!(cloned.name, original.name);
        assert_eq!(cloned.description, original.description);
        assert_eq!(cloned.input_schema, original.input_schema);
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Edge case: Empty tool name registration.
#[test]
fn edge_empty_tool_name() {
    let reg = McpToolRegistry::new();
    // Empty name should still be registrable (no validation against empty)
    reg.register_tool(make_tool("")).unwrap();
    assert!(reg.get_tool("").is_some());
}

/// Edge case: Very long tool names.
#[test]
fn edge_long_tool_name() {
    let reg = McpToolRegistry::new();
    let long_name = "a".repeat(10_000);
    reg.register_tool(make_tool(&long_name)).unwrap();
    assert!(reg.get_tool(&long_name).is_some());
    let tools = reg.list_mcp_tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, long_name);
}

/// Edge case: Duplicate tool registration returns error.
#[test]
fn edge_duplicate_tool_registration() {
    let reg = McpToolRegistry::new();
    reg.register_tool(make_tool("dup")).unwrap();
    let result = reg.register_tool(make_tool("dup"));
    assert!(result.is_err());
    // Original tool should remain
    assert_eq!(reg.len(), 1);
}

/// Edge case: Unregister non-existent tool.
#[test]
fn edge_unregister_nonexistent() {
    let reg = McpToolRegistry::new();
    let result = reg.unregister_tool("ghost");
    assert!(result.is_err());
}

/// Edge case: Call non-existent tool.
#[tokio::test]
async fn edge_call_nonexistent_tool() {
    let reg = McpToolRegistry::new();
    let result = reg.call_tool("does-not-exist", serde_json::Value::Null).await;
    assert!(result.is_err());
}

/// Edge case: Validate args for non-existent tool.
#[test]
fn edge_validate_args_nonexistent() {
    let reg = McpToolRegistry::new();
    let result = reg.validate_args("ghost", &serde_json::json!({}));
    assert!(result.is_err());
}

/// Edge case: JsonRpcRequest with null id.
#[test]
fn edge_jsonrpc_null_id() {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: serde_json::Value::Null,
        method: "ping".into(),
        params: None,
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: JsonRpcRequest = serde_json::from_str(&json).unwrap();
    assert!(back.id.is_null());
}

/// Edge case: JsonRpcRequest with string id.
#[test]
fn edge_jsonrpc_string_id() {
    let req = JsonRpcRequest::new("req-abc-123", "tools/list");
    let json = serde_json::to_string(&req).unwrap();
    let back: JsonRpcRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.id.as_str(), Some("req-abc-123"));
}

/// Edge case: Content with empty text.
#[test]
fn edge_content_empty_text() {
    let c = Content::text("");
    let json = serde_json::to_string(&c).unwrap();
    let back: Content = serde_json::from_str(&json).unwrap();
    match back {
        Content::Text { text } => assert!(text.is_empty()),
        _ => panic!("expected text content"),
    }
}

/// Edge case: ToolCallResult with is_error=true.
#[test]
fn edge_tool_call_result_error() {
    let result = ToolCallResult {
        content: vec![Content::text("something went wrong")],
        is_error: true,
    };
    let json = serde_json::to_string(&result).unwrap();
    let back: ToolCallResult = serde_json::from_str(&json).unwrap();
    assert!(back.is_error);
    assert_eq!(back.content.len(), 1);
}

/// Edge case: McpResource with no optional fields.
#[test]
fn edge_resource_minimal() {
    let r = McpResource {
        uri: "rvagent://minimal".into(),
        name: "minimal".into(),
        description: None,
        mime_type: None,
    };
    let json = serde_json::to_string(&r).unwrap();
    // Optional fields should be omitted
    assert!(!json.contains("description"));
    assert!(!json.contains("mimeType"));
    let back: McpResource = serde_json::from_str(&json).unwrap();
    assert_eq!(back.uri, "rvagent://minimal");
}

/// Edge case: McpError conversion from serde_json::Error.
#[test]
fn edge_mcp_error_from_json() {
    let bad: std::result::Result<serde_json::Value, _> = serde_json::from_str("{invalid json");
    let mcp_err: McpError = bad.unwrap_err().into();
    assert!(matches!(mcp_err, McpError::Json(_)));
    assert!(!mcp_err.to_string().is_empty());
}

/// Edge case: McpError all variants display correctly.
#[test]
fn edge_mcp_error_display_all_variants() {
    let variants: Vec<McpError> = vec![
        McpError::protocol("protocol fail"),
        McpError::tool("tool fail"),
        McpError::resource("resource fail"),
        McpError::transport("transport fail"),
        McpError::server("server fail"),
        McpError::client("client fail"),
    ];
    for e in &variants {
        let s = e.to_string();
        assert!(!s.is_empty());
        assert!(s.contains("fail"));
    }
}

/// Stress: Initialize params serde at scale.
#[test]
fn stress_initialize_params_serde() {
    for i in 0..200 {
        let params = InitializeParams {
            protocol_version: "2024-11-05".into(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: format!("client-{}", i),
                version: format!("{}.0.0", i),
            },
        };
        let json = serde_json::to_string(&params).unwrap();
        let back: InitializeParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.client_info.name, format!("client-{}", i));
    }
}

/// Stress: McpPrompt with many arguments serde roundtrip.
#[test]
fn stress_prompt_many_arguments() {
    let args: Vec<PromptArgument> = (0..50)
        .map(|i| PromptArgument {
            name: format!("arg-{}", i),
            description: Some(format!("Argument number {}", i)),
            required: i % 2 == 0,
        })
        .collect();

    let prompt = McpPrompt {
        name: "big-prompt".into(),
        description: Some("A prompt with many arguments".into()),
        arguments: args,
    };

    let json = serde_json::to_string(&prompt).unwrap();
    let back: McpPrompt = serde_json::from_str(&json).unwrap();
    assert_eq!(back.arguments.len(), 50);
    assert!(back.arguments[0].required);
    assert!(!back.arguments[1].required);
}

/// Stress: Registry clone shares state via Arc.
#[test]
fn stress_registry_clone_shared_state() {
    let reg = McpToolRegistry::new();
    for i in 0..100 {
        reg.register_tool(make_tool(&format!("shared-{}", i))).unwrap();
    }

    let reg2 = reg.clone();
    assert_eq!(reg2.len(), 100);

    // Modifications through clone are visible in original (shared Arc)
    reg2.register_tool(make_tool("from-clone")).unwrap();
    assert!(reg.get_tool("from-clone").is_some());
    assert_eq!(reg.len(), 101);
}
