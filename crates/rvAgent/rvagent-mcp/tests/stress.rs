//! Stress and property-based tests for rvagent-mcp.
//! Tests topology scaling, concurrent access patterns, and edge cases.

use std::sync::Arc;

use rvagent_mcp::protocol::*;
use rvagent_mcp::registry::*;
use rvagent_mcp::resources::*;
use rvagent_mcp::skills_bridge::*;
use rvagent_mcp::topology::*;
use rvagent_mcp::McpError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
// Stress: Topology scaling
// ---------------------------------------------------------------------------

/// Stress test: Scale topology to 100 nodes.
#[test]
fn stress_topology_100_nodes() {
    let mut router = TopologyRouter::mesh(200);
    for i in 0..100 {
        router.add_node(TopologyNode {
            id: format!("node-{}", i),
            role: if i == 0 {
                NodeRole::Queen
            } else {
                NodeRole::Worker
            },
            status: match i % 4 {
                0 => NodeStatus::Active,
                1 => NodeStatus::Idle,
                2 => NodeStatus::Busy,
                _ => NodeStatus::Active,
            },
            tools: vec![format!("tool-{}", i % 10)],
            connections: if i > 0 {
                vec![format!("node-{}", i - 1)]
            } else {
                vec![]
            },
        });
    }
    assert_eq!(router.node_count(), 100);

    // Routing should still work
    for i in 0..10 {
        let tool = format!("tool-{}", i);
        let target = router.route_tool_call(&tool);
        assert!(target.is_some(), "Should find node for {}", tool);
    }

    // Status should be valid JSON
    let status = router.status();
    let nodes = status.get("nodes").unwrap().as_array().unwrap();
    assert_eq!(nodes.len(), 100);
}

/// Stress test: Rapid add/remove nodes.
#[test]
fn stress_topology_churn() {
    let mut router = TopologyRouter::adaptive(50);

    // Add 50 nodes
    for i in 0..50 {
        router.add_node(TopologyNode {
            id: format!("churn-{}", i),
            role: NodeRole::Worker,
            status: NodeStatus::Active,
            tools: vec!["read_file".into()],
            connections: vec![],
        });
    }
    assert_eq!(router.node_count(), 50);

    // Remove every other node
    for i in (0..50).step_by(2) {
        router.remove_node(&format!("churn-{}", i));
    }
    assert_eq!(router.node_count(), 25);

    // Routing should still work with remaining nodes
    assert!(router.route_tool_call("read_file").is_some());
}

/// Stress test: All nodes failed.
#[test]
fn stress_all_nodes_failed() {
    let mut router = TopologyRouter::hierarchical(10);
    for i in 0..5 {
        router.add_node(TopologyNode {
            id: format!("fail-{}", i),
            role: NodeRole::Worker,
            status: NodeStatus::Failed,
            tools: vec!["grep".into()],
            connections: vec![],
        });
    }

    // Should return None since all nodes are failed
    assert!(router.route_tool_call("grep").is_none());
    assert_eq!(router.active_nodes().len(), 0);
}

// ---------------------------------------------------------------------------
// Property: Resource URIs and providers
// ---------------------------------------------------------------------------

/// Property: Static resource URIs are consistent after add/list roundtrip.
#[tokio::test]
async fn property_resource_uris_valid() {
    let provider = StaticResourceProvider::new();
    provider.add(
        "rvagent://state/overview",
        "overview",
        "state data",
        Some("application/json"),
        Some("Agent state overview"),
    );
    provider.add(
        "rvagent://skills/catalog",
        "catalog",
        "skills list",
        Some("application/json"),
        Some("Available skills"),
    );
    provider.add(
        "rvagent://topology/status",
        "status",
        "topology info",
        Some("application/json"),
        Some("Topology status"),
    );

    for resource in provider.list().await.unwrap() {
        assert!(
            resource.uri.starts_with("rvagent://"),
            "URI must use rvagent:// scheme: {}",
            resource.uri
        );
        assert!(!resource.name.is_empty());
        assert!(resource.description.is_some());
    }
}

/// Property: All built-in tools have valid schemas.
#[test]
fn property_tool_schemas_valid() {
    let registry = McpToolRegistry::new();
    register_builtins(&registry, serde_json::json!({"tools": true})).unwrap();
    let tools = registry.list_tools();
    assert_eq!(tools.len(), 3);
    for tool in &tools {
        assert!(!tool.name.is_empty());
        assert!(!tool.description.is_empty());
        assert!(
            tool.input_schema.is_object(),
            "Schema for {} must be object",
            tool.name
        );
    }
}

/// Property: Topology status JSON is well-formed.
#[test]
fn property_topology_status_shape() {
    let topologies: Vec<(&str, TopologyRouter)> = vec![
        ("standalone", TopologyRouter::standalone()),
        ("hierarchical", TopologyRouter::hierarchical(8)),
        ("mesh", TopologyRouter::mesh(8)),
        ("adaptive", TopologyRouter::adaptive(8)),
    ];

    for (name, router) in &topologies {
        let status = router.status();
        assert!(
            status.get("topology").is_some(),
            "{} missing topology",
            name
        );
        assert!(
            status.get("max_agents").is_some(),
            "{} missing max_agents",
            name
        );
        assert!(
            status.get("node_count").is_some(),
            "{} missing node_count",
            name
        );
        assert!(
            status.get("active_nodes").is_some(),
            "{} missing active_nodes",
            name
        );
        assert!(
            status.get("consensus").is_some(),
            "{} missing consensus",
            name
        );
        assert!(status.get("nodes").is_some(), "{} missing nodes", name);
    }
}

/// Property: Skills conversion is idempotent.
#[test]
fn property_skills_roundtrip_idempotent() {
    let original = rvagent_middleware::skills::SkillMetadata {
        path: ".skills/test/SKILL.md".into(),
        name: "test-skill".into(),
        description: "A test skill for roundtrip testing".into(),
        license: Some("MIT".into()),
        compatibility: Some("claude-code".into()),
        metadata: std::collections::HashMap::new(),
        allowed_tools: vec!["read_file".into(), "write_file".into()],
    };

    // rvAgent -> Claude Code -> rvAgent
    let claude = SkillBridge::to_claude_code(&original);
    let back = SkillBridge::from_claude_code(&claude);
    assert_eq!(back.name, original.name);
    assert_eq!(back.description, original.description);
    assert_eq!(back.allowed_tools, original.allowed_tools);

    // rvAgent -> Codex -> rvAgent
    let codex = SkillBridge::to_codex(&original);
    let back2 = SkillBridge::from_codex(&codex);
    assert_eq!(back2.name, original.name);
    assert_eq!(back2.allowed_tools, original.allowed_tools);
}

// ---------------------------------------------------------------------------
// Stress: Protocol serde throughput
// ---------------------------------------------------------------------------

/// Stress: Serialize/deserialize many MCP requests.
#[test]
fn stress_mcp_serde_throughput() {
    let req = JsonRpcRequest::new(1, "tools/call").with_params(
        serde_json::json!({"name": "read_file", "arguments": {"file_path": "/test.txt"}}),
    );

    for i in 0..1000 {
        let mut r = req.clone();
        r.id = serde_json::json!(i);
        let json = serde_json::to_string(&r).unwrap();
        let back: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.method, "tools/call");
    }
}

/// Stress: Create and query many resources.
#[tokio::test]
async fn stress_resource_reads() {
    let provider = StaticResourceProvider::new();
    provider.add(
        "rvagent://state/overview",
        "overview",
        "{}",
        Some("application/json"),
        None,
    );
    provider.add(
        "rvagent://skills/catalog",
        "catalog",
        "[]",
        Some("application/json"),
        None,
    );
    provider.add(
        "rvagent://topology/status",
        "status",
        "{}",
        Some("application/json"),
        None,
    );

    // Read all static resources many times
    for _ in 0..100 {
        assert!(provider.read("rvagent://state/overview").await.is_ok());
        assert!(provider.read("rvagent://skills/catalog").await.is_ok());
        assert!(provider.read("rvagent://topology/status").await.is_ok());
    }

    // Non-existent resources
    assert!(provider.read("rvagent://nonexistent").await.is_err());
}

/// Stress: TopologyNode serde roundtrip at scale.
#[test]
fn stress_node_serde_roundtrip() {
    for i in 0..500 {
        let node = TopologyNode {
            id: format!("node-{}", i),
            role: match i % 5 {
                0 => NodeRole::Queen,
                1 => NodeRole::Worker,
                2 => NodeRole::Scout,
                3 => NodeRole::Specialist,
                _ => NodeRole::Router,
            },
            status: match i % 5 {
                0 => NodeStatus::Active,
                1 => NodeStatus::Idle,
                2 => NodeStatus::Busy,
                3 => NodeStatus::Failed,
                _ => NodeStatus::Draining,
            },
            tools: (0..3).map(|j| format!("tool-{}", j)).collect(),
            connections: (0..2).map(|j| format!("conn-{}", j)).collect(),
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: TopologyNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, node.id);
        assert_eq!(back.role, node.role);
        assert_eq!(back.status, node.status);
    }
}

/// Stress: Registry register/unregister churn.
#[test]
fn stress_registry_churn() {
    let reg = McpToolRegistry::new();

    for i in 0..100 {
        reg.register_tool(make_tool(&format!("churn-{}", i)))
            .unwrap();
    }
    assert_eq!(reg.len(), 100);

    for i in (0..100).step_by(2) {
        reg.unregister_tool(&format!("churn-{}", i)).unwrap();
    }
    assert_eq!(reg.len(), 50);

    for i in (1..100).step_by(2) {
        assert!(reg.get_tool(&format!("churn-{}", i)).is_some());
    }
}

/// Stress: Call builtins many times.
#[tokio::test]
async fn stress_builtins_repeated_calls() {
    let reg = McpToolRegistry::new();
    register_builtins(&reg, serde_json::json!({"tools": true})).unwrap();

    for _ in 0..100 {
        let ping = reg
            .call_tool("ping", serde_json::Value::Null)
            .await
            .unwrap();
        assert!(!ping.is_error);

        let echo = reg
            .call_tool("echo", serde_json::json!({"text": "hello"}))
            .await
            .unwrap();
        match &echo.content[0] {
            Content::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("expected text content"),
        }
    }
}

/// Stress: Registry clone shares state via Arc.
#[test]
fn stress_registry_clone_shared_state() {
    let reg = McpToolRegistry::new();
    for i in 0..100 {
        reg.register_tool(make_tool(&format!("shared-{}", i)))
            .unwrap();
    }
    let reg2 = reg.clone();
    assert_eq!(reg2.len(), 100);
    reg2.register_tool(make_tool("from-clone")).unwrap();
    assert!(reg.get_tool("from-clone").is_some());
    assert_eq!(reg.len(), 101);
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

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Edge case: Empty tool name routing.
#[test]
fn edge_empty_tool_name() {
    let mut router = TopologyRouter::hierarchical(4);
    router.add_node(TopologyNode {
        id: "q".into(),
        role: NodeRole::Queen,
        status: NodeStatus::Active,
        tools: vec![],
        connections: vec![],
    });
    // Empty tool name should still not panic
    let _ = router.route_tool_call("");
}

/// Edge case: Very long tool/node names.
#[test]
fn edge_long_names() {
    let mut router = TopologyRouter::mesh(2);
    let long_id = "a".repeat(1000);
    let long_tool = "b".repeat(1000);
    router.add_node(TopologyNode {
        id: long_id.clone(),
        role: NodeRole::Worker,
        status: NodeStatus::Active,
        tools: vec![long_tool.clone()],
        connections: vec![],
    });
    assert_eq!(router.route_tool_call(&long_tool), Some(long_id));
}

/// Edge case: Duplicate node IDs.
#[test]
fn edge_duplicate_node_id() {
    let mut router = TopologyRouter::hierarchical(4);
    router.add_node(TopologyNode {
        id: "dup".into(),
        role: NodeRole::Worker,
        status: NodeStatus::Active,
        tools: vec!["ls".into()],
        connections: vec![],
    });
    // Adding same ID should overwrite
    router.add_node(TopologyNode {
        id: "dup".into(),
        role: NodeRole::Specialist,
        status: NodeStatus::Idle,
        tools: vec!["grep".into()],
        connections: vec![],
    });
    assert_eq!(router.node_count(), 1);
    let node = router.get_node("dup").unwrap();
    assert_eq!(node.role, NodeRole::Specialist);
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

/// Edge case: McpError from serde_json::Error.
#[test]
fn edge_mcp_error_from_json() {
    let bad: std::result::Result<serde_json::Value, _> = serde_json::from_str("{invalid");
    let mcp_err: McpError = bad.unwrap_err().into();
    assert!(matches!(mcp_err, McpError::Json(_)));
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

/// Edge case: Duplicate tool registration returns error.
#[test]
fn edge_duplicate_tool_registration() {
    let reg = McpToolRegistry::new();
    reg.register_tool(make_tool("dup")).unwrap();
    let result = reg.register_tool(make_tool("dup"));
    assert!(result.is_err());
    assert_eq!(reg.len(), 1);
}

/// Edge case: Call non-existent tool.
#[tokio::test]
async fn edge_call_nonexistent_tool() {
    let reg = McpToolRegistry::new();
    let result = reg
        .call_tool("does-not-exist", serde_json::Value::Null)
        .await;
    assert!(result.is_err());
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
        assert_eq!(
            parsed.as_ref(),
            Some(method),
            "Failed roundtrip for {:?}",
            method
        );
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
        let json = serde_json::to_string(&err).unwrap();
        let back: JsonRpcError = serde_json::from_str(&json).unwrap();
        assert_eq!(back.code, *expected_code);
    }
}

/// Property: validate_args is consistent for all registered tools.
#[test]
fn property_validate_args_consistency() {
    let reg = McpToolRegistry::new();
    reg.register_tool(make_tool_with_schema(
        "obj-tool",
        serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}),
    )).unwrap();
    reg.register_tool(make_tool_with_schema(
        "no-req",
        serde_json::json!({"type": "object", "properties": {"b": {"type": "number"}}}),
    ))
    .unwrap();

    assert!(reg
        .validate_args("obj-tool", &serde_json::json!({}))
        .is_err());
    assert!(reg
        .validate_args("obj-tool", &serde_json::json!({"a": "val"}))
        .is_ok());
    assert!(reg.validate_args("no-req", &serde_json::json!({})).is_ok());
    assert!(reg
        .validate_args("obj-tool", &serde_json::json!("string"))
        .is_err());
}

/// Property: McpToolDefinition clone preserves all fields.
#[test]
fn property_tool_definition_clone_preserves_fields() {
    for i in 0..50 {
        let original = make_tool_with_schema(
            &format!("clone-test-{}", i),
            serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}),
        );
        let cloned = original.clone();
        assert_eq!(cloned.name, original.name);
        assert_eq!(cloned.description, original.description);
        assert_eq!(cloned.input_schema, original.input_schema);
    }
}

/// Stress: ResourceRegistry with multiple providers.
#[tokio::test]
async fn stress_resource_registry_multiple_providers() {
    let mut registry = ResourceRegistry::new();
    for i in 0..10 {
        let provider = Arc::new(StaticResourceProvider::new());
        for j in 0..10 {
            provider.add(
                &format!("memory://provider-{}/resource-{}", i, j),
                &format!("resource-{}-{}", i, j),
                &format!("content for {}-{}", i, j),
                None,
                None,
            );
        }
        registry.register(provider);
    }
    assert_eq!(registry.provider_count(), 10);

    let all = registry.list_resources().await.unwrap();
    assert_eq!(all.len(), 100);

    // Read from various providers
    for i in 0..10 {
        let uri = format!("memory://provider-{}/resource-0", i);
        let result = registry.read_resource(&uri).await.unwrap();
        assert_eq!(result.contents.len(), 1);
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
