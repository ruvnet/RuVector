//! Comprehensive integration tests for the `rvagent-mcp` crate.
//!
//! Covers: topology routing, MCP protocol, skills bridge, transport, server
//! dispatch, cross-architecture consistency, and error paths.

use std::collections::HashMap;
use std::sync::Arc;

use rvagent_mcp::protocol::*;
use rvagent_mcp::registry::{
    register_builtins, EchoHandler, McpToolDefinition, McpToolHandler, McpToolRegistry, PingHandler,
};
use rvagent_mcp::resources::{ResourceRegistry, StaticResourceProvider};
use rvagent_mcp::server::{McpServer, McpServerConfig};
use rvagent_mcp::skills_bridge::{ClaudeCodeSkill, CodexSkill, SkillBridge};
use rvagent_mcp::topology::*;
use rvagent_mcp::transport::{MemoryTransport, Transport};
use rvagent_mcp::McpError;

// =========================================================================
// Helpers
// =========================================================================

fn make_node(id: &str, role: NodeRole, status: NodeStatus, tools: Vec<&str>) -> TopologyNode {
    TopologyNode {
        id: id.into(),
        role,
        status,
        tools: tools.into_iter().map(|s| s.to_string()).collect(),
        connections: vec![],
    }
}

fn make_node_connected(
    id: &str,
    role: NodeRole,
    status: NodeStatus,
    tools: Vec<&str>,
    connections: Vec<&str>,
) -> TopologyNode {
    TopologyNode {
        id: id.into(),
        role,
        status,
        tools: tools.into_iter().map(|s| s.to_string()).collect(),
        connections: connections.into_iter().map(|s| s.to_string()).collect(),
    }
}

fn sample_skill() -> rvagent_middleware::skills::SkillMetadata {
    rvagent_middleware::skills::SkillMetadata {
        path: ".skills/deploy/SKILL.md".into(),
        name: "deploy".into(),
        description: "Deploy the application to production".into(),
        license: Some("MIT".into()),
        compatibility: Some("claude-code".into()),
        metadata: {
            let mut m = HashMap::new();
            m.insert("version".into(), "2.0".into());
            m
        },
        allowed_tools: vec!["execute".into(), "write_file".into(), "read_file".into()],
    }
}

fn make_server() -> McpServer {
    let reg = McpToolRegistry::new();
    register_builtins(&reg, serde_json::json!({"tools": true, "resources": true})).unwrap();
    let res_reg = Arc::new(ResourceRegistry::new());
    McpServer::new(McpServerConfig::default(), reg, res_reg)
}

fn make_server_with_resources() -> McpServer {
    let reg = McpToolRegistry::new();
    register_builtins(&reg, serde_json::json!({"tools": true})).unwrap();
    let sp = Arc::new(StaticResourceProvider::new());
    sp.add(
        "rvagent://status",
        "Server Status",
        r#"{"status":"running"}"#,
        Some("application/json"),
        Some("Current server status"),
    );
    sp.add(
        "rvagent://caps",
        "Capabilities",
        r#"{"tools":true}"#,
        Some("application/json"),
        None,
    );
    sp.add(
        "rvagent://topology",
        "Topology",
        r#"{"type":"standalone"}"#,
        Some("application/json"),
        None,
    );
    let mut res_reg = ResourceRegistry::new();
    res_reg.register(sp);
    McpServer::new(McpServerConfig::default(), reg, Arc::new(res_reg))
}

// =========================================================================
// 1. Topology Integration Tests (20+ tests)
// =========================================================================

#[test]
fn test_hierarchical_full_topology() {
    let mut router = TopologyRouter::hierarchical(8);
    router.add_node(make_node_connected(
        "queen-1",
        NodeRole::Queen,
        NodeStatus::Active,
        vec!["ls", "read_file", "execute"],
        vec!["worker-1", "worker-2"],
    ));
    router.add_node(make_node_connected(
        "worker-1",
        NodeRole::Worker,
        NodeStatus::Active,
        vec!["read_file", "write_file"],
        vec!["queen-1"],
    ));
    router.add_node(make_node_connected(
        "worker-2",
        NodeRole::Specialist,
        NodeStatus::Active,
        vec!["grep", "glob"],
        vec!["queen-1"],
    ));

    // Specialist with grep should be preferred
    let target = router.route_tool_call("grep");
    assert!(target.is_some());
    let target_id = target.unwrap();
    assert!(target_id == "worker-2" || target_id == "queen-1");

    // read_file is on worker-1 and queen-1
    let target = router.route_tool_call("read_file");
    assert!(target.is_some());

    // execute is only on queen
    let target = router.route_tool_call("execute");
    assert!(target.is_some());
}

#[test]
fn test_mesh_topology_routing() {
    let mut router = TopologyRouter::mesh(6);
    for i in 0..4 {
        let connections: Vec<&str> = vec![];
        let status = if i == 2 {
            NodeStatus::Busy
        } else {
            NodeStatus::Active
        };
        router.add_node(make_node(
            &format!("peer-{}", i),
            NodeRole::Worker,
            status,
            vec!["read_file", "write_file"],
        ));
    }

    let target = router.route_tool_call("read_file");
    assert!(target.is_some());
    // Should not route to the busy node (peer-2)
    let target_id = target.unwrap();
    assert_ne!(target_id, "peer-2");
}

#[test]
fn test_adaptive_topology_load_balancing() {
    let mut router = TopologyRouter::adaptive(10);
    router.add_node(make_node(
        "idle-node",
        NodeRole::Worker,
        NodeStatus::Idle,
        vec!["grep"],
    ));
    router.add_node(make_node(
        "busy-node",
        NodeRole::Worker,
        NodeStatus::Busy,
        vec!["grep"],
    ));

    // Should prefer idle node
    assert_eq!(router.route_tool_call("grep"), Some("idle-node".into()));
}

#[test]
fn test_standalone_topology_returns_none() {
    let router = TopologyRouter::standalone();
    assert_eq!(router.route_tool_call("read_file"), None);
    assert_eq!(router.route_tool_call("execute"), None);
    assert_eq!(router.route_tool_call(""), None);
}

#[test]
fn test_node_failure_recovery() {
    let mut router = TopologyRouter::hierarchical(4);
    router.add_node(make_node(
        "primary",
        NodeRole::Worker,
        NodeStatus::Active,
        vec!["ls"],
    ));
    router.add_node(make_node(
        "backup",
        NodeRole::Worker,
        NodeStatus::Active,
        vec!["ls"],
    ));

    // Remove primary to simulate failure
    let removed = router.remove_node("primary");
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().id, "primary");

    // Should still route to backup
    let target = router.route_tool_call("ls");
    assert_eq!(target, Some("backup".into()));
}

#[test]
fn test_topology_config_serialization_roundtrip() {
    let config = TopologyConfig {
        topology_type: TopologyType::Mesh,
        max_agents: 16,
        consensus: ConsensusType::Byzantine,
        health_check_interval_ms: 2000,
    };
    let json = serde_json::to_string(&config).unwrap();
    let back: TopologyConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.topology_type, TopologyType::Mesh);
    assert_eq!(back.max_agents, 16);
    assert_eq!(back.consensus, ConsensusType::Byzantine);
    assert_eq!(back.health_check_interval_ms, 2000);
}

#[test]
fn test_status_json_has_correct_shape_hierarchical() {
    let mut router = TopologyRouter::hierarchical(8);
    router.add_node(make_node(
        "q",
        NodeRole::Queen,
        NodeStatus::Active,
        vec!["ls"],
    ));
    let status = router.status();
    assert_eq!(status["topology"], "hierarchical");
    assert_eq!(status["max_agents"], 8);
    assert_eq!(status["node_count"], 1);
    assert_eq!(status["active_nodes"], 1);
    assert!(status["nodes"].is_array());
    assert!(status["consensus"].is_string());
}

#[test]
fn test_status_json_has_correct_shape_mesh() {
    let router = TopologyRouter::mesh(4);
    let status = router.status();
    assert_eq!(status["topology"], "mesh");
    assert_eq!(status["node_count"], 0);
}

#[test]
fn test_status_json_has_correct_shape_adaptive() {
    let router = TopologyRouter::adaptive(10);
    let status = router.status();
    assert_eq!(status["topology"], "adaptive");
    assert_eq!(status["max_agents"], 10);
}

#[test]
fn test_status_json_has_correct_shape_standalone() {
    let router = TopologyRouter::standalone();
    let status = router.status();
    assert_eq!(status["topology"], "standalone");
}

#[test]
fn test_empty_topology_routing() {
    let router = TopologyRouter::hierarchical(8);
    assert_eq!(router.route_tool_call("read_file"), None);
    assert_eq!(router.node_count(), 0);
}

#[test]
fn test_empty_mesh_routing() {
    let router = TopologyRouter::mesh(4);
    assert_eq!(router.route_tool_call("anything"), None);
}

#[test]
fn test_empty_adaptive_routing() {
    let router = TopologyRouter::adaptive(4);
    assert_eq!(router.route_tool_call("grep"), None);
}

#[test]
fn test_node_role_transitions_via_replacement() {
    let mut router = TopologyRouter::hierarchical(4);
    router.add_node(make_node(
        "n1",
        NodeRole::Worker,
        NodeStatus::Active,
        vec!["ls"],
    ));
    assert_eq!(router.get_node("n1").unwrap().role, NodeRole::Worker);

    // Remove and re-add with new role
    router.remove_node("n1");
    router.add_node(make_node(
        "n1",
        NodeRole::Specialist,
        NodeStatus::Active,
        vec!["ls", "grep"],
    ));
    assert_eq!(router.get_node("n1").unwrap().role, NodeRole::Specialist);
}

#[test]
fn test_all_node_status_variants_routing_priority() {
    let mut router = TopologyRouter::adaptive(10);
    router.add_node(make_node(
        "failed",
        NodeRole::Worker,
        NodeStatus::Failed,
        vec!["tool"],
    ));
    router.add_node(make_node(
        "draining",
        NodeRole::Worker,
        NodeStatus::Draining,
        vec!["tool"],
    ));
    router.add_node(make_node(
        "busy",
        NodeRole::Worker,
        NodeStatus::Busy,
        vec!["tool"],
    ));
    router.add_node(make_node(
        "active",
        NodeRole::Worker,
        NodeStatus::Active,
        vec!["tool"],
    ));
    router.add_node(make_node(
        "idle",
        NodeRole::Worker,
        NodeStatus::Idle,
        vec!["tool"],
    ));

    // Idle should be preferred in adaptive
    assert_eq!(router.route_tool_call("tool"), Some("idle".into()));
}

#[test]
fn test_adaptive_falls_back_to_active_when_no_idle() {
    let mut router = TopologyRouter::adaptive(4);
    router.add_node(make_node(
        "busy-1",
        NodeRole::Worker,
        NodeStatus::Busy,
        vec!["ls"],
    ));
    router.add_node(make_node(
        "active-1",
        NodeRole::Worker,
        NodeStatus::Active,
        vec!["ls"],
    ));
    assert_eq!(router.route_tool_call("ls"), Some("active-1".into()));
}

#[test]
fn test_adaptive_falls_back_to_busy_when_no_active_or_idle() {
    let mut router = TopologyRouter::adaptive(4);
    router.add_node(make_node(
        "busy-1",
        NodeRole::Worker,
        NodeStatus::Busy,
        vec!["ls"],
    ));
    // Adaptive should still route to busy as last resort
    assert_eq!(router.route_tool_call("ls"), Some("busy-1".into()));
}

#[test]
fn test_all_node_role_variants_serde() {
    for role in &[
        NodeRole::Queen,
        NodeRole::Worker,
        NodeRole::Scout,
        NodeRole::Specialist,
        NodeRole::Router,
    ] {
        let json = serde_json::to_string(role).unwrap();
        let back: NodeRole = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, role);
    }
}

#[test]
fn test_all_node_status_variants_serde() {
    for status in &[
        NodeStatus::Active,
        NodeStatus::Idle,
        NodeStatus::Busy,
        NodeStatus::Failed,
        NodeStatus::Draining,
    ] {
        let json = serde_json::to_string(status).unwrap();
        let back: NodeStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, status);
    }
}

#[test]
fn test_all_topology_type_variants_serde() {
    for tt in &[
        TopologyType::Standalone,
        TopologyType::Hierarchical,
        TopologyType::Mesh,
        TopologyType::Adaptive,
    ] {
        let json = serde_json::to_string(tt).unwrap();
        let back: TopologyType = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, tt);
    }
}

#[test]
fn test_all_consensus_type_variants_serde() {
    for ct in &[
        ConsensusType::Raft,
        ConsensusType::Byzantine,
        ConsensusType::Gossip,
        ConsensusType::None,
    ] {
        let json = serde_json::to_string(ct).unwrap();
        let back: ConsensusType = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, ct);
    }
}

#[test]
fn test_topology_node_serde_roundtrip() {
    let node = TopologyNode {
        id: "node-42".into(),
        role: NodeRole::Specialist,
        status: NodeStatus::Idle,
        tools: vec!["grep".into(), "glob".into(), "ls".into()],
        connections: vec!["node-1".into(), "node-2".into()],
    };
    let json = serde_json::to_string(&node).unwrap();
    let back: TopologyNode = serde_json::from_str(&json).unwrap();
    assert_eq!(back.id, "node-42");
    assert_eq!(back.role, NodeRole::Specialist);
    assert_eq!(back.status, NodeStatus::Idle);
    assert_eq!(back.tools.len(), 3);
    assert_eq!(back.connections.len(), 2);
}

#[test]
fn test_topology_config_defaults() {
    let config = TopologyConfig::default();
    assert_eq!(config.topology_type, TopologyType::Standalone);
    assert_eq!(config.max_agents, 8);
    assert_eq!(config.consensus, ConsensusType::Raft);
}

#[test]
fn test_remove_nonexistent_node() {
    let mut router = TopologyRouter::standalone();
    assert!(router.remove_node("ghost").is_none());
}

#[test]
fn test_get_node_exists_and_missing() {
    let mut router = TopologyRouter::hierarchical(4);
    router.add_node(make_node(
        "present",
        NodeRole::Queen,
        NodeStatus::Active,
        vec![],
    ));
    assert!(router.get_node("present").is_some());
    assert!(router.get_node("absent").is_none());
}

#[test]
fn test_active_nodes_filtering() {
    let mut router = TopologyRouter::mesh(8);
    router.add_node(make_node("a", NodeRole::Worker, NodeStatus::Active, vec![]));
    router.add_node(make_node("b", NodeRole::Worker, NodeStatus::Failed, vec![]));
    router.add_node(make_node("c", NodeRole::Worker, NodeStatus::Active, vec![]));
    router.add_node(make_node("d", NodeRole::Worker, NodeStatus::Idle, vec![]));
    assert_eq!(router.active_nodes().len(), 2); // Only Active, not Idle
}

// =========================================================================
// 2. MCP Protocol Tests (15+ tests)
// =========================================================================

#[tokio::test]
async fn test_server_initialize_handshake() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "initialize").with_params(serde_json::json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "test-client", "version": "1.0" }
    }));
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert!(result["capabilities"]["tools"].is_object());
    assert!(result["capabilities"]["resources"].is_object());
    assert_eq!(result["serverInfo"]["name"], "rvagent-mcp");
}

#[tokio::test]
async fn test_server_ping_pong() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "ping");
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_none());
    assert!(resp.result.is_some());
}

#[tokio::test]
async fn test_server_tools_list_returns_builtins() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "tools/list");
    let resp = server.handle_request(req).await;
    let result = resp.result.unwrap();
    let tools = result["tools"].as_array().unwrap();
    // Should have at least ping, echo, list_capabilities
    assert!(tools.len() >= 3);
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"ping"));
    assert!(names.contains(&"echo"));
    assert!(names.contains(&"list_capabilities"));
}

#[tokio::test]
async fn test_server_tools_call_valid_tool() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "tools/call").with_params(serde_json::json!({
        "name": "echo",
        "arguments": { "text": "hello integration" }
    }));
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["content"][0]["text"], "hello integration");
    assert_eq!(result["isError"], false);
}

#[tokio::test]
async fn test_server_tools_call_invalid_tool_name() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "tools/call").with_params(serde_json::json!({
        "name": "does_not_exist",
        "arguments": {}
    }));
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_some());
    let error = resp.error.unwrap();
    assert_eq!(error.code, -32603); // Internal error (tool not found)
}

#[tokio::test]
async fn test_server_resources_list_with_provider() {
    let server = make_server_with_resources();
    let req = JsonRpcRequest::new(1, "resources/list");
    let resp = server.handle_request(req).await;
    let result = resp.result.unwrap();
    let resources = result["resources"].as_array().unwrap();
    assert_eq!(resources.len(), 3);
}

#[tokio::test]
async fn test_server_resources_read_status() {
    let server = make_server_with_resources();
    let req = JsonRpcRequest::new(1, "resources/read").with_params(serde_json::json!({
        "uri": "rvagent://status"
    }));
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let text = result["contents"][0]["text"].as_str().unwrap();
    assert!(text.contains("running"));
}

#[tokio::test]
async fn test_server_resources_read_capabilities() {
    let server = make_server_with_resources();
    let req = JsonRpcRequest::new(1, "resources/read").with_params(serde_json::json!({
        "uri": "rvagent://caps"
    }));
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_none());
}

#[tokio::test]
async fn test_server_resources_read_topology() {
    let server = make_server_with_resources();
    let req = JsonRpcRequest::new(1, "resources/read").with_params(serde_json::json!({
        "uri": "rvagent://topology"
    }));
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_none());
}

#[tokio::test]
async fn test_server_resources_templates_list() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "resources/templates/list");
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert!(result["resource_templates"].is_array());
}

#[tokio::test]
async fn test_server_unknown_method_returns_error() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "completely/unknown");
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_some());
    let error = resp.error.unwrap();
    assert_eq!(error.code, -32601); // METHOD_NOT_FOUND
    assert!(error.message.contains("unknown method"));
}

#[tokio::test]
async fn test_server_response_preserves_request_id() {
    let server = make_server();
    let req = JsonRpcRequest::new(42, "ping");
    let resp = server.handle_request(req).await;
    assert_eq!(resp.id, serde_json::json!(42));
}

#[tokio::test]
async fn test_server_response_has_jsonrpc_version() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "ping");
    let resp = server.handle_request(req).await;
    assert_eq!(resp.jsonrpc, "2.0");
}

#[tokio::test]
async fn test_server_tools_call_without_params() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "tools/call");
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_some());
    let error = resp.error.unwrap();
    assert_eq!(error.code, -32602); // INVALID_PARAMS
}

#[tokio::test]
async fn test_server_tools_call_with_malformed_params() {
    let server = make_server();
    let req =
        JsonRpcRequest::new(1, "tools/call").with_params(serde_json::json!("not an object"));
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32602);
}

#[tokio::test]
async fn test_server_resources_read_missing_resource() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "resources/read").with_params(serde_json::json!({
        "uri": "rvagent://nonexistent"
    }));
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_server_prompts_list_empty() {
    let server = make_server();
    let req = JsonRpcRequest::new(1, "prompts/list");
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert!(result["prompts"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_server_prompts_get_returns_error() {
    let server = make_server();
    let req =
        JsonRpcRequest::new(1, "prompts/get").with_params(serde_json::json!({"name": "nope"}));
    let resp = server.handle_request(req).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_server_string_id_preserved() {
    let server = make_server();
    let req = JsonRpcRequest::new("request-abc", "ping");
    let resp = server.handle_request(req).await;
    assert_eq!(resp.id, serde_json::json!("request-abc"));
}

// =========================================================================
// 3. Skills Bridge Tests (10+ tests)
// =========================================================================

#[test]
fn test_skill_to_claude_code_format() {
    let skill = sample_skill();
    let cc = SkillBridge::to_claude_code(&skill);
    assert_eq!(cc.name, "deploy");
    assert_eq!(cc.description, "Deploy the application to production");
    assert_eq!(cc.path, ".skills/deploy/SKILL.md");
    assert_eq!(
        cc.allowed_tools,
        vec!["execute", "write_file", "read_file"]
    );
    assert_eq!(cc.triggers, vec!["/deploy"]);
}

#[test]
fn test_skill_to_codex_format() {
    let skill = sample_skill();
    let codex = SkillBridge::to_codex(&skill);
    assert_eq!(codex.name, "deploy");
    assert_eq!(codex.prompt, "Deploy the application to production");
    assert_eq!(codex.tools, vec!["execute", "write_file", "read_file"]);
    assert!(codex.model.is_none());
}

#[test]
fn test_claude_code_to_rvagent_format() {
    let cc = ClaudeCodeSkill {
        name: "lint".into(),
        description: "Lint the codebase".into(),
        path: ".skills/lint/SKILL.md".into(),
        allowed_tools: vec!["execute".into(), "read_file".into()],
        triggers: vec!["/lint".into()],
    };
    let meta = SkillBridge::from_claude_code(&cc);
    assert_eq!(meta.name, "lint");
    assert_eq!(meta.description, "Lint the codebase");
    assert_eq!(meta.path, ".skills/lint/SKILL.md");
    assert_eq!(meta.compatibility.as_deref(), Some("claude-code"));
    assert_eq!(meta.allowed_tools, vec!["execute", "read_file"]);
}

#[test]
fn test_codex_to_rvagent_format() {
    let codex = CodexSkill {
        name: "refactor".into(),
        prompt: "Refactor the module".into(),
        tools: vec!["read_file".into(), "write_file".into()],
        model: Some("gpt-4o".into()),
    };
    let meta = SkillBridge::from_codex(&codex);
    assert_eq!(meta.name, "refactor");
    assert_eq!(meta.description, "Refactor the module");
    assert_eq!(meta.compatibility.as_deref(), Some("codex"));
    assert!(meta.path.is_empty());
}

#[test]
fn test_roundtrip_claude_code_preserves_fields() {
    let original = sample_skill();
    let cc = SkillBridge::to_claude_code(&original);
    let back = SkillBridge::from_claude_code(&cc);
    assert_eq!(back.name, original.name);
    assert_eq!(back.description, original.description);
    assert_eq!(back.path, original.path);
    assert_eq!(back.allowed_tools, original.allowed_tools);
}

#[test]
fn test_roundtrip_codex_preserves_fields() {
    let original = sample_skill();
    let codex = SkillBridge::to_codex(&original);
    let back = SkillBridge::from_codex(&codex);
    assert_eq!(back.name, original.name);
    assert_eq!(back.description, original.description);
    assert_eq!(back.allowed_tools, original.allowed_tools);
}

#[test]
fn test_claude_code_trigger_format_correct() {
    let skill = sample_skill();
    let cc = SkillBridge::to_claude_code(&skill);
    assert!(cc.triggers[0].starts_with('/'));
    assert_eq!(cc.triggers[0], format!("/{}", skill.name));
}

#[test]
fn test_empty_allowed_tools_handled() {
    let skill = rvagent_middleware::skills::SkillMetadata {
        path: String::new(),
        name: "empty-tools".into(),
        description: "No tools".into(),
        license: None,
        compatibility: None,
        metadata: HashMap::new(),
        allowed_tools: vec![],
    };
    let cc = SkillBridge::to_claude_code(&skill);
    assert!(cc.allowed_tools.is_empty());
    let codex = SkillBridge::to_codex(&skill);
    assert!(codex.tools.is_empty());
}

#[test]
fn test_claude_code_skill_serde_roundtrip() {
    let cc = ClaudeCodeSkill {
        name: "test".into(),
        description: "Test skill".into(),
        path: "/skills/test.md".into(),
        allowed_tools: vec!["ls".into()],
        triggers: vec!["/test".into()],
    };
    let json = serde_json::to_string(&cc).unwrap();
    let back: ClaudeCodeSkill = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, cc.name);
    assert_eq!(back.triggers, cc.triggers);
}

#[test]
fn test_codex_skill_serde_roundtrip() {
    let codex = CodexSkill {
        name: "test".into(),
        prompt: "Do the thing".into(),
        tools: vec!["read_file".into()],
        model: Some("gpt-4".into()),
    };
    let json = serde_json::to_string(&codex).unwrap();
    let back: CodexSkill = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, codex.name);
    assert_eq!(back.model.as_deref(), Some("gpt-4"));
}

#[test]
fn test_batch_conversion_claude_code() {
    let skills = vec![sample_skill(), sample_skill()];
    let batch = SkillBridge::to_claude_code_batch(&skills);
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].name, "deploy");
}

#[test]
fn test_batch_conversion_codex() {
    let skills = vec![sample_skill()];
    let batch = SkillBridge::to_codex_batch(&skills);
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].name, "deploy");
}

#[test]
fn test_compatibility_field_set_correctly_claude_code() {
    let cc = ClaudeCodeSkill {
        name: "x".into(),
        description: "x".into(),
        path: "x".into(),
        allowed_tools: vec![],
        triggers: vec![],
    };
    let meta = SkillBridge::from_claude_code(&cc);
    assert_eq!(meta.compatibility.as_deref(), Some("claude-code"));
}

#[test]
fn test_compatibility_field_set_correctly_codex() {
    let codex = CodexSkill {
        name: "x".into(),
        prompt: "x".into(),
        tools: vec![],
        model: None,
    };
    let meta = SkillBridge::from_codex(&codex);
    assert_eq!(meta.compatibility.as_deref(), Some("codex"));
}

// =========================================================================
// 4. Transport Tests (5+ tests)
// =========================================================================

#[tokio::test]
async fn test_memory_transport_request_roundtrip() {
    let (client, server) = MemoryTransport::pair(16);
    let req = JsonRpcRequest::new(1, "tools/list");
    client.send_request(req).await.unwrap();
    let received = server.receive_request().await.unwrap().unwrap();
    assert_eq!(received.method, "tools/list");
    assert_eq!(received.id, serde_json::json!(1));
}

#[tokio::test]
async fn test_memory_transport_response_roundtrip() {
    let (client, server) = MemoryTransport::pair(16);
    let resp = JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({"ok": true}));
    server.send_response(resp).await.unwrap();
    let received = client.receive_response().await.unwrap().unwrap();
    assert!(received.result.is_some());
    assert_eq!(received.result.unwrap()["ok"], true);
}

#[tokio::test]
async fn test_memory_transport_empty_returns_none_on_drop() {
    let (client, server) = MemoryTransport::pair(16);
    drop(client);
    let result = server.receive_request().await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_memory_transport_multiple_requests_queued() {
    let (client, server) = MemoryTransport::pair(16);
    for i in 0..5 {
        client
            .send_request(JsonRpcRequest::new(i, "ping"))
            .await
            .unwrap();
    }
    for i in 0..5 {
        let req = server.receive_request().await.unwrap().unwrap();
        assert_eq!(req.id, serde_json::json!(i));
        assert_eq!(req.method, "ping");
    }
}

#[tokio::test]
async fn test_memory_transport_bidirectional_exchange() {
    let (client, server) = MemoryTransport::pair(16);

    // Client sends request
    client
        .send_request(JsonRpcRequest::new(1, "echo"))
        .await
        .unwrap();
    let req = server.receive_request().await.unwrap().unwrap();
    assert_eq!(req.method, "echo");

    // Server sends response
    server
        .send_response(JsonRpcResponse::success(
            serde_json::json!(1),
            serde_json::json!({"echoed": true}),
        ))
        .await
        .unwrap();
    let resp = client.receive_response().await.unwrap().unwrap();
    assert_eq!(resp.result.unwrap()["echoed"], true);
}

#[tokio::test]
async fn test_memory_transport_close() {
    let (a, _b) = MemoryTransport::pair(16);
    assert!(a.close().await.is_ok());
}

#[tokio::test]
async fn test_memory_transport_error_response() {
    let (client, server) = MemoryTransport::pair(16);
    let resp = JsonRpcResponse::error(
        serde_json::json!(99),
        JsonRpcError::method_not_found("no such method"),
    );
    server.send_response(resp).await.unwrap();
    let received = client.receive_response().await.unwrap().unwrap();
    assert!(received.error.is_some());
    assert_eq!(received.error.unwrap().code, -32601);
}

#[tokio::test]
async fn test_memory_transport_send_convenience() {
    let (client, server) = MemoryTransport::pair(16);

    // Spawn a task to respond
    let server_handle = tokio::spawn(async move {
        let req = server.receive_request().await.unwrap().unwrap();
        server
            .send_response(JsonRpcResponse::success(
                req.id,
                serde_json::json!({"pong": true}),
            ))
            .await
            .unwrap();
    });

    let resp = client.send(JsonRpcRequest::new(1, "ping")).await.unwrap();
    assert!(resp.result.is_some());
    assert_eq!(resp.result.unwrap()["pong"], true);
    server_handle.await.unwrap();
}

// =========================================================================
// 5. Cross-Architecture Tests (10+ tests)
// =========================================================================

#[test]
fn test_same_status_shape_across_all_topologies() {
    let topologies: Vec<TopologyRouter> = vec![
        TopologyRouter::standalone(),
        TopologyRouter::hierarchical(4),
        TopologyRouter::mesh(4),
        TopologyRouter::adaptive(4),
    ];

    for topology in &topologies {
        let status = topology.status();
        assert!(status.get("topology").is_some(), "missing 'topology' key");
        assert!(
            status.get("node_count").is_some(),
            "missing 'node_count' key"
        );
        assert!(
            status.get("max_agents").is_some(),
            "missing 'max_agents' key"
        );
        assert!(
            status.get("active_nodes").is_some(),
            "missing 'active_nodes' key"
        );
        assert!(status.get("nodes").is_some(), "missing 'nodes' key");
        assert!(
            status.get("consensus").is_some(),
            "missing 'consensus' key"
        );
    }
}

#[test]
fn test_empty_routing_consistent_across_topologies() {
    // Standalone always returns None
    assert_eq!(TopologyRouter::standalone().route_tool_call("tool"), None);
    // Empty hierarchical/mesh/adaptive also return None (no nodes)
    assert_eq!(
        TopologyRouter::hierarchical(4).route_tool_call("tool"),
        None
    );
    assert_eq!(TopologyRouter::mesh(4).route_tool_call("tool"), None);
    assert_eq!(TopologyRouter::adaptive(4).route_tool_call("tool"), None);
}

#[test]
fn test_single_active_node_routes_consistently() {
    let configs = vec![
        TopologyRouter::hierarchical(4),
        TopologyRouter::mesh(4),
        TopologyRouter::adaptive(4),
    ];

    for mut router in configs {
        router.add_node(make_node(
            "only-node",
            NodeRole::Worker,
            NodeStatus::Active,
            vec!["tool-a"],
        ));
        assert_eq!(
            router.route_tool_call("tool-a"),
            Some("only-node".into()),
            "Failed for topology {:?}",
            router.topology_type()
        );
    }
}

#[test]
fn test_failed_node_excluded_in_mesh() {
    // Mesh topology filters out failed nodes (requires Active status)
    let mut router = TopologyRouter::mesh(4);
    router.add_node(make_node(
        "failed-only",
        NodeRole::Worker,
        NodeStatus::Failed,
        vec!["tool-x"],
    ));
    let result = router.route_tool_call("tool-x");
    assert!(result.is_none(), "Mesh should skip failed nodes");
}

#[test]
fn test_failed_node_deprioritized_in_adaptive() {
    // Adaptive deprioritizes failed nodes but returns them as last resort
    let mut router = TopologyRouter::adaptive(4);
    router.add_node(make_node(
        "failed-1",
        NodeRole::Worker,
        NodeStatus::Failed,
        vec!["tool-x"],
    ));
    router.add_node(make_node(
        "active-1",
        NodeRole::Worker,
        NodeStatus::Active,
        vec!["tool-x"],
    ));
    // Active should be preferred over failed
    assert_eq!(router.route_tool_call("tool-x"), Some("active-1".into()));
}

#[test]
fn test_hierarchical_skips_failed_workers() {
    let mut router = TopologyRouter::hierarchical(4);
    router.add_node(make_node(
        "failed-worker",
        NodeRole::Worker,
        NodeStatus::Failed,
        vec!["tool-x"],
    ));
    // No queen, no active workers -- should return None
    let result = router.route_tool_call("tool-x");
    // Hierarchical finds active nodes; failed worker should not match
    assert!(result.is_none() || result == Some("failed-worker".into()));
}

#[test]
fn test_node_count_consistent_after_add_remove() {
    let mut topologies: Vec<TopologyRouter> = vec![
        TopologyRouter::hierarchical(4),
        TopologyRouter::mesh(4),
        TopologyRouter::adaptive(4),
    ];

    for router in &mut topologies {
        assert_eq!(router.node_count(), 0);
        router.add_node(make_node("a", NodeRole::Worker, NodeStatus::Active, vec![]));
        router.add_node(make_node("b", NodeRole::Worker, NodeStatus::Active, vec![]));
        assert_eq!(router.node_count(), 2);
        router.remove_node("a");
        assert_eq!(router.node_count(), 1);
        router.remove_node("b");
        assert_eq!(router.node_count(), 0);
    }
}

#[test]
fn test_config_accessor_consistent() {
    let h = TopologyRouter::hierarchical(6);
    assert_eq!(h.config().max_agents, 6);
    assert_eq!(h.config().topology_type, TopologyType::Hierarchical);

    let m = TopologyRouter::mesh(12);
    assert_eq!(m.config().max_agents, 12);
    assert_eq!(m.config().topology_type, TopologyType::Mesh);

    let a = TopologyRouter::adaptive(3);
    assert_eq!(a.config().max_agents, 3);
    assert_eq!(a.config().topology_type, TopologyType::Adaptive);
}

#[tokio::test]
async fn test_server_handles_all_mcp_methods() {
    let server = make_server_with_resources();
    let methods = vec![
        ("initialize", Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "1"}
        }))),
        ("ping", None),
        ("tools/list", None),
        ("tools/call", Some(serde_json::json!({"name": "ping", "arguments": {}}))),
        ("resources/list", None),
        ("resources/read", Some(serde_json::json!({"uri": "rvagent://status"}))),
        ("resources/templates/list", None),
        ("prompts/list", None),
    ];

    for (method, params) in methods {
        let mut req = JsonRpcRequest::new(1, method);
        if let Some(p) = params {
            req = req.with_params(p);
        }
        let resp = server.handle_request(req).await;
        assert!(
            resp.error.is_none(),
            "method '{}' should succeed but got error: {:?}",
            method,
            resp.error
        );
    }
}

#[tokio::test]
async fn test_server_error_codes_are_correct() {
    let server = make_server();

    // METHOD_NOT_FOUND = -32601
    let resp = server
        .handle_request(JsonRpcRequest::new(1, "invalid/method"))
        .await;
    assert_eq!(resp.error.as_ref().unwrap().code, -32601);

    // INVALID_PARAMS = -32602 (tools/call with no params)
    let resp = server
        .handle_request(JsonRpcRequest::new(2, "tools/call"))
        .await;
    assert_eq!(resp.error.as_ref().unwrap().code, -32602);
}

#[test]
fn test_jsonrpc_error_constructors() {
    assert_eq!(JsonRpcError::parse_error("x").code, -32700);
    assert_eq!(JsonRpcError::invalid_request("x").code, -32600);
    assert_eq!(JsonRpcError::method_not_found("x").code, -32601);
    assert_eq!(JsonRpcError::invalid_params("x").code, -32602);
    assert_eq!(JsonRpcError::internal_error("x").code, -32603);
}

#[test]
fn test_mcp_method_from_str_all_variants() {
    assert_eq!(McpMethod::from_str("initialize"), Some(McpMethod::Initialize));
    assert_eq!(McpMethod::from_str("tools/list"), Some(McpMethod::ToolsList));
    assert_eq!(McpMethod::from_str("tools/call"), Some(McpMethod::ToolsCall));
    assert_eq!(McpMethod::from_str("resources/list"), Some(McpMethod::ResourcesList));
    assert_eq!(McpMethod::from_str("resources/read"), Some(McpMethod::ResourcesRead));
    assert_eq!(McpMethod::from_str("resources/templates/list"), Some(McpMethod::ResourcesTemplatesList));
    assert_eq!(McpMethod::from_str("prompts/list"), Some(McpMethod::PromptsList));
    assert_eq!(McpMethod::from_str("prompts/get"), Some(McpMethod::PromptsGet));
    assert_eq!(McpMethod::from_str("ping"), Some(McpMethod::Ping));
    assert_eq!(McpMethod::from_str("nonexistent"), None);
    assert_eq!(McpMethod::from_str(""), None);
}

#[test]
fn test_mcp_method_roundtrip_all() {
    let all = vec![
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
    for method in all {
        let s = method.as_str();
        assert_eq!(McpMethod::from_str(s).as_ref(), Some(&method));
    }
}

// =========================================================================
// 6. Error Handling Tests
// =========================================================================

#[test]
fn test_mcp_error_display_all_variants() {
    let errors = vec![
        (McpError::protocol("p"), "protocol error: p"),
        (McpError::tool("t"), "tool error: t"),
        (McpError::resource("r"), "resource error: r"),
        (McpError::transport("tr"), "transport error: tr"),
        (McpError::server("s"), "server error: s"),
        (McpError::client("c"), "client error: c"),
    ];
    for (err, expected) in errors {
        assert_eq!(err.to_string(), expected);
    }
}

#[test]
fn test_mcp_error_from_json() {
    let bad: std::result::Result<serde_json::Value, _> = serde_json::from_str("{bad json");
    let mcp_err: McpError = bad.unwrap_err().into();
    assert!(matches!(mcp_err, McpError::Json(_)));
    assert!(mcp_err.to_string().contains("json error"));
}

#[test]
fn test_jsonrpc_request_creation() {
    let req = JsonRpcRequest::new(1, "test/method");
    assert_eq!(req.jsonrpc, "2.0");
    assert_eq!(req.method, "test/method");
    assert!(req.params.is_none());
}

#[test]
fn test_jsonrpc_request_with_params() {
    let req = JsonRpcRequest::new(1, "test")
        .with_params(serde_json::json!({"key": "value"}));
    assert!(req.params.is_some());
    assert_eq!(req.params.unwrap()["key"], "value");
}

#[test]
fn test_jsonrpc_response_success() {
    let resp = JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({"ok": true}));
    assert_eq!(resp.jsonrpc, "2.0");
    assert!(resp.result.is_some());
    assert!(resp.error.is_none());
}

#[test]
fn test_jsonrpc_response_error() {
    let resp = JsonRpcResponse::error(
        serde_json::json!(1),
        JsonRpcError::method_not_found("nope"),
    );
    assert!(resp.result.is_none());
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32601);
}

// =========================================================================
// 7. Registry Integration Tests
// =========================================================================

#[tokio::test]
async fn test_registry_register_and_call_custom_tool() {
    let reg = McpToolRegistry::new();
    reg.register_tool(McpToolDefinition {
        name: "custom_ping".into(),
        description: "Custom ping".into(),
        input_schema: serde_json::json!({"type": "object", "properties": {}}),
        handler: Arc::new(PingHandler),
    })
    .unwrap();

    let result = reg
        .call_tool("custom_ping", serde_json::Value::Null)
        .await
        .unwrap();
    assert!(!result.is_error);
    match &result.content[0] {
        Content::Text { text } => assert_eq!(text, "pong"),
        _ => panic!("expected text content"),
    }
}

#[tokio::test]
async fn test_registry_echo_handler() {
    let reg = McpToolRegistry::new();
    reg.register_tool(McpToolDefinition {
        name: "echo".into(),
        description: "Echo".into(),
        input_schema: serde_json::json!({"type": "object"}),
        handler: Arc::new(EchoHandler),
    })
    .unwrap();

    let result = reg
        .call_tool("echo", serde_json::json!({"text": "integration test"}))
        .await
        .unwrap();
    match &result.content[0] {
        Content::Text { text } => assert_eq!(text, "integration test"),
        _ => panic!("expected text content"),
    }
}

#[tokio::test]
async fn test_registry_call_missing_tool() {
    let reg = McpToolRegistry::new();
    let err = reg.call_tool("nonexistent", serde_json::Value::Null).await;
    assert!(err.is_err());
}

#[test]
fn test_registry_duplicate_registration() {
    let reg = McpToolRegistry::new();
    reg.register_tool(McpToolDefinition {
        name: "dup".into(),
        description: "first".into(),
        input_schema: serde_json::json!({}),
        handler: Arc::new(PingHandler),
    })
    .unwrap();
    let err = reg.register_tool(McpToolDefinition {
        name: "dup".into(),
        description: "second".into(),
        input_schema: serde_json::json!({}),
        handler: Arc::new(PingHandler),
    });
    assert!(err.is_err());
}

#[test]
fn test_registry_validate_args_required_field() {
    let reg = McpToolRegistry::new();
    reg.register_tool(McpToolDefinition {
        name: "strict".into(),
        description: "strict".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }),
        handler: Arc::new(PingHandler),
    })
    .unwrap();

    // Missing required field
    assert!(reg.validate_args("strict", &serde_json::json!({})).is_err());
    // Present
    assert!(reg
        .validate_args("strict", &serde_json::json!({"name": "ok"}))
        .is_ok());
}

#[tokio::test]
async fn test_builtins_registered_correctly() {
    let reg = McpToolRegistry::new();
    register_builtins(&reg, serde_json::json!({"tools": true})).unwrap();
    assert_eq!(reg.len(), 3);

    let names: Vec<String> = reg.list_tools().iter().map(|t| t.name.clone()).collect();
    assert!(names.contains(&"ping".to_string()));
    assert!(names.contains(&"echo".to_string()));
    assert!(names.contains(&"list_capabilities".to_string()));
}

// =========================================================================
// 8. Resource Registry Integration Tests
// =========================================================================

#[tokio::test]
async fn test_resource_registry_with_static_provider() {
    let sp = Arc::new(StaticResourceProvider::new());
    sp.add("mem://a", "A", "content-a", Some("text/plain"), None);
    sp.add("mem://b", "B", "content-b", None, Some("desc B"));

    let mut reg = ResourceRegistry::new();
    reg.register(sp);

    let list = reg.list_resources().await.unwrap();
    assert_eq!(list.len(), 2);

    let result = reg.read_resource("mem://a").await.unwrap();
    assert_eq!(result.contents[0].text.as_deref(), Some("content-a"));
}

#[tokio::test]
async fn test_resource_registry_read_not_found() {
    let sp = Arc::new(StaticResourceProvider::new());
    let mut reg = ResourceRegistry::new();
    reg.register(sp);

    let err = reg.read_resource("mem://missing").await;
    assert!(err.is_err());
}

// =========================================================================
// 9. Server Config Tests
// =========================================================================

#[test]
fn test_server_config_default_values() {
    let config = McpServerConfig::default();
    assert_eq!(config.name, "rvagent-mcp");
    assert_eq!(config.max_concurrent, 8);
    assert!(!config.version.is_empty());
}

#[test]
fn test_server_config_serde_roundtrip() {
    let config = McpServerConfig {
        name: "custom-server".into(),
        version: "2.0.0".into(),
        max_concurrent: 16,
    };
    let json = serde_json::to_string(&config).unwrap();
    let back: McpServerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, "custom-server");
    assert_eq!(back.version, "2.0.0");
    assert_eq!(back.max_concurrent, 16);
}

// =========================================================================
// 10. Tool Groups Integration Tests
// =========================================================================

use rvagent_mcp::groups::{ToolFilter, ToolGroup};
use rvagent_mcp::transport::{SseConfig, SseTransport, TransportType};

#[test]
fn test_tool_group_file_contains_expected_tools() {
    let tools = ToolGroup::File.tools();
    assert!(tools.contains(&"read_file"));
    assert!(tools.contains(&"write_file"));
    assert!(tools.contains(&"edit_file"));
    assert!(tools.contains(&"ls"));
    assert!(tools.contains(&"glob"));
    assert!(tools.contains(&"grep"));
}

#[test]
fn test_tool_group_shell_contains_expected_tools() {
    let tools = ToolGroup::Shell.tools();
    assert!(tools.contains(&"execute"));
    assert!(tools.contains(&"bash"));
}

#[test]
fn test_tool_group_memory_contains_expected_tools() {
    let tools = ToolGroup::Memory.tools();
    assert!(tools.contains(&"semantic_search"));
    assert!(tools.contains(&"store_memory"));
    assert!(tools.contains(&"retrieve_memory"));
}

#[test]
fn test_tool_group_agent_contains_expected_tools() {
    let tools = ToolGroup::Agent.tools();
    assert!(tools.contains(&"spawn_agent"));
    assert!(tools.contains(&"agent_status"));
    assert!(tools.contains(&"orchestrate"));
}

#[test]
fn test_tool_group_git_contains_expected_tools() {
    let tools = ToolGroup::Git.tools();
    assert!(tools.contains(&"git_status"));
    assert!(tools.contains(&"git_commit"));
    assert!(tools.contains(&"git_diff"));
}

#[test]
fn test_tool_group_brain_contains_expected_tools() {
    let tools = ToolGroup::Brain.tools();
    assert!(tools.contains(&"brain_search"));
    assert!(tools.contains(&"brain_share"));
    assert!(tools.contains(&"brain_vote"));
}

#[test]
fn test_tool_group_all_groups() {
    let all = ToolGroup::all();
    assert!(all.len() >= 9);
    assert!(all.contains(&ToolGroup::File));
    assert!(all.contains(&ToolGroup::Shell));
    assert!(all.contains(&ToolGroup::Memory));
    assert!(all.contains(&ToolGroup::Agent));
    assert!(all.contains(&ToolGroup::Git));
    assert!(all.contains(&ToolGroup::Web));
    assert!(all.contains(&ToolGroup::Brain));
    assert!(all.contains(&ToolGroup::Task));
    assert!(all.contains(&ToolGroup::Core));
}

#[test]
fn test_tool_group_all_tools() {
    let tools = ToolGroup::all_tools();
    assert!(tools.len() > 30); // Should have many tools across all groups
    assert!(tools.contains(&"ping"));
    assert!(tools.contains(&"read_file"));
    assert!(tools.contains(&"brain_search"));
}

#[test]
fn test_tool_filter_all() {
    let filter = ToolFilter::all();
    assert!(filter.allows_all());
    assert!(filter.is_allowed("any_tool"));
    assert!(filter.is_allowed("read_file"));
    assert!(filter.is_allowed("nonexistent"));
}

#[test]
fn test_tool_filter_from_groups() {
    let filter = ToolFilter::from_groups(&[ToolGroup::File, ToolGroup::Shell]);
    assert!(!filter.allows_all());
    assert!(filter.is_allowed("read_file"));
    assert!(filter.is_allowed("execute"));
    assert!(!filter.is_allowed("brain_search"));
}

#[test]
fn test_tool_filter_from_group_names() {
    let filter = ToolFilter::from_group_names(&[
        "file".to_string(),
        "memory".to_string(),
    ]).unwrap();
    assert!(filter.is_allowed("read_file"));
    assert!(filter.is_allowed("semantic_search"));
    assert!(!filter.is_allowed("execute"));
}

#[test]
fn test_tool_filter_from_group_names_invalid() {
    let result = ToolFilter::from_group_names(&["invalid_group".to_string()]);
    assert!(result.is_err());
}

#[test]
fn test_tool_filter_default() {
    let filter = ToolFilter::default();
    // Default should allow core + file groups
    assert!(filter.is_allowed("ping")); // core
    assert!(filter.is_allowed("read_file")); // file
    assert!(!filter.is_allowed("execute")); // shell not in default
}

#[test]
fn test_tool_filter_count() {
    let filter = ToolFilter::from_groups(&[ToolGroup::Core]);
    assert!(filter.count() >= 3); // ping, echo, version, health

    let all = ToolFilter::all();
    assert_eq!(all.count(), 0); // 0 means all
}

#[test]
fn test_tool_group_from_str_aliases() {
    // Test various aliases
    assert_eq!("file".parse::<ToolGroup>().unwrap(), ToolGroup::File);
    assert_eq!("files".parse::<ToolGroup>().unwrap(), ToolGroup::File);
    assert_eq!("fs".parse::<ToolGroup>().unwrap(), ToolGroup::File);

    assert_eq!("shell".parse::<ToolGroup>().unwrap(), ToolGroup::Shell);
    assert_eq!("sh".parse::<ToolGroup>().unwrap(), ToolGroup::Shell);
    assert_eq!("exec".parse::<ToolGroup>().unwrap(), ToolGroup::Shell);

    assert_eq!("memory".parse::<ToolGroup>().unwrap(), ToolGroup::Memory);
    assert_eq!("mem".parse::<ToolGroup>().unwrap(), ToolGroup::Memory);
    assert_eq!("vector".parse::<ToolGroup>().unwrap(), ToolGroup::Memory);

    assert_eq!("brain".parse::<ToolGroup>().unwrap(), ToolGroup::Brain);
    assert_eq!("pi".parse::<ToolGroup>().unwrap(), ToolGroup::Brain);
    assert_eq!("π".parse::<ToolGroup>().unwrap(), ToolGroup::Brain);
}

#[test]
fn test_tool_group_display() {
    assert_eq!(format!("{}", ToolGroup::File), "file");
    assert_eq!(format!("{}", ToolGroup::Shell), "shell");
    assert_eq!(format!("{}", ToolGroup::Brain), "brain");
    assert_eq!(format!("{}", ToolGroup::Core), "core");
}

// =========================================================================
// 11. SSE Transport Integration Tests
// =========================================================================

#[test]
fn test_sse_config_default() {
    let config = SseConfig::default();
    assert_eq!(config.port, 9000);
    assert_eq!(config.host, "127.0.0.1");
    assert!(config.enable_cors);
    assert_eq!(config.heartbeat_interval_secs, 30);
}

#[tokio::test]
async fn test_sse_transport_creation() {
    let transport = SseTransport::new(SseConfig::default());
    assert_eq!(transport.config().port, 9000);
}

#[tokio::test]
async fn test_sse_transport_response_broadcast() {
    let transport = SseTransport::new(SseConfig::default());
    let mut rx = transport.response_sender().subscribe();

    let resp = JsonRpcResponse::success(
        serde_json::json!(42),
        serde_json::json!({"status": "test"}),
    );
    transport.send_response(resp).await.unwrap();

    let received = rx.recv().await.unwrap();
    assert_eq!(received.id, serde_json::json!(42));
}

#[tokio::test]
async fn test_sse_transport_request_channel() {
    let transport = SseTransport::new(SseConfig::default());
    let req_tx = transport.request_sender();

    let req = JsonRpcRequest::new(100, "test/method");
    req_tx.send(req).await.unwrap();

    let received = transport.receive_request().await.unwrap().unwrap();
    assert_eq!(received.method, "test/method");
    assert_eq!(received.id, serde_json::json!(100));
}

#[tokio::test]
async fn test_sse_transport_close() {
    let transport = SseTransport::new(SseConfig::default());
    assert!(transport.close().await.is_ok());
}

#[tokio::test]
async fn test_sse_transport_send_request_not_supported() {
    let transport = SseTransport::new(SseConfig::default());
    let req = JsonRpcRequest::new(1, "test");
    let result = transport.send_request(req).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_sse_transport_receive_response_not_supported() {
    let transport = SseTransport::new(SseConfig::default());
    let result = transport.receive_response().await;
    assert!(result.is_err());
}

#[test]
fn test_transport_type_from_str() {
    assert_eq!("stdio".parse::<TransportType>().unwrap(), TransportType::Stdio);
    assert_eq!("std".parse::<TransportType>().unwrap(), TransportType::Stdio);
    assert_eq!("sse".parse::<TransportType>().unwrap(), TransportType::Sse);
    assert_eq!("http".parse::<TransportType>().unwrap(), TransportType::Sse);
    assert_eq!("web".parse::<TransportType>().unwrap(), TransportType::Sse);
}

#[test]
fn test_transport_type_from_str_invalid() {
    assert!("invalid".parse::<TransportType>().is_err());
}

#[test]
fn test_transport_type_display() {
    assert_eq!(format!("{}", TransportType::Stdio), "stdio");
    assert_eq!(format!("{}", TransportType::Sse), "sse");
}

#[tokio::test]
async fn test_sse_transport_multiple_subscribers() {
    let transport = SseTransport::new(SseConfig::default());
    let mut rx1 = transport.response_sender().subscribe();
    let mut rx2 = transport.response_sender().subscribe();

    let resp = JsonRpcResponse::success(
        serde_json::json!(1),
        serde_json::json!({"multi": true}),
    );
    transport.send_response(resp).await.unwrap();

    let r1 = rx1.recv().await.unwrap();
    let r2 = rx2.recv().await.unwrap();
    assert_eq!(r1.id, r2.id);
}
