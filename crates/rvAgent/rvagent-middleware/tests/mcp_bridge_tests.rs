//! Integration tests for the MCP bridge middleware.

use rvagent_middleware::{
    AgentState, Middleware, ModelHandler, ModelRequest, ModelResponse,
    Message, Runtime, RunnableConfig,
};
use rvagent_middleware::mcp_bridge::{McpBridgeConfig, McpBridgeMiddleware};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

struct PassthroughHandler;

impl ModelHandler for PassthroughHandler {
    fn call(&self, request: ModelRequest) -> ModelResponse {
        ModelResponse::text(format!("handled:{}", request.messages.len()))
    }
}

// ---------------------------------------------------------------------------
// Tests: McpBridgeConfig
// ---------------------------------------------------------------------------

#[test]
fn test_default_config_values() {
    let config = McpBridgeConfig::default();
    assert!(config.enabled);
    assert_eq!(config.max_concurrent, 10);
    assert!(config.allowed_transports.contains(&"stdio".to_string()));
    assert!(config.allowed_transports.contains(&"sse".to_string()));
    assert!(config.allowed_transports.contains(&"memory".to_string()));
    assert!(config.tool_allowlist.is_empty());
}

#[test]
fn test_config_serialization_roundtrip() {
    let config = McpBridgeConfig {
        enabled: false,
        max_concurrent: 5,
        allowed_transports: vec!["stdio".into()],
        tool_allowlist: vec!["read_file".into(), "ls".into()],
    };
    let json = serde_json::to_string(&config).unwrap();
    let restored: McpBridgeConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.enabled, false);
    assert_eq!(restored.max_concurrent, 5);
    assert_eq!(restored.allowed_transports, vec!["stdio"]);
    assert_eq!(restored.tool_allowlist, vec!["read_file", "ls"]);
}

// ---------------------------------------------------------------------------
// Tests: Tool allowlist
// ---------------------------------------------------------------------------

#[test]
fn test_tool_allowed_with_empty_allowlist_permits_all() {
    let mw = McpBridgeMiddleware::new();
    assert!(mw.is_tool_allowed("execute"));
    assert!(mw.is_tool_allowed("read_file"));
    assert!(mw.is_tool_allowed("arbitrary_name"));
}

#[test]
fn test_tool_allowed_with_populated_allowlist() {
    let config = McpBridgeConfig {
        tool_allowlist: vec!["read_file".into(), "glob".into()],
        ..Default::default()
    };
    let mw = McpBridgeMiddleware::with_config(config);
    assert!(mw.is_tool_allowed("read_file"));
    assert!(mw.is_tool_allowed("glob"));
    assert!(!mw.is_tool_allowed("execute"));
    assert!(!mw.is_tool_allowed("write_file"));
}

// ---------------------------------------------------------------------------
// Tests: Transport allowlist
// ---------------------------------------------------------------------------

#[test]
fn test_transport_allowed_default() {
    let mw = McpBridgeMiddleware::new();
    assert!(mw.is_transport_allowed("stdio"));
    assert!(mw.is_transport_allowed("sse"));
    assert!(mw.is_transport_allowed("memory"));
    assert!(!mw.is_transport_allowed("websocket"));
    assert!(!mw.is_transport_allowed("http"));
}

#[test]
fn test_transport_allowed_custom() {
    let config = McpBridgeConfig {
        allowed_transports: vec!["websocket".into()],
        ..Default::default()
    };
    let mw = McpBridgeMiddleware::with_config(config);
    assert!(mw.is_transport_allowed("websocket"));
    assert!(!mw.is_transport_allowed("stdio"));
}

// ---------------------------------------------------------------------------
// Tests: Middleware trait methods
// ---------------------------------------------------------------------------

#[test]
fn test_middleware_name() {
    let mw = McpBridgeMiddleware::new();
    assert_eq!(mw.name(), "mcp_bridge");
}

#[test]
fn test_before_agent_when_enabled_injects_config() {
    let mw = McpBridgeMiddleware::new();
    let state = AgentState::default();
    let runtime = Runtime::new();
    let config = RunnableConfig::default();

    let update = mw.before_agent(&state, &runtime, &config);
    assert!(update.is_some(), "enabled bridge should produce state update");

    let update = update.unwrap();
    assert!(
        update.extensions.contains_key("mcp_bridge_config"),
        "should inject mcp_bridge_config extension"
    );
}

#[test]
fn test_before_agent_when_disabled_returns_none() {
    let config = McpBridgeConfig {
        enabled: false,
        ..Default::default()
    };
    let mw = McpBridgeMiddleware::with_config(config);
    let state = AgentState::default();
    let runtime = Runtime::new();
    let runnable_config = RunnableConfig::default();

    let update = mw.before_agent(&state, &runtime, &runnable_config);
    assert!(update.is_none(), "disabled bridge should not produce update");
}

#[test]
fn test_modify_request_when_enabled_sets_flag() {
    let mw = McpBridgeMiddleware::new();
    let request = ModelRequest::new(vec![Message::user("hello")]);
    let modified = mw.modify_request(request);

    assert_eq!(
        modified.extensions.get("mcp_bridge_active"),
        Some(&serde_json::json!(true)),
        "enabled bridge should set mcp_bridge_active flag"
    );
}

#[test]
fn test_modify_request_when_disabled_does_not_set_flag() {
    let config = McpBridgeConfig {
        enabled: false,
        ..Default::default()
    };
    let mw = McpBridgeMiddleware::with_config(config);
    let request = ModelRequest::new(vec![Message::user("hello")]);
    let modified = mw.modify_request(request);

    assert!(
        !modified.extensions.contains_key("mcp_bridge_active"),
        "disabled bridge should not set flag"
    );
}

#[test]
fn test_wrap_model_call_passes_through() {
    let mw = McpBridgeMiddleware::new();
    let request = ModelRequest::new(vec![Message::user("hi")]);
    let response = mw.wrap_model_call(request, &PassthroughHandler);

    assert!(
        response.message.content.contains("handled:1"),
        "wrap_model_call should pass through to handler"
    );
}

// ---------------------------------------------------------------------------
// Tests: Status tool
// ---------------------------------------------------------------------------

#[test]
fn test_tools_when_enabled_provides_status_tool() {
    let mw = McpBridgeMiddleware::new();
    let tools = mw.tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "mcp_bridge_status");
}

#[test]
fn test_tools_when_disabled_provides_no_tools() {
    let config = McpBridgeConfig {
        enabled: false,
        ..Default::default()
    };
    let mw = McpBridgeMiddleware::with_config(config);
    let tools = mw.tools();
    assert!(tools.is_empty());
}

#[test]
fn test_status_tool_returns_config_values() {
    let mw = McpBridgeMiddleware::new();
    let tools = mw.tools();
    let status_tool = &tools[0];

    let result = status_tool.invoke(serde_json::json!({}));
    assert!(result.is_ok());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(json["enabled"], true);
    assert_eq!(json["max_concurrent"], 10);
    assert!(json["allowed_transports"].is_array());
}

#[test]
fn test_status_tool_schema() {
    let mw = McpBridgeMiddleware::new();
    let tools = mw.tools();
    let schema = tools[0].parameters_schema();
    assert!(schema.is_object());
    assert!(schema["properties"].is_object());
}

#[test]
fn test_status_tool_description() {
    let mw = McpBridgeMiddleware::new();
    let tools = mw.tools();
    assert!(
        !tools[0].description().is_empty(),
        "status tool should have a description"
    );
}
