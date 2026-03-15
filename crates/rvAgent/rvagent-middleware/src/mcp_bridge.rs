//! MCP Bridge Middleware — routes MCP tool calls through the middleware pipeline.
//!
//! Enables the rvAgent middleware pipeline to handle MCP-originated tool calls
//! alongside native tool calls, with proper security and validation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    AgentState, AgentStateUpdate, Middleware, ModelHandler, ModelRequest, ModelResponse,
    RunnableConfig, Runtime,
};

/// MCP tool call origin tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCallOrigin {
    pub transport: String,
    pub client_id: Option<String>,
    pub request_id: Option<serde_json::Value>,
}

/// Configuration for the MCP bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpBridgeConfig {
    /// Whether to allow MCP tool calls through the pipeline.
    pub enabled: bool,
    /// Maximum concurrent MCP tool calls.
    pub max_concurrent: usize,
    /// Allowed MCP transports.
    pub allowed_transports: Vec<String>,
    /// Tool allowlist (empty = all allowed).
    pub tool_allowlist: Vec<String>,
}

impl Default for McpBridgeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent: 10,
            allowed_transports: vec!["stdio".into(), "sse".into(), "memory".into()],
            tool_allowlist: vec![],
        }
    }
}

/// Middleware that bridges MCP tool calls into the rvAgent pipeline.
pub struct McpBridgeMiddleware {
    config: McpBridgeConfig,
}

impl McpBridgeMiddleware {
    pub fn new() -> Self {
        Self {
            config: McpBridgeConfig::default(),
        }
    }

    pub fn with_config(config: McpBridgeConfig) -> Self {
        Self { config }
    }

    /// Check if a tool is allowed by the bridge configuration.
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        if self.config.tool_allowlist.is_empty() {
            return true;
        }
        self.config.tool_allowlist.contains(&tool_name.to_string())
    }

    /// Check if a transport is allowed.
    pub fn is_transport_allowed(&self, transport: &str) -> bool {
        self.config
            .allowed_transports
            .contains(&transport.to_string())
    }
}

#[async_trait]
impl Middleware for McpBridgeMiddleware {
    fn name(&self) -> &str {
        "mcp_bridge"
    }

    fn before_agent(
        &self,
        _state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if !self.config.enabled {
            return None;
        }

        let mut update = AgentStateUpdate::default();
        update.extensions.insert(
            "mcp_bridge_config".into(),
            serde_json::to_value(&self.config).unwrap_or_default(),
        );
        Some(update)
    }

    fn modify_request(&self, mut request: ModelRequest) -> ModelRequest {
        if !self.config.enabled {
            return request;
        }
        request
            .extensions
            .insert("mcp_bridge_active".into(), serde_json::json!(true));
        request
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        handler.call(request)
    }

    fn tools(&self) -> Vec<Box<dyn crate::Tool>> {
        if !self.config.enabled {
            return vec![];
        }
        vec![Box::new(McpStatusTool {
            config: self.config.clone(),
        })]
    }
}

/// Introspection tool that reports MCP bridge status.
struct McpStatusTool {
    config: McpBridgeConfig,
}

impl crate::Tool for McpStatusTool {
    fn name(&self) -> &str {
        "mcp_bridge_status"
    }

    fn description(&self) -> &str {
        "Returns the current MCP bridge configuration and status"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    fn invoke(&self, _args: serde_json::Value) -> Result<String, String> {
        Ok(serde_json::json!({
            "enabled": self.config.enabled,
            "max_concurrent": self.config.max_concurrent,
            "allowed_transports": self.config.allowed_transports,
            "tool_allowlist": self.config.tool_allowlist,
        })
        .to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_bridge_default_config() {
        let config = McpBridgeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_concurrent, 10);
        assert_eq!(config.allowed_transports.len(), 3);
        assert!(config.tool_allowlist.is_empty());
    }

    #[test]
    fn test_mcp_bridge_tool_allowed_empty_list() {
        let mw = McpBridgeMiddleware::new();
        assert!(mw.is_tool_allowed("any_tool"));
    }

    #[test]
    fn test_mcp_bridge_tool_allowed_with_allowlist() {
        let config = McpBridgeConfig {
            tool_allowlist: vec!["read_file".into(), "ls".into()],
            ..Default::default()
        };
        let mw = McpBridgeMiddleware::with_config(config);
        assert!(mw.is_tool_allowed("read_file"));
        assert!(mw.is_tool_allowed("ls"));
        assert!(!mw.is_tool_allowed("execute"));
    }

    #[test]
    fn test_mcp_bridge_transport_allowed() {
        let mw = McpBridgeMiddleware::new();
        assert!(mw.is_transport_allowed("stdio"));
        assert!(mw.is_transport_allowed("sse"));
        assert!(!mw.is_transport_allowed("websocket"));
    }

    #[test]
    fn test_mcp_bridge_disabled() {
        let config = McpBridgeConfig {
            enabled: false,
            ..Default::default()
        };
        let mw = McpBridgeMiddleware::with_config(config);
        let state = AgentState::default();
        let runtime = Runtime::new();
        let runnable_config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &runnable_config).is_none());
        assert!(mw.tools().is_empty());
    }

    #[test]
    fn test_mcp_bridge_enabled_injects_config() {
        let mw = McpBridgeMiddleware::new();
        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        let update = mw.before_agent(&state, &runtime, &config);
        assert!(update.is_some());
        assert!(update.unwrap().extensions.contains_key("mcp_bridge_config"));
    }

    #[test]
    fn test_mcp_bridge_provides_status_tool() {
        let mw = McpBridgeMiddleware::new();
        let tools = mw.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "mcp_bridge_status");
    }

    #[test]
    fn test_mcp_status_tool_invoke() {
        use crate::Tool;
        let tool = McpStatusTool {
            config: McpBridgeConfig::default(),
        };
        let result = tool.invoke(serde_json::json!({}));
        assert!(result.is_ok());
        let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(json["enabled"], true);
        assert_eq!(json["max_concurrent"], 10);
    }

    #[test]
    fn test_mcp_bridge_modify_request() {
        let mw = McpBridgeMiddleware::new();
        let request = ModelRequest::new(vec![]);
        let modified = mw.modify_request(request);
        assert_eq!(
            modified.extensions.get("mcp_bridge_active"),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn test_mcp_bridge_middleware_name() {
        let mw = McpBridgeMiddleware::new();
        assert_eq!(mw.name(), "mcp_bridge");
    }

    #[test]
    fn test_mcp_bridge_config_serde() {
        let config = McpBridgeConfig {
            enabled: true,
            max_concurrent: 5,
            allowed_transports: vec!["stdio".into()],
            tool_allowlist: vec!["ls".into()],
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: McpBridgeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_concurrent, 5);
        assert_eq!(back.allowed_transports, vec!["stdio"]);
    }

    #[test]
    fn test_mcp_bridge_disabled_modify_request_passthrough() {
        let config = McpBridgeConfig {
            enabled: false,
            ..Default::default()
        };
        let mw = McpBridgeMiddleware::with_config(config);
        let request = ModelRequest::new(vec![]);
        let modified = mw.modify_request(request);
        assert!(!modified.extensions.contains_key("mcp_bridge_active"));
    }
}
