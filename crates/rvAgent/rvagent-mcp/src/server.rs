//! MCP server that routes requests to tools, resources, and prompts.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::protocol::*;
use crate::registry::McpToolRegistry;
use crate::resources::ResourceRegistry;
// McpError and Result are used in tests
#[allow(unused_imports)]
use crate::{McpError, Result};

/// Configuration for the MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Server name.
    pub name: String,
    /// Server version.
    pub version: String,
    /// Maximum concurrent tool calls.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
}

fn default_max_concurrent() -> usize {
    8
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            name: "rvagent-mcp".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            max_concurrent: default_max_concurrent(),
        }
    }
}

/// MCP server that processes JSON-RPC requests.
pub struct McpServer {
    config: McpServerConfig,
    tool_registry: McpToolRegistry,
    resource_registry: Arc<ResourceRegistry>,
}

impl McpServer {
    /// Create a new MCP server with the given config.
    pub fn new(
        config: McpServerConfig,
        tool_registry: McpToolRegistry,
        resource_registry: Arc<ResourceRegistry>,
    ) -> Self {
        Self {
            config,
            tool_registry,
            resource_registry,
        }
    }

    /// Server configuration.
    pub fn config(&self) -> &McpServerConfig {
        &self.config
    }

    /// Handle a JSON-RPC request and produce a response.
    pub async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone();
        match self.dispatch(request).await {
            Ok(result) => JsonRpcResponse::success(id, result),
            Err(err) => JsonRpcResponse::error(id, err),
        }
    }

    async fn dispatch(&self, request: JsonRpcRequest) -> std::result::Result<serde_json::Value, JsonRpcError> {
        match McpMethod::from_str(&request.method) {
            Some(McpMethod::Initialize) => self.handle_initialize(),
            Some(McpMethod::Ping) => Ok(serde_json::json!({})),
            Some(McpMethod::ToolsList) => self.handle_tools_list(),
            Some(McpMethod::ToolsCall) => self.handle_tools_call(request.params).await,
            Some(McpMethod::ResourcesList) => self.handle_resources_list().await,
            Some(McpMethod::ResourcesRead) => self.handle_resources_read(request.params).await,
            Some(McpMethod::ResourcesTemplatesList) => self.handle_templates_list(),
            Some(McpMethod::PromptsList) => Ok(serde_json::json!({ "prompts": [] })),
            Some(McpMethod::PromptsGet) => {
                Err(JsonRpcError::invalid_params("prompt not found"))
            }
            None => Err(JsonRpcError::method_not_found(format!(
                "unknown method: {}",
                request.method
            ))),
        }
    }

    fn handle_initialize(&self) -> std::result::Result<serde_json::Value, JsonRpcError> {
        let result = InitializeResult {
            protocol_version: "2024-11-05".into(),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability { list_changed: false }),
                resources: Some(ResourcesCapability {
                    subscribe: false,
                    list_changed: false,
                }),
                prompts: Some(PromptsCapability { list_changed: false }),
            },
            server_info: ServerInfo {
                name: self.config.name.clone(),
                version: self.config.version.clone(),
            },
        };
        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    fn handle_tools_list(&self) -> std::result::Result<serde_json::Value, JsonRpcError> {
        let tools = self.tool_registry.list_mcp_tools();
        let result = ToolsListResult { tools };
        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    async fn handle_tools_call(
        &self,
        params: Option<serde_json::Value>,
    ) -> std::result::Result<serde_json::Value, JsonRpcError> {
        let params = params.ok_or_else(|| JsonRpcError::invalid_params("missing params"))?;
        let call: ToolCallParams = serde_json::from_value(params)
            .map_err(|e| JsonRpcError::invalid_params(e.to_string()))?;

        match self.tool_registry.call_tool(&call.name, call.arguments).await {
            Ok(result) => {
                serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
            }
            Err(e) => Err(JsonRpcError::internal_error(e.to_string())),
        }
    }

    async fn handle_resources_list(&self) -> std::result::Result<serde_json::Value, JsonRpcError> {
        match self.resource_registry.list_resources().await {
            Ok(resources) => {
                let result = ResourcesListResult { resources };
                serde_json::to_value(result)
                    .map_err(|e| JsonRpcError::internal_error(e.to_string()))
            }
            Err(e) => Err(JsonRpcError::internal_error(e.to_string())),
        }
    }

    async fn handle_resources_read(
        &self,
        params: Option<serde_json::Value>,
    ) -> std::result::Result<serde_json::Value, JsonRpcError> {
        let params = params.ok_or_else(|| JsonRpcError::invalid_params("missing params"))?;
        let read: ResourceReadParams = serde_json::from_value(params)
            .map_err(|e| JsonRpcError::invalid_params(e.to_string()))?;

        match self.resource_registry.read_resource(&read.uri).await {
            Ok(result) => {
                serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
            }
            Err(e) => Err(JsonRpcError::internal_error(e.to_string())),
        }
    }

    fn handle_templates_list(&self) -> std::result::Result<serde_json::Value, JsonRpcError> {
        let templates = self.resource_registry.list_templates();
        let result = ResourceTemplatesListResult {
            resource_templates: templates,
        };
        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    /// Get the tool registry.
    pub fn tool_registry(&self) -> &McpToolRegistry {
        &self.tool_registry
    }

    /// Get the resource registry.
    pub fn resource_registry(&self) -> &ResourceRegistry {
        &*self.resource_registry
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{McpToolDefinition, PingHandler};
    use crate::resources::StaticResourceProvider;

    fn make_server() -> McpServer {
        let reg = McpToolRegistry::new();
        crate::registry::register_builtins(&reg, serde_json::json!({})).unwrap();
        let res = Arc::new(ResourceRegistry::new());
        McpServer::new(McpServerConfig::default(), reg, res)
    }

    fn make_server_with_resources() -> McpServer {
        let reg = McpToolRegistry::new();
        crate::registry::register_builtins(&reg, serde_json::json!({})).unwrap();
        let sp = Arc::new(StaticResourceProvider::new());
        sp.add("memory://doc", "doc", "hello", Some("text/plain"), None);
        let mut rr = ResourceRegistry::new();
        rr.register(sp);
        McpServer::new(McpServerConfig::default(), reg, Arc::new(rr))
    }

    #[tokio::test]
    async fn test_handle_initialize() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "initialize").with_params(
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "test", "version": "1.0" }
            }),
        );
        let resp = server.handle_request(req).await;
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "rvagent-mcp");
    }

    #[tokio::test]
    async fn test_handle_ping() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "ping");
        let resp = server.handle_request(req).await;
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[tokio::test]
    async fn test_handle_tools_list() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "tools/list");
        let resp = server.handle_request(req).await;
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert!(tools.len() >= 3);
    }

    #[tokio::test]
    async fn test_handle_tools_call_ping() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "tools/call").with_params(
            serde_json::json!({"name": "ping", "arguments": {}}),
        );
        let resp = server.handle_request(req).await;
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[tokio::test]
    async fn test_handle_tools_call_echo() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "tools/call").with_params(
            serde_json::json!({"name": "echo", "arguments": {"text": "hello"}}),
        );
        let resp = server.handle_request(req).await;
        let result = resp.result.unwrap();
        assert_eq!(result["content"][0]["text"], "hello");
    }

    #[tokio::test]
    async fn test_handle_tools_call_missing_tool() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "tools/call").with_params(
            serde_json::json!({"name": "nonexistent", "arguments": {}}),
        );
        let resp = server.handle_request(req).await;
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn test_handle_tools_call_no_params() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "tools/call");
        let resp = server.handle_request(req).await;
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn test_handle_tools_call_invalid_params() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "tools/call")
            .with_params(serde_json::json!("not an object"));
        let resp = server.handle_request(req).await;
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn test_handle_resources_list_empty() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "resources/list");
        let resp = server.handle_request(req).await;
        let result = resp.result.unwrap();
        assert!(result["resources"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_resources_list_with_data() {
        let server = make_server_with_resources();
        let req = JsonRpcRequest::new(1, "resources/list");
        let resp = server.handle_request(req).await;
        let result = resp.result.unwrap();
        assert_eq!(result["resources"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_handle_resources_read() {
        let server = make_server_with_resources();
        let req = JsonRpcRequest::new(1, "resources/read")
            .with_params(serde_json::json!({"uri": "memory://doc"}));
        let resp = server.handle_request(req).await;
        let result = resp.result.unwrap();
        assert_eq!(result["contents"][0]["text"], "hello");
    }

    #[tokio::test]
    async fn test_handle_resources_read_not_found() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "resources/read")
            .with_params(serde_json::json!({"uri": "memory://missing"}));
        let resp = server.handle_request(req).await;
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn test_handle_resources_read_no_params() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "resources/read");
        let resp = server.handle_request(req).await;
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn test_handle_resources_templates_list() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "resources/templates/list");
        let resp = server.handle_request(req).await;
        assert!(resp.result.is_some());
    }

    #[tokio::test]
    async fn test_handle_unknown_method() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "unknown/method");
        let resp = server.handle_request(req).await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_handle_prompts_list() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "prompts/list");
        let resp = server.handle_request(req).await;
        let result = resp.result.unwrap();
        assert!(result["prompts"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_prompts_get_not_found() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "prompts/get")
            .with_params(serde_json::json!({"name": "missing"}));
        let resp = server.handle_request(req).await;
        assert!(resp.error.is_some());
    }

    #[test]
    fn test_server_config_default() {
        let config = McpServerConfig::default();
        assert_eq!(config.name, "rvagent-mcp");
        assert_eq!(config.max_concurrent, 8);
    }

    #[test]
    fn test_server_config_serde() {
        let config = McpServerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let back: McpServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, config.name);
    }

    #[tokio::test]
    async fn test_tool_registry_accessible() {
        let server = make_server();
        assert!(server.tool_registry().len() >= 3);
    }

    #[tokio::test]
    async fn test_register_custom_tool() {
        let server = make_server();
        server.tool_registry().register_tool(McpToolDefinition {
            name: "custom".into(),
            description: "Custom tool".into(),
            input_schema: serde_json::json!({"type": "object"}),
            handler: std::sync::Arc::new(PingHandler),
        }).unwrap();
        assert!(server.tool_registry().get_tool("custom").is_some());
    }

    #[tokio::test]
    async fn test_response_has_correct_id() {
        let server = make_server();
        let req = JsonRpcRequest::new(42, "ping");
        let resp = server.handle_request(req).await;
        assert_eq!(resp.id, serde_json::json!(42));
    }

    #[tokio::test]
    async fn test_response_jsonrpc_version() {
        let server = make_server();
        let req = JsonRpcRequest::new(1, "ping");
        let resp = server.handle_request(req).await;
        assert_eq!(resp.jsonrpc, "2.0");
    }
}
