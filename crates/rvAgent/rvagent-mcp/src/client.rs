//! MCP client for connecting to external MCP servers.

use std::sync::Arc;

use crate::protocol::{
    InitializeParams, InitializeResult, JsonRpcRequest, McpTool, ResourceReadResult, ToolCallResult,
};
use crate::transport::Transport;
use crate::{McpError, Result};

/// MCP client that communicates over a [`Transport`].
pub struct McpClient {
    transport: Arc<dyn Transport>,
    initialized: bool,
}

impl McpClient {
    /// Create a new MCP client with the given transport.
    pub fn new(transport: Arc<dyn Transport>) -> Self {
        Self {
            transport,
            initialized: false,
        }
    }

    /// Whether the client has completed initialization.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Initialize the connection with the server.
    pub async fn initialize(&mut self, params: InitializeParams) -> Result<InitializeResult> {
        let req = JsonRpcRequest::new(1, "initialize")
            .with_params(serde_json::to_value(&params).map_err(McpError::from)?);
        self.transport.send_request(req).await?;
        let resp = self.transport.receive_response().await?
            .ok_or_else(|| McpError::client("connection closed"))?;
        if let Some(error) = resp.error {
            return Err(McpError::client(error.message));
        }
        let result: InitializeResult = serde_json::from_value(
            resp.result
                .ok_or_else(|| McpError::client("missing result"))?,
        )
        .map_err(McpError::from)?;
        self.initialized = true;
        Ok(result)
    }

    /// Ping the server.
    pub async fn ping(&self) -> Result<()> {
        let req = JsonRpcRequest::new(2, "ping");
        self.transport.send_request(req).await?;
        let resp = self.transport.receive_response().await?
            .ok_or_else(|| McpError::client("connection closed"))?;
        if resp.error.is_some() {
            return Err(McpError::client("ping failed"));
        }
        Ok(())
    }

    /// List available tools.
    pub async fn list_tools(&self) -> Result<Vec<McpTool>> {
        let req = JsonRpcRequest::new(3, "tools/list");
        self.transport.send_request(req).await?;
        let resp = self.transport.receive_response().await?
            .ok_or_else(|| McpError::client("connection closed"))?;
        if let Some(error) = resp.error {
            return Err(McpError::client(error.message));
        }
        let result = resp
            .result
            .ok_or_else(|| McpError::client("missing result"))?;
        let tools = result
            .get("tools")
            .ok_or_else(|| McpError::client("missing tools field"))?;
        let tools: Vec<McpTool> =
            serde_json::from_value(tools.clone()).map_err(McpError::from)?;
        Ok(tools)
    }

    /// Call a tool by name.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult> {
        let req = JsonRpcRequest::new(4, "tools/call")
            .with_params(serde_json::json!({ "name": name, "arguments": arguments }));
        self.transport.send_request(req).await?;
        let resp = self.transport.receive_response().await?
            .ok_or_else(|| McpError::client("connection closed"))?;
        if let Some(error) = resp.error {
            return Err(McpError::client(error.message));
        }
        let result: ToolCallResult = serde_json::from_value(
            resp.result
                .ok_or_else(|| McpError::client("missing result"))?,
        )
        .map_err(McpError::from)?;
        Ok(result)
    }

    /// Read a resource by URI.
    pub async fn read_resource(&self, uri: &str) -> Result<ResourceReadResult> {
        let req = JsonRpcRequest::new(5, "resources/read")
            .with_params(serde_json::json!({ "uri": uri }));
        self.transport.send_request(req).await?;
        let resp = self.transport.receive_response().await?
            .ok_or_else(|| McpError::client("connection closed"))?;
        if let Some(error) = resp.error {
            return Err(McpError::client(error.message));
        }
        let result: ResourceReadResult = serde_json::from_value(
            resp.result
                .ok_or_else(|| McpError::client("missing result"))?,
        )
        .map_err(McpError::from)?;
        Ok(result)
    }

    /// List available resources.
    pub async fn list_resources(&self) -> Result<Vec<crate::protocol::McpResource>> {
        let req = JsonRpcRequest::new(6, "resources/list");
        self.transport.send_request(req).await?;
        let resp = self.transport.receive_response().await?
            .ok_or_else(|| McpError::client("connection closed"))?;
        if let Some(error) = resp.error {
            return Err(McpError::client(error.message));
        }
        let result = resp
            .result
            .ok_or_else(|| McpError::client("missing result"))?;
        let resources = result
            .get("resources")
            .ok_or_else(|| McpError::client("missing resources field"))?;
        serde_json::from_value(resources.clone()).map_err(McpError::from)
    }

    /// Close the client and underlying transport.
    pub async fn close(self) -> Result<()> {
        self.transport.close().await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::*;
    use crate::transport::MemoryTransport;

    fn setup() -> (McpClient, Arc<MemoryTransport>) {
        let (client_transport, server_transport) = MemoryTransport::pair(32);
        let client_transport = Arc::new(client_transport);
        let server_transport = Arc::new(server_transport);
        let client = McpClient::new(client_transport);
        (client, server_transport)
    }

    async fn respond_with(server: &Arc<MemoryTransport>, result: serde_json::Value) {
        let req = server.receive_request().await.unwrap().unwrap();
        server
            .send_response(JsonRpcResponse::success(req.id, result))
            .await
            .unwrap();
    }

    async fn respond_with_error(server: &Arc<MemoryTransport>, error: JsonRpcError) {
        let req = server.receive_request().await.unwrap().unwrap();
        server
            .send_response(JsonRpcResponse::error(req.id, error))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_client_initialize() {
        let (mut client, server) = setup();
        let h = tokio::spawn(async move {
            respond_with(
                &server,
                serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "test", "version": "1.0"}
                }),
            )
            .await;
        });
        let result = client
            .initialize(InitializeParams {
                protocol_version: "2024-11-05".into(),
                capabilities: ClientCapabilities::default(),
                client_info: ClientInfo { name: "t".into(), version: "1".into() },
            })
            .await
            .unwrap();
        assert_eq!(result.server_info.name, "test");
        assert!(client.is_initialized());
        h.await.unwrap();
    }

    #[tokio::test]
    async fn test_client_initialize_error() {
        let (mut client, server) = setup();
        let h = tokio::spawn(async move {
            respond_with_error(&server, JsonRpcError::internal_error("fail")).await;
        });
        let err = client
            .initialize(InitializeParams {
                protocol_version: "2024-11-05".into(),
                capabilities: ClientCapabilities::default(),
                client_info: ClientInfo { name: "t".into(), version: "0".into() },
            })
            .await;
        assert!(err.is_err());
        h.await.unwrap();
    }

    #[tokio::test]
    async fn test_client_ping() {
        let (client, server) = setup();
        let h = tokio::spawn(async move {
            respond_with(&server, serde_json::json!({})).await;
        });
        client.ping().await.unwrap();
        h.await.unwrap();
    }

    #[tokio::test]
    async fn test_client_ping_error() {
        let (client, server) = setup();
        let h = tokio::spawn(async move {
            respond_with_error(&server, JsonRpcError::internal_error("err")).await;
        });
        assert!(client.ping().await.is_err());
        h.await.unwrap();
    }

    #[tokio::test]
    async fn test_client_list_tools() {
        let (client, server) = setup();
        let h = tokio::spawn(async move {
            respond_with(
                &server,
                serde_json::json!({"tools": [
                    {"name": "ping", "description": "ping", "inputSchema": {}}
                ]}),
            )
            .await;
        });
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "ping");
        h.await.unwrap();
    }

    #[tokio::test]
    async fn test_client_call_tool() {
        let (client, server) = setup();
        let h = tokio::spawn(async move {
            respond_with(
                &server,
                serde_json::json!({
                    "content": [{"type": "text", "text": "pong"}],
                    "isError": false
                }),
            )
            .await;
        });
        let result = client.call_tool("ping", serde_json::json!({})).await.unwrap();
        assert!(!result.is_error);
        h.await.unwrap();
    }

    #[tokio::test]
    async fn test_client_call_tool_error() {
        let (client, server) = setup();
        let h = tokio::spawn(async move {
            respond_with_error(&server, JsonRpcError::internal_error("fail")).await;
        });
        assert!(client.call_tool("bad", serde_json::json!({})).await.is_err());
        h.await.unwrap();
    }

    #[tokio::test]
    async fn test_client_read_resource() {
        let (client, server) = setup();
        let h = tokio::spawn(async move {
            respond_with(
                &server,
                serde_json::json!({"contents": [{"uri": "m://d", "text": "hi"}]}),
            )
            .await;
        });
        let result = client.read_resource("m://d").await.unwrap();
        assert_eq!(result.contents[0].text.as_deref(), Some("hi"));
        h.await.unwrap();
    }

    #[tokio::test]
    async fn test_client_list_resources() {
        let (client, server) = setup();
        let h = tokio::spawn(async move {
            respond_with(
                &server,
                serde_json::json!({"resources": [{"uri": "m://a", "name": "a"}]}),
            )
            .await;
        });
        let resources = client.list_resources().await.unwrap();
        assert_eq!(resources.len(), 1);
        h.await.unwrap();
    }

    #[tokio::test]
    async fn test_client_not_initialized() {
        let (client, _server) = setup();
        assert!(!client.is_initialized());
    }

    #[tokio::test]
    async fn test_client_close() {
        let (client, _server) = setup();
        assert!(client.close().await.is_ok());
    }
}
