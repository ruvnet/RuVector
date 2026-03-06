//! MCP protocol server implementation (stdio JSON-RPC)

use crate::tools::McpBrainTools;
use crate::types::*;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, info, warn};

/// MCP Brain Server
pub struct McpBrainServer {
    tools: McpBrainTools,
}

impl McpBrainServer {
    pub fn new() -> Self {
        Self {
            tools: McpBrainTools::new(),
        }
    }

    pub fn with_backend_url(url: String) -> Self {
        Self {
            tools: McpBrainTools::with_backend_url(url),
        }
    }

    /// Run the server on stdio
    pub async fn run_stdio(&self) -> Result<(), std::io::Error> {
        info!("Starting MCP Brain server on stdio");

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let reader = BufReader::new(stdin);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }

            debug!("Received: {}", line);

            let response = self.handle_message(&line).await;

            if let Some(resp) = response {
                let resp_json = serde_json::to_string(&resp).unwrap_or_default();
                debug!("Sending: {}", resp_json);
                stdout.write_all(resp_json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
        }

        info!("MCP Brain server shutting down");
        Ok(())
    }

    async fn handle_message(&self, message: &str) -> Option<JsonRpcResponse> {
        let request: JsonRpcRequest = match serde_json::from_str(message) {
            Ok(req) => req,
            Err(e) => {
                error!("Failed to parse request: {}", e);
                return Some(JsonRpcResponse::error(
                    serde_json::Value::Null,
                    -32700,
                    format!("Parse error: {}", e),
                ));
            }
        };

        Some(self.handle_request(&request).await)
    }

    async fn handle_request(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "initialized" => JsonRpcResponse::success(request.id.clone(), serde_json::json!({})),
            "tools/list" => self.handle_tools_list(request),
            "tools/call" => self.handle_tools_call(request).await,
            "shutdown" => {
                info!("Received shutdown request");
                JsonRpcResponse::success(request.id.clone(), serde_json::json!({}))
            }
            _ => {
                warn!("Unknown method: {}", request.method);
                JsonRpcResponse::error(
                    request.id.clone(),
                    -32601,
                    format!("Method not found: {}", request.method),
                )
            }
        }
    }

    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        info!("Handling initialize request");
        JsonRpcResponse::success(
            request.id.clone(),
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "mcp-brain",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
    }

    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        info!("Handling tools/list request");
        let tools = McpBrainTools::list_tools();
        JsonRpcResponse::success(request.id.clone(), serde_json::json!({ "tools": tools }))
    }

    async fn handle_tools_call(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        info!("Handling tools/call request");
        let tool_call: McpToolCall = match serde_json::from_value(request.params.clone()) {
            Ok(tc) => tc,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    -32602,
                    format!("Invalid params: {}", e),
                );
            }
        };

        match self.tools.call_tool(tool_call).await {
            Ok(result) => {
                let response_content = match result {
                    McpToolResult::Success { content } => serde_json::json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&content).unwrap_or_default()
                        }]
                    }),
                    McpToolResult::Error { error } => serde_json::json!({
                        "content": [{
                            "type": "text",
                            "text": error
                        }],
                        "isError": true
                    }),
                };
                JsonRpcResponse::success(request.id.clone(), response_content)
            }
            Err(e) => JsonRpcResponse::error(request.id.clone(), e.code(), e.to_string()),
        }
    }
}

impl Default for McpBrainServer {
    fn default() -> Self {
        Self::new()
    }
}
