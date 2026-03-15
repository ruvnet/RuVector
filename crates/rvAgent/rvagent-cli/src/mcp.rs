//! MCP (Model Context Protocol) client integration for rvAgent CLI.
//!
//! Connects to external MCP servers, discovers their available tools,
//! and translates MCP tool schemas into the rvAgent `Tool` trait format
//! so they can be used seamlessly in the agent pipeline.
//!
//! Supports both stdio (subprocess) and SSE (HTTP) transports.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// MCP tool schema types
// ---------------------------------------------------------------------------

/// An MCP tool definition received from a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    /// Tool name (must be unique across all MCP servers).
    pub name: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

/// An MCP server connection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Display name for this server.
    pub name: String,
    /// Transport type: "stdio" or "sse".
    pub transport: McpTransport,
    /// Whether this server is currently connected.
    #[serde(default)]
    pub connected: bool,
}

/// MCP transport types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpTransport {
    /// Standard I/O transport — launch a subprocess.
    Stdio {
        /// Command to execute.
        command: String,
        /// Arguments for the command.
        #[serde(default)]
        args: Vec<String>,
        /// Environment variables to set.
        #[serde(default)]
        env: HashMap<String, String>,
    },
    /// Server-Sent Events transport — connect to an HTTP endpoint.
    Sse {
        /// The SSE endpoint URL.
        url: String,
        /// Optional authorization header value.
        #[serde(default)]
        auth: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// MCP tool call / result
// ---------------------------------------------------------------------------

/// A tool invocation request to an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCall {
    /// Tool name.
    pub name: String,
    /// Arguments as a JSON object.
    pub arguments: serde_json::Value,
}

/// A tool execution result from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    /// Whether the tool call succeeded.
    pub is_error: bool,
    /// Result content.
    pub content: Vec<McpContent>,
}

/// Content block in an MCP tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpContent {
    /// Plain text content.
    Text { text: String },
    /// Image content (base64-encoded).
    Image { data: String, mime_type: String },
    /// Resource reference.
    Resource { uri: String },
}

impl McpToolResult {
    /// Extract the text content from the result, joining multiple text blocks.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                McpContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ---------------------------------------------------------------------------
// MCP client
// ---------------------------------------------------------------------------

/// A client connection to a single MCP server.
pub struct McpClient {
    /// Server configuration.
    config: McpServerConfig,
    /// Discovered tools from this server.
    tools: Vec<McpToolDef>,
    /// Stdin handle for the stdio subprocess (if connected via stdio transport).
    #[allow(dead_code)]
    stdin: Option<tokio::process::ChildStdin>,
    /// Buffered stdout reader for the stdio subprocess.
    #[allow(dead_code)]
    stdout: Option<BufReader<tokio::process::ChildStdout>>,
    /// Child process handle for the stdio subprocess.
    #[allow(dead_code)]
    child: Option<tokio::process::Child>,
    /// Next JSON-RPC request ID.
    next_id: u64,
}

impl McpClient {
    /// Create a new MCP client for the given server config.
    pub fn new(config: McpServerConfig) -> Self {
        Self {
            config,
            tools: Vec::new(),
            stdin: None,
            stdout: None,
            child: None,
            next_id: 1,
        }
    }

    /// Connect to the MCP server and discover available tools.
    ///
    /// For stdio transport, this spawns the subprocess and performs the
    /// initialize / tools/list handshake. For SSE, it connects to the
    /// endpoint and subscribes to events.
    pub async fn connect(&mut self) -> Result<()> {
        info!(server = %self.config.name, "connecting to MCP server");

        // Clone transport to avoid borrow conflict with &self and &mut self.
        let transport = self.config.transport.clone();
        match &transport {
            McpTransport::Stdio {
                command,
                args,
                env,
            } => {
                self.connect_stdio(command, args, env).await?;
            }
            McpTransport::Sse { url, auth } => {
                self.connect_sse(url, auth.as_deref()).await?;
            }
        }

        self.config.connected = true;
        info!(
            server = %self.config.name,
            tools = self.tools.len(),
            "MCP server connected"
        );
        Ok(())
    }

    /// Discover tools — returns the list of tools from this server.
    pub fn tools(&self) -> &[McpToolDef] {
        &self.tools
    }

    /// Check if the server is currently connected.
    pub fn is_connected(&self) -> bool {
        self.config.connected
    }

    /// Server name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Call a tool on this MCP server via JSON-RPC over the stdio transport.
    pub async fn call_tool(&mut self, call: &McpToolCall) -> Result<McpToolResult> {
        if !self.config.connected {
            anyhow::bail!("MCP server '{}' is not connected", self.config.name);
        }

        let stdin = self.stdin.as_mut().context(
            "MCP server not connected via stdio — call_tool requires an active subprocess",
        )?;
        let stdout = self.stdout.as_mut().context(
            "MCP server stdout not available",
        )?;

        let id = self.next_id;
        self.next_id += 1;

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {
                "name": call.name,
                "arguments": call.arguments,
            }
        });

        let mut request_line = serde_json::to_string(&request)
            .context("failed to serialize tools/call request")?;
        request_line.push('\n');

        stdin
            .write_all(request_line.as_bytes())
            .await
            .context("failed to write to MCP subprocess stdin")?;
        stdin
            .flush()
            .await
            .context("failed to flush MCP subprocess stdin")?;

        let mut response_line = String::new();
        stdout
            .read_line(&mut response_line)
            .await
            .context("failed to read tools/call response from MCP subprocess")?;

        let response: serde_json::Value = serde_json::from_str(response_line.trim())
            .context("failed to parse tools/call JSON-RPC response")?;

        debug!(
            server = %self.config.name,
            tool = %call.name,
            "MCP tools/call response received"
        );

        // Parse the result from the JSON-RPC response.
        if let Some(error) = response.get("error") {
            let msg = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Ok(McpToolResult {
                is_error: true,
                content: vec![McpContent::Text {
                    text: format!("MCP error: {}", msg),
                }],
            });
        }

        let result = response
            .get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let is_error = result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let content = if let Some(content_array) = result.get("content").and_then(|c| c.as_array())
        {
            content_array
                .iter()
                .filter_map(|item| {
                    let content_type = item.get("type")?.as_str()?;
                    match content_type {
                        "text" => {
                            let text = item.get("text")?.as_str()?.to_string();
                            Some(McpContent::Text { text })
                        }
                        "image" => {
                            let data = item.get("data")?.as_str()?.to_string();
                            let mime_type = item
                                .get("mimeType")
                                .and_then(|m| m.as_str())
                                .unwrap_or("application/octet-stream")
                                .to_string();
                            Some(McpContent::Image { data, mime_type })
                        }
                        "resource" => {
                            let uri = item.get("uri")?.as_str()?.to_string();
                            Some(McpContent::Resource { uri })
                        }
                        _ => None,
                    }
                })
                .collect()
        } else {
            // Fallback: wrap the entire result as text.
            vec![McpContent::Text {
                text: result.to_string(),
            }]
        };

        Ok(McpToolResult { is_error, content })
    }

    // -- Private transport methods --

    async fn connect_stdio(
        &mut self,
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<()> {
        info!(
            command = %command,
            args = ?args,
            "spawning MCP subprocess via stdio transport"
        );

        // Spawn the MCP server subprocess with stdin/stdout piped.
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn MCP server: {} {:?}", command, args))?;

        let mut stdin = child
            .stdin
            .take()
            .context("failed to capture MCP subprocess stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("failed to capture MCP subprocess stdout")?;
        let mut stdout_reader = BufReader::new(stdout);

        // --- JSON-RPC initialize handshake ---

        // Step 1: Send initialize request.
        let init_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "rvagent",
                    "version": "0.1.0"
                }
            }
        });
        self.next_id += 1;

        let mut init_line = serde_json::to_string(&init_request)
            .context("failed to serialize initialize request")?;
        init_line.push('\n');

        stdin
            .write_all(init_line.as_bytes())
            .await
            .context("failed to write initialize request")?;
        stdin
            .flush()
            .await
            .context("failed to flush after initialize")?;

        // Step 2: Read initialize response.
        let mut response_line = String::new();
        stdout_reader
            .read_line(&mut response_line)
            .await
            .context("failed to read initialize response")?;

        let init_response: serde_json::Value = serde_json::from_str(response_line.trim())
            .context("failed to parse initialize JSON-RPC response")?;

        debug!(
            response = %init_response,
            "MCP initialize response received"
        );

        if init_response.get("error").is_some() {
            let msg = init_response["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            anyhow::bail!("MCP initialize failed: {}", msg);
        }

        // Step 3: Send initialized notification (no id, no response expected).
        let initialized_notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        let mut notif_line = serde_json::to_string(&initialized_notification)
            .context("failed to serialize initialized notification")?;
        notif_line.push('\n');

        stdin
            .write_all(notif_line.as_bytes())
            .await
            .context("failed to write initialized notification")?;
        stdin
            .flush()
            .await
            .context("failed to flush after initialized notification")?;

        // Step 4: Call tools/list to discover available tools.
        let tools_list_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": "tools/list",
            "params": {}
        });
        self.next_id += 1;

        let mut tools_line = serde_json::to_string(&tools_list_request)
            .context("failed to serialize tools/list request")?;
        tools_line.push('\n');

        stdin
            .write_all(tools_line.as_bytes())
            .await
            .context("failed to write tools/list request")?;
        stdin
            .flush()
            .await
            .context("failed to flush after tools/list")?;

        let mut tools_response_line = String::new();
        stdout_reader
            .read_line(&mut tools_response_line)
            .await
            .context("failed to read tools/list response")?;

        let tools_response: serde_json::Value = serde_json::from_str(tools_response_line.trim())
            .context("failed to parse tools/list JSON-RPC response")?;

        debug!(
            response = %tools_response,
            "MCP tools/list response received"
        );

        // Parse tools from the response.
        if let Some(tools_array) = tools_response
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(|t| t.as_array())
        {
            self.tools = tools_array
                .iter()
                .filter_map(|t| serde_json::from_value::<McpToolDef>(t.clone()).ok())
                .collect();
        } else {
            warn!(
                server = %self.config.name,
                "tools/list response did not contain a tools array"
            );
            self.tools = Vec::new();
        }

        info!(
            server = %self.config.name,
            tool_count = self.tools.len(),
            "MCP stdio handshake complete"
        );

        // Store handles for later communication.
        self.stdin = Some(stdin);
        self.stdout = Some(stdout_reader);
        self.child = Some(child);

        Ok(())
    }

    async fn connect_sse(&mut self, url: &str, _auth: Option<&str>) -> Result<()> {
        // TODO: Connect to SSE endpoint, perform initialize, discover tools.
        info!(url = %url, "SSE MCP transport — stub connect");

        self.tools = Vec::new();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MCP registry
// ---------------------------------------------------------------------------

/// Registry of all connected MCP servers and their tools.
pub struct McpRegistry {
    clients: Vec<McpClient>,
}

impl McpRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            clients: Vec::new(),
        }
    }

    /// Add and connect an MCP server.
    pub async fn add_server(&mut self, config: McpServerConfig) -> Result<()> {
        let mut client = McpClient::new(config);
        client.connect().await?;
        self.clients.push(client);
        Ok(())
    }

    /// Get all discovered tools across all connected servers.
    pub fn all_tools(&self) -> Vec<&McpToolDef> {
        self.clients
            .iter()
            .flat_map(|c| c.tools())
            .collect()
    }

    /// Find which client owns a given tool name (immutable).
    #[allow(dead_code)]
    pub fn find_tool_client(&self, tool_name: &str) -> Option<&McpClient> {
        self.clients
            .iter()
            .find(|c| c.tools().iter().any(|t| t.name == tool_name))
    }

    /// Find the index of the client that owns a given tool name.
    fn find_tool_client_index(&self, tool_name: &str) -> Option<usize> {
        self.clients
            .iter()
            .position(|c| c.tools().iter().any(|t| t.name == tool_name))
    }

    /// Call a tool by name, routing to the appropriate MCP server.
    pub async fn call_tool(&mut self, name: &str, arguments: serde_json::Value) -> Result<McpToolResult> {
        let idx = self
            .find_tool_client_index(name)
            .with_context(|| format!("no MCP server provides tool '{}'", name))?;

        self.clients[idx]
            .call_tool(&McpToolCall {
                name: name.to_string(),
                arguments,
            })
            .await
    }

    /// Number of connected servers.
    pub fn server_count(&self) -> usize {
        self.clients.len()
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers: MCP tool → rvAgent Tool schema
// ---------------------------------------------------------------------------

/// Convert an MCP tool definition to the format expected by rvAgent's
/// tool registration system.
pub fn mcp_tool_to_agent_schema(tool: &McpToolDef) -> serde_json::Value {
    serde_json::json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "source": "mcp",
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_tool_def_serde() {
        let tool = McpToolDef {
            name: "read_file".into(),
            description: "Read a file".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                }
            }),
        };
        let json = serde_json::to_string(&tool).unwrap();
        let back: McpToolDef = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "read_file");
    }

    #[test]
    fn test_mcp_transport_stdio_serde() {
        let config = McpServerConfig {
            name: "test".into(),
            transport: McpTransport::Stdio {
                command: "node".into(),
                args: vec!["server.js".into()],
                env: HashMap::new(),
            },
            connected: false,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("stdio"));
        let back: McpServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "test");
    }

    #[test]
    fn test_mcp_transport_sse_serde() {
        let config = McpServerConfig {
            name: "remote".into(),
            transport: McpTransport::Sse {
                url: "https://example.com/sse".into(),
                auth: Some("Bearer token".into()),
            },
            connected: false,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("sse"));
    }

    #[test]
    fn test_mcp_tool_result_text_content() {
        let result = McpToolResult {
            is_error: false,
            content: vec![
                McpContent::Text {
                    text: "line1".into(),
                },
                McpContent::Text {
                    text: "line2".into(),
                },
                McpContent::Image {
                    data: "...".into(),
                    mime_type: "image/png".into(),
                },
            ],
        };
        assert_eq!(result.text_content(), "line1\nline2");
    }

    #[test]
    fn test_mcp_tool_to_agent_schema() {
        let tool = McpToolDef {
            name: "search".into(),
            description: "Search files".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let schema = mcp_tool_to_agent_schema(&tool);
        assert_eq!(schema["name"], "search");
        assert_eq!(schema["source"], "mcp");
    }

    #[test]
    fn test_mcp_registry_new() {
        let registry = McpRegistry::new();
        assert_eq!(registry.server_count(), 0);
        assert!(registry.all_tools().is_empty());
    }
}
