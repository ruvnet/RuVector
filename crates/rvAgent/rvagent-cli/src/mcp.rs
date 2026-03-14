//! MCP (Model Context Protocol) client integration for rvAgent CLI.
//!
//! Connects to external MCP servers, discovers their available tools,
//! and translates MCP tool schemas into the rvAgent `Tool` trait format
//! so they can be used seamlessly in the agent pipeline.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

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
}

impl McpClient {
    /// Create a new MCP client for the given server config.
    pub fn new(config: McpServerConfig) -> Self {
        Self {
            config,
            tools: Vec::new(),
        }
    }

    /// Connect to the MCP server and discover available tools.
    ///
    /// For stdio transport, this spawns the subprocess and performs the
    /// initialize / tools/list handshake. For SSE, it connects to the
    /// endpoint and subscribes to events.
    pub async fn connect(&mut self) -> Result<()> {
        info!(server = %self.config.name, "connecting to MCP server");

        match &self.config.transport {
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

    /// Call a tool on this MCP server.
    pub async fn call_tool(&self, call: &McpToolCall) -> Result<McpToolResult> {
        if !self.config.connected {
            anyhow::bail!("MCP server '{}' is not connected", self.config.name);
        }

        // TODO: Implement actual MCP protocol communication.
        // For now, return a stub result.
        warn!(
            server = %self.config.name,
            tool = %call.name,
            "MCP tool call stub — not yet implemented"
        );

        Ok(McpToolResult {
            is_error: false,
            content: vec![McpContent::Text {
                text: format!(
                    "[MCP stub] Tool '{}' called on server '{}'",
                    call.name, self.config.name
                ),
            }],
        })
    }

    // -- Private transport methods --

    async fn connect_stdio(
        &mut self,
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<()> {
        // TODO: Spawn subprocess, perform JSON-RPC initialize handshake,
        // then call tools/list to populate self.tools.
        info!(
            command = %command,
            args = ?args,
            "stdio MCP transport — stub connect"
        );

        // Placeholder: no tools discovered until real protocol is implemented.
        self.tools = Vec::new();
        Ok(())
    }

    async fn connect_sse(&mut self, url: &str, auth: Option<&str>) -> Result<()> {
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

    /// Find which client owns a given tool name.
    pub fn find_tool_client(&self, tool_name: &str) -> Option<&McpClient> {
        self.clients
            .iter()
            .find(|c| c.tools().iter().any(|t| t.name == tool_name))
    }

    /// Call a tool by name, routing to the appropriate MCP server.
    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<McpToolResult> {
        let client = self
            .find_tool_client(name)
            .with_context(|| format!("no MCP server provides tool '{}'", name))?;

        client
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
