//! `rvagent-mcp` — Model Context Protocol integration for rvAgent.
//!
//! This crate provides a complete MCP implementation including:
//!
//! - [`protocol`] — JSON-RPC 2.0 protocol types for MCP
//! - [`registry`] — Thread-safe MCP tool registry with handler dispatch
//! - [`resources`] — Resource providers (static, file, template) and registry
//! - [`transport`] — Transport abstraction (stdio, memory) for MCP messages
//! - [`server`] — MCP server that routes requests to tools/resources
//! - [`client`] — MCP client for connecting to external MCP servers
//! - [`middleware`] — MCP middleware for the rvagent pipeline
//! - [`topology`] — Topology strategies for multi-agent routing
//! - [`skills_bridge`] — Skills format bridge (Claude Code, Codex)

pub mod client;
pub mod middleware;
pub mod protocol;
pub mod registry;
pub mod resources;
pub mod server;
pub mod skills_bridge;
pub mod topology;
pub mod transport;

// Re-export key types at crate root.
pub use client::McpClient;
pub use protocol::{
    Content, JsonRpcError, JsonRpcRequest, JsonRpcResponse, McpMethod, McpPrompt, McpResource,
    McpResourceTemplate, McpTool, ServerCapabilities,
};
pub use registry::{McpToolDefinition, McpToolHandler, McpToolRegistry};
pub use resources::{ResourceProvider, ResourceRegistry};
pub use server::{McpServer, McpServerConfig};
pub use topology::{
    ConsensusType, NodeRole, NodeStatus, TopologyConfig, TopologyNode, TopologyRouter, TopologyType,
};
pub use transport::{MemoryTransport, Transport};

/// Error types for the MCP crate.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// JSON-RPC protocol error.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// Tool not found or execution error.
    #[error("tool error: {0}")]
    Tool(String),

    /// Resource not found or read error.
    #[error("resource error: {0}")]
    Resource(String),

    /// Transport layer error.
    #[error("transport error: {0}")]
    Transport(String),

    /// Server configuration or lifecycle error.
    #[error("server error: {0}")]
    Server(String),

    /// Client connection or request error.
    #[error("client error: {0}")]
    Client(String),

    /// Serialization/deserialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, McpError>;

impl McpError {
    pub fn protocol(msg: impl Into<String>) -> Self {
        Self::Protocol(msg.into())
    }
    pub fn tool(msg: impl Into<String>) -> Self {
        Self::Tool(msg.into())
    }
    pub fn resource(msg: impl Into<String>) -> Self {
        Self::Resource(msg.into())
    }
    pub fn transport(msg: impl Into<String>) -> Self {
        Self::Transport(msg.into())
    }
    pub fn server(msg: impl Into<String>) -> Self {
        Self::Server(msg.into())
    }
    pub fn client(msg: impl Into<String>) -> Self {
        Self::Client(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = McpError::protocol("invalid request");
        assert_eq!(e.to_string(), "protocol error: invalid request");
    }

    #[test]
    fn test_error_variants() {
        let cases: Vec<McpError> = vec![
            McpError::protocol("p"),
            McpError::tool("t"),
            McpError::resource("r"),
            McpError::transport("tr"),
            McpError::server("s"),
            McpError::client("c"),
        ];
        for e in &cases {
            assert!(!e.to_string().is_empty());
        }
    }

    #[test]
    fn test_from_json_error() {
        let bad: std::result::Result<serde_json::Value, _> = serde_json::from_str("{invalid");
        let mcp_err: McpError = bad.unwrap_err().into();
        assert!(matches!(mcp_err, McpError::Json(_)));
    }
}
