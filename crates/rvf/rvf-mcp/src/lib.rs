//! RVF MCP Server — exposes RVF memory, pattern, witness, and coordination
//! tools over the Model Context Protocol (JSON-RPC over stdio).
//!
//! Composes two adapters:
//! - `rvf-adapter-claude-flow` for memory, witness chains, coordination, learning
//! - `rvf-adapter-agentdb` for pattern storage (rewards, critiques, success tracking)

mod handlers;
mod protocol;
mod tools;

pub use handlers::{ToolCall, ToolResult};
pub use protocol::JsonRpcError;
pub use tools::ToolDef;

use std::collections::HashMap;
use std::path::PathBuf;

use rvf_adapter_agentdb::RvfPatternStore;
use rvf_adapter_claude_flow::{ClaudeFlowConfig, RvfMemoryStore};

/// Server configuration.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Directory for .rvf data files.
    pub data_dir: PathBuf,
    /// Vector dimensionality for embeddings.
    pub dimension: u16,
}

/// The RVF MCP server. Holds memory and pattern stores, dispatches tool calls.
pub struct RvfMcpServer {
    memory_store: RvfMemoryStore,
    pattern_store: RvfPatternStore,
    /// Key-value lookup for memories (key → vector_id).
    key_index: HashMap<String, (u64, String, Option<String>)>,
    config: ServerConfig,
}

impl RvfMcpServer {
    /// Create a new server, initializing .rvf stores in `config.data_dir`.
    pub fn new(config: ServerConfig) -> Result<Self, String> {
        std::fs::create_dir_all(&config.data_dir).map_err(|e| e.to_string())?;

        let cf_config = ClaudeFlowConfig::new(&config.data_dir, "mcp-server")
            .with_dimension(config.dimension)
            .with_namespace("default");

        let memory_store =
            RvfMemoryStore::create(cf_config).map_err(|e| format!("memory store: {e}"))?;

        let pattern_path = config.data_dir.join("patterns.rvf");
        let pattern_store = RvfPatternStore::create(&pattern_path, config.dimension)
            .map_err(|e| format!("pattern store: {e}"))?;

        Ok(Self {
            memory_store,
            pattern_store,
            key_index: HashMap::new(),
            config,
        })
    }

    /// List all available tools with their descriptions and input schemas.
    pub fn list_tools(&self) -> Vec<ToolDef> {
        tools::all_tools()
    }

    /// Handle a tool call by name and arguments. Returns a ToolResult.
    pub fn handle_tool_call(&mut self, call: ToolCall) -> Result<ToolResult, String> {
        handlers::dispatch(self, call)
    }

    /// Handle a raw JSON-RPC request string. Returns a JSON-RPC response string.
    pub fn handle_jsonrpc(&mut self, request: &str) -> Result<String, String> {
        protocol::handle_jsonrpc(self, request)
    }
}
