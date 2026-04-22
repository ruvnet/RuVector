//! MCP tool registry — thread-safe registration and lookup of MCP tools.
//!
//! Provides [`McpToolRegistry`] backed by `DashMap` for concurrent access,
//! the [`McpToolHandler`] trait for tool execution, and a bridge adapter
//! to wrap rvagent-tools `Tool` trait implementations.

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use serde_json::Value;

use crate::protocol::{Content, McpTool, ToolCallResult};
use crate::{McpError, Result};

// ---------------------------------------------------------------------------
// McpToolHandler trait
// ---------------------------------------------------------------------------

/// Async handler for an MCP tool invocation.
#[async_trait]
pub trait McpToolHandler: Send + Sync {
    /// Execute the tool with the given arguments.
    async fn execute(&self, arguments: Value) -> Result<ToolCallResult>;
}

// ---------------------------------------------------------------------------
// McpToolDefinition
// ---------------------------------------------------------------------------

/// A registered MCP tool: metadata + handler.
pub struct McpToolDefinition {
    /// Tool name (unique identifier).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: Value,
    /// The handler that executes this tool.
    pub handler: Arc<dyn McpToolHandler>,
}

impl McpToolDefinition {
    /// Convert to the wire-format `McpTool` (without handler).
    pub fn to_mcp_tool(&self) -> McpTool {
        McpTool {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }
}

impl Clone for McpToolDefinition {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
            handler: Arc::clone(&self.handler),
        }
    }
}

impl std::fmt::Debug for McpToolDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolDefinition")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// McpToolRegistry
// ---------------------------------------------------------------------------

/// Thread-safe registry of MCP tools, backed by `DashMap`.
#[derive(Clone)]
pub struct McpToolRegistry {
    tools: Arc<DashMap<String, McpToolDefinition>>,
}

impl McpToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(DashMap::new()),
        }
    }

    /// Register a tool. Returns an error if a tool with the same name exists.
    pub fn register_tool(&self, tool: McpToolDefinition) -> Result<()> {
        if self.tools.contains_key(&tool.name) {
            return Err(McpError::tool(format!(
                "tool '{}' is already registered",
                tool.name
            )));
        }
        self.tools.insert(tool.name.clone(), tool);
        Ok(())
    }

    /// Unregister a tool by name. Returns an error if the tool does not exist.
    pub fn unregister_tool(&self, name: &str) -> Result<()> {
        if self.tools.remove(name).is_none() {
            return Err(McpError::tool(format!("tool '{}' not found", name)));
        }
        Ok(())
    }

    /// Look up a tool by name.
    pub fn get_tool(&self, name: &str) -> Option<McpToolDefinition> {
        self.tools.get(name).map(|r| r.value().clone())
    }

    /// List all registered tools (sorted by name for determinism).
    pub fn list_tools(&self) -> Vec<McpToolDefinition> {
        let mut tools: Vec<_> = self.tools.iter().map(|r| r.value().clone()).collect();
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        tools
    }

    /// List as wire-format `McpTool` objects.
    pub fn list_mcp_tools(&self) -> Vec<McpTool> {
        self.list_tools().iter().map(|t| t.to_mcp_tool()).collect()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Execute a tool by name with the given arguments.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolCallResult> {
        let tool = self
            .get_tool(name)
            .ok_or_else(|| McpError::tool(format!("tool '{}' not found", name)))?;
        tool.handler.execute(arguments).await
    }

    /// Validate arguments against a tool's input schema (basic check).
    pub fn validate_args(&self, name: &str, args: &Value) -> Result<()> {
        let tool = self
            .get_tool(name)
            .ok_or_else(|| McpError::tool(format!("tool '{}' not found", name)))?;
        // Basic validation: if schema requires an object, args must be object
        if let Some(schema_type) = tool.input_schema.get("type").and_then(|v| v.as_str()) {
            if schema_type == "object" && !args.is_object() {
                return Err(McpError::tool(format!(
                    "tool '{}' expects object arguments",
                    name
                )));
            }
        }
        // Check required properties
        if let Some(required) = tool.input_schema.get("required").and_then(|v| v.as_array()) {
            if let Some(obj) = args.as_object() {
                for req in required {
                    if let Some(field) = req.as_str() {
                        if !obj.contains_key(field) {
                            return Err(McpError::tool(format!(
                                "tool '{}' missing required argument '{}'",
                                name, field
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for McpToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in tool handlers
// ---------------------------------------------------------------------------

/// Ping handler — returns "pong".
pub struct PingHandler;

#[async_trait]
impl McpToolHandler for PingHandler {
    async fn execute(&self, _arguments: Value) -> Result<ToolCallResult> {
        Ok(ToolCallResult {
            content: vec![Content::text("pong")],
            is_error: false,
        })
    }
}

/// Echo handler — returns the input text.
pub struct EchoHandler;

#[async_trait]
impl McpToolHandler for EchoHandler {
    async fn execute(&self, arguments: Value) -> Result<ToolCallResult> {
        let text = arguments.get("text").and_then(|v| v.as_str()).unwrap_or("");
        Ok(ToolCallResult {
            content: vec![Content::text(text)],
            is_error: false,
        })
    }
}

/// ListCapabilities handler — returns the server capability summary.
pub struct ListCapabilitiesHandler {
    capabilities: Value,
}

impl ListCapabilitiesHandler {
    /// Create with serialized capabilities.
    pub fn new(capabilities: Value) -> Self {
        Self { capabilities }
    }
}

#[async_trait]
impl McpToolHandler for ListCapabilitiesHandler {
    async fn execute(&self, _arguments: Value) -> Result<ToolCallResult> {
        let text =
            serde_json::to_string_pretty(&self.capabilities).unwrap_or_else(|_| "{}".to_string());
        Ok(ToolCallResult {
            content: vec![Content::text(text)],
            is_error: false,
        })
    }
}

/// Register built-in MCP tools (ping, echo, list_capabilities).
pub fn register_builtins(registry: &McpToolRegistry, capabilities: Value) -> Result<()> {
    registry.register_tool(McpToolDefinition {
        name: "ping".into(),
        description: "Responds with pong — used for health checks".into(),
        input_schema: serde_json::json!({"type": "object", "properties": {}}),
        handler: Arc::new(PingHandler),
    })?;

    registry.register_tool(McpToolDefinition {
        name: "echo".into(),
        description: "Echoes back the provided text".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "Text to echo" }
            },
            "required": ["text"]
        }),
        handler: Arc::new(EchoHandler),
    })?;

    registry.register_tool(McpToolDefinition {
        name: "list_capabilities".into(),
        description: "Lists the server's capabilities".into(),
        input_schema: serde_json::json!({"type": "object", "properties": {}}),
        handler: Arc::new(ListCapabilitiesHandler::new(capabilities)),
    })?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_handler() -> Arc<dyn McpToolHandler> {
        Arc::new(PingHandler)
    }

    fn make_tool(name: &str) -> McpToolDefinition {
        McpToolDefinition {
            name: name.into(),
            description: format!("{} tool", name),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
            handler: make_handler(),
        }
    }

    #[test]
    fn test_register_and_get() {
        let reg = McpToolRegistry::new();
        reg.register_tool(make_tool("alpha")).unwrap();
        let t = reg.get_tool("alpha");
        assert!(t.is_some());
        assert_eq!(t.unwrap().name, "alpha");
    }

    #[test]
    fn test_register_duplicate() {
        let reg = McpToolRegistry::new();
        reg.register_tool(make_tool("dup")).unwrap();
        let err = reg.register_tool(make_tool("dup"));
        assert!(err.is_err());
    }

    #[test]
    fn test_unregister() {
        let reg = McpToolRegistry::new();
        reg.register_tool(make_tool("rm")).unwrap();
        assert_eq!(reg.len(), 1);
        reg.unregister_tool("rm").unwrap();
        assert_eq!(reg.len(), 0);
        assert!(reg.get_tool("rm").is_none());
    }

    #[test]
    fn test_unregister_not_found() {
        let reg = McpToolRegistry::new();
        let err = reg.unregister_tool("nope");
        assert!(err.is_err());
    }

    #[test]
    fn test_list_tools_sorted() {
        let reg = McpToolRegistry::new();
        reg.register_tool(make_tool("charlie")).unwrap();
        reg.register_tool(make_tool("alpha")).unwrap();
        reg.register_tool(make_tool("bravo")).unwrap();
        let names: Vec<_> = reg.list_tools().iter().map(|t| t.name.clone()).collect();
        assert_eq!(names, vec!["alpha", "bravo", "charlie"]);
    }

    #[test]
    fn test_list_mcp_tools() {
        let reg = McpToolRegistry::new();
        reg.register_tool(make_tool("test")).unwrap();
        let mcp_tools = reg.list_mcp_tools();
        assert_eq!(mcp_tools.len(), 1);
        assert_eq!(mcp_tools[0].name, "test");
    }

    #[test]
    fn test_len_and_is_empty() {
        let reg = McpToolRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        reg.register_tool(make_tool("x")).unwrap();
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_get_nonexistent() {
        let reg = McpToolRegistry::new();
        assert!(reg.get_tool("missing").is_none());
    }

    #[test]
    fn test_to_mcp_tool() {
        let def = make_tool("test");
        let mcp = def.to_mcp_tool();
        assert_eq!(mcp.name, "test");
        assert_eq!(mcp.description, "test tool");
    }

    #[test]
    fn test_tool_definition_debug() {
        let def = make_tool("dbg");
        let dbg = format!("{:?}", def);
        assert!(dbg.contains("dbg"));
    }

    #[test]
    fn test_tool_definition_clone() {
        let def = make_tool("orig");
        let cloned = def.clone();
        assert_eq!(cloned.name, "orig");
    }

    #[tokio::test]
    async fn test_call_tool_ping() {
        let reg = McpToolRegistry::new();
        reg.register_tool(make_tool("ping")).unwrap();
        // Replace with actual ping handler
        let reg2 = McpToolRegistry::new();
        reg2.register_tool(McpToolDefinition {
            name: "ping".into(),
            description: "ping".into(),
            input_schema: serde_json::json!({}),
            handler: Arc::new(PingHandler),
        })
        .unwrap();
        let result = reg2.call_tool("ping", Value::Null).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
    }

    #[tokio::test]
    async fn test_call_tool_echo() {
        let reg = McpToolRegistry::new();
        reg.register_tool(McpToolDefinition {
            name: "echo".into(),
            description: "echo".into(),
            input_schema: serde_json::json!({}),
            handler: Arc::new(EchoHandler),
        })
        .unwrap();
        let result = reg
            .call_tool("echo", serde_json::json!({"text": "hello"}))
            .await
            .unwrap();
        match &result.content[0] {
            Content::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn test_call_tool_not_found() {
        let reg = McpToolRegistry::new();
        let err = reg.call_tool("missing", Value::Null).await;
        assert!(err.is_err());
    }

    #[test]
    fn test_validate_args_object_check() {
        let reg = McpToolRegistry::new();
        reg.register_tool(McpToolDefinition {
            name: "obj".into(),
            description: "needs object".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
            handler: make_handler(),
        })
        .unwrap();
        // Object passes
        assert!(reg.validate_args("obj", &serde_json::json!({})).is_ok());
        // Non-object fails
        assert!(reg
            .validate_args("obj", &serde_json::json!("string"))
            .is_err());
    }

    #[test]
    fn test_validate_args_required() {
        let reg = McpToolRegistry::new();
        reg.register_tool(McpToolDefinition {
            name: "req".into(),
            description: "has required".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
            handler: make_handler(),
        })
        .unwrap();
        // Missing required
        assert!(reg.validate_args("req", &serde_json::json!({})).is_err());
        // Present
        assert!(reg
            .validate_args("req", &serde_json::json!({"name": "val"}))
            .is_ok());
    }

    #[test]
    fn test_validate_args_tool_not_found() {
        let reg = McpToolRegistry::new();
        assert!(reg.validate_args("nope", &Value::Null).is_err());
    }

    #[tokio::test]
    async fn test_register_builtins() {
        let reg = McpToolRegistry::new();
        register_builtins(&reg, serde_json::json!({"tools": true})).unwrap();
        assert_eq!(reg.len(), 3);
        assert!(reg.get_tool("ping").is_some());
        assert!(reg.get_tool("echo").is_some());
        assert!(reg.get_tool("list_capabilities").is_some());
    }

    #[tokio::test]
    async fn test_list_capabilities_handler() {
        let h = ListCapabilitiesHandler::new(serde_json::json!({"tools": true}));
        let result = h.execute(Value::Null).await.unwrap();
        match &result.content[0] {
            Content::Text { text } => assert!(text.contains("tools")),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn test_registry_default() {
        let reg = McpToolRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_clone() {
        let reg = McpToolRegistry::new();
        reg.register_tool(make_tool("shared")).unwrap();
        let reg2 = reg.clone();
        assert!(reg2.get_tool("shared").is_some());
    }
}
