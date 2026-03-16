//! MCP Protocol support for rvagent-wasm.
//!
//! Provides `WasmMcpServer` which implements the MCP protocol in WASM,
//! allowing the agent's tools to be exposed via MCP JSON-RPC in the browser.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::backends::WasmStateBackend;
use crate::bridge::to_js_value;
use crate::gallery::WasmGallery;
use crate::tools::{TodoItem, ToolRequest, ToolResult, WasmToolExecutor};

// ---------------------------------------------------------------------------
// Security Constants
// ---------------------------------------------------------------------------

/// Maximum JSON-RPC request size (100 KB)
pub const MAX_REQUEST_SIZE: usize = 100 * 1024;

/// Maximum path argument length (256 characters)
pub const MAX_PATH_LENGTH: usize = 256;

/// Maximum content argument length (1 MB)
pub const MAX_CONTENT_LENGTH: usize = 1024 * 1024;

/// Maximum gallery search query length
pub const MAX_SEARCH_QUERY_LENGTH: usize = 256;

// ---------------------------------------------------------------------------
// MCP Protocol Types
// ---------------------------------------------------------------------------

/// MCP tool definition (for tools/list response).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<serde_json::Value>, code: i32, message: &str) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.to_string(),
                data: None,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// MCP Server Info
// ---------------------------------------------------------------------------

/// Server capabilities for MCP initialize response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolCapabilities>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceCapabilities>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapabilities {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapabilities {
    #[serde(default)]
    pub list_changed: bool,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            tools: Some(ToolCapabilities { list_changed: false }),
            resources: Some(ResourceCapabilities { list_changed: false }),
        }
    }
}

// ---------------------------------------------------------------------------
// WasmMcpServer — browser-based MCP server
// ---------------------------------------------------------------------------

/// WASM MCP Server — runs the MCP protocol entirely in the browser.
///
/// This server exposes rvAgent tools via MCP JSON-RPC, enabling integration
/// with MCP clients without requiring a separate server process.
///
/// # Example (JavaScript)
/// ```js
/// const mcp = new WasmMcpServer("rvagent-wasm");
///
/// // Handle request
/// const response = mcp.handleRequest(JSON.stringify({
///     jsonrpc: "2.0",
///     id: 1,
///     method: "tools/list",
///     params: {}
/// }));
/// console.log(response);
/// ```
#[wasm_bindgen]
pub struct WasmMcpServer {
    name: String,
    version: String,
    backend: WasmStateBackend,
    todos: Vec<TodoItem>,
    gallery: WasmGallery,
    initialized: bool,
}

#[wasm_bindgen]
impl WasmMcpServer {
    /// Create a new WasmMcpServer with the given name.
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend: WasmStateBackend::new(),
            todos: Vec::new(),
            gallery: WasmGallery::new(),
            initialized: false,
        }
    }

    /// Get the gallery instance for direct access.
    pub fn gallery(&self) -> Result<JsValue, JsValue> {
        to_js_value(&serde_json::json!({
            "count": self.gallery.count(),
            "active": self.gallery.get_active(),
        }))
    }

    /// Handle a JSON-RPC request and return a JSON-RPC response.
    pub fn handle_request(&mut self, request_json: &str) -> Result<JsValue, JsValue> {
        // Security: Check request size
        if request_json.len() > MAX_REQUEST_SIZE {
            return to_js_value(&JsonRpcResponse::error(
                None,
                -32600,
                &format!("Request size {} exceeds maximum {}", request_json.len(), MAX_REQUEST_SIZE),
            ));
        }

        let request: JsonRpcRequest = serde_json::from_str(request_json)
            .map_err(|e| JsValue::from_str(&format!("invalid JSON-RPC: {}", e)))?;

        let response = self.process_request(&request);
        to_js_value(&response)
    }

    /// Get the list of available tools as JSON.
    pub fn list_tools(&self) -> Result<JsValue, JsValue> {
        to_js_value(&self.get_mcp_tools())
    }

    /// Execute a tool by name with JSON parameters.
    pub fn call_tool(&mut self, name: &str, params_json: &str) -> Result<JsValue, JsValue> {
        let params: serde_json::Value = serde_json::from_str(params_json)
            .map_err(|e| JsValue::from_str(&format!("invalid params: {}", e)))?;

        let result = self.execute_tool(name, &params);
        to_js_value(&result)
    }

    /// Check if the server has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the server name.
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Get the server version.
    pub fn version(&self) -> String {
        self.version.clone()
    }
}

impl WasmMcpServer {
    /// Process a JSON-RPC request and return a response.
    fn process_request(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request.id.clone()),
            "ping" => self.handle_ping(request.id.clone()),
            "tools/list" => self.handle_tools_list(request.id.clone()),
            "tools/call" => self.handle_tools_call(request.id.clone(), &request.params),
            "resources/list" => self.handle_resources_list(request.id.clone()),
            "prompts/list" => self.handle_prompts_list(request.id.clone()),
            // Gallery methods
            "gallery/list" => self.handle_gallery_list(request.id.clone()),
            "gallery/search" => self.handle_gallery_search(request.id.clone(), &request.params),
            "gallery/get" => self.handle_gallery_get(request.id.clone(), &request.params),
            "gallery/load" => self.handle_gallery_load(request.id.clone(), &request.params),
            "gallery/configure" => self.handle_gallery_configure(request.id.clone(), &request.params),
            "gallery/categories" => self.handle_gallery_categories(request.id.clone()),
            _ => JsonRpcResponse::error(
                request.id.clone(),
                -32601,
                &format!("method not found: {}", request.method),
            ),
        }
    }

    fn handle_initialize(&mut self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        self.initialized = true;
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": self.name,
                    "version": self.version,
                },
                "capabilities": ServerCapabilities::default(),
            }),
        )
    }

    fn handle_ping(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        JsonRpcResponse::success(id, serde_json::json!({}))
    }

    fn handle_tools_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "tools": self.get_mcp_tools(),
            }),
        )
    }

    fn handle_tools_call(
        &mut self,
        id: Option<serde_json::Value>,
        params: &serde_json::Value,
    ) -> JsonRpcResponse {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        let result = self.execute_tool(name, &arguments);

        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": result.output,
                }],
                "isError": !result.success,
            }),
        )
    }

    fn handle_resources_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        // Return virtual filesystem contents as resources
        let resources: Vec<serde_json::Value> = self
            .backend
            .list_files()
            .iter()
            .map(|path| {
                serde_json::json!({
                    "uri": format!("file://{}", path),
                    "name": path,
                    "mimeType": "text/plain",
                })
            })
            .collect();

        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "resources": resources,
            }),
        )
    }

    fn handle_prompts_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        // Return prompts from active gallery template if any
        if let Some(ref active_id) = self.gallery.get_active() {
            if let Some(template) = self.gallery.all_templates()
                .find(|t| &t.id == active_id)
            {
                let prompts: Vec<serde_json::Value> = template.prompts.iter()
                    .map(|p| serde_json::json!({
                        "name": p.name,
                        "description": format!("Prompt v{}", p.version),
                    }))
                    .collect();
                return JsonRpcResponse::success(id, serde_json::json!({ "prompts": prompts }));
            }
        }
        JsonRpcResponse::success(id, serde_json::json!({ "prompts": [] }))
    }

    // -------------------------------------------------------------------------
    // Gallery Handlers
    // -------------------------------------------------------------------------

    fn handle_gallery_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        let templates: Vec<serde_json::Value> = self.gallery.all_templates()
            .map(|t| serde_json::json!({
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "version": t.version,
                "author": t.author,
                "tags": t.tags,
                "builtin": t.builtin,
            }))
            .collect();

        JsonRpcResponse::success(id, serde_json::json!({
            "templates": templates,
            "count": templates.len(),
            "active": self.gallery.get_active(),
        }))
    }

    fn handle_gallery_search(
        &self,
        id: Option<serde_json::Value>,
        params: &serde_json::Value,
    ) -> JsonRpcResponse {
        let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");

        // Security: Validate query length
        if query.len() > MAX_SEARCH_QUERY_LENGTH {
            return JsonRpcResponse::error(
                id,
                -32602,
                &format!("Query too long (max {} chars)", MAX_SEARCH_QUERY_LENGTH),
            );
        }

        let query_lower = query.to_lowercase();
        let terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut results: Vec<serde_json::Value> = self.gallery.all_templates()
            .filter_map(|t| {
                let mut score = 0.0f32;
                let name_lower = t.name.to_lowercase();
                let desc_lower = t.description.to_lowercase();

                for term in &terms {
                    if name_lower.contains(term) { score += 0.4; }
                    if desc_lower.contains(term) { score += 0.3; }
                    if t.tags.iter().any(|tag| tag.to_lowercase().contains(term)) { score += 0.3; }
                }

                if score > 0.0 {
                    Some(serde_json::json!({
                        "id": t.id,
                        "name": t.name,
                        "description": t.description,
                        "category": t.category,
                        "tags": t.tags,
                        "relevance": (score * 100.0).round() / 100.0,
                    }))
                } else {
                    None
                }
            })
            .collect();

        // Sort by relevance (descending)
        results.sort_by(|a, b| {
            let ra = a.get("relevance").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let rb = b.get("relevance").and_then(|v| v.as_f64()).unwrap_or(0.0);
            rb.partial_cmp(&ra).unwrap()
        });

        JsonRpcResponse::success(id, serde_json::json!({
            "results": results,
            "query": query,
        }))
    }

    fn handle_gallery_get(
        &self,
        id: Option<serde_json::Value>,
        params: &serde_json::Value,
    ) -> JsonRpcResponse {
        let template_id = params.get("id").and_then(|v| v.as_str()).unwrap_or("");

        if template_id.is_empty() {
            return JsonRpcResponse::error(id, -32602, "missing 'id' parameter");
        }

        let template = self.gallery.find_template(template_id);

        match template {
            Some(t) => JsonRpcResponse::success(id, serde_json::json!({
                "template": t,
            })),
            None => JsonRpcResponse::error(
                id,
                -32602,
                &format!("template not found: {}", template_id),
            ),
        }
    }

    fn handle_gallery_load(
        &mut self,
        id: Option<serde_json::Value>,
        params: &serde_json::Value,
    ) -> JsonRpcResponse {
        let template_id = params.get("id").and_then(|v| v.as_str()).unwrap_or("");

        if template_id.is_empty() {
            return JsonRpcResponse::error(id, -32602, "missing 'id' parameter");
        }

        // Find and activate template
        let template = self.gallery.find_template(template_id).cloned();

        match template {
            Some(t) => {
                self.gallery.set_active_id(Some(template_id.to_string()));

                // Build RVF container
                let rvf_bytes = t.to_rvf();

                JsonRpcResponse::success(id, serde_json::json!({
                    "loaded": true,
                    "template_id": template_id,
                    "name": t.name,
                    "rvf_size": rvf_bytes.len(),
                    "tools_count": t.tools.len(),
                    "prompts_count": t.prompts.len(),
                    "skills_count": t.skills.len(),
                    "mcp_tools_count": t.mcp_tools.len(),
                    "capabilities_count": t.capabilities.len(),
                    "has_orchestrator": t.orchestrator.is_some(),
                }))
            }
            None => JsonRpcResponse::error(
                id,
                -32602,
                &format!("template not found: {}", template_id),
            ),
        }
    }

    fn handle_gallery_configure(
        &mut self,
        id: Option<serde_json::Value>,
        params: &serde_json::Value,
    ) -> JsonRpcResponse {
        let config = params.get("config").cloned().unwrap_or(serde_json::json!({}));

        if self.gallery.get_active().is_none() {
            return JsonRpcResponse::error(id, -32602, "no active template - load one first");
        }

        self.gallery.set_config_overrides(config.clone());

        JsonRpcResponse::success(id, serde_json::json!({
            "configured": true,
            "active": self.gallery.get_active(),
            "config": config,
        }))
    }

    fn handle_gallery_categories(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        use std::collections::HashMap;

        let mut counts: HashMap<String, usize> = HashMap::new();

        for template in self.gallery.all_templates() {
            let cat = serde_json::to_string(&template.category)
                .unwrap()
                .trim_matches('"')
                .to_string();
            *counts.entry(cat).or_insert(0) += 1;
        }

        let categories: Vec<serde_json::Value> = counts.iter()
            .map(|(name, count)| serde_json::json!({
                "name": name,
                "count": count,
            }))
            .collect();

        JsonRpcResponse::success(id, serde_json::json!({
            "categories": categories,
        }))
    }

    /// Get MCP tool definitions from available tools.
    fn get_mcp_tools(&self) -> Vec<McpToolDef> {
        vec![
            McpToolDef {
                name: "read_file".to_string(),
                description: Some("Read the contents of a file from the virtual filesystem".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path to read" }
                    },
                    "required": ["path"]
                })),
            },
            McpToolDef {
                name: "write_file".to_string(),
                description: Some("Write content to a file in the virtual filesystem".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path to write" },
                        "content": { "type": "string", "description": "Content to write" }
                    },
                    "required": ["path", "content"]
                })),
            },
            McpToolDef {
                name: "edit_file".to_string(),
                description: Some("Apply a string replacement edit to a file".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path to edit" },
                        "old_string": { "type": "string", "description": "String to replace" },
                        "new_string": { "type": "string", "description": "Replacement string" }
                    },
                    "required": ["path", "old_string", "new_string"]
                })),
            },
            McpToolDef {
                name: "list_files".to_string(),
                description: Some("List all files in the virtual filesystem".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {}
                })),
            },
            McpToolDef {
                name: "write_todos".to_string(),
                description: Some("Write/update the todo list".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": { "type": "string" },
                                    "status": { "type": "string", "enum": ["pending", "in_progress", "completed"] }
                                },
                                "required": ["content", "status"]
                            }
                        }
                    },
                    "required": ["todos"]
                })),
            },
            // Gallery tools
            McpToolDef {
                name: "gallery_list".to_string(),
                description: Some("List all available agent templates in the gallery".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {}
                })),
            },
            McpToolDef {
                name: "gallery_search".to_string(),
                description: Some("Search for templates by name, description, or tags".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query" }
                    },
                    "required": ["query"]
                })),
            },
            McpToolDef {
                name: "gallery_get".to_string(),
                description: Some("Get detailed information about a specific template".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Template ID" }
                    },
                    "required": ["id"]
                })),
            },
            McpToolDef {
                name: "gallery_load".to_string(),
                description: Some("Load a template as the active agent configuration".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Template ID to load" }
                    },
                    "required": ["id"]
                })),
            },
            McpToolDef {
                name: "gallery_configure".to_string(),
                description: Some("Configure the active template with custom settings".to_string()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "config": { "type": "object", "description": "Configuration overrides" }
                    },
                    "required": ["config"]
                })),
            },
        ]
    }

    /// Execute a tool by name with the given arguments.
    fn execute_tool(&mut self, name: &str, args: &serde_json::Value) -> ToolResult {
        // Security: Validate input lengths
        let validate_path = |p: &str| -> Result<(), String> {
            if p.len() > MAX_PATH_LENGTH {
                return Err(format!("Path length {} exceeds maximum {}", p.len(), MAX_PATH_LENGTH));
            }
            if p.contains("..") {
                return Err("Path traversal (..) is not allowed".to_string());
            }
            Ok(())
        };

        let validate_content = |c: &str| -> Result<(), String> {
            if c.len() > MAX_CONTENT_LENGTH {
                return Err(format!("Content length {} exceeds maximum {}", c.len(), MAX_CONTENT_LENGTH));
            }
            Ok(())
        };

        // Convert MCP tool call to ToolRequest
        let request = match name {
            "read_file" => {
                let path = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if let Err(e) = validate_path(path) {
                    return ToolResult { success: false, output: e };
                }
                ToolRequest::ReadFile { path: path.into() }
            }
            "write_file" => {
                let path = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let content = args
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if let Err(e) = validate_path(path) {
                    return ToolResult { success: false, output: e };
                }
                if let Err(e) = validate_content(content) {
                    return ToolResult { success: false, output: e };
                }
                ToolRequest::WriteFile {
                    path: path.into(),
                    content: content.into(),
                }
            }
            "edit_file" => {
                let path = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let old_string = args
                    .get("old_string")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let new_string = args
                    .get("new_string")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if let Err(e) = validate_path(path) {
                    return ToolResult { success: false, output: e };
                }
                if let Err(e) = validate_content(old_string) {
                    return ToolResult { success: false, output: e };
                }
                if let Err(e) = validate_content(new_string) {
                    return ToolResult { success: false, output: e };
                }
                ToolRequest::EditFile {
                    path: path.into(),
                    old_string: old_string.into(),
                    new_string: new_string.into(),
                }
            }
            "list_files" => ToolRequest::ListFiles,
            "write_todos" => {
                let todos_val = args.get("todos").cloned().unwrap_or(serde_json::json!([]));
                let todos: Vec<TodoItem> = serde_json::from_value(todos_val).unwrap_or_default();
                // Security: Limit number of todos
                if todos.len() > 1000 {
                    return ToolResult {
                        success: false,
                        output: format!("Todo count {} exceeds maximum 1000", todos.len()),
                    };
                }
                ToolRequest::WriteTodos { todos }
            }
            _ => {
                return ToolResult {
                    success: false,
                    output: format!("unknown tool: {}", name),
                };
            }
        };

        let mut executor = WasmToolExecutor::new(&mut self.backend, &mut self.todos);
        executor.execute(&request)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rpc_response_success() {
        let resp = JsonRpcResponse::success(Some(serde_json::json!(1)), serde_json::json!({"ok": true}));
        assert_eq!(resp.jsonrpc, "2.0");
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_json_rpc_response_error() {
        let resp = JsonRpcResponse::error(Some(serde_json::json!(1)), -32600, "invalid request");
        assert_eq!(resp.jsonrpc, "2.0");
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, -32600);
    }

    #[test]
    fn test_mcp_server_initialize() {
        let mut server = WasmMcpServer::new("test-server");
        assert!(!server.is_initialized());

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "initialize".to_string(),
            params: serde_json::json!({}),
        };

        let response = server.process_request(&request);
        assert!(server.is_initialized());
        assert!(response.result.is_some());
    }

    #[test]
    fn test_mcp_server_ping() {
        let mut server = WasmMcpServer::new("test");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "ping".to_string(),
            params: serde_json::json!({}),
        };

        let response = server.process_request(&request);
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_mcp_server_tools_list() {
        let mut server = WasmMcpServer::new("test");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "tools/list".to_string(),
            params: serde_json::json!({}),
        };

        let response = server.process_request(&request);
        assert!(response.result.is_some());

        let result = response.result.unwrap();
        assert!(result.get("tools").is_some());
    }

    #[test]
    fn test_mcp_server_tools_call_write_file() {
        let mut server = WasmMcpServer::new("test");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".to_string(),
            params: serde_json::json!({
                "name": "write_file",
                "arguments": {
                    "path": "test.txt",
                    "content": "hello mcp",
                },
            }),
        };

        let response = server.process_request(&request);
        assert!(response.result.is_some());

        // Verify file was written
        let files = server.backend.list_files();
        assert!(files.contains(&"test.txt".to_string()));
    }

    #[test]
    fn test_mcp_server_unknown_method() {
        let mut server = WasmMcpServer::new("test");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "unknown/method".to_string(),
            params: serde_json::json!({}),
        };

        let response = server.process_request(&request);
        assert!(response.error.is_some());
        assert_eq!(response.error.as_ref().unwrap().code, -32601);
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_wasm_mcp_server_create() {
        let server = WasmMcpServer::new("test-wasm");
        assert_eq!(server.name(), "test-wasm");
        assert!(!server.is_initialized());
    }

    #[wasm_bindgen_test]
    fn test_wasm_mcp_handle_request() {
        let mut server = WasmMcpServer::new("test-wasm");
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}"#;
        let result = server.handle_request(request);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_wasm_mcp_list_tools() {
        let server = WasmMcpServer::new("test-wasm");
        let tools = server.list_tools();
        assert!(tools.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_wasm_mcp_call_tool() {
        let mut server = WasmMcpServer::new("test-wasm");
        let result = server.call_tool("write_file", r#"{"path":"a.txt","content":"wasm"}"#);
        assert!(result.is_ok());
    }
}
