//! JSON-RPC 2.0 protocol types for the Model Context Protocol.
//!
//! Implements the MCP wire format: requests, responses, errors, capabilities,
//! tool/resource/prompt definitions, and content types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 base types
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request.
    pub fn new(id: impl Into<serde_json::Value>, method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: id.into(),
            method: method.into(),
            params: None,
        }
    }

    /// Attach params to the request.
    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params);
        self
    }
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response.
    pub fn error(id: serde_json::Value, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcError {
    /// Standard parse error (-32700).
    pub fn parse_error(msg: impl Into<String>) -> Self {
        Self { code: -32700, message: msg.into(), data: None }
    }

    /// Standard invalid request (-32600).
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self { code: -32600, message: msg.into(), data: None }
    }

    /// Standard method not found (-32601).
    pub fn method_not_found(msg: impl Into<String>) -> Self {
        Self { code: -32601, message: msg.into(), data: None }
    }

    /// Standard invalid params (-32602).
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self { code: -32602, message: msg.into(), data: None }
    }

    /// Standard internal error (-32603).
    pub fn internal_error(msg: impl Into<String>) -> Self {
        Self { code: -32603, message: msg.into(), data: None }
    }
}

// ---------------------------------------------------------------------------
// MCP method enumeration
// ---------------------------------------------------------------------------

/// MCP protocol methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum McpMethod {
    #[serde(rename = "initialize")]
    Initialize,
    #[serde(rename = "tools/list")]
    ToolsList,
    #[serde(rename = "tools/call")]
    ToolsCall,
    #[serde(rename = "resources/list")]
    ResourcesList,
    #[serde(rename = "resources/read")]
    ResourcesRead,
    #[serde(rename = "resources/templates/list")]
    ResourcesTemplatesList,
    #[serde(rename = "prompts/list")]
    PromptsList,
    #[serde(rename = "prompts/get")]
    PromptsGet,
    #[serde(rename = "ping")]
    Ping,
}

impl McpMethod {
    /// Parse a method string into an `McpMethod`.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "initialize" => Some(Self::Initialize),
            "tools/list" => Some(Self::ToolsList),
            "tools/call" => Some(Self::ToolsCall),
            "resources/list" => Some(Self::ResourcesList),
            "resources/read" => Some(Self::ResourcesRead),
            "resources/templates/list" => Some(Self::ResourcesTemplatesList),
            "prompts/list" => Some(Self::PromptsList),
            "prompts/get" => Some(Self::PromptsGet),
            "ping" => Some(Self::Ping),
            _ => None,
        }
    }

    /// Return the wire-format string for this method.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Initialize => "initialize",
            Self::ToolsList => "tools/list",
            Self::ToolsCall => "tools/call",
            Self::ResourcesList => "resources/list",
            Self::ResourcesRead => "resources/read",
            Self::ResourcesTemplatesList => "resources/templates/list",
            Self::PromptsList => "prompts/list",
            Self::PromptsGet => "prompts/get",
            Self::Ping => "ping",
        }
    }
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/// Server capabilities advertised during initialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
}

/// Tools capability descriptor.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

/// Resources capability descriptor.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    #[serde(default)]
    pub subscribe: bool,
    #[serde(default)]
    pub list_changed: bool,
}

/// Prompts capability descriptor.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

/// Client capabilities sent during initialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<serde_json::Value>,
}

/// Roots capability from client.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

// ---------------------------------------------------------------------------
// Initialize handshake
// ---------------------------------------------------------------------------

/// Parameters for the `initialize` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    pub client_info: ClientInfo,
}

/// Client identification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// Result of the `initialize` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    pub server_info: ServerInfo,
}

/// Server identification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

// ---------------------------------------------------------------------------
// Tool types
// ---------------------------------------------------------------------------

/// An MCP tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Parameters for `tools/call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Result of `tools/call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallResult {
    pub content: Vec<Content>,
    #[serde(default)]
    pub is_error: bool,
}

// ---------------------------------------------------------------------------
// Resource types
// ---------------------------------------------------------------------------

/// An MCP resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// An MCP resource template.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResourceTemplate {
    pub uri_template: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Parameters for `resources/read`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReadParams {
    pub uri: String,
}

/// Result of `resources/read`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReadResult {
    pub contents: Vec<ResourceContent>,
}

/// A resource content entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceContent {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

// ---------------------------------------------------------------------------
// Prompt types
// ---------------------------------------------------------------------------

/// An MCP prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPrompt {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<PromptArgument>,
}

/// A prompt argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// Parameters for `prompts/get`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptGetParams {
    pub name: String,
    #[serde(default)]
    pub arguments: HashMap<String, String>,
}

/// Result of `prompts/get`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptGetResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub messages: Vec<PromptMessage>,
}

/// A message within a prompt result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: Content,
}

// ---------------------------------------------------------------------------
// Content types
// ---------------------------------------------------------------------------

/// MCP content — text, image, or embedded resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Content {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        text: String,
    },
    /// Base64-encoded image content.
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Embedded resource reference.
    #[serde(rename = "resource")]
    Resource {
        resource: ResourceContent,
    },
}

impl Content {
    /// Create text content.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    /// Create image content.
    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Image { data: data.into(), mime_type: mime_type.into() }
    }
}

// ---------------------------------------------------------------------------
// List response wrappers
// ---------------------------------------------------------------------------

/// Response for `tools/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsListResult {
    pub tools: Vec<McpTool>,
}

/// Response for `resources/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcesListResult {
    pub resources: Vec<McpResource>,
}

/// Response for `resources/templates/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTemplatesListResult {
    pub resource_templates: Vec<McpResourceTemplate>,
}

/// Response for `prompts/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsListResult {
    pub prompts: Vec<McpPrompt>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonrpc_request_roundtrip() {
        let req = JsonRpcRequest::new(1, "tools/list");
        let json = serde_json::to_string(&req).unwrap();
        let back: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.method, "tools/list");
        assert_eq!(back.jsonrpc, "2.0");
    }

    #[test]
    fn test_jsonrpc_request_with_params() {
        let req = JsonRpcRequest::new(42, "tools/call")
            .with_params(serde_json::json!({"name": "ping"}));
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"params\""));
        let back: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert!(back.params.is_some());
    }

    #[test]
    fn test_jsonrpc_response_success() {
        let resp = JsonRpcResponse::success(
            serde_json::json!(1),
            serde_json::json!({"tools": []}),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn test_jsonrpc_response_error() {
        let resp = JsonRpcResponse::error(
            serde_json::json!(1),
            JsonRpcError::method_not_found("no such method"),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"error\""));
        let back: JsonRpcResponse = serde_json::from_str(&json).unwrap();
        assert!(back.error.is_some());
        assert_eq!(back.error.unwrap().code, -32601);
    }

    #[test]
    fn test_jsonrpc_error_codes() {
        assert_eq!(JsonRpcError::parse_error("x").code, -32700);
        assert_eq!(JsonRpcError::invalid_request("x").code, -32600);
        assert_eq!(JsonRpcError::method_not_found("x").code, -32601);
        assert_eq!(JsonRpcError::invalid_params("x").code, -32602);
        assert_eq!(JsonRpcError::internal_error("x").code, -32603);
    }

    #[test]
    fn test_mcp_method_parse() {
        assert_eq!(McpMethod::from_str("initialize"), Some(McpMethod::Initialize));
        assert_eq!(McpMethod::from_str("tools/list"), Some(McpMethod::ToolsList));
        assert_eq!(McpMethod::from_str("tools/call"), Some(McpMethod::ToolsCall));
        assert_eq!(McpMethod::from_str("resources/list"), Some(McpMethod::ResourcesList));
        assert_eq!(McpMethod::from_str("resources/read"), Some(McpMethod::ResourcesRead));
        assert_eq!(McpMethod::from_str("resources/templates/list"), Some(McpMethod::ResourcesTemplatesList));
        assert_eq!(McpMethod::from_str("prompts/list"), Some(McpMethod::PromptsList));
        assert_eq!(McpMethod::from_str("prompts/get"), Some(McpMethod::PromptsGet));
        assert_eq!(McpMethod::from_str("ping"), Some(McpMethod::Ping));
        assert_eq!(McpMethod::from_str("unknown"), None);
    }

    #[test]
    fn test_mcp_method_roundtrip() {
        for method in &[
            McpMethod::Initialize,
            McpMethod::ToolsList,
            McpMethod::ToolsCall,
            McpMethod::ResourcesList,
            McpMethod::ResourcesRead,
            McpMethod::Ping,
        ] {
            let s = method.as_str();
            assert_eq!(McpMethod::from_str(s).as_ref(), Some(method));
        }
    }

    #[test]
    fn test_server_capabilities_roundtrip() {
        let caps = ServerCapabilities {
            tools: Some(ToolsCapability { list_changed: true }),
            resources: Some(ResourcesCapability { subscribe: true, list_changed: false }),
            prompts: None,
        };
        let json = serde_json::to_string(&caps).unwrap();
        let back: ServerCapabilities = serde_json::from_str(&json).unwrap();
        assert!(back.tools.unwrap().list_changed);
        assert!(back.resources.unwrap().subscribe);
        assert!(back.prompts.is_none());
    }

    #[test]
    fn test_initialize_params_roundtrip() {
        let params = InitializeParams {
            protocol_version: "2024-11-05".into(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "test-client".into(),
                version: "1.0".into(),
            },
        };
        let json = serde_json::to_string(&params).unwrap();
        let back: InitializeParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.protocol_version, "2024-11-05");
        assert_eq!(back.client_info.name, "test-client");
    }

    #[test]
    fn test_initialize_result_roundtrip() {
        let result = InitializeResult {
            protocol_version: "2024-11-05".into(),
            capabilities: ServerCapabilities::default(),
            server_info: ServerInfo {
                name: "test-server".into(),
                version: "0.1.0".into(),
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: InitializeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.server_info.name, "test-server");
    }

    #[test]
    fn test_mcp_tool_roundtrip() {
        let tool = McpTool {
            name: "read_file".into(),
            description: "Read a file".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } }
            }),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("inputSchema"));
        let back: McpTool = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "read_file");
    }

    #[test]
    fn test_tool_call_params_roundtrip() {
        let params = ToolCallParams {
            name: "echo".into(),
            arguments: serde_json::json!({"text": "hello"}),
        };
        let json = serde_json::to_string(&params).unwrap();
        let back: ToolCallParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "echo");
    }

    #[test]
    fn test_tool_call_result_roundtrip() {
        let result = ToolCallResult {
            content: vec![Content::text("result text")],
            is_error: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ToolCallResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content.len(), 1);
        assert!(!back.is_error);
    }

    #[test]
    fn test_content_text_roundtrip() {
        let c = Content::text("hello world");
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        let back: Content = serde_json::from_str(&json).unwrap();
        match back {
            Content::Text { text } => assert_eq!(text, "hello world"),
            _ => panic!("expected text content"),
        }
    }

    #[test]
    fn test_content_image_roundtrip() {
        let c = Content::image("base64data==", "image/png");
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("\"type\":\"image\""));
        let back: Content = serde_json::from_str(&json).unwrap();
        match back {
            Content::Image { data, mime_type } => {
                assert_eq!(data, "base64data==");
                assert_eq!(mime_type, "image/png");
            }
            _ => panic!("expected image content"),
        }
    }

    #[test]
    fn test_mcp_resource_roundtrip() {
        let r = McpResource {
            uri: "file:///readme.md".into(),
            name: "readme".into(),
            description: Some("Project readme".into()),
            mime_type: Some("text/markdown".into()),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("mimeType"));
        let back: McpResource = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uri, "file:///readme.md");
    }

    #[test]
    fn test_mcp_resource_template_roundtrip() {
        let t = McpResourceTemplate {
            uri_template: "file:///{path}".into(),
            name: "file".into(),
            description: None,
            mime_type: None,
        };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("uriTemplate"));
        let back: McpResourceTemplate = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uri_template, "file:///{path}");
    }

    #[test]
    fn test_mcp_prompt_roundtrip() {
        let p = McpPrompt {
            name: "summarize".into(),
            description: Some("Summarize text".into()),
            arguments: vec![PromptArgument {
                name: "text".into(),
                description: Some("Text to summarize".into()),
                required: true,
            }],
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: McpPrompt = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "summarize");
        assert_eq!(back.arguments.len(), 1);
        assert!(back.arguments[0].required);
    }

    #[test]
    fn test_resource_content_text() {
        let rc = ResourceContent {
            uri: "file:///test.txt".into(),
            mime_type: Some("text/plain".into()),
            text: Some("file content".into()),
            blob: None,
        };
        let json = serde_json::to_string(&rc).unwrap();
        let back: ResourceContent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text.as_deref(), Some("file content"));
        assert!(back.blob.is_none());
    }

    #[test]
    fn test_resource_read_result_roundtrip() {
        let result = ResourceReadResult {
            contents: vec![ResourceContent {
                uri: "file:///a.txt".into(),
                mime_type: None,
                text: Some("hello".into()),
                blob: None,
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ResourceReadResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.contents.len(), 1);
    }

    #[test]
    fn test_client_capabilities_default() {
        let caps = ClientCapabilities::default();
        let json = serde_json::to_string(&caps).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_prompt_get_result_roundtrip() {
        let result = PromptGetResult {
            description: Some("A prompt".into()),
            messages: vec![PromptMessage {
                role: "user".into(),
                content: Content::text("What is 2+2?"),
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: PromptGetResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.messages[0].role, "user");
    }

    #[test]
    fn test_tools_list_result() {
        let result = ToolsListResult { tools: vec![] };
        let json = serde_json::to_string(&result).unwrap();
        let back: ToolsListResult = serde_json::from_str(&json).unwrap();
        assert!(back.tools.is_empty());
    }
}
