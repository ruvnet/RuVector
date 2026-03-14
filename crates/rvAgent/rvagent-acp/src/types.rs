//! Request/response types for the ACP server.
//!
//! Defines the wire format for prompt submission, session management,
//! and error responses per ADR-099 and ADR-103 C6.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

/// A content block within a prompt request or response message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content.
    Text { text: String },

    /// A tool use request (model → server).
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    /// A tool execution result (server → model).
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

// ---------------------------------------------------------------------------
// Prompt request / response
// ---------------------------------------------------------------------------

/// Request body for `POST /prompt`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptRequest {
    /// Target session (created automatically if absent).
    pub session_id: Option<String>,

    /// Content blocks to send to the agent.
    pub content: Vec<ContentBlock>,
}

/// A single message in a prompt response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    /// Role: "assistant" or "tool".
    pub role: String,

    /// Content blocks returned by the agent.
    pub content: Vec<ContentBlock>,
}

/// Response body for `POST /prompt`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptResponse {
    /// The session that handled this prompt.
    pub session_id: String,

    /// Response messages from the agent.
    pub messages: Vec<ResponseMessage>,
}

// ---------------------------------------------------------------------------
// Session types
// ---------------------------------------------------------------------------

/// Summary information about an active session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    /// Unique session identifier.
    pub id: String,

    /// When the session was created (RFC 3339 timestamp).
    pub created_at: String,

    /// Number of messages exchanged in this session.
    pub message_count: usize,
}

/// Request body for `POST /sessions`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreateSessionRequest {
    /// Optional working directory for the session.
    #[serde(default)]
    pub cwd: Option<String>,
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// Response body for `GET /health`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Standard error response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Machine-readable error code.
    pub error: String,

    /// Human-readable description.
    pub message: String,

    /// HTTP status code (mirrored in body for convenience).
    pub status: u16,
}

impl ErrorResponse {
    /// Create a new error response.
    pub fn new(error: impl Into<String>, message: impl Into<String>, status: u16) -> Self {
        Self {
            error: error.into(),
            message: message.into(),
            status,
        }
    }

    /// 400 Bad Request.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new("bad_request", message, 400)
    }

    /// 401 Unauthorized.
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new("unauthorized", message, 401)
    }

    /// 404 Not Found.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new("not_found", message, 404)
    }

    /// 413 Payload Too Large.
    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::new("payload_too_large", message, 413)
    }

    /// 429 Too Many Requests.
    pub fn too_many_requests(message: impl Into<String>) -> Self {
        Self::new("too_many_requests", message, 429)
    }

    /// 500 Internal Server Error.
    #[allow(dead_code)]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new("internal_error", message, 500)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_block_text_serde() {
        let block = ContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"text""#));
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, back);
    }

    #[test]
    fn test_content_block_tool_use_serde() {
        let block = ContentBlock::ToolUse {
            id: "call_1".into(),
            name: "read_file".into(),
            input: serde_json::json!({"path": "/tmp/f.txt"}),
        };
        let json = serde_json::to_string(&block).unwrap();
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, back);
    }

    #[test]
    fn test_content_block_tool_result_serde() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "call_1".into(),
            content: "file contents".into(),
            is_error: false,
        };
        let json = serde_json::to_string(&block).unwrap();
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, back);
    }

    #[test]
    fn test_prompt_request_serde() {
        let req = PromptRequest {
            session_id: Some("sess_1".into()),
            content: vec![ContentBlock::Text {
                text: "hello".into(),
            }],
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: PromptRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, Some("sess_1".into()));
        assert_eq!(back.content.len(), 1);
    }

    #[test]
    fn test_error_response_constructors() {
        let e = ErrorResponse::unauthorized("missing token");
        assert_eq!(e.status, 401);
        assert_eq!(e.error, "unauthorized");

        let e = ErrorResponse::too_many_requests("slow down");
        assert_eq!(e.status, 429);

        let e = ErrorResponse::not_found("session gone");
        assert_eq!(e.status, 404);

        let e = ErrorResponse::payload_too_large("body too big");
        assert_eq!(e.status, 413);

        let e = ErrorResponse::internal("oops");
        assert_eq!(e.status, 500);
    }

    #[test]
    fn test_session_info_serde() {
        let info = SessionInfo {
            id: "s1".into(),
            created_at: "2026-03-14T00:00:00Z".into(),
            message_count: 5,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: SessionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "s1");
        assert_eq!(back.message_count, 5);
    }

    #[test]
    fn test_health_response() {
        let h = HealthResponse {
            status: "ok".into(),
            version: "0.1.0".into(),
        };
        let json = serde_json::to_string(&h).unwrap();
        assert!(json.contains("ok"));
    }
}
