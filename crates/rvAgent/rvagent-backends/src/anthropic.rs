//! Anthropic Messages API backend for rvAgent.
//!
//! Implements the [`ChatModel`] trait using the Anthropic Messages API (`v1/messages`).
//! Supports text completions, tool-use responses, and automatic retry with exponential
//! backoff for transient errors (429, 500, 502, 503).

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use rvagent_core::error::{Result, RvAgentError};
use rvagent_core::messages::{AiMessage, Message, ToolCall};
use rvagent_core::models::{ApiKeySource, ChatModel, ModelConfig};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const MAX_RETRIES: u32 = 3;
const INITIAL_BACKOFF_MS: u64 = 500;

/// Status codes that should trigger an automatic retry.
const RETRYABLE_STATUS_CODES: &[u16] = &[429, 500, 502, 503];

// ---------------------------------------------------------------------------
// Anthropic API request / response types
// ---------------------------------------------------------------------------

/// A single message in the Anthropic Messages API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiMessage {
    role: String,
    content: ApiContent,
}

/// Content can be a plain string or a list of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ApiContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block in a response (text or tool_use).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// The request body sent to the Anthropic Messages API.
#[derive(Debug, Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// The response body from the Anthropic Messages API.
#[derive(Debug, Deserialize)]
struct ApiResponse {
    content: Vec<ContentBlock>,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    stop_reason: Option<String>,
    #[allow(dead_code)]
    usage: Option<Usage>,
}

/// Token usage information.
#[derive(Debug, Deserialize)]
struct Usage {
    #[allow(dead_code)]
    input_tokens: u64,
    #[allow(dead_code)]
    output_tokens: u64,
}

/// Error body returned by the Anthropic API.
#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    #[allow(dead_code)]
    r#type: String,
    message: String,
}

// ---------------------------------------------------------------------------
// AnthropicClient
// ---------------------------------------------------------------------------

/// Client for the Anthropic Messages API.
///
/// # Example
///
/// ```rust,no_run
/// use rvagent_core::models::{resolve_model, ChatModel};
/// use rvagent_backends::anthropic::AnthropicClient;
/// use rvagent_core::messages::Message;
///
/// # async fn example() -> rvagent_core::error::Result<()> {
/// let config = resolve_model("anthropic:claude-sonnet-4-20250514");
/// let client = AnthropicClient::new(config)?;
/// let response = client.complete(&[Message::human("Hello!")]).await?;
/// println!("{}", response.content());
/// # Ok(())
/// # }
/// ```
pub struct AnthropicClient {
    config: ModelConfig,
    http: reqwest::Client,
    api_key: String,
}

impl AnthropicClient {
    /// Create a new `AnthropicClient` from a [`ModelConfig`].
    ///
    /// The API key is resolved eagerly from the configured [`ApiKeySource`].
    /// Returns an error if the key cannot be resolved.
    pub fn new(config: ModelConfig) -> Result<Self> {
        let api_key = resolve_api_key(&config.api_key_source)?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| RvAgentError::model(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            config,
            http,
            api_key,
        })
    }

    /// Create an `AnthropicClient` with a pre-built `reqwest::Client` (useful for testing).
    #[cfg(test)]
    pub(crate) fn with_http(config: ModelConfig, http: reqwest::Client, api_key: String) -> Self {
        Self {
            config,
            http,
            api_key,
        }
    }

    /// Build the API request body from rvAgent messages.
    fn build_request(&self, messages: &[Message], stream: bool) -> ApiRequest {
        let mut system_text: Option<String> = None;
        let mut api_messages: Vec<ApiMessage> = Vec::new();

        for msg in messages {
            match msg {
                Message::System(s) => {
                    // Anthropic uses a top-level `system` field; merge multiple system messages.
                    match &mut system_text {
                        Some(existing) => {
                            existing.push('\n');
                            existing.push_str(&s.content);
                        }
                        None => system_text = Some(s.content.clone()),
                    }
                }
                Message::Human(h) => {
                    api_messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: ApiContent::Text(h.content.clone()),
                    });
                }
                Message::Ai(ai) => {
                    if ai.tool_calls.is_empty() {
                        api_messages.push(ApiMessage {
                            role: "assistant".to_string(),
                            content: ApiContent::Text(ai.content.clone()),
                        });
                    } else {
                        // Include text + tool_use blocks.
                        let mut blocks = Vec::new();
                        if !ai.content.is_empty() {
                            blocks.push(ContentBlock::Text {
                                text: ai.content.clone(),
                            });
                        }
                        for tc in &ai.tool_calls {
                            blocks.push(ContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                input: tc.args.clone(),
                            });
                        }
                        api_messages.push(ApiMessage {
                            role: "assistant".to_string(),
                            content: ApiContent::Blocks(blocks),
                        });
                    }
                }
                Message::Tool(t) => {
                    api_messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: ApiContent::Blocks(vec![ContentBlock::ToolResult {
                            tool_use_id: t.tool_call_id.clone(),
                            content: t.content.clone(),
                        }]),
                    });
                }
            }
        }

        ApiRequest {
            model: self.config.model_id.clone(),
            max_tokens: self.config.max_tokens,
            temperature: if self.config.temperature == 0.0 {
                None
            } else {
                Some(self.config.temperature)
            },
            system: system_text,
            messages: api_messages,
            stream: if stream { Some(true) } else { None },
        }
    }

    /// Send a request to the API with retry logic.
    async fn send_with_retry(&self, request_body: &ApiRequest, url: &str) -> Result<ApiResponse> {
        let mut last_err: Option<RvAgentError> = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let backoff = Duration::from_millis(INITIAL_BACKOFF_MS * 2u64.pow(attempt - 1));
                debug!(attempt, ?backoff, "retrying Anthropic API request");
                tokio::time::sleep(backoff).await;
            }

            // Serialize the request body to JSON string first for better error handling
            let body_json = serde_json::to_string(request_body).map_err(|e| {
                RvAgentError::model(format!("failed to serialize request body: {e}"))
            })?;

            debug!(body = %body_json, "Sending Anthropic API request");

            let result = self
                .http
                .post(url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .body(body_json)
                .send()
                .await;

            let response = match result {
                Ok(r) => r,
                Err(e) => {
                    warn!(attempt, error = %e, "Anthropic API network error");
                    last_err = Some(RvAgentError::model(format!(
                        "Anthropic API request failed: {e}"
                    )));
                    continue;
                }
            };

            let status = response.status();

            if status.is_success() {
                let body = response.text().await.map_err(|e| {
                    RvAgentError::model(format!("failed to read response body: {e}"))
                })?;
                let api_response: ApiResponse = serde_json::from_str(&body).map_err(|e| {
                    RvAgentError::model(format!(
                        "failed to parse Anthropic response: {e}; body: {body}"
                    ))
                })?;
                return Ok(api_response);
            }

            // Read error body for diagnostics.
            let error_body = response.text().await.unwrap_or_default();
            let error_message = serde_json::from_str::<ApiErrorResponse>(&error_body)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| error_body.clone());

            if RETRYABLE_STATUS_CODES.contains(&status.as_u16()) && attempt < MAX_RETRIES {
                warn!(
                    attempt,
                    status = status.as_u16(),
                    %error_message,
                    "retryable Anthropic API error"
                );
                last_err = Some(RvAgentError::model(format!(
                    "Anthropic API error {}: {}",
                    status.as_u16(),
                    error_message
                )));
                continue;
            }

            // Non-retryable or exhausted retries.
            return Err(RvAgentError::model(format!(
                "Anthropic API error {}: {}",
                status.as_u16(),
                error_message
            )));
        }

        Err(last_err.unwrap_or_else(|| RvAgentError::model("Anthropic API request failed")))
    }

    /// Convert an API response into an rvAgent [`Message`].
    fn parse_response(response: ApiResponse) -> Message {
        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in response.content {
            match block {
                ContentBlock::Text { text } => text_parts.push(text),
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        args: input,
                    });
                }
                ContentBlock::ToolResult { .. } => {
                    // Unexpected in a response; ignore.
                }
            }
        }

        let content = text_parts.join("");

        if tool_calls.is_empty() {
            Message::ai(content)
        } else {
            Message::Ai(AiMessage {
                content,
                tool_calls,
                metadata: HashMap::new(),
            })
        }
    }
}

#[async_trait]
impl ChatModel for AnthropicClient {
    /// Send messages and receive a complete response.
    async fn complete(&self, messages: &[Message]) -> Result<Message> {
        let request_body = self.build_request(messages, false);
        let response = self
            .send_with_retry(&request_body, ANTHROPIC_API_URL)
            .await?;
        Ok(Self::parse_response(response))
    }

    /// Non-streaming fallback: sends a single request and returns the full response.
    ///
    /// True SSE streaming is not yet implemented; this method calls the non-streaming
    /// endpoint and returns a single-element vector containing the complete message.
    async fn stream(&self, messages: &[Message]) -> Result<Vec<Message>> {
        let msg = self.complete(messages).await?;
        Ok(vec![msg])
    }
}

/// Resolve an API key from the configured source.
fn resolve_api_key(source: &ApiKeySource) -> Result<String> {
    match source {
        ApiKeySource::Env(var_name) => std::env::var(var_name).map_err(|_| {
            RvAgentError::config(format!(
                "API key environment variable '{var_name}' is not set"
            ))
        }),
        ApiKeySource::File(path) => std::fs::read_to_string(path)
            .map(|s| s.trim().to_string())
            .map_err(|e| {
                RvAgentError::config(format!("failed to read API key from file '{path}': {e}"))
            }),
        ApiKeySource::None => Err(RvAgentError::config(
            "no API key source configured for Anthropic",
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rvagent_core::messages::Message;
    use rvagent_core::models::{ApiKeySource, ModelConfig, Provider};
    use serde_json::json;

    /// Helper to build a test `ModelConfig`.
    fn test_config() -> ModelConfig {
        ModelConfig {
            provider: Provider::Anthropic,
            model_id: "claude-sonnet-4-20250514".to_string(),
            api_key_source: ApiKeySource::Env("ANTHROPIC_API_KEY".to_string()),
            max_tokens: 1024,
            temperature: 0.0,
        }
    }

    /// Helper to create an `AnthropicClient` pointing at a mock server.
    fn test_client(_base_url: &str) -> AnthropicClient {
        let config = test_config();
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("failed to build test HTTP client");
        AnthropicClient {
            config,
            http,
            api_key: "test-key".to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // Unit tests (no HTTP)
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let client =
            AnthropicClient::with_http(test_config(), reqwest::Client::new(), "key".to_string());
        let messages = vec![
            Message::system("You are helpful."),
            Message::human("Hello!"),
        ];
        let req = client.build_request(&messages, false);

        assert_eq!(req.model, "claude-sonnet-4-20250514");
        assert_eq!(req.max_tokens, 1024);
        assert_eq!(req.system, Some("You are helpful.".to_string()));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert!(req.stream.is_none());
    }

    #[test]
    fn test_build_request_multiple_system_messages() {
        let client =
            AnthropicClient::with_http(test_config(), reqwest::Client::new(), "key".to_string());
        let messages = vec![
            Message::system("First instruction."),
            Message::system("Second instruction."),
            Message::human("Go."),
        ];
        let req = client.build_request(&messages, false);

        assert_eq!(
            req.system,
            Some("First instruction.\nSecond instruction.".to_string())
        );
    }

    #[test]
    fn test_build_request_with_tool_calls() {
        let client =
            AnthropicClient::with_http(test_config(), reqwest::Client::new(), "key".to_string());
        let messages = vec![
            Message::human("Read that file."),
            Message::ai_with_tools(
                "Let me read it.",
                vec![ToolCall {
                    id: "tc_1".to_string(),
                    name: "read_file".to_string(),
                    args: json!({"path": "/tmp/test.txt"}),
                }],
            ),
            Message::tool("tc_1", "file contents here"),
        ];
        let req = client.build_request(&messages, false);

        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[1].role, "assistant");
        assert_eq!(req.messages[2].role, "user");

        // The assistant message should have blocks.
        match &req.messages[1].content {
            ApiContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2); // text + tool_use
            }
            _ => panic!("expected Blocks content for assistant with tool calls"),
        }
    }

    #[test]
    fn test_build_request_stream_flag() {
        let client =
            AnthropicClient::with_http(test_config(), reqwest::Client::new(), "key".to_string());
        let messages = vec![Message::human("Hi")];
        let req = client.build_request(&messages, true);
        assert_eq!(req.stream, Some(true));
    }

    #[test]
    fn test_parse_response_text_only() {
        let response = ApiResponse {
            content: vec![ContentBlock::Text {
                text: "Hello there!".to_string(),
            }],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some("end_turn".to_string()),
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
            }),
        };
        let msg = AnthropicClient::parse_response(response);
        assert_eq!(msg.content(), "Hello there!");
        assert!(!msg.has_tool_calls());
    }

    #[test]
    fn test_parse_response_with_tool_use() {
        let response = ApiResponse {
            content: vec![
                ContentBlock::Text {
                    text: "I'll read that file.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_01".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path": "/etc/hosts"}),
                },
            ],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some("tool_use".to_string()),
            usage: Some(Usage {
                input_tokens: 20,
                output_tokens: 15,
            }),
        };
        let msg = AnthropicClient::parse_response(response);
        assert_eq!(msg.content(), "I'll read that file.");
        assert!(msg.has_tool_calls());

        if let Message::Ai(ai) = &msg {
            assert_eq!(ai.tool_calls.len(), 1);
            assert_eq!(ai.tool_calls[0].id, "toolu_01");
            assert_eq!(ai.tool_calls[0].name, "read_file");
            assert_eq!(ai.tool_calls[0].args, json!({"path": "/etc/hosts"}));
        } else {
            panic!("expected Ai message");
        }
    }

    #[test]
    fn test_parse_response_multiple_tool_calls() {
        let response = ApiResponse {
            content: vec![
                ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path": "a.txt"}),
                },
                ContentBlock::ToolUse {
                    id: "t2".to_string(),
                    name: "write_file".to_string(),
                    input: json!({"path": "b.txt", "content": "data"}),
                },
            ],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some("tool_use".to_string()),
            usage: None,
        };
        let msg = AnthropicClient::parse_response(response);
        assert!(msg.has_tool_calls());
        if let Message::Ai(ai) = &msg {
            assert_eq!(ai.tool_calls.len(), 2);
            assert_eq!(ai.tool_calls[0].name, "read_file");
            assert_eq!(ai.tool_calls[1].name, "write_file");
        }
    }

    #[test]
    fn test_resolve_api_key_env() {
        std::env::set_var("TEST_ANTHROPIC_KEY_42", "sk-test-123");
        let key = resolve_api_key(&ApiKeySource::Env("TEST_ANTHROPIC_KEY_42".to_string()));
        assert_eq!(key.unwrap(), "sk-test-123");
        std::env::remove_var("TEST_ANTHROPIC_KEY_42");
    }

    #[test]
    fn test_resolve_api_key_env_missing() {
        let key = resolve_api_key(&ApiKeySource::Env(
            "DEFINITELY_NOT_SET_RVAGENT_TEST".to_string(),
        ));
        assert!(key.is_err());
    }

    #[test]
    fn test_resolve_api_key_none() {
        let key = resolve_api_key(&ApiKeySource::None);
        assert!(key.is_err());
    }

    #[test]
    fn test_resolve_api_key_file() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let key_path = dir.path().join("api_key.txt");
        std::fs::write(&key_path, "  sk-file-key  \n").expect("failed to write key file");

        let key = resolve_api_key(&ApiKeySource::File(key_path.to_string_lossy().to_string()));
        assert_eq!(key.unwrap(), "sk-file-key");
    }

    #[test]
    fn test_resolve_api_key_file_missing() {
        let key = resolve_api_key(&ApiKeySource::File("/nonexistent/key.txt".to_string()));
        assert!(key.is_err());
    }

    #[test]
    fn test_temperature_serialization() {
        let client =
            AnthropicClient::with_http(test_config(), reqwest::Client::new(), "key".to_string());
        let req = client.build_request(&[Message::human("Hi")], false);
        // temperature=0.0 => None (omitted)
        assert!(req.temperature.is_none());

        let mut config = test_config();
        config.temperature = 0.7;
        let client2 = AnthropicClient::with_http(config, reqwest::Client::new(), "key".to_string());
        let req2 = client2.build_request(&[Message::human("Hi")], false);
        assert_eq!(req2.temperature, Some(0.7));
    }

    #[test]
    fn test_api_request_serialization() {
        let req = ApiRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            temperature: Some(0.5),
            system: Some("Be helpful".to_string()),
            messages: vec![ApiMessage {
                role: "user".to_string(),
                content: ApiContent::Text("Hello".to_string()),
            }],
            stream: None,
        };
        let json = serde_json::to_value(&req).expect("serialization failed");
        assert_eq!(json["model"], "claude-sonnet-4-20250514");
        assert_eq!(json["max_tokens"], 1024);
        assert_eq!(json["temperature"], 0.5);
        assert_eq!(json["system"], "Be helpful");
        assert!(json.get("stream").is_none());
    }

    #[test]
    fn test_api_response_deserialization() {
        let json = r#"{
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3}
        }"#;
        let resp: ApiResponse = serde_json::from_str(json).expect("deserialization failed");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.model, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_api_response_tool_use_deserialization() {
        let json = r#"{
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"city": "London"}}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15}
        }"#;
        let resp: ApiResponse = serde_json::from_str(json).expect("deserialization failed");
        assert_eq!(resp.content.len(), 2);
        let msg = AnthropicClient::parse_response(resp);
        assert!(msg.has_tool_calls());
    }

    #[test]
    fn test_api_error_response_deserialization() {
        let json =
            r#"{"error": {"type": "invalid_request_error", "message": "max_tokens must be > 0"}}"#;
        let err: ApiErrorResponse = serde_json::from_str(json).expect("deserialization failed");
        assert_eq!(err.error.message, "max_tokens must be > 0");
    }

    // -----------------------------------------------------------------------
    // Integration-style tests (mock HTTP server)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_complete_success() {
        let mock_response = json!({
            "content": [{"type": "text", "text": "Hi there!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3}
        });

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_response.to_string())
            .create_async()
            .await;

        let client = test_client(&server.url());
        let url = format!("{}/v1/messages", server.url());
        let req = client.build_request(&[Message::human("Hello")], false);
        let resp = client.send_with_retry(&req, &url).await;

        assert!(resp.is_ok());
        let msg = AnthropicClient::parse_response(resp.unwrap());
        assert_eq!(msg.content(), "Hi there!");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_complete_tool_use() {
        let mock_response = json!({
            "content": [
                {"type": "text", "text": "I'll look that up."},
                {"type": "tool_use", "id": "toolu_xyz", "name": "search", "input": {"query": "rust"}}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 20}
        });

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_response.to_string())
            .create_async()
            .await;

        let client = test_client(&server.url());
        let url = format!("{}/v1/messages", server.url());
        let req = client.build_request(&[Message::human("Search for Rust")], false);
        let resp = client.send_with_retry(&req, &url).await;

        assert!(resp.is_ok());
        let msg = AnthropicClient::parse_response(resp.unwrap());
        assert!(msg.has_tool_calls());
        assert_eq!(msg.content(), "I'll look that up.");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_complete_auth_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/messages")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body(
                json!({"error": {"type": "authentication_error", "message": "invalid api key"}})
                    .to_string(),
            )
            .create_async()
            .await;

        let client = test_client(&server.url());
        let url = format!("{}/v1/messages", server.url());
        let req = client.build_request(&[Message::human("Hi")], false);
        let result = client.send_with_retry(&req, &url).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("401"));
        assert!(err_msg.contains("invalid api key"));
        // 401 is non-retryable, so only one request.
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_retry_on_500() {
        let mut server = mockito::Server::new_async().await;

        // First call returns 500, second returns 200.
        let fail_mock = server
            .mock("POST", "/v1/messages")
            .with_status(500)
            .with_header("content-type", "application/json")
            .with_body(
                json!({"error": {"type": "api_error", "message": "internal error"}}).to_string(),
            )
            .expect(1)
            .create_async()
            .await;

        let success_mock = server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "content": [{"type": "text", "text": "recovered"}],
                    "model": "claude-sonnet-4-20250514",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 2}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let client = test_client(&server.url());
        let url = format!("{}/v1/messages", server.url());
        let req = client.build_request(&[Message::human("Hi")], false);
        let result = client.send_with_retry(&req, &url).await;

        assert!(result.is_ok());
        let msg = AnthropicClient::parse_response(result.unwrap());
        assert_eq!(msg.content(), "recovered");

        fail_mock.assert_async().await;
        success_mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let mut server = mockito::Server::new_async().await;

        // All 4 attempts (1 initial + 3 retries) return 429.
        let mock = server
            .mock("POST", "/v1/messages")
            .with_status(429)
            .with_header("content-type", "application/json")
            .with_body(
                json!({"error": {"type": "rate_limit_error", "message": "rate limited"}})
                    .to_string(),
            )
            .expect(4) // initial + 3 retries
            .create_async()
            .await;

        let client = test_client(&server.url());
        let url = format!("{}/v1/messages", server.url());
        let req = client.build_request(&[Message::human("Hi")], false);
        let result = client.send_with_retry(&req, &url).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("429"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_request_headers() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/messages")
            .match_header("x-api-key", "test-key")
            .match_header("anthropic-version", "2023-06-01")
            .match_header("content-type", "application/json")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "claude-sonnet-4-20250514",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let client = test_client(&server.url());
        let url = format!("{}/v1/messages", server.url());
        let req = client.build_request(&[Message::human("Hi")], false);
        let result = client.send_with_retry(&req, &url).await;

        assert!(result.is_ok());
        mock.assert_async().await;
    }
}
