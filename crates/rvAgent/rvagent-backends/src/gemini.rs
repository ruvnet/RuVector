//! Google Gemini API backend for rvAgent.
//!
//! Implements the [`ChatModel`] trait using the Google Generative AI API.
//! Supports text completions and automatic retry with exponential backoff.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use std::collections::HashMap;

use rvagent_core::error::{Result, RvAgentError};
use rvagent_core::messages::{AiMessage, Message, ToolCall};
use rvagent_core::models::{ApiKeySource, ChatModel, ModelConfig, ToolDefinition};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const MAX_RETRIES: u32 = 3;
const INITIAL_BACKOFF_MS: u64 = 500;

/// Status codes that should trigger an automatic retry.
const RETRYABLE_STATUS_CODES: &[u16] = &[429, 500, 502, 503];

// ---------------------------------------------------------------------------
// Gemini API request / response types
// ---------------------------------------------------------------------------

/// A function call emitted by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionCall {
    name: String,
    #[serde(default)]
    args: serde_json::Value,
}

/// A function result sent back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionResponse {
    name: String,
    response: serde_json::Value,
}

/// Content part in a Gemini message: text, a function call, or a function response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<FunctionResponse>,
}

impl Part {
    fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            ..Self::default()
        }
    }
}

/// A function declaration advertised to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// A tool group in the Gemini API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<FunctionDeclaration>,
}

/// A single message in the Gemini API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<Part>,
}

/// Generation config for the Gemini API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    max_output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

/// The request body sent to the Gemini API.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    generation_config: GenerationConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GeminiTool>,
}

/// A candidate response from Gemini.
#[derive(Debug, Deserialize)]
struct Candidate {
    content: GeminiContent,
    #[allow(dead_code)]
    #[serde(default)]
    finish_reason: Option<String>,
}

/// The response body from the Gemini API.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<Candidate>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
}

/// Token usage information.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    prompt_token_count: Option<u64>,
    candidates_token_count: Option<u64>,
}

/// Error response from Gemini API.
#[derive(Debug, Deserialize)]
struct GeminiError {
    error: GeminiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct GeminiErrorDetail {
    message: String,
    #[allow(dead_code)]
    code: Option<i32>,
}

// ---------------------------------------------------------------------------
// GeminiClient
// ---------------------------------------------------------------------------

/// Client for the Google Gemini API.
///
/// # Example
///
/// ```rust,no_run
/// use rvagent_core::models::{resolve_model, ChatModel};
/// use rvagent_backends::gemini::GeminiClient;
/// use rvagent_core::messages::Message;
///
/// # async fn example() -> rvagent_core::error::Result<()> {
/// let config = resolve_model("google:gemini-2.5-pro-preview-06-05");
/// let client = GeminiClient::new(config)?;
/// let response = client.complete(&[Message::human("Hello!")], &[]).await?;
/// println!("{}", response.content());
/// # Ok(())
/// # }
/// ```
pub struct GeminiClient {
    config: ModelConfig,
    http: reqwest::Client,
    api_key: String,
}

impl GeminiClient {
    /// Create a new `GeminiClient` from a [`ModelConfig`].
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

    /// Build the API request body from rvAgent messages and tool definitions.
    fn build_request(&self, messages: &[Message], tools: &[ToolDefinition]) -> GeminiRequest {
        let mut system_instruction: Option<GeminiContent> = None;
        let mut contents: Vec<GeminiContent> = Vec::new();
        // Gemini has no tool-call IDs on the wire; map our synthesized IDs
        // back to function names so tool results can be sent as
        // functionResponse parts.
        let mut call_id_to_name: HashMap<String, String> = HashMap::new();

        for msg in messages {
            match msg {
                Message::System(s) => {
                    system_instruction = Some(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![Part::text(s.content.clone())],
                    });
                }
                Message::Human(h) => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![Part::text(h.content.clone())],
                    });
                }
                Message::Ai(ai) => {
                    let mut parts: Vec<Part> = Vec::new();
                    if !ai.content.is_empty() {
                        parts.push(Part::text(ai.content.clone()));
                    }
                    for tc in &ai.tool_calls {
                        call_id_to_name.insert(tc.id.clone(), tc.name.clone());
                        parts.push(Part {
                            function_call: Some(FunctionCall {
                                name: tc.name.clone(),
                                args: tc.args.clone(),
                            }),
                            ..Part::default()
                        });
                    }
                    if parts.is_empty() {
                        parts.push(Part::text(String::new()));
                    }
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
                Message::Tool(t) => {
                    let name = call_id_to_name
                        .get(&t.tool_call_id)
                        .cloned()
                        .unwrap_or_else(|| t.tool_call_id.clone());
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![Part {
                            function_response: Some(FunctionResponse {
                                name,
                                response: serde_json::json!({ "result": t.content }),
                            }),
                            ..Part::default()
                        }],
                    });
                }
            }
        }

        let gemini_tools = if tools.is_empty() {
            Vec::new()
        } else {
            vec![GeminiTool {
                function_declarations: tools
                    .iter()
                    .map(|t| FunctionDeclaration {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.input_schema.clone(),
                    })
                    .collect(),
            }]
        };

        GeminiRequest {
            contents,
            generation_config: GenerationConfig {
                max_output_tokens: self.config.max_tokens,
                temperature: if self.config.temperature == 0.0 {
                    None
                } else {
                    Some(self.config.temperature)
                },
            },
            system_instruction,
            tools: gemini_tools,
        }
    }

    /// Send a request to the API with retry logic.
    async fn send_with_retry(&self, request_body: &GeminiRequest) -> Result<GeminiResponse> {
        let url = format!(
            "{}/{}:generateContent?key={}",
            GEMINI_API_BASE, self.config.model_id, self.api_key
        );

        let mut last_err: Option<RvAgentError> = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let backoff = Duration::from_millis(INITIAL_BACKOFF_MS * 2u64.pow(attempt - 1));
                debug!(attempt, ?backoff, "retrying Gemini API request");
                tokio::time::sleep(backoff).await;
            }

            let body_json = serde_json::to_string(request_body).map_err(|e| {
                RvAgentError::model(format!("failed to serialize request body: {e}"))
            })?;

            debug!(body = %body_json, "Sending Gemini API request");

            let result = self
                .http
                .post(&url)
                .header("content-type", "application/json")
                .body(body_json)
                .send()
                .await;

            let response = match result {
                Ok(r) => r,
                Err(e) => {
                    warn!(attempt, error = %e, "Gemini API network error");
                    last_err = Some(RvAgentError::model(format!(
                        "Gemini API request failed: {e}"
                    )));
                    continue;
                }
            };

            let status = response.status();

            if status.is_success() {
                let body = response.text().await.map_err(|e| {
                    RvAgentError::model(format!("failed to read response body: {e}"))
                })?;
                let api_response: GeminiResponse = serde_json::from_str(&body).map_err(|e| {
                    RvAgentError::model(format!(
                        "failed to parse Gemini response: {e}; body: {body}"
                    ))
                })?;
                return Ok(api_response);
            }

            // Read error body for diagnostics.
            let error_body = response.text().await.unwrap_or_default();
            let error_message = serde_json::from_str::<GeminiError>(&error_body)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| error_body.clone());

            let status_code = status.as_u16();
            if RETRYABLE_STATUS_CODES.contains(&status_code) {
                warn!(attempt, status_code, %error_message, "retryable Gemini API error");
                last_err = Some(RvAgentError::model(format!(
                    "Gemini API error {status_code}: {error_message}"
                )));
                continue;
            }

            // Non-retryable error.
            return Err(RvAgentError::model(format!(
                "Gemini API error {status_code}: {error_message}"
            )));
        }

        Err(last_err
            .unwrap_or_else(|| RvAgentError::model("Gemini API request failed after all retries")))
    }
}

#[async_trait]
impl ChatModel for GeminiClient {
    async fn complete(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Message> {
        let request = self.build_request(messages, tools);
        let response = self.send_with_retry(&request).await?;

        // Collect text and function-call parts from the first candidate.
        // Gemini does not assign tool-call IDs, so synthesize stable ones.
        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        if let Some(candidate) = response.candidates.first() {
            for (idx, part) in candidate.content.parts.iter().enumerate() {
                if let Some(text) = &part.text {
                    text_parts.push(text.clone());
                }
                if let Some(fc) = &part.function_call {
                    tool_calls.push(ToolCall {
                        id: format!("gemini_call_{idx}_{}", fc.name),
                        name: fc.name.clone(),
                        args: fc.args.clone(),
                    });
                }
            }
        }

        let mut metadata = HashMap::new();
        if let Some(usage) = &response.usage_metadata {
            metadata.insert(
                "usage".to_string(),
                serde_json::json!({
                    "input_tokens": usage.prompt_token_count.unwrap_or(0),
                    "output_tokens": usage.candidates_token_count.unwrap_or(0),
                }),
            );
        }

        Ok(Message::Ai(AiMessage {
            content: text_parts.join(""),
            tool_calls,
            metadata,
        }))
    }

    async fn stream(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Vec<Message>> {
        // For now, use non-streaming completion
        let msg = self.complete(messages, tools).await?;
        Ok(vec![msg])
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_api_key(source: &ApiKeySource) -> Result<String> {
    match source {
        ApiKeySource::Env(var) => std::env::var(var).map_err(|_| {
            RvAgentError::config(format!("API key environment variable '{var}' not set"))
        }),
        ApiKeySource::File(path) => std::fs::read_to_string(path)
            .map(|s| s.trim().to_string())
            .map_err(|e| {
                RvAgentError::config(format!("failed to read API key from '{path}': {e}"))
            }),
        ApiKeySource::None => Err(RvAgentError::config(
            "no API key source configured for Gemini",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_request_serialization() {
        let request = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".to_string(),
                parts: vec![Part::text("Hello")],
            }],
            generation_config: GenerationConfig {
                max_output_tokens: 1024,
                temperature: Some(0.7),
            },
            system_instruction: None,
            tools: Vec::new(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"maxOutputTokens\":1024"));
        // Empty tools must be omitted entirely.
        assert!(!json.contains("\"tools\""));
    }

    #[test]
    fn test_gemini_response_parsing() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello there!"}]
                },
                "finishReason": "STOP"
            }]
        }"#;

        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert_eq!(
            response.candidates[0].content.parts[0].text.as_deref(),
            Some("Hello there!")
        );
    }

    #[test]
    fn test_gemini_request_with_tools() {
        let config = ModelConfig {
            provider: rvagent_core::models::Provider::Google,
            model_id: "gemini-2.5-pro".to_string(),
            api_key_source: ApiKeySource::None,
            max_tokens: 1024,
            temperature: 0.0,
        };
        let client = GeminiClient {
            config,
            http: reqwest::Client::new(),
            api_key: "test".to_string(),
        };

        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }];
        let req = client.build_request(&[Message::human("read it")], &tools);
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].function_declarations.len(), 1);
        assert_eq!(req.tools[0].function_declarations[0].name, "read_file");

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("functionDeclarations"));
    }

    #[test]
    fn test_gemini_function_call_response_parsing() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Let me read that."},
                        {"functionCall": {"name": "read_file", "args": {"path": "/tmp/x"}}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 7}
        }"#;

        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        let parts = &response.candidates[0].content.parts;
        assert_eq!(parts.len(), 2);
        assert!(parts[0].text.is_some());
        let fc = parts[1].function_call.as_ref().unwrap();
        assert_eq!(fc.name, "read_file");
        assert_eq!(fc.args, serde_json::json!({"path": "/tmp/x"}));
        let usage = response.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, Some(12));
        assert_eq!(usage.candidates_token_count, Some(7));
    }

    #[test]
    fn test_gemini_tool_result_roundtrip() {
        let config = ModelConfig {
            provider: rvagent_core::models::Provider::Google,
            model_id: "gemini-2.5-pro".to_string(),
            api_key_source: ApiKeySource::None,
            max_tokens: 1024,
            temperature: 0.0,
        };
        let client = GeminiClient {
            config,
            http: reqwest::Client::new(),
            api_key: "test".to_string(),
        };

        let messages = vec![
            Message::human("read it"),
            Message::ai_with_tools(
                "",
                vec![ToolCall {
                    id: "gemini_call_0_read_file".to_string(),
                    name: "read_file".to_string(),
                    args: serde_json::json!({"path": "/tmp/x"}),
                }],
            ),
            Message::tool("gemini_call_0_read_file", "contents"),
        ];
        let req = client.build_request(&messages, &[]);

        // AI turn carries the functionCall part; tool turn carries the
        // functionResponse part with the recovered function name.
        let model_turn = &req.contents[1];
        assert_eq!(model_turn.role, "model");
        assert!(model_turn.parts[0].function_call.is_some());

        let tool_turn = &req.contents[2];
        assert_eq!(tool_turn.role, "user");
        let fr = tool_turn.parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "read_file");
        assert_eq!(fr.response, serde_json::json!({"result": "contents"}));
    }
}
