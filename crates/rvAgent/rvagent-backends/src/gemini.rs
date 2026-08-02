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
    /// Omitted entirely for parameterless tools: Gemini rejects an OBJECT
    /// schema with an empty property map, and `parameters` is optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

/// A tool group in the Gemini API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<FunctionDeclaration>,
}

/// A single message in the Gemini API format.
///
/// Both fields default on deserialization: Gemini omits `parts` entirely on a
/// truncated or filtered candidate, and omits `role` on some content blocks.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GeminiContent {
    #[serde(default)]
    role: String,
    #[serde(default)]
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
#[serde(rename_all = "camelCase")]
struct Candidate {
    /// Absent when Gemini blocks the candidate outright (e.g. a SAFETY stop
    /// yields `{"finishReason": "SAFETY"}` with no content at all).
    #[serde(default)]
    content: Option<GeminiContent>,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// Feedback about the prompt itself, present when Gemini refuses to answer.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptFeedback {
    #[serde(default)]
    block_reason: Option<String>,
}

/// The response body from the Gemini API.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
    #[serde(default)]
    prompt_feedback: Option<PromptFeedback>,
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
        // Gemini requires every functionResponse answering one model turn to
        // live in a single user Content. Buffer consecutive tool results and
        // flush them together.
        let mut pending_tool_parts: Vec<Part> = Vec::new();

        for msg in messages {
            if !matches!(msg, Message::Tool(_)) && !pending_tool_parts.is_empty() {
                contents.push(GeminiContent {
                    role: "user".to_string(),
                    parts: std::mem::take(&mut pending_tool_parts),
                });
            }

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
                    // Prefer the tool name recorded on the message itself,
                    // then the id→name map from the preceding AI turn. The
                    // raw id is a last resort only (it is not a valid
                    // function name and Gemini may reject it).
                    let name = t
                        .tool_name
                        .clone()
                        .or_else(|| call_id_to_name.get(&t.tool_call_id).cloned())
                        .unwrap_or_else(|| t.tool_call_id.clone());
                    pending_tool_parts.push(Part {
                        function_response: Some(FunctionResponse {
                            name,
                            response: serde_json::json!({ "result": t.content }),
                        }),
                        ..Part::default()
                    });
                }
            }
        }

        if !pending_tool_parts.is_empty() {
            contents.push(GeminiContent {
                role: "user".to_string(),
                parts: pending_tool_parts,
            });
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
                        parameters: tool_parameters(&t.input_schema),
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
        // The API key travels in the `x-goog-api-key` header, never in the
        // URL query string — URLs end up in error messages, proxy logs, and
        // tracing output, which must never contain credentials.
        let url = format!(
            "{}/{}:generateContent",
            GEMINI_API_BASE, self.config.model_id
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
                .header("x-goog-api-key", &self.api_key)
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

    /// Convert an API response into an rvAgent [`Message`].
    fn parse_response(response: GeminiResponse) -> Result<Message> {
        // Collect text and function-call parts from the first candidate.
        // Gemini does not assign tool-call IDs, so synthesize unique ones:
        // a process-wide counter prevents id collisions across turns (the
        // per-part index alone repeats every turn, which would let one
        // turn's tool result satisfy another turn's call in id-keyed logic).
        static CALL_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        // An empty candidate list means Gemini produced nothing — typically a
        // safety block, which carries a promptFeedback.blockReason. Surfacing
        // it as an error keeps the agent loop from treating a refusal as an
        // ordinary empty reply.
        let candidate = response.candidates.first().ok_or_else(|| {
            match response
                .prompt_feedback
                .as_ref()
                .and_then(|f| f.block_reason.as_deref())
            {
                Some(reason) => RvAgentError::model(format!(
                    "Gemini returned no candidates: prompt blocked (blockReason: {reason})"
                )),
                None => RvAgentError::model(
                    "Gemini returned no candidates (response may have been blocked)",
                ),
            }
        })?;

        // A candidate can arrive with no content (SAFETY block) or with
        // content but no parts (MAX_TOKENS truncation on the 2.5 series).
        // Both are failures, not empty replies — report the finishReason so
        // the caller can tell truncation from filtering.
        let finish_reason = candidate.finish_reason.as_deref().unwrap_or("unspecified");
        let content = candidate.content.as_ref().ok_or_else(|| {
            RvAgentError::model(format!(
                "Gemini candidate had no content (finishReason: {finish_reason})"
            ))
        })?;
        if content.parts.is_empty() {
            return Err(RvAgentError::model(format!(
                "Gemini candidate had no content parts (finishReason: {finish_reason})"
            )));
        }

        for part in &content.parts {
            if let Some(text) = &part.text {
                text_parts.push(text.clone());
            }
            if let Some(fc) = &part.function_call {
                let n = CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tool_calls.push(ToolCall {
                    id: format!("gemini_call_{n}_{}", fc.name),
                    name: fc.name.clone(),
                    args: fc.args.clone(),
                });
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
}

#[async_trait]
impl ChatModel for GeminiClient {
    async fn complete(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Message> {
        let request = self.build_request(messages, tools);
        let response = self.send_with_retry(&request).await?;
        Self::parse_response(response)
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

/// JSON Schema keywords Gemini's restricted OpenAPI subset accepts.
const GEMINI_SCHEMA_KEYS: &[&str] = &[
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "anyOf",
];

/// Sanitize a tool's input schema into the `parameters` field of a
/// [`FunctionDeclaration`].
///
/// A parameterless tool sanitizes down to a bare `{"type": "object"}` (the
/// in-tree rvagent-mcp registry `ping` tool declares `{"type": "object",
/// "properties": {}}`). `parameters` is optional in the Gemini API, so the
/// field is omitted outright rather than sent as a property-less object.
fn tool_parameters(input_schema: &serde_json::Value) -> Option<serde_json::Value> {
    let sanitized = sanitize_schema(input_schema);
    let is_bare_object = sanitized.get("type").and_then(serde_json::Value::as_str)
        == Some("object")
        && sanitized.get("properties").is_none();
    if is_bare_object {
        None
    } else {
        Some(sanitized)
    }
}

/// `format` values Gemini accepts, keyed by the schema's resolved type.
/// Anything else (uri, email, uuid, date, binary, …) is a 400.
const GEMINI_STRING_FORMATS: &[&str] = &["enum", "date-time"];
const GEMINI_INTEGER_FORMATS: &[&str] = &["int32", "int64"];
const GEMINI_NUMBER_FORMATS: &[&str] = &["float", "double"];

/// Rewrite a JSON Schema into Gemini's restricted OpenAPI subset.
///
/// Beyond stripping keywords Gemini rejects outright (`default`, `$ref`,
/// `oneOf`, `additionalProperties`, `$schema`, `examples`, …), this repairs
/// four shapes that are legal JSON Schema but 400 on Gemini:
///
/// - an OBJECT with an empty (or emptied) property map — the `properties` key
///   is dropped, leaving a bare `{"type": "object"}`;
/// - the OpenAPI 3.1 / pydantic array type form `{"type": ["string", "null"]}`
///   — collapsed to the first non-null entry plus `"nullable": true`;
/// - a `format` the resolved type does not support — dropped;
/// - a subschema left type-less by stripping (it carried only `$ref`/`oneOf`)
///   or an `enum` with no type — defaulted to `"string"`, since a best-effort
///   type beats a guaranteed 400.
///
/// Applied recursively so nested `properties`, `items`, and `anyOf`
/// subschemas are repaired too.
fn sanitize_schema(schema: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;

    let Value::Object(map) = schema else {
        return schema.clone();
    };

    let mut out = serde_json::Map::new();
    for (key, value) in map {
        if !GEMINI_SCHEMA_KEYS.contains(&key.as_str()) {
            continue;
        }
        let cleaned = match key.as_str() {
            "properties" => match value {
                Value::Object(props) => Value::Object(
                    props
                        .iter()
                        .map(|(name, sub)| (name.clone(), sanitize_schema(sub)))
                        .collect(),
                ),
                other => other.clone(),
            },
            // `items` is a subschema, or an array of subschemas in tuple form.
            "items" => match value {
                Value::Array(subs) => Value::Array(subs.iter().map(sanitize_schema).collect()),
                sub => sanitize_schema(sub),
            },
            "anyOf" => match value {
                Value::Array(subs) => Value::Array(subs.iter().map(sanitize_schema).collect()),
                other => other.clone(),
            },
            _ => value.clone(),
        };
        out.insert(key.clone(), cleaned);
    }

    // Collapse the array type form to a single type + nullable flag.
    if let Some(Value::Array(variants)) = out.get("type").cloned() {
        let has_null = variants.iter().any(|v| v.as_str() == Some("null"));
        let primary = variants
            .iter()
            .find(|v| matches!(v.as_str(), Some(s) if s != "null"))
            .cloned()
            .unwrap_or_else(|| Value::String("string".to_string()));
        out.insert("type".to_string(), primary);
        if has_null {
            out.insert("nullable".to_string(), Value::Bool(true));
        }
    }

    // An enum is a string enum unless the schema says otherwise.
    if out.contains_key("enum") && !out.contains_key("type") {
        out.insert("type".to_string(), Value::String("string".to_string()));
    }

    // Gemini rejects OBJECT with an empty property map; a bare
    // `{"type": "object"}` is accepted.
    let properties_empty = out
        .get("properties")
        .is_some_and(|p| p.as_object().is_none_or(|m| m.is_empty()));
    if properties_empty {
        out.remove("properties");
    }

    // `format` is only valid for the types listed above.
    if let Some(format) = out
        .get("format")
        .and_then(Value::as_str)
        .map(str::to_string)
    {
        let supported = match out.get("type").and_then(Value::as_str) {
            Some("string") => GEMINI_STRING_FORMATS.contains(&format.as_str()),
            Some("integer") => GEMINI_INTEGER_FORMATS.contains(&format.as_str()),
            Some("number") => GEMINI_NUMBER_FORMATS.contains(&format.as_str()),
            _ => false,
        };
        if !supported {
            out.remove("format");
        }
    }

    // Nothing left to describe the shape (e.g. the schema was only a `$ref`).
    if !["type", "anyOf", "enum", "properties", "items"]
        .iter()
        .any(|k| out.contains_key(*k))
    {
        out.insert("type".to_string(), Value::String("string".to_string()));
    }

    Value::Object(out)
}

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
            response.candidates[0].content.as_ref().unwrap().parts[0]
                .text
                .as_deref(),
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
        let parts = &response.candidates[0].content.as_ref().unwrap().parts;
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

    #[test]
    fn test_gemini_tool_result_prefers_recorded_tool_name() {
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

        // Orphaned tool message (no preceding AI turn, e.g. after history
        // trimming): the recorded tool_name must win over the raw id.
        let messages = vec![
            Message::human("read it"),
            Message::tool_with_name("some_opaque_id", "contents", "read_file"),
        ];
        let req = client.build_request(&messages, &[]);
        let fr = req.contents[1].parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "read_file");
    }

    #[test]
    fn test_gemini_parallel_tool_results_share_one_content() {
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
            Message::human("read both"),
            Message::ai_with_tools(
                "",
                vec![
                    ToolCall {
                        id: "gemini_call_0_read_file".to_string(),
                        name: "read_file".to_string(),
                        args: serde_json::json!({"path": "/tmp/a"}),
                    },
                    ToolCall {
                        id: "gemini_call_1_list_dir".to_string(),
                        name: "list_dir".to_string(),
                        args: serde_json::json!({"path": "/tmp"}),
                    },
                ],
            ),
            Message::tool("gemini_call_0_read_file", "a contents"),
            Message::tool("gemini_call_1_list_dir", "a\nb"),
        ];
        let req = client.build_request(&messages, &[]);

        // human + model + ONE grouped tool-result turn.
        assert_eq!(req.contents.len(), 3);
        let tool_turn = &req.contents[2];
        assert_eq!(tool_turn.role, "user");
        assert_eq!(tool_turn.parts.len(), 2);
        assert_eq!(
            tool_turn.parts[0]
                .function_response
                .as_ref()
                .unwrap()
                .name
                .as_str(),
            "read_file"
        );
        assert_eq!(
            tool_turn.parts[1]
                .function_response
                .as_ref()
                .unwrap()
                .name
                .as_str(),
            "list_dir"
        );
    }

    #[test]
    fn test_gemini_tool_results_flush_before_following_turn() {
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

        // Two separate model turns, each answered by its own tool result:
        // grouping must not merge across the intervening model turn.
        let messages = vec![
            Message::ai_with_tools(
                "",
                vec![ToolCall {
                    id: "c0".to_string(),
                    name: "first".to_string(),
                    args: serde_json::json!({}),
                }],
            ),
            Message::tool("c0", "one"),
            Message::ai_with_tools(
                "",
                vec![ToolCall {
                    id: "c1".to_string(),
                    name: "second".to_string(),
                    args: serde_json::json!({}),
                }],
            ),
            Message::tool("c1", "two"),
        ];
        let req = client.build_request(&messages, &[]);

        assert_eq!(req.contents.len(), 4);
        assert_eq!(req.contents[0].role, "model");
        assert_eq!(req.contents[1].parts.len(), 1);
        assert_eq!(req.contents[2].role, "model");
        assert_eq!(req.contents[3].parts.len(), 1);
    }

    #[test]
    fn test_gemini_candidate_finish_reason_deserializes() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "truncated..."}]
                },
                "finishReason": "MAX_TOKENS",
                "index": 0
            }],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 1024}
        }"#;

        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            response.candidates[0].finish_reason.as_deref(),
            Some("MAX_TOKENS")
        );
    }

    #[test]
    fn test_gemini_empty_candidates_is_error() {
        let json = r#"{
            "candidates": [],
            "promptFeedback": {"blockReason": "SAFETY"}
        }"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        let err = GeminiClient::parse_response(response).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no candidates"), "unexpected error: {msg}");
        assert!(msg.contains("SAFETY"), "block reason missing: {msg}");
    }

    #[test]
    fn test_gemini_missing_candidates_is_error() {
        // No candidates key at all and no promptFeedback.
        let response: GeminiResponse = serde_json::from_str("{}").unwrap();
        let err = GeminiClient::parse_response(response).unwrap_err();
        assert!(err.to_string().contains("no candidates"));
    }

    #[test]
    fn test_sanitize_schema_strips_unsupported_keys() {
        let schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "path": {
                    "type": "string",
                    "description": "file path",
                    "default": "/tmp",
                    "examples": ["/tmp/a"]
                },
                "mode": {
                    "$ref": "#/definitions/Mode",
                    "oneOf": [{"type": "string"}],
                    "enum": ["read", "write"]
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "default": "x"}
                }
            },
            "required": ["path"]
        });

        let cleaned = sanitize_schema(&schema);

        assert!(cleaned.get("$schema").is_none());
        assert!(cleaned.get("additionalProperties").is_none());
        assert_eq!(cleaned["type"], "object");
        assert_eq!(cleaned["required"], serde_json::json!(["path"]));

        let path = &cleaned["properties"]["path"];
        assert!(path.get("default").is_none());
        assert!(path.get("examples").is_none());
        assert_eq!(path["description"], "file path");

        let mode = &cleaned["properties"]["mode"];
        assert!(mode.get("$ref").is_none());
        assert!(mode.get("oneOf").is_none());
        assert_eq!(mode["enum"], serde_json::json!(["read", "write"]));
        // An enum stripped down to no type must not stay type-less.
        assert_eq!(mode["type"], "string");

        // Nested subschemas under `items` are sanitized too.
        assert!(cleaned["properties"]["tags"]["items"]
            .get("default")
            .is_none());

        // And nothing unsupported survives serialization into the request.
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
            input_schema: schema,
        }];
        let req = client.build_request(&[Message::human("go")], &tools);
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("\"default\""));
        assert!(!json.contains("$ref"));
        assert!(!json.contains("additionalProperties"));
    }

    fn test_client() -> GeminiClient {
        GeminiClient {
            config: ModelConfig {
                provider: rvagent_core::models::Provider::Google,
                model_id: "gemini-2.5-pro".to_string(),
                api_key_source: ApiKeySource::None,
                max_tokens: 1024,
                temperature: 0.0,
            },
            http: reqwest::Client::new(),
            api_key: "test".to_string(),
        }
    }

    #[test]
    fn test_sanitize_schema_drops_empty_properties() {
        // The rvagent-mcp registry `ping` tool declares exactly this.
        let cleaned = sanitize_schema(&serde_json::json!({
            "type": "object",
            "properties": {}
        }));
        assert_eq!(cleaned, serde_json::json!({"type": "object"}));

        // Same when the map is only emptied by sanitizing (map schemas whose
        // sole key was additionalProperties).
        let cleaned = sanitize_schema(&serde_json::json!({
            "type": "object",
            "additionalProperties": {"type": "string"}
        }));
        assert_eq!(cleaned, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn test_parameterless_tool_omits_parameters() {
        let tools = vec![ToolDefinition {
            name: "ping".to_string(),
            description: "Ping the server".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        }];
        let req = test_client().build_request(&[Message::human("ping")], &tools);
        assert!(req.tools[0].function_declarations[0].parameters.is_none());

        // `parameters` must not appear on the wire at all.
        let json = serde_json::to_value(&req).unwrap();
        assert!(json["tools"][0]["functionDeclarations"][0]
            .get("parameters")
            .is_none());
        // A tool that does take parameters still carries them.
        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}}
            }),
        }];
        let req = test_client().build_request(&[Message::human("go")], &tools);
        assert!(req.tools[0].function_declarations[0].parameters.is_some());
    }

    #[test]
    fn test_sanitize_schema_normalizes_array_type_form() {
        // OpenAPI 3.1 / pydantic nullable form.
        let cleaned = sanitize_schema(&serde_json::json!({
            "type": ["string", "null"],
            "description": "maybe a name"
        }));
        assert_eq!(cleaned["type"], "string");
        assert_eq!(cleaned["nullable"], true);
        assert_eq!(cleaned["description"], "maybe a name");

        // No null present: collapse to the first entry, no nullable flag.
        let cleaned = sanitize_schema(&serde_json::json!({"type": ["integer", "string"]}));
        assert_eq!(cleaned["type"], "integer");
        assert!(cleaned.get("nullable").is_none());

        // Degenerate null-only form still yields a usable type.
        let cleaned = sanitize_schema(&serde_json::json!({"type": ["null"]}));
        assert_eq!(cleaned["type"], "string");
        assert_eq!(cleaned["nullable"], true);

        // Nested under properties too.
        let cleaned = sanitize_schema(&serde_json::json!({
            "type": "object",
            "properties": {"nick": {"type": ["string", "null"]}}
        }));
        assert_eq!(cleaned["properties"]["nick"]["type"], "string");
        assert_eq!(cleaned["properties"]["nick"]["nullable"], true);
    }

    #[test]
    fn test_sanitize_schema_filters_unsupported_formats() {
        // Unsupported string formats are dropped, the type survives.
        for format in ["uri", "email", "uuid", "date", "binary", "hostname"] {
            let cleaned = sanitize_schema(&serde_json::json!({"type": "string", "format": format}));
            assert_eq!(
                cleaned,
                serde_json::json!({"type": "string"}),
                "format {format} should have been dropped"
            );
        }

        // Supported ones are kept.
        for format in ["enum", "date-time"] {
            let cleaned = sanitize_schema(&serde_json::json!({"type": "string", "format": format}));
            assert_eq!(cleaned["format"], format);
        }
        for (ty, format) in [
            ("integer", "int32"),
            ("integer", "int64"),
            ("number", "float"),
            ("number", "double"),
        ] {
            let cleaned = sanitize_schema(&serde_json::json!({"type": ty, "format": format}));
            assert_eq!(cleaned["format"], format);
        }

        // Right format, wrong type => dropped.
        let cleaned = sanitize_schema(&serde_json::json!({"type": "string", "format": "int32"}));
        assert!(cleaned.get("format").is_none());
        let cleaned = sanitize_schema(&serde_json::json!({"type": "integer", "format": "double"}));
        assert!(cleaned.get("format").is_none());
    }

    #[test]
    fn test_sanitize_schema_defaults_typeless_subschemas() {
        // A schema carrying only unsupported keys must not end up type-less.
        let cleaned = sanitize_schema(&serde_json::json!({"$ref": "#/definitions/Mode"}));
        assert_eq!(cleaned, serde_json::json!({"type": "string"}));

        let cleaned = sanitize_schema(&serde_json::json!({
            "allOf": [{"type": "string"}],
            "oneOf": [{"type": "integer"}]
        }));
        assert_eq!(cleaned, serde_json::json!({"type": "string"}));

        // Enum without a type gains one.
        let cleaned = sanitize_schema(&serde_json::json!({"enum": ["a", "b"]}));
        assert_eq!(cleaned["type"], "string");
        assert_eq!(cleaned["enum"], serde_json::json!(["a", "b"]));

        // Schemas that already describe a shape are left alone.
        let cleaned = sanitize_schema(&serde_json::json!({"anyOf": [{"type": "string"}]}));
        assert!(cleaned.get("type").is_none());
        let cleaned = sanitize_schema(&serde_json::json!({
            "items": {"type": "string"}, "type": "array"
        }));
        assert_eq!(cleaned["type"], "array");
    }

    #[test]
    fn test_gemini_candidate_without_content_deserializes_and_errors() {
        // SAFETY stop: candidate present, content absent entirely.
        let json = r#"{"candidates": [{"finishReason": "SAFETY", "index": 0}]}"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert!(response.candidates[0].content.is_none());

        let err = GeminiClient::parse_response(response)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no content"), "unexpected error: {err}");
        assert!(err.contains("SAFETY"), "finishReason missing: {err}");
    }

    #[test]
    fn test_gemini_candidate_with_empty_parts_deserializes_and_errors() {
        // MAX_TOKENS truncation on the 2.5 series: content with no parts.
        let json = r#"{
            "candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 1024}
        }"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        assert!(response.candidates[0]
            .content
            .as_ref()
            .unwrap()
            .parts
            .is_empty());

        let err = GeminiClient::parse_response(response)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no content parts"), "unexpected error: {err}");
        assert!(err.contains("MAX_TOKENS"), "finishReason missing: {err}");
    }

    #[test]
    fn test_gemini_candidate_missing_finish_reason_reports_unspecified() {
        let json = r#"{"candidates": [{}]}"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        let err = GeminiClient::parse_response(response)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unspecified"), "unexpected error: {err}");
    }
}
