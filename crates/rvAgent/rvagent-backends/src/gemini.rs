//! Google Gemini API backend for rvAgent.
//!
//! Implements the [`ChatModel`] trait using the Google Generative AI API.
//! Supports text completions and automatic retry with exponential backoff.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use rvagent_core::error::{Result, RvAgentError};
use rvagent_core::messages::{Message, AiMessage};
use rvagent_core::models::{ApiKeySource, ChatModel, ModelConfig};

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

/// Content part in a Gemini message.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Part {
    text: String,
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
struct GeminiResponse {
    candidates: Vec<Candidate>,
    #[allow(dead_code)]
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
}

/// Token usage information.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    #[allow(dead_code)]
    prompt_token_count: Option<u64>,
    #[allow(dead_code)]
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
/// let response = client.complete(&[Message::human("Hello!")]).await?;
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

    /// Build the API request body from rvAgent messages.
    fn build_request(&self, messages: &[Message]) -> GeminiRequest {
        let mut system_instruction: Option<GeminiContent> = None;
        let mut contents: Vec<GeminiContent> = Vec::new();

        for msg in messages {
            match msg {
                Message::System(s) => {
                    system_instruction = Some(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![Part { text: s.content.clone() }],
                    });
                }
                Message::Human(h) => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![Part { text: h.content.clone() }],
                    });
                }
                Message::Ai(ai) => {
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts: vec![Part { text: ai.content.clone() }],
                    });
                }
                Message::Tool(t) => {
                    // Tool results go as user messages
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![Part { text: format!("Tool result: {}", t.content) }],
                    });
                }
            }
        }

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
        }
    }

    /// Send a request to the API with retry logic.
    async fn send_with_retry(&self, request_body: &GeminiRequest) -> Result<GeminiResponse> {
        let url = format!(
            "{}/{}:generateContent?key={}",
            GEMINI_API_BASE,
            self.config.model_id,
            self.api_key
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

        Err(last_err.unwrap_or_else(|| {
            RvAgentError::model("Gemini API request failed after all retries")
        }))
    }
}

#[async_trait]
impl ChatModel for GeminiClient {
    async fn complete(&self, messages: &[Message]) -> Result<Message> {
        let request = self.build_request(messages);
        let response = self.send_with_retry(&request).await?;

        // Extract text from first candidate
        let text = response
            .candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .map(|p| p.text.clone())
            .unwrap_or_default();

        Ok(Message::Ai(AiMessage {
            content: text,
            tool_calls: vec![],
            metadata: std::collections::HashMap::new(),
        }))
    }

    async fn stream(&self, messages: &[Message]) -> Result<Vec<Message>> {
        // For now, use non-streaming completion
        let msg = self.complete(messages).await?;
        Ok(vec![msg])
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_api_key(source: &ApiKeySource) -> Result<String> {
    match source {
        ApiKeySource::Env(var) => std::env::var(var).map_err(|_| {
            RvAgentError::config(format!(
                "API key environment variable '{var}' not set"
            ))
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
                parts: vec![Part { text: "Hello".to_string() }],
            }],
            generation_config: GenerationConfig {
                max_output_tokens: 1024,
                temperature: Some(0.7),
            },
            system_instruction: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"maxOutputTokens\":1024"));
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
        assert_eq!(response.candidates[0].content.parts[0].text, "Hello there!");
    }
}
