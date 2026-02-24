//! Unified inference backend trait and adapters.
//!
//! Provides a common async trait (`UnifiedInferenceBackend`) that abstracts
//! over three different inference backends:
//!
//! - **LocalCandleAdapter**: Wraps the sync `CandleBackend` via `spawn_blocking`
//! - **OllamaAdapter**: Wraps the async `OllamaBackend`
//! - **ClaudeAdapter**: Calls Claude Messages API via reqwest
//!
//! This trait exists alongside `LlmBackend` (sync) to avoid breaking the
//! existing candle ecosystem while enabling async multi-backend routing.

use crate::backends::LlmBackend;
use crate::error::{Result, RuvLLMError};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================================
// Unified Types
// ============================================================================

/// Backend-agnostic inference request
#[derive(Debug, Clone)]
pub struct UnifiedRequest {
    /// Input prompt or message
    pub prompt: String,
    /// System prompt (optional)
    pub system: Option<String>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

impl Default for UnifiedRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            system: None,
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            stop_sequences: Vec::new(),
        }
    }
}

impl UnifiedRequest {
    /// Create a new request with a prompt
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Default::default()
        }
    }

    /// Set system prompt
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
}

/// Backend-agnostic inference response
#[derive(Debug, Clone)]
pub struct UnifiedResponse {
    /// Generated text
    pub text: String,
    /// Input tokens used (estimated if not exact)
    pub input_tokens: usize,
    /// Output tokens generated
    pub output_tokens: usize,
    /// Time to first token in milliseconds
    pub ttft_ms: u64,
    /// Total generation time in milliseconds
    pub total_ms: u64,
    /// Which backend produced this response
    pub backend_name: String,
}

/// A single streaming token from any backend
#[derive(Debug, Clone)]
pub struct UnifiedStreamToken {
    /// Token text
    pub text: String,
    /// Token index in sequence
    pub index: usize,
    /// Whether this is the final token
    pub done: bool,
}

// ============================================================================
// Unified Trait
// ============================================================================

/// Async trait for unified inference across backends.
///
/// Each adapter implements this trait to provide a consistent interface
/// regardless of whether the underlying backend is sync (Candle), async
/// REST (Ollama), or async HTTP API (Claude).
#[async_trait::async_trait]
pub trait UnifiedInferenceBackend: Send + Sync {
    /// Human-readable backend name (e.g., "local-candle", "ollama", "claude-opus")
    fn name(&self) -> &str;

    /// Check if this backend is currently available and ready
    async fn is_available(&self) -> bool;

    /// Generate a complete response (non-streaming)
    async fn generate(&self, request: &UnifiedRequest) -> Result<UnifiedResponse>;

    /// Generate with streaming, returning a channel of tokens.
    ///
    /// Default implementation calls `generate()` and sends the full
    /// response as a single token. Backends should override for true streaming.
    async fn generate_stream(
        &self,
        request: &UnifiedRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<UnifiedStreamToken>>> {
        let response = self.generate(request).await?;
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let _ = tx
            .send(Ok(UnifiedStreamToken {
                text: response.text,
                index: 0,
                done: true,
            }))
            .await;
        Ok(rx)
    }
}

// ============================================================================
// LocalCandleAdapter
// ============================================================================

/// Adapter that wraps the sync `CandleBackend` for use with the unified trait.
///
/// Uses `tokio::task::spawn_blocking` to run sync inference off the async runtime.
#[cfg(feature = "candle")]
pub struct LocalCandleAdapter {
    backend: std::sync::Arc<parking_lot::Mutex<crate::backends::CandleBackend>>,
}

#[cfg(feature = "candle")]
impl LocalCandleAdapter {
    /// Create from an existing CandleBackend
    pub fn new(backend: crate::backends::CandleBackend) -> Self {
        Self {
            backend: std::sync::Arc::new(parking_lot::Mutex::new(backend)),
        }
    }
}

#[cfg(feature = "candle")]
#[async_trait::async_trait]
impl UnifiedInferenceBackend for LocalCandleAdapter {
    fn name(&self) -> &str {
        "local-candle"
    }

    async fn is_available(&self) -> bool {
        let backend = self.backend.clone();
        tokio::task::spawn_blocking(move || {
            let guard = backend.lock();
            guard.is_model_loaded()
        })
        .await
        .unwrap_or(false)
    }

    async fn generate(&self, request: &UnifiedRequest) -> Result<UnifiedResponse> {
        let backend = self.backend.clone();
        let prompt = request.prompt.clone();
        let params = crate::backends::GenerateParams {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences: request.stop_sequences.clone(),
            ..Default::default()
        };

        let start = Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            let guard = backend.lock();
            guard.generate(&prompt, params)
        })
        .await
        .map_err(|e| RuvLLMError::Backend(format!("Spawn blocking failed: {}", e)))??;

        let total_ms = start.elapsed().as_millis() as u64;

        Ok(UnifiedResponse {
            input_tokens: request.prompt.len() / 4,
            output_tokens: result.len() / 4,
            text: result,
            ttft_ms: total_ms.min(50), // Estimate TTFT for local inference
            total_ms,
            backend_name: "local-candle".to_string(),
        })
    }
}

// ============================================================================
// OllamaAdapter
// ============================================================================

/// Adapter that wraps the `OllamaBackend` for unified inference.
#[cfg(feature = "ollama")]
pub struct OllamaAdapter {
    backend: super::ollama_backend::OllamaBackend,
}

#[cfg(feature = "ollama")]
impl OllamaAdapter {
    /// Create from an existing OllamaBackend
    pub fn new(backend: super::ollama_backend::OllamaBackend) -> Self {
        Self { backend }
    }

    /// Create with default Ollama configuration
    pub fn with_defaults() -> Result<Self> {
        Ok(Self {
            backend: super::ollama_backend::OllamaBackend::with_defaults()?,
        })
    }

    /// Create with a specific model name
    pub fn with_model(model: &str) -> Result<Self> {
        let config = super::ollama_backend::OllamaConfig {
            model: model.to_string(),
            ..Default::default()
        };
        Ok(Self {
            backend: super::ollama_backend::OllamaBackend::new(config)?,
        })
    }
}

#[cfg(feature = "ollama")]
#[async_trait::async_trait]
impl UnifiedInferenceBackend for OllamaAdapter {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn is_available(&self) -> bool {
        self.backend.health_check().await.unwrap_or(false)
    }

    async fn generate(&self, request: &UnifiedRequest) -> Result<UnifiedResponse> {
        let model = &self.backend.config().model;
        let options = Some(super::ollama_backend::OllamaOptions {
            temperature: Some(request.temperature),
            top_p: Some(request.top_p),
            top_k: None,
            num_predict: Some(request.max_tokens),
            repeat_penalty: None,
            seed: None,
            stop: if request.stop_sequences.is_empty() {
                None
            } else {
                Some(request.stop_sequences.clone())
            },
        });

        let start = Instant::now();
        let response = self
            .backend
            .generate(model, &request.prompt, options)
            .await?;

        let total_ms = start.elapsed().as_millis() as u64;
        let ttft_ms = if response.prompt_eval_duration > 0 {
            response.prompt_eval_duration / 1_000_000 // ns to ms
        } else {
            total_ms.min(200)
        };

        Ok(UnifiedResponse {
            text: response.response,
            input_tokens: response.prompt_eval_count,
            output_tokens: response.eval_count,
            ttft_ms,
            total_ms,
            backend_name: "ollama".to_string(),
        })
    }

    async fn generate_stream(
        &self,
        request: &UnifiedRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<UnifiedStreamToken>>> {
        let model = &self.backend.config().model;
        let options = Some(super::ollama_backend::OllamaOptions {
            temperature: Some(request.temperature),
            top_p: Some(request.top_p),
            top_k: None,
            num_predict: Some(request.max_tokens),
            repeat_penalty: None,
            seed: None,
            stop: if request.stop_sequences.is_empty() {
                None
            } else {
                Some(request.stop_sequences.clone())
            },
        });

        let mut ollama_rx = self
            .backend
            .generate_stream(model, &request.prompt, options)
            .await?;

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::spawn(async move {
            let mut index = 0;
            while let Some(chunk_result) = ollama_rx.recv().await {
                match chunk_result {
                    Ok(chunk) => {
                        let token = UnifiedStreamToken {
                            text: chunk.response.clone(),
                            index,
                            done: chunk.done,
                        };
                        index += 1;
                        if tx.send(Ok(token)).await.is_err() {
                            return;
                        }
                        if chunk.done {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                }
            }
        });

        Ok(rx)
    }
}

// ============================================================================
// ClaudeAdapter
// ============================================================================

/// Adapter that calls the Claude Messages API via reqwest.
///
/// Uses the existing `ClaudeRequest`/`ClaudeResponse` types from claude_integration.
#[cfg(feature = "ollama")]
pub struct ClaudeAdapter {
    /// API key for authentication
    api_key: String,
    /// Model ID to use (e.g., "claude-opus-4-20250514")
    model_id: String,
    /// API base URL
    base_url: String,
    /// HTTP client
    client: reqwest::Client,
}

#[cfg(feature = "ollama")]
impl ClaudeAdapter {
    /// Create a new Claude adapter
    ///
    /// # Arguments
    ///
    /// * `api_key` - Anthropic API key
    /// * `model` - Claude model to use
    pub fn new(api_key: impl Into<String>, model: crate::claude_flow::ClaudeModel) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| RuvLLMError::Http(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            api_key: api_key.into(),
            model_id: model.model_id().to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            client,
        })
    }

    /// Create from environment variable ANTHROPIC_API_KEY
    pub fn from_env(model: crate::claude_flow::ClaudeModel) -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            RuvLLMError::Config("ANTHROPIC_API_KEY environment variable not set".to_string())
        })?;
        Self::new(api_key, model)
    }
}

#[cfg(feature = "ollama")]
#[async_trait::async_trait]
impl UnifiedInferenceBackend for ClaudeAdapter {
    fn name(&self) -> &str {
        "claude-api"
    }

    async fn is_available(&self) -> bool {
        // Claude API is available if we have an API key
        !self.api_key.is_empty()
    }

    async fn generate(&self, request: &UnifiedRequest) -> Result<UnifiedResponse> {
        use crate::claude_flow::{ClaudeRequest, ClaudeResponse, ContentBlock, Message, MessageRole};

        let claude_request = ClaudeRequest {
            model: self.model_id.clone(),
            messages: vec![Message::user(&request.prompt)],
            max_tokens: request.max_tokens,
            system: request.system.clone(),
            temperature: Some(request.temperature),
            stream: Some(false),
        };

        let start = Instant::now();

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&claude_request)
            .send()
            .await?;

        let ttft_ms = start.elapsed().as_millis() as u64;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(RuvLLMError::Config(format!(
                    "Claude API auth failure ({}): {}",
                    status, body
                )));
            }
            return Err(RuvLLMError::Http(format!(
                "Claude API error ({}): {}",
                status, body
            )));
        }

        let claude_response: ClaudeResponse = resp.json().await.map_err(|e| {
            RuvLLMError::Serialization(format!("Failed to parse Claude response: {}", e))
        })?;

        let total_ms = start.elapsed().as_millis() as u64;

        // Extract text from response content blocks
        let text = claude_response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(UnifiedResponse {
            text,
            input_tokens: claude_response.usage.input_tokens,
            output_tokens: claude_response.usage.output_tokens,
            ttft_ms,
            total_ms,
            backend_name: format!("claude-{}", self.model_id),
        })
    }

    async fn generate_stream(
        &self,
        request: &UnifiedRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<UnifiedStreamToken>>> {
        use crate::claude_flow::{ClaudeRequest, Message};

        let claude_request = ClaudeRequest {
            model: self.model_id.clone(),
            messages: vec![Message::user(&request.prompt)],
            max_tokens: request.max_tokens,
            system: request.system.clone(),
            temperature: Some(request.temperature),
            stream: Some(true),
        };

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&claude_request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(RuvLLMError::Http(format!(
                "Claude API stream error ({}): {}",
                status, body
            )));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::spawn(async move {
            use tokio_stream::StreamExt;

            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();
            let mut index = 0;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // Claude SSE format: "data: {json}\n\n"
                        while let Some(data_pos) = buffer.find("data: ") {
                            let after_data = &buffer[data_pos + 6..];
                            if let Some(end_pos) = after_data.find('\n') {
                                let json_str = after_data[..end_pos].trim();

                                if json_str == "[DONE]" {
                                    let _ = tx
                                        .send(Ok(UnifiedStreamToken {
                                            text: String::new(),
                                            index,
                                            done: true,
                                        }))
                                        .await;
                                    return;
                                }

                                // Parse the SSE event
                                if let Ok(event) =
                                    serde_json::from_str::<serde_json::Value>(json_str)
                                {
                                    // Extract text delta from content_block_delta events
                                    if event.get("type").and_then(|t| t.as_str())
                                        == Some("content_block_delta")
                                    {
                                        if let Some(text) = event
                                            .get("delta")
                                            .and_then(|d| d.get("text"))
                                            .and_then(|t| t.as_str())
                                        {
                                            let token = UnifiedStreamToken {
                                                text: text.to_string(),
                                                index,
                                                done: false,
                                            };
                                            index += 1;
                                            if tx.send(Ok(token)).await.is_err() {
                                                return;
                                            }
                                        }
                                    } else if event.get("type").and_then(|t| t.as_str())
                                        == Some("message_stop")
                                    {
                                        let _ = tx
                                            .send(Ok(UnifiedStreamToken {
                                                text: String::new(),
                                                index,
                                                done: true,
                                            }))
                                            .await;
                                        return;
                                    }
                                }

                                buffer = buffer[data_pos + 6 + end_pos + 1..].to_string();
                            } else {
                                break; // Incomplete line, wait for more data
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(RuvLLMError::Http(e.to_string()))).await;
                        return;
                    }
                }
            }

            // Stream ended without explicit done signal
            let _ = tx
                .send(Ok(UnifiedStreamToken {
                    text: String::new(),
                    index,
                    done: true,
                }))
                .await;
        });

        Ok(rx)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_request_builder() {
        let req = UnifiedRequest::new("Hello, world!")
            .with_system("You are helpful")
            .with_max_tokens(512)
            .with_temperature(0.5);

        assert_eq!(req.prompt, "Hello, world!");
        assert_eq!(req.system, Some("You are helpful".to_string()));
        assert_eq!(req.max_tokens, 512);
        assert_eq!(req.temperature, 0.5);
    }

    #[test]
    fn test_unified_request_defaults() {
        let req = UnifiedRequest::default();
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, 0.7);
        assert_eq!(req.top_p, 0.9);
        assert!(req.stop_sequences.is_empty());
    }
}
