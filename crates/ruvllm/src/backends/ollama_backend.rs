//! Ollama REST API backend for local LLM inference.
//!
//! Provides a client for the Ollama API, supporting both streaming and
//! non-streaming generation. Ollama runs models like Llama 3.1 locally
//! and exposes a REST API on localhost:11434.
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::backends::ollama_backend::{OllamaBackend, OllamaConfig};
//!
//! let config = OllamaConfig::default();
//! let backend = OllamaBackend::new(config);
//!
//! // Check health
//! let healthy = backend.health_check().await?;
//!
//! // Generate text
//! let response = backend.generate("llama3.1:8b", "Hello!", None).await?;
//! println!("{}", response.response);
//! ```

use crate::error::{Result, RuvLLMError};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the Ollama backend
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Base URL for Ollama API (default: http://localhost:11434)
    pub base_url: String,
    /// Default model name (default: llama3.1:8b)
    pub model: String,
    /// Request timeout
    pub timeout: Duration,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Keep-alive duration for connection pool
    pub keep_alive: Option<Duration>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3.1:8b".to_string(),
            timeout: Duration::from_secs(120),
            connect_timeout: Duration::from_secs(5),
            keep_alive: Some(Duration::from_secs(300)),
        }
    }
}

/// Request body for Ollama /api/generate endpoint
#[derive(Debug, Clone, Serialize)]
pub struct OllamaGenerateRequest {
    /// Model name
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// System prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Whether to stream the response
    pub stream: bool,
    /// Generation options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
}

/// Request body for Ollama /api/chat endpoint
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatRequest {
    /// Model name
    pub model: String,
    /// Chat messages
    pub messages: Vec<OllamaChatMessage>,
    /// Whether to stream the response
    pub stream: bool,
    /// Generation options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
}

/// A chat message for the Ollama API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatMessage {
    /// Message role (system, user, assistant)
    pub role: String,
    /// Message content
    pub content: String,
}

/// Generation options for Ollama
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaOptions {
    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<usize>,
    /// Repetition penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    /// Random seed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// Response from Ollama /api/generate (non-streaming, final response)
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaGenerateResponse {
    /// Model used
    pub model: String,
    /// Generated text
    pub response: String,
    /// Whether generation is complete
    pub done: bool,
    /// Total duration in nanoseconds
    #[serde(default)]
    pub total_duration: u64,
    /// Model load duration in nanoseconds
    #[serde(default)]
    pub load_duration: u64,
    /// Prompt evaluation count
    #[serde(default)]
    pub prompt_eval_count: usize,
    /// Prompt evaluation duration in nanoseconds
    #[serde(default)]
    pub prompt_eval_duration: u64,
    /// Token evaluation count
    #[serde(default)]
    pub eval_count: usize,
    /// Token evaluation duration in nanoseconds
    #[serde(default)]
    pub eval_duration: u64,
}

/// Response from Ollama /api/chat (non-streaming, final response)
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatResponse {
    /// Model used
    pub model: String,
    /// Assistant message
    pub message: OllamaChatMessage,
    /// Whether generation is complete
    pub done: bool,
    /// Total duration in nanoseconds
    #[serde(default)]
    pub total_duration: u64,
    /// Token evaluation count
    #[serde(default)]
    pub eval_count: usize,
    /// Token evaluation duration in nanoseconds
    #[serde(default)]
    pub eval_duration: u64,
}

/// A single streaming chunk from Ollama (newline-delimited JSON)
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaStreamChunk {
    /// Model used
    pub model: String,
    /// Partial response text (for /api/generate)
    #[serde(default)]
    pub response: String,
    /// Partial message (for /api/chat)
    pub message: Option<OllamaChatMessage>,
    /// Whether this is the final chunk
    pub done: bool,
    /// Total duration (only on final chunk)
    #[serde(default)]
    pub total_duration: u64,
    /// Eval count (only on final chunk)
    #[serde(default)]
    pub eval_count: usize,
    /// Eval duration (only on final chunk)
    #[serde(default)]
    pub eval_duration: u64,
}

/// Ollama REST API client
pub struct OllamaBackend {
    /// Configuration
    config: OllamaConfig,
    /// HTTP client
    client: reqwest::Client,
}

impl OllamaBackend {
    /// Create a new Ollama backend with the given configuration
    pub fn new(config: OllamaConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .pool_idle_timeout(config.keep_alive)
            .build()
            .map_err(|e| RuvLLMError::Http(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { config, client })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(OllamaConfig::default())
    }

    /// Get the current configuration
    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }

    /// Check if Ollama is running and accessible
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.config.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(e) if e.is_connect() => Ok(false),
            Err(e) if e.is_timeout() => Ok(false),
            Err(e) => Err(RuvLLMError::Http(format!(
                "Health check failed: {}",
                e
            ))),
        }
    }

    /// Generate text using /api/generate (non-streaming)
    pub async fn generate(
        &self,
        model: &str,
        prompt: &str,
        options: Option<OllamaOptions>,
    ) -> Result<OllamaGenerateResponse> {
        let url = format!("{}/api/generate", self.config.base_url);
        let request = OllamaGenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            system: None,
            stream: false,
            options,
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(RuvLLMError::Http(format!(
                "Ollama generate failed ({}): {}",
                status, body
            )));
        }

        resp.json::<OllamaGenerateResponse>()
            .await
            .map_err(|e| RuvLLMError::Serialization(format!("Failed to parse response: {}", e)))
    }

    /// Generate text using /api/generate with streaming.
    ///
    /// Returns a receiver that yields token strings as they arrive.
    /// The final message on the channel is `None`, indicating completion.
    pub async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        options: Option<OllamaOptions>,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<OllamaStreamChunk>>> {
        let url = format!("{}/api/generate", self.config.base_url);
        let request = OllamaGenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            system: None,
            stream: true,
            options,
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(RuvLLMError::Http(format!(
                "Ollama stream failed ({}): {}",
                status, body
            )));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        // Spawn task to read newline-delimited JSON chunks
        tokio::spawn(async move {
            use futures_core::Stream;
            use tokio_stream::StreamExt;

            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // Process complete lines (Ollama sends newline-delimited JSON)
                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer = buffer[newline_pos + 1..].to_string();

                            if line.is_empty() {
                                continue;
                            }

                            match serde_json::from_str::<OllamaStreamChunk>(&line) {
                                Ok(chunk) => {
                                    let done = chunk.done;
                                    if tx.send(Ok(chunk)).await.is_err() {
                                        return; // Receiver dropped
                                    }
                                    if done {
                                        return;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx
                                        .send(Err(RuvLLMError::Serialization(format!(
                                            "Failed to parse stream chunk: {}",
                                            e
                                        ))))
                                        .await;
                                    return;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(RuvLLMError::Http(e.to_string()))).await;
                        return;
                    }
                }
            }
        });

        Ok(rx)
    }

    /// Chat using /api/chat (non-streaming)
    pub async fn chat(
        &self,
        model: &str,
        messages: Vec<OllamaChatMessage>,
        options: Option<OllamaOptions>,
    ) -> Result<OllamaChatResponse> {
        let url = format!("{}/api/chat", self.config.base_url);
        let request = OllamaChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
            options,
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(RuvLLMError::Http(format!(
                "Ollama chat failed ({}): {}",
                status, body
            )));
        }

        resp.json::<OllamaChatResponse>()
            .await
            .map_err(|e| RuvLLMError::Serialization(format!("Failed to parse response: {}", e)))
    }

    /// Chat using /api/chat with streaming.
    ///
    /// Returns a receiver that yields stream chunks as they arrive.
    pub async fn chat_stream(
        &self,
        model: &str,
        messages: Vec<OllamaChatMessage>,
        options: Option<OllamaOptions>,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<OllamaStreamChunk>>> {
        let url = format!("{}/api/chat", self.config.base_url);
        let request = OllamaChatRequest {
            model: model.to_string(),
            messages,
            stream: true,
            options,
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(RuvLLMError::Http(format!(
                "Ollama chat stream failed ({}): {}",
                status, body
            )));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::spawn(async move {
            use futures_core::Stream;
            use tokio_stream::StreamExt;

            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer = buffer[newline_pos + 1..].to_string();

                            if line.is_empty() {
                                continue;
                            }

                            match serde_json::from_str::<OllamaStreamChunk>(&line) {
                                Ok(chunk) => {
                                    let done = chunk.done;
                                    if tx.send(Ok(chunk)).await.is_err() {
                                        return;
                                    }
                                    if done {
                                        return;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx
                                        .send(Err(RuvLLMError::Serialization(format!(
                                            "Failed to parse stream chunk: {}",
                                            e
                                        ))))
                                        .await;
                                    return;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(RuvLLMError::Http(e.to_string()))).await;
                        return;
                    }
                }
            }
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama3.1:8b");
        assert_eq!(config.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_generate_request_serialization() {
        let request = OllamaGenerateRequest {
            model: "llama3.1:8b".to_string(),
            prompt: "Hello".to_string(),
            system: None,
            stream: false,
            options: Some(OllamaOptions {
                temperature: Some(0.7),
                top_p: None,
                top_k: None,
                num_predict: Some(256),
                repeat_penalty: None,
                seed: None,
                stop: None,
            }),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("llama3.1:8b"));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_stream_chunk_deserialization() {
        let json = r#"{"model":"llama3.1:8b","response":"Hello","done":false}"#;
        let chunk: OllamaStreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.response, "Hello");
        assert!(!chunk.done);

        let final_json = r#"{"model":"llama3.1:8b","response":"","done":true,"total_duration":1234567890,"eval_count":42,"eval_duration":987654321}"#;
        let final_chunk: OllamaStreamChunk = serde_json::from_str(final_json).unwrap();
        assert!(final_chunk.done);
        assert_eq!(final_chunk.eval_count, 42);
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = OllamaChatMessage {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
    }
}
