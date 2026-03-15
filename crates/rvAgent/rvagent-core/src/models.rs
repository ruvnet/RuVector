//! Model resolution and chat model trait.
//!
//! Parses "provider:model" format strings and provides the async `ChatModel` trait.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::messages::Message;

/// A single chunk from a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Incremental text content.
    pub text: String,
    /// Whether this is the final chunk.
    pub is_final: bool,
    /// Accumulated usage (available on final chunk).
    pub usage: Option<StreamUsage>,
}

/// Token usage from a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Known LLM providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Anthropic,
    OpenAi,
    Google,
    Bedrock,
    Fireworks,
    /// Catch-all for unknown / user-defined providers.
    Other(String),
}

impl Provider {
    /// Parse a provider string.
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "anthropic" => Self::Anthropic,
            "openai" => Self::OpenAi,
            "google" | "vertex" => Self::Google,
            "bedrock" | "aws" => Self::Bedrock,
            "fireworks" => Self::Fireworks,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Source for the API key (never store the key directly in config).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ApiKeySource {
    /// Read from an environment variable.
    Env(String),
    /// Read from a file path.
    File(String),
    /// No key required (e.g. local models).
    None,
}

impl Default for ApiKeySource {
    fn default() -> Self {
        Self::None
    }
}

/// Resolved model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Provider enum.
    pub provider: Provider,
    /// Model identifier (the part after the colon).
    pub model_id: String,
    /// Where to obtain the API key.
    #[serde(default)]
    pub api_key_source: ApiKeySource,
    /// Maximum tokens for completion.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> u32 {
    16_384
}

fn default_temperature() -> f32 {
    0.0
}

/// Parse a "provider:model" string into a `ModelConfig`.
///
/// # Examples
/// ```
/// use rvagent_core::models::resolve_model;
/// let cfg = rvagent_core::models::resolve_model("anthropic:claude-sonnet-4-20250514");
/// assert_eq!(cfg.model_id, "claude-sonnet-4-20250514");
/// ```
pub fn resolve_model(model_str: &str) -> ModelConfig {
    let (provider_str, model_id) = match model_str.split_once(':') {
        Some((p, m)) => (p, m.to_string()),
        None => ("anthropic", model_str.to_string()),
    };

    let provider = Provider::from_str_lossy(provider_str);

    let api_key_source = match &provider {
        Provider::Anthropic => ApiKeySource::Env("ANTHROPIC_API_KEY".into()),
        Provider::OpenAi => ApiKeySource::Env("OPENAI_API_KEY".into()),
        Provider::Google => ApiKeySource::Env("GOOGLE_API_KEY".into()),
        Provider::Bedrock => ApiKeySource::Env("AWS_ACCESS_KEY_ID".into()),
        Provider::Fireworks => ApiKeySource::Env("FIREWORKS_API_KEY".into()),
        Provider::Other(_) => ApiKeySource::None,
    };

    ModelConfig {
        provider,
        model_id,
        api_key_source,
        max_tokens: default_max_tokens(),
        temperature: default_temperature(),
    }
}

/// Async trait for chat model implementations.
///
/// Provider-specific crates implement this trait (e.g. `rvagent-anthropic`).
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// Send messages and receive a complete response.
    async fn complete(&self, messages: &[Message]) -> Result<Message>;

    /// Stream a response token-by-token. Returns a vector of incremental messages.
    /// The final element is the complete assembled message.
    async fn stream(&self, messages: &[Message]) -> Result<Vec<Message>>;
}

/// Extended trait for models that support chunk-based streaming.
///
/// Provides incremental `StreamChunk` delivery. Models that do not natively
/// support streaming can fall back to `ChatModel::complete` and return a
/// single final chunk.
#[async_trait]
pub trait StreamingChatModel: ChatModel {
    /// Stream response chunks incrementally.
    async fn stream_chunks(&self, messages: &[Message]) -> Result<Vec<StreamChunk>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_anthropic() {
        let cfg = resolve_model("anthropic:claude-sonnet-4-20250514");
        assert_eq!(cfg.provider, Provider::Anthropic);
        assert_eq!(cfg.model_id, "claude-sonnet-4-20250514");
        assert_eq!(
            cfg.api_key_source,
            ApiKeySource::Env("ANTHROPIC_API_KEY".into())
        );
    }

    #[test]
    fn test_resolve_openai() {
        let cfg = resolve_model("openai:gpt-4o");
        assert_eq!(cfg.provider, Provider::OpenAi);
        assert_eq!(cfg.model_id, "gpt-4o");
    }

    #[test]
    fn test_resolve_no_provider() {
        let cfg = resolve_model("claude-sonnet-4-20250514");
        assert_eq!(cfg.provider, Provider::Anthropic);
        assert_eq!(cfg.model_id, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_resolve_unknown_provider() {
        let cfg = resolve_model("custom:my-model");
        assert!(matches!(cfg.provider, Provider::Other(ref s) if s == "custom"));
        assert_eq!(cfg.model_id, "my-model");
        assert_eq!(cfg.api_key_source, ApiKeySource::None);
    }

    #[test]
    fn test_resolve_google_aliases() {
        let cfg1 = resolve_model("google:gemini-pro");
        assert_eq!(cfg1.provider, Provider::Google);
        let cfg2 = resolve_model("vertex:gemini-pro");
        assert_eq!(cfg2.provider, Provider::Google);
    }

    #[test]
    fn test_model_config_defaults() {
        let cfg = resolve_model("anthropic:test");
        assert_eq!(cfg.max_tokens, 16_384);
        assert_eq!(cfg.temperature, 0.0);
    }

    #[test]
    fn test_model_config_serde() {
        let cfg = resolve_model("openai:gpt-4o");
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.provider, cfg.provider);
        assert_eq!(back.model_id, cfg.model_id);
    }

    #[test]
    fn test_stream_chunk_serialization() {
        let chunk = StreamChunk {
            text: "Hello".into(),
            is_final: false,
            usage: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: StreamChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text, "Hello");
        assert!(!back.is_final);
        assert!(back.usage.is_none());
    }

    #[test]
    fn test_stream_chunk_with_usage() {
        let chunk = StreamChunk {
            text: "done".into(),
            is_final: true,
            usage: Some(StreamUsage {
                input_tokens: 10,
                output_tokens: 25,
            }),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: StreamChunk = serde_json::from_str(&json).unwrap();
        assert!(back.is_final);
        let usage = back.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 25);
    }

    #[test]
    fn test_stream_usage_serialization() {
        let usage = StreamUsage {
            input_tokens: 42,
            output_tokens: 100,
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("42"));
        assert!(json.contains("100"));
        let back: StreamUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.input_tokens, 42);
        assert_eq!(back.output_tokens, 100);
    }

    #[test]
    fn test_stream_chunk_vec_serialization() {
        let chunks = vec![
            StreamChunk {
                text: "Hel".into(),
                is_final: false,
                usage: None,
            },
            StreamChunk {
                text: "lo!".into(),
                is_final: true,
                usage: Some(StreamUsage {
                    input_tokens: 5,
                    output_tokens: 2,
                }),
            },
        ];
        let json = serde_json::to_string(&chunks).unwrap();
        let back: Vec<StreamChunk> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].text, "Hel");
        assert!(!back[0].is_final);
        assert_eq!(back[1].text, "lo!");
        assert!(back[1].is_final);
    }
}
