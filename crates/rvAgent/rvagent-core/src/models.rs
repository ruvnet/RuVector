//! Model resolution and chat model trait.
//!
//! Parses "provider:model" format strings and provides the async `ChatModel` trait.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::messages::Message;

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
}
