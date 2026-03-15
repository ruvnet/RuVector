//! Integration tests for model resolution in `rvagent_core::models`.
//!
//! Tests cover provider parsing, API key source mapping, and handling
//! of unknown/custom provider strings.

use rvagent_core::models::{resolve_model, ApiKeySource, Provider};

/// Anthropic provider should be recognized and map to the correct API key env var.
#[test]
fn test_resolve_model_anthropic() {
    let cfg = resolve_model("anthropic:claude-sonnet-4-20250514");

    assert_eq!(cfg.provider, Provider::Anthropic);
    assert_eq!(cfg.model_id, "claude-sonnet-4-20250514");
    assert_eq!(
        cfg.api_key_source,
        ApiKeySource::Env("ANTHROPIC_API_KEY".to_string())
    );
    assert_eq!(cfg.max_tokens, 16_384);
    assert_eq!(cfg.temperature, 0.0);

    // Bare model name (no provider prefix) defaults to Anthropic.
    let bare = resolve_model("claude-sonnet-4-20250514");
    assert_eq!(bare.provider, Provider::Anthropic);
    assert_eq!(bare.model_id, "claude-sonnet-4-20250514");
}

/// OpenAI provider should be recognized with correct key source.
#[test]
fn test_resolve_model_openai() {
    let cfg = resolve_model("openai:gpt-4o");

    assert_eq!(cfg.provider, Provider::OpenAi);
    assert_eq!(cfg.model_id, "gpt-4o");
    assert_eq!(
        cfg.api_key_source,
        ApiKeySource::Env("OPENAI_API_KEY".to_string())
    );

    // Case-insensitive (the implementation lowercases).
    let cfg2 = resolve_model("OpenAI:gpt-4o-mini");
    assert_eq!(cfg2.provider, Provider::OpenAi);
    assert_eq!(cfg2.model_id, "gpt-4o-mini");
}

/// Unknown / custom providers should use Provider::Other with no API key.
#[test]
fn test_resolve_model_custom_provider() {
    let cfg = resolve_model("custom:my-local-model");

    assert!(
        matches!(cfg.provider, Provider::Other(ref s) if s == "custom"),
        "expected Provider::Other(\"custom\"), got {:?}",
        cfg.provider
    );
    assert_eq!(cfg.model_id, "my-local-model");
    assert_eq!(cfg.api_key_source, ApiKeySource::None);

    // Another custom provider.
    let cfg2 = resolve_model("ollama:llama3");
    assert!(matches!(cfg2.provider, Provider::Other(ref s) if s == "ollama"));
    assert_eq!(cfg2.model_id, "llama3");
    assert_eq!(cfg2.api_key_source, ApiKeySource::None);
}

/// A model string with no colon should default to Anthropic provider.
#[test]
fn test_invalid_model_string() {
    // No colon -> treated as anthropic:<full string>
    let cfg = resolve_model("some-model-name");
    assert_eq!(cfg.provider, Provider::Anthropic);
    assert_eq!(cfg.model_id, "some-model-name");
    assert_eq!(
        cfg.api_key_source,
        ApiKeySource::Env("ANTHROPIC_API_KEY".to_string())
    );

    // Empty string -> anthropic with empty model id
    let empty = resolve_model("");
    assert_eq!(empty.provider, Provider::Anthropic);
    assert_eq!(empty.model_id, "");

    // Google aliases
    let g1 = resolve_model("google:gemini-pro");
    assert_eq!(g1.provider, Provider::Google);
    let g2 = resolve_model("vertex:gemini-pro");
    assert_eq!(g2.provider, Provider::Google);

    // Bedrock aliases
    let b1 = resolve_model("bedrock:claude-v2");
    assert_eq!(b1.provider, Provider::Bedrock);
    let b2 = resolve_model("aws:claude-v2");
    assert_eq!(b2.provider, Provider::Bedrock);
}
