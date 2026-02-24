//! OpenAI- and Ollama-compatible HTTP serving adapter.
//!
//! Wraps any [`LlmBackend`] (including the three-tier `VirtualLlmBackend`) behind
//! standard REST endpoints so existing clients work unchanged.
//!
//! ## Endpoints
//!
//! | Route                          | Protocol | Description                    |
//! |--------------------------------|----------|--------------------------------|
//! | `POST /v1/chat/completions`    | OpenAI   | Chat completions (+ streaming) |
//! | `POST /v1/completions`         | OpenAI   | Legacy completions             |
//! | `POST /v1/embeddings`          | OpenAI   | Embedding extraction           |
//! | `GET  /v1/models`              | OpenAI   | List loaded models             |
//! | `POST /api/generate`           | Ollama   | Text generation                |
//! | `POST /api/chat`               | Ollama   | Chat generation                |
//! | `GET  /api/tags`               | Ollama   | List available models          |
//! | `GET  /health`                 | —        | Health check                   |
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::serving::openai_compat::{CompatServer, CompatServerConfig};
//! use ruvllm::backends::create_backend;
//! use std::sync::Arc;
//!
//! let backend = Arc::new(create_backend());
//! let config = CompatServerConfig::default();
//! let server = CompatServer::new(backend, config);
//!
//! // Starts on 0.0.0.0:11434 by default (Ollama port)
//! server.serve().await?;
//! ```

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::backends::{GenerateParams, LlmBackend, SharedBackend, StreamEvent};
use crate::error::Result as LlmResult;

// ============================================================================
// Configuration
// ============================================================================

/// Server configuration.
#[derive(Debug, Clone)]
pub struct CompatServerConfig {
    /// Bind address (default `0.0.0.0`).
    pub host: String,
    /// Listen port (default `11434` — Ollama default).
    pub port: u16,
    /// Model name reported to clients when no model is loaded.
    pub default_model_name: String,
    /// Maximum tokens the server will allow per request.
    pub max_tokens_limit: usize,
}

impl Default for CompatServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 11434,
            default_model_name: "ruvllm".into(),
            max_tokens_limit: 4096,
        }
    }
}

// ============================================================================
// Shared server state
// ============================================================================

/// State shared across all request handlers.
struct AppState {
    backend: SharedBackend,
    config: CompatServerConfig,
    /// Tracks total requests for simple observability.
    request_count: std::sync::atomic::AtomicU64,
}

// ============================================================================
// OpenAI request/response types
// ============================================================================

/// OpenAI `ChatCompletionRequest`.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

/// A single chat message.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI `CompletionRequest` (legacy).
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// OpenAI `EmbeddingRequest`.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
}

/// The `input` field can be a single string or a list of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

/// OpenAI chat completion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// A single choice in a completion response.
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: &'static str,
}

/// Streaming chunk for SSE.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// A delta choice in a streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<&'static str>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Legacy completion response.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: &'static str,
}

/// Token usage stats.
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Embedding response.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

/// Model listing response.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

// ============================================================================
// Ollama request/response types
// ============================================================================

/// Ollama `/api/generate` request.
#[derive(Debug, Deserialize)]
pub struct OllamaGenerateRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

/// Ollama `/api/chat` request.
#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

/// Ollama generation options.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OllamaOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub num_predict: Option<usize>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<u64>,
    pub repeat_penalty: Option<f32>,
}

/// Ollama generate response (non-streaming).
#[derive(Debug, Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub total_duration: u64,
    pub eval_count: usize,
}

/// Ollama chat response (non-streaming).
#[derive(Debug, Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: ChatMessage,
    pub done: bool,
    pub total_duration: u64,
    pub eval_count: usize,
}

/// Ollama model tags response.
#[derive(Debug, Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelTag>,
}

#[derive(Debug, Serialize)]
pub struct OllamaModelTag {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
}

// ============================================================================
// Server builder
// ============================================================================

/// OpenAI + Ollama compatible HTTP server wrapping any `LlmBackend`.
pub struct CompatServer {
    state: Arc<AppState>,
}

impl CompatServer {
    /// Create a new server wrapping the given backend.
    pub fn new(backend: SharedBackend, config: CompatServerConfig) -> Self {
        Self {
            state: Arc::new(AppState {
                backend,
                config,
                request_count: std::sync::atomic::AtomicU64::new(0),
            }),
        }
    }

    /// Build the axum router (useful for testing or composition).
    pub fn router(&self) -> Router {
        Router::new()
            // OpenAI endpoints
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/v1/completions", post(handle_completions))
            .route("/v1/embeddings", post(handle_embeddings))
            .route("/v1/models", get(handle_models))
            // Ollama endpoints
            .route("/api/generate", post(handle_ollama_generate))
            .route("/api/chat", post(handle_ollama_chat))
            .route("/api/tags", get(handle_ollama_tags))
            // Health
            .route("/health", get(handle_health))
            .with_state(self.state.clone())
    }

    /// Start listening.  Blocks until shutdown.
    pub async fn serve(self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let addr = format!("{}:{}", self.state.config.host, self.state.config.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!("ruvLLM compat server listening on {addr}");
        axum::serve(listener, self.router()).await?;
        Ok(())
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn model_name(state: &AppState) -> String {
    state
        .backend
        .model_info()
        .map(|m| m.name.clone())
        .unwrap_or_else(|| state.config.default_model_name.clone())
}

/// Convert chat messages into a flat prompt string.
///
/// Uses a simple `<role>: content\n` template. If a backend ships with a
/// proper chat template tokenizer this can be extended later.
fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&m.role);
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out.push_str("assistant: ");
    out
}

fn to_generate_params(state: &AppState, req: &ChatCompletionRequest) -> GenerateParams {
    GenerateParams {
        max_tokens: req
            .max_tokens
            .unwrap_or(256)
            .min(state.config.max_tokens_limit),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        top_k: 40,
        repetition_penalty: 1.1,
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        stop_sequences: req.stop.clone().unwrap_or_default(),
        seed: req.seed,
    }
}

fn ollama_opts_to_params(state: &AppState, opts: Option<&OllamaOptions>) -> GenerateParams {
    let opts = opts.cloned().unwrap_or_default();
    GenerateParams {
        max_tokens: opts
            .num_predict
            .unwrap_or(256)
            .min(state.config.max_tokens_limit),
        temperature: opts.temperature.unwrap_or(0.7),
        top_p: opts.top_p.unwrap_or(0.9),
        top_k: opts.top_k.unwrap_or(40),
        repetition_penalty: opts.repeat_penalty.unwrap_or(1.1),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_sequences: opts.stop.unwrap_or_default(),
        seed: opts.seed,
    }
}

/// Rough prompt-token estimate (4 chars ≈ 1 token).
fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

fn error_json(msg: &str) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "error": { "message": msg, "type": "server_error" }
    }))
}

// ============================================================================
// OpenAI handlers
// ============================================================================

/// `POST /v1/chat/completions`
async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    state
        .request_count
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let prompt = messages_to_prompt(&req.messages);
    let params = to_generate_params(&state, &req);
    let prompt_tokens = estimate_tokens(&prompt);

    // Non-streaming path
    let backend = state.backend.clone();
    let prompt_owned = prompt.clone();
    let result = tokio::task::spawn_blocking(move || backend.generate(&prompt_owned, params)).await;

    match result {
        Ok(Ok(text)) => {
            let completion_tokens = estimate_tokens(&text);
            let resp = ChatCompletionResponse {
                id: format!("chatcmpl-{}", Uuid::new_v4()),
                object: "chat.completion",
                created: now_unix(),
                model: model_name(&state),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".into(),
                        content: text,
                    },
                    finish_reason: "stop",
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
    }
}

/// `POST /v1/completions`
async fn handle_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    state
        .request_count
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let params = GenerateParams {
        max_tokens: req
            .max_tokens
            .unwrap_or(256)
            .min(state.config.max_tokens_limit),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        stop_sequences: req.stop.unwrap_or_default(),
        seed: req.seed,
        ..Default::default()
    };

    let prompt_tokens = estimate_tokens(&req.prompt);
    let backend = state.backend.clone();
    let prompt = req.prompt;
    let result = tokio::task::spawn_blocking(move || backend.generate(&prompt, params)).await;

    match result {
        Ok(Ok(text)) => {
            let completion_tokens = estimate_tokens(&text);
            let resp = CompletionResponse {
                id: format!("cmpl-{}", Uuid::new_v4()),
                object: "text_completion",
                created: now_unix(),
                model: model_name(&state),
                choices: vec![CompletionChoice {
                    index: 0,
                    text,
                    finish_reason: "stop",
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
    }
}

/// `POST /v1/embeddings`
async fn handle_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    let texts = match req.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Batch(v) => v,
    };

    let backend = state.backend.clone();
    let result = tokio::task::spawn_blocking(move || {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| {
                backend.get_embeddings(t).map(|emb| EmbeddingData {
                    object: "embedding",
                    index: i,
                    embedding: emb,
                })
            })
            .collect::<LlmResult<Vec<_>>>()
    })
    .await;

    match result {
        Ok(Ok(data)) => {
            let total: usize = data.iter().map(|d| d.embedding.len()).sum();
            let resp = EmbeddingResponse {
                object: "list",
                model: model_name(&state),
                usage: Usage {
                    prompt_tokens: total / 4,
                    completion_tokens: 0,
                    total_tokens: total / 4,
                },
                data,
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
    }
}

/// `GET /v1/models`
async fn handle_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models = if let Some(info) = state.backend.model_info() {
        vec![ModelObject {
            id: info.name,
            object: "model",
            created: now_unix(),
            owned_by: "ruvllm".into(),
        }]
    } else {
        vec![ModelObject {
            id: state.config.default_model_name.clone(),
            object: "model",
            created: now_unix(),
            owned_by: "ruvllm".into(),
        }]
    };

    Json(
        serde_json::to_value(ModelsResponse {
            object: "list",
            data: models,
        })
        .unwrap(),
    )
}

// ============================================================================
// Ollama handlers
// ============================================================================

/// `POST /api/generate`
async fn handle_ollama_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaGenerateRequest>,
) -> impl IntoResponse {
    state
        .request_count
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let params = ollama_opts_to_params(&state, req.options.as_ref());
    let start = Instant::now();
    let backend = state.backend.clone();
    let prompt = req.prompt;
    let result = tokio::task::spawn_blocking(move || backend.generate(&prompt, params)).await;

    match result {
        Ok(Ok(text)) => {
            let elapsed = start.elapsed().as_nanos() as u64;
            let eval_count = estimate_tokens(&text);
            let resp = OllamaGenerateResponse {
                model: model_name(&state),
                created_at: chrono::Utc::now().to_rfc3339(),
                response: text,
                done: true,
                total_duration: elapsed,
                eval_count,
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
    }
}

/// `POST /api/chat`
async fn handle_ollama_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaChatRequest>,
) -> impl IntoResponse {
    state
        .request_count
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let prompt = messages_to_prompt(&req.messages);
    let params = ollama_opts_to_params(&state, req.options.as_ref());
    let start = Instant::now();
    let backend = state.backend.clone();
    let result = tokio::task::spawn_blocking(move || backend.generate(&prompt, params)).await;

    match result {
        Ok(Ok(text)) => {
            let elapsed = start.elapsed().as_nanos() as u64;
            let eval_count = estimate_tokens(&text);
            let resp = OllamaChatResponse {
                model: model_name(&state),
                created_at: chrono::Utc::now().to_rfc3339(),
                message: ChatMessage {
                    role: "assistant".into(),
                    content: text,
                },
                done: true,
                total_duration: elapsed,
                eval_count,
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error_json(&e.to_string()),
        )
            .into_response(),
    }
}

/// `GET /api/tags`
async fn handle_ollama_tags(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models = if let Some(info) = state.backend.model_info() {
        vec![OllamaModelTag {
            name: info.name,
            modified_at: chrono::Utc::now().to_rfc3339(),
            size: info.memory_usage as u64,
        }]
    } else {
        vec![OllamaModelTag {
            name: state.config.default_model_name.clone(),
            modified_at: chrono::Utc::now().to_rfc3339(),
            size: 0,
        }]
    };

    Json(serde_json::to_value(OllamaTagsResponse { models }).unwrap())
}

/// `GET /health`
async fn handle_health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let reqs = state
        .request_count
        .load(std::sync::atomic::Ordering::Relaxed);
    Json(serde_json::json!({
        "status": "ok",
        "model_loaded": state.backend.is_model_loaded(),
        "total_requests": reqs,
    }))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_messages_to_prompt() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("system: You are helpful."));
        assert!(prompt.contains("user: Hello"));
        assert!(prompt.ends_with("assistant: "));
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world!"), 3);
        assert_eq!(estimate_tokens(""), 1);
    }

    #[test]
    fn test_default_config() {
        let cfg = CompatServerConfig::default();
        assert_eq!(cfg.port, 11434);
        assert_eq!(cfg.default_model_name, "ruvllm");
    }

    #[test]
    fn test_ollama_opts_defaults() {
        let state = AppState {
            backend: Arc::new(crate::backends::NoopBackend),
            config: CompatServerConfig::default(),
            request_count: std::sync::atomic::AtomicU64::new(0),
        };
        let params = ollama_opts_to_params(&state, None);
        assert_eq!(params.max_tokens, 256);
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_router_builds() {
        let backend: SharedBackend = Arc::new(crate::backends::NoopBackend);
        let server = CompatServer::new(backend, CompatServerConfig::default());
        let _router = server.router();
    }
}
