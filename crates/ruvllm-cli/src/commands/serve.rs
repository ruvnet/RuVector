//! Inference server command implementation
//!
//! Starts an OpenAI-compatible HTTP server for model inference,
//! providing endpoints for chat completions, health checks, and metrics.
//! Supports Server-Sent Events (SSE) for streaming responses.
//!
//! The server exits when the model fails to load. Placeholder ("mock mode")
//! completions are served only with `--allow-mock`, and are labeled: an
//! `x-ruvllm-mode: mock` header, `"system_fingerprint": "ruvllm-mock"` and
//! text that says it is not model output.

use anyhow::{Context, Result};
use axum::{
    extract::{rejection::JsonRejection, Json, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use colored::Colorize;
use console::style;
use futures::stream;
use ruvllm::{ChatTemplate, FinishReason, GenerateParams, LlmBackend, ModelArchitecture, Role};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::models::{resolve_model_id, QuantPreset};

/// `system_fingerprint` of every placeholder (mock mode) response.
const MOCK_FINGERPRINT: &str = "ruvllm-mock";

/// Server state
struct ServerState {
    model_id: String,
    /// The loaded model. `None` only in mock mode (`--allow-mock` after a
    /// failed load).
    backend: Option<Arc<dyn LlmBackend>>,
    /// Serve labeled placeholder completions while no model is loaded.
    allow_mock: bool,
    /// Prompt format used to render `messages` for the loaded model.
    chat_template: ChatTemplate,
    request_count: u64,
    total_tokens: u64,
    start_time: Instant,
}

type SharedState = Arc<RwLock<ServerState>>;

/// Run the serve command
pub async fn run(
    model: &str,
    host: &str,
    port: u16,
    max_concurrent: usize,
    max_context: usize,
    quantization: &str,
    cache_dir: &str,
    allow_mock: bool,
    strict: bool,
) -> Result<()> {
    if strict && allow_mock {
        anyhow::bail!(
            "--allow-mock conflicts with --strict/RUVLLM_STRICT. Exiting on a failed model \
             load is now the default, so --strict is no longer needed."
        );
    }

    let quant = QuantPreset::from_str(quantization)
        .ok_or_else(|| anyhow::anyhow!("Invalid quantization format: {}", quantization))?;
    // Resolve to the repo that hosts the weights for this quantization
    // (GGUF twin for safetensors-only aliases) — must match `download`'s cache key.
    let model_id = crate::models::resolve_weights_repo(model, quant);

    println!();
    println!("{}", style("RuvLLM Inference Server").bold().cyan());
    println!();
    println!("  {} {}", "Model:".dimmed(), model_id);
    println!("  {} {}", "Quantization:".dimmed(), quant);
    println!("  {} {}", "Max Concurrent:".dimmed(), max_concurrent);
    println!("  {} {}", "Max Context:".dimmed(), max_context);
    println!();

    // Initialize backend
    println!("{}", "Loading model...".yellow());

    let mut backend = ruvllm::create_backend();
    let config = ruvllm::ModelConfig {
        architecture: detect_architecture(&model_id),
        quantization: Some(map_quantization(quant)),
        max_sequence_length: max_context,
        ..Default::default()
    };

    // The downloaded GGUF for this quantization, else the id or path given.
    let load_result = crate::models::local_model_source(cache_dir, &model_id, quant)
        .map_err(|e| format!("{e:#}"))
        .and_then(|source| {
            backend
                .load_model(&source, config)
                .map_err(|e| e.to_string())
        });

    // A load that leaves no model or no tokenizer cannot answer a request.
    let load_error = match load_result {
        Err(e) => Some(e),
        Ok(()) if !backend.is_model_loaded() => {
            Some("the backend reported success but no model is loaded".to_string())
        }
        Ok(()) if backend.tokenizer().is_none() => Some(
            "the model loaded but no tokenizer did (no tokenizer.json next to the weights, \
             and none could be built from the GGUF metadata)"
                .to_string(),
        ),
        Ok(()) => None,
    };

    let backend: Option<Arc<dyn LlmBackend>> = match load_error {
        None => {
            if let Some(info) = backend.model_info() {
                println!(
                    "{} Loaded {} ({:.1}B params, {} memory)",
                    style("Success!").green().bold(),
                    info.name,
                    info.num_parameters as f64 / 1e9,
                    bytesize::ByteSize(info.memory_usage as u64)
                );
            } else {
                println!("{} Model loaded", style("Success!").green().bold());
            }
            Some(Arc::from(backend))
        }
        Some(e) if allow_mock => {
            println!(
                "{} Model loading failed: {}. Running in MOCK MODE (--allow-mock).",
                style("Warning:").yellow().bold(),
                e
            );
            println!(
                "{} Every completion will be placeholder text, not model output. Responses \
                 carry `x-ruvllm-mode: mock` and `\"system_fingerprint\": \"{}\"`, and \
                 /health reports \"mode\": \"mock\".",
                style("Warning:").yellow().bold(),
                MOCK_FINGERPRINT
            );
            None
        }
        Some(e) => {
            // `RUVLLM_STRICT=0` used to opt into mock mode; say that it no longer does.
            let legacy_hint = if std::env::var_os("RUVLLM_STRICT").is_some() && !strict {
                " (RUVLLM_STRICT=0 no longer enables mock mode.)"
            } else {
                ""
            };
            anyhow::bail!(
                "model {} failed to load: {}. Refusing to serve placeholder responses; pass \
                 --allow-mock (RUVLLM_ALLOW_MOCK=1) for labeled mock mode.{}",
                model_id,
                e,
                legacy_hint
            );
        }
    };

    let chat_template = select_chat_template(backend.as_deref(), &model_id);
    println!("  {} {:?}", "Chat Template:".dimmed(), chat_template);

    // Create server state
    let mock_mode = backend.is_none();
    let state = Arc::new(RwLock::new(ServerState {
        model_id: model_id.clone(),
        backend,
        allow_mock,
        chat_template,
        request_count: 0,
        total_tokens: 0,
        start_time: Instant::now(),
    }));

    let app = build_router(state);

    // Start server
    let addr = format!("{}:{}", host, port)
        .parse::<SocketAddr>()
        .context("Invalid address")?;

    println!();
    if mock_mode {
        println!(
            "{}",
            style("Server ready (MOCK MODE: placeholder responses, not model output)")
                .bold()
                .yellow()
        );
    } else {
        println!("{}", style("Server ready!").bold().green());
    }
    println!();
    println!("  {} http://{}/v1/chat/completions", "API:".cyan(), addr);
    println!("  {} http://{}/health", "Health:".cyan(), addr);
    println!("  {} http://{}/metrics", "Metrics:".cyan(), addr);
    println!();
    println!("{}", "Example curl:".dimmed());
    println!(
        r#"  curl http://{}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{{"model": "{}", "messages": [{{"role": "user", "content": "Hello!"}}]}}'"#,
        addr, model_id
    );
    println!();
    println!("Press Ctrl+C to stop the server.");
    println!();

    // Set up graceful shutdown
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server error")?;

    println!();
    println!("{}", "Server stopped.".dimmed());

    Ok(())
}

/// Routes, the `x-ruvllm-mode` label, CORS and tracing.
fn build_router(state: SharedState) -> Router {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        // Health and metrics
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        .route("/", get(root))
        // Label every response with whether a real model produced it.
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            label_mode,
        ))
        // State and middleware
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
}

/// The template the backend read from the model file; else Qwen's for a
/// Qwen-architecture model; else a guess from the last component of the
/// model id (a parent directory such as `/home/phil/` must not pick Phi).
pub(crate) fn select_chat_template(
    backend: Option<&dyn LlmBackend>,
    model_id: &str,
) -> ChatTemplate {
    if let Some(backend) = backend {
        if let Some(template) = backend.chat_template() {
            return template;
        }
        if backend.model_info().map(|info| info.architecture) == Some(ModelArchitecture::Qwen) {
            return ChatTemplate::Qwen;
        }
    }
    let name = std::path::Path::new(model_id)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(model_id);
    ChatTemplate::detect_from_model_id(name)
}

/// OpenAI-compatible chat completion request
#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    /// Accepted for compatibility; responses name the served model.
    #[serde(default)]
    model: Option<String>,
    messages: Vec<RequestMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    /// Newer OpenAI name for `max_tokens`; wins when both are set.
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    stop: Option<StopSequences>,
}

const DEFAULT_MAX_TOKENS: usize = 512;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_P: f32 = 0.9;

/// OpenAI accepts `stop` as one string or a list of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopSequences::One(s) => vec![s],
            StopSequences::Many(v) => v,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct StreamOptions {
    /// Send a final chunk with `usage` (and no choices) before `[DONE]`.
    #[serde(default)]
    include_usage: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// A request message. `content` is a string, an array of content parts (what
/// current OpenAI SDKs send), or null (an assistant turn without text).
#[derive(Debug, Deserialize)]
struct RequestMessage {
    role: String,
    #[serde(default)]
    content: Option<MessageContent>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

impl RequestMessage {
    /// The message text: text parts joined by newlines. Non-text parts are
    /// rejected; null content is accepted only for an assistant turn.
    fn text(&self) -> Result<String, ApiError> {
        match &self.content {
            Some(MessageContent::Text(text)) => Ok(text.clone()),
            Some(MessageContent::Parts(parts)) => parts
                .iter()
                .map(|part| match (part.kind.as_str(), &part.text) {
                    ("text", Some(text)) => Ok(text.as_str()),
                    ("text", None) => Err(ApiError::invalid_request(
                        "a content part of type 'text' needs a 'text' field",
                    )),
                    (other, _) => Err(ApiError::invalid_request(format!(
                        "unsupported content part type '{}' (only 'text' is supported)",
                        other
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|texts| texts.join("\n")),
            None if self.role == "assistant" => Ok(String::new()),
            None => Err(ApiError::invalid_request(format!(
                "a '{}' message needs content",
                self.role
            ))),
        }
    }
}

/// OpenAI-compatible chat completion response
#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_fingerprint: Option<String>,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl Usage {
    fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// OpenAI-compatible streaming chunk response
#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_fingerprint: Option<String>,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

/// An OpenAI-style error: `{"error": {"message", "type", "param", "code"}}`.
#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    kind: &'static str,
    code: Option<&'static str>,
    message: String,
}

impl ApiError {
    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: None,
            message: message.into(),
        }
    }

    fn server(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            code: None,
            message: message.into(),
        }
    }

    fn body(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": null,
                "code": self.code,
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

/// A validated request, ready for the backend.
struct Job {
    prompt: String,
    params: GenerateParams,
    stream: bool,
    include_usage: bool,
}

/// Where a response id, timestamp and model name come from.
struct ResponseMeta {
    id: String,
    created: u64,
    model: String,
}

impl ResponseMeta {
    fn new(model: String) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            created: chrono::Utc::now().timestamp() as u64,
            model,
        }
    }

    fn chunk(
        &self,
        delta: Delta,
        finish_reason: Option<&str>,
        fingerprint: Option<&str>,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: fingerprint.map(str::to_string),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason: finish_reason.map(str::to_string),
            }],
            usage: None,
        }
    }

    /// The `stream_options.include_usage` chunk: usage and no choices.
    fn usage_chunk(&self, usage: Usage, fingerprint: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: fingerprint.map(str::to_string),
            choices: Vec::new(),
            usage: Some(usage),
        }
    }

    fn response(
        &self,
        content: String,
        finish_reason: &str,
        usage: Usage,
        fingerprint: Option<&str>,
    ) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: self.id.clone(),
            object: "chat.completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: fingerprint.map(str::to_string),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: finish_reason.to_string(),
            }],
            usage,
        }
    }
}

fn sse_json<T: Serialize>(value: &T) -> Event {
    Event::default().data(serde_json::to_string(value).unwrap_or_default())
}

/// Chat completions endpoint - handles both streaming and non-streaming
async fn chat_completions(
    State(state): State<SharedState>,
    request: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    // Malformed JSON, a missing field or a wrong type is an OpenAI-shaped 400,
    // not axum's plain-text 415/422.
    let request = match request {
        Ok(Json(request)) => request,
        Err(rejection) => return ApiError::invalid_request(rejection.body_text()).into_response(),
    };
    let (model_id, backend, allow_mock, template) = {
        let mut state_lock = state.write().await;
        state_lock.request_count += 1;
        (
            state_lock.model_id.clone(),
            state_lock.backend.clone(),
            state_lock.allow_mock,
            state_lock.chat_template.clone(),
        )
    };

    let job = match prepare_job(&template, request) {
        Ok(job) => job,
        Err(e) => return e.into_response(),
    };

    let Some(backend) = backend else {
        if allow_mock {
            return mock_completion(&model_id, &job);
        }
        // Unreachable from `run`, which exits when the model fails to load.
        return ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "server_error",
            code: Some("model_not_loaded"),
            message: format!("model {} is not loaded", model_id),
        }
        .into_response();
    };

    if let Err(e) = check_prompt_fits(backend.as_ref(), &job.prompt) {
        return e.into_response();
    }

    let meta = ResponseMeta::new(model_id);
    if job.stream {
        stream_completion(state, backend, meta, job).into_response()
    } else {
        match complete(state, backend, meta, job).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => e.into_response(),
        }
    }
}

/// Validate the request and render its messages with the model's template.
fn prepare_job(template: &ChatTemplate, request: ChatCompletionRequest) -> Result<Job, ApiError> {
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens == 0 {
        return Err(ApiError::invalid_request("max_tokens must be at least 1"));
    }
    let prompt = build_prompt(template, &request.messages)?;
    Ok(Job {
        prompt,
        params: GenerateParams {
            max_tokens,
            temperature: request.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            top_p: request.top_p.unwrap_or(DEFAULT_TOP_P),
            stop_sequences: request
                .stop
                .map(StopSequences::into_vec)
                .unwrap_or_default()
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect(),
            ..Default::default()
        },
        stream: request.stream,
        include_usage: request
            .stream_options
            .map(|o| o.include_usage)
            .unwrap_or(false),
    })
}

/// Render chat messages as the model's prompt. Unknown roles are rejected
/// rather than rewritten.
fn build_prompt(template: &ChatTemplate, messages: &[RequestMessage]) -> Result<String, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::invalid_request(
            "messages must contain at least one message",
        ));
    }
    let messages = messages
        .iter()
        .map(|msg| {
            let role = match msg.role.as_str() {
                "system" | "developer" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => {
                    return Err(ApiError::invalid_request(format!(
                        "unsupported message role '{}' (supported: system, developer, user, \
                         assistant)",
                        other
                    )))
                }
            };
            Ok(ruvllm::ChatMessage::new(role, msg.text()?))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(template.format(&messages))
}

/// Reject, before generation starts, a prompt that leaves no room in the
/// model's context window. Returns the prompt's token count.
fn check_prompt_fits(backend: &dyn LlmBackend, prompt: &str) -> Result<usize, ApiError> {
    let tokenizer = backend
        .tokenizer()
        .ok_or_else(|| ApiError::server("the loaded model has no tokenizer"))?;
    let prompt_tokens = tokenizer
        .encode(prompt)
        .map_err(|e| ApiError::server(format!("failed to tokenize the prompt: {}", e)))?
        .len();
    let context = backend
        .model_info()
        .map(|info| info.max_context_length)
        .filter(|&n| n > 0);
    if let Some(context) = context {
        if prompt_tokens >= context {
            return Err(ApiError {
                code: Some("context_length_exceeded"),
                ..ApiError::invalid_request(format!(
                    "the prompt is {} tokens; this model's context window is {} tokens",
                    prompt_tokens, context
                ))
            });
        }
    }
    Ok(prompt_tokens)
}

/// OpenAI `finish_reason` for a backend stop cause; `None` for a failure.
/// `Cancelled` only happens after the client went away, so nobody sees it.
fn openai_finish_reason(reason: FinishReason) -> Option<&'static str> {
    match reason {
        FinishReason::Length => Some("length"),
        FinishReason::Stop | FinishReason::EndOfSequence | FinishReason::Cancelled => Some("stop"),
        FinishReason::Error => None,
    }
}

/// Sets its flag when dropped: a non-streaming handler holds one while it
/// waits, so a dropped (client disconnected) request cancels its generation.
#[derive(Default)]
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Non-streaming chat completion
async fn complete(
    state: SharedState,
    backend: Arc<dyn LlmBackend>,
    meta: ResponseMeta,
    job: Job,
) -> Result<ChatCompletionResponse, ApiError> {
    let start = Instant::now();
    let Job { prompt, params, .. } = job;
    // If the client goes away, this future is dropped and the guard stops the
    // generation, which would otherwise hold the generation lock to the end.
    let cancel = CancelOnDrop::default();
    let cancelled = Arc::clone(&cancel.0);
    // Generation blocks; keep it off the async workers and outside the state lock.
    let output = tokio::task::spawn_blocking(move || {
        backend.generate_detailed(&prompt, params, &mut |_| !cancelled.load(Ordering::Relaxed))
    })
    .await
    .map_err(|e| ApiError::server(format!("generation task failed: {}", e)))?
    .map_err(|e| ApiError::server(format!("generation failed: {}", e)))?;

    let finish_reason = openai_finish_reason(output.finish_reason)
        .ok_or_else(|| ApiError::server("generation ended with an error"))?;
    let usage = Usage::new(output.prompt_tokens, output.completion_tokens);
    state.write().await.total_tokens += usage.total_tokens as u64;

    tracing::info!(
        "Chat completion: {} prompt + {} completion tokens in {:.2}ms (finish_reason={})",
        usage.prompt_tokens,
        usage.completion_tokens,
        start.elapsed().as_secs_f64() * 1000.0,
        finish_reason
    );

    Ok(meta.response(output.text, finish_reason, usage, None))
}

/// SSE streaming chat completion. Deltas are sent as the backend produces
/// them; the last chunk carries the backend's finish reason. A failure sends
/// an `{"error": ...}` event instead of a finish chunk.
fn stream_completion(
    state: SharedState,
    backend: Arc<dyn LlmBackend>,
    meta: ResponseMeta,
    job: Job,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let Job {
        prompt,
        params,
        include_usage,
        ..
    } = job;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    // A failed send means the client went away: returning false cancels.
    let generation = tokio::task::spawn_blocking(move || {
        backend.generate_detailed(&prompt, params, &mut |token| {
            tx.send(token.text.clone()).is_ok()
        })
    });

    let stream = async_stream::stream! {
        let role = Delta { role: Some("assistant".to_string()), content: None };
        yield Ok(sse_json(&meta.chunk(role, None, None)));

        // The sender lives in the generation task, so this ends when it returns.
        while let Some(text) = rx.recv().await {
            let delta = Delta { role: None, content: Some(text) };
            yield Ok(sse_json(&meta.chunk(delta, None, None)));
        }

        let failure = match generation.await {
            Ok(Ok(output)) => match openai_finish_reason(output.finish_reason) {
                Some(finish_reason) => {
                    let usage = Usage::new(output.prompt_tokens, output.completion_tokens);
                    state.write().await.total_tokens += usage.total_tokens as u64;
                    yield Ok(sse_json(&meta.chunk(Delta::default(), Some(finish_reason), None)));
                    if include_usage {
                        yield Ok(sse_json(&meta.usage_chunk(usage, None)));
                    }
                    None
                }
                None => Some(ApiError::server("generation ended with an error")),
            },
            Ok(Err(e)) => Some(ApiError::server(format!("generation failed: {}", e))),
            Err(e) => Some(ApiError::server(format!("generation task failed: {}", e))),
        };
        if let Some(error) = failure {
            tracing::error!("Stream error: {}", error.message);
            yield Ok(sse_json(&error.body()));
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// The one placeholder completion text. It names itself as placeholder text.
fn mock_text(model_id: &str) -> String {
    format!(
        "[ruvllm mock mode] Model '{}' is not loaded. This is placeholder text, not model output.",
        model_id
    )
}

/// A labeled placeholder completion (`--allow-mock` only): fixed text, zero
/// usage, `system_fingerprint: "ruvllm-mock"`.
fn mock_completion(model_id: &str, job: &Job) -> Response {
    let meta = ResponseMeta::new(model_id.to_string());
    let text = mock_text(model_id);
    let fingerprint = Some(MOCK_FINGERPRINT);
    if !job.stream {
        return Json(meta.response(text, "stop", Usage::default(), fingerprint)).into_response();
    }

    let mut events = vec![
        sse_json(&meta.chunk(
            Delta {
                role: Some("assistant".to_string()),
                content: None,
            },
            None,
            fingerprint,
        )),
        sse_json(&meta.chunk(
            Delta {
                role: None,
                content: Some(text),
            },
            None,
            fingerprint,
        )),
        sse_json(&meta.chunk(Delta::default(), Some("stop"), fingerprint)),
    ];
    if job.include_usage {
        events.push(sse_json(&meta.usage_chunk(Usage::default(), fingerprint)));
    }
    events.push(Event::default().data("[DONE]"));
    Sse::new(stream::iter(events.into_iter().map(Ok::<_, Infallible>))).into_response()
}

/// List available models
async fn list_models(State(state): State<SharedState>) -> impl IntoResponse {
    let state_lock = state.read().await;

    let models = serde_json::json!({
        "object": "list",
        "data": [{
            "id": state_lock.model_id,
            "object": "model",
            "owned_by": "ruvllm",
            "permission": []
        }]
    });

    Json(models)
}

/// `"model"` when a real model is loaded, `"mock"` when completions are
/// placeholder text.
fn mode_of(state: &ServerState) -> &'static str {
    if state
        .backend
        .as_ref()
        .map(|b| b.is_model_loaded())
        .unwrap_or(false)
    {
        "model"
    } else {
        "mock"
    }
}

/// Middleware: add `x-ruvllm-mode: model|mock` to every response, so a client
/// can tell placeholder completions from model output without parsing text.
async fn label_mode(
    State(state): State<SharedState>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let mode = mode_of(&*state.read().await);
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        axum::http::HeaderName::from_static("x-ruvllm-mode"),
        axum::http::HeaderValue::from_static(mode),
    );
    response
}

/// Health check endpoint
async fn health_check(State(state): State<SharedState>) -> impl IntoResponse {
    let state_lock = state.read().await;

    let status = if state_lock
        .backend
        .as_ref()
        .map(|b| b.is_model_loaded())
        .unwrap_or(false)
    {
        "healthy"
    } else {
        "degraded"
    };

    let health = serde_json::json!({
        "status": status,
        "mode": mode_of(&state_lock),
        "model": state_lock.model_id,
        "uptime_seconds": state_lock.start_time.elapsed().as_secs()
    });

    Json(health)
}

/// Metrics endpoint
async fn metrics(State(state): State<SharedState>) -> impl IntoResponse {
    let state_lock = state.read().await;
    let uptime = state_lock.start_time.elapsed();

    let metrics = serde_json::json!({
        "model": state_lock.model_id,
        "requests_total": state_lock.request_count,
        "tokens_total": state_lock.total_tokens,
        "uptime_seconds": uptime.as_secs(),
        "requests_per_second": if uptime.as_secs() > 0 {
            state_lock.request_count as f64 / uptime.as_secs() as f64
        } else {
            0.0
        },
        "tokens_per_second": if uptime.as_secs() > 0 {
            state_lock.total_tokens as f64 / uptime.as_secs() as f64
        } else {
            0.0
        }
    });

    Json(metrics)
}

/// Root endpoint
async fn root() -> impl IntoResponse {
    let info = serde_json::json!({
        "name": "RuvLLM Inference Server",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "metrics": "/metrics"
        }
    });

    Json(info)
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    println!();
    println!("{}", "Shutting down...".yellow());
}

/// Detect model architecture from model ID
fn detect_architecture(model_id: &str) -> ruvllm::ModelArchitecture {
    let lower = model_id.to_lowercase();
    if lower.contains("mistral") {
        ruvllm::ModelArchitecture::Mistral
    } else if lower.contains("llama") {
        ruvllm::ModelArchitecture::Llama
    } else if lower.contains("phi") {
        ruvllm::ModelArchitecture::Phi
    } else if lower.contains("qwen") {
        ruvllm::ModelArchitecture::Qwen
    } else if lower.contains("gemma") {
        ruvllm::ModelArchitecture::Gemma
    } else {
        ruvllm::ModelArchitecture::Llama // Default
    }
}

/// Map our quantization preset to ruvllm quantization
fn map_quantization(quant: QuantPreset) -> ruvllm::Quantization {
    match quant {
        QuantPreset::Q4K => ruvllm::Quantization::Q4K,
        QuantPreset::Q8 => ruvllm::Quantization::Q8,
        QuantPreset::F16 => ruvllm::Quantization::F16,
        QuantPreset::None => ruvllm::Quantization::None,
    }
}

#[cfg(test)]
#[path = "serve_tests.rs"]
mod tests;
