//! `ruview-ruvllm-h10` — gRPC + HTTP LLM serving on cognitum-cluster-3 Hailo-10H.
//!
//! Boot order:
//! 1. Parse Config from env.
//! 2. Spawn hailo-ollama subprocess via bridge (waits for ready).
//! 3. Pull the configured model if not present.
//! 4. Start gRPC LlmService on GRPC_LISTEN (:50058).
//! 5. Start HTTP proxy on HTTP_LISTEN (:8880).

mod bridge;

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_stream::try_stream;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use bridge::HailoOllamaBridge;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Semaphore};
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tracing_subscriber::EnvFilter;

// ──────────────────────────────────────────────── generated proto

pub mod llm_proto {
    tonic::include_proto!("ruview.llm.v1");
}
use llm_proto::llm_service_server::{LlmService, LlmServiceServer};
use llm_proto::{
    GenerateChunk, GenerateRequest, HealthRequest, HealthResponse, PullRequest, PullResponse,
};

// ──────────────────────────────────────────────── error / result

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("bridge: {0}")]
    Bridge(String),
    #[error("config: {0}")]
    Config(String),
}
pub type Result<T> = std::result::Result<T, Error>;

// ──────────────────────────────────────────────── rate limiter

/// Token-bucket rate limiter for the /generate HTTP endpoint.
///
/// Tokens refill at `tokens_per_min / 60` per second, capped at
/// `burst`. Each /generate call consumes 1 token. If empty → 429.
/// Implemented with atomics so it is `Send + Sync` without a Mutex.
struct RateLimiter {
    /// Tokens currently available (scaled ×1000 to avoid float).
    tokens_millis: AtomicU64,
    /// Timestamp of last refill (UNIX ms).
    last_refill_ms: AtomicU64,
    /// Refill rate: tokens per millisecond (×1000, i.e. tokens_per_min / 60_000).
    refill_rate_per_ms_millis: u64,
    /// Maximum burst (×1000).
    burst_millis: u64,
    /// Concurrency semaphore — hailo-ollama is single-threaded.
    semaphore: Arc<Semaphore>,
}

impl RateLimiter {
    fn new(tokens_per_min: u64, burst: u64, max_concurrent: usize) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            tokens_millis: AtomicU64::new(burst * 1000),
            last_refill_ms: AtomicU64::new(now_ms),
            refill_rate_per_ms_millis: tokens_per_min.max(1) * 1000 / 60_000,
            burst_millis: burst * 1000,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    /// Returns `true` if a token was acquired (request allowed).
    fn try_acquire(&self) -> bool {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let last = self.last_refill_ms.load(Ordering::Relaxed);
        let elapsed = now_ms.saturating_sub(last);
        let added = elapsed * self.refill_rate_per_ms_millis;

        if added > 0 {
            self.last_refill_ms.store(now_ms, Ordering::Relaxed);
            let cur = self.tokens_millis.load(Ordering::Relaxed);
            let new = (cur + added).min(self.burst_millis);
            self.tokens_millis.store(new, Ordering::Relaxed);
        }

        let cur = self.tokens_millis.load(Ordering::Relaxed);
        if cur >= 1000 {
            self.tokens_millis.fetch_sub(1000, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

// ──────────────────────────────────────────────── config

struct Config {
    grpc_listen:       SocketAddr,
    http_listen:       SocketAddr,
    model:             String,
    rate_limit_rpm:    u64,
    rate_limit_burst:  u64,
    max_concurrent:    usize,
}

impl Config {
    fn from_env() -> Result<Self> {
        let parse_u64 = |var: &str, default: u64| -> Result<u64> {
            std::env::var(var)
                .ok()
                .map(|v| v.parse::<u64>().map_err(|e| Error::Config(format!("{var}: {e}"))))
                .transpose()
                .map(|o| o.unwrap_or(default))
        };
        Ok(Self {
            grpc_listen: std::env::var("RUVIEW_RUVLLM_GRPC_LISTEN")
                .unwrap_or_else(|_| "0.0.0.0:50058".into())
                .parse()
                .map_err(|e| Error::Config(format!("GRPC_LISTEN: {e}")))?,
            http_listen: std::env::var("RUVIEW_RUVLLM_HTTP_LISTEN")
                .unwrap_or_else(|_| "0.0.0.0:8880".into())
                .parse()
                .map_err(|e| Error::Config(format!("HTTP_LISTEN: {e}")))?,
            model:            std::env::var("RUVIEW_RUVLLM_MODEL")
                .unwrap_or_else(|_| "llama3.2:1b".into()),
            rate_limit_rpm:   parse_u64("RUVIEW_RUVLLM_RATE_LIMIT_RPM", 20)?,
            rate_limit_burst: parse_u64("RUVIEW_RUVLLM_RATE_LIMIT_BURST", 5)?,
            max_concurrent:   parse_u64("RUVIEW_RUVLLM_MAX_CONCURRENT", 1)? as usize,
        })
    }
}

// ──────────────────────────────────────────────── gRPC service

struct LlmSvc {
    bridge: Arc<HailoOllamaBridge>,
    started: Instant,
}

#[tonic::async_trait]
impl LlmService for LlmSvc {
    type GenerateStream = std::pin::Pin<
        Box<dyn futures_core::Stream<Item = std::result::Result<GenerateChunk, Status>> + Send>,
    >;

    async fn generate(
        &self,
        req: Request<GenerateRequest>,
    ) -> std::result::Result<Response<Self::GenerateStream>, Status> {
        let r = req.into_inner();
        let model = if r.model.is_empty() {
            self.bridge.model().to_string()
        } else {
            r.model
        };
        let prompt    = r.prompt;
        let max_toks  = if r.max_tokens <= 0 { 256 } else { r.max_tokens };
        let temp      = if r.temperature == 0.0 { 0.4 } else { r.temperature };

        let (tx, mut rx) = mpsc::channel::<(String, bool, i64)>(512);
        let bridge = Arc::clone(&self.bridge);

        // Spawn bridge call in background to avoid blocking the gRPC task.
        tokio::spawn(async move {
            if let Err(e) = bridge.generate_stream(&prompt, max_toks, temp, tx).await {
                tracing::error!(error = %e, %model, "generate_stream error");
            }
        });

        let stream = try_stream! {
            while let Some((token, done, latency_us)) = rx.recv().await {
                yield GenerateChunk { token, done, latency_us };
                if done { break; }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    async fn pull_model(
        &self,
        req: Request<PullRequest>,
    ) -> std::result::Result<Response<PullResponse>, Status> {
        let model = req.into_inner().model;
        match self.bridge.pull(&model).await {
            Ok(()) => Ok(Response::new(PullResponse { ok: true, message: "pulled".into() })),
            Err(e) => Ok(Response::new(PullResponse {
                ok: false,
                message: e.to_string(),
            })),
        }
    }

    async fn health(
        &self,
        _req: Request<HealthRequest>,
    ) -> std::result::Result<Response<HealthResponse>, Status> {
        let (hailo_ok, backend) = self.bridge.health_check().await;
        let tok_per_sec = self
            .bridge
            .stats
            .tok_per_sec_window(self.started.elapsed());
        Ok(Response::new(HealthResponse {
            model: self.bridge.model().to_string(),
            backend,
            tok_per_sec,
            hailo_ok,
            firmware_ver: "5.1.1".into(),
        }))
    }
}

// ──────────────────────────────────────────────── HTTP proxy

#[derive(Clone)]
struct HttpState {
    bridge:  Arc<HailoOllamaBridge>,
    started: Instant,
    rl:      Arc<RateLimiter>,
}

#[derive(Serialize)]
struct HealthJson {
    model:        String,
    backend:      String,
    tok_per_sec:  f32,
    hailo_ok:     bool,
    firmware_ver: String,
}

#[derive(Deserialize)]
struct GenerateBodyJson {
    prompt:      String,
    #[serde(default = "default_max_tokens")]
    max_tokens:  i32,
    #[serde(default = "default_temperature")]
    temperature: f32,
}
fn default_max_tokens() -> i32 { 256 }
fn default_temperature() -> f32 { 0.4 }

async fn http_health(State(s): State<HttpState>) -> impl IntoResponse {
    let (hailo_ok, backend) = s.bridge.health_check().await;
    let tok_per_sec = s.bridge.stats.tok_per_sec_window(s.started.elapsed());
    Json(HealthJson {
        model: s.bridge.model().to_string(),
        backend,
        tok_per_sec,
        hailo_ok,
        firmware_ver: "5.1.1".into(),
    })
}

async fn http_generate(
    State(s): State<HttpState>,
    Json(body): Json<GenerateBodyJson>,
) -> impl IntoResponse {
    // Rate-limit check (token bucket).
    if !s.rl.try_acquire() {
        tracing::warn!("rate limit exceeded — returning 429");
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({"error": "rate limit exceeded", "retry_after_s": 3})),
        ).into_response();
    }
    // Concurrency semaphore — try immediately; don't queue forever.
    let _permit = match s.rl.semaphore.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            tracing::warn!("max concurrent requests reached — returning 429");
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "server busy", "retry_after_s": 10})),
            ).into_response();
        }
    };

    let (tx, mut rx) = mpsc::channel::<(String, bool, i64)>(512);
    let bridge = Arc::clone(&s.bridge);
    let prompt  = body.prompt.clone();
    let max_tok = body.max_tokens;
    let temp    = body.temperature;

    tokio::spawn(async move {
        if let Err(e) = bridge.generate_stream(&prompt, max_tok, temp, tx).await {
            tracing::error!(error = %e, "http generate_stream error");
        }
    });

    let mut out = String::new();
    while let Some((token, done, _)) = rx.recv().await {
        out.push_str(&token);
        if done { break; }
    }
    (StatusCode::OK, Json(serde_json::json!({"text": out, "model": s.bridge.model()}))).into_response()
}

// ──────────────────────────────────────────────── main

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let cfg = Config::from_env().map_err(|e| format!("{e}"))?;

    tracing::info!(
        grpc = %cfg.grpc_listen,
        http = %cfg.http_listen,
        model = %cfg.model,
        "ruview-ruvllm-h10 starting"
    );

    // Spawn hailo-ollama + wait for it to be ready.
    let bridge = Arc::new(
        HailoOllamaBridge::spawn(&cfg.model)
            .await
            .map_err(|e| format!("{e}"))?,
    );

    // Pull model if not already present.
    if let Err(e) = bridge.pull(&cfg.model).await {
        tracing::warn!(error = %e, "model pull failed (may already be cached)");
    }

    tracing::info!(model = %cfg.model, "model ready");

    let started = Instant::now();

    // gRPC server.
    let svc = LlmServiceServer::new(LlmSvc {
        bridge: Arc::clone(&bridge),
        started,
    });
    let grpc_addr = cfg.grpc_listen;
    tokio::spawn(async move {
        tracing::info!(addr = %grpc_addr, "gRPC LlmService starting");
        if let Err(e) = Server::builder()
            .add_service(svc)
            .serve(grpc_addr)
            .await
        {
            tracing::error!(error = %e, "gRPC server exited");
        }
    });

    // HTTP proxy with rate limiter.
    let rl = Arc::new(RateLimiter::new(cfg.rate_limit_rpm, cfg.rate_limit_burst, cfg.max_concurrent));
    tracing::info!(rpm = cfg.rate_limit_rpm, burst = cfg.rate_limit_burst,
                   max_concurrent = cfg.max_concurrent, "rate limiter initialised");

    let http_state = HttpState { bridge: Arc::clone(&bridge), started, rl };
    let app = Router::new()
        .route("/health", get(http_health))
        .route("/generate", post(http_generate))
        .with_state(http_state);

    tracing::info!(addr = %cfg.http_listen, "HTTP proxy starting");
    let listener = tokio::net::TcpListener::bind(cfg.http_listen).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_env("RUVIEW_RUVLLM_LOG")
        .or_else(|_| EnvFilter::try_new("info,ruview_ruvllm_h10=info"))
        .expect("tracing filter");
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_writer(std::io::stderr)
        .init();
}
