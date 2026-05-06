//! `ruview-ruvllm-router` — multi-backend LLM router for cognitum cluster.
//!
//! Accepts gRPC `LlmService` requests on `:50060` and HTTP on `:8882`,
//! routing each request to the least-busy healthy backend using the same
//! proto as `ruview-ruvllm-h10`.
//!
//! Config env vars:
//!   RUVIEW_ROUTER_GRPC_LISTEN   — default 0.0.0.0:50060
//!   RUVIEW_ROUTER_HTTP_LISTEN   — default 0.0.0.0:8882
//!   RUVIEW_ROUTER_BACKENDS      — comma-separated "addr:port[:model]" entries
//!                                  e.g. "100.73.75.53:50058:llama3.2:1b,100.77.59.83:50058:llama3.2:1b"
//!   RUVIEW_ROUTER_HEALTH_SEC    — health-check interval (default 30)
//!   RUVIEW_ROUTER_LOG           — tracing filter (default info)

mod pool;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_stream::try_stream;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use pool::{ActiveGuard, Backend, Pool};
use serde::{Deserialize, Serialize};
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tracing_subscriber::EnvFilter;

pub mod llm_proto {
    tonic::include_proto!("ruview.llm.v1");
}
use llm_proto::llm_service_server::{LlmService, LlmServiceServer};
use llm_proto::{
    GenerateChunk, GenerateRequest, HealthRequest, HealthResponse, PullRequest, PullResponse,
};

// ──────────────────────────────────────────────── config

struct Config {
    grpc_listen:  SocketAddr,
    http_listen:  SocketAddr,
    backends:     Vec<(String, String)>, // (addr, model)
    health_sec:   u64,
}

impl Config {
    fn from_env() -> Self {
        let grpc_listen = std::env::var("RUVIEW_ROUTER_GRPC_LISTEN")
            .unwrap_or_else(|_| "0.0.0.0:50060".into())
            .parse()
            .expect("RUVIEW_ROUTER_GRPC_LISTEN must be a SocketAddr");
        let http_listen = std::env::var("RUVIEW_ROUTER_HTTP_LISTEN")
            .unwrap_or_else(|_| "0.0.0.0:8882".into())
            .parse()
            .expect("RUVIEW_ROUTER_HTTP_LISTEN must be a SocketAddr");
        let health_sec = std::env::var("RUVIEW_ROUTER_HEALTH_SEC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30u64);

        let backends_str = std::env::var("RUVIEW_ROUTER_BACKENDS")
            .unwrap_or_default();
        let backends = backends_str
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|spec| {
                // Format: "addr:port" or "addr:port:model"
                // addr may contain dots so we split from end for port
                let parts: Vec<&str> = spec.splitn(3, ':').collect();
                match parts.len() {
                    // "host:port:model" — but host may be IP so rejoin correctly
                    // Actually spec is "ip:port" or "ip:port:model-name" where model may contain ':'
                    _ => {
                        // Find the last ':model' chunk after the second ':'
                        if let Some(idx) = spec.find(':') {
                            let after_first = &spec[idx + 1..];
                            if let Some(idx2) = after_first.find(':') {
                                let addr = &spec[..idx + 1 + idx2];
                                let model = &after_first[idx2 + 1..];
                                (addr.to_string(), model.to_string())
                            } else {
                                (spec.to_string(), "llama3.2:1b".to_string())
                            }
                        } else {
                            (spec.to_string(), "llama3.2:1b".to_string())
                        }
                    }
                }
            })
            .collect();

        Self { grpc_listen, http_listen, backends, health_sec }
    }
}

// ──────────────────────────────────────────────── gRPC router service

struct RouterSvc {
    backends: Arc<Vec<Arc<Backend>>>,
}

#[tonic::async_trait]
impl LlmService for RouterSvc {
    type GenerateStream = std::pin::Pin<
        Box<dyn futures_core::Stream<Item = Result<GenerateChunk, Status>> + Send>,
    >;

    async fn generate(
        &self,
        req: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let backend = Pool::least_busy(&self.backends)
            .ok_or_else(|| Status::unavailable("no healthy LLM backends available"))?;

        let guard = ActiveGuard::new(&backend);
        let mut client = backend.client();
        let inner = req.into_inner();

        let mut upstream = client
            .generate(inner)
            .await
            .map_err(|e| Status::internal(format!("backend {}: {e}", backend.addr)))?
            .into_inner();

        let stream = try_stream! {
            let _guard = guard;
            loop {
                match upstream.message().await? {
                    Some(chunk) => yield chunk,
                    None => break,
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    async fn pull_model(
        &self,
        req: Request<PullRequest>,
    ) -> Result<Response<PullResponse>, Status> {
        let model = req.into_inner().model.clone();
        let mut results: Vec<String> = Vec::new();
        for b in self.backends.iter() {
            match b.client().pull_model(PullRequest { model: model.clone() }).await {
                Ok(r) => {
                    let r = r.into_inner();
                    results.push(format!("{}: {}", b.addr, r.message));
                }
                Err(e) => results.push(format!("{}: ERROR {e}", b.addr)),
            }
        }
        Ok(Response::new(PullResponse { ok: true, message: results.join("; ") }))
    }

    async fn health(
        &self,
        _req: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let healthy = self.backends.iter().filter(|b| b.healthy.load(std::sync::atomic::Ordering::Relaxed)).count();
        let total   = self.backends.len();
        let tok_per_sec: f32 = self.backends.iter()
            .filter_map(|b| {
                if b.healthy.load(std::sync::atomic::Ordering::Relaxed) {
                    Some(b.active.load(std::sync::atomic::Ordering::Relaxed))
                } else {
                    None
                }
            })
            .count() as f32;  // placeholder: count of active backends

        Ok(Response::new(HealthResponse {
            model:        "router".into(),
            backend:      format!("{healthy}/{total} backends healthy"),
            tok_per_sec,
            hailo_ok:     healthy > 0,
            firmware_ver: env!("CARGO_PKG_VERSION").into(),
        }))
    }
}

// ──────────────────────────────────────────────── HTTP handlers

#[derive(Clone)]
struct HttpState {
    backends: Arc<Vec<Arc<Backend>>>,
}

#[derive(Serialize)]
struct BackendStatus {
    addr:     String,
    model:    String,
    healthy:  bool,
    active:   u32,
}

#[derive(Serialize)]
struct RouteHealth {
    backends_healthy: usize,
    backends_total:   usize,
    backends:         Vec<BackendStatus>,
}

#[derive(Deserialize)]
struct GenerateBody {
    prompt:      String,
    #[serde(default = "default_max_tokens")]
    max_tokens:  i32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    model:       String,
}
fn default_max_tokens() -> i32 { 256 }
fn default_temperature() -> f32 { 0.4 }

async fn http_health(State(s): State<HttpState>) -> impl IntoResponse {
    let statuses: Vec<BackendStatus> = s.backends.iter().map(|b| BackendStatus {
        addr:    b.addr.clone(),
        model:   b.model.clone(),
        healthy: b.healthy.load(std::sync::atomic::Ordering::Relaxed),
        active:  b.active.load(std::sync::atomic::Ordering::Relaxed),
    }).collect();
    let healthy = statuses.iter().filter(|b| b.healthy).count();
    Json(RouteHealth {
        backends_healthy: healthy,
        backends_total:   statuses.len(),
        backends:         statuses,
    })
}

async fn http_generate(
    State(s): State<HttpState>,
    Json(body): Json<GenerateBody>,
) -> impl IntoResponse {
    let Some(backend) = Pool::least_busy(&s.backends) else {
        return (StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "no healthy backends"}))).into_response();
    };

    let _guard = ActiveGuard::new(&backend);
    let model = if body.model.is_empty() { backend.model.clone() } else { body.model.clone() };
    let req = GenerateRequest {
        model:       model.clone(),
        prompt:      body.prompt.clone(),
        max_tokens:  body.max_tokens,
        temperature: body.temperature,
    };

    match backend.client().generate(req).await {
        Ok(stream) => {
            let mut text = String::new();
            let mut upstream = stream.into_inner();
            loop {
                match upstream.message().await {
                    Ok(Some(chunk)) => {
                        text.push_str(&chunk.token);
                        if chunk.done { break; }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        tracing::error!(error = %e, "http generate stream error");
                        break;
                    }
                }
            }
            (StatusCode::OK, Json(serde_json::json!({"text": text, "model": model, "backend": backend.addr}))).into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, backend = %backend.addr, "generate rpc failed");
            (StatusCode::BAD_GATEWAY,
             Json(serde_json::json!({"error": e.to_string()}))).into_response()
        }
    }
}

// ──────────────────────────────────────────────── main

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter = EnvFilter::try_from_env("RUVIEW_ROUTER_LOG")
        .or_else(|_| EnvFilter::try_new("info,ruview_ruvllm_router=info"))
        .expect("tracing filter");
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_writer(std::io::stderr)
        .init();

    let cfg = Config::from_env();

    if cfg.backends.is_empty() {
        tracing::warn!(
            "RUVIEW_ROUTER_BACKENDS is empty — router will start but return 503 on all requests. \
             Set e.g. RUVIEW_ROUTER_BACKENDS=100.73.75.53:50058,100.77.59.83:50058"
        );
    }

    tracing::info!(
        grpc        = %cfg.grpc_listen,
        http        = %cfg.http_listen,
        backends    = cfg.backends.len(),
        health_sec  = cfg.health_sec,
        "ruview-ruvllm-router starting"
    );

    let backends = Arc::new(Pool::new(&cfg.backends).await);

    // Initial health sweep.
    for b in backends.iter() { b.check_health().await; }

    // Ongoing health check loop.
    let backends_hc = Arc::clone(&backends);
    let interval = Duration::from_secs(cfg.health_sec);
    tokio::spawn(async move {
        Pool::health_loop(backends_hc, interval).await;
    });

    // gRPC server.
    let svc = LlmServiceServer::new(RouterSvc { backends: Arc::clone(&backends) });
    let grpc_addr = cfg.grpc_listen;
    tokio::spawn(async move {
        tracing::info!(addr = %grpc_addr, "gRPC router starting");
        if let Err(e) = Server::builder().add_service(svc).serve(grpc_addr).await {
            tracing::error!(error = %e, "gRPC server exited");
        }
    });

    // HTTP server.
    let state = HttpState { backends: Arc::clone(&backends) };
    let app = Router::new()
        .route("/health",   get(http_health))
        .route("/generate", post(http_generate))
        .with_state(state);

    tracing::info!(addr = %cfg.http_listen, "HTTP router starting");
    let listener = tokio::net::TcpListener::bind(cfg.http_listen).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
