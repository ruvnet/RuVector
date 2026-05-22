//! ruvllm-embedder: standalone HTTP embedder service for obsidian-brain and other clients.
//!
//! Exposes the EmbeddingEngine from mcp-brain-server over a minimal HTTP API on port 9877.
//!
//! Build:
//!   cargo build --release -p mcp-brain-server --bin ruvllm-embedder
//!
//! Endpoints:
//!   POST /embed   {"texts": ["..."]}  → {"vectors": [[...]], "engine": "...", "corpus_size": N}
//!   GET  /health                      → {"status": "ok", "engine": "...", "embed_dim": N, ...}

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mcp_brain_server::embeddings::EmbeddingEngine;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<EmbeddingEngine>>,
}

#[derive(Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Serialize)]
struct EmbedResponse {
    vectors: Vec<Vec<f32>>,
    engine: String,
    corpus_size: usize,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    engine: String,
    embed_dim: usize,
    corpus_size: usize,
    rlm_active: bool,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

async fn embed(State(state): State<AppState>, Json(req): Json<EmbedRequest>) -> impl IntoResponse {
    if req.texts.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::to_value(ErrorResponse {
                    error: "texts array must not be empty".into(),
                })
                .unwrap(),
            ),
        );
    }

    let engine = state.engine.lock().unwrap();
    let vectors: Vec<Vec<f32>> = req.texts.iter().map(|t| engine.embed(t)).collect();
    let response = EmbedResponse {
        engine: engine.engine_name().to_owned(),
        corpus_size: engine.corpus_size(),
        vectors,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap()),
    )
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let engine = state.engine.lock().unwrap();
    let response = HealthResponse {
        status: "ok",
        engine: engine.engine_name().to_owned(),
        embed_dim: engine.dim(),
        corpus_size: engine.corpus_size(),
        rlm_active: engine.is_rlm_active(),
    };
    Json(response)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("EMBEDDER_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(9877);

    let state = AppState {
        engine: Arc::new(Mutex::new(EmbeddingEngine::new())),
    };

    let app = Router::new()
        .route("/embed", post(embed))
        .route("/health", get(health))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("ruvllm-embedder listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
