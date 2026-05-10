//! ruvllm-embedder: local HTTP embedder honoring the original EmbeddingEngine design.
//!
//! Three-phase architecture (from `src/embeddings.rs`):
//! - Phase 1: HashEmbedder base (FNV-1a + bigrams, L2-normalized) — fallback for empty corpus
//! - Phase 2: RlmEmbedder recursive context-aware embeddings (active when corpus ≥ 50)
//! - Phase 3: candle sentence transformer (future — requires `candle` feature)
//!
//! Endpoints:
//! - POST /embed          → query-conditioned embeddings (retrieval-optimized)
//! - POST /embed/storage   → corpus-conditioned embeddings (stable over time)
//! - POST /corpus/add     → add documents to grow the neighbor store
//! - GET  /health         → engine status, corpus size, active phase

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use mcp_brain_server::embeddings::EmbeddingEngine;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::sync::Arc;
use tracing_subscriber::prelude::*;

const EMBED_DIM: usize = 128;

// ── Request / Response ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedResponse {
    vectors: Vec<Vec<f32>>,
    embeddings: Vec<Vec<f32>>,
    engine: String,
    corpus_size: usize,
}

#[derive(Debug, Deserialize)]
struct CorpusAddRequest {
    id: String,
    text: String,
    cluster_id: Option<usize>,
}

#[derive(Debug, Serialize)]
struct CorpusAddResponse {
    ok: bool,
    corpus_size: usize,
    rlm_active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct HealthResponse {
    status: String,
    engine: String,
    embed_dim: usize,
    corpus_size: usize,
    rlm_active: bool,
}

// ── State ──────────────────────────────────────────────────────────────────

struct AppState {
    engine: Mutex<EmbeddingEngine>,
}

// ── Handlers ────────────────────────────────────────────────────────────────

async fn embed_handler(
    State(st): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<serde_json::Value>)> {
    let engine = st.engine.lock().unwrap();
    let mut vectors = Vec::with_capacity(req.texts.len());

    for text in &req.texts {
        let v = engine.embed(text);
        if v.len() != EMBED_DIM {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("unexpected embedding dimension: expected {EMBED_DIM}, got {}", v.len())
                })),
            ));
        }
        vectors.push(v);
    }

    let engine_name = engine.engine_name().to_string();
    let corpus_size = engine.corpus_size();

    Ok(Json(EmbedResponse {
        embeddings: vectors.clone(),
        vectors,
        engine: engine_name,
        corpus_size,
    }))
}

async fn embed_storage_handler(
    State(st): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<serde_json::Value>)> {
    let engine = st.engine.lock().unwrap();
    let mut vectors = Vec::with_capacity(req.texts.len());

    for text in &req.texts {
        let v = engine.embed_for_storage(text);
        if v.len() != EMBED_DIM {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("unexpected embedding dimension: expected {EMBED_DIM}, got {}", v.len())
                })),
            ));
        }
        vectors.push(v);
    }

    let engine_name = engine.engine_name().to_string();
    let corpus_size = engine.corpus_size();

    Ok(Json(EmbedResponse {
        embeddings: vectors.clone(),
        vectors,
        engine: engine_name,
        corpus_size,
    }))
}

async fn corpus_add_handler(
    State(st): State<Arc<AppState>>,
    Json(req): Json<CorpusAddRequest>,
) -> Json<CorpusAddResponse> {
    let mut engine = st.engine.lock().unwrap();
    let emb = engine.embed(&req.text); // use query embedding as base for corpus
    engine.add_to_corpus(&req.id, emb, req.cluster_id);

    Json(CorpusAddResponse {
        ok: true,
        corpus_size: engine.corpus_size(),
        rlm_active: engine.is_rlm_active(),
    })
}

async fn health(State(st): State<Arc<AppState>>) -> Json<HealthResponse> {
    let engine = st.engine.lock().unwrap();
    Json(HealthResponse {
        status: "ok".to_string(),
        engine: engine.engine_name().to_string(),
        embed_dim: EMBED_DIM,
        corpus_size: engine.corpus_size(),
        rlm_active: engine.is_rlm_active(),
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    fn test_state() -> Arc<AppState> {
        Arc::new(AppState {
            engine: Mutex::new(EmbeddingEngine::new()),
        })
    }

    #[tokio::test]
    async fn test_embed_endpoint() {
        let state = test_state();
        let app = Router::new()
            .route("/embed", post(embed_handler))
            .with_state(state);

        let req_body = serde_json::json!({"texts": ["hello", "world"]});
        let req = Request::builder()
            .method(Method::POST)
            .uri("/embed")
            .header("content-type", "application/json")
            .body(Body::from(req_body.to_string()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let resp: EmbedResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp.vectors.len(), 2);
        assert_eq!(resp.vectors[0].len(), EMBED_DIM);
        assert_eq!(resp.embeddings.len(), 2);
        assert_eq!(resp.embeddings[0], resp.vectors[0]);
        assert!(resp.corpus_size < 50); // HashEmbedder phase
    }

    #[tokio::test]
    async fn test_embed_storage_endpoint() {
        let state = test_state();
        let app = Router::new()
            .route("/embed/storage", post(embed_storage_handler))
            .with_state(state);

        let req_body = serde_json::json!({"texts": ["hello"]});
        let req = Request::builder()
            .method(Method::POST)
            .uri("/embed/storage")
            .header("content-type", "application/json")
            .body(Body::from(req_body.to_string()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let resp: EmbedResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp.vectors.len(), 1);
        assert_eq!(resp.vectors[0].len(), EMBED_DIM);
    }

    #[tokio::test]
    async fn test_corpus_add_activates_rlm() {
        let state = test_state();
        let app = Router::new()
            .route("/corpus/add", post(corpus_add_handler))
            .route("/health", get(health))
            .with_state(state);

        // Add enough documents to activate RLM
        for i in 0..55 {
            let req_body = serde_json::json!({
                "id": format!("doc-{i}"),
                "text": format!("document about topic {i} in domain {}", i % 10),
                "cluster_id": i % 5,
            });
            let req = Request::builder()
                .method(Method::POST)
                .uri("/corpus/add")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        // Health should now report RLM active
        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert!(health.rlm_active, "RLM should be active after 55 corpus entries");
        assert_eq!(health.corpus_size, 55);
    }

    #[tokio::test]
    async fn test_health_initially_hash() {
        let state = test_state();
        let app = Router::new()
            .route("/health", get(health))
            .with_state(state);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(health.status, "ok");
        assert!(!health.rlm_active);
        assert_eq!(health.embed_dim, EMBED_DIM);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .init();

    let engine = EmbeddingEngine::new();
    tracing::info!(
        engine = %engine.engine_name(),
        corpus_size = engine.corpus_size(),
        "embedder starting"
    );

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
    });

    let app = Router::new()
        .route("/embed", post(embed_handler))
        .route("/embed/storage", post(embed_storage_handler))
        .route("/corpus/add", post(corpus_add_handler))
        .route("/health", get(health))
        .with_state(state);

    let port: u16 = std::env::var("RUVLLM_EMBEDDER_PORT")
        .unwrap_or_else(|_| "9877".to_string())
        .parse()
        .unwrap_or(9877);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    tracing::info!(port, "ruvllm-embedder listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}
