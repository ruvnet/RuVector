//! ruvbrain-sse: Thin SSE proxy for MCP transport decoupling (ADR-130).
//!
//! This binary has NO business logic. It:
//! - Accepts SSE connections on GET /sse (max 200 concurrent)
//! - Accepts JSON-RPC on POST /messages?sessionId=X
//! - Forwards JSON-RPC to the brain API at BRAIN_API_URL
//! - Polls API at /internal/queue/drain?sessionId=X for responses
//! - Streams responses back to SSE clients
//! - Provides /v1/ready and /v1/health endpoints

use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tokio::sync::broadcast;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Shared application state for the SSE proxy.
#[derive(Clone)]
struct AppState {
    /// HTTP client for forwarding requests to the brain API.
    client: reqwest::Client,
    /// Base URL of the brain API (e.g. http://localhost:8080).
    brain_api_url: String,
    /// Number of active SSE connections.
    active_connections: Arc<AtomicUsize>,
    /// Maximum allowed concurrent SSE connections.
    max_connections: usize,
    /// Shutdown signal sender — all SSE streams listen on a receiver.
    shutdown_tx: broadcast::Sender<()>,
}

// ---------------------------------------------------------------------------
// Query params
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SessionQuery {
    #[serde(rename = "sessionId")]
    session_id: String,
}

// ---------------------------------------------------------------------------
// Health / readiness
// ---------------------------------------------------------------------------

async fn ready() -> StatusCode {
    StatusCode::OK
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    active_connections: usize,
    max_connections: usize,
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "ruvbrain-sse",
        active_connections: state.active_connections.load(Ordering::Relaxed),
        max_connections: state.max_connections,
    })
}

// ---------------------------------------------------------------------------
// GET /sse — SSE connection with drain polling
// ---------------------------------------------------------------------------

async fn sse_handler(
    State(state): State<AppState>,
    Query(params): Query<SessionQuery>,
) -> impl IntoResponse {
    let current = state.active_connections.fetch_add(1, Ordering::SeqCst);
    if current >= state.max_connections {
        // Undo the increment — we are rejecting this connection.
        state.active_connections.fetch_sub(1, Ordering::SeqCst);
        let mut headers = HeaderMap::new();
        headers.insert("Retry-After", "10".parse().unwrap());
        return (StatusCode::TOO_MANY_REQUESTS, headers, "connection limit reached")
            .into_response();
    }

    tracing::info!(
        session_id = %params.session_id,
        active = current + 1,
        "SSE connection opened"
    );

    let session_id = params.session_id.clone();
    let active = Arc::clone(&state.active_connections);
    let client = state.client.clone();
    let api_url = state.brain_api_url.clone();
    let mut shutdown_rx = state.shutdown_tx.subscribe();

    // Build an async SSE stream that polls the drain endpoint.
    let stream = async_stream::stream! {
        let drain_url = format!(
            "{}/internal/queue/drain?sessionId={}",
            api_url,
            urlencoding::encode(&session_id)
        );

        // Send an initial comment so the client knows the stream is live.
        yield Ok::<_, Infallible>(Event::default().comment("connected"));

        let mut interval = tokio::time::interval(Duration::from_millis(100));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = interval.tick() => {}
                _ = shutdown_rx.recv() => {
                    tracing::info!(session_id = %session_id, "SSE stream closing (shutdown)");
                    break;
                }
            }

            // Poll the API drain endpoint for queued responses.
            let resp = match client.get(&drain_url).send().await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(error = %e, "drain poll failed");
                    continue;
                }
            };

            if !resp.status().is_success() {
                continue;
            }

            // The drain endpoint returns a JSON array of messages.
            let body = match resp.text().await {
                Ok(b) => b,
                Err(_) => continue,
            };

            // Skip empty arrays.
            let trimmed = body.trim();
            if trimmed.is_empty() || trimmed == "[]" {
                continue;
            }

            // Parse as array of raw JSON values and emit each as an SSE event.
            if let Ok(messages) = serde_json::from_str::<Vec<serde_json::Value>>(trimmed) {
                for msg in messages {
                    let data = serde_json::to_string(&msg).unwrap_or_default();
                    yield Ok::<_, Infallible>(Event::default().data(data));
                }
            } else {
                // Single object — emit as-is.
                yield Ok::<_, Infallible>(Event::default().data(trimmed.to_owned()));
            }
        }

        // Decrement connection count on stream end.
        let remaining = active.fetch_sub(1, Ordering::SeqCst) - 1;
        tracing::info!(session_id = %session_id, active = remaining, "SSE connection closed");
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

// ---------------------------------------------------------------------------
// POST /messages?sessionId=X — forward JSON-RPC to brain API
// ---------------------------------------------------------------------------

async fn messages_handler(
    State(state): State<AppState>,
    Query(params): Query<SessionQuery>,
    body: String,
) -> impl IntoResponse {
    let forward_url = format!(
        "{}/messages?sessionId={}",
        state.brain_api_url,
        urlencoding::encode(&params.session_id)
    );

    match state
        .client
        .post(&forward_url)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let response_body = resp.text().await.unwrap_or_default();
            (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY), response_body)
                .into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "failed to forward JSON-RPC");
            (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Logging
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .init();

    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()?;

    let brain_api_url = std::env::var("BRAIN_API_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_string())
        .trim_end_matches('/')
        .to_string();

    let max_connections: usize = std::env::var("MAX_SSE_CONNECTIONS")
        .unwrap_or_else(|_| "200".to_string())
        .parse()
        .unwrap_or(200);

    let (shutdown_tx, _) = broadcast::channel::<()>(1);

    let state = AppState {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(20)
            .build()?,
        brain_api_url: brain_api_url.clone(),
        active_connections: Arc::new(AtomicUsize::new(0)),
        max_connections,
        shutdown_tx: shutdown_tx.clone(),
    };

    let app = Router::new()
        .route("/sse", get(sse_handler))
        .route("/messages", post(messages_handler))
        .route("/v1/ready", get(ready))
        .route("/v1/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    tracing::info!(
        port,
        brain_api = %brain_api_url,
        max_sse = max_connections,
        "ruvbrain-sse proxy started"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_tx))
        .await?;

    tracing::info!("ruvbrain-sse shut down cleanly");
    Ok(())
}

/// Wait for SIGTERM (Cloud Run) or SIGINT (local dev), then broadcast shutdown.
async fn shutdown_signal(shutdown_tx: broadcast::Sender<()>) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT"),
        _ = terminate => tracing::info!("received SIGTERM"),
    }

    let _ = shutdown_tx.send(());
}
