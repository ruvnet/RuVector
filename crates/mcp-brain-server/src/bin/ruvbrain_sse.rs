//! ruvbrain-sse: Thin SSE proxy for MCP transport decoupling (ADR-130).
//!
//! This binary has NO business logic. It:
//! - Accepts SSE connections on GET /sse (generates session ID per MCP spec)
//! - Sends initial `endpoint` event with /messages?sessionId=X
//! - Accepts JSON-RPC on POST /messages?sessionId=X
//! - Forwards JSON-RPC to the brain API at BRAIN_API_URL/internal/queue/push
//! - Polls API at /internal/queue/drain?sessionId=X for responses
//! - Streams responses back to SSE clients as `message` events
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
use uuid::Uuid;

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
    session_id: Option<String>,
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
// GET /sse — MCP SSE connection with drain polling
// ---------------------------------------------------------------------------

async fn sse_handler(
    State(state): State<AppState>,
    params: Option<Query<SessionQuery>>,
) -> impl IntoResponse {
    let current = state.active_connections.fetch_add(1, Ordering::SeqCst);
    if current >= state.max_connections {
        state.active_connections.fetch_sub(1, Ordering::SeqCst);
        let mut headers = HeaderMap::new();
        headers.insert("Retry-After", "10".parse().unwrap());
        return (
            StatusCode::TOO_MANY_REQUESTS,
            headers,
            "connection limit reached",
        )
            .into_response();
    }

    // Generate a session ID (MCP spec: server assigns it, not the client)
    let session_id = params
        .and_then(|q| q.0.session_id.clone())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    tracing::info!(
        session_id = %session_id,
        active = current + 1,
        "SSE connection opened"
    );

    // Register session with the brain API
    let create_url = format!("{}/internal/session/create", state.brain_api_url);
    if let Err(e) = state
        .client
        .post(&create_url)
        .json(&serde_json::json!({ "session_id": &session_id }))
        .send()
        .await
    {
        tracing::error!(error = %e, "failed to create session on brain API");
        state.active_connections.fetch_sub(1, Ordering::SeqCst);
        return (StatusCode::BAD_GATEWAY, "failed to create upstream session").into_response();
    }

    let active = Arc::clone(&state.active_connections);
    let client = state.client.clone();
    let api_url = state.brain_api_url.clone();
    let mut shutdown_rx = state.shutdown_tx.subscribe();
    let sid_for_cleanup = session_id.clone();
    let client_for_cleanup = state.client.clone();
    let api_for_cleanup = state.brain_api_url.clone();

    let stream = async_stream::stream! {
        // MCP protocol: first event is `endpoint` with the messages URL
        yield Ok::<_, Infallible>(
            Event::default()
                .event("endpoint")
                .data(format!("/messages?sessionId={session_id}"))
        );

        let drain_url = format!(
            "{}/internal/queue/drain?sessionId={}",
            api_url,
            urlencoding::encode(&session_id)
        );

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

            let body = match resp.text().await {
                Ok(b) => b,
                Err(_) => continue,
            };

            let trimmed = body.trim();
            if trimmed.is_empty() || trimmed == "[]" {
                continue;
            }

            // Parse as array of raw JSON strings and emit each as an SSE `message` event.
            if let Ok(messages) = serde_json::from_str::<Vec<serde_json::Value>>(trimmed) {
                for msg in messages {
                    // Messages are JSON-RPC response strings; emit as SSE data
                    let data = if msg.is_string() {
                        msg.as_str().unwrap_or_default().to_string()
                    } else {
                        serde_json::to_string(&msg).unwrap_or_default()
                    };
                    yield Ok::<_, Infallible>(Event::default().event("message").data(data));
                }
            } else {
                // Single object — emit as-is
                yield Ok::<_, Infallible>(Event::default().event("message").data(trimmed.to_owned()));
            }
        }

        // Decrement connection count on stream end.
        let remaining = active.fetch_sub(1, Ordering::SeqCst) - 1;
        tracing::info!(session_id = %sid_for_cleanup, active = remaining, "SSE connection closed");

        // Clean up session on brain API
        let delete_url = format!("{}/internal/session/{}", api_for_cleanup, sid_for_cleanup);
        let _ = client_for_cleanup.delete(&delete_url).send().await;
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

/// Alias so both `/` and `/sse` serve the SSE handler.
async fn sse_handler_alias(
    state: State<AppState>,
    params: Option<Query<SessionQuery>>,
) -> impl IntoResponse {
    sse_handler(state, params).await
}

// ---------------------------------------------------------------------------
// POST /messages?sessionId=X — forward JSON-RPC to brain API
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct MessagesQuery {
    #[serde(rename = "sessionId")]
    session_id: String,
}

async fn messages_handler(
    State(state): State<AppState>,
    Query(params): Query<MessagesQuery>,
    body: String,
) -> impl IntoResponse {
    // Forward to brain API's /messages endpoint which processes JSON-RPC
    // and sends the response through the session's mpsc channel → response_queues
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
            (
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                response_body,
            )
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
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(20)
            .build()?,
        brain_api_url: brain_api_url.clone(),
        active_connections: Arc::new(AtomicUsize::new(0)),
        max_connections,
        shutdown_tx: shutdown_tx.clone(),
    };

    let app = Router::new()
        .route("/", get(sse_handler))
        .route("/sse", get(sse_handler_alias))
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
