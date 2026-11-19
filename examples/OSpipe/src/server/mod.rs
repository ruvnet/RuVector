//! Lightweight HTTP REST API server for OSpipe.
//!
//! Exposes the ingestion pipeline, search, routing, and health endpoints
//! that the TypeScript SDK (`@ruvector/ospipe`) expects. Built on
//! [axum](https://docs.rs/axum) and gated behind
//! `cfg(not(target_arch = "wasm32"))` since WASM targets cannot bind
//! TCP sockets.
//!
//! ## Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | `POST` | `/v2/search` | Semantic / hybrid vector search |
//! | `POST` | `/v2/route` | Query routing |
//! | `GET`  | `/v2/stats` | Pipeline statistics |
//! | `GET`  | `/v2/health` | Health check |
//! | `GET`  | `/search` | Legacy Screenpipe v1 search |

use std::sync::Arc;

use axum::{
    extract::{Query, Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::pipeline::ingestion::{IngestionPipeline, PipelineStats};
use crate::search::router::{QueryRoute, QueryRouter};
use crate::storage::vector_store::SearchResult;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Shared server state holding the pipeline behind a read-write lock.
#[derive(Clone)]
pub struct ServerState {
    /// The ingestion pipeline (search + store).
    pub pipeline: Arc<RwLock<IngestionPipeline>>,
    /// The query router.
    pub router: Arc<QueryRouter>,
    /// Server start instant for uptime calculation.
    pub started_at: std::time::Instant,
}

// ---------------------------------------------------------------------------
// Request / response DTOs
// ---------------------------------------------------------------------------

/// Request body for `POST /v2/search`.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// Natural-language query string.
    pub query: String,
    /// Search mode hint (semantic, keyword, hybrid).
    #[serde(default = "default_search_mode")]
    pub mode: String,
    /// Number of results to return.
    #[serde(default = "default_k")]
    pub k: usize,
    /// Distance metric (cosine, euclidean, dot).
    #[serde(default = "default_metric")]
    pub metric: String,
    /// Optional metadata filters.
    pub filters: Option<SearchFilters>,
    /// Whether to apply MMR reranking.
    #[serde(default)]
    pub rerank: bool,
}

fn default_search_mode() -> String {
    "semantic".to_string()
}
fn default_k() -> usize {
    10
}
fn default_metric() -> String {
    "cosine".to_string()
}

/// Metadata filters mirroring the TypeScript SDK `SearchFilters` type.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchFilters {
    pub app: Option<String>,
    pub window: Option<String>,
    pub content_type: Option<String>,
    pub time_range: Option<TimeRange>,
    pub monitor: Option<u32>,
    pub speaker: Option<String>,
    pub language: Option<String>,
}

/// ISO-8601 time range.
#[derive(Debug, Deserialize)]
pub struct TimeRange {
    pub start: String,
    pub end: String,
}

/// Request body for `POST /v2/route`.
#[derive(Debug, Deserialize)]
pub struct RouteRequest {
    pub query: String,
}

/// Response body for `POST /v2/route`.
#[derive(Debug, Serialize, Deserialize)]
pub struct RouteResponse {
    pub route: String,
}

/// Response body for `GET /v2/stats`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StatsResponse {
    pub total_ingested: u64,
    pub total_deduplicated: u64,
    pub total_denied: u64,
    pub total_redacted: u64,
    pub storage_bytes: u64,
    pub index_size: usize,
    pub uptime: u64,
}

/// Response body for `GET /v2/health`.
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub backends: Vec<String>,
}

/// API-facing search result that matches the TypeScript SDK `SearchResult`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiSearchResult {
    pub id: String,
    pub score: f32,
    pub content: String,
    pub source: String,
    pub timestamp: String,
    pub metadata: serde_json::Value,
}

/// Query parameters for `GET /search` (legacy v1).
#[derive(Debug, Deserialize)]
pub struct LegacySearchParams {
    pub q: Option<String>,
    pub content_type: Option<String>,
    pub limit: Option<usize>,
}

/// Wrapper for JSON error responses.
#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `POST /v2/search` - Semantic / hybrid search.
async fn search_handler(
    State(state): State<ServerState>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let pipeline = state.pipeline.read().await;
    let embedding = pipeline.embedding_engine().embed(&req.query);
    let k = if req.k == 0 {
        10
    } else {
        req.k.min(MAX_SEARCH_K)
    };

    let filter = build_search_filter(&req.filters);

    let results = if filter_is_empty(&filter) {
        pipeline.vector_store().search(&embedding, k)
    } else {
        pipeline
            .vector_store()
            .search_filtered(&embedding, k, &filter)
    };

    match results {
        Ok(results) => {
            let api_results: Vec<ApiSearchResult> =
                results.into_iter().map(to_api_result).collect();
            (StatusCode::OK, Json(api_results)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

/// `POST /v2/route` - Query routing.
async fn route_handler(
    State(state): State<ServerState>,
    Json(req): Json<RouteRequest>,
) -> impl IntoResponse {
    let route = state.router.route(&req.query);
    let route_str = match route {
        QueryRoute::Semantic => "semantic",
        QueryRoute::Keyword => "keyword",
        QueryRoute::Graph => "graph",
        QueryRoute::Temporal => "temporal",
        QueryRoute::Hybrid => "hybrid",
    };
    Json(RouteResponse {
        route: route_str.to_string(),
    })
}

/// `GET /v2/stats` - Pipeline statistics.
async fn stats_handler(State(state): State<ServerState>) -> impl IntoResponse {
    let pipeline = state.pipeline.read().await;
    let stats: &PipelineStats = pipeline.stats();
    let index_size = pipeline.vector_store().len();
    let uptime = state.started_at.elapsed().as_secs();

    Json(StatsResponse {
        total_ingested: stats.total_ingested,
        total_deduplicated: stats.total_deduplicated,
        total_denied: stats.total_denied,
        total_redacted: stats.total_redacted,
        storage_bytes: 0, // not tracked in the in-memory store
        index_size,
        uptime,
    })
}

/// `GET /v2/health` - Health check.
async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        backends: vec![
            "hnsw".to_string(),
            "keyword".to_string(),
            "graph".to_string(),
        ],
    })
}

/// `GET /search` - Legacy Screenpipe v1 search endpoint.
async fn legacy_search_handler(
    State(state): State<ServerState>,
    Query(params): Query<LegacySearchParams>,
) -> impl IntoResponse {
    let q = match params.q {
        Some(q) if !q.is_empty() => q,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: "Missing required query parameter 'q'".to_string(),
                }),
            )
                .into_response();
        }
    };

    let k = params.limit.unwrap_or(10).min(MAX_SEARCH_K);
    let pipeline = state.pipeline.read().await;
    let embedding = pipeline.embedding_engine().embed(&q);

    let filter = if let Some(ref ct) = params.content_type {
        let mapped = match ct.as_str() {
            "ocr" => "ocr",
            "audio" => "transcription",
            "ui" => "ui_event",
            _ => "",
        };
        if mapped.is_empty() {
            crate::storage::vector_store::SearchFilter::default()
        } else {
            crate::storage::vector_store::SearchFilter {
                content_type: Some(mapped.to_string()),
                ..Default::default()
            }
        }
    } else {
        crate::storage::vector_store::SearchFilter::default()
    };

    let results = if filter_is_empty(&filter) {
        pipeline.vector_store().search(&embedding, k)
    } else {
        pipeline
            .vector_store()
            .search_filtered(&embedding, k, &filter)
    };

    match results {
        Ok(results) => {
            let api_results: Vec<ApiSearchResult> =
                results.into_iter().map(to_api_result).collect();
            (StatusCode::OK, Json(api_results)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `SearchFilter` from optional API filters.
fn build_search_filter(
    filters: &Option<SearchFilters>,
) -> crate::storage::vector_store::SearchFilter {
    let Some(f) = filters else {
        return crate::storage::vector_store::SearchFilter::default();
    };

    let content_type = f.content_type.as_deref().map(|ct| {
        match ct {
            "screen" => "ocr",
            "audio" => "transcription",
            "ui" => "ui_event",
            other => other,
        }
        .to_string()
    });

    let (time_start, time_end) = if let Some(ref tr) = f.time_range {
        (
            chrono::DateTime::parse_from_rfc3339(&tr.start)
                .ok()
                .map(|dt| dt.with_timezone(&chrono::Utc)),
            chrono::DateTime::parse_from_rfc3339(&tr.end)
                .ok()
                .map(|dt| dt.with_timezone(&chrono::Utc)),
        )
    } else {
        (None, None)
    };

    crate::storage::vector_store::SearchFilter {
        app: f.app.clone(),
        time_start,
        time_end,
        content_type,
        monitor: f.monitor,
    }
}

/// Check whether a filter is effectively empty (no criteria set).
fn filter_is_empty(f: &crate::storage::vector_store::SearchFilter) -> bool {
    f.app.is_none()
        && f.time_start.is_none()
        && f.time_end.is_none()
        && f.content_type.is_none()
        && f.monitor.is_none()
}

/// Convert an internal `SearchResult` to the API-facing DTO.
fn to_api_result(r: SearchResult) -> ApiSearchResult {
    let content = r
        .metadata
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let source = r
        .metadata
        .get("content_type")
        .and_then(|v| v.as_str())
        .map(|ct| match ct {
            "ocr" => "screen",
            "transcription" => "audio",
            "ui_event" => "ui",
            other => other,
        })
        .unwrap_or("screen")
        .to_string();

    ApiSearchResult {
        id: r.id.to_string(),
        score: r.score,
        content,
        source,
        timestamp: chrono::Utc::now().to_rfc3339(),
        metadata: r.metadata,
    }
}

// ---------------------------------------------------------------------------
// Router & startup
// ---------------------------------------------------------------------------

/// Upper bound on results returned by a single search request.
const MAX_SEARCH_K: usize = 1000;

/// True when `host` (a Host header value or URL authority, optionally with a
/// port) names the local machine.
fn is_loopback_host(host: &str) -> bool {
    let host = host.trim();
    let name = if let Some(rest) = host.strip_prefix('[') {
        match rest.split_once(']') {
            Some((inner, after)) if after.is_empty() || after.starts_with(':') => inner,
            _ => return false,
        }
    } else {
        match host.rsplit_once(':') {
            Some((name, port)) if port.chars().all(|c| c.is_ascii_digit()) => name,
            Some(_) => return false,
            None => host,
        }
    };
    name.eq_ignore_ascii_case("localhost")
        || name
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

/// True for browser origins served from the local machine (or a Tauri shell).
/// Indexed screen/audio content must not be readable by arbitrary websites.
fn is_local_origin(origin: &HeaderValue) -> bool {
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    let Some((scheme, authority)) = origin.split_once("://") else {
        return false;
    };
    match scheme {
        "http" | "https" => is_loopback_host(authority),
        "tauri" => authority == "localhost",
        _ => false,
    }
}

/// Reject requests whose Host header is not a loopback name, so a remote
/// page cannot reach the API through DNS rebinding.
async fn require_loopback_host(req: Request, next: Next) -> Response {
    let ok = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .map(is_loopback_host)
        .unwrap_or(false);
    if !ok {
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorBody {
                error: "Host header must name the local machine".to_string(),
            }),
        )
            .into_response();
    }
    next.run(req).await
}

/// Build the axum [`Router`] with all OSpipe endpoints.
pub fn build_router(state: ServerState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| is_local_origin(origin)))
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // v2 API
        .route("/v2/search", post(search_handler))
        .route("/v2/route", post(route_handler))
        .route("/v2/stats", get(stats_handler))
        .route("/v2/health", get(health_handler))
        // Legacy v1
        .route("/search", get(legacy_search_handler))
        .layer(middleware::from_fn(require_loopback_host))
        .layer(cors)
        .with_state(state)
}

/// Start the OSpipe HTTP server on the given loopback port.
///
/// This function blocks until the server is shut down (e.g. via Ctrl-C).
///
/// # Errors
///
/// Returns an error if the TCP listener cannot bind to the requested port.
pub async fn start_server(state: ServerState, port: u16) -> crate::error::Result<()> {
    let app = build_router(state);
    // Loopback only: the index holds captured screen and audio content.
    let addr = std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, port));
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| OsPipeError::Pipeline(format!("Failed to bind to {}: {}", addr, e)))?;

    tracing::info!("OSpipe server listening on {}", addr);

    axum::serve(listener, app)
        .await
        .map_err(|e| OsPipeError::Pipeline(format!("Server error: {}", e)))?;

    Ok(())
}

use crate::error::OsPipeError;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OsPipeConfig;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for oneshot

    fn test_state() -> ServerState {
        let config = OsPipeConfig::default();
        let pipeline = IngestionPipeline::new(config).unwrap();
        ServerState {
            pipeline: Arc::new(RwLock::new(pipeline)),
            router: Arc::new(QueryRouter::new()),
            started_at: std::time::Instant::now(),
        }
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = test_state();
        let app = build_router(state);

        let req = Request::builder()
            .header("host", "localhost:3030")
            .uri("/v2/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(health.status, "ok");
        assert_eq!(health.version, env!("CARGO_PKG_VERSION"));
        assert!(!health.backends.is_empty());
    }

    #[tokio::test]
    async fn test_stats_endpoint() {
        let state = test_state();
        let app = build_router(state);

        let req = Request::builder()
            .header("host", "localhost:3030")
            .uri("/v2/stats")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let stats: StatsResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(stats.total_ingested, 0);
        assert_eq!(stats.index_size, 0);
    }

    #[tokio::test]
    async fn test_route_endpoint() {
        let state = test_state();
        let app = build_router(state);

        let req = Request::builder()
            .header("host", "localhost:3030")
            .method("POST")
            .uri("/v2/route")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"query": "what happened yesterday"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let route: RouteResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(route.route, "temporal");
    }

    #[tokio::test]
    async fn test_search_endpoint_empty_store() {
        let state = test_state();
        let app = build_router(state);

        let req = Request::builder()
            .header("host", "localhost:3030")
            .method("POST")
            .uri("/v2/search")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"query": "test", "mode": "semantic", "k": 5}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let results: Vec<ApiSearchResult> = serde_json::from_slice(&body).unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_legacy_search_missing_q() {
        let state = test_state();
        let app = build_router(state);

        let req = Request::builder()
            .header("host", "localhost:3030")
            .uri("/search")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_search_with_ingested_data() {
        let state = test_state();
        // Ingest a frame so there is data to search
        {
            let mut pipeline = state.pipeline.write().await;
            let frame = crate::capture::CapturedFrame::new_screen(
                "VSCode",
                "main.rs",
                "fn main() { println!(\"hello\"); }",
                0,
            );
            pipeline.ingest(frame).unwrap();
        }

        let app = build_router(state);

        let req = Request::builder()
            .header("host", "localhost:3030")
            .method("POST")
            .uri("/v2/search")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"query": "fn main", "k": 5}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let results: Vec<ApiSearchResult> = serde_json::from_slice(&body).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("fn main"));
        assert_eq!(results[0].source, "screen");
    }
    #[tokio::test]
    async fn test_remote_origin_gets_no_cors_grant() {
        let app = build_router(test_state());
        let req = Request::builder()
            .uri("/search?q=password")
            .header("host", "localhost:3030")
            .header("origin", "https://attacker.example")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "a remote website must not be allowed to read search results"
        );
    }

    #[tokio::test]
    async fn test_local_origin_gets_cors_grant() {
        let app = build_router(test_state());
        let req = Request::builder()
            .uri("/v2/health")
            .header("host", "localhost:3030")
            .header("origin", "http://localhost:5173")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.headers().get("access-control-allow-origin").unwrap(),
            "http://localhost:5173"
        );
    }

    #[tokio::test]
    async fn test_non_loopback_host_header_rejected() {
        let app = build_router(test_state());
        let req = Request::builder()
            .uri("/search?q=password")
            .header("host", "rebind.attacker.example:3030")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn test_loopback_host_parsing() {
        for ok in [
            "localhost",
            "localhost:3030",
            "127.0.0.1:3030",
            "[::1]:3030",
            "LOCALHOST",
        ] {
            assert!(is_loopback_host(ok), "{ok}");
        }
        for bad in [
            "evil.example",
            "localhost.evil.example",
            "0.0.0.0:3030",
            "192.168.1.5:3030",
            "[::1]x",
            "localhost:abc",
        ] {
            assert!(!is_loopback_host(bad), "{bad}");
        }
    }
}
