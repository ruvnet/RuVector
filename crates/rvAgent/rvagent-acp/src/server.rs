//! Axum-based ACP server implementation.
//!
//! Per ADR-099 and ADR-103 C6, provides:
//! - `POST /prompt` — send prompt to agent
//! - `GET  /sessions` — list sessions
//! - `POST /sessions` — create session
//! - `GET  /sessions/:id` — get session
//! - `DELETE /sessions/:id` — delete session
//! - `GET  /health` — health check
//!
//! With authentication, rate limiting, and body size enforcement middleware.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    middleware,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::agent::AcpAgent;
use crate::auth::{
    rate_limiter, request_size_limit, require_api_key, require_tls_middleware, ApiKeyState,
    MaxBodySize, RateLimiterState, RequireTls,
};
use crate::types::{CreateSessionRequest, ErrorResponse, HealthResponse, PromptRequest};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// ACP server configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AcpConfig {
    /// Listen address (default: "0.0.0.0").
    pub host: String,

    /// Listen port (default: 3100).
    pub port: u16,

    /// Optional API key for Bearer authentication. `None` disables auth.
    pub api_key: Option<String>,

    /// Maximum requests per minute per IP (default: 60).
    pub rate_limit: u32,

    /// Maximum request body size in bytes (default: 1 MB).
    pub max_body_size: usize,

    /// Whether to require TLS for non-localhost connections.
    pub require_tls: bool,
}

impl Default for AcpConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 3100,
            api_key: None,
            rate_limit: 60,
            max_body_size: 1024 * 1024, // 1 MB
            require_tls: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// Application state shared across all handlers.
#[derive(Clone)]
pub struct AppState {
    pub agent: Arc<AcpAgent>,
}

/// The ACP server wrapping an `AcpAgent` with HTTP routes and middleware.
#[allow(dead_code)]
pub struct AcpServer {
    agent: Arc<AcpAgent>,
    config: AcpConfig,
}

impl AcpServer {
    /// Create a new server with the given agent and configuration.
    pub fn new(agent: AcpAgent, config: AcpConfig) -> Self {
        Self {
            agent: Arc::new(agent),
            config,
        }
    }

    /// Build the axum `Router` with all routes and middleware layers.
    pub fn router(&self) -> Router {
        let state = AppState {
            agent: Arc::clone(&self.agent),
        };

        let api_key_state = ApiKeyState {
            api_key: self.config.api_key.clone(),
        };
        let rate_limiter_state = RateLimiterState::new(self.config.rate_limit);
        let max_body = MaxBodySize(self.config.max_body_size);
        let require_tls = RequireTls(self.config.require_tls);

        Router::new()
            // Routes
            .route("/prompt", post(handle_prompt))
            .route(
                "/sessions",
                get(handle_list_sessions).post(handle_create_session),
            )
            .route(
                "/sessions/{id}",
                get(handle_get_session).delete(handle_delete_session),
            )
            .route("/health", get(handle_health))
            // Shared state
            .with_state(state)
            // Middleware layers (applied bottom-up: TLS first, then auth, rate limit, size check, body limit)
            .layer(middleware::from_fn(require_api_key))
            .layer(middleware::from_fn(rate_limiter))
            .layer(middleware::from_fn(request_size_limit))
            .layer(middleware::from_fn(require_tls_middleware))
            .layer(axum::Extension(api_key_state))
            .layer(axum::Extension(rate_limiter_state))
            .layer(axum::Extension(max_body))
            .layer(axum::Extension(require_tls))
            .layer(RequestBodyLimitLayer::new(self.config.max_body_size))
            .layer(TraceLayer::new_for_http())
            .layer(CorsLayer::permissive())
    }

    /// Start listening and serving requests.
    pub async fn serve(self) -> anyhow::Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!("ACP server listening on {}", addr);

        let router = self.router();
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await?;

        Ok(())
    }

    /// Access the server configuration.
    #[allow(dead_code)]
    pub fn config(&self) -> &AcpConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `POST /prompt` — send a prompt to the agent.
async fn handle_prompt(
    State(state): State<AppState>,
    Json(req): Json<PromptRequest>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    match state
        .agent
        .prompt(req.session_id.as_deref(), req.content)
        .await
    {
        Ok(resp) => Ok((StatusCode::OK, Json(resp))),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse::bad_request(e)))),
    }
}

/// `GET /sessions` — list all sessions.
async fn handle_list_sessions(State(state): State<AppState>) -> impl IntoResponse {
    let sessions = state.agent.list_sessions().await;
    (StatusCode::OK, Json(sessions))
}

/// `POST /sessions` — create a new session.
async fn handle_create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> impl IntoResponse {
    let info = state.agent.create_session(&req).await;
    (StatusCode::CREATED, Json(info))
}

/// `GET /sessions/:id` — get a session by ID.
async fn handle_get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    match state.agent.get_session(&id).await {
        Some(info) => Ok((StatusCode::OK, Json(info))),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::not_found(format!(
                "session not found: {}",
                id
            ))),
        )),
    }
}

/// `DELETE /sessions/:id` — delete a session.
async fn handle_delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if state.agent.delete_session(&id).await {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

/// `GET /health` — health check (no auth required).
async fn handle_health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PromptResponse, SessionInfo};
    use axum::body::Body;
    use axum::http::{self, Request};
    use rvagent_core::config::RvAgentConfig;
    use tower::ServiceExt;

    fn test_server(api_key: Option<&str>) -> AcpServer {
        let agent = AcpAgent::new(RvAgentConfig::default());
        let config = AcpConfig {
            api_key: api_key.map(|s| s.to_string()),
            rate_limit: 60,
            max_body_size: 1024 * 1024,
            ..AcpConfig::default()
        };
        AcpServer::new(agent, config)
    }

    #[tokio::test]
    async fn test_health_no_auth_required() {
        let server = test_server(Some("secret-key"));
        let app = server.router();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_rejects_missing_key() {
        let server = test_server(Some("secret-key"));
        let app = server.router();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_rejects_invalid_key() {
        let server = test_server(Some("secret-key"));
        let app = server.router();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .header("Authorization", "Bearer wrong-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_accepts_valid_key() {
        let server = test_server(Some("secret-key"));
        let app = server.router();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .header("Authorization", "Bearer secret-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_no_auth_when_key_not_configured() {
        let server = test_server(None);
        let app = server.router();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_create_and_get_session() {
        let server = test_server(None);
        let app = server.router();

        // Create a session.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/sessions")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::CREATED);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let info: SessionInfo = serde_json::from_slice(&body).unwrap();
        assert!(!info.id.is_empty());

        // Get the session.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/sessions/{}", info.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_delete_session() {
        let server = test_server(None);
        let app = server.router();

        // Create.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/sessions")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let info: SessionInfo = serde_json::from_slice(&body).unwrap();

        // Delete.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(http::Method::DELETE)
                    .uri(&format!("/sessions/{}", info.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        // Get should 404.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/sessions/{}", info.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_prompt_endpoint() {
        let server = test_server(None);
        let app = server.router();

        let req_body = serde_json::json!({
            "content": [{"type": "text", "text": "hello agent"}]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/prompt")
                    .header("Content-Type", "application/json")
                    .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let prompt_resp: PromptResponse = serde_json::from_slice(&body).unwrap();
        assert!(!prompt_resp.session_id.is_empty());
        assert!(!prompt_resp.messages.is_empty());
    }

    #[tokio::test]
    async fn test_get_nonexistent_session_returns_404() {
        let server = test_server(None);
        let app = server.router();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/sessions/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_tls_not_required_by_default() {
        let agent = AcpAgent::new(RvAgentConfig::default());
        let config = AcpConfig {
            require_tls: false,
            ..AcpConfig::default()
        };
        let server = AcpServer::new(agent, config);
        let app = server.router();

        // Non-HTTPS request should succeed when TLS is not required.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .header("Host", "example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_tls_allows_localhost() {
        let agent = AcpAgent::new(RvAgentConfig::default());
        let config = AcpConfig {
            require_tls: true,
            ..AcpConfig::default()
        };
        let server = AcpServer::new(agent, config);
        let app = server.router();

        // Localhost should bypass TLS requirement.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .header("Host", "localhost:3100")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_tls_requires_https_for_remote() {
        let agent = AcpAgent::new(RvAgentConfig::default());
        let config = AcpConfig {
            require_tls: true,
            ..AcpConfig::default()
        };
        let server = AcpServer::new(agent, config);
        let app = server.router();

        // Non-localhost, non-HTTPS request should be forbidden.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .header("Host", "example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        // Same request with x-forwarded-proto: https should succeed.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .header("Host", "example.com")
                    .header("x-forwarded-proto", "https")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }
}
