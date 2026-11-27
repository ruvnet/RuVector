//! MCP transport layers (STDIO and SSE)

use super::{handlers::McpHandler, protocol::*};
use anyhow::Result;
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde_json;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tower_http::cors::{AllowOrigin, CorsLayer};

/// STDIO transport for local MCP communication
pub struct StdioTransport {
    handler: Arc<McpHandler>,
}

impl StdioTransport {
    pub fn new(handler: Arc<McpHandler>) -> Self {
        Self { handler }
    }

    /// Run STDIO transport loop
    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        tracing::info!("MCP STDIO transport started");

        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;

            if n == 0 {
                // EOF
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Parse request
            let request: McpRequest = match serde_json::from_str(trimmed) {
                Ok(req) => req,
                Err(e) => {
                    let error_response = McpResponse::error(
                        None,
                        McpError::new(error_codes::PARSE_ERROR, e.to_string()),
                    );
                    let response_json = serde_json::to_string(&error_response)?;
                    stdout.write_all(response_json.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                    continue;
                }
            };

            // Handle request
            let response = self.handler.handle_request(request).await;

            // Send response
            let response_json = serde_json::to_string(&response)?;
            stdout.write_all(response_json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }

        tracing::info!("MCP STDIO transport stopped");
        Ok(())
    }
}

/// SSE (Server-Sent Events) transport for HTTP streaming
pub struct SseTransport {
    handler: Arc<McpHandler>,
    host: String,
    port: u16,
}

impl SseTransport {
    pub fn new(handler: Arc<McpHandler>, host: String, port: u16) -> Self {
        Self {
            handler,
            host,
            port,
        }
    }

    /// Run SSE transport server
    pub async fn run(&self) -> Result<()> {
        let app = sse_router(self.handler.clone());

        let addr = format!("{}:{}", self.host, self.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;

        tracing::info!("MCP SSE transport listening on http://{}", addr);
        axum::serve(listener, app).await?;

        Ok(())
    }
}

/// HTTP routes for the SSE transport.
fn sse_router(handler: Arc<McpHandler>) -> Router {
    // Use restrictive CORS: only allow localhost origins by default
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| {
            origin.to_str().map(is_loopback_origin).unwrap_or(false)
        }))
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    Router::new()
        .route("/", get(root))
        .route("/mcp", post(mcp_handler))
        .route("/mcp/sse", get(mcp_sse_handler))
        .layer(cors)
        .with_state(handler)
}

/// True when `origin` is exactly `http(s)://localhost|127.0.0.1|[::1]` with
/// an optional numeric port. A prefix match would also accept origins such as
/// `http://localhost.example.com`.
fn is_loopback_origin(origin: &str) -> bool {
    let rest = match origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    {
        Some(r) => r,
        None => return false,
    };
    let (host, port) = match rest.strip_prefix("[::1]") {
        Some(after) => ("[::1]", after),
        None => match rest.find(':') {
            Some(i) => rest.split_at(i),
            None => (rest, ""),
        },
    };
    let port_ok = match port.strip_prefix(':') {
        None => port.is_empty(),
        Some(p) => !p.is_empty() && p.len() <= 5 && p.bytes().all(|b| b.is_ascii_digit()),
    };
    port_ok && matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

// HTTP handlers

async fn root() -> &'static str {
    "Ruvector MCP Server"
}

async fn mcp_handler(
    State(handler): State<Arc<McpHandler>>,
    Json(request): Json<McpRequest>,
) -> Json<McpResponse> {
    let response = handler.handle_request(request).await;
    Json(response)
}

async fn mcp_sse_handler(
    State(handler): State<Arc<McpHandler>>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let stream = async_stream::stream! {
        // Send initial connection event
        yield Ok(Event::default().data("connected"));

        // Keep connection alive with periodic pings
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            yield Ok(Event::default().event("ping").data("keep-alive"));
        }
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(tokio::time::Duration::from_secs(30))
            .text("keep-alive"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[tokio::test]
    async fn test_stdio_transport_creation() {
        let config = Config::default();
        let handler = Arc::new(McpHandler::new(config));
        let _transport = StdioTransport::new(handler);
    }

    async fn preflight_allowed_origin(origin: &str) -> Option<String> {
        use tower::ServiceExt;
        let app = sse_router(Arc::new(McpHandler::new(Config::default())));
        let req = axum::http::Request::builder()
            .method("OPTIONS")
            .uri("/mcp")
            .header("origin", origin)
            .header("access-control-request-method", "POST")
            .header("access-control-request-headers", "content-type")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        resp.headers()
            .get("access-control-allow-origin")
            .map(|v| v.to_str().unwrap().to_string())
    }

    #[tokio::test]
    async fn cors_allows_loopback_origins() {
        for o in [
            "http://localhost",
            "http://localhost:3000",
            "https://127.0.0.1:8443",
            "http://[::1]:3000",
        ] {
            assert_eq!(preflight_allowed_origin(o).await.as_deref(), Some(o));
        }
    }

    #[tokio::test]
    async fn cors_rejects_lookalike_origins() {
        for o in [
            "http://localhost.example.com",
            "https://localhost-example.com",
            "http://127.0.0.1.example.com",
            "http://localhost:3000.example.com",
            "http://localhost:3000@example.com",
            "null",
        ] {
            assert_eq!(preflight_allowed_origin(o).await, None, "{o} was allowed");
        }
    }

    #[tokio::test]
    async fn test_sse_transport_creation() {
        let config = Config::default();
        let handler = Arc::new(McpHandler::new(config));
        let _transport = SseTransport::new(handler, "127.0.0.1".to_string(), 3000);
    }
}
