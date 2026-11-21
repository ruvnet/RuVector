//! ruvector-server: REST API server for rUvector vector database
//!
//! This crate provides a REST API server built on axum for interacting with rUvector.

pub mod error;
pub mod routes;
pub mod state;

use axum::{routing::get, Router};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tower_http::{
    compression::CompressionLayer,
    cors::{AllowOrigin, Any, CorsLayer},
    trace::TraceLayer,
};

pub use error::{Error, Result};
pub use state::AppState;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server host address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Enable CORS for browser pages served from loopback origins
    /// (`localhost`, `127.0.0.1`, `[::1]`). The API has no authentication, so
    /// pages from any other origin are not allowed to call it.
    pub enable_cors: bool,
    /// Enable compression
    pub enable_compression: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 6333,
            enable_cors: true,
            enable_compression: true,
        }
    }
}

/// Whether a browser `Origin` is an http(s) page on a loopback host.
fn is_loopback_origin(origin: &[u8]) -> bool {
    let Ok(origin) = std::str::from_utf8(origin) else {
        return false;
    };
    let Some(rest) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    let host = if rest.starts_with('[') {
        match rest.find(']') {
            Some(end) if rest[end + 1..].is_empty() || rest[end + 1..].starts_with(':') => {
                &rest[..=end]
            }
            _ => return false,
        }
    } else {
        rest.split(':').next().unwrap_or("")
    };
    let port = &rest[host.len()..];
    if !(port.is_empty()
        || (port.len() > 1
            && port.starts_with(':')
            && port[1..].bytes().all(|b| b.is_ascii_digit())))
    {
        return false;
    }
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

/// Main server structure
pub struct RuvectorServer {
    config: Config,
    state: AppState,
}

impl RuvectorServer {
    /// Create a new server instance with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            state: AppState::new(),
        }
    }

    /// Create a new server instance with custom configuration
    pub fn with_config(config: Config) -> Self {
        Self {
            config,
            state: AppState::new(),
        }
    }

    /// Build the router with all routes
    fn build_router(&self) -> Router {
        let mut router = Router::new()
            .route("/health", get(routes::health::health_check))
            .route("/ready", get(routes::health::readiness))
            .nest("/collections", routes::collections::routes())
            .merge(routes::points::routes())
            .with_state(self.state.clone());

        // Add middleware layers
        router = router.layer(TraceLayer::new_for_http());

        if self.config.enable_compression {
            router = router.layer(CompressionLayer::new());
        }

        if self.config.enable_cors {
            let cors = CorsLayer::new()
                .allow_origin(AllowOrigin::predicate(|origin, _| {
                    is_loopback_origin(origin.as_bytes())
                }))
                .allow_methods(Any)
                .allow_headers(Any);
            router = router.layer(cors);
        }

        router
    }

    /// Start the server
    ///
    /// # Errors
    ///
    /// Returns an error if the server fails to bind or start
    pub async fn start(self) -> Result<()> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| Error::Config(format!("Invalid address: {}", e)))?;

        let router = self.build_router();

        tracing::info!("Starting ruvector-server on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| Error::Server(format!("Failed to bind to {}: {}", addr, e)))?;

        axum::serve(listener, router)
            .await
            .map_err(|e| Error::Server(format!("Server error: {}", e)))?;

        Ok(())
    }
}

impl Default for RuvectorServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use tower::ServiceExt;

    async fn send(router: Router, req: Request<Body>) -> axum::response::Response {
        router.oneshot(req).await.expect("request")
    }

    fn get_with_origin(origin: &str) -> Request<Body> {
        Request::builder()
            .uri("/collections")
            .header(header::ORIGIN, origin)
            .body(Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn cors_does_not_grant_arbitrary_web_origins() {
        let router = RuvectorServer::new().build_router();
        let resp = send(router.clone(), get_with_origin("https://attacker.example")).await;
        assert!(
            resp.headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .is_none(),
            "a foreign web origin must not be allowed to read responses"
        );

        let preflight = Request::builder()
            .method(Method::OPTIONS)
            .uri("/collections/x")
            .header(header::ORIGIN, "https://attacker.example")
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "DELETE")
            .body(Body::empty())
            .unwrap();
        let resp = send(router, preflight).await;
        assert!(resp
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none());
    }

    #[tokio::test]
    async fn cors_allows_loopback_origins() {
        let router = RuvectorServer::new().build_router();
        for origin in [
            "http://localhost:3000",
            "http://127.0.0.1:8080",
            "http://[::1]:5173",
        ] {
            let resp = send(router.clone(), get_with_origin(origin)).await;
            assert_eq!(
                resp.headers()
                    .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                    .map(|v| v.as_bytes()),
                Some(origin.as_bytes()),
                "{origin} should be allowed"
            );
        }
        for origin in [
            "http://localhost.attacker.example",
            "http://127.0.0.1.attacker.example",
        ] {
            let resp = send(router.clone(), get_with_origin(origin)).await;
            assert!(
                resp.headers()
                    .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                    .is_none(),
                "{origin}"
            );
        }
    }
}
