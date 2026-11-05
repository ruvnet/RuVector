//! rvf-server -- RuVector Format TCP/HTTP streaming server.
//!
//! Provides a network-accessible interface to `rvf-runtime`,
//! supporting HTTP REST endpoints and a binary TCP streaming protocol
//! for inter-agent vector exchange.
//!
//! The server binds to loopback by default. Binding any other address
//! requires an [`auth::AuthToken`]; HTTP requests then need
//! `Authorization: Bearer <token>` (except `GET /v1/health`). The binary TCP
//! protocol has no authentication, so it always stays on loopback.

pub mod auth;
pub mod error;
pub mod http;
pub mod stream;
pub mod stream_client;
pub mod tcp;
pub mod ws;

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use axum::Router;
use tokio::sync::Mutex;

use rvf_runtime::{RvfOptions, RvfStore};

use crate::auth::AuthToken;
use crate::http::SharedStore;

/// Server configuration.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// HTTP listen port.
    pub http_port: u16,
    /// TCP streaming listen port.
    pub tcp_port: u16,
    /// Path to the RVF data file.
    pub data_path: std::path::PathBuf,
    /// Dimension for new stores (only used when creating).
    pub dimension: u16,
    /// HTTP listen address (default `127.0.0.1`). A non-loopback address
    /// requires `token`.
    pub host: IpAddr,
    /// Bearer token required on every HTTP route except `/v1/health`.
    pub token: Option<AuthToken>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            http_port: 8080,
            tcp_port: 9090,
            data_path: std::path::PathBuf::from("data.rvf"),
            dimension: 128,
            host: IpAddr::V4(Ipv4Addr::LOCALHOST),
            token: None,
        }
    }
}

impl ServerConfig {
    /// Refuse a configuration that would expose the store without
    /// authentication.
    pub fn validate(&self) -> Result<(), String> {
        if !self.host.is_loopback() && self.token.is_none() {
            return Err(format!(
                "refusing to bind {} without a token: set {} (or bind a loopback address)",
                self.host,
                auth::TOKEN_ENV
            ));
        }
        Ok(())
    }

    /// Address the HTTP listener binds.
    pub fn http_addr(&self) -> SocketAddr {
        SocketAddr::new(self.host, self.http_port)
    }

    /// Address the TCP listener binds: the configured host when it is
    /// loopback, otherwise loopback of the same family, since the binary
    /// protocol cannot carry the token.
    pub fn tcp_addr(&self) -> SocketAddr {
        let host = match self.host {
            h if h.is_loopback() => h,
            IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::LOCALHOST),
        };
        SocketAddr::new(host, self.tcp_port)
    }
}

/// Open or create the store at the configured path, returning a shared handle.
pub fn open_or_create_store(config: &ServerConfig) -> Result<SharedStore, rvf_types::RvfError> {
    let path = &config.data_path;

    let store = if path.exists() {
        RvfStore::open(path)?
    } else {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .map_err(|_| rvf_types::RvfError::Code(rvf_types::ErrorCode::FsyncFailed))?;
            }
        }
        let options = RvfOptions {
            dimension: config.dimension,
            ..Default::default()
        };
        RvfStore::create(path, options)?
    };

    Ok(Arc::new(Mutex::new(store)))
}

/// The complete HTTP application: REST, dashboard and (when configured) the
/// verified-stream routes, behind the bearer-token check when `token` is set.
pub fn build_router(
    store: SharedStore,
    events: ws::EventSender,
    stream: Option<Arc<stream::StreamServer>>,
    token: Option<AuthToken>,
) -> Router {
    let mut app = http::router_with_static(store, events, None);
    if let Some(s) = stream {
        app = app.merge(stream::router(s));
    }
    auth::protect(app, token)
}

/// Start both HTTP and TCP servers. Blocks until shutdown.
pub async fn run(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    run_with_stream(config, None).await
}

/// [`run`] plus the verified-stream endpoints (HTTP and TCP `0x05`/`0x85`)
/// when `stream` is configured.
pub async fn run_with_stream(
    config: ServerConfig,
    stream: Option<stream::StreamServerConfig>,
) -> Result<(), Box<dyn std::error::Error>> {
    config.validate()?;
    let store = open_or_create_store(&config).map_err(|e| format!("failed to open store: {e}"))?;

    let http_addr = config.http_addr();
    let tcp_addr = config.tcp_addr();

    let (event_tx, _rx) = ws::event_channel();
    let stream_srv = match stream {
        Some(cfg) => Some(stream::StreamServer::load(cfg)?),
        None => None,
    };
    let app = build_router(
        store.clone(),
        event_tx,
        stream_srv.clone(),
        config.token.clone(),
    );
    let listener = tokio::net::TcpListener::bind(http_addr).await?;
    eprintln!(
        "rvf-server HTTP listening on {http_addr}{}",
        if config.token.is_some() {
            " (bearer token required)"
        } else {
            ""
        }
    );
    eprintln!("rvf-server TCP  listening on {tcp_addr}");

    let tcp_addr = tcp_addr.to_string();
    tokio::select! {
        result = axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()) => {
            result?;
        }
        result = tcp::serve_tcp_with_stream(&tcp_addr, store, stream_srv) => {
            result?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn router(dir: &tempfile::TempDir, token: Option<AuthToken>) -> Router {
        let store = open_or_create_store(&ServerConfig {
            data_path: dir.path().join("auth.rvf"),
            dimension: 4,
            ..Default::default()
        })
        .unwrap();
        let (tx, _rx) = ws::event_channel();
        build_router(store, tx, None, token)
    }

    async fn status(app: &Router, path: &str, auth: Option<&str>) -> StatusCode {
        let mut req = Request::get(path);
        if let Some(a) = auth {
            req = req.header("authorization", a);
        }
        app.clone()
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[test]
    fn default_config_binds_loopback() {
        let c = ServerConfig::default();
        assert!(c.host.is_loopback());
        assert!(c.http_addr().ip().is_loopback());
        assert!(c.tcp_addr().ip().is_loopback());
        assert!(c.validate().is_ok());
    }

    #[test]
    fn non_loopback_bind_requires_a_token() {
        for host in ["0.0.0.0", "::", "192.168.1.10"] {
            let mut c = ServerConfig {
                host: host.parse().unwrap(),
                ..Default::default()
            };
            assert!(c.validate().is_err(), "{host} accepted without a token");
            c.token = AuthToken::new("s3cret");
            assert!(c.validate().is_ok());
            // The unauthenticated binary protocol stays local.
            assert!(c.tcp_addr().ip().is_loopback());
        }
    }

    #[test]
    fn empty_token_is_no_token_and_debug_is_redacted() {
        assert!(AuthToken::new("").is_none());
        assert!(AuthToken::new("   ").is_none());
        let c = ServerConfig {
            token: AuthToken::new("s3cret-value"),
            ..Default::default()
        };
        assert!(!format!("{c:?}").contains("s3cret-value"));
    }

    #[tokio::test]
    async fn token_is_required_on_every_route_but_health() {
        let dir = tempfile::tempdir().unwrap();
        let app = router(&dir, AuthToken::new("s3cret"));
        for path in ["/v1/status", "/api/status", "/", "/no-such-file.wasm"] {
            assert_eq!(status(&app, path, None).await, StatusCode::UNAUTHORIZED);
            assert_eq!(
                status(&app, path, Some("Bearer wrong")).await,
                StatusCode::UNAUTHORIZED
            );
            assert_eq!(
                status(&app, path, Some("s3cret")).await,
                StatusCode::UNAUTHORIZED
            );
        }
        assert_eq!(
            status(&app, "/v1/status", Some("Bearer s3cret")).await,
            StatusCode::OK
        );
        assert_eq!(status(&app, "/v1/health", None).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn no_token_leaves_routes_open() {
        let dir = tempfile::tempdir().unwrap();
        let app = router(&dir, None);
        assert_eq!(status(&app, "/v1/status", None).await, StatusCode::OK);
    }
}
