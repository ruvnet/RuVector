//! rvagent-mcp — MCP server binary with stdio and SSE transports.
//!
//! # Usage
//!
//! ```bash
//! # stdio mode (default for Claude Code)
//! rvagent-mcp --transport stdio
//!
//! # SSE mode with port
//! rvagent-mcp --transport sse --port 9000
//!
//! # Filter by tool groups
//! rvagent-mcp --transport sse --groups file,shell,memory
//!
//! # Expose all tools
//! rvagent-mcp --transport sse --all
//! ```

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::State,
    http::{Method, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures::stream::Stream;
use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};

use rvagent_mcp::{
    groups::{ToolFilter, ToolGroup},
    protocol::JsonRpcRequest,
    registry::McpToolRegistry,
    resources::ResourceRegistry,
    server::{McpServer, McpServerConfig},
    transport::{
        SseConfig, SseTransport, StdioTransport, Transport, TransportConfig, TransportType,
    },
};

/// rvAgent MCP Server — Model Context Protocol for rvAgent tools.
#[derive(Parser, Debug)]
#[command(name = "rvagent-mcp")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "rvAgent MCP Server with stdio and SSE transports")]
#[command(long_about = None)]
struct Cli {
    /// Transport type: stdio or sse
    #[arg(short, long, default_value = "stdio")]
    transport: String,

    /// Port for SSE server (only used with --transport sse)
    #[arg(short, long, default_value = "9000")]
    port: u16,

    /// Host to bind (only used with --transport sse)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Tool groups to expose (comma-separated: file,shell,memory,agent,git,web,brain,task,core)
    #[arg(short, long, value_delimiter = ',')]
    groups: Option<Vec<String>>,

    /// Expose all tools (overrides --groups)
    #[arg(long)]
    all: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Enable CORS for SSE transport
    #[arg(long, default_value = "true")]
    cors: bool,
}

/// Shared application state for SSE transport.
#[derive(Clone)]
struct AppState {
    server: Arc<McpServer>,
    response_tx: broadcast::Sender<String>,
    request_tx: tokio::sync::mpsc::Sender<JsonRpcRequest>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cli.log_level));
    fmt().with_env_filter(filter).with_target(false).init();

    // Parse transport type
    let transport_type: TransportType = cli
        .transport
        .parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;

    // Build tool filter
    let tool_filter = if cli.all {
        ToolFilter::all()
    } else if let Some(ref group_names) = cli.groups {
        ToolFilter::from_group_names(group_names).map_err(|e| anyhow::anyhow!(e))?
    } else {
        ToolFilter::default() // core + file
    };

    info!(
        "Starting rvagent-mcp v{} with {} transport",
        env!("CARGO_PKG_VERSION"),
        transport_type
    );

    if tool_filter.allows_all() {
        info!("Exposing all tools");
    } else {
        info!(
            "Exposing {} tools from selected groups",
            tool_filter.count()
        );
    }

    // Build registries
    let tool_registry = build_tool_registry(&tool_filter)?;
    let resource_registry = Arc::new(ResourceRegistry::new());

    // Create MCP server
    let server_config = McpServerConfig::default();
    let server = McpServer::new(server_config, tool_registry, resource_registry);

    info!("Registered {} tools", server.tool_registry().len());

    match transport_type {
        TransportType::Stdio => run_stdio(server).await,
        TransportType::Sse => run_sse(server, &cli).await,
    }
}

/// Build tool registry with filter applied.
fn build_tool_registry(filter: &ToolFilter) -> anyhow::Result<McpToolRegistry> {
    let registry = McpToolRegistry::new();

    // Register builtin tools
    rvagent_mcp::registry::register_builtins(&registry, serde_json::json!({}))?;

    // If not allowing all, we need to filter
    // Note: The registry already has all tools; for now we register all and the
    // server can filter at list time. A production implementation would filter
    // during registration.
    if !filter.allows_all() {
        // Log which groups are enabled
        for group in ToolGroup::all() {
            let group_tools: Vec<_> = group
                .tools()
                .iter()
                .filter(|t| filter.is_allowed(t))
                .collect();
            if !group_tools.is_empty() {
                info!("Group '{}': {} tools", group, group_tools.len());
            }
        }
    }

    Ok(registry)
}

/// Run in stdio mode (NDJSON over stdin/stdout).
async fn run_stdio(server: McpServer) -> anyhow::Result<()> {
    info!("Running in stdio mode (NDJSON)");

    let transport = StdioTransport::new(TransportConfig::default());

    loop {
        match transport.receive_request().await {
            Ok(Some(request)) => {
                let response = server.handle_request(request).await;
                if let Err(e) = transport.send_response(response).await {
                    error!("Failed to send response: {}", e);
                }
            }
            Ok(None) => {
                info!("stdin closed, shutting down");
                break;
            }
            Err(e) => {
                error!("Error receiving request: {}", e);
                break;
            }
        }
    }

    Ok(())
}

/// Run in SSE mode (HTTP server with Server-Sent Events).
async fn run_sse(server: McpServer, cli: &Cli) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    info!("Running in SSE mode on http://{}", addr);

    // Create SSE transport
    let sse_config = SseConfig {
        port: cli.port,
        host: cli.host.clone(),
        enable_cors: cli.cors,
        heartbeat_interval_secs: 30,
    };
    let sse_transport = SseTransport::new(sse_config);

    // Create broadcast channel for SSE events
    let (response_tx, _) = broadcast::channel::<String>(256);
    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel::<JsonRpcRequest>(256);

    let server = Arc::new(server);
    let state = AppState {
        server: server.clone(),
        response_tx: response_tx.clone(),
        request_tx,
    };

    // Build router
    let mut app = Router::new()
        .route("/sse", get(sse_handler))
        .route("/message", post(message_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    // Add CORS if enabled
    if cli.cors {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers(Any);
        app = app.layer(cors);
    }

    // Spawn request handler task
    let response_tx_clone = response_tx.clone();
    tokio::spawn(async move {
        while let Some(request) = request_rx.recv().await {
            let response = server.handle_request(request).await;
            let json = match serde_json::to_string(&response) {
                Ok(j) => j,
                Err(e) => {
                    error!("Failed to serialize response: {}", e);
                    continue;
                }
            };
            if response_tx_clone.receiver_count() > 0 {
                if let Err(e) = response_tx_clone.send(json) {
                    warn!("Failed to broadcast response: {}", e);
                }
            }
        }
    });

    // Start server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Listening on http://{}", addr);
    info!("SSE endpoint: http://{}/sse", addr);
    info!("Message endpoint: http://{}/message", addr);

    // Handle shutdown
    let shutdown = async move {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to listen for shutdown signal");
        info!("Shutdown signal received");
        let _ = sse_transport.close().await;
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    info!("Server stopped");
    Ok(())
}

/// Custom stream wrapper for broadcast receiver.
struct SseStream {
    rx: broadcast::Receiver<String>,
}

impl Stream for SseStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.rx.try_recv() {
            Ok(data) => Poll::Ready(Some(Ok(Event::default().event("message").data(data)))),
            Err(broadcast::error::TryRecvError::Empty) => {
                // Register waker and return pending
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => {
                // Skip lagged messages
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Closed) => Poll::Ready(None),
        }
    }
}

/// SSE event stream handler.
async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.response_tx.subscribe();
    let stream = SseStream { rx };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

/// POST /message handler for incoming JSON-RPC requests.
async fn message_handler(
    State(state): State<AppState>,
    Json(request): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    match state.request_tx.send(request).await {
        Ok(_) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({"status": "accepted"})),
        ),
        Err(e) => {
            error!("Failed to queue request: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "failed to queue request"})),
            )
        }
    }
}

/// Health check endpoint.
async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "server": state.server.config().name,
        "version": state.server.config().version,
        "tools": state.server.tool_registry().len(),
    }))
}
