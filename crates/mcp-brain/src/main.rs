//! MCP Brain server binary
//!
//! Runs the MCP Brain server on stdio for integration with Claude Code.

use mcp_brain::McpBrainServer;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .init();

    let server = McpBrainServer::new();

    tracing::info!("MCP Brain server v{} starting", env!("CARGO_PKG_VERSION"));
    tracing::info!("Backend: brain.ruv.io");

    server.run_stdio().await?;

    Ok(())
}
