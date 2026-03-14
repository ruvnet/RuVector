//! ACP server entry point.
//!
//! Parses CLI arguments, loads configuration, and starts the ACP server
//! per ADR-099 and ADR-103 C6.

mod agent;
mod auth;
mod server;
mod types;

use clap::Parser;
use rvagent_core::config::RvAgentConfig;
use tracing_subscriber::EnvFilter;

use crate::agent::AcpAgent;
use crate::server::{AcpConfig, AcpServer};

/// rvAgent ACP Server — Agent Communication Protocol
#[derive(Parser, Debug)]
#[command(name = "rvagent-acp", version, about)]
struct Cli {
    /// Host address to listen on.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on.
    #[arg(short, long, default_value_t = 3100)]
    port: u16,

    /// API key for Bearer authentication. If omitted, auth is disabled.
    #[arg(long, env = "RVAGENT_ACP_API_KEY")]
    api_key: Option<String>,

    /// Maximum requests per minute per IP.
    #[arg(long, default_value_t = 60)]
    rate_limit: u32,

    /// Maximum request body size in bytes.
    #[arg(long, default_value_t = 1_048_576)]
    max_body_size: usize,

    /// Require TLS for non-localhost connections.
    #[arg(long)]
    require_tls: bool,

    /// Model to use (provider:model format).
    #[arg(short, long)]
    model: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Build agent configuration.
    let mut agent_config = RvAgentConfig::default();
    if let Some(model) = &cli.model {
        agent_config.model = model.clone();
    }

    // Build server configuration.
    let server_config = AcpConfig {
        host: cli.host,
        port: cli.port,
        api_key: cli.api_key,
        rate_limit: cli.rate_limit,
        max_body_size: cli.max_body_size,
        require_tls: cli.require_tls,
    };

    let agent = AcpAgent::new(agent_config);
    let server = AcpServer::new(agent, server_config);

    tracing::info!("Starting rvAgent ACP server");
    server.serve().await
}
