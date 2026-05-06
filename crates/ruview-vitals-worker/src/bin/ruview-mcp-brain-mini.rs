//! `ruview-mcp-brain-mini` — minimal HTTP brain for the cognitum
//! cluster (ADR-183 Tier 2 iter 8).
//!
//! Wire-compatible with the existing `mcp-brain-serve` REST shape so
//! both [`ruview_vitals_worker::brain::BrainClient`] and RuView's own
//! `brain_bridge.rs` POST into it without code change. Most of the
//! behaviour lives in the [`ruview_vitals_worker::mcp_brain`] lib
//! module so integration tests can spin it up in-process.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

use ruview_vitals_worker::mcp_brain::{build_app, Store, DEFAULT_BODY_LIMIT_BYTES};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let bind = std::env::var("RUVIEW_BRAIN_BIND")
        .unwrap_or_else(|_| "0.0.0.0:9876".to_string());
    let store_path = std::env::var("RUVIEW_BRAIN_STORE_PATH")
        .ok()
        .map(PathBuf::from);
    let body_limit_bytes = std::env::var("RUVIEW_BRAIN_BODY_LIMIT_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BODY_LIMIT_BYTES);

    let store = Arc::new(RwLock::new(Store::load(store_path)));
    let app = build_app(store.clone(), body_limit_bytes);

    let listener = TcpListener::bind(&bind).await?;
    tracing::info!(
        addr = %listener.local_addr()?,
        memories = store.read().await.memories.len(),
        body_limit_bytes,
        "ruview-mcp-brain-mini up"
    );
    axum::serve(listener, app).await?;
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_env("RUVIEW_BRAIN_LOG")
        .or_else(|_| EnvFilter::try_new("info"))
        .expect("default tracing filter");
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();
}
