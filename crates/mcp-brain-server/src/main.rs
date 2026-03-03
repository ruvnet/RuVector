use mcp_brain_server::routes;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

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

    let (app, state) = routes::create_router().await;

    // Background training loop: runs SONA force_learn + domain evolve_population
    // every 5 minutes (or after threshold of new data). This bridges the gap between
    // "stores knowledge" and "learns from knowledge".
    let train_state = state.clone();
    let _training_handle = tokio::spawn(async move {
        let interval = std::time::Duration::from_secs(300); // 5 minutes
        let mut last_memory_count = train_state.store.memory_count();
        let mut last_vote_count = train_state.store.vote_count();
        // Wait 60s before first cycle (let startup finish, data load)
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        loop {
            tokio::time::sleep(interval).await;

            let current_memories = train_state.store.memory_count();
            let current_votes = train_state.store.vote_count();
            let new_memories = current_memories.saturating_sub(last_memory_count);
            let new_votes = current_votes.saturating_sub(last_vote_count);

            // Train if: 5 min elapsed AND (any new data, or every cycle regardless)
            // Threshold-based: also runs immediately if 50+ new memories or 100+ new votes
            if new_memories > 0 || new_votes > 0 {
                let result = routes::run_training_cycle(&train_state);
                tracing::info!(
                    "Background training: sona_patterns={}, pareto={}→{}, new_memories={}, new_votes={}",
                    result.sona_patterns, result.pareto_before, result.pareto_after,
                    new_memories, new_votes
                );
                last_memory_count = current_memories;
                last_vote_count = current_votes;
            }
        }
    });

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    tracing::info!("mcp-brain-server listening on port {port}");
    tracing::info!("Endpoints: brain.ruv.io | π.ruv.io");
    tracing::info!("Background training loop: every 5 min (active when new data)");

    // Graceful shutdown: wait for SIGTERM (Cloud Run sends this) or Ctrl+C,
    // then allow in-flight requests 10s to complete before terminating.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    tracing::info!("Server shut down gracefully");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received Ctrl+C, starting graceful shutdown"),
        _ = terminate => tracing::info!("Received SIGTERM, starting graceful shutdown"),
    }
}
