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

    // Start server IMMEDIATELY — data loads in background.
    // This ensures health/ready endpoints respond during Firestore hydration.
    let (app, state) = routes::create_router().await;
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    tracing::info!("mcp-brain-server listening on port {port} (data loading in background)");

    // ── Enhanced cognitive loop (replaces basic training loop) ──
    // Runs every 5 min: SONA + symbolic reasoning + internal voice + curiosity +
    // strange-loop self-awareness + GWT workspace broadcast + LoRA federation.
    // Also runs a lightweight cognitive tick every 60s for GWT + curiosity.
    let train_state = state.clone();
    let _training_handle = tokio::spawn(async move {
        let train_interval = std::time::Duration::from_secs(300); // 5 min: full enhanced cycle
        let tick_interval = std::time::Duration::from_secs(60); // 60s: lightweight cognitive tick
        let mut tick_count = 0u64;

        // Wait 30s before first cycle (let startup finish, data load)
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;

        // Run an initial enhanced cycle on startup to bootstrap cognitive state (full retrain)
        let result = routes::run_enhanced_training_cycle(&train_state, true);
        tracing::info!(
            "Initial cognitive bootstrap: props={}, inferences={}, voice={}, curiosity={}, strange_loop={:.4}",
            result.propositions_extracted, result.inferences_derived,
            result.voice_thoughts, result.curiosity_triggered, result.strange_loop_score
        );
        let mut last_memory_count = train_state.store.memory_count();
        let mut last_vote_count = train_state.store.vote_count();

        loop {
            tokio::time::sleep(tick_interval).await;
            tick_count += 1;

            // ── Lightweight cognitive tick (every 60s) ──
            // 1. GWT workspace: broadcast top memory as a representation
            //    This keeps the workspace active with fresh content
            {
                let memories = train_state.store.recent_memories(1);
                if let Some(mem) = memories.first() {
                    if !mem.embedding.is_empty() {
                        let rep = ruvector_nervous_system::routing::workspace::WorkspaceItem::new(
                            mem.embedding.clone(),
                            mem.quality_score.mean() as f32,
                            0, // source module: brain-core
                            tick_count,
                        );
                        train_state.workspace.write().broadcast(rep);
                    }
                }
                // Run competitive dynamics to decay stale items
                train_state.workspace.write().compete();
            }

            // 2. Curiosity: record a visit for the most recently active category
            {
                let memories = train_state.store.recent_memories(5);
                let mut cats = std::collections::HashMap::new();
                for m in &memories {
                    *cats.entry(m.category.to_string()).or_insert(0u32) += 1;
                }
                if let Some((top_cat, _)) = cats.iter().max_by_key(|(_, c)| *c) {
                    let bucket = ruvector_domain_expansion::transfer::ContextBucket {
                        difficulty_tier: "medium".to_string(),
                        category: top_cat.clone(),
                    };
                    let arm = ruvector_domain_expansion::transfer::ArmId(top_cat.clone());
                    train_state
                        .domain_engine
                        .write()
                        .meta
                        .curiosity
                        .record_visit(&bucket, &arm);
                }
            }

            // 3. Internal voice: periodic observation about brain state
            if tick_count % 5 == 0 {
                let mem_count = train_state.store.memory_count();
                let ws_load = train_state.workspace.read().current_load();
                train_state.internal_voice.write().observe(
                    format!(
                        "tick {}: {} memories, GWT load {:.2}",
                        tick_count, mem_count, ws_load
                    ),
                    uuid::Uuid::nil(),
                );
            }

            // ── Full enhanced training cycle (every 5 min = every 5th tick) ──
            if tick_count % 5 == 0 {
                let current_memories = train_state.store.memory_count();
                let current_votes = train_state.store.vote_count();
                let new_memories = current_memories.saturating_sub(last_memory_count);
                let new_votes = current_votes.saturating_sub(last_vote_count);

                // Run enhanced cycle if there's new data, or every 3rd full cycle regardless
                // (keeps curiosity + self-reflection active even during quiet periods)
                // ADR-149 P4: The incremental filter inside run_enhanced_training_cycle
                // handles skipping unchanged memories automatically. Pass force_full=false
                // to benefit from incremental processing; the function auto-forces a full
                // retrain every 24h.
                if new_memories > 0 || new_votes > 0 || tick_count % 15 == 0 {
                    let result = routes::run_enhanced_training_cycle(&train_state, false);
                    tracing::info!(
                        "Cognitive cycle #{} ({}): props={}, inferences={}, voice={}, auto_votes={}, \
                         curiosity={}, sona_patterns={}, strange_loop={:.4}, lora_auto={}, processed={}/{}",
                        tick_count / 5,
                        if result.was_full_retrain { "full" } else { "incremental" },
                        result.propositions_extracted, result.inferences_derived,
                        result.voice_thoughts, result.auto_votes,
                        result.curiosity_triggered, result.sona_patterns,
                        result.strange_loop_score, result.lora_auto_submitted,
                        result.memories_processed, result.memory_count
                    );
                    last_memory_count = current_memories;
                    last_vote_count = current_votes;
                }
            }
        }
    });

    // ── Background sparsifier build for large graphs ──
    // Deferred from startup to avoid blocking the health probe.
    // For very large graphs (>5M edges), skip the sparsifier entirely — it
    // holds a write lock that blocks all readers and pegs the CPU, causing
    // Cloud Run to 504 every request while it runs.
    let spar_state = state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        let edge_count = spar_state.graph.read().edge_count();
        if edge_count > 5_000_000 {
            tracing::info!(
                "Skipping sparsifier build: graph too large ({edge_count} edges, >5M threshold)"
            );
        } else if edge_count > 100_000 && spar_state.graph.read().sparsifier_stats().is_none() {
            tracing::info!("Background sparsifier build starting ({edge_count} edges)");
            // Run in spawn_blocking to avoid starving the tokio runtime
            let graph = spar_state.graph.clone();
            tokio::task::spawn_blocking(move || {
                graph.write().rebuild_sparsifier();
            })
            .await
            .ok();
            let stats = spar_state.graph.read().sparsifier_stats();
            if let Some(s) = stats {
                tracing::info!(
                    "Sparsifier built: {} edges, {:.1}x compression",
                    s.sparsified_edges,
                    s.compression_ratio
                );
            } else {
                tracing::warn!("Sparsifier build returned no stats");
            }
        }
    });

    tracing::info!("Endpoints: brain.ruv.io | π.ruv.io");
    tracing::info!("Cognitive loop: tick every 60s, full cycle every 5 min");

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
