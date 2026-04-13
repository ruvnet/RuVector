//! ruvbrain-worker: batch worker for Cloud Run Jobs
//!
//! Runs scheduler actions (train, drift_check, transfer_all, rebuild_graph,
//! cleanup, attractor_analysis) as a one-shot CLI, then exits.
//!
//! Reads WORKER_ACTIONS env var (comma-separated) to select actions.
//! If unset, runs all actions. Reuses the same AppState as the API server.

use mcp_brain_server::routes;
use mcp_brain_server::types::AppState;
use mcp_brain_server::midstream;
use ruvector_domain_expansion::DomainId;
use std::collections::{HashMap, HashSet};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

const ALL_ACTIONS: &[&str] = &[
    "train",
    "drift_check",
    "transfer_all",
    "rebuild_graph",
    "cleanup",
    "attractor_analysis",
];

#[tokio::main]
async fn main() {
    // ── Tracing ──
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .init();

    tracing::info!("ruvbrain-worker starting");

    // ── Build AppState (reuse create_router; we discard the Router) ──
    let (_router, state) = routes::create_router().await;
    tracing::info!("AppState initialised");

    // ── Parse requested actions ──
    let actions: Vec<&str> = match std::env::var("WORKER_ACTIONS") {
        Ok(val) if !val.is_empty() => {
            let requested: Vec<String> = val.split(',').map(|s| s.trim().to_string()).collect();
            // Validate action names
            for a in &requested {
                if !ALL_ACTIONS.contains(&a.as_str()) {
                    tracing::error!("Unknown action: {a}");
                    std::process::exit(1);
                }
            }
            tracing::info!("WORKER_ACTIONS={val}");
            // Leak into 'static so we can collect &str — fine for a one-shot CLI
            requested
                .into_iter()
                .map(|s| &*Box::leak(s.into_boxed_str()) as &str)
                .collect()
        }
        _ => {
            tracing::info!("WORKER_ACTIONS unset — running all actions");
            ALL_ACTIONS.to_vec()
        }
    };

    // ── Execute each action ──
    let total_start = std::time::Instant::now();
    let mut any_failure = false;

    for action in &actions {
        let action_start = std::time::Instant::now();
        let (success, message) = run_action(action, &state);
        let elapsed_ms = action_start.elapsed().as_millis();

        if success {
            tracing::info!(action, elapsed_ms, message, "action completed");
        } else {
            tracing::error!(action, elapsed_ms, message, "action failed");
            any_failure = true;
        }
    }

    let total_ms = total_start.elapsed().as_millis();
    tracing::info!(total_ms, actions = ?actions, "worker finished");

    if any_failure {
        std::process::exit(1);
    }
}

/// Run a single pipeline action against the shared AppState.
/// Returns (success, human-readable message).
fn run_action(action: &str, state: &AppState) -> (bool, String) {
    match action {
        "train" => {
            let result = routes::run_training_cycle(state);
            (
                true,
                format!(
                    "Training complete: sona_patterns={}, pareto={}->{}",
                    result.sona_patterns, result.pareto_before, result.pareto_after,
                ),
            )
        }

        "drift_check" => {
            let drift = state.drift.read();
            let report = drift.compute_drift(None);
            (
                true,
                format!(
                    "Drift check: drifting={}, cv={:.4}, trend={}",
                    report.is_drifting, report.coefficient_of_variation, report.trend,
                ),
            )
        }

        "transfer_all" => {
            let categories: Vec<String> = {
                let all_mems = state.store.all_memories();
                let mut cats: HashSet<String> = HashSet::new();
                for m in &all_mems {
                    cats.insert(m.category.to_string());
                }
                cats.into_iter().collect()
            };
            let mut transfers = 0usize;
            let mut engine = state.domain_engine.write();
            for i in 0..categories.len() {
                for j in (i + 1)..categories.len() {
                    engine.initiate_transfer(
                        &DomainId(categories[i].clone()),
                        &DomainId(categories[j].clone()),
                    );
                    transfers += 1;
                }
            }
            (
                true,
                format!(
                    "Domain transfers initiated: {transfers} pairs across {} categories",
                    categories.len()
                ),
            )
        }

        "rebuild_graph" => {
            let all_mems = state.store.all_memories();
            let mut graph = state.graph.write();
            // ADR-149 P3: batch rebuild instead of one-at-a-time add_memory loop
            graph.rebuild_from_batch(&all_mems);
            graph.rebuild_sparsifier();
            (
                true,
                format!(
                    "Graph rebuilt: {} nodes, {} edges",
                    graph.node_count(),
                    graph.edge_count()
                ),
            )
        }

        "cleanup" => {
            if state.rvf_flags.sona_enabled {
                let _ = state.sona.read().tick();
            }
            (true, "Cleanup complete".into())
        }

        "attractor_analysis" => {
            if state.rvf_flags.midstream_attractor {
                let all_mems = state.store.all_memories();
                let mut categories: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
                for m in &all_mems {
                    categories
                        .entry(m.category.to_string())
                        .or_default()
                        .push(m.embedding.clone());
                }
                let mut analyzed = 0usize;
                for (cat, embeddings) in &categories {
                    if let Some(result) = midstream::analyze_category_attractor(embeddings) {
                        state.attractor_results.write().insert(cat.clone(), result);
                        analyzed += 1;
                    }
                }
                (
                    true,
                    format!(
                        "Attractor analysis: {analyzed}/{} categories analyzed",
                        categories.len()
                    ),
                )
            } else {
                (false, "Midstream attractor feature not enabled".into())
            }
        }

        other => (false, format!("Unknown action: {other}")),
    }
}
