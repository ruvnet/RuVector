//! Async orchestration for the off-lock rebuild (ADR-349 item 6): the
//! single-flight guard, the detached task, and the follow-on sparsifier.

use super::{default_build_threads, InstallReport, KnowledgeGraph, RebuildTicket};
use crate::types::BrainMemory;
use parking_lot::RwLock;
use std::sync::Arc;

/// Result of [`rebuild_off_lock`].
#[derive(Debug)]
pub enum RebuildOutcome {
    Installed(InstallReport),
    /// Another rebuild was already in flight (single-flight); nothing done.
    AlreadyRunning,
    /// The rebuild was cancelled while building (an in-place
    /// `rebuild_from_batch` ran meanwhile); the stale build was discarded.
    Superseded,
    /// The build task panicked; the in-flight marker was released.
    Failed(String),
}

/// Releases the single-flight marker if the rebuild ends any way other than
/// a successful install (panic in the build, task cancellation).
struct AbortOnDrop {
    graph: Arc<RwLock<KnowledgeGraph>>,
    ticket: RebuildTicket,
    armed: bool,
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if self.armed && self.graph.write().abort_rebuild(self.ticket) {
            tracing::warn!("Graph rebuild abandoned before install; in-flight marker released");
        }
    }
}

/// Rebuild the graph without holding its lock across the O(n²) pass.
///
/// `load` produces the memory snapshot; it runs on the blocking pool too,
/// because cloning ~60k memories is itself a stall on a 2-worker runtime.
/// It is invoked *after* the mutation log is armed, so anything added to the
/// graph from then on is either in the snapshot or replayed at install.
///
/// `sparsifier_max_edges`: if `Some(cap)` and the new graph has at most `cap`
/// edges, the sparsifier is rebuilt off-lock afterwards via the ADR-349 P3
/// snapshot → build → guarded-install path.
///
/// Prefer [`spawn_rebuild`] from request handlers, so a client disconnect
/// cannot cancel a half-finished rebuild.
pub async fn rebuild_off_lock<F>(
    graph: Arc<RwLock<KnowledgeGraph>>,
    load: F,
    sparsifier_max_edges: Option<usize>,
) -> RebuildOutcome
where
    F: FnOnce() -> Vec<BrainMemory> + Send + 'static,
{
    let Some(ticket) = graph.write().begin_rebuild() else {
        tracing::info!("Graph rebuild already in progress; skipping");
        return RebuildOutcome::AlreadyRunning;
    };
    let mut guard = AbortOnDrop {
        graph: graph.clone(),
        ticket,
        armed: true,
    };

    let threads = default_build_threads();
    let built = tokio::task::spawn_blocking(move || {
        let memories = load();
        KnowledgeGraph::build_batch(&memories, ticket.threshold, threads)
    })
    .await;
    let build = match built {
        Ok(b) => b,
        // `guard` drops here and releases the marker.
        Err(e) => return RebuildOutcome::Failed(e.to_string()),
    };

    let installed = graph.write().install_batch(ticket, build);
    guard.armed = false;
    let Some((report, retired)) = installed else {
        tracing::warn!("Graph rebuild superseded while building; discarded");
        return RebuildOutcome::Superseded;
    };
    drop(retired); // outside the write lock
    tracing::info!(
        nodes = report.nodes,
        edges = report.edges,
        build_ms = report.build.elapsed.as_millis() as u64,
        threads = report.build.threads,
        install_ms = report.install_elapsed.as_millis() as u64,
        replayed_adds = report.replayed_adds,
        replayed_removes = report.replayed_removes,
        "Graph rebuilt from batch (ADR-149 P3)"
    );

    if let Some(cap) = sparsifier_max_edges {
        if report.edges <= cap {
            rebuild_sparsifier_off_lock(&graph).await;
        } else {
            tracing::info!(
                "Skipping sparsifier build: {} edges exceeds cap {cap}",
                report.edges
            );
        }
    }
    RebuildOutcome::Installed(report)
}

/// Run [`rebuild_off_lock`] as a detached task. Dropping the returned handle
/// does not cancel the rebuild (Cloud Scheduler's HTTP deadline is shorter
/// than a 60k-node build, and a disconnect must not waste or wedge it).
pub fn spawn_rebuild<F>(
    graph: Arc<RwLock<KnowledgeGraph>>,
    load: F,
    sparsifier_max_edges: Option<usize>,
) -> tokio::task::JoinHandle<RebuildOutcome>
where
    F: FnOnce() -> Vec<BrainMemory> + Send + 'static,
{
    tokio::spawn(rebuild_off_lock(graph, load, sparsifier_max_edges))
}

/// Snapshot → build (blocking pool, no lock) → guarded install. Returns
/// whether a sparsifier was installed.
pub async fn rebuild_sparsifier_off_lock(graph: &Arc<RwLock<KnowledgeGraph>>) -> bool {
    let snapshot = graph.read().sparsifier_snapshot();
    let Some((entries, nodes, edges, gen)) = snapshot else {
        return false;
    };
    let built =
        tokio::task::spawn_blocking(move || KnowledgeGraph::build_sparsifier_from(&entries, nodes))
            .await
            .ok()
            .flatten();
    match built {
        Some(spar) => graph.write().install_sparsifier(spar, nodes, edges, gen),
        None => false,
    }
}
