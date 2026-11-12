//! Off-lock full graph rebuild (ADR-349 item 6).
//!
//! `rebuild_from_batch` used to run its O(n²) all-pairs similarity pass
//! (~1.8B cosines at the live 59,758 nodes) with `graph.write()` held, so every
//! search/status request queued behind it until Cloud Run's timeout. This
//! module splits the rebuild the same way ADR-349 P3 split the sparsifier:
//!
//! 1. [`KnowledgeGraph::begin_rebuild`] — brief write lock. Single-flight: a
//!    second begin is refused while one is in flight. Arms a mutation log.
//! 2. [`KnowledgeGraph::build_batch`] — associated fn over plain data, run in
//!    `spawn_blocking` with **no graph lock held**. Readers keep using the old
//!    graph; writers keep mutating it, and every `add_memory` /
//!    `remove_memory` is recorded in the log.
//! 3. [`KnowledgeGraph::install_batch`] — brief write lock. Swaps the new
//!    adjacency in, bumps `index_generation` (so any in-flight sparsifier
//!    build refuses to install), and replays the logged mutations onto it.
//!
//! The edge pass is exact: same pairs, same order, bitwise-identical weights
//! (norms are precomputed with the identical accumulation order that
//! `cosine_similarity` uses, so `dot / (|a|·|b|)` is the same f64).

use super::{cosine_similarity, GraphEdge, GraphNode, KnowledgeGraph};
use crate::types::{BrainCategory, BrainMemory};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// ADR-149 P2: nodes below this mean quality get no edges.
const EDGE_QUALITY_FLOOR: f64 = 0.01;

/// Below this many edge-eligible nodes the pass runs on the calling thread;
/// spawning workers costs more than it saves.
const PARALLEL_MIN_NODES: usize = 4096;

/// Upper bound for the post-hydration sparsifier build, matching the existing
/// `> 5_000_000` skip in `main.rs`'s background sparsifier task.
pub const SPARSIFIER_MAX_EDGES: usize = 5_000_000;

/// One mutation applied to the live graph while a rebuild was in flight.
pub(super) enum Mutation {
    Add {
        id: Uuid,
        embedding: Vec<f32>,
        category: BrainCategory,
        quality: f64,
    },
    Remove(Uuid),
}

/// Mutations recorded between `begin_rebuild` and `install_batch`.
pub(super) struct RebuildLog {
    ticket: u64,
    entries: Vec<Mutation>,
}

impl RebuildLog {
    pub(super) fn record_add(&mut self, id: Uuid, node: &GraphNode) {
        self.entries.push(Mutation::Add {
            id,
            embedding: node.embedding.clone(),
            category: node.category.clone(),
            quality: node.quality,
        });
    }

    pub(super) fn record_remove(&mut self, id: Uuid) {
        self.entries.push(Mutation::Remove(id));
    }
}

/// Handle for one in-flight rebuild. Only the matching ticket can install or
/// abort it.
#[derive(Debug, Clone, Copy)]
pub struct RebuildTicket {
    id: u64,
    threshold: f64,
}

impl RebuildTicket {
    /// The similarity threshold captured at `begin_rebuild`.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// Timing and shape of one edge pass.
#[derive(Debug, Clone, Copy)]
pub struct BuildStats {
    pub elapsed: Duration,
    pub threads: usize,
    pub pairs_compared: u64,
}

/// A fully built graph, not yet installed. Plain data: holds no lock.
pub struct BatchBuild {
    nodes: HashMap<Uuid, GraphNode>,
    node_ids: Vec<Uuid>,
    node_index: HashMap<Uuid, usize>,
    edges: Vec<GraphEdge>,
    stats: BuildStats,
}

impl BatchBuild {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
    pub fn stats(&self) -> BuildStats {
        self.stats
    }
}

/// The structures replaced by an install. Returned so the caller can drop
/// them (60k embedding allocations, a >1M-entry edge Vec) *after* releasing
/// the write lock rather than inside it.
pub struct Retired {
    _nodes: HashMap<Uuid, GraphNode>,
    _node_ids: Vec<Uuid>,
    _node_index: HashMap<Uuid, usize>,
    _edges: Vec<GraphEdge>,
    _mincut: Option<ruvector_mincut::DynamicMinCut>,
    _sparsifier: Option<ruvector_sparsifier::AdaptiveGeoSpar>,
}

/// What an install did.
#[derive(Debug, Clone, Copy)]
pub struct InstallReport {
    pub nodes: usize,
    pub edges: usize,
    pub replayed_adds: usize,
    pub replayed_removes: usize,
    pub build: BuildStats,
    /// Time the install held the write lock (swap + replay).
    pub install_elapsed: Duration,
}

/// Worker threads for the edge pass: `GRAPH_REBUILD_THREADS` if set, else
/// all cores but one, so at least one core is left for request serving
/// (Cloud Run runs `ruvbrain` at `--cpu=2`, i.e. one build thread).
pub fn default_build_threads() -> usize {
    if let Some(n) = std::env::var("GRAPH_REBUILD_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

/// `sqrt(Σ a²)` with exactly the accumulation order `cosine_similarity` uses.
#[inline]
fn split_norm(a: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let (mut n0, mut n1) = (0.0f64, 0.0f64);
    for c in 0..chunks {
        let i = c * 4;
        let (a0, a1, a2, a3) = (
            a[i] as f64,
            a[i + 1] as f64,
            a[i + 2] as f64,
            a[i + 3] as f64,
        );
        n0 += a0 * a0 + a2 * a2;
        n1 += a1 * a1 + a3 * a3;
    }
    for i in (chunks * 4)..n {
        let ai = a[i] as f64;
        n0 += ai * ai;
    }
    (n0 + n1).sqrt()
}

/// `Σ a·b` with exactly the accumulation order `cosine_similarity` uses.
#[inline]
fn split_dot(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let (mut d0, mut d1) = (0.0f64, 0.0f64);
    for c in 0..chunks {
        let i = c * 4;
        d0 += (a[i] as f64) * (b[i] as f64) + (a[i + 2] as f64) * (b[i + 2] as f64);
        d1 += (a[i + 1] as f64) * (b[i + 1] as f64) + (a[i + 3] as f64) * (b[i + 3] as f64);
    }
    for i in (chunks * 4)..n {
        d0 += (a[i] as f64) * (b[i] as f64);
    }
    d0 + d1
}

/// Bitwise-identical to `cosine_similarity(a, b)` given `na = split_norm(a)`
/// and `nb = split_norm(b)`; the per-pair norm work is hoisted out.
#[inline]
fn cosine_prenormed(a: &[f32], na: f64, b: &[f32], nb: f64) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    split_dot(a, b) / (na * nb)
}

/// Edges for rows `rows` of the upper triangle over `eligible`.
fn edge_rows(
    rows: std::ops::Range<usize>,
    eligible: &[usize],
    embs: &[&[f32]],
    norms: &[f64],
    memories: &[BrainMemory],
    threshold: f64,
) -> Vec<GraphEdge> {
    let m = eligible.len();
    let mut out = Vec::new();
    for p in rows {
        let (a, na) = (embs[p], norms[p]);
        let source = memories[eligible[p]].id;
        for q in (p + 1)..m {
            let sim = cosine_prenormed(a, na, embs[q], norms[q]);
            if sim >= threshold {
                out.push(GraphEdge {
                    source,
                    target: memories[eligible[q]].id,
                    weight: sim,
                });
            }
        }
    }
    out
}

/// Split rows `0..m` of the upper triangle into `parts` contiguous ranges of
/// roughly equal pair count (row `p` has `m-1-p` pairs).
fn balanced_row_ranges(m: usize, parts: usize) -> Vec<std::ops::Range<usize>> {
    let total = (m as u128) * (m.saturating_sub(1) as u128) / 2;
    let mut ranges = Vec::with_capacity(parts);
    let (mut start, mut acc) = (0usize, 0u128);
    for p in 0..m {
        acc += (m - 1 - p) as u128;
        let k = ranges.len() as u128 + 1;
        if ranges.len() + 1 < parts && acc * (parts as u128) >= total * k {
            ranges.push(start..p + 1);
            start = p + 1;
        }
    }
    ranges.push(start..m);
    ranges
}

impl KnowledgeGraph {
    /// Build a whole graph from `memories`. Pure: no `self`, so it cannot
    /// hold a graph lock, and is safe to run in `spawn_blocking`.
    ///
    /// Node positions follow `memories` order and edges are emitted in
    /// `(i, j)` row-major order — exactly what the previous in-place loop
    /// produced — regardless of `threads`.
    pub fn build_batch(memories: &[BrainMemory], threshold: f64, threads: usize) -> BatchBuild {
        Self::build_batch_inner(memories, threshold, threads, PARALLEL_MIN_NODES)
    }

    /// `build_batch` with an explicit parallel cut-over, so tests can force
    /// the multi-threaded path on a small graph.
    pub(crate) fn build_batch_inner(
        memories: &[BrainMemory],
        threshold: f64,
        threads: usize,
        parallel_min_nodes: usize,
    ) -> BatchBuild {
        let start = Instant::now();
        let n = memories.len();
        let mut nodes = HashMap::with_capacity(n);
        let mut node_ids = Vec::with_capacity(n);
        let mut node_index = HashMap::with_capacity(n);
        let mut eligible = Vec::with_capacity(n);
        for (idx, m) in memories.iter().enumerate() {
            let quality = m.quality_score.mean();
            nodes.insert(
                m.id,
                GraphNode {
                    embedding: m.embedding.clone(),
                    category: m.category.clone(),
                    quality,
                },
            );
            node_index.insert(m.id, idx);
            node_ids.push(m.id);
            if quality >= EDGE_QUALITY_FLOOR {
                eligible.push(idx);
            }
        }

        let embs: Vec<&[f32]> = eligible
            .iter()
            .map(|&i| memories[i].embedding.as_slice())
            .collect();
        let norms: Vec<f64> = embs.iter().map(|e| split_norm(e)).collect();
        let m = eligible.len();
        let pairs_compared = (m as u64) * (m.saturating_sub(1) as u64) / 2;

        let threads = if m < parallel_min_nodes || m < 2 {
            1
        } else {
            threads.max(1)
        };
        let edges = if threads == 1 {
            edge_rows(0..m, &eligible, &embs, &norms, memories, threshold)
        } else {
            let ranges = balanced_row_ranges(m, threads);
            let parts: Vec<Vec<GraphEdge>> = std::thread::scope(|s| {
                let handles: Vec<_> = ranges
                    .into_iter()
                    .map(|r| {
                        let (eligible, embs, norms) = (&eligible, &embs, &norms);
                        s.spawn(move || edge_rows(r, eligible, embs, norms, memories, threshold))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("graph rebuild worker panicked"))
                    .collect()
            });
            // Concatenating the contiguous row ranges in order reproduces the
            // sequential edge order exactly.
            let mut edges = Vec::with_capacity(parts.iter().map(Vec::len).sum());
            for p in parts {
                edges.extend(p);
            }
            edges
        };

        BatchBuild {
            nodes,
            node_ids,
            node_index,
            edges,
            stats: BuildStats {
                elapsed: start.elapsed(),
                threads,
                pairs_compared,
            },
        }
    }

    /// Replace the adjacency with `build`. Invalidates every position-keyed
    /// derivative (CSR, mincut, sparsifier) and bumps `index_generation`.
    pub(super) fn swap_in_batch(&mut self, build: BatchBuild) -> (Retired, BuildStats) {
        let retired = Retired {
            _nodes: std::mem::replace(&mut self.nodes, build.nodes),
            _node_ids: std::mem::replace(&mut self.node_ids, build.node_ids),
            _node_index: std::mem::replace(&mut self.node_index, build.node_index),
            _edges: std::mem::replace(&mut self.edges, build.edges),
            _mincut: self.mincut.take(),
            _sparsifier: self.sparsifier.take(),
        };
        self.mark_csr_dirty();
        // Every node position was reassigned: an in-flight sparsifier build
        // (ADR-349 P3) must refuse to install.
        self.index_generation = self.index_generation.wrapping_add(1);
        (retired, build.stats)
    }

    /// Start an off-lock rebuild. Returns `None` if one is already in flight
    /// (single-flight across the scheduler action and the cold-start path).
    pub fn begin_rebuild(&mut self) -> Option<RebuildTicket> {
        if self.rebuild_log.is_some() {
            return None;
        }
        self.rebuild_seq = self.rebuild_seq.wrapping_add(1);
        self.rebuild_log = Some(RebuildLog {
            ticket: self.rebuild_seq,
            entries: Vec::new(),
        });
        Some(RebuildTicket {
            id: self.rebuild_seq,
            threshold: self.similarity_threshold,
        })
    }

    /// Whether an off-lock rebuild is currently in flight.
    pub fn rebuild_in_flight(&self) -> bool {
        self.rebuild_log.is_some()
    }

    /// Release the in-flight marker without installing. Returns whether
    /// `ticket` was the in-flight rebuild.
    pub fn abort_rebuild(&mut self, ticket: RebuildTicket) -> bool {
        match &self.rebuild_log {
            Some(log) if log.ticket == ticket.id => {
                self.rebuild_log = None;
                true
            }
            _ => false,
        }
    }

    /// Install a build under the caller's (brief) write lock, then replay
    /// every mutation recorded since `begin_rebuild`, in order:
    ///
    /// - an add whose id the snapshot already contains is skipped (it was
    ///   in the store when the snapshot was taken, so its edges are exact);
    /// - any other add is inserted exactly as `add_memory` would;
    /// - a remove deletes the node if the snapshot still has it.
    ///
    /// Returns `None` (and changes nothing) if `ticket` is not the in-flight
    /// rebuild — e.g. an in-place `rebuild_from_batch` superseded it.
    pub fn install_batch(
        &mut self,
        ticket: RebuildTicket,
        build: BatchBuild,
    ) -> Option<(InstallReport, Retired)> {
        match &self.rebuild_log {
            Some(log) if log.ticket == ticket.id => {}
            _ => return None,
        }
        let t0 = Instant::now();
        let log = self.rebuild_log.take()?;
        let (retired, build_stats) = self.swap_in_batch(build);
        let (mut adds, mut removes) = (0usize, 0usize);
        for m in log.entries {
            match m {
                Mutation::Add {
                    id,
                    embedding,
                    category,
                    quality,
                } => {
                    if !self.nodes.contains_key(&id) {
                        self.add_node(
                            id,
                            GraphNode {
                                embedding,
                                category,
                                quality,
                            },
                        );
                        adds += 1;
                    }
                }
                Mutation::Remove(id) => {
                    if self.nodes.contains_key(&id) {
                        self.remove_node(&id);
                        removes += 1;
                    }
                }
            }
        }
        Some((
            InstallReport {
                nodes: self.nodes.len(),
                edges: self.edges.len(),
                replayed_adds: adds,
                replayed_removes: removes,
                build: build_stats,
                install_elapsed: t0.elapsed(),
            },
            retired,
        ))
    }

    /// All edges as `(source, target, weight)` in storage order. For
    /// equivalence tests.
    pub fn edges_snapshot(&self) -> Vec<(Uuid, Uuid, f64)> {
        self.edges
            .iter()
            .map(|e| (e.source, e.target, e.weight))
            .collect()
    }
}

mod task;
pub use task::{rebuild_off_lock, rebuild_sparsifier_off_lock, spawn_rebuild, RebuildOutcome};

#[cfg(test)]
mod tests;
