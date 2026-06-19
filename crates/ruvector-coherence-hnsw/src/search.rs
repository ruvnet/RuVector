//! Three beam-search variants on a flat proximity graph.
//!
//! All variants share the same priority-queue beam-search skeleton. The
//! difference is whether and how the coherence gate prunes candidate expansion.
//!
//! ## Variants
//!
//! * **Baseline** – Standard beam search. Every popped candidate's neighbors
//!   are expanded unconditionally.
//!
//! * **CoherenceGated** – Fixed threshold. A candidate's neighbors are only
//!   expanded if its traversal coherence exceeds `threshold`. The candidate
//!   itself is still considered as a result regardless.
//!
//! * **AdaptiveCoherence** – The threshold starts at `initial_threshold` and
//!   rises as the beam finds better results, falls when stuck.
//!
//! ## Entry point
//!
//! All variants use the node at index `entry_id` as the search starting point.
//! Passing a fixed entry (e.g., 0) simulates HNSW layer-0 navigation where the
//! starting position comes from a coarse upper-layer greedy descent that may be
//! distant from the query. This is where the coherence gate has the most effect:
//! when the beam must navigate through many nodes before converging on the query,
//! off-path branches are pruned by the gate while on-path branches are expanded.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::coherence::{traversal_coherence, update_adaptive_threshold};
use crate::graph::{l2_sq, FlatGraph};

/// Ordered f32 wrapper for use in BinaryHeap.
#[derive(Clone, Copy, PartialEq)]
pub struct OrdF32(pub f32);

impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Output from a single query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Top-k results as (node_id, squared_L2_distance), sorted nearest first.
    pub neighbors: Vec<(u32, f32)>,
    /// How many candidates were popped from the heap (evaluations).
    pub pops: usize,
    /// How many candidates' neighbor lists were iterated (expanded).
    /// This is the main metric: gated search skips expanding low-coherence nodes.
    pub expansions: usize,
}

/// Common trait for all search backends.
pub trait Searcher {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult;
}

// ──────────────────────────────────────────────────────────────────────────────
// Baseline
// ──────────────────────────────────────────────────────────────────────────────

/// Standard beam search — all candidate neighborhoods are expanded.
pub struct BaselineSearch;

impl Searcher for BaselineSearch {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult {
        beam_search(graph, query, k, ef, entry_id, |_entry, _cand, _q| true)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CoherenceGated (fixed threshold)
// ──────────────────────────────────────────────────────────────────────────────

/// Beam search with a fixed traversal-coherence gate.
///
/// Candidates with `traversal_coherence(entry, candidate, query) < threshold`
/// are popped and considered as result candidates, but their neighbors are NOT
/// expanded. This prunes off-path branches while preserving on-path exploration.
pub struct CoherenceGatedSearch {
    pub threshold: f32,
}

impl Searcher for CoherenceGatedSearch {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult {
        let t = self.threshold;
        beam_search(graph, query, k, ef, entry_id, |entry, cand, q| {
            traversal_coherence(entry, cand, q) >= t
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AdaptiveCoherence (dynamic threshold)
// ──────────────────────────────────────────────────────────────────────────────

/// Beam search where the coherence threshold adapts to beam progress.
///
/// Threshold rises by `adaptation_rate` when the beam finds a new best
/// candidate (the search is converging; be selective about which branches
/// to expand). Falls by half that rate when stuck (be more exploratory).
pub struct AdaptiveCoherenceSearch {
    pub initial_threshold: f32,
    pub adaptation_rate: f32,
    pub max_threshold: f32,
}

impl Default for AdaptiveCoherenceSearch {
    fn default() -> Self {
        AdaptiveCoherenceSearch {
            initial_threshold: 0.0,
            adaptation_rate: 0.08,
            max_threshold: 0.65,
        }
    }
}

impl Searcher for AdaptiveCoherenceSearch {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult {
        adaptive_beam_search(
            graph,
            query,
            k,
            ef,
            entry_id,
            self.initial_threshold,
            self.adaptation_rate,
            self.max_threshold,
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core implementation
// ──────────────────────────────────────────────────────────────────────────────

/// Generic beam search with a pluggable expansion gate.
///
/// `expand_gate(entry_vec, candidate_vec, query_vec) -> bool`
///   - returning `true`  → expand candidate's neighbors
///   - returning `false` → skip neighbor expansion (candidate still a result)
fn beam_search<F>(
    graph: &FlatGraph,
    query: &[f32],
    k: usize,
    ef: usize,
    entry_id: usize,
    expand_gate: F,
) -> SearchResult
where
    F: Fn(&[f32], &[f32], &[f32]) -> bool,
{
    if graph.is_empty() {
        return SearchResult {
            neighbors: vec![],
            pops: 0,
            expansions: 0,
        };
    }

    let n = graph.n;
    let ef = ef.max(k);
    let entry = entry_id.min(n - 1);
    let entry_vec = graph.row(entry);

    let mut visited: Vec<bool> = vec![false; n];
    // Min-heap: pop closest candidate first.
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::with_capacity(ef + 1);
    // Max-heap: top-k results; peek = farthest accepted result.
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::with_capacity(k + 1);

    let d0 = l2_sq(query, entry_vec);
    candidates.push(Reverse((OrdF32(d0), entry as u32)));
    visited[entry] = true;

    let mut pops: usize = 0;
    let mut expansions: usize = 0;

    while let Some(Reverse((OrdF32(curr_d), curr))) = candidates.pop() {
        pops += 1;

        // Early stop: if this candidate is already worse than our worst result,
        // no neighbor can improve things.
        if results.len() >= k {
            if let Some(&(OrdF32(worst), _)) = results.peek() {
                if curr_d > worst {
                    break;
                }
            }
        }

        // Always consider curr as a result candidate.
        results.push((OrdF32(curr_d), curr));
        if results.len() > k {
            results.pop();
        }

        let curr_vec = graph.row(curr as usize);

        // Coherence gate: check whether to expand this candidate's neighbors.
        if !expand_gate(entry_vec, curr_vec, query) {
            continue; // skip expansion, pops already counted
        }

        expansions += 1;

        // Compute farthest pending candidate distance for the bounded beam.
        // We scan candidates — O(ef) per expansion. Acceptable for small ef.
        let worst_pending = candidates
            .iter()
            .map(|Reverse((OrdF32(d), _))| *d)
            .fold(f32::NEG_INFINITY, f32::max);

        for &nbr in &graph.neighbors[curr as usize] {
            let ni = nbr as usize;
            if visited[ni] {
                continue;
            }
            visited[ni] = true;
            let nd = l2_sq(query, graph.row(ni));

            if candidates.len() < ef || nd < worst_pending {
                candidates.push(Reverse((OrdF32(nd), nbr)));
                if candidates.len() > ef {
                    // Evict the farthest candidate to maintain bounded beam.
                    let mut items: Vec<_> = candidates.drain().collect();
                    items.sort_unstable_by(|a, b| a.0 .0 .0.total_cmp(&b.0 .0 .0));
                    items.truncate(ef);
                    candidates.extend(items);
                }
            }
        }
    }

    let mut neighbors: Vec<(u32, f32)> =
        results.into_iter().map(|(OrdF32(d), id)| (id, d)).collect();
    neighbors.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    SearchResult {
        neighbors,
        pops,
        expansions,
    }
}

/// Adaptive beam search: threshold evolves as the beam progresses.
#[allow(clippy::too_many_arguments)]
fn adaptive_beam_search(
    graph: &FlatGraph,
    query: &[f32],
    k: usize,
    ef: usize,
    entry_id: usize,
    initial_threshold: f32,
    adaptation_rate: f32,
    max_threshold: f32,
) -> SearchResult {
    if graph.is_empty() {
        return SearchResult {
            neighbors: vec![],
            pops: 0,
            expansions: 0,
        };
    }

    let n = graph.n;
    let ef = ef.max(k);
    let entry = entry_id.min(n - 1);
    let entry_vec = graph.row(entry);

    let mut visited: Vec<bool> = vec![false; n];
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::with_capacity(ef + 1);
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::with_capacity(k + 1);

    let d0 = l2_sq(query, entry_vec);
    candidates.push(Reverse((OrdF32(d0), entry as u32)));
    visited[entry] = true;

    let mut pops: usize = 0;
    let mut expansions: usize = 0;
    let mut threshold = initial_threshold;
    let mut best_dist = d0;

    while let Some(Reverse((OrdF32(curr_d), curr))) = candidates.pop() {
        pops += 1;

        if results.len() >= k {
            if let Some(&(OrdF32(worst), _)) = results.peek() {
                if curr_d > worst {
                    break;
                }
            }
        }

        results.push((OrdF32(curr_d), curr));
        if results.len() > k {
            results.pop();
        }

        let new_best = curr_d.min(best_dist);
        threshold = update_adaptive_threshold(
            threshold,
            best_dist,
            new_best,
            adaptation_rate,
            max_threshold,
        );
        best_dist = new_best;

        let curr_vec = graph.row(curr as usize);

        if traversal_coherence(entry_vec, curr_vec, query) < threshold {
            continue;
        }

        expansions += 1;

        let worst_pending = candidates
            .iter()
            .map(|Reverse((OrdF32(d), _))| *d)
            .fold(f32::NEG_INFINITY, f32::max);

        for &nbr in &graph.neighbors[curr as usize] {
            let ni = nbr as usize;
            if visited[ni] {
                continue;
            }
            visited[ni] = true;
            let nd = l2_sq(query, graph.row(ni));

            if candidates.len() < ef || nd < worst_pending {
                candidates.push(Reverse((OrdF32(nd), nbr)));
                if candidates.len() > ef {
                    let mut items: Vec<_> = candidates.drain().collect();
                    items.sort_unstable_by(|a, b| a.0 .0 .0.total_cmp(&b.0 .0 .0));
                    items.truncate(ef);
                    candidates.extend(items);
                }
            }
        }
    }

    let mut neighbors: Vec<(u32, f32)> =
        results.into_iter().map(|(OrdF32(d), id)| (id, d)).collect();
    neighbors.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    SearchResult {
        neighbors,
        pops,
        expansions,
    }
}
