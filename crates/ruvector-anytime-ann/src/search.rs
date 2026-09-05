//! Three anytime beam-search variants.
//!
//! All variants share the same priority-queue skeleton. They differ only in
//! their stopping criterion — when to interrupt the search and return results.
//!
//! ## Variants
//!
//! * **FixedEfSearch** — Standard HNSW beam search. Maintains a candidate heap
//!   of up to `ef` entries; stops when all remaining candidates are farther
//!   than the current kth result. This is the baseline.
//!
//! * **BudgetedEvalsSearch** — Stops when the total number of distance
//!   evaluations exceeds `max_evals`. Returns the best k results found so
//!   far. This bounds compute cost regardless of graph structure.
//!
//! * **EarlyConvergenceSearch** — Stops when the kth-nearest distance has not
//!   improved by at least `min_improvement` for `patience` consecutive
//!   candidate expansions. Implements the anytime property: stop when
//!   marginal returns fall below a threshold.
//!
//! ## Anytime guarantee
//!
//! At any point during graph traversal, the result max-heap holds the k
//! nearest vectors seen so far. All three variants can be interrupted and
//! will return the best available answer.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::graph::{l2_sq, FlatGraph};

/// Orderable f32 for BinaryHeap (natural order = max-heap by value).
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

/// Result from a single ANN query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Top-k results as (node_id, squared_L2_distance), nearest first.
    pub neighbors: Vec<(u32, f32)>,
    /// Total distance evaluations performed.
    pub evaluations: usize,
    /// Number of candidate neighborhoods expanded.
    pub expansions: usize,
}

/// Common interface for all three search backends.
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

// ─────────────────────────────────────────────────────────────────────────────
// Variant 1: FixedEfSearch (baseline)
// ─────────────────────────────────────────────────────────────────────────────

/// Standard beam search with fixed exploration factor.
///
/// Stops only when all candidates in the heap are farther than the kth result.
/// This is the reference HNSW layer-0 implementation.
pub struct FixedEfSearch;

impl Searcher for FixedEfSearch {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult {
        beam_search(graph, query, k, ef, entry_id, FixedStop)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant 2: BudgetedEvalsSearch
// ─────────────────────────────────────────────────────────────────────────────

/// Beam search with a hard limit on total distance evaluations.
///
/// When `evaluations >= max_evals`, the search halts and returns the best k
/// results seen so far. This bounds compute cost to a known constant — useful
/// for edge deployment where CPU time is the scarce resource.
pub struct BudgetedEvalsSearch {
    /// Maximum number of distance evaluations before forced termination.
    pub max_evals: usize,
}

impl Searcher for BudgetedEvalsSearch {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult {
        beam_search(graph, query, k, ef, entry_id, BudgetStop { max: self.max_evals })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant 3: EarlyConvergenceSearch
// ─────────────────────────────────────────────────────────────────────────────

/// Beam search with early termination on convergence.
///
/// Tracks how much the kth-nearest distance improves per expansion. When it
/// fails to improve by at least `min_improvement` for `patience` consecutive
/// expansions, the search terminates early. This is the "anytime" property:
/// stop when marginal returns fall below a practical threshold.
pub struct EarlyConvergenceSearch {
    /// Consecutive non-improving expansions before termination.
    pub patience: usize,
    /// Minimum absolute improvement in kth distance to reset the counter.
    pub min_improvement: f32,
}

impl Searcher for EarlyConvergenceSearch {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult {
        beam_search(
            graph, query, k, ef, entry_id,
            ConvergeStop { patience: self.patience, min_imp: self.min_improvement, stalls: 0 },
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stopping policy trait
// ─────────────────────────────────────────────────────────────────────────────

/// Controls when to stop the beam search.
trait StopPolicy {
    /// Return `true` to continue, `false` to stop.
    fn should_continue(&mut self, evals: usize, kth_dist: f32, prev_kth: f32) -> bool;
}

struct FixedStop;
impl StopPolicy for FixedStop {
    fn should_continue(&mut self, _evals: usize, _kth: f32, _prev: f32) -> bool {
        true
    }
}

struct BudgetStop {
    max: usize,
}
impl StopPolicy for BudgetStop {
    fn should_continue(&mut self, evals: usize, _kth: f32, _prev: f32) -> bool {
        evals < self.max
    }
}

struct ConvergeStop {
    patience: usize,
    min_imp: f32,
    stalls: usize,
}
impl StopPolicy for ConvergeStop {
    fn should_continue(&mut self, _evals: usize, kth: f32, prev_kth: f32) -> bool {
        let improved = prev_kth - kth;
        if improved >= self.min_imp {
            self.stalls = 0;
        } else {
            self.stalls += 1;
        }
        self.stalls < self.patience
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared beam-search kernel
// ─────────────────────────────────────────────────────────────────────────────

fn beam_search<S: StopPolicy>(
    graph: &FlatGraph,
    query: &[f32],
    k: usize,
    ef: usize,
    entry_id: usize,
    mut stop: S,
) -> SearchResult {
    // candidates: min-heap by distance (Reverse turns BinaryHeap into min-heap)
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
    // results: max-heap by distance (pop removes the farthest)
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
    let mut visited: HashSet<u32> = HashSet::new();

    let entry = entry_id as u32;
    let d_entry = l2_sq(query, graph.vector(entry_id));
    candidates.push(Reverse((OrdF32(d_entry), entry)));
    results.push((OrdF32(d_entry), entry));
    visited.insert(entry);

    let mut evaluations = 1usize;
    let mut expansions = 0usize;
    let mut prev_kth = f32::INFINITY;

    while let Some(Reverse((OrdF32(d_cand), cand_id))) = candidates.peek().cloned() {
        // Standard HNSW stop: candidate is farther than worst current result.
        let worst = results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY);
        if results.len() >= k && d_cand > worst {
            break;
        }

        let kth = if results.len() >= k { worst } else { f32::INFINITY };
        if !stop.should_continue(evaluations, kth, prev_kth) {
            break;
        }
        prev_kth = kth;

        candidates.pop();
        expansions += 1;

        for &nb_id in &graph.neighbors[cand_id as usize] {
            if visited.contains(&nb_id) {
                continue;
            }
            visited.insert(nb_id);
            let d_nb = l2_sq(query, graph.vector(nb_id as usize));
            evaluations += 1;

            let cur_worst = results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY);
            if results.len() < k || d_nb < cur_worst {
                results.push((OrdF32(d_nb), nb_id));
                if results.len() > k {
                    results.pop();
                }
                if candidates.len() < ef {
                    candidates.push(Reverse((OrdF32(d_nb), nb_id)));
                }
            } else if candidates.len() < ef {
                candidates.push(Reverse((OrdF32(d_nb), nb_id)));
            }
        }
    }

    let mut out: Vec<(u32, f32)> = results.into_iter().map(|(d, id)| (id, d.0)).collect();
    out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    out.truncate(k);

    SearchResult { neighbors: out, evaluations, expansions }
}
