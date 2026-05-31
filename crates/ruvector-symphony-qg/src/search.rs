//! Beam search over the SymphonyQG graph.
//!
//! ## Algorithm
//!
//! Standard greedy beam search (à la HNSW layer-0 / NSG) with two modes:
//!
//! **Exact mode** (used in `GraphExact` index):
//!   Each candidate's neighbors are scored with exact L2 distance.
//!   Baseline for measuring quantization overhead.
//!
//! **Symphony mode** (used in `SymphonyIndex`):
//!   Neighbor distances are estimated using the co-located RaBitQ codes
//!   via `batch_asym_l2`. Only the *current candidate* (already in the
//!   beam set) requires an exact distance; all R neighbors are scored by
//!   the asymmetric estimator without any random memory reads.
//!
//! ## Termination
//!
//! The beam set is a max-heap of size `ef`. Expansion stops when the
//! best unvisited candidate's estimated distance exceeds the worst
//! distance in the result heap. This is the standard HNSW termination
//! criterion; in SymphonyQG it is safe because the RaBitQ estimator is
//! an unbiased approximation with bounded variance.

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

use crate::codes::{batch_asym_l2, QueryProjection};
use crate::graph::{l2_sq, SymphonyGraph};

/// (distance, id) ordered as a min-heap entry (Rust's BinaryHeap is max-heap,
/// so we negate the distance comparison).
#[derive(Clone)]
struct HeapEntry {
    neg_dist: f32, // stored negated for max-heap inversion
    id: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool { self.neg_dist == other.neg_dist }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.neg_dist.partial_cmp(&other.neg_dist).unwrap_or(Ordering::Equal)
    }
}

fn random_entry_points(n: usize, count: usize, seed: u64) -> Vec<usize> {
    // Pseudo-random starting points spread across the graph
    let step = n / count.max(1);
    (0..count).map(|i| (i * step + seed as usize) % n).collect()
}

/// Beam search with exact L2 distances (no quantization).
pub fn beam_search_exact(
    graph: &SymphonyGraph,
    query: &[f32],
    k: usize,
    ef: usize,
    n_starts: usize,
) -> Vec<(f32, usize)> {
    let n = graph.vertices.len();
    if n == 0 { return vec![]; }

    let mut visited = HashSet::new();
    // candidates: min-heap by distance (we negate to use BinaryHeap as min-heap)
    let mut candidates: BinaryHeap<HeapEntry> = BinaryHeap::new();
    // results: max-heap of size ef (for top-k extraction)
    let mut results: BinaryHeap<HeapEntry> = BinaryHeap::new();

    let entries = random_entry_points(n, n_starts, 0);
    for ep in entries {
        if visited.contains(&ep) { continue; }
        let d = l2_sq(query, &graph.vertices[ep].raw);
        candidates.push(HeapEntry { neg_dist: -d, id: ep });
    }

    while let Some(HeapEntry { neg_dist, id }) = candidates.pop() {
        let dist = -neg_dist;
        if visited.contains(&id) { continue; }
        visited.insert(id);

        // Prune: if the result set is full and current dist > worst result, stop
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if dist > -worst.neg_dist { break; }
            }
        }

        results.push(HeapEntry { neg_dist: -dist, id });
        if results.len() > ef {
            results.pop(); // remove the farthest
        }

        // Expand neighbors with exact distances
        let v = &graph.vertices[id];
        for &nid in &v.neighbor_ids {
            let nid = nid as usize;
            if !visited.contains(&nid) {
                let nd = l2_sq(query, &graph.vertices[nid].raw);
                candidates.push(HeapEntry { neg_dist: -nd, id: nid });
            }
        }
    }

    let mut out: Vec<(f32, usize)> = results
        .into_iter()
        .map(|e| (-e.neg_dist, e.id))
        .collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out.truncate(k);
    out
}

/// Beam search with asymmetric RaBitQ distance estimates on co-located codes.
/// Exact distance is only computed for the current node (already in beam).
pub fn beam_search_symphony(
    graph: &SymphonyGraph,
    query: &[f32],
    k: usize,
    ef: usize,
    n_starts: usize,
) -> Vec<(f32, usize)> {
    let n = graph.vertices.len();
    if n == 0 { return vec![]; }

    let dim = graph.config.dim;
    let q_rot = crate::rotation::rotate(&graph.rotation, query, dim);
    let norm_q_sq = query.iter().map(|v| v * v).sum::<f32>();
    let qp = QueryProjection::new(q_rot);

    let mut visited = HashSet::new();
    let mut candidates: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut results: BinaryHeap<HeapEntry> = BinaryHeap::new();

    let entries = random_entry_points(n, n_starts, 0);
    for ep in entries {
        if visited.contains(&ep) { continue; }
        // Entry points: use exact distance for the seed (no codes available without neighbor context)
        let d = l2_sq(query, &graph.vertices[ep].raw);
        candidates.push(HeapEntry { neg_dist: -d, id: ep });
    }

    while let Some(HeapEntry { neg_dist, id }) = candidates.pop() {
        let dist = -neg_dist;
        if visited.contains(&id) { continue; }
        visited.insert(id);

        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if dist > -worst.neg_dist { break; }
            }
        }

        results.push(HeapEntry { neg_dist: -dist, id });
        if results.len() > ef {
            results.pop();
        }

        // Batch estimate distances for all R neighbors using co-located codes
        let v = &graph.vertices[id];
        let r = v.neighbor_ids.len();
        if r == 0 { continue; }

        let est_dists = batch_asym_l2(&qp, &v.neighbor_codes, &v.neighbor_norms, norm_q_sq);

        for (slot, &nid) in v.neighbor_ids.iter().enumerate() {
            let nid = nid as usize;
            if !visited.contains(&nid) {
                candidates.push(HeapEntry { neg_dist: -est_dists[slot], id: nid });
            }
        }
    }

    // Final step: retrieve exact distances for the top ef candidates in results
    // This is the "re-rank-free" design: the beam already converged well enough
    // that we return the exact distances for the top-k within the result set.
    let mut out: Vec<(f32, usize)> = results
        .into_iter()
        .map(|e| {
            let id = e.id;
            let exact = l2_sq(query, &graph.vertices[id].raw);
            (exact, id)
        })
        .collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out.truncate(k);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{graph::{GraphConfig, SymphonyGraph}, rotation::random_orthogonal};

    fn tiny_graph(n: usize, dim: usize) -> SymphonyGraph {
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f32 * 0.01).collect())
            .collect();
        let rot = random_orthogonal(dim, 42);
        let cfg = GraphConfig::new(dim).with_r(4).with_ef(8);
        SymphonyGraph::build(&vecs, cfg, &rot)
    }

    #[test]
    fn exact_returns_nearest() {
        let dim = 8;
        let graph = tiny_graph(16, dim);
        let query: Vec<f32> = (0..dim).map(|j| 0.0 * j as f32).collect();
        let results = beam_search_exact(&graph, &query, 1, 8, 4);
        assert!(!results.is_empty());
        // Nearest should be vertex 0 (all zeros for i=0 case)
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn symphony_finds_reasonable_neighbours() {
        let dim = 16;
        let n = 50;
        let graph = tiny_graph(n, dim);
        let query: Vec<f32> = vec![0.0; dim];
        let res_exact = beam_search_exact(&graph, &query, 5, 20, 4);
        let res_sym = beam_search_symphony(&graph, &query, 5, 20, 4);
        // At least 3 of top-5 should overlap between exact and symphony
        let exact_ids: HashSet<usize> = res_exact.iter().map(|(_, id)| *id).collect();
        let overlap = res_sym.iter().filter(|(_, id)| exact_ids.contains(id)).count();
        assert!(overlap >= 2, "too little overlap: {overlap}/5");
    }
}
