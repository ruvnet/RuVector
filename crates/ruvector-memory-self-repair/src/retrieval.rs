//! Associative retrieval quality: graph-walk top-k retrieval whose accuracy
//! depends directly on structural connectivity, so it is the honest,
//! measurable stand-in for "does drift/repair actually matter" rather than
//! a proxy metric decoupled from the graph itself.

use crate::graph::MemoryGraph;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoredNode {
    score: f64,
    node: usize,
}
impl Eq for ScoredNode {}
impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}
impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Best-first graph walk from `query`: score of a reached node is the
/// maximum path weight (product of edge weights) over all paths explored
/// within `budget` node expansions. This is exactly the kind of traversal a
/// real associative-memory read would perform, and it degrades gracefully
/// (not catastrophically) as bridge edges weaken, then recovers once a
/// repair reconnects them.
pub fn graph_walk_topk(graph: &MemoryGraph, query: usize, k: usize, budget: usize) -> Vec<usize> {
    let mut best: Vec<f64> = vec![0.0; graph.nodes.len()];
    let mut heap = BinaryHeap::new();
    best[query] = 1.0;
    heap.push(ScoredNode {
        score: 1.0,
        node: query,
    });
    let mut expansions = 0usize;
    let mut visited_order: Vec<(usize, f64)> = Vec::new();

    while let Some(ScoredNode { score, node }) = heap.pop() {
        if score < best[node] - 1e-12 {
            continue; // stale heap entry
        }
        if node != query {
            visited_order.push((node, score));
        }
        expansions += 1;
        if expansions >= budget {
            break;
        }
        for (&next, &w) in &graph.adjacency[node] {
            if !graph.nodes[next].alive {
                continue;
            }
            let cand = score * w;
            if cand > best[next] {
                best[next] = cand;
                heap.push(ScoredNode {
                    score: cand,
                    node: next,
                });
            }
        }
    }

    visited_order.sort_by(|a, b| b.1.total_cmp(&a.1));
    visited_order.into_iter().take(k).map(|(n, _)| n).collect()
}

/// Recall@k of graph-walk retrieval against each query's *original*
/// associative neighbors (the ones it was wired to at formation time,
/// still alive), averaged over `queries`. Queries whose ground truth is
/// empty are skipped (undefined, not scored as 0 or 1).
///
/// This is deliberately not "any other same-topic memory": with a few
/// hundred same-topic memories typically alive, top-k against that broad a
/// truth set is satisfied by whatever is locally abundant and never
/// exercises reachability, which saturates recall at 1.0 for every variant
/// regardless of drift (see git history of this file). Restricting ground
/// truth to a memory's own originally-formed, still-alive neighbors makes
/// recall directly sensitive to whether the graph retained/recovered the
/// specific structure that existed before drift and compaction acted on it.
pub fn mean_recall_at_k(
    graph: &MemoryGraph,
    queries: &[usize],
    k: usize,
    budget: usize,
) -> (f64, usize) {
    let mut total = 0.0;
    let mut counted = 0usize;
    for &q in queries {
        if !graph.nodes[q].alive {
            continue;
        }
        let truth: HashSet<usize> = graph.nodes[q]
            .original_neighbors
            .iter()
            .copied()
            .filter(|&n| graph.nodes[n].alive)
            .collect();
        if truth.is_empty() {
            continue;
        }
        let predicted = graph_walk_topk(graph, q, k, budget);
        let hits = predicted.iter().filter(|p| truth.contains(p)).count();
        let denom = k.min(truth.len());
        total += hits as f64 / denom as f64;
        counted += 1;
    }
    if counted == 0 {
        (0.0, 0)
    } else {
        (total / counted as f64, counted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::TopicSpace;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn reachable_original_neighbors_are_recalled() {
        let mut rng = StdRng::seed_from_u64(2);
        let space = TopicSpace::new(8, 2, &mut rng);
        let mut g = MemoryGraph::new();
        let mut ids = Vec::new();
        for _ in 0..6 {
            ids.push(g.add_node(space.sample(0, 0.02, &mut rng), 0));
        }
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                g.add_edge(ids[i], ids[j], 0.9);
            }
        }
        // Every node's "original neighbors" are declared to be all the
        // others, matching the fully-connected wiring above.
        for &id in &ids {
            g.nodes[id].original_neighbors = ids.iter().copied().filter(|&o| o != id).collect();
        }
        let (recall, counted) = mean_recall_at_k(&g, &ids, 5, 50);
        assert_eq!(counted, ids.len());
        assert!(recall > 0.9, "recall = {recall}");
    }

    #[test]
    fn unreachable_original_neighbor_is_not_recalled() {
        let mut rng = StdRng::seed_from_u64(2);
        let space = TopicSpace::new(8, 2, &mut rng);
        let mut g = MemoryGraph::new();
        let a = g.add_node(space.sample(0, 0.02, &mut rng), 0);
        let b = g.add_node(space.sample(0, 0.02, &mut rng), 0);
        // a's original neighbor was b, but no edge exists any more (e.g.
        // decayed away): graph walk from a can never reach b.
        g.nodes[a].original_neighbors = vec![b];
        g.nodes[b].original_neighbors = vec![a];
        let (recall, counted) = mean_recall_at_k(&g, &[a, b], 5, 50);
        assert_eq!(counted, 2);
        assert_eq!(recall, 0.0);
    }
}
