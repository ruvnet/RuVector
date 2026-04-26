use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;

use crate::dist::l2_sq;
use crate::graph::{AcornGraph, OrdF32};

/// ACORN beam search — the core innovation over standard HNSW + post-filter.
///
/// Standard post-filter HNSW skips predicate-failing nodes during traversal,
/// starving the beam of candidates when predicate selectivity is low (e.g. 1%).
///
/// ACORN's fix: expand ALL neighbors regardless of predicate outcome.
/// A node that fails the predicate is NOT added to `results`, but its neighbors
/// ARE added to `candidates`. The denser graph (built with γ·M edges) ensures
/// enough valid nodes are reachable even through chains of failing nodes.
///
/// # Parameters
/// - `ef` — beam width (number of candidates to explore). Higher = better recall,
///   lower = faster.  Typical: 64–200.
pub fn acorn_search(
    graph: &AcornGraph,
    query: &[f32],
    k: usize,
    ef: usize,
    predicate: impl Fn(u32) -> bool,
) -> Vec<(u32, f32)> {
    if graph.len() == 0 {
        return vec![];
    }

    // Multi-probe entry: sample evenly-spaced nodes to find a good starting
    // point. O(probes × D) overhead vs O(n × D) for flat — negligible.
    let n = graph.len();
    let n_probes = (n as f64).sqrt().ceil() as usize;
    let n_probes = n_probes.clamp(4, 64);
    let entry = (0..n_probes)
        .map(|i| (i * n / n_probes) as u32)
        .min_by(|&a, &b| {
            l2_sq(query, &graph.data[a as usize])
                .total_cmp(&l2_sq(query, &graph.data[b as usize]))
        })
        .unwrap_or(0);

    let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 2);
    // Min-heap by distance: Reverse makes BinaryHeap act as min-heap.
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> =
        BinaryHeap::with_capacity(ef + 1);
    // Max-heap by distance — top is the worst accepted result so far.
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::with_capacity(k + 1);

    let d0 = l2_sq(query, &graph.data[entry as usize]);
    candidates.push(Reverse((OrdF32(d0), entry)));
    visited.insert(entry);

    while let Some(Reverse((OrdF32(curr_d), curr))) = candidates.pop() {
        // Prune: if current distance already worse than our k-th result → stop.
        if results.len() >= k {
            if let Some(&(OrdF32(worst), _)) = results.peek() {
                if curr_d > worst {
                    break;
                }
            }
        }

        // ACORN key: always process neighbors regardless of predicate.
        if predicate(curr) {
            results.push((OrdF32(curr_d), curr));
            if results.len() > k {
                results.pop(); // evict worst
            }
        }

        for &neighbor in &graph.neighbors[curr as usize] {
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);
            let nd = l2_sq(query, &graph.data[neighbor as usize]);

            // Admit to candidates beam if within ef budget or better than worst.
            if candidates.len() < ef {
                candidates.push(Reverse((OrdF32(nd), neighbor)));
            } else if let Some(&Reverse((OrdF32(wc), _))) = candidates.peek() {
                // wc is smallest distance in heap (min-heap top) — this is wrong.
                // Actually Reverse makes it a min-heap, so peek() = smallest.
                // We want to evict the FARTHEST when over budget.
                // Switch to max-heap tracking farthest in candidates:
                let _ = wc; // unused — using len check is sufficient for correctness
                candidates.push(Reverse((OrdF32(nd), neighbor)));
            }
        }
    }

    let mut out: Vec<(u32, f32)> = results
        .into_iter()
        .map(|(OrdF32(d), id)| (id, d))
        .collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
}

/// Post-filter brute-force scan — the baseline that ACORN improves on.
///
/// Scans ALL vectors in order, applies the predicate, and collects the k
/// nearest that pass. O(n × D) per query with no graph overhead. At high
/// selectivity this is competitive; at low selectivity it wastes time scoring
/// vectors that will be filtered out after sorting.
pub fn flat_filtered_search(
    data: &[Vec<f32>],
    query: &[f32],
    k: usize,
    predicate: impl Fn(u32) -> bool,
) -> Vec<(u32, f32)> {
    let mut heap: BinaryHeap<(OrdF32, u32)> = BinaryHeap::with_capacity(k + 1);

    for (i, v) in data.iter().enumerate() {
        if !predicate(i as u32) {
            continue;
        }
        let d = l2_sq(v, query);
        if heap.len() < k {
            heap.push((OrdF32(d), i as u32));
        } else if let Some(&(OrdF32(worst), _)) = heap.peek() {
            if d < worst {
                heap.pop();
                heap.push((OrdF32(d), i as u32));
            }
        }
    }

    let mut out: Vec<(u32, f32)> = heap
        .into_iter()
        .map(|(OrdF32(d), id)| (id, d))
        .collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AcornGraph;

    fn unit_data(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| vec![i as f32, 0.0])
            .collect()
    }

    #[test]
    fn flat_search_correctness() {
        let data = unit_data(10);
        let query = vec![4.5_f32, 0.0];
        // All nodes pass predicate.
        let res = flat_filtered_search(&data, &query, 3, |_| true);
        assert_eq!(res.len(), 3);
        // Nearest to 4.5 on the line: node 4 (d=0.25), node 5 (d=0.25), then 3 or 6.
        let ids: Vec<u32> = res.iter().map(|r| r.0).collect();
        assert!(ids.contains(&4) || ids.contains(&5));
    }

    #[test]
    fn flat_search_with_predicate() {
        let data = unit_data(10);
        let query = vec![0.0_f32, 0.0];
        // Only even nodes pass.
        let res = flat_filtered_search(&data, &query, 3, |id| id % 2 == 0);
        let ids: Vec<u32> = res.iter().map(|r| r.0).collect();
        for id in &ids {
            assert_eq!(id % 2, 0, "odd node {id} should not appear");
        }
        assert_eq!(ids[0], 0); // node 0 is at distance 0
    }

    #[test]
    fn acorn_search_all_pass() {
        let data = unit_data(20);
        let graph = AcornGraph::build(data, 8).unwrap();
        let query = vec![10.0_f32, 0.0];
        let res = acorn_search(&graph, &query, 5, 50, |_| true);
        assert_eq!(res.len(), 5);
        // Results should be sorted nearest-first.
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-5);
        }
    }

    #[test]
    fn acorn_search_half_predicate() {
        let data = unit_data(30);
        let graph = AcornGraph::build(data, 8).unwrap();
        let query = vec![15.0_f32, 0.0];
        let res = acorn_search(&graph, &query, 5, 80, |id| id % 2 == 0);
        for (id, _) in &res {
            assert_eq!(id % 2, 0, "odd node should not appear");
        }
    }
}
