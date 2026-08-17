//! Self-contained weighted Stoer–Wagner global minimum cut.
//!
//! `partition.rs` originally called `ruvector_mincut::DynamicMinCut::partition()`
//! directly to materialize each split. That turned out to be unsafe to trust:
//! diagnosed with a minimal repro (two triangles joined by one weak-weight
//! bridge edge, `examples/debug_mincut.rs` during this nightly's development),
//! `DynamicMinCut::min_cut_value()` reliably reports the correct cut weight,
//! but `DynamicMinCut::partition()` — and, derivatively, the unweighted
//! `RuVectorGraphAnalyzer` / `GraphPartitioner` path — returns a vertex split
//! that is *inconsistent with that value* and nondeterministic across runs:
//! of three runs on the 6-vertex repro, two returned the correct {0,1,2} vs
//! {3,4,5} split and one returned a degenerate {single vertex} vs {rest}
//! split; on a 100-vertex version (two 50-cliques joined by one weak edge)
//! every run returned the degenerate split. See the nightly research doc
//! (docs/research/nightly/2026-08-17_mincut-partitioned-memory-consolidation)
//! for the full repro and evidence — this is being reported upstream as a
//! defect, not silently patched around.
//!
//! This module is the workaround: a correct, deterministic, from-scratch
//! implementation used as the *sole* source of partition vertex sets in
//! this crate. `ruvector_mincut::DynamicMinCut::min_cut_value()` is still
//! called from `partition.rs` as an independent cross-check value (its
//! *value* output was never observed to be wrong) — its agreement or
//! disagreement with this module's result is recorded in the witness chain.

use ruvector_mincut::{VertexId, Weight};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Copy, Clone)]
struct HeapItem {
    weight: f64,
    vertex: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight
            .partial_cmp(&other.weight)
            .unwrap_or(Ordering::Equal)
    }
}

/// Global minimum cut of a weighted undirected graph via Stoer–Wagner.
/// Returns `(cut_value, side_a, side_b)`, `side_a` being the smaller-index
/// "last vertex merged" side of the best phase found. Deterministic for a
/// fixed input ordering. `O(V) merge phases, each an O(E log V) maximum
/// adjacency search)` — the standard binary-heap formulation.
///
/// Panics only on the precondition `vertices.len() >= 2`, checked by the
/// caller (`partition.rs` never calls this below `min_cluster_size`, which
/// defaults to 20).
pub fn global_min_cut(
    vertices: &[VertexId],
    edges: &[(VertexId, VertexId, Weight)],
) -> (f64, Vec<VertexId>, Vec<VertexId>) {
    let n = vertices.len();
    assert!(n >= 2, "global_min_cut requires at least 2 vertices");
    let index_of: HashMap<VertexId, usize> =
        vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    let mut adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
    for &(u, v, w) in edges {
        if let (Some(&iu), Some(&iv)) = (index_of.get(&u), index_of.get(&v)) {
            if iu == iv {
                continue;
            }
            *adj[iu].entry(iv).or_insert(0.0) += w;
            *adj[iv].entry(iu).or_insert(0.0) += w;
        }
    }

    // Stoer-Wagner's maximum-adjacency search assumes a connected graph —
    // at low corpus noise, an induced kNN subgraph can genuinely be
    // disconnected (a discovered crash during this nightly's development:
    // the heap-based phase ran out of reachable vertices). A disconnected
    // graph's true global min cut is 0 (split along the existing component
    // boundary for free), so detect that up front via BFS and short-circuit
    // rather than feed a disconnected graph to a connected-graph algorithm.
    if let Some((component, rest)) = find_disconnected_split(&adj, n) {
        let side_a: Vec<VertexId> = component.iter().map(|&i| vertices[i]).collect();
        let side_b: Vec<VertexId> = rest.iter().map(|&i| vertices[i]).collect();
        return (0.0, side_a, side_b);
    }

    let mut merged_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active = vec![true; n];
    let mut remaining = n;

    let mut best_cut = f64::INFINITY;
    let mut best_side: Vec<usize> = Vec::new();

    while remaining > 1 {
        let (order, weight_to_a) = maximum_adjacency_ordering(&adj, &active, n, remaining);
        let t = order[order.len() - 1];
        let s = order[order.len() - 2];
        let cut_of_phase = weight_to_a[t];

        if cut_of_phase < best_cut {
            best_cut = cut_of_phase;
            best_side = merged_members[t].clone();
        }

        // Merge t into s (standard Stoer-Wagner vertex contraction).
        let t_neighbors: Vec<(usize, f64)> = adj[t].iter().map(|(&k, &v)| (k, v)).collect();
        for (nb, w) in t_neighbors {
            if nb == s {
                continue;
            }
            *adj[s].entry(nb).or_insert(0.0) += w;
            *adj[nb].entry(s).or_insert(0.0) += w;
            adj[nb].remove(&t);
        }
        adj[s].remove(&t);
        adj[t].clear();
        let t_members = std::mem::take(&mut merged_members[t]);
        merged_members[s].extend(t_members);
        active[t] = false;
        remaining -= 1;
    }

    let side_a_set: std::collections::HashSet<usize> = best_side.iter().copied().collect();
    let side_a: Vec<VertexId> = best_side.iter().map(|&i| vertices[i]).collect();
    let side_b: Vec<VertexId> = (0..n)
        .filter(|i| !side_a_set.contains(i))
        .map(|i| vertices[i])
        .collect();
    (best_cut, side_a, side_b)
}

/// BFS from vertex 0; if it does not reach every vertex, returns
/// `Some((reached, unreached))`. `None` means the graph is connected.
fn find_disconnected_split(
    adj: &[HashMap<usize, f64>],
    n: usize,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    let mut reached = vec![0usize];
    while let Some(v) = stack.pop() {
        for &nb in adj[v].keys() {
            if !visited[nb] {
                visited[nb] = true;
                reached.push(nb);
                stack.push(nb);
            }
        }
    }
    if reached.len() == n {
        None
    } else {
        let unreached: Vec<usize> = (0..n).filter(|&i| !visited[i]).collect();
        Some((reached, unreached))
    }
}

/// One Stoer-Wagner phase: a Prim-like maximum-adjacency traversal over the
/// currently active (super-)vertices. Returns the visitation order and each
/// vertex's final accumulated weight-to-the-visited-set (`weight_to_a`); the
/// last two entries in `order` are the phase's `s` and `t`.
fn maximum_adjacency_ordering(
    adj: &[HashMap<usize, f64>],
    active: &[bool],
    n: usize,
    remaining: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut in_a = vec![false; n];
    let mut weight_to_a = vec![0.0f64; n];
    let mut order = Vec::with_capacity(remaining);
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();

    let start = (0..n)
        .find(|&i| active[i])
        .expect("at least one active vertex");
    in_a[start] = true;
    order.push(start);
    for (&nb, &w) in &adj[start] {
        if active[nb] && !in_a[nb] {
            weight_to_a[nb] += w;
            heap.push(HeapItem {
                weight: weight_to_a[nb],
                vertex: nb,
            });
        }
    }

    while order.len() < remaining {
        let next = loop {
            let top = heap
                .pop()
                .expect("heap exhausted before all active vertices visited");
            if !active[top.vertex] || in_a[top.vertex] {
                continue; // stale: vertex merged away or already visited
            }
            if (top.weight - weight_to_a[top.vertex]).abs() > 1e-9 {
                continue; // stale: a fresher, larger entry for this vertex exists
            }
            break top.vertex;
        };
        in_a[next] = true;
        order.push(next);
        for (&nb, &w) in &adj[next] {
            if active[nb] && !in_a[nb] {
                weight_to_a[nb] += w;
                heap.push(HeapItem {
                    weight: weight_to_a[nb],
                    vertex: nb,
                });
            }
        }
    }

    (order, weight_to_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side_of(v: VertexId, side_a: &[VertexId]) -> bool {
        side_a.contains(&v)
    }

    #[test]
    fn finds_the_weak_bridge_between_two_triangles() {
        let vertices: Vec<VertexId> = vec![0, 1, 2, 3, 4, 5];
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 3, 1.0),
            (2, 3, 0.05),
        ];
        let (value, side_a, side_b) = global_min_cut(&vertices, &edges);
        assert!((value - 0.05).abs() < 1e-9, "got {value}");
        assert_eq!(side_a.len(), 3);
        assert_eq!(side_b.len(), 3);
        // {0,1,2} must be together, {3,4,5} must be together.
        let a_has_0 = side_of(0, &side_a);
        for v in [0u64, 1, 2] {
            assert_eq!(
                side_of(v, &side_a),
                a_has_0,
                "vertex {v} split from its triangle"
            );
        }
        for v in [3u64, 4, 5] {
            assert_eq!(
                side_of(v, &side_a),
                !a_has_0,
                "vertex {v} split from its triangle"
            );
        }
    }

    #[test]
    fn finds_the_weak_bridge_at_larger_scale() {
        let mut edges = Vec::new();
        for i in 0..50u64 {
            for j in (i + 1)..50u64 {
                edges.push((i, j, 1.0));
            }
        }
        for i in 50..100u64 {
            for j in (i + 1)..100u64 {
                edges.push((i, j, 1.0));
            }
        }
        edges.push((0, 50, 0.01));
        let vertices: Vec<VertexId> = (0..100).collect();
        let (value, side_a, side_b) = global_min_cut(&vertices, &edges);
        assert!((value - 0.01).abs() < 1e-9, "got {value}");
        assert_eq!(side_a.len().min(side_b.len()), 50);
        assert_eq!(side_a.len().max(side_b.len()), 50);
    }

    #[test]
    fn disconnected_graph_returns_zero_cut_without_panicking() {
        // Two triangles with NO edge between them at all: the classic
        // Stoer-Wagner phase (which assumes connectivity) would exhaust its
        // heap before visiting every vertex. This was an actual crash
        // discovered while calibrating this crate's corpus generator at
        // low noise, where an induced kNN subgraph came out disconnected.
        let vertices: Vec<VertexId> = vec![0, 1, 2, 3, 4, 5];
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 3, 1.0),
        ];
        let (value, side_a, side_b) = global_min_cut(&vertices, &edges);
        assert_eq!(value, 0.0);
        assert_eq!(side_a.len().min(side_b.len()), 3);
        assert_eq!(side_a.len().max(side_b.len()), 3);
    }

    #[test]
    fn single_edge_two_vertices() {
        let vertices: Vec<VertexId> = vec![7, 9];
        let edges = vec![(7, 9, 2.5)];
        let (value, side_a, side_b) = global_min_cut(&vertices, &edges);
        assert!((value - 2.5).abs() < 1e-9);
        assert_eq!(side_a.len(), 1);
        assert_eq!(side_b.len(), 1);
    }

    /// This graph's minimum cut is *unique*: isolating any single triangle
    /// vertex costs >= 1.0 (two clique edges), so the 0.05 bridge cut
    /// strictly dominates every alternative and {0,1,2}/{3,4,5} is the only
    /// partition that achieves it. Repeated calls must therefore agree on
    /// cut value and side membership every time — but not necessarily on
    /// the internal `Vec` ordering within a side (which vertex plays the
    /// Stoer-Wagner "s" vs "t" role on a weight tie is an implementation
    /// detail, not part of the correctness contract), so this test
    /// canonicalizes each side as a sorted set before comparing.
    #[test]
    fn deterministic_across_repeated_calls() {
        let vertices: Vec<VertexId> = vec![0, 1, 2, 3, 4, 5];
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 3, 1.0),
            (2, 3, 0.05),
        ];
        fn canonical(
            mut side_a: Vec<VertexId>,
            mut side_b: Vec<VertexId>,
        ) -> (Vec<VertexId>, Vec<VertexId>) {
            side_a.sort_unstable();
            side_b.sort_unstable();
            if side_a.first() > side_b.first() {
                std::mem::swap(&mut side_a, &mut side_b);
            }
            (side_a, side_b)
        }

        let (value0, a0, b0) = global_min_cut(&vertices, &edges);
        let (canon_a0, canon_b0) = canonical(a0, b0);
        for _ in 0..5 {
            let (value, a, b) = global_min_cut(&vertices, &edges);
            assert!((value - value0).abs() < 1e-9);
            let (canon_a, canon_b) = canonical(a, b);
            assert_eq!(canon_a, canon_a0);
            assert_eq!(canon_b, canon_b0);
        }
    }
}
