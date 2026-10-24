//! Reference global minimum-cut algorithm: Stoer-Wagner.
//!
//! This is a self-contained implementation, not an integration with
//! `ruvector-mincut`. `ruvector-mincut` is a mature, feature-gated crate
//! built around dynamic (incremental edge insert/delete) subpolynomial
//! min-cut maintenance, canonical/deterministic partition selection among
//! tied cuts, and provable cut certificates (`certificate::CutCertificate`)
//! — a materially different, larger API surface aimed at a different
//! problem (maintaining a cut under updates) than "compute one global
//! min-cut for a static graph, once, as a timing baseline." Reusing it
//! properly (picking the right builder config, feature flags, and output
//! shape, and confirming its canonical-partition semantics line up with
//! what this benchmark needs for correctness comparison) was judged, after
//! reading `src/certificate/mod.rs` and `src/canonical/mod.rs`, to be a
//! multi-hour integration rather than the 30-45 minute exploration budget
//! available for this nightly run. Stoer-Wagner is well-known, simple to
//! get correct, and isolates the import-time hypothesis (this crate's
//! actual subject) from any question about which min-cut algorithm a real
//! deployment would use — it is not a claim about `ruvector-mincut`'s own
//! performance, and should not be read as one.
//!
//! ## Algorithm
//!
//! Classic Stoer-Wagner (Stoer & Wagner, 1997, "A Simple Min-Cut
//! Algorithm"). Each of `n - 1` "minimum cut phases" grows a set `A` from
//! an arbitrary start vertex by repeatedly adding the active vertex most
//! tightly connected to `A` (a maximum-adjacency / Prim-like search), until
//! all active (possibly already-merged) vertices are in `A`. The weight of
//! edges from the last vertex added (`t`) to the rest of `A` at the moment
//! it was added is the "cut of the phase" — a legitimate `s-t` cut for the
//! current contracted graph (and, by the merge invariant below, for the
//! original graph, between `t`'s original-vertex group and everything
//! else). The global minimum cut is the minimum cut-of-phase over all `n -
//! 1` phases; `t` and `s` (the last two vertices added) are then merged
//! (summing parallel edge weights) before the next phase.
//!
//! Complexity: each phase does a lazy-deletion binary-heap maximum
//! adjacency search, `O((V + E) log V)`; `V - 1` phases, `O(VE log V)`
//! overall for a connected graph with `E` edges after accounting for
//! shrinking active-vertex counts. This is adequate for the graph sizes
//! this benchmark exercises (hundreds to tens of thousands of sparse
//! nodes) but is not intended to compete with `ruvector-mincut`'s
//! subpolynomial dynamic algorithm, nor with Karger-Stein-style randomized
//! approaches, at larger scale.

use crate::graph::{Partition, WeightedGraph};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// The result of a global minimum-cut computation: the cut's total edge
/// weight and the two-sided partition that achieves it.
#[derive(Clone, Debug)]
pub struct MinCutResult {
    pub cut_value: f64,
    pub partition: Partition,
}

#[derive(Clone, Copy, PartialEq)]
struct HeapEntry {
    weight: f64,
    vertex: u32,
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight
            .partial_cmp(&other.weight)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

/// Compute the global minimum cut of `graph` via Stoer-Wagner.
///
/// Panics if `graph.n < 2` (a cut is undefined for 0 or 1 vertices).
pub fn global_min_cut(graph: &WeightedGraph) -> MinCutResult {
    let n = graph.n;
    assert!(n >= 2, "min cut requires at least 2 vertices");

    // adj[v] is None for a vertex that has been merged away.
    let mut adj: Vec<Option<HashMap<u32, f64>>> = vec![Some(HashMap::new()); n];
    for &(u, v, w) in &graph.edges {
        *adj[u as usize].as_mut().unwrap().entry(v).or_insert(0.0) += w;
        *adj[v as usize].as_mut().unwrap().entry(u).or_insert(0.0) += w;
    }

    let mut groups: Vec<Option<Vec<u32>>> = (0..n as u32).map(|v| Some(vec![v])).collect();
    let mut active: Vec<bool> = vec![true; n];
    let mut active_count = n;

    let mut best_cut = f64::INFINITY;
    let mut best_group: Vec<u32> = Vec::new();

    while active_count > 1 {
        let start = (0..n)
            .find(|&v| active[v])
            .expect("active_count > 1 implies an active vertex exists");

        let mut in_a = vec![false; n];
        in_a[start] = true;
        let mut order: Vec<usize> = vec![start];
        let mut w: HashMap<u32, f64> = HashMap::new();
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        for (&nbr, &weight) in adj[start].as_ref().unwrap() {
            if active[nbr as usize] {
                w.insert(nbr, weight);
                heap.push(HeapEntry {
                    weight,
                    vertex: nbr,
                });
            }
        }

        let mut last_cut_value = 0.0f64;

        for _ in 1..active_count {
            // Pop until we find a fresh, not-yet-added entry.
            let next = loop {
                let top = heap.pop();
                match top {
                    None => {
                        // Defensive fallback: should not happen for a connected
                        // graph, but pick any remaining active, not-yet-added
                        // vertex with weight 0 rather than panic.
                        let fallback = (0..n).find(|&v| active[v] && !in_a[v]);
                        break fallback.map(|v| (v as u32, 0.0));
                    }
                    Some(entry) => {
                        let v = entry.vertex;
                        if !in_a[v as usize] && w.get(&v) == Some(&entry.weight) {
                            break Some((v, entry.weight));
                        }
                        // stale entry: discard and keep popping
                    }
                }
            };

            let (v, weight_to_a) = next.expect("connected graph guarantees a next vertex");
            in_a[v as usize] = true;
            order.push(v as usize);
            last_cut_value = weight_to_a;

            for (&nbr, &edge_w) in adj[v as usize].as_ref().unwrap() {
                if active[nbr as usize] && !in_a[nbr as usize] {
                    let updated = w.get(&nbr).copied().unwrap_or(0.0) + edge_w;
                    w.insert(nbr, updated);
                    heap.push(HeapEntry {
                        weight: updated,
                        vertex: nbr,
                    });
                }
            }
        }

        // t = last vertex added this phase; s = the one added just before it.
        let t = order[order.len() - 1];
        let s = order[order.len() - 2];

        if last_cut_value < best_cut {
            best_cut = last_cut_value;
            best_group = groups[t].clone().unwrap();
        }

        merge(&mut adj, &mut groups, &mut active, s, t);
        active_count -= 1;
    }

    let mut side = vec![false; n];
    for v in best_group {
        side[v as usize] = true;
    }

    MinCutResult {
        cut_value: best_cut,
        partition: Partition::new(side),
    }
}

/// Merge vertex `t` into vertex `s`: fold `t`'s adjacency into `s`'s,
/// re-point `t`'s neighbors at `s`, deactivate `t`, and move `t`'s original
/// vertex group into `s`'s group.
fn merge(
    adj: &mut [Option<HashMap<u32, f64>>],
    groups: &mut [Option<Vec<u32>>],
    active: &mut [bool],
    s: usize,
    t: usize,
) {
    let t_adj = adj[t].take().unwrap();
    for (nbr, weight) in t_adj {
        if nbr as usize == s {
            continue;
        }
        *adj[s].as_mut().unwrap().entry(nbr).or_insert(0.0) += weight;
        let nbr_adj = adj[nbr as usize].as_mut().unwrap();
        nbr_adj.remove(&(t as u32));
        *nbr_adj.entry(s as u32).or_insert(0.0) += weight;
    }
    adj[s].as_mut().unwrap().remove(&(t as u32));

    let t_group = groups[t].take().unwrap();
    groups[s].as_mut().unwrap().extend(t_group);
    active[t] = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generate_graph;

    #[test]
    fn two_triangles_bridge_has_cut_one() {
        // Two triangles {0,1,2} and {3,4,5} joined by a single weight-1
        // bridge edge (2,3). The global min cut must be exactly 1, at the
        // bridge.
        let edges = vec![
            (0, 1, 5.0),
            (1, 2, 5.0),
            (0, 2, 5.0),
            (3, 4, 5.0),
            (4, 5, 5.0),
            (3, 5, 5.0),
            (2, 3, 1.0),
        ];
        let g = WeightedGraph { n: 6, edges };
        let result = global_min_cut(&g);
        assert!((result.cut_value - 1.0).abs() < 1e-9);
        // The two triangles must be on opposite sides.
        let side = &result.partition.side;
        assert_eq!(side[0], side[1]);
        assert_eq!(side[1], side[2]);
        assert_eq!(side[3], side[4]);
        assert_eq!(side[4], side[5]);
        assert_ne!(side[0], side[3]);
    }

    #[test]
    fn single_edge_two_vertices() {
        let g = WeightedGraph {
            n: 2,
            edges: vec![(0, 1, 3.5)],
        };
        let result = global_min_cut(&g);
        assert!((result.cut_value - 3.5).abs() < 1e-9);
    }

    #[test]
    fn cut_value_matches_partition_recomputation() {
        for seed in [1u64, 2, 3, 4, 5] {
            let g = generate_graph(80, seed, 6);
            let result = global_min_cut(&g);
            let recomputed = result.partition.cut_value(&g);
            assert!(
                (recomputed - result.cut_value).abs() < 1e-6,
                "seed {seed}: reported cut {} != recomputed cut {}",
                result.cut_value,
                recomputed
            );
        }
    }

    #[test]
    fn complete_graph_k4_unit_weights_cut_is_three() {
        // K4 with unit weights: isolating any single vertex cuts 3 edges.
        let mut edges = Vec::new();
        for u in 0..4u32 {
            for v in (u + 1)..4u32 {
                edges.push((u, v, 1.0));
            }
        }
        let g = WeightedGraph { n: 4, edges };
        let result = global_min_cut(&g);
        assert!((result.cut_value - 3.0).abs() < 1e-9);
    }
}
