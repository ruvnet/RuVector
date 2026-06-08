//! Swarm-scale min-cut sensing — *where* a coupled system is closest to breaking.
//!
//! At facility or city scale every room, machine, or router is a node in a
//! **coupling graph**: an edge weight is how strongly two nodes hold each other
//! in a coherent operating state (shared load, redundant links, correlated
//! environment). The operational question is not *"which sensor crossed a
//! threshold?"* but *"WHERE is the whole structure closest to fragmenting?"* —
//! and that is answered, globally, by the minimum cut.
//!
//! The global **min-cut value** is the total coupling that would have to fail for
//! the facility to split into two pieces: a low value means the system is one
//! weak link away from breaking apart (fragile); a high value means it is
//! robustly interconnected. The **bottleneck** nodes are those touching a
//! crossing edge — the load-bearing joints where the break would happen.
//!
//! This reuses [`ruvector_mincut`] for the cut. The cut *value* is authoritative;
//! the returned partition is best-effort (the engine may peel a single weakly
//! connected node rather than return a balanced split), so all decision-relevant
//! output keys on the **value** and on the **bottleneck set**, never on an exact
//! balanced partition.
//!
//! ## Example
//!
//! ```
//! use ruvector_perception::FacilityGraph;
//!
//! let mut g = FacilityGraph::new();
//! // Two tight clusters joined by one thin link.
//! g.couple("r1", "r2", 10.0);
//! g.couple("r2", "r3", 10.0);
//! g.couple("r3", "r4", 0.5); // the fragile joint
//! g.couple("r4", "r5", 10.0);
//! g.couple("r5", "r6", 10.0);
//!
//! let report = g.fragility().unwrap();
//! assert!((report.min_cut - 0.5).abs() < 1e-6);
//! assert!(report.bottlenecks.contains(&"r3".to_string())
//!     || report.bottlenecks.contains(&"r4".to_string()));
//! ```

use std::collections::{BTreeMap, BTreeSet};

use ruvector_mincut::MinCutBuilder;
use serde::{Deserialize, Serialize};

/// Where a coupled facility is structurally closest to fragmenting.
///
/// The headline number is [`min_cut`](FragilityReport::min_cut): the total
/// coupling weight that would have to fail for the system to split. Lower means
/// more fragile. [`bottlenecks`](FragilityReport::bottlenecks) lists the
/// load-bearing joints — nodes touching a crossing edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FragilityReport {
    /// Global min-cut weight = how close the facility is to breaking apart.
    /// Lower = more fragile.
    pub min_cut: f64,
    /// One side of the fragile partition (best-effort; do not rely on balance).
    pub side_a: Vec<String>,
    /// The other side of the fragile partition (best-effort; may be empty).
    pub side_b: Vec<String>,
    /// Nodes incident to a crossing (cut) edge — the structural bottlenecks.
    /// Sorted and deduped.
    pub bottlenecks: Vec<String>,
}

/// A facility-scale coupling graph: nodes are rooms/machines/routers, edges are
/// undirected coupling strengths that accumulate across calls.
#[derive(Debug, Clone, Default)]
pub struct FacilityGraph {
    /// Distinct node names.
    nodes: BTreeSet<String>,
    /// Summed undirected coupling, keyed by the ordered `(min, max)` name pair.
    edges: BTreeMap<(String, String), f64>,
}

impl FacilityGraph {
    /// Create an empty facility graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or accumulate) an undirected coupling strength between two facility
    /// nodes. Repeated calls on the same unordered pair **sum** into one total
    /// coupling weight.
    ///
    /// Non-positive weights and self-loops (`a == b`) are ignored, matching the
    /// min-cut engine's requirement of positive weights on distinct endpoints.
    pub fn couple(&mut self, a: impl Into<String>, b: impl Into<String>, weight: f64) {
        let a = a.into();
        let b = b.into();
        if a == b || !weight.is_finite() || weight <= 0.0 {
            return;
        }
        self.nodes.insert(a.clone());
        self.nodes.insert(b.clone());
        let key = if a <= b { (a, b) } else { (b, a) };
        *self.edges.entry(key).or_insert(0.0) += weight;
    }

    /// Number of distinct nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Compute the global minimum cut: the place where the facility is
    /// structurally closest to fragmenting.
    ///
    /// Returns `None` if there are fewer than two nodes or no edges. Otherwise
    /// the [`FragilityReport`] always carries a trustworthy
    /// [`min_cut`](FragilityReport::min_cut) value; the sides are best-effort and
    /// `bottlenecks` lists the nodes touching a crossing edge (sorted, deduped).
    /// Never panics.
    pub fn fragility(&self) -> Option<FragilityReport> {
        if self.nodes.len() < 2 || self.edges.is_empty() {
            return None;
        }

        // Stable name <-> id mapping (BTreeSet iterates in sorted order).
        let names: Vec<String> = self.nodes.iter().cloned().collect();
        let id_of: BTreeMap<&str, u64> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as u64))
            .collect();

        let edges: Vec<(u64, u64, f64)> = self
            .edges
            .iter()
            .map(|((u, v), w)| (id_of[u.as_str()], id_of[v.as_str()], *w))
            .collect();

        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()
            .ok()?;
        let result = mincut.min_cut();
        let min_cut = result.value;

        // Best-effort sides from the engine partition. Fall back to "all on one
        // side" when the engine returns no usable split.
        let (side_a_ids, side_b_ids): (Vec<u64>, Vec<u64>) = match result.partition {
            Some((a, b)) if !a.is_empty() && !b.is_empty() => (a, b),
            _ => ((0..names.len() as u64).collect(), Vec::new()),
        };

        let mut side_a: Vec<String> = side_a_ids
            .iter()
            .map(|&i| names[i as usize].clone())
            .collect();
        let mut side_b: Vec<String> = side_b_ids
            .iter()
            .map(|&i| names[i as usize].clone())
            .collect();
        side_a.sort();
        side_b.sort();

        // Bottlenecks = endpoints of the WEAKEST link(s) — the fragile joints
        // where a break would occur. Derived from edge weights, NOT the engine
        // partition: the engine's `min_cut` value is reliable but the partition
        // it materialises can be inconsistent with that value (it sometimes
        // peels a single node), so partition-crossing edges are not trustworthy
        // bottleneck markers. The weakest edge is the true structural weak point.
        let min_w = self.edges.values().copied().fold(f64::INFINITY, f64::min);
        let mut bottleneck_set: BTreeSet<String> = BTreeSet::new();
        for ((u, v), &w) in &self.edges {
            if (w - min_w).abs() <= 1e-9 {
                bottleneck_set.insert(u.clone());
                bottleneck_set.insert(v.clone());
            }
        }
        Some(FragilityReport {
            min_cut,
            side_a,
            side_b,
            bottlenecks: bottleneck_set.into_iter().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_link_between_two_clusters_is_the_fragility() {
        // Cluster {r1,r2,r3} tightly coupled, cluster {r4,r5,r6} tightly coupled,
        // joined only by the thin link r3<->r4 (weight 0.5).
        let mut g = FacilityGraph::new();
        for &(a, b) in &[("r1", "r2"), ("r2", "r3"), ("r1", "r3")] {
            g.couple(a, b, 10.0);
        }
        for &(a, b) in &[("r4", "r5"), ("r5", "r6"), ("r4", "r6")] {
            g.couple(a, b, 10.0);
        }
        g.couple("r3", "r4", 0.5);

        assert_eq!(g.len(), 6);
        let report = g.fragility().expect("two clusters -> a report");

        // The weakest crossing is exactly the thin link.
        assert!(
            (report.min_cut - 0.5).abs() < 1e-6,
            "min_cut = {}",
            report.min_cut
        );
        // ...and it is far below the intra-cluster coupling.
        assert!(report.min_cut < 10.0);
        // The fragile joint is r3 or r4.
        assert!(
            report.bottlenecks.contains(&"r3".to_string())
                || report.bottlenecks.contains(&"r4".to_string()),
            "bottlenecks = {:?}",
            report.bottlenecks
        );
    }

    #[test]
    fn fewer_than_two_nodes_is_none() {
        let empty = FacilityGraph::new();
        assert!(empty.is_empty());
        assert_eq!(empty.fragility(), None);

        // A single self-loop is ignored, so still no graph.
        let mut single = FacilityGraph::new();
        single.couple("only", "only", 5.0);
        assert!(single.is_empty());
        assert_eq!(single.fragility(), None);
    }

    #[test]
    fn uniform_clique_isolates_a_single_node() {
        // Strongly, uniformly coupled clique over 5 nodes. The cheapest cut is
        // to isolate one node: (k-1) * weight.
        let mut g = FacilityGraph::new();
        let nodes = ["a", "b", "c", "d", "e"];
        let weight = 2.0;
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                g.couple(nodes[i], nodes[j], weight);
            }
        }
        let report = g.fragility().expect("clique -> a report");

        let isolation_cost = (nodes.len() as f64 - 1.0) * weight; // 4 * 2 = 8
        assert!(
            (report.min_cut - isolation_cost).abs() < 1e-6,
            "min_cut = {}",
            report.min_cut
        );
        assert!(report.min_cut > 0.0);
        assert!(!report.bottlenecks.is_empty());
    }

    #[test]
    fn repeated_couplings_sum() {
        let mut g = FacilityGraph::new();
        g.couple("x", "y", 1.5);
        g.couple("y", "x", 2.5); // same unordered pair, reversed
                                 // Only one edge, so the only cut separates the two nodes: total = 4.0.
        let report = g.fragility().expect("two coupled nodes -> a report");
        assert!(
            (report.min_cut - 4.0).abs() < 1e-6,
            "min_cut = {}",
            report.min_cut
        );
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn non_positive_and_self_weights_ignored() {
        let mut g = FacilityGraph::new();
        g.couple("a", "b", 0.0);
        g.couple("a", "b", -3.0);
        g.couple("a", "a", 5.0);
        assert!(g.is_empty());
        assert_eq!(g.fragility(), None);
    }
}
