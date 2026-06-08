//! # Self-healing sensor topology
//!
//! Sensors are not equal, and which ones *matter* changes over time. This module
//! keeps a running **agreement graph** between sensors (how often each pair
//! corroborates the other) and lets that graph reorganise itself so the system
//! can answer one operational question: *what is each sensor's structural role
//! right now?*
//!
//! Every node is classified into one of four [`NodeRole`]s:
//!
//! - **Critical** — removing it would fragment the topology. It is the sole (or
//!   near-sole) strong link bridging two otherwise-disconnected clusters.
//!   Detected with a dynamic global **min-cut**: a node on the min-cut boundary
//!   that carries a crossing edge and has few strong alternatives is a bridge.
//! - **Redundant** — it has a near-duplicate peer (very high agreement with at
//!   least one other sensor), so it could be put to sleep without losing
//!   coverage.
//! - **Noisy** — it disagrees with essentially everyone (low mean agreement);
//!   its readings are not corroborated and should be discounted.
//! - **Normal** — none of the above.
//!
//! The agreement between two sensors is accumulated as an **EWMA** (exponential
//! weighted moving average, `alpha = 0.3`) over repeated [`record_agreement`]
//! calls, so the topology drifts toward recent behaviour while staying stable.
//!
//! ```
//! use ruvector_perception::topology::{TopologyManager, NodeRole};
//!
//! let mut topo = TopologyManager::new();
//! topo.record_agreement("cam_a", "cam_b", 0.95); // near-duplicates
//! topo.record_agreement("cam_a", "mic_x", 0.6);
//! topo.record_agreement("cam_b", "mic_x", 0.6);
//! let report = topo.assess();
//! assert!(report.iter().any(|a| a.role == NodeRole::Redundant));
//! ```
//!
//! [`record_agreement`]: TopologyManager::record_agreement

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// EWMA smoothing factor for accumulated pairwise agreement.
const ALPHA: f32 = 0.3;
/// Minimum agreement weight for an edge to count as a topology link at all.
/// Edges below this floor are treated as "no meaningful link".
const EDGE_FLOOR: f32 = 0.05;
/// Below this *mean* incident agreement a node is considered [`NodeRole::Noisy`].
const NOISY_MEAN: f32 = 0.3;
/// At or above this *max* incident agreement a node has a near-duplicate peer
/// and is considered [`NodeRole::Redundant`].
const REDUNDANT_MAX: f32 = 0.85;
/// Minimum number of sensors for articulation (bridge) detection to be meaningful.
const MIN_NODES_FOR_BRIDGE: usize = 3;

/// Structural role of a sensor within the agreement topology.
///
/// Ordering of precedence when more than one rule fires is documented on
/// [`TopologyManager::assess`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRole {
    /// Bridges two clusters; its loss would fragment the topology.
    Critical,
    /// Has a near-duplicate peer and could be put to sleep.
    Redundant,
    /// Disagrees with (almost) everyone; readings are uncorroborated.
    Noisy,
    /// No special structural role.
    Normal,
}

/// Per-node assessment produced by [`TopologyManager::assess`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeAssessment {
    /// Sensor name.
    pub node: String,
    /// Classified structural role.
    pub role: NodeRole,
    /// Maximum incident agreement (how close its nearest peer is, in `[0, 1]`).
    pub redundancy: f32,
    /// Mean incident agreement across all of its links, in `[0, 1]`.
    pub agreement: f32,
}

/// Maintains a self-healing sensor agreement graph and classifies node roles.
///
/// Pairwise agreement is stored once per unordered pair, keyed by the
/// lexicographically ordered `(min, max)` name tuple, and accumulated as an
/// EWMA. The set of known sensor names is tracked separately so isolated
/// sensors (no edges yet) are still assessed.
#[derive(Debug, Clone, Default)]
pub struct TopologyManager {
    /// EWMA agreement per unordered pair, key = `(min_name, max_name)`.
    edges: BTreeMap<(String, String), f32>,
    /// All sensor names ever observed.
    nodes: BTreeSet<String>,
}

impl TopologyManager {
    /// Create an empty topology manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record (and accumulate) a pairwise agreement score in `[0, 1]` between
    /// two sensors.
    ///
    /// Repeated calls update the stored value as an EWMA
    /// (`new = alpha * score + (1 - alpha) * old`), so the topology adapts to
    /// recent behaviour. The score is clamped to `[0, 1]`. Self-pairs
    /// (`a == b`) are ignored. Both names are registered as known sensors even
    /// if the pair is a self-pair.
    pub fn record_agreement(&mut self, a: impl Into<String>, b: impl Into<String>, score: f32) {
        let a = a.into();
        let b = b.into();
        self.nodes.insert(a.clone());
        self.nodes.insert(b.clone());
        if a == b {
            return; // no self-loops
        }
        let score = score.clamp(0.0, 1.0);
        let key = if a <= b { (a, b) } else { (b, a) };
        self.edges
            .entry(key)
            .and_modify(|w| *w = ALPHA * score + (1.0 - ALPHA) * *w)
            .or_insert(score);
    }

    /// Number of known sensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether no sensors are known yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Assess every node's role from the accumulated agreement graph.
    ///
    /// Output is sorted by node name for determinism. Role precedence when
    /// multiple rules could apply: **Critical > Redundant > Noisy > Normal**.
    /// (A bridge that also happens to be noisy is reported Critical, because its
    /// structural fragility dominates the operational decision.)
    ///
    /// Graceful degenerate handling: with fewer than two sensors, or with no
    /// edges above [`EDGE_FLOOR`], every node is [`NodeRole::Normal`] — except a
    /// truly isolated node (no incident links at all) which is reported
    /// [`NodeRole::Noisy`], since nothing corroborates it. Never panics.
    #[must_use]
    pub fn assess(&self) -> Vec<NodeAssessment> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // Stable index for each sensor (BTreeSet iterates in sorted order).
        let names: Vec<String> = self.nodes.iter().cloned().collect();
        let index: BTreeMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let n = names.len();

        // Incident weights per node (only edges above the floor count).
        let mut incident: Vec<Vec<f32>> = vec![Vec::new(); n];
        // Adjacency restricted to floor-passing edges, for the bridge rule.
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for ((a, b), &w) in &self.edges {
            if w < EDGE_FLOOR {
                continue;
            }
            let (ia, ib) = (index[a.as_str()], index[b.as_str()]);
            incident[ia].push(w);
            incident[ib].push(w);
            adj[ia].push((ib, w));
            adj[ib].push((ia, w));
        }

        // Identify the min-cut bridge boundary once for the whole graph.
        let critical = self.critical_nodes(n, &adj);

        names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let inc = &incident[i];
                let (agreement, redundancy) = if inc.is_empty() {
                    (0.0_f32, 0.0_f32)
                } else {
                    let sum: f32 = inc.iter().sum();
                    let mean = sum / inc.len() as f32;
                    let max = inc.iter().copied().fold(0.0_f32, f32::max);
                    (mean, max)
                };

                let role = if critical.contains(&i) {
                    NodeRole::Critical
                } else if redundancy >= REDUNDANT_MAX {
                    NodeRole::Redundant
                } else if inc.is_empty() || agreement < NOISY_MEAN {
                    NodeRole::Noisy
                } else {
                    NodeRole::Normal
                };

                NodeAssessment {
                    node: name.clone(),
                    role,
                    redundancy,
                    agreement,
                }
            })
            .collect()
    }

    /// Determine which node indices are **structural bridges** (articulation
    /// points): a node whose removal fragments the strong-edge agreement graph
    /// into more connected components than before. A bridge is the extreme,
    /// most fragile cut — a single-edge min cut — so losing such a node splits
    /// the topology.
    ///
    /// This is robust where a global-min-cut partition is not: it directly tests
    /// "does removing this node disconnect the graph?", which cleanly separates a
    /// true inter-cluster bridge (Critical) from a lone outlier that merely
    /// peels off (Noisy/Redundant). Isolated nodes (no strong edges) are never
    /// Critical. Needs at least [`MIN_NODES_FOR_BRIDGE`] sensors to be meaningful.
    fn critical_nodes(&self, n: usize, adj: &[Vec<(usize, f32)>]) -> BTreeSet<usize> {
        let mut critical = BTreeSet::new();
        if n < MIN_NODES_FOR_BRIDGE {
            return critical;
        }
        let base = components(n, adj, None);
        for v in 0..n {
            if adj[v].is_empty() {
                continue; // isolated node can't be a bridge
            }
            if components(n, adj, Some(v)) > base {
                critical.insert(v);
            }
        }
        critical
    }
}

/// Count connected components among non-isolated nodes, optionally excluding one
/// `removed` node (and its incident edges). Used for articulation detection.
fn components(n: usize, adj: &[Vec<(usize, f32)>], removed: Option<usize>) -> usize {
    let mut visited = vec![false; n];
    if let Some(r) = removed {
        visited[r] = true;
    }
    let mut comps = 0;
    for start in 0..n {
        if visited[start] || adj[start].is_empty() {
            continue; // skip visited and truly isolated nodes
        }
        comps += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(u) = stack.pop() {
            for &(w, _) in &adj[u] {
                if Some(w) == removed || visited[w] {
                    continue;
                }
                visited[w] = true;
                stack.push(w);
            }
        }
    }
    comps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn role_of<'a>(report: &'a [NodeAssessment], node: &str) -> &'a NodeRole {
        &report
            .iter()
            .find(|a| a.node == node)
            .unwrap_or_else(|| panic!("node {node} missing from report"))
            .role
    }

    #[test]
    fn empty_manager_is_empty_and_safe() {
        let topo = TopologyManager::new();
        assert!(topo.is_empty());
        assert_eq!(topo.len(), 0);
        assert!(topo.assess().is_empty()); // no panic, empty result
    }

    #[test]
    fn near_duplicate_peer_is_redundant() {
        let mut topo = TopologyManager::new();
        // a and b are near-duplicates; both also moderately agree with c.
        topo.record_agreement("a", "b", 0.95);
        topo.record_agreement("a", "c", 0.6);
        topo.record_agreement("b", "c", 0.6);
        let report = topo.assess();

        // At least one of the duplicate pair is flagged Redundant.
        let redundant = report
            .iter()
            .filter(|x| x.role == NodeRole::Redundant)
            .count();
        assert!(redundant >= 1, "expected a redundant node, got {report:?}");
        // The redundant node should be a or b (high mutual agreement).
        for a in &report {
            if a.role == NodeRole::Redundant {
                assert!(a.node == "a" || a.node == "b", "unexpected redundant {a:?}");
                assert!(a.redundancy >= REDUNDANT_MAX);
            }
        }
    }

    #[test]
    fn lone_disagreeing_node_is_noisy() {
        let mut topo = TopologyManager::new();
        // x corroborates y and z strongly; n disagrees with all (~0.1).
        topo.record_agreement("x", "y", 0.8);
        topo.record_agreement("x", "z", 0.8);
        topo.record_agreement("y", "z", 0.8);
        topo.record_agreement("n", "x", 0.1);
        topo.record_agreement("n", "y", 0.1);
        topo.record_agreement("n", "z", 0.1);
        let report = topo.assess();

        assert_eq!(
            *role_of(&report, "n"),
            NodeRole::Noisy,
            "report: {report:?}"
        );
        // The well-corroborated nodes are not Noisy.
        assert_ne!(*role_of(&report, "x"), NodeRole::Noisy);
    }

    #[test]
    fn bridge_node_between_two_clusters_is_critical() {
        let mut topo = TopologyManager::new();
        // Cluster 1: {a, b, c} tightly agree.
        topo.record_agreement("a", "b", 0.95);
        topo.record_agreement("a", "c", 0.95);
        topo.record_agreement("b", "c", 0.95);
        // Cluster 2: {d, e, f} tightly agree.
        topo.record_agreement("d", "e", 0.95);
        topo.record_agreement("d", "f", 0.95);
        topo.record_agreement("e", "f", 0.95);
        // Single fragile link joining the clusters: c <-> d.
        topo.record_agreement("c", "d", 0.6);

        let report = topo.assess();
        let critical: Vec<&str> = report
            .iter()
            .filter(|x| x.role == NodeRole::Critical)
            .map(|x| x.node.as_str())
            .collect();

        // The bridge endpoints (c and d) carry the sole crossing link and have
        // strongly-connected same-side peers; the min-cut should isolate them
        // as the boundary, marking at least one bridge endpoint Critical.
        assert!(
            critical.contains(&"c") || critical.contains(&"d"),
            "expected a bridge node (c or d) to be Critical, got critical={critical:?} report={report:?}"
        );
    }

    #[test]
    fn ewma_accumulates_repeated_scores() {
        let mut topo = TopologyManager::new();
        topo.record_agreement("p", "q", 1.0); // first observation -> stored as-is
        topo.record_agreement("p", "q", 0.0); // EWMA pulls it down
        let report = topo.assess();
        let p = report.iter().find(|x| x.node == "p").unwrap();
        // After 1.0 then 0.0: 0.3*0.0 + 0.7*1.0 = 0.7.
        assert!((p.agreement - 0.7).abs() < 1e-4, "got {}", p.agreement);
    }

    #[test]
    fn output_is_sorted_by_name() {
        let mut topo = TopologyManager::new();
        topo.record_agreement("zebra", "alpha", 0.5);
        topo.record_agreement("mid", "alpha", 0.5);
        let report = topo.assess();
        let names: Vec<&str> = report.iter().map(|a| a.node.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mid", "zebra"]);
    }

    #[test]
    fn single_isolated_sensor_is_noisy_not_panic() {
        let mut topo = TopologyManager::new();
        topo.record_agreement("solo", "solo", 0.9); // self-pair ignored as edge
        let report = topo.assess();
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].role, NodeRole::Noisy); // nothing corroborates it
    }
}
