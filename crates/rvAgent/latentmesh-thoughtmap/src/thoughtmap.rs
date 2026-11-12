//! The shared thought map (ADR-326 Decision §3, §5).
//!
//! DeAR's confirmed mechanism (arXiv:2608.17282, part 2) is *thought-map
//! navigation for targeted peer interactions*. We persist the evolving
//! collaborative reasoning process as a graph: each **node** is a reasoning
//! step, each **edge** is a targeted peer interaction. Successful reasoning
//! paths **reinforce** their edges; failed paths **weaken** them — an evolving,
//! experience-weighted topology rather than a static one.
//!
//! ## RuVector-integration seam
//!
//! ADR-326 §3 states the thought graph should be persisted in RuVector's durable
//! graph substrate (as ADR-307 memory tiers / ADR-320 causal episodic graph do).
//! This first shippable slice ships a **first-party in-memory graph** so WP23 is
//! testable without blocking on the WP5 transport crates, and exposes a
//! documented seam — the [`ThoughtGraphBackend`] trait — that a RuVector-backed
//! store (`ruvector-graph`) implements later without changing callers. The
//! in-memory [`ThoughtMap`] is one implementation of that seam.

use serde::{Deserialize, Serialize};

use crate::capability::PeerId;

/// Identifier for a reasoning step (a node in the thought map).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StepId(pub u64);

/// A reasoning step. Carries a `latent` signature — the compressed latent state
/// (KV-cache-style, ADR-326 §4) that summarizes the step — used both for
/// navigation and as the anchor for causal attribution of incoming messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThoughtNode {
    pub id: StepId,
    /// The agent whose contribution produced this step.
    pub author: PeerId,
    /// Compressed latent signature of the step.
    pub latent: Vec<f32>,
    /// Optional human-readable summary (debugging / witness records).
    #[serde(default)]
    pub summary: String,
}

/// A directed edge: a targeted peer interaction connecting two reasoning steps.
/// `weight` is the experience-weighted strength; reinforcement raises it, failure
/// lowers it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThoughtEdge {
    pub from: StepId,
    pub to: StepId,
    /// The peer interaction this edge represents.
    pub peer: PeerId,
    /// Experience-weighted strength in `[0, 1]`.
    pub weight: f32,
}

/// The RuVector-integration seam. A RuVector-backed thought graph (e.g. over
/// `ruvector-graph`) implements this without changing any caller; [`ThoughtMap`]
/// is the first-party in-memory implementation used by this slice.
pub trait ThoughtGraphBackend {
    fn add_step(&mut self, author: PeerId, latent: Vec<f32>, summary: impl Into<String>) -> StepId;
    fn connect(&mut self, from: StepId, to: StepId, peer: PeerId, weight: f32);
    fn node(&self, id: StepId) -> Option<&ThoughtNode>;
    fn reinforce_path(&mut self, path: &[StepId], delta: f32);
    fn weaken_path(&mut self, path: &[StepId], delta: f32);
}

/// First-party in-memory thought map (the default [`ThoughtGraphBackend`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThoughtMap {
    nodes: Vec<ThoughtNode>,
    edges: Vec<ThoughtEdge>,
    next_step: u64,
    /// Edges below this weight are treated as pruned for navigation: a node whose
    /// every outgoing edge is below it is a dead end.
    pub dead_edge_threshold: f32,
}

impl ThoughtMap {
    pub fn new() -> Self {
        Self {
            dead_edge_threshold: 0.15,
            ..Default::default()
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Look up a reasoning step by id. Inherent method so concrete callers don't
    /// need [`ThoughtGraphBackend`] in scope; the trait impl delegates here.
    pub fn node(&self, id: StepId) -> Option<&ThoughtNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn edges(&self) -> &[ThoughtEdge] {
        &self.edges
    }

    /// The signatures of every node except `exclude` — the control population for
    /// controlled-replacement causal attribution (ADR-310). See [`crate::causal`].
    pub fn control_signatures(&self, exclude: StepId) -> Vec<Vec<f32>> {
        self.nodes
            .iter()
            .filter(|n| n.id != exclude)
            .map(|n| n.latent.clone())
            .collect()
    }

    /// Look up the strength of a specific edge (for assertions/tests).
    pub fn edge_weight(&self, from: StepId, to: StepId) -> Option<f32> {
        self.edges
            .iter()
            .find(|e| e.from == from && e.to == to)
            .map(|e| e.weight)
    }

    /// Outgoing edges of a node, strongest first.
    pub fn neighbors(&self, node: StepId) -> Vec<&ThoughtEdge> {
        let mut out: Vec<&ThoughtEdge> = self.edges.iter().filter(|e| e.from == node).collect();
        out.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        out
    }

    /// A node is a dead end when it has no outgoing edge strong enough to follow.
    /// (Dead-end *handling* — the continue-vs-restart design choice — lives in
    /// [`crate::topology`], deliberately separated from detection.)
    pub fn is_dead_end(&self, node: StepId) -> bool {
        !self
            .edges
            .iter()
            .any(|e| e.from == node && e.weight >= self.dead_edge_threshold)
    }
}

impl ThoughtGraphBackend for ThoughtMap {
    fn add_step(&mut self, author: PeerId, latent: Vec<f32>, summary: impl Into<String>) -> StepId {
        let id = StepId(self.next_step);
        self.next_step += 1;
        // Defense-in-depth: never store a non-finite latent, even if a caller
        // bypassed the `incorporate` choke point. A non-finite element is
        // sanitized to 0.0 so the map's signatures — and the control population
        // derived from them — stay finite and the causal gate cannot silently
        // degrade. The real fix rejects such messages before they reach here.
        let latent: Vec<f32> = latent
            .into_iter()
            .map(|x| if x.is_finite() { x } else { 0.0 })
            .collect();
        self.nodes.push(ThoughtNode {
            id,
            author,
            latent,
            summary: summary.into(),
        });
        id
    }

    fn connect(&mut self, from: StepId, to: StepId, peer: PeerId, weight: f32) {
        self.edges.push(ThoughtEdge {
            from,
            to,
            peer,
            weight: weight.clamp(0.0, 1.0),
        });
    }

    fn node(&self, id: StepId) -> Option<&ThoughtNode> {
        ThoughtMap::node(self, id)
    }

    /// Reinforce every edge along a successful path.
    fn reinforce_path(&mut self, path: &[StepId], delta: f32) {
        for pair in path.windows(2) {
            if let Some(edge) = self
                .edges
                .iter_mut()
                .find(|e| e.from == pair[0] && e.to == pair[1])
            {
                edge.weight = (edge.weight + delta).clamp(0.0, 1.0);
            }
        }
    }

    /// Weaken every edge along a failed path.
    fn weaken_path(&mut self, path: &[StepId], delta: f32) {
        for pair in path.windows(2) {
            if let Some(edge) = self
                .edges
                .iter_mut()
                .find(|e| e.from == pair[0] && e.to == pair[1])
            {
                edge.weight = (edge.weight - delta).clamp(0.0, 1.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reinforce_and_weaken_move_edge_weights() {
        let mut map = ThoughtMap::new();
        let a = map.add_step(PeerId(1), vec![1.0, 0.0], "start");
        let b = map.add_step(PeerId(2), vec![0.9, 0.1], "next");
        map.connect(a, b, PeerId(2), 0.5);

        map.reinforce_path(&[a, b], 0.3);
        assert!((map.edge_weight(a, b).unwrap() - 0.8).abs() < 1e-6);

        map.weaken_path(&[a, b], 0.6);
        assert!((map.edge_weight(a, b).unwrap() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn weights_stay_clamped() {
        let mut map = ThoughtMap::new();
        let a = map.add_step(PeerId(1), vec![1.0], "a");
        let b = map.add_step(PeerId(2), vec![1.0], "b");
        map.connect(a, b, PeerId(2), 0.9);
        map.reinforce_path(&[a, b], 5.0);
        assert_eq!(map.edge_weight(a, b).unwrap(), 1.0);
        map.weaken_path(&[a, b], 5.0);
        assert_eq!(map.edge_weight(a, b).unwrap(), 0.0);
    }

    #[test]
    fn dead_end_detection() {
        let mut map = ThoughtMap::new();
        let a = map.add_step(PeerId(1), vec![1.0], "a");
        let b = map.add_step(PeerId(2), vec![1.0], "b");
        assert!(map.is_dead_end(a), "no outgoing edges → dead end");
        map.connect(a, b, PeerId(2), 0.05); // below threshold
        assert!(map.is_dead_end(a), "only a weak edge → still a dead end");
        map.connect(a, b, PeerId(2), 0.5); // strong
        assert!(!map.is_dead_end(a));
    }
}
