//! Field engine — orchestrates ingest, promotion, retrieval, drift, routing.
//!
//! The engine owns:
//! * an [`EmbeddingStore`] for interned vectors,
//! * a [`LinearIndex`] (or any [`SemanticIndex`]) for candidate generation,
//! * the node + edge graph,
//! * [`TemporalBuckets`] for historical queries,
//! * a [`PolicyRegistry`] for policy fit scoring,
//! * a [`WitnessLog`] that records every mutation exactly once,
//! * a [`Clock`] abstraction for deterministic tests,
//! * [`FieldEngineConfig`] for hysteresis windows and thresholds.
//!
//! See the submodules for the individual phases.

pub mod drift;
pub mod ingest;
pub mod promote;
pub mod retrieve;
pub mod route;

use std::collections::HashMap;
use std::sync::Arc;

use crate::clock::{Clock, SystemClock};
use crate::model::node::AxisKind;
use crate::model::{
    EdgeKind, Embedding, EmbeddingStore, FieldEdge, FieldNode, HintId, NodeId, Shell,
};
use crate::policy::PolicyRegistry;
use crate::scoring::resonance_score;
use crate::storage::{LinearIndex, TemporalBuckets};
use crate::witness::WitnessLog;

pub use promote::{PromotionReason, PromotionRecord};

/// Engine configuration.
///
/// # Example
///
/// ```
/// use ruvector_field::engine::FieldEngineConfig;
/// let cfg = FieldEngineConfig::default();
/// assert!(cfg.hysteresis_window >= 2);
/// ```
#[derive(Debug, Clone)]
pub struct FieldEngineConfig {
    /// Expected node capacity — used to preallocate.
    pub expected_nodes: usize,
    /// Expected edge capacity.
    pub expected_edges: usize,
    /// Window for oscillation detection / promotion hysteresis.
    pub hysteresis_window: usize,
    /// Minimum residence window in ns before a shell can be promoted out of.
    pub min_residence_ns: u64,
    /// Drift alert threshold (total channel sum).
    pub drift_threshold: f32,
    /// Threshold above which an individual drift channel counts as "agreeing".
    pub drift_channel_threshold: f32,
    /// Number of consecutive promotion passes required before a promotion fires.
    pub promotion_passes: u32,
    /// Maximum k for the contradiction frontier walk.
    pub frontier_k: usize,
}

impl Default for FieldEngineConfig {
    fn default() -> Self {
        Self {
            expected_nodes: 64,
            expected_edges: 128,
            hysteresis_window: 4,
            min_residence_ns: 0,
            drift_threshold: 0.4,
            drift_channel_threshold: 0.1,
            promotion_passes: 2,
            frontier_k: 8,
        }
    }
}

/// Field engine.
pub struct FieldEngine {
    /// Configuration.
    pub config: FieldEngineConfig,
    /// Nodes keyed by id.
    pub nodes: HashMap<NodeId, FieldNode>,
    /// Flat edge list.
    pub edges: Vec<FieldEdge>,
    /// Interned embedding store.
    pub store: EmbeddingStore,
    /// Linear semantic index (swap out for HNSW via [`SemanticIndex`]).
    pub index: LinearIndex,
    /// Temporal buckets.
    pub temporal: TemporalBuckets,
    /// Policy registry — may be empty.
    pub policies: PolicyRegistry,
    /// Witness log.
    pub witness: WitnessLog,
    /// Active routing hints by id.
    pub active_hints: HashMap<HintId, crate::scoring::RoutingHint>,
    next_id: u64,
    next_hint_id: u64,
    clock: Arc<dyn Clock>,
    last_tick_ts: u64,
}

impl FieldEngine {
    /// Create an engine with default config and a [`SystemClock`].
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::engine::FieldEngine;
    /// let engine = FieldEngine::new();
    /// assert!(engine.nodes.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::with_config(FieldEngineConfig::default())
    }

    /// Create an engine with the given config and a [`SystemClock`].
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::engine::{FieldEngine, FieldEngineConfig};
    /// let engine = FieldEngine::with_config(FieldEngineConfig::default());
    /// assert_eq!(engine.node_count(), 0);
    /// ```
    pub fn with_config(config: FieldEngineConfig) -> Self {
        Self::with_config_and_clock(config, Arc::new(SystemClock))
    }

    /// Create an engine with a custom clock — used by tests.
    pub fn with_clock(clock: Arc<dyn Clock>) -> Self {
        Self::with_config_and_clock(FieldEngineConfig::default(), clock)
    }

    /// Full constructor.
    pub fn with_config_and_clock(config: FieldEngineConfig, clock: Arc<dyn Clock>) -> Self {
        let nodes = HashMap::with_capacity(config.expected_nodes);
        let edges = Vec::with_capacity(config.expected_edges);
        Self {
            config,
            nodes,
            edges,
            store: EmbeddingStore::new(),
            index: LinearIndex::new(),
            temporal: TemporalBuckets::new(),
            policies: PolicyRegistry::new(),
            witness: WitnessLog::new(),
            active_hints: HashMap::new(),
            next_id: 1,
            next_hint_id: 1,
            clock,
            last_tick_ts: 0,
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Look up a node by id.
    pub fn node(&self, id: NodeId) -> Option<&FieldNode> {
        self.nodes.get(&id)
    }

    /// Mutable node lookup.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut FieldNode> {
        self.nodes.get_mut(&id)
    }

    /// Current clock reading.
    pub fn now_ns(&self) -> u64 {
        self.clock.now_ns()
    }

    pub(crate) fn next_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    pub(crate) fn next_hint_id(&mut self) -> HintId {
        let id = HintId(self.next_hint_id);
        self.next_hint_id += 1;
        id
    }

    /// Tick — recompute coherence, continuity, and axis scores for every node.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::engine::FieldEngine;
    /// let mut engine = FieldEngine::new();
    /// engine.tick(); // tick on an empty engine is a no-op
    /// ```
    pub fn tick(&mut self) {
        let ts = self.now_ns();
        self.recompute_coherence();
        self.recompute_continuity();
        self.update_axis_scores();
        // Refresh resonance across every node.
        for node in self.nodes.values_mut() {
            node.resonance = resonance_score(node);
        }
        self.last_tick_ts = ts;
    }

    /// Apply Laplacian-proxy coherence across all nodes using same-shell
    /// neighbors weighted by support/refines edges.
    pub fn recompute_coherence(&mut self) {
        use crate::scoring::local_coherence;
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        // Precompute support weights per (src,dst).
        let mut support_w: HashMap<(NodeId, NodeId), f32> = HashMap::new();
        for e in &self.edges {
            if matches!(e.kind, EdgeKind::Supports | EdgeKind::Refines) {
                let w = support_w.entry((e.src, e.dst)).or_insert(0.0);
                *w = (*w + e.weight).clamp(0.0, 1.0);
                let w2 = support_w.entry((e.dst, e.src)).or_insert(0.0);
                *w2 = (*w2 + e.weight).clamp(0.0, 1.0);
            }
        }
        for id in node_ids {
            let Some(node) = self.nodes.get(&id).cloned() else { continue };
            let Some(center) = self.store.get(node.semantic_embedding).cloned() else {
                continue;
            };
            // Collect (embedding, weight) pairs for same-shell neighbors.
            let mut owned: Vec<(Embedding, f32)> = Vec::new();
            for other in self.nodes.values() {
                if other.id == id || other.shell != node.shell {
                    continue;
                }
                let Some(e) = self.store.get(other.semantic_embedding) else { continue };
                let support = *support_w.get(&(id, other.id)).unwrap_or(&0.3);
                owned.push((e.clone(), support));
            }
            let neighbors: Vec<(&Embedding, f32)> =
                owned.iter().map(|(e, w)| (e, *w)).collect();
            let coh = local_coherence(&center, &neighbors, 8);
            if let Some(n) = self.nodes.get_mut(&id) {
                n.coherence = coh;
            }
        }
    }

    /// Continuity: `1 / (1 + normalized_edge_churn_since_last_tick)`.
    ///
    /// Nodes whose first tick is now (i.e. `edges_at_last_tick == 0` AND
    /// `last_tick_ts == 0`) get a full continuity of 1.0 — there is no prior
    /// observation to churn against. On subsequent ticks the ratio is
    /// `|now - prev| / max(1, prev)`.
    pub fn recompute_continuity(&mut self) {
        let mut current: HashMap<NodeId, u32> = HashMap::new();
        for e in &self.edges {
            *current.entry(e.src).or_insert(0) += 1;
            *current.entry(e.dst).or_insert(0) += 1;
        }
        let first_tick = self.last_tick_ts == 0;
        for (id, node) in self.nodes.iter_mut() {
            let now = *current.get(id).unwrap_or(&0);
            if first_tick {
                // Baseline: no prior observation. Give full continuity
                // provided the node actually has any structural anchor.
                node.continuity = 1.0;
            } else {
                let churn = (now as i64 - node.edges_at_last_tick as i64).unsigned_abs() as f32;
                let denom = (node.edges_at_last_tick.max(1)) as f32;
                let normalized = churn / denom;
                node.continuity = (1.0 / (1.0 + normalized)).clamp(0.0, 1.0);
            }
            node.edges_at_last_tick = now;
        }
    }

    /// Reinforce or decay axis scores based on usage / contradictions / policy.
    pub fn update_axis_scores(&mut self) {
        let mask = self.policies.is_empty();
        for node in self.nodes.values_mut() {
            // Reinforcement from successful retrievals — clarity and bridge.
            if node.selection_count > 0 {
                node.axes.reinforce(AxisKind::Clarity, 0.01 * node.selection_count as f32);
                node.axes.reinforce(AxisKind::Bridge, 0.005 * node.selection_count as f32);
                node.selection_count = 0;
            }
            // Contradictions hurt limit and care.
            if node.contradiction_hits > 0 {
                node.axes.decay(AxisKind::Limit, 0.02 * node.contradiction_hits as f32);
                node.axes.decay(AxisKind::Care, 0.01 * node.contradiction_hits as f32);
                node.contradiction_hits = 0;
            }
            // Natural slow decay so unused nodes don't stay max forever.
            if mask {
                node.axes.decay(AxisKind::Bridge, 0.0005);
            }
        }
    }

    /// BFS partition distance via `SharesRegion` + `RoutesTo` edges.
    pub fn partition_distance(&self, from: NodeId, to: NodeId) -> u32 {
        if from == to {
            return 0;
        }
        let mut frontier = vec![from];
        let mut seen: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        seen.insert(from);
        for depth in 1..=8u32 {
            let mut next: Vec<NodeId> = Vec::new();
            for n in &frontier {
                for e in &self.edges {
                    if !matches!(e.kind, EdgeKind::SharesRegion | EdgeKind::RoutesTo) {
                        continue;
                    }
                    let neighbor = if e.src == *n {
                        Some(e.dst)
                    } else if e.dst == *n {
                        Some(e.src)
                    } else {
                        None
                    };
                    if let Some(nb) = neighbor {
                        if nb == to {
                            return depth;
                        }
                        if seen.insert(nb) {
                            next.push(nb);
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        // Unreachable — far penalty.
        8
    }

    /// Count edges matching one of `kinds` incident at each node.
    pub fn count_edges(&self, kinds: &[EdgeKind]) -> HashMap<NodeId, usize> {
        let mut out = HashMap::new();
        for e in &self.edges {
            if kinds.contains(&e.kind) {
                *out.entry(e.dst).or_insert(0) += 1;
                *out.entry(e.src).or_insert(0) += 1;
            }
        }
        out
    }

    /// Mean cosine similarity into the shell's centroid — used in drift.
    pub fn shell_centroid(&self, shell: Shell) -> Option<Embedding> {
        let mut acc: Vec<f32> = Vec::new();
        let mut count = 0usize;
        for node in self.nodes.values() {
            if node.shell != shell {
                continue;
            }
            let Some(e) = self.store.get(node.semantic_embedding) else { continue };
            if acc.is_empty() {
                acc = vec![0.0; e.values.len()];
            }
            for (i, v) in e.values.iter().enumerate().take(acc.len()) {
                acc[i] += v;
            }
            count += 1;
        }
        if count == 0 {
            None
        } else {
            for v in &mut acc {
                *v /= count as f32;
            }
            Some(Embedding::new(acc))
        }
    }
}

impl Default for FieldEngine {
    fn default() -> Self {
        Self::new()
    }
}
