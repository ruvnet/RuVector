//! Deterministic simulation timeline for an evolving agent memory graph:
//! memory arrivals, coherence-weighted compaction, and injected structural
//! drift (bridge-edge decay between topic regions). The exact same arrival
//! and drift *rules* are replayed against every variant's own graph so
//! comparisons stay fair even after a repair variant's graph structurally
//! diverges from the baseline.

use crate::embedding::{Embedding, TopicSpace};
use crate::graph::MemoryGraph;
use crate::scoring::{lowest_scoring, ScoringWeights};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub dim: usize,
    pub n_topics: usize,
    pub total_steps: u64,
    pub semantic_k: usize,
    pub embedding_noise: f32,
    pub compaction_interval: u64,
    pub compaction_evict_count: usize,
    pub drift_interval: u64,
    pub drift_decay_factor: f64,
    pub drift_max_edges_per_wave: usize,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dim: 24,
            n_topics: 8,
            total_steps: 3000,
            semantic_k: 5,
            embedding_noise: 0.12,
            compaction_interval: 25,
            compaction_evict_count: 3,
            drift_interval: 60,
            drift_decay_factor: 0.15,
            drift_max_edges_per_wave: 40,
            seed: 42,
        }
    }
}

/// A ground-truth drift injection: which topic pair had its bridge edges
/// weakened at which step. Used only for scoring monitor detection quality,
/// never fed into the repair logic itself.
#[derive(Debug, Clone, Copy)]
pub struct DriftEvent {
    pub step: u64,
    pub topic_a: usize,
    pub topic_b: usize,
}

/// Pre-generates the full memory arrival stream once, up front, from a
/// single RNG pass — so it is byte-identical across every variant no matter
/// how their graphs later diverge via repair.
pub fn build_arrivals(cfg: &SimConfig) -> (TopicSpace, Vec<Embedding>) {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let space = TopicSpace::new(cfg.dim, cfg.n_topics, &mut rng);
    let arrivals = (0..cfg.total_steps)
        .map(|_| {
            let topic = rng.gen_range(0..cfg.n_topics);
            space.sample(topic, cfg.embedding_noise, &mut rng)
        })
        .collect();
    (space, arrivals)
}

/// Deterministic drift schedule: which topic pair gets targeted at each
/// drift wave, cycling through all pairs so every topic eventually
/// experiences a weakened bridge.
pub fn build_drift_schedule(cfg: &SimConfig) -> Vec<DriftEvent> {
    let mut pairs = Vec::new();
    for a in 0..cfg.n_topics {
        for b in (a + 1)..cfg.n_topics {
            pairs.push((a, b));
        }
    }
    if pairs.is_empty() {
        return Vec::new();
    }
    let mut events = Vec::new();
    let mut step = cfg.drift_interval;
    let mut i = 0;
    while step < cfg.total_steps {
        let (a, b) = pairs[i % pairs.len()];
        events.push(DriftEvent {
            step,
            topic_a: a,
            topic_b: b,
        });
        i += 1;
        step += cfg.drift_interval;
    }
    events
}

/// Adds one new memory at `step`, wiring it to its top-k alive semantic
/// neighbors plus a temporal edge to the immediately preceding node.
pub fn apply_arrival(graph: &mut MemoryGraph, embedding: Embedding, step: u64, cfg: &SimConfig) {
    let id = graph.add_node(embedding, step);
    let neighbors = graph.semantic_topk(id, cfg.semantic_k);
    let mut original = Vec::new();
    for (n, sim) in neighbors {
        if sim > 0.0 {
            graph.add_edge(id, n, sim);
            original.push(n);
        }
    }
    if id > 0 && graph.nodes[id - 1].alive {
        graph.add_edge(id, id - 1, 0.6);
        original.push(id - 1);
    }
    graph.nodes[id].original_neighbors = original;
}

/// Coherence-weighted compaction: evicts the lowest-retention-score alive
/// memories, exactly the mechanism from the agent-memory-compaction nightly.
pub fn apply_compaction(graph: &mut MemoryGraph, now: u64, cfg: &SimConfig) {
    let weights = ScoringWeights::default();
    let victims = lowest_scoring(graph, now, &weights, cfg.compaction_evict_count);
    for v in victims {
        graph.evict(v);
    }
}

/// Weakens (and, below a floor, removes) bridge edges between two topic
/// regions among currently alive nodes — simulates topic staleness /
/// context drift without ever touching the embeddings themselves, so a
/// repair can always legitimately recover the true similarity.
pub fn apply_drift_wave(graph: &mut MemoryGraph, event: DriftEvent, cfg: &SimConfig) -> usize {
    let alive = graph.alive_ids();
    let mut touched = 0usize;
    let mut to_remove = Vec::new();
    let mut to_decay = Vec::new();
    'outer: for &u in &alive {
        if graph.nodes[u].embedding.topic != event.topic_a {
            continue;
        }
        for (&v, &w) in &graph.adjacency[u] {
            if graph.nodes[v].embedding.topic == event.topic_b {
                let new_w = w * cfg.drift_decay_factor;
                if new_w < 0.05 {
                    to_remove.push((u, v));
                } else {
                    to_decay.push((u, v, new_w));
                }
                touched += 1;
                if touched >= cfg.drift_max_edges_per_wave {
                    break 'outer;
                }
            }
        }
    }
    for (u, v) in to_remove {
        graph.remove_edge(u, v);
    }
    for (u, v, w) in to_decay {
        graph.add_edge(u, v, w);
    }
    touched
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrivals_are_reproducible_for_same_seed() {
        let cfg = SimConfig {
            total_steps: 50,
            ..Default::default()
        };
        let (_, a1) = build_arrivals(&cfg);
        let (_, a2) = build_arrivals(&cfg);
        for (x, y) in a1.iter().zip(a2.iter()) {
            assert_eq!(x.topic, y.topic);
            assert_eq!(x.vec, y.vec);
        }
    }

    #[test]
    fn drift_wave_reduces_or_removes_bridge_edges() {
        let cfg = SimConfig {
            drift_decay_factor: 0.1,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(1);
        let space = TopicSpace::new(cfg.dim, cfg.n_topics, &mut rng);
        let mut g = MemoryGraph::new();
        let a = g.add_node(space.sample(0, 0.05, &mut rng), 0);
        let b = g.add_node(space.sample(1, 0.05, &mut rng), 0);
        g.add_edge(a, b, 0.9);
        let touched = apply_drift_wave(
            &mut g,
            DriftEvent {
                step: 0,
                topic_a: 0,
                topic_b: 1,
            },
            &cfg,
        );
        assert_eq!(touched, 1);
        assert!(g.adjacency[a].get(&b).copied().unwrap_or(0.0) < 0.9);
    }
}
