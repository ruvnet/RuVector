//! Coherence-weighted compaction scoring: recency + frequency + local
//! semantic coherence, in the spirit of the 2026-06-14 agent-memory
//! compaction nightly (docs/research/nightly/2026-06-14-agent-memory-compaction).
//! Reimplemented locally (not depended on as a crate) since that PoC is a
//! standalone binary crate outside the cargo workspace; the formula shape is
//! the same three-signal blend, used here as a *reasonable baseline*, not a
//! strawman, driving which memories get evicted and thus what structural
//! drift looks like.

use crate::graph::MemoryGraph;

#[derive(Debug, Clone, Copy)]
pub struct ScoringWeights {
    pub recency: f64,
    pub frequency: f64,
    pub coherence: f64,
    pub half_life_steps: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            recency: 0.4,
            frequency: 0.25,
            coherence: 0.35,
            half_life_steps: 300.0,
        }
    }
}

/// Retention score in [0, ~1]; higher survives compaction.
pub fn retention_score(graph: &MemoryGraph, id: usize, now: u64, w: &ScoringWeights) -> f64 {
    let node = &graph.nodes[id];
    let age = (now.saturating_sub(node.last_access_step)) as f64;
    let recency = 0.5_f64.powf(age / w.half_life_steps.max(1.0));
    let frequency = (1.0 + node.access_count as f64).ln() / (1.0 + 50.0_f64).ln();
    let frequency = frequency.min(1.0);
    let coherence = local_coherence(graph, id);
    (w.recency * recency + w.frequency * frequency + w.coherence * coherence).clamp(0.0, 1.0)
}

/// Mean edge weight to currently alive neighbors; 0 for isolated nodes.
/// This is the signal that structural drift (weakened/removed bridge edges)
/// directly degrades, which is what couples compaction to graph health.
pub fn local_coherence(graph: &MemoryGraph, id: usize) -> f64 {
    let neighbors = &graph.adjacency[id];
    if neighbors.is_empty() {
        return 0.0;
    }
    let sum: f64 = neighbors.values().sum();
    (sum / neighbors.len() as f64).clamp(0.0, 1.0)
}

/// Selects the `count` lowest-scoring alive nodes for eviction.
pub fn lowest_scoring(
    graph: &MemoryGraph,
    now: u64,
    w: &ScoringWeights,
    count: usize,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = graph
        .alive_ids()
        .into_iter()
        .map(|id| (id, retention_score(graph, id, now, w)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(count);
    scored.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::TopicSpace;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn well_connected_recent_node_outscores_isolated_stale_node() {
        let mut rng = StdRng::seed_from_u64(3);
        let space = TopicSpace::new(8, 3, &mut rng);
        let mut g = MemoryGraph::new();
        let fresh = g.add_node(space.sample(0, 0.05, &mut rng), 100);
        let stale = g.add_node(space.sample(1, 0.05, &mut rng), 0);
        let hub = g.add_node(space.sample(0, 0.05, &mut rng), 100);
        g.add_edge(fresh, hub, 0.9);
        let w = ScoringWeights::default();
        let s_fresh = retention_score(&g, fresh, 100, &w);
        let s_stale = retention_score(&g, stale, 100, &w);
        assert!(s_fresh > s_stale, "{s_fresh} should exceed {s_stale}");
    }

    #[test]
    fn lowest_scoring_returns_requested_count() {
        let mut rng = StdRng::seed_from_u64(3);
        let space = TopicSpace::new(8, 3, &mut rng);
        let mut g = MemoryGraph::new();
        for i in 0..10 {
            g.add_node(space.sample(i % 3, 0.05, &mut rng), i as u64);
        }
        let picked = lowest_scoring(&g, 10, &ScoringWeights::default(), 4);
        assert_eq!(picked.len(), 4);
    }
}
