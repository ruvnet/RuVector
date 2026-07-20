//! Directed weighted adjacency-list graph for entity relationships.
//!
//! Edges represent semantic or relational connections: citations, hyperlinks,
//! co-occurrence, ontological subsumption, or workflow dependencies.

use std::collections::HashMap;

/// A single directed weighted edge.
#[derive(Debug, Clone)]
pub struct Edge {
    pub target: u32,
    /// Edge weight in [0, 1]; higher = stronger semantic link.
    pub weight: f32,
}

/// Lightweight adjacency-list graph.
pub struct KnowledgeGraph {
    // source_id -> list of outgoing edges
    adjacency: HashMap<u32, Vec<Edge>>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self { adjacency: HashMap::new() }
    }

    pub fn add_edge(&mut self, source: u32, target: u32, weight: f32) {
        self.adjacency
            .entry(source)
            .or_default()
            .push(Edge { target, weight });
    }

    /// Undirected convenience wrapper (adds both directions).
    pub fn add_undirected_edge(&mut self, a: u32, b: u32, weight: f32) {
        self.add_edge(a, b, weight);
        self.add_edge(b, a, weight);
    }

    pub fn neighbors(&self, id: u32) -> &[Edge] {
        self.adjacency.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    /// Expand a set of seed ids by one hop, returning newly discovered ids
    /// sorted by descending edge weight. Excludes seeds already in `visited`.
    pub fn expand_one_hop<'a>(
        &'a self,
        seeds: impl Iterator<Item = u32>,
        visited: &std::collections::HashSet<u32>,
    ) -> Vec<(u32, f32)> {
        let mut candidates: HashMap<u32, f32> = HashMap::new();
        for seed in seeds {
            for edge in self.neighbors(seed) {
                if !visited.contains(&edge.target) {
                    let entry = candidates.entry(edge.target).or_insert(0.0);
                    if edge.weight > *entry {
                        *entry = edge.weight;
                    }
                }
            }
        }
        let mut result: Vec<(u32, f32)> = candidates.into_iter().collect();
        result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}
