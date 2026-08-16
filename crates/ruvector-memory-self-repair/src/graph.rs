//! Agent memory graph: nodes are memories, edges are semantic + temporal
//! associations. This is the structure whose spectral health we monitor.

use crate::embedding::{cosine, Embedding};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct MemoryNode {
    pub id: usize,
    pub embedding: Embedding,
    pub created_step: u64,
    pub last_access_step: u64,
    pub access_count: u32,
    pub alive: bool,
    /// Snapshot of the neighbor ids this memory was originally wired to at
    /// formation time (semantic top-k + temporal chain). Frozen at
    /// creation; used as ground truth for "did the graph retain/recover
    /// this memory's original associations" independent of whatever the
    /// live adjacency looks like later.
    pub original_neighbors: Vec<usize>,
}

/// Undirected weighted memory graph. Node ids are stable slot indices;
/// `alive[i] == false` marks an evicted memory whose slot is retained only
/// so edge lists don't need renumbering.
#[derive(Debug, Clone, Default)]
pub struct MemoryGraph {
    pub nodes: Vec<MemoryNode>,
    /// adjacency[u] = { v: weight }, symmetric (adjacency[v] also has u).
    pub adjacency: Vec<BTreeMap<usize, f64>>,
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, embedding: Embedding, step: u64) -> usize {
        let id = self.nodes.len();
        self.nodes.push(MemoryNode {
            id,
            embedding,
            created_step: step,
            last_access_step: step,
            access_count: 1,
            alive: true,
            original_neighbors: Vec::new(),
        });
        self.adjacency.push(BTreeMap::new());
        id
    }

    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        if u == v {
            return;
        }
        self.adjacency[u].insert(v, weight);
        self.adjacency[v].insert(u, weight);
    }

    pub fn remove_edge(&mut self, u: usize, v: usize) {
        self.adjacency[u].remove(&v);
        self.adjacency[v].remove(&u);
    }

    pub fn evict(&mut self, id: usize) {
        self.nodes[id].alive = false;
        let neighbors: Vec<usize> = self.adjacency[id].keys().copied().collect();
        for n in neighbors {
            self.adjacency[n].remove(&id);
        }
        self.adjacency[id].clear();
    }

    pub fn alive_ids(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.alive)
            .map(|n| n.id)
            .collect()
    }

    pub fn degree(&self, id: usize) -> usize {
        self.adjacency[id].len()
    }

    pub fn edge_count(&self) -> usize {
        self.adjacency.iter().map(|m| m.len()).sum::<usize>() / 2
    }

    /// Top-k semantic neighbors among currently alive nodes (excluding self),
    /// by cosine similarity of stored embeddings.
    pub fn semantic_topk(&self, id: usize, k: usize) -> Vec<(usize, f64)> {
        let e = &self.nodes[id].embedding;
        let mut sims: Vec<(usize, f64)> = self
            .nodes
            .iter()
            .filter(|n| n.alive && n.id != id)
            .map(|n| (n.id, cosine(&e.vec, &n.embedding.vec) as f64))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sims.truncate(k);
        sims
    }

    /// Exports the alive-node induced subgraph as a dense edge list plus a
    /// mapping from local (0..n) index back to original node ids, in the
    /// format `ruvector_coherence::spectral::CsrMatrixView::build_laplacian`
    /// expects.
    pub fn to_laplacian_edges(&self) -> (usize, Vec<(usize, usize, f64)>, Vec<usize>) {
        let alive: Vec<usize> = self.alive_ids();
        let index: BTreeMap<usize, usize> = alive
            .iter()
            .enumerate()
            .map(|(local, &orig)| (orig, local))
            .collect();
        let mut edges = Vec::new();
        let mut seen: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &u in &alive {
            for (&v, &w) in &self.adjacency[u] {
                if !self.nodes[v].alive {
                    continue;
                }
                let (lu, lv) = (index[&u], index[&v]);
                let key = if lu < lv { (lu, lv) } else { (lv, lu) };
                if seen.insert(key) {
                    edges.push((key.0, key.1, w));
                }
            }
        }
        (alive.len(), edges, alive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::TopicSpace;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn evict_removes_incident_edges() {
        let mut rng = StdRng::seed_from_u64(1);
        let space = TopicSpace::new(4, 2, &mut rng);
        let mut g = MemoryGraph::new();
        let a = g.add_node(space.sample(0, 0.1, &mut rng), 0);
        let b = g.add_node(space.sample(0, 0.1, &mut rng), 0);
        g.add_edge(a, b, 1.0);
        assert_eq!(g.edge_count(), 1);
        g.evict(a);
        assert_eq!(g.edge_count(), 0);
        assert!(g.adjacency[b].is_empty());
    }

    #[test]
    fn laplacian_export_excludes_dead_nodes() {
        let mut rng = StdRng::seed_from_u64(1);
        let space = TopicSpace::new(4, 2, &mut rng);
        let mut g = MemoryGraph::new();
        let a = g.add_node(space.sample(0, 0.1, &mut rng), 0);
        let b = g.add_node(space.sample(0, 0.1, &mut rng), 0);
        let c = g.add_node(space.sample(1, 0.1, &mut rng), 0);
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        g.evict(a);
        let (n, edges, alive) = g.to_laplacian_edges();
        assert_eq!(n, 2);
        assert_eq!(alive, vec![b, c]);
        assert_eq!(edges.len(), 1);
    }
}
