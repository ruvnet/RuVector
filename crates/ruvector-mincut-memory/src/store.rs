//! Agent working memory store: vector entries + similarity graph.
//!
//! The graph is rebuilt lazily on demand.  Edge weights are cosine similarities
//! above a configurable threshold; they are stored in a dense adjacency matrix
//! over the live entry set so that graph-cut heuristics can run in O(N²) with
//! simple index arithmetic.

use crate::l2_sq;
use std::collections::HashMap;

/// One memory entry: a vector, a logical timestamp, and an access count.
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: u64,
    pub vector: Vec<f32>,
    /// Logical insertion time — lower means older.
    pub timestamp: u64,
    /// Number of times this entry has been retrieved.
    pub access_count: u32,
}

impl Entry {
    pub fn new(id: u64, vector: Vec<f32>, timestamp: u64) -> Self {
        Self {
            id,
            vector,
            timestamp,
            access_count: 0,
        }
    }
}

/// In-memory vector store with configurable similarity graph support.
pub struct MemoryStore {
    pub entries: Vec<Entry>,
    pub dims: usize,
    /// Edge weight threshold for the similarity graph.
    pub similarity_threshold: f32,
    /// Cached adjacency weights.  `graph[i][j]` = cosine similarity if ≥
    /// threshold, else 0.0.  Rebuilt by `rebuild_graph()`.
    pub graph: Vec<Vec<f32>>,
    graph_dirty: bool,
    next_id: u64,
}

impl MemoryStore {
    pub fn new(dims: usize, similarity_threshold: f32) -> Self {
        Self {
            entries: Vec::new(),
            dims,
            similarity_threshold,
            graph: Vec::new(),
            graph_dirty: true,
            next_id: 0,
        }
    }

    /// Insert a vector and return its id.
    pub fn insert(&mut self, vector: Vec<f32>, timestamp: u64) -> u64 {
        assert_eq!(vector.len(), self.dims);
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Entry::new(id, vector, timestamp));
        self.graph_dirty = true;
        id
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Brute-force ANN search: returns the `k` nearest entry indices (not ids)
    /// together with their squared L2 distances.
    pub fn search_k(&mut self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, l2_sq(query, &e.vector)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        scored
    }

    /// Rebuild the cosine-similarity graph.  O(N²·D).
    pub fn rebuild_graph(&mut self) {
        use crate::cosine_similarity;
        let n = self.entries.len();
        self.graph = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&self.entries[i].vector, &self.entries[j].vector);
                if sim >= self.similarity_threshold {
                    self.graph[i][j] = sim;
                    self.graph[j][i] = sim;
                }
            }
        }
        self.graph_dirty = false;
    }

    /// Ensure the graph is up to date.
    pub fn ensure_graph(&mut self) {
        if self.graph_dirty {
            self.rebuild_graph();
        }
    }

    /// Remove entries by their position indices (largest first to preserve
    /// positions).
    pub fn remove_indices(&mut self, mut indices: Vec<usize>) {
        indices.sort_unstable_by(|a, b| b.cmp(a));
        indices.dedup();
        for idx in indices {
            self.entries.swap_remove(idx);
        }
        self.graph_dirty = true;
    }

    /// Recall@k: fraction of `ground_truth_ids` found in the top-k search
    /// results for the given query.
    pub fn recall_at_k(&mut self, query: &[f32], ground_truth_ids: &[u64], k: usize) -> f32 {
        let results = self.search_k(query, k);
        let found_ids: std::collections::HashSet<u64> =
            results.iter().map(|(i, _)| self.entries[*i].id).collect();
        let hits = ground_truth_ids
            .iter()
            .filter(|id| found_ids.contains(id))
            .count();
        hits as f32 / ground_truth_ids.len().min(k) as f32
    }

    /// Build an id→index map over the current entries.
    pub fn id_to_index(&self) -> HashMap<u64, usize> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id, i))
            .collect()
    }

    /// Estimate memory usage in bytes (vectors only).
    pub fn memory_bytes(&self) -> usize {
        self.entries.len() * self.dims * std::mem::size_of::<f32>()
            + self.entries.len() * std::mem::size_of::<Entry>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> MemoryStore {
        let mut s = MemoryStore::new(4, 0.3);
        s.insert(vec![1.0, 0.0, 0.0, 0.0], 0);
        s.insert(vec![0.9, 0.1, 0.0, 0.0], 1);
        s.insert(vec![0.0, 1.0, 0.0, 0.0], 2);
        s.insert(vec![0.0, 0.0, 1.0, 0.0], 3);
        s
    }

    #[test]
    fn insert_and_len() {
        let s = make_store();
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn search_returns_nearest() {
        let mut s = make_store();
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let results = s.search_k(&q, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 0.0); // exact match
    }

    #[test]
    fn rebuild_graph_edges_above_threshold() {
        let mut s = make_store();
        s.rebuild_graph();
        // entries 0 and 1 are similar (cosine ~ 0.995)
        assert!(s.graph[0][1] > 0.3, "expected edge between similar entries");
        // entries 0 and 2 are orthogonal (cosine = 0)
        assert_eq!(s.graph[0][2], 0.0);
    }

    #[test]
    fn remove_indices_shrinks_store() {
        let mut s = make_store();
        s.remove_indices(vec![0, 2]);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn memory_bytes_nonzero() {
        let s = make_store();
        assert!(s.memory_bytes() > 0);
    }
}
