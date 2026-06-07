use std::collections::HashMap;

/// Sparse adjacency list graph over vector IDs.
#[derive(Default, Clone)]
pub struct Graph {
    /// edges[id] = list of neighbour ids
    edges: HashMap<usize, Vec<usize>>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.entry(from).or_default().push(to);
        self.edges.entry(to).or_default().push(from);
    }

    pub fn neighbours(&self, id: usize) -> &[usize] {
        self.edges.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum::<usize>() / 2
    }

    pub fn node_count(&self) -> usize {
        self.edges.len()
    }
}
