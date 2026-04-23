/// Compressed Sparse Row (CSR) representation of an undirected graph.
///
/// Edges are stored deduplicated and sorted per node. Self-loops are excluded.
pub struct CsrGraph {
    /// offsets[i]..offsets[i+1] gives the range of neighbors for node i.
    offsets: Vec<usize>,
    /// Flat list of neighbor node IDs.
    neighbors: Vec<usize>,
    num_nodes: usize,
}

impl CsrGraph {
    /// Build an undirected CSR graph from the given edge list.
    ///
    /// Edges are treated as undirected (each (u,v) also inserts (v,u)).
    /// Duplicate edges and self-loops are removed.
    pub fn new(num_nodes: usize, edges: &[(usize, usize)]) -> Self {
        // Build adjacency list, excluding self-loops
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for &(u, v) in edges {
            if u != v && u < num_nodes && v < num_nodes {
                adj[u].push(v);
                adj[v].push(u);
            }
        }

        // Sort and deduplicate each adjacency list
        for nbrs in adj.iter_mut() {
            nbrs.sort_unstable();
            nbrs.dedup();
        }

        // Build CSR arrays
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut neighbors_flat = Vec::new();
        offsets.push(0);
        for nbrs in &adj {
            neighbors_flat.extend_from_slice(nbrs);
            offsets.push(neighbors_flat.len());
        }

        CsrGraph {
            offsets,
            neighbors: neighbors_flat,
            num_nodes,
        }
    }

    /// Returns the slice of neighbor IDs for the given node.
    pub fn neighbors(&self, node: usize) -> &[usize] {
        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        &self.neighbors[start..end]
    }

    /// Returns the degree (number of neighbors) of the given node.
    pub fn degree(&self, node: usize) -> usize {
        self.offsets[node + 1] - self.offsets[node]
    }

    /// Returns the total number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }
}
