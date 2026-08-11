//! Brute-force k-NN graph for the graph-neighbour cascade.
//!
//! At index time, every vector's K_GRAPH nearest neighbours are computed via
//! exact f32 distances (O(N² D)).  For N=10 000, D=128 this takes ~1 s in
//! release and produces a 160 KB adjacency list.
//!
//! During search the graph is used only to expand the "uncertain zone" —
//! candidates near the cascade boundary whose true nearest neighbours may
//! have been displaced by quantization error.

/// Compact flat adjacency list.  Neighbours are stored as u32 ids.
pub struct KnnGraph {
    pub n: usize,
    pub k: usize,      // neighbours per node
    pub adj: Vec<u32>, // row-major, n * k entries
}

impl KnnGraph {
    /// Build by brute-force.  O(N² D) — only used at index construction time.
    pub fn build(vecs: &[Vec<f32>], k: usize) -> Self {
        let n = vecs.len();
        let mut adj = vec![0u32; n * k];

        for i in 0..n {
            // Compute distances to all other vectors.
            let mut dists: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j as u32, crate::dist_sq(&vecs[i], &vecs[j])))
                .collect();

            // Partial sort: we only need the k smallest.
            let pivot = k.min(dists.len()) - 1;
            dists.select_nth_unstable_by(pivot, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            for (slot, &(id, _)) in dists.iter().take(k).enumerate() {
                adj[i * k + slot] = id;
            }
        }

        KnnGraph { n, k, adj }
    }

    /// Neighbours of node `id`.
    #[inline]
    pub fn neighbours(&self, id: usize) -> &[u32] {
        &self.adj[id * self.k..(id + 1) * self.k]
    }

    /// Heap bytes: n * k * 4.
    pub fn memory_bytes(&self) -> usize {
        self.adj.len() * 4
    }
}
