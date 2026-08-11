use crate::dataset::l2_sq;

/// Lightweight k-NN graph for in-degree computation.
/// We compute the graph using brute force — acceptable for PoC dataset sizes.
pub struct KnnGraph {
    pub adj: Vec<Vec<usize>>,
    pub in_degree: Vec<usize>,
    pub n: usize,
    pub k: usize,
}

impl KnnGraph {
    /// Build a k-NN graph from data. O(n^2) brute force.
    pub fn build(data: &[Vec<f32>], k: usize) -> Self {
        let n = data.len();
        let k = k.min(n.saturating_sub(1));
        let mut adj: Vec<Vec<usize>> = vec![Vec::with_capacity(k); n];
        let mut in_degree = vec![0usize; n];

        for i in 0..n {
            let mut dists: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (l2_sq(&data[i], &data[j]), j))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let neighbors: Vec<usize> = dists[..k].iter().map(|(_, j)| *j).collect();
            for &nb in &neighbors {
                in_degree[nb] += 1;
            }
            adj[i] = neighbors;
        }

        Self {
            adj,
            in_degree,
            n,
            k,
        }
    }

    /// Degree of node as a fraction of maximum observed in-degree.
    pub fn normalized_degree(&self) -> Vec<f32> {
        let max_deg = self.in_degree.iter().copied().max().unwrap_or(1) as f32;
        self.in_degree.iter().map(|&d| d as f32 / max_deg).collect()
    }

    /// In-degree percentile threshold: returns a bitmask of "high degree" nodes.
    /// `fraction` ∈ (0, 1]: top `fraction` by in-degree are marked true.
    pub fn high_degree_mask(&self, fraction: f32) -> Vec<bool> {
        let threshold = self.degree_threshold(fraction);
        self.in_degree.iter().map(|&d| d >= threshold).collect()
    }

    fn degree_threshold(&self, fraction: f32) -> usize {
        let mut sorted = self.in_degree.clone();
        sorted.sort_unstable();
        let cut_idx = ((1.0 - fraction) * sorted.len() as f32) as usize;
        let cut_idx = cut_idx.min(sorted.len().saturating_sub(1));
        sorted[cut_idx]
    }

    pub fn mean_in_degree(&self) -> f32 {
        self.in_degree.iter().sum::<usize>() as f32 / self.n as f32
    }

    pub fn max_in_degree(&self) -> usize {
        self.in_degree.iter().copied().max().unwrap_or(0)
    }

    pub fn hub_fraction(&self, threshold: usize) -> f32 {
        let count = self.in_degree.iter().filter(|&&d| d >= threshold).count();
        count as f32 / self.n as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::generate_clustered;

    #[test]
    fn test_graph_builds_correctly() {
        let data = generate_clustered(100, 16, 5, 42);
        let graph = KnnGraph::build(&data, 8);
        assert_eq!(graph.n, 100);
        assert_eq!(graph.k, 8);
        // Each node has exactly k neighbors
        for adj in &graph.adj {
            assert_eq!(adj.len(), 8);
        }
    }

    #[test]
    fn test_in_degree_sums_to_n_times_k() {
        let data = generate_clustered(50, 8, 3, 77);
        let graph = KnnGraph::build(&data, 4);
        let total: usize = graph.in_degree.iter().sum();
        // Each of 50 vectors contributes 4 edges
        assert_eq!(total, 50 * 4);
    }

    #[test]
    fn test_high_degree_mask_fraction() {
        let data = generate_clustered(100, 16, 5, 11);
        let graph = KnnGraph::build(&data, 8);
        let mask = graph.high_degree_mask(0.20);
        let high_count = mask.iter().filter(|&&v| v).count();
        // Should be roughly 20%, allow ±10 due to ties
        assert!(
            high_count >= 10 && high_count <= 35,
            "Expected ~20 high-degree nodes, got {}",
            high_count
        );
    }
}
