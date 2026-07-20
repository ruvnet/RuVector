//! Simple k-NN graph construction for topology analysis.
//!
//! Builds a symmetric k-nearest-neighbor graph and computes per-node degree,
//! which serves as a proxy for betweenness centrality: high-degree nodes are
//! hubs that lie on many traversal paths through the graph.

/// Squared Euclidean distance between two f32 vectors.
#[inline]
pub fn sq_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// k-NN graph: `neighbors[i]` is the list of the k nearest neighbors of i.
/// The graph is directed (not symmetric) — each node records its own k NN.
pub fn build_knn_directed(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    let n = vectors.len();
    let k = k.min(n.saturating_sub(1));
    let mut graph = vec![Vec::new(); n];
    for i in 0..n {
        let mut dists: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (sq_euclidean(&vectors[i], &vectors[j]), j))
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        graph[i] = dists.into_iter().take(k).map(|(_, j)| j).collect();
    }
    graph
}

/// Node degree counting in-edges (how many nodes have this node as a neighbor).
/// High in-degree ≈ hub — many traversal paths pass through this node.
pub fn in_degree(graph: &[Vec<usize>]) -> Vec<usize> {
    let n = graph.len();
    let mut deg = vec![0usize; n];
    for neighbors in graph {
        for &nb in neighbors {
            deg[nb] += 1;
        }
    }
    deg
}

/// Classify nodes: `true` = hub (in-degree > threshold), `false` = leaf.
pub fn classify_hubs(degrees: &[usize], threshold: usize) -> Vec<bool> {
    degrees.iter().map(|&d| d > threshold).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_graph_self_excluded() {
        let vecs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0]).collect();
        let g = build_knn_directed(&vecs, 3);
        for (i, neighbors) in g.iter().enumerate() {
            assert!(!neighbors.contains(&i), "node {} has itself as neighbor", i);
            assert!(neighbors.len() <= 3);
        }
    }

    #[test]
    fn in_degree_sums_to_n_times_k() {
        let vecs: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32]).collect();
        let k = 2;
        let g = build_knn_directed(&vecs, k);
        let deg = in_degree(&g);
        let total: usize = deg.iter().sum();
        // Each node contributes exactly k out-edges → total in-degree = n*k
        assert_eq!(total, 8 * k);
    }

    #[test]
    fn hub_classification_at_threshold() {
        let degrees = vec![0, 1, 3, 5, 2];
        let hubs = classify_hubs(&degrees, 2);
        // threshold=2 → in_degree > 2 → nodes with degree 3, 5
        assert_eq!(hubs, vec![false, false, true, true, false]);
    }
}
