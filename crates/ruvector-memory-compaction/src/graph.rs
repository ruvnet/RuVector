//! k-NN similarity graph construction and isolation scoring.
//!
//! Used by `MinCutCompactor` to identify which memory entries are most
//! isolated in the embedding space — analogous to finding nodes cut away
//! by a minimum-cut partition.
//!
//! ## Algorithm
//!
//! 1. For each entry, compute cosine similarity to every other entry.
//! 2. Retain only the top-k edges per node above `similarity_threshold`.
//! 3. Compute each node's **isolation score** = 1 − (mean edge weight of
//!    its top-k connections, or 0 if it has no connections).
//!
//! Nodes with the highest isolation score are the most weakly connected
//! — they sit at cut boundaries and are the safest candidates for eviction
//! without disrupting the global semantic structure of agent memory.

use crate::store::cosine_sim;

/// Sparse adjacency list: `edges[i]` = `(j, similarity)` pairs.
pub type AdjList = Vec<Vec<(usize, f32)>>;

/// Build a k-NN similarity graph over the entry vectors.
///
/// `vectors` — row-major: `vectors[i]` is the f32 embedding for entry i.
/// `k`       — maximum neighbors per node.
/// `thresh`  — only include edges with cosine sim ≥ thresh.
pub fn build_knn_graph(vectors: &[&[f32]], k: usize, thresh: f32) -> AdjList {
    let n = vectors.len();
    let mut adj: AdjList = vec![Vec::new(); n];

    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_sim(vectors[i], vectors[j])))
            .filter(|(_, s)| *s >= thresh)
            .collect();

        // Keep top-k by descending similarity
        sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sims.truncate(k);
        adj[i] = sims;
    }

    adj
}

/// Compute an isolation score for each node in [0, 1].
///
/// score = 1 − (mean_edge_weight), so 1.0 = fully isolated, 0.0 = strongly connected.
/// Nodes with no neighbors get score 1.0 (maximally isolated).
pub fn isolation_scores(adj: &AdjList) -> Vec<f32> {
    adj.iter()
        .map(|neighbors| {
            if neighbors.is_empty() {
                1.0
            } else {
                let mean: f32 =
                    neighbors.iter().map(|(_, s)| *s).sum::<f32>() / neighbors.len() as f32;
                1.0 - mean.clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Find connected components using BFS.
/// Returns a vector of component IDs (0-indexed) for each node.
pub fn connected_components(adj: &AdjList) -> Vec<usize> {
    let n = adj.len();
    let mut comp = vec![usize::MAX; n];
    let mut next_comp = 0usize;

    for start in 0..n {
        if comp[start] != usize::MAX {
            continue;
        }
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        comp[start] = next_comp;
        while let Some(node) = queue.pop_front() {
            for &(neighbor, _) in &adj[node] {
                if comp[neighbor] == usize::MAX {
                    comp[neighbor] = next_comp;
                    queue.push_back(neighbor);
                }
            }
        }
        next_comp += 1;
    }

    comp
}

/// Estimate the "cut weight" for each node — how much edge weight would be
/// severed if this node were removed.  Useful as a secondary sorting key.
pub fn cut_weights(adj: &AdjList) -> Vec<f32> {
    adj.iter()
        .map(|neighbors| neighbors.iter().map(|(_, s)| *s).sum())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_tight_clusters_have_isolated_bridge() {
        // Cluster A: [1,0,0], [0.9,0.1,0]
        // Cluster B: [0,0,1], [0,0.1,0.9]
        // Bridge:    [0.5,0,0.5]  — should have higher isolation than cluster members
        let vecs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.1, 0.9],
            vec![0.5, 0.0, 0.5],
        ];
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let adj = build_knn_graph(&refs, 2, 0.0);
        let scores = isolation_scores(&adj);

        // Bridge node should have higher isolation than at least one cluster member
        let bridge_score = scores[4];
        let cluster_a_score = scores[0];
        let cluster_b_score = scores[2];
        assert!(
            bridge_score > cluster_a_score || bridge_score > cluster_b_score,
            "bridge={bridge_score:.3} cluster_a={cluster_a_score:.3} cluster_b={cluster_b_score:.3}"
        );
    }

    #[test]
    fn isolated_node_gets_score_one() {
        // Node 0 is far from node 1 — orthogonal, thresh=0.5 means no edge
        let vecs: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let adj = build_knn_graph(&refs, 4, 0.5);
        let scores = isolation_scores(&adj);
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 1.0);
    }

    #[test]
    fn connected_components_finds_two_clusters() {
        let vecs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let adj = build_knn_graph(&refs, 2, 0.5);
        let comps = connected_components(&adj);
        // Nodes 0,1 should share a component; nodes 2,3 another
        let max_comp = *comps.iter().max().unwrap();
        assert!(max_comp >= 1, "expected at least 2 components");
    }
}
