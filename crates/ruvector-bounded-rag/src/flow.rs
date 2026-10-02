//! Max-flow / min-cut source-side partition, extracted from the original
//! [`crate::MinCutRetriever`] so the same Edmonds–Karp solver can be reused
//! against a *sparse* inter-chunk edge set (see [`crate::sparse_knn`])
//! without duplicating the flow algorithm.
//!
//! This module is a pure graph routine: it knows nothing about chunks,
//! corpora, or cosine similarity. Callers build the flow network (source
//! capacities, sink capacities, undirected inter-node edges) and get back
//! the boolean source-side membership for nodes `0..n`.

use std::collections::{HashMap, VecDeque};

/// Computes the max-flow / min-cut over a flow network shaped as:
///
/// - a virtual source with an edge to every node `i` of capacity `source_cap[i]`
/// - a virtual sink with an edge from every node `i` of capacity `sink_cap[i]`
/// - undirected inter-node edges from `edges` (`(i, j, weight)`), added in
///   both directions
///
/// Runs Edmonds-Karp (BFS-augmented Ford-Fulkerson) to find the max flow,
/// then BFS on the residual graph from the source to find the source-side
/// partition. Returns a `Vec<bool>` of length `n`: `true` means node `i` is
/// on the source side of the min cut.
///
/// `source_cap` and `sink_cap` must both have length `n`. `edges` indices
/// must be `< n`.
pub fn source_side_partition(
    n: usize,
    source_cap: &[f32],
    sink_cap: &[f32],
    edges: &[(usize, usize, f32)],
) -> Vec<bool> {
    assert_eq!(source_cap.len(), n, "source_cap length must equal n");
    assert_eq!(sink_cap.len(), n, "sink_cap length must equal n");

    // Node indices: 0..n = chunks, n = source, n+1 = sink
    let source = n;
    let sink = n + 1;
    let total = n + 2;

    let mut cap: Vec<HashMap<usize, f32>> = vec![HashMap::new(); total];

    for i in 0..n {
        cap[source].insert(i, source_cap[i]);
        cap[i].entry(source).or_insert(0.0);
    }
    for i in 0..n {
        cap[i].insert(sink, sink_cap[i]);
        cap[sink].entry(i).or_insert(0.0);
    }
    for &(i, j, w) in edges {
        *cap[i].entry(j).or_insert(0.0) += w;
        *cap[j].entry(i).or_insert(0.0) += w;
    }

    let mut flow: Vec<HashMap<usize, f32>> = vec![HashMap::new(); total];

    loop {
        let mut prev = vec![usize::MAX; total];
        prev[source] = source;
        let mut queue = VecDeque::new();
        queue.push_back(source);

        'bfs: while let Some(u) = queue.pop_front() {
            for (&v, &c) in &cap[u] {
                if prev[v] == usize::MAX {
                    let f = *flow[u].get(&v).unwrap_or(&0.0);
                    if c - f > 1e-6 {
                        prev[v] = u;
                        if v == sink {
                            break 'bfs;
                        }
                        queue.push_back(v);
                    }
                }
            }
        }

        if prev[sink] == usize::MAX {
            break; // No augmenting path — max flow reached.
        }

        let mut bottleneck = f32::INFINITY;
        let mut v = sink;
        while v != source {
            let u = prev[v];
            let c = *cap[u].get(&v).unwrap_or(&0.0);
            let f = *flow[u].get(&v).unwrap_or(&0.0);
            bottleneck = bottleneck.min(c - f);
            v = u;
        }

        let mut v = sink;
        while v != source {
            let u = prev[v];
            *flow[u].entry(v).or_insert(0.0) += bottleneck;
            *flow[v].entry(u).or_insert(0.0) -= bottleneck;
            v = u;
        }
    }

    let mut in_source_set = vec![false; total];
    in_source_set[source] = true;
    let mut queue = VecDeque::new();
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        for (&v, &c) in &cap[u] {
            if !in_source_set[v] {
                let f = *flow[u].get(&v).unwrap_or(&0.0);
                if c - f > 1e-6 {
                    in_source_set[v] = true;
                    queue.push_back(v);
                }
            }
        }
    }

    in_source_set[..n].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_tight_clusters_separate_cleanly() {
        // 4 nodes: {0,1} strongly query-coherent and linked, {2,3} not.
        let n = 4;
        let source_cap = vec![0.9, 0.9, 0.1, 0.1];
        let sink_cap = vec![0.1, 0.1, 0.9, 0.9];
        let edges = vec![(0, 1, 0.9), (2, 3, 0.9)];
        let partition = source_side_partition(n, &source_cap, &sink_cap, &edges);
        assert!(partition[0] && partition[1]);
        assert!(!partition[2] && !partition[3]);
    }

    #[test]
    fn no_inter_edges_falls_back_to_source_capacity() {
        let n = 3;
        let source_cap = vec![0.8, 0.2, 0.9];
        let sink_cap = vec![0.2, 0.8, 0.1];
        let partition = source_side_partition(n, &source_cap, &sink_cap, &[]);
        assert!(partition[0]);
        assert!(!partition[1]);
        assert!(partition[2]);
    }
}
