//! Mutual k-NN boundary / coherence scoring.
//!
//! For each vector, computes the fraction of its k nearest neighbours for
//! which the relationship is *mutual*: `v` is among `u`'s k nearest
//! neighbours as well as `u` being among `v`'s. Vectors deep inside a dense
//! cluster tend to share most of their nearest neighbours mutually, because
//! everyone nearby agrees on who is close. Vectors that sit on a cluster
//! boundary or bridge between clusters have more asymmetric neighbour
//! relationships: their neighbours' own nearest points often lie inside a
//! different, denser region, so the relationship is one-directional.
//!
//! This is a lightweight, ground-truth-free structural proxy for local graph
//! conductance / cut-boundary detection -- the same family of signal the
//! `ruvector-mincut` crate computes with subpolynomial dynamic algorithms
//! over general graphs. This PoC intentionally does not depend on
//! `ruvector-mincut` (kept evaluator-independent and dependency-light for
//! benchmarking); production adoption should replace
//! [`mutual_knn_coherence`] with a `ruvector-mincut` conductance query over
//! the same k-NN graph. See the research README for that integration path.

use crate::dataset::l2sq;

pub struct KnnGraph {
    pub k: usize,
    /// `neighbours[i]` = the k nearest neighbour indices of vector `i`,
    /// sorted by ascending distance.
    pub neighbours: Vec<Vec<usize>>,
}

/// Brute-force k-NN graph construction. O(n^2 * dim); fine for PoC-scale
/// corpora (a production index would reuse the HNSW/graph build already
/// paid for).
pub fn build_knn_graph(corpus: &[Vec<f32>], k: usize) -> KnnGraph {
    let n = corpus.len();
    let neighbours: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, l2sq(&corpus[i], &corpus[j])))
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.truncate(k);
            dists.into_iter().map(|(j, _)| j).collect()
        })
        .collect();
    KnnGraph { k, neighbours }
}

/// coherence(v) in `[0, 1]`: fraction of v's k nearest neighbours that are
/// *mutual* (v is also among that neighbour's k nearest neighbours).
///
/// 1.0 = fully mutual neighbourhood (deep cluster core).
/// 0.0 = no mutual neighbours (isolated / bridging point).
pub fn mutual_knn_coherence(graph: &KnnGraph) -> Vec<f32> {
    let n = graph.neighbours.len();
    let sets: Vec<std::collections::HashSet<usize>> = graph
        .neighbours
        .iter()
        .map(|nb| nb.iter().copied().collect())
        .collect();
    (0..n)
        .map(|i| {
            let nb = &graph.neighbours[i];
            if nb.is_empty() {
                return 0.0;
            }
            let mutual = nb.iter().filter(|&&j| sets[j].contains(&i)).count();
            mutual as f32 / nb.len() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{clustered_vectors, random_unit_vectors};

    #[test]
    fn clustered_corpus_has_higher_mean_coherence_than_random() {
        // Tight clusters should produce systematically higher mutual-kNN
        // coherence than uniformly random points, because random points
        // have no consistent local density structure to agree on.
        let clustered = clustered_vectors(600, 12, 6, 0.05, 7);
        let random = random_unit_vectors(600, 12, 7);

        let g_clustered = build_knn_graph(&clustered, 10);
        let g_random = build_knn_graph(&random, 10);

        let c_clustered = mutual_knn_coherence(&g_clustered);
        let c_random = mutual_knn_coherence(&g_random);

        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let mean_clustered = mean(&c_clustered);
        let mean_random = mean(&c_random);

        assert!(
            mean_clustered > mean_random,
            "clustered mean coherence {mean_clustered:.3} should exceed random {mean_random:.3}"
        );
    }

    #[test]
    fn coherence_is_bounded_zero_one() {
        let corpus = clustered_vectors(200, 8, 4, 0.1, 3);
        let graph = build_knn_graph(&corpus, 8);
        let scores = mutual_knn_coherence(&graph);
        for &s in &scores {
            assert!((0.0..=1.0).contains(&s), "coherence out of range: {s}");
        }
    }

    #[test]
    fn fully_mutual_pair_has_coherence_one() {
        // Two isolated points, k=1: each other's only (hence mutual) neighbour.
        let corpus = vec![vec![0.0_f32, 0.0], vec![0.1_f32, 0.0]];
        let graph = build_knn_graph(&corpus, 1);
        let scores = mutual_knn_coherence(&graph);
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!((scores[1] - 1.0).abs() < 1e-6);
    }
}
