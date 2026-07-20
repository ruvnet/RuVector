//! Recall computation and distance utilities.

use crate::graph::sq_euclidean;

/// Compute recall@k: fraction of true nearest neighbors that appear in results.
/// `ground_truth[q]` = sorted true neighbor IDs for query q (length >= k).
/// `results[q]` = returned neighbor IDs for query q (length >= k).
pub fn recall_at_k(ground_truth: &[Vec<usize>], results: &[Vec<usize>], k: usize) -> f64 {
    assert_eq!(ground_truth.len(), results.len(), "query count mismatch");
    let mut hits = 0u64;
    let mut total = 0u64;
    for (gt, res) in ground_truth.iter().zip(results.iter()) {
        let gt_set: std::collections::HashSet<usize> = gt.iter().take(k).copied().collect();
        for &r in res.iter().take(k) {
            if gt_set.contains(&r) {
                hits += 1;
            }
        }
        total += k as u64;
    }
    if total == 0 {
        1.0
    } else {
        hits as f64 / total as f64
    }
}

/// Brute-force exact k-NN for a single query over a dataset.
pub fn exact_knn(query: &[f32], dataset: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut dists: Vec<(f32, usize)> = dataset
        .iter()
        .enumerate()
        .map(|(i, v)| (sq_euclidean(query, v), i))
        .collect();
    dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Compute ground-truth k-NN for a batch of queries.
pub fn ground_truth_knn(queries: &[Vec<f32>], dataset: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    queries.iter().map(|q| exact_knn(q, dataset, k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_recall() {
        let gt = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let res = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let r = recall_at_k(&gt, &res, 3);
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }

    #[test]
    fn zero_recall() {
        let gt = vec![vec![0, 1, 2]];
        let res = vec![vec![3, 4, 5]];
        let r = recall_at_k(&gt, &res, 3);
        assert!((r - 0.0).abs() < 1e-9, "expected 0.0, got {}", r);
    }

    #[test]
    fn partial_recall() {
        let gt = vec![vec![0, 1, 2]];
        let res = vec![vec![0, 3, 5]];
        let r = recall_at_k(&gt, &res, 3);
        let expected = 1.0 / 3.0;
        assert!((r - expected).abs() < 1e-6, "expected ~0.333, got {}", r);
    }

    #[test]
    fn exact_knn_finds_nearest() {
        let dataset: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ];
        let query = vec![0.1, 0.1];
        let nn = exact_knn(&query, &dataset, 2);
        assert_eq!(nn[0], 0, "closest should be origin");
        assert!(
            nn.contains(&1) || nn.contains(&2),
            "second should be an axis point"
        );
    }
}
