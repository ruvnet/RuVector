//! Baseline retriever: pure top-K by L2 distance.
//!
//! Maximises relevance with no diversity guarantee.  This serves as the
//! baseline against which MMR and PartitionMMR are compared.

use crate::{l2, DiverseRetriever, RetrievalResult};

/// Brute-force top-K nearest-neighbour retriever.
///
/// Scans all stored vectors and returns the `k` closest to the query.
/// O(N·D) per query where N = dataset size and D = dimension.
pub struct TopKRetriever<'a> {
    vectors: &'a [Vec<f32>],
}

impl<'a> TopKRetriever<'a> {
    /// Create a retriever over `vectors`.
    pub fn new(vectors: &'a [Vec<f32>]) -> Self {
        Self { vectors }
    }
}

impl<'a> DiverseRetriever for TopKRetriever<'a> {
    fn search(&self, query: &[f32], k: usize) -> Vec<RetrievalResult> {
        let k = k.min(self.vectors.len());
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2(query, v)))
            .collect();

        // Partial sort: bring k smallest distances to the front.
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(id, distance)| RetrievalResult {
                id,
                distance,
                diversity_score: 0.0,
            })
            .collect()
    }

    fn name(&self) -> &str {
        "TopK (baseline)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vecs() -> Vec<Vec<f32>> {
        vec![
            vec![1.0f32, 0.0],
            vec![2.0f32, 0.0],
            vec![0.5f32, 0.0],
            vec![3.0f32, 0.0],
        ]
    }

    #[test]
    fn returns_k_results() {
        let v = tiny_vecs();
        let r = TopKRetriever::new(&v);
        let res = r.search(&[0.0, 0.0], 2);
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn closest_first() {
        let v = tiny_vecs();
        let r = TopKRetriever::new(&v);
        let res = r.search(&[0.0f32, 0.0], 4);
        // Distances: 1.0, 2.0, 0.5, 3.0 → order: 2, 0, 1, 3
        assert_eq!(res[0].id, 2, "closest should be index 2 ([0.5,0.0])");
        assert_eq!(res[1].id, 0, "second should be index 0 ([1.0,0.0])");
    }

    #[test]
    fn k_clamped_to_dataset_size() {
        let v = tiny_vecs();
        let r = TopKRetriever::new(&v);
        let res = r.search(&[0.0, 0.0], 100);
        assert_eq!(res.len(), 4);
    }

    #[test]
    fn distances_are_non_decreasing() {
        let v = tiny_vecs();
        let r = TopKRetriever::new(&v);
        let res = r.search(&[0.0, 0.0], 4);
        for w in res.windows(2) {
            assert!(
                w[0].distance <= w[1].distance,
                "results must be sorted by distance"
            );
        }
    }
}
