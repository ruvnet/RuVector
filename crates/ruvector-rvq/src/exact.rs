//! Exact brute-force f32 inner-product search.
//!
//! Ground-truth baseline: recall = 1.0 by definition.
//! Memory = N × D × 4 bytes (raw vectors).

use crate::{dot, Hit, VectorIndex};

pub struct ExactSearch {
    vectors: Vec<Vec<f32>>,
}

impl ExactSearch {
    pub fn build(vectors: &[Vec<f32>]) -> Self {
        Self {
            vectors: vectors.to_vec(),
        }
    }
}

impl VectorIndex for ExactSearch {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut hits: Vec<Hit> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(id, v)| Hit {
                id,
                score: dot(query, v),
            })
            .collect();
        hits.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str {
        "Exact-f32"
    }

    fn memory_bytes(&self) -> usize {
        self.vectors.len() * self.vectors.first().map(|v| v.len()).unwrap_or(0) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Dataset, DatasetConfig};

    #[test]
    fn exact_returns_k_hits() {
        let cfg = DatasetConfig {
            n_vectors: 200,
            dims: 32,
            n_queries: 5,
            ..Default::default()
        };
        let ds = Dataset::generate(&cfg);
        let idx = ExactSearch::build(&ds.vectors);
        for q in &ds.queries {
            let hits = idx.search(q, 10);
            assert_eq!(hits.len(), 10);
        }
    }

    #[test]
    fn exact_top1_is_self_when_query_in_corpus() {
        let vecs = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.0, 1.0],
        ];
        let idx = ExactSearch::build(&vecs);
        let hits = idx.search(&vecs[1], 1);
        assert_eq!(hits[0].id, 1, "top-1 should be the query vector itself");
    }
}
