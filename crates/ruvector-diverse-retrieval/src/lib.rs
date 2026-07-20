//! # ruvector-diverse-retrieval
//!
//! Partition-aware diverse ANN retrieval using graph-cut-inspired partitioning
//! to guarantee semantic spread across retrieved results.
//!
//! ## Background
//!
//! Standard top-K ANN retrieval maximises relevance but makes no diversity
//! guarantee. In agent memory and RAG pipelines, receiving ten near-identical
//! memories from the same semantic cluster is wasteful—and can cause LLM
//! context collapse. Maximal Marginal Relevance (MMR, Carbonell & Goldstein
//! 1998) addresses this with a greedy λ-weighted score, but it treats all
//! inter-candidate distances uniformly. This crate adds a **partition layer**:
//! candidates are clustered by connectivity threshold (akin to a soft mincut),
//! and the MMR penalty is amplified for same-partition selections, forcing the
//! result set to span multiple semantic regions.
//!
//! ## Variants
//!
//! | Variant | Description |
//! |---------|-------------|
//! | [`TopKRetriever`] | Baseline: pure nearest-neighbour |
//! | [`MmrRetriever`] | Maximal Marginal Relevance (standard) |
//! | [`PartitionMmrRetriever`] | MMR + graph-partition same-cluster penalty |
//!
//! ## Quick start
//!
//! ```rust
//! use ruvector_diverse_retrieval::topk::TopKRetriever;
//! use ruvector_diverse_retrieval::DiverseRetriever;
//!
//! let vectors: Vec<Vec<f32>> = vec![
//!     vec![1.0, 0.0], vec![1.1, 0.0], vec![5.0, 0.0], vec![5.1, 0.0],
//! ];
//! let r = TopKRetriever::new(&vectors);
//! let results = r.search(&[0.0, 0.0], 2);
//! assert_eq!(results.len(), 2);
//! ```

pub mod dataset;
pub mod graph;
pub mod mmr;
pub mod partition_mmr;
pub mod topk;

/// A single retrieval result.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Index into the dataset.
    pub id: usize,
    /// L2 distance from query vector to this result.
    pub distance: f32,
    /// MMR diversity contribution score (0.0 for TopK baseline).
    pub diversity_score: f32,
}

/// Core trait implemented by all three retrieval variants.
pub trait DiverseRetriever {
    /// Search for `k` results, balancing relevance and diversity.
    fn search(&self, query: &[f32], k: usize) -> Vec<RetrievalResult>;
    /// Human-readable name for this variant.
    fn name(&self) -> &str;
}

/// L2 squared distance between two equal-length slices.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// L2 (Euclidean) distance between two equal-length slices.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2sq(a, b).sqrt()
}

/// Mean pairwise L2 distance among results — higher means more diverse.
pub fn mean_diversity(results: &[RetrievalResult], vectors: &[Vec<f32>]) -> f32 {
    let n = results.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            total += l2(&vectors[results[i].id], &vectors[results[j].id]);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// Mean L2 distance from query to results — lower means more relevant.
pub fn mean_relevance(results: &[RetrievalResult]) -> f32 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(|r| r.distance).sum::<f32>() / results.len() as f32
}

/// Count distinct partition labels among a result set.
pub fn distinct_partitions(
    results: &[RetrievalResult],
    partition_fn: impl Fn(usize) -> usize,
) -> usize {
    let mut seen = std::collections::HashSet::new();
    for r in results {
        seen.insert(partition_fn(r.id));
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2sq_identity() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert_eq!(l2sq(&v, &v), 0.0);
    }

    #[test]
    fn l2_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let dist = l2(&a, &b);
        assert!(
            (dist - std::f32::consts::SQRT_2).abs() < 1e-5,
            "dist={dist}"
        );
    }

    #[test]
    fn mean_diversity_two_results() {
        let vecs = vec![vec![0.0f32, 0.0], vec![3.0f32, 4.0]];
        let results = vec![
            RetrievalResult {
                id: 0,
                distance: 0.0,
                diversity_score: 0.0,
            },
            RetrievalResult {
                id: 1,
                distance: 5.0,
                diversity_score: 0.0,
            },
        ];
        let div = mean_diversity(&results, &vecs);
        assert!((div - 5.0).abs() < 1e-4, "div={div}");
    }

    #[test]
    fn mean_relevance_basic() {
        let results = vec![
            RetrievalResult {
                id: 0,
                distance: 1.0,
                diversity_score: 0.0,
            },
            RetrievalResult {
                id: 1,
                distance: 3.0,
                diversity_score: 0.0,
            },
        ];
        assert!((mean_relevance(&results) - 2.0).abs() < 1e-4);
    }
}
