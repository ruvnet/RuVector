//! Hybrid sparse-dense fusion with coherence-adaptive per-query weighting.
//!
//! Combines a BM25 inverted index (sparse leg) with a flat cosine scan
//! (dense leg) into three fusion strategies, measured honestly against
//! a mixed text-dominant / vector-dominant corpus.
//!
//! # Variants
//! - [`SparseOnly`]: BM25 keyword retrieval only
//! - [`DenseOnly`]: flat cosine-similarity scan only
//! - [`HybridRRF`]: Reciprocal Rank Fusion (k=60, standard baseline)
//! - [`HybridCoherence`]: per-query alpha from score-concentration ratio

pub mod bm25;
pub mod dataset;
pub mod dense;
pub mod fusion;

use std::time::{Duration, Instant};

pub use bm25::Bm25Index;
pub use dataset::{Corpus, DocType, Query, QueryType};
pub use dense::DenseIndex;
pub use fusion::{coherence_fuse, rrf_fuse, linear_fuse};

/// A retrieval result: (doc_id, score).
pub type Hit = (usize, f32);

/// Common interface for all retrieval backends.
pub trait Retriever {
    fn name(&self) -> &str;
    fn search_query(&self, q: &Query, top_k: usize) -> Vec<Hit>;
}

/// Thin adapter so callers can benchmark uniformly.
pub struct BenchResult {
    pub variant: String,
    pub recalls: Vec<f64>,
    pub latencies_us: Vec<u64>,
}

impl BenchResult {
    pub fn mean_recall(&self) -> f64 {
        self.recalls.iter().sum::<f64>() / self.recalls.len() as f64
    }

    pub fn mean_latency_us(&self) -> f64 {
        self.latencies_us.iter().sum::<u64>() as f64 / self.latencies_us.len() as f64
    }

    pub fn p50_latency_us(&self) -> u64 {
        let mut v = self.latencies_us.clone();
        v.sort_unstable();
        v[v.len() / 2]
    }

    pub fn p95_latency_us(&self) -> u64 {
        let mut v = self.latencies_us.clone();
        v.sort_unstable();
        v[(v.len() * 95) / 100]
    }

    pub fn throughput_qps(&self) -> f64 {
        let mean_s = self.mean_latency_us() / 1_000_000.0;
        if mean_s < 1e-12 { return 0.0; }
        1.0 / mean_s
    }
}

/// Recall@K: fraction of ground-truth top-k found in `results`.
pub fn recall_at_k(results: &[Hit], ground_truth: &[usize], k: usize) -> f64 {
    let relevant: std::collections::HashSet<usize> =
        ground_truth.iter().take(k).copied().collect();
    let found = results.iter().take(k).filter(|&&(id, _)| relevant.contains(&id)).count();
    found as f64 / k.min(relevant.len()) as f64
}

/// Run one variant over all queries, collecting recall and latency.
pub fn bench_retriever<F>(
    corpus: &Corpus,
    f: F,
    variant_name: &str,
    top_k: usize,
) -> BenchResult
where
    F: Fn(&Query) -> Vec<Hit>,
{
    let mut recalls = Vec::with_capacity(corpus.queries.len());
    let mut latencies_us = Vec::with_capacity(corpus.queries.len());

    for (qi, q) in corpus.queries.iter().enumerate() {
        let t0 = Instant::now();
        let results = f(q);
        let elapsed: Duration = t0.elapsed();
        latencies_us.push(elapsed.as_micros() as u64);
        let r = recall_at_k(&results, &corpus.ground_truth[qi], top_k);
        recalls.push(r);
    }

    BenchResult {
        variant: variant_name.to_string(),
        recalls,
        latencies_us,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::generate_corpus;

    #[test]
    fn recall_at_k_perfect() {
        let hits: Vec<Hit> = (0..10).map(|i| (i, 1.0 - i as f32 * 0.1)).collect();
        let gt: Vec<usize> = (0..10).collect();
        assert!((recall_at_k(&hits, &gt, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn recall_at_k_none() {
        let hits: Vec<Hit> = (10..20).map(|i| (i, 0.5)).collect();
        let gt: Vec<usize> = (0..10).collect();
        assert!((recall_at_k(&hits, &gt, 10) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn recall_at_k_half() {
        let hits: Vec<Hit> = (0..5).chain(10..15).map(|i| (i, 0.5)).collect();
        let gt: Vec<usize> = (0..10).collect();
        let r = recall_at_k(&hits, &gt, 10);
        assert!((r - 0.5).abs() < 1e-9);
    }

    #[test]
    fn bench_result_stats() {
        let br = BenchResult {
            variant: "test".into(),
            recalls: vec![0.5, 0.7, 0.9],
            latencies_us: vec![100, 200, 300],
        };
        assert!((br.mean_recall() - 0.7).abs() < 1e-9);
        assert_eq!(br.p50_latency_us(), 200);
        assert_eq!(br.p95_latency_us(), 300);
    }

    #[test]
    fn hybrid_recall_beats_single_leg() {
        let corpus = generate_corpus(42);
        let bm25 = Bm25Index::build(
            &corpus.docs.iter().map(|d| d.tokens.clone()).collect::<Vec<_>>(),
        );
        let dense = DenseIndex::build(corpus.docs.iter().map(|d| d.vector.clone()).collect());
        let top_k = 10;
        const FETCH: usize = 50;

        let r_sparse = bench_retriever(
            &corpus,
            |q| bm25.score(&q.tokens, top_k),
            "sparse",
            top_k,
        );
        let r_dense = bench_retriever(
            &corpus,
            |q| dense.search(&q.vector, top_k),
            "dense",
            top_k,
        );
        let r_rrf = bench_retriever(
            &corpus,
            |q| {
                let sp = bm25.score(&q.tokens, FETCH);
                let de = dense.search(&q.vector, FETCH);
                rrf_fuse(&sp, &de, 60.0, top_k)
            },
            "rrf",
            top_k,
        );
        let r_coh = bench_retriever(
            &corpus,
            |q| {
                let sp = bm25.score(&q.tokens, FETCH);
                let de = dense.search(&q.vector, FETCH);
                coherence_fuse(&sp, &de, top_k)
            },
            "coh",
            top_k,
        );

        let recall_sparse = r_sparse.mean_recall();
        let recall_dense = r_dense.mean_recall();
        let recall_rrf = r_rrf.mean_recall();
        let recall_coh = r_coh.mean_recall();

        // Both hybrid variants must beat both single legs
        assert!(
            recall_rrf > recall_sparse,
            "rrf {:.3} should beat sparse {:.3}",
            recall_rrf, recall_sparse
        );
        assert!(
            recall_rrf > recall_dense,
            "rrf {:.3} should beat dense {:.3}",
            recall_rrf, recall_dense
        );
        assert!(
            recall_coh > recall_sparse,
            "coherence {:.3} should beat sparse {:.3}",
            recall_coh, recall_sparse
        );
        assert!(
            recall_coh > recall_dense,
            "coherence {:.3} should beat dense {:.3}",
            recall_coh, recall_dense
        );
    }
}
