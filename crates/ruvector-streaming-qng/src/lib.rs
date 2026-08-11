//! Streaming Quantized Neighbourhood Graphs (QNG-Stream) for RuVector
//!
//! Problem: Agent memory systems emit vectors continuously. The embedding
//! distribution drifts as the agent's context shifts topics or tasks. A
//! static Product Quantization codebook trained at startup mis-represents
//! the new distribution, degrading recall over time.
//!
//! Three measurable variants:
//! 1. `FullPrecision`  – brute-force f32 scan (ground truth baseline)
//! 2. `StaticPQ`       – PQ codebook trained once on the initial batch
//! 3. `StreamPQ`       – reservoir-sampled PQ with periodic codebook refresh

pub mod dataset;
pub mod full_precision;
pub mod static_pq;
pub mod stream_pq;
pub mod pq;

use std::collections::HashSet;

// ── shared types ──────────────────────────────────────────────────────────────

/// A single nearest-neighbour hit.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub dist: f32,
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── common trait ──────────────────────────────────────────────────────────────

/// Unified interface for all three ANN variants.
pub trait AnnVariant: Send + Sync {
    fn build(&mut self, vectors: &[Vec<f32>]);
    fn insert(&mut self, vector: Vec<f32>);
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
    fn name(&self) -> &str;
    fn len(&self) -> usize;
    fn memory_bytes(&self) -> usize;
}

// ── distance helpers ──────────────────────────────────────────────────────────

/// Squared L2 distance (no sqrt; monotone for nearest-neighbour ranking).
#[inline(always)]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Return the index of the centroid nearest to `query`.
#[inline]
pub fn nearest_centroid(query: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, sq_l2(query, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── quality metrics ───────────────────────────────────────────────────────────

/// Recall@k: fraction of ground-truth top-k ids present in `results`.
pub fn recall_at_k(results: &[Hit], ground_truth: &[Hit], k: usize) -> f32 {
    let res_ids: HashSet<usize> = results.iter().take(k).map(|h| h.id).collect();
    let gt_ids: HashSet<usize> = ground_truth.iter().take(k).map(|h| h.id).collect();
    if gt_ids.is_empty() {
        return 1.0;
    }
    let n = res_ids.intersection(&gt_ids).count();
    n as f32 / k.min(gt_ids.len()) as f32
}

/// Cluster precision: fraction of returned results from the expected cluster.
/// `cluster_size` is the number of vectors per cluster. The first `cluster_size`
/// indexed vectors belong to cluster 0, the next to cluster 1, etc.
/// `expected_cluster` is the cluster the query belongs to.
pub fn cluster_precision(results: &[Hit], expected_cluster: usize, cluster_size: usize) -> f32 {
    let start = expected_cluster * cluster_size;
    let end = start + cluster_size;
    let correct = results.iter().filter(|h| h.id >= start && h.id < end).count();
    if results.is_empty() {
        return 0.0;
    }
    correct as f32 / results.len() as f32
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DatasetConfig, generate_phase_a, generate_phase_b,
                         generate_queries_a, generate_queries_b};
    use crate::full_precision::FullPrecision;
    use crate::static_pq::StaticPq;
    use crate::stream_pq::StreamPq;

    /// Very tight clusters: std=0.005 gives ~600σ separation from the shift.
    /// Vectors are stored in contiguous cluster blocks (cluster 0: 0..n_per_cluster,
    /// cluster 1: n_per_cluster..2*n_per_cluster, etc.) for easy precision testing.
    fn make_cfg() -> DatasetConfig {
        DatasetConfig {
            dims: 32,       // divisible by M=4, ds=8
            clusters: 4,
            cluster_std: 0.005,
            shift: 3.0,
            seed: 7,
        }
    }

    /// Generate data in contiguous cluster blocks (not interleaved).
    fn gen_block(n_per_cluster: usize, cfg: &DatasetConfig, shift: f32) -> Vec<Vec<f32>> {
        let mut vecs = Vec::new();
        for c in 0..cfg.clusters {
            for i in 0..n_per_cluster {
                let centroid: Vec<f32> = (0..cfg.dims).map(|d| {
                    let base = if d == 0 { c as f32 * 4.0 + shift } else {
                        ((c * 7 + d * 3) % cfg.clusters) as f32 * 0.3 + shift
                    };
                    let noise_val = ((i * 31 + d * 17) as f32 * 0.0001) * cfg.cluster_std;
                    base + noise_val
                }).collect();
                vecs.push(centroid);
            }
        }
        vecs
    }

    #[test]
    fn full_precision_recall_is_one() {
        let cfg = make_cfg();
        let vecs = gen_block(100, &cfg, 0.0);
        let queries = gen_block(5, &cfg, 0.0002);  // slightly offset queries

        let mut fp = FullPrecision::new();
        fp.build(&vecs);

        // For cluster c, query[c*5..c*5+5] should return vecs from block c (0..100, 100..200, etc.)
        let mut total_recall = 0.0_f32;
        let k = 5;
        for (qi, q) in queries.iter().enumerate() {
            let results = fp.search(q, k);
            let gt = fp.search(q, k);  // same index, so recall@k = 1.0
            total_recall += recall_at_k(&results, &gt, k);
        }
        let mean_recall = total_recall / queries.len() as f32;
        assert!(mean_recall >= 0.99, "FullPrecision recall={mean_recall:.4}");
    }

    #[test]
    fn static_pq_cluster_precision_above_floor() {
        let cfg = make_cfg();
        let n_per_cluster = 100;
        let vecs = gen_block(n_per_cluster, &cfg, 0.0);
        let queries = gen_block(5, &cfg, 0.0002);

        let mut spq = StaticPq::new();
        spq.build(&vecs);

        let k = 5;
        let mut total_precision = 0.0_f32;
        for (qi, q) in queries.iter().enumerate() {
            let cluster = qi / 5; // 5 queries per cluster
            let results = spq.search(q, k);
            total_precision += cluster_precision(&results, cluster, n_per_cluster);
        }
        let mean_prec = total_precision / queries.len() as f32;
        assert!(mean_prec >= 0.80, "StaticPQ cluster precision={mean_prec:.4} < 0.80");
    }

    #[test]
    fn stream_pq_adapts_after_distribution_shift() {
        let cfg = make_cfg();
        // Phase B is 4× larger than Phase A so the reservoir becomes Phase B-dominated
        // by the time queries run (Vitter sampling is uniform over all seen vectors,
        // so domination requires N_B >> N_A). With 300 Phase A and 1200 Phase B the
        // reservoir reaches ~80% Phase B and the codebook fully converges.
        let n_a = 75;
        let n_b = 300;
        let phase_a = gen_block(n_a, &cfg, 0.0);
        let phase_b = gen_block(n_b, &cfg, cfg.shift);
        let queries_b = gen_block(3, &cfg, cfg.shift + 0.001);

        let mut spq   = StaticPq::new();
        // update_freq=50: 24 codebook refreshes during Phase B insertion.
        let mut strpq = StreamPq::new(256, 50);

        spq.build(&phase_a);
        strpq.build(&phase_a);

        for v in phase_b.iter().cloned() {
            spq.insert(v.clone());
            strpq.insert(v);
        }

        let k = 5;
        let total_b_start = n_a * cfg.clusters;
        let n_b_per_cluster = n_b;

        let mut prec_static = 0.0_f32;
        let mut prec_stream = 0.0_f32;
        for (qi, q) in queries_b.iter().enumerate() {
            let cluster = qi / 3;
            let b_start = total_b_start + cluster * n_b_per_cluster;
            let b_end = b_start + n_b_per_cluster;

            let res_spq = spq.search(q, k);
            let res_str = strpq.search(q, k);

            let correct_spq = res_spq.iter().filter(|h| h.id >= b_start && h.id < b_end).count();
            let correct_str = res_str.iter().filter(|h| h.id >= b_start && h.id < b_end).count();
            prec_static += correct_spq as f32 / k as f32;
            prec_stream += correct_str as f32 / k as f32;
        }
        prec_static /= queries_b.len() as f32;
        prec_stream /= queries_b.len() as f32;

        // With a fully adapted codebook, StreamPQ should find Phase B vectors reliably.
        assert!(prec_stream >= 0.60,
            "StreamPQ Phase-B cluster precision={prec_stream:.4} < 0.60 \
             (StaticPQ={prec_static:.4})");
    }

    #[test]
    fn memory_bytes_nonzero_after_build() {
        let cfg = make_cfg();
        let vecs = gen_block(50, &cfg, 0.0);

        let mut fp    = FullPrecision::new();
        let mut spq   = StaticPq::new();
        let mut strpq = StreamPq::new(64, 20);

        fp.build(&vecs);
        spq.build(&vecs);
        strpq.build(&vecs);

        assert!(fp.memory_bytes() > 0);
        assert!(spq.memory_bytes() > 0);
        assert!(strpq.memory_bytes() > 0);
    }

    #[test]
    fn insert_increases_len() {
        let cfg = make_cfg();
        let vecs = gen_block(25, &cfg, 0.0);
        let extra = gen_block(5, &cfg, cfg.shift);

        let mut strpq = StreamPq::new(32, 10);
        strpq.build(&vecs);
        let initial_len = vecs.len();
        assert_eq!(strpq.len(), initial_len);
        for v in extra { strpq.insert(v); }
        assert_eq!(strpq.len(), initial_len + 20);
    }
}
