//! Deterministic fitness evaluation of a [`Genome`] against a fixed
//! `ruvector-coherence-hnsw` workload.
//!
//! Fitness is built **only** from quantities that are exact functions of the
//! genome and the (seeded, immutable) workload: recall@k and expansion
//! counts. Wall-clock latency is deliberately excluded from fitness — timer
//! noise would make two runs of the identical (1+1)-ES with the identical
//! seed diverge in their accept/reject decisions, which would break both the
//! "witnessed run == unwitnessed run" comparison and replay verification
//! (`replay_verify` recomputes fitness from scratch and expects an exact
//! match). Latency is still measured and reported separately, at the
//! benchmark-harness level, purely as an overhead metric.

use ruvector_coherence_hnsw::{
    dataset::{clustered_queries, clustered_unit_vectors, ground_truth},
    graph::FlatGraph,
    graph::GraphConfig,
    metrics::recall_at_k,
    search::{CoherenceGatedSearch, Searcher},
};

use crate::genome::Genome;

/// Penalty weight on normalized expansions in the composite fitness score.
/// Fixed before any generation is evaluated — see ADR rationale for why this
/// must not change after benchmarking begins.
pub const EXPANSION_PENALTY: f32 = 0.30;

/// A fixed, seeded coherence-hnsw workload: dataset, graph, queries, ground
/// truth top-k. Built once per benchmark run; every genome is evaluated
/// against the exact same workload.
pub struct Workload {
    pub graph: FlatGraph,
    pub queries: Vec<Vec<f32>>,
    pub gt: Vec<Vec<u32>>,
    pub k: usize,
    pub entry_id: usize,
}

/// Dataset/graph/query parameters for [`Workload::build`], grouped to keep
/// the constructor's arity sane.
pub struct WorkloadConfig {
    pub n_clusters: usize,
    pub n_per_cluster: usize,
    pub dims: usize,
    pub cluster_std: f32,
    pub m: usize,
    pub m_longjump: usize,
    pub n_queries: usize,
    pub k: usize,
    pub entry_id: usize,
    pub data_seed: u64,
    pub query_seed: u64,
}

/// Fitness of one genome against one workload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fitness {
    pub recall_mean: f32,
    pub avg_expansions: f32,
    pub composite: f32,
}

impl Workload {
    pub fn build(cfg: &WorkloadConfig) -> Workload {
        let (data, _assignments) = clustered_unit_vectors(
            cfg.n_clusters,
            cfg.n_per_cluster,
            cfg.dims,
            cfg.cluster_std,
            cfg.data_seed,
        );
        let queries = clustered_queries(
            cfg.n_queries,
            cfg.dims,
            &data,
            cfg.n_per_cluster,
            cfg.cluster_std,
            cfg.query_seed,
        );
        let gt = ground_truth(&data, &queries, cfg.dims, cfg.k);
        let graph = FlatGraph::build(
            data,
            GraphConfig {
                m: cfg.m,
                m_longjump: cfg.m_longjump,
                dims: cfg.dims,
            },
        );
        Workload {
            graph,
            queries,
            gt,
            k: cfg.k,
            entry_id: cfg.entry_id,
        }
    }

    /// Deterministically evaluate `genome`: run every query, average recall
    /// and expansions, fold into the fixed composite formula.
    pub fn evaluate(&self, genome: Genome) -> Fitness {
        let searcher = CoherenceGatedSearch {
            threshold: genome.threshold,
        };
        let ef = genome.ef_usize();
        let n = self.queries.len().max(1);
        let mut recall_sum = 0.0f64;
        let mut expansions_sum = 0.0f64;
        for (q, gt) in self.queries.iter().zip(self.gt.iter()) {
            let result = searcher.search(&self.graph, q, self.k, ef, self.entry_id);
            recall_sum += recall_at_k(&result, gt) as f64;
            expansions_sum += result.expansions as f64;
        }
        let recall_mean = (recall_sum / n as f64) as f32;
        let avg_expansions = (expansions_sum / n as f64) as f32;
        let normalized_expansions = avg_expansions / self.graph.len().max(1) as f32;
        let composite = recall_mean - EXPANSION_PENALTY * normalized_expansions;
        Fitness {
            recall_mean,
            avg_expansions,
            composite,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::DEFAULT_GENOME;

    fn tiny_workload() -> Workload {
        Workload::build(&WorkloadConfig {
            n_clusters: 4,
            n_per_cluster: 40,
            dims: 16,
            cluster_std: 0.15,
            m: 12,
            m_longjump: 4,
            n_queries: 40,
            k: 10,
            entry_id: 0,
            data_seed: 0xAAAA,
            query_seed: 0xBBBB,
        })
    }

    #[test]
    fn evaluation_is_deterministic() {
        let w = tiny_workload();
        let f1 = w.evaluate(DEFAULT_GENOME);
        let f2 = w.evaluate(DEFAULT_GENOME);
        assert_eq!(f1, f2);
    }

    #[test]
    fn recall_is_a_fraction() {
        let w = tiny_workload();
        let f = w.evaluate(DEFAULT_GENOME);
        assert!((0.0..=1.0).contains(&f.recall_mean));
    }
}
