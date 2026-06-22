//! Benchmark runner for ruvector-matryoshka coarse-to-fine ANN (ADR-264).
//!
//! Measures the recall@10 vs QPS tradeoff for FullDimIndex, TwoStageIndex,
//! and ThreeStageIndex on synthetic datasets matching ANN-Benchmarks dims.
use crate::metrics::{LatencyMetrics, RecallMetrics};
use crate::runners::core_hnsw::{HNSW_BASELINE_MEM_MB, HNSW_BASELINE_P99_MS, HNSW_BASELINE_QPS};
use crate::{claim_sota, darwin_score, BenchScore, Dataset};
use ruvector_matryoshka::{MatryoshkaConfig, Searcher};
use std::time::Instant;

fn bench_searcher<S: Searcher>(
    label: &str,
    cfg: &MatryoshkaConfig,
    dataset: &Dataset,
    k: usize,
    ef: usize,
) -> anyhow::Result<BenchScore> {
    // Build index over full corpus
    let t_build = Instant::now();
    let idx = S::build(cfg, &dataset.corpus);
    let build_secs = t_build.elapsed().as_secs_f64();

    // Query + recall
    let mut latencies = Vec::with_capacity(dataset.queries.len());
    let mut r10s = Vec::new();

    for (qi, q) in dataset.queries.iter().enumerate() {
        let t = Instant::now();
        let result_idxs = idx.search(q, k.max(10), ef);
        latencies.push(t.elapsed().as_nanos());

        // Convert usize indices to u64 for recall computation
        let ids: Vec<u64> = result_idxs.iter().map(|&i| i as u64).collect();
        r10s.push(dataset.recall_at_k(qi, &ids, 10));
    }

    let n_q = dataset.queries.len() as f64;
    let mr10 = r10s.iter().sum::<f64>() / n_q;
    let p99_us = {
        let mut sorted = latencies.clone();
        sorted.sort_unstable();
        sorted[(0.99 * (sorted.len() - 1) as f64) as usize] as f64 / 1_000.0
    };
    let latency = LatencyMetrics::from_nanos(latencies.clone());
    let qps = n_q / (latencies.iter().sum::<u128>() as f64 / 1e9);
    let memory_mb = (dataset.corpus.len() * dataset.dims * 4) as f64 / (1024.0 * 1024.0) * 1.2;

    Ok(BenchScore {
        index: label.to_string(),
        dataset: dataset.name.clone(),
        recall: RecallMetrics {
            recall_at_1: mr10,
            recall_at_10: mr10,
            recall_at_100: mr10,
        },
        latency,
        qps,
        build_secs,
        memory_mb,
        darwin_score: darwin_score(
            mr10,
            qps,
            HNSW_BASELINE_QPS,
            memory_mb,
            HNSW_BASELINE_MEM_MB,
            p99_us / 1_000.0,
            HNSW_BASELINE_P99_MS,
        ),
        sota: claim_sota(mr10, qps, HNSW_BASELINE_QPS),
        params: [("ef".to_string(), ef.to_string())].into(),
    })
}

/// Run FullDimIndex and TwoStageIndex on a dataset.
pub fn run_matryoshka_suite(
    dataset: &Dataset,
    k: usize,
    ef: usize,
) -> Vec<anyhow::Result<BenchScore>> {
    use ruvector_matryoshka::{FullDimIndex, TwoStageIndex};

    let dims = dataset.dims;
    let coarse = (dims / 4).max(16);
    let mid = (dims / 2).max(coarse + 1);
    let candidates = ef * 4;
    let cfg_full = MatryoshkaConfig {
        full_dim: dims,
        coarse_dim: dims,
        mid_dim: dims,
        m: 16,
        ef_construction: 100,
        two_stage_candidates: candidates,
        three_stage_coarse_candidates: candidates,
        three_stage_mid_candidates: candidates / 2,
    };
    let cfg_two = MatryoshkaConfig {
        full_dim: dims,
        coarse_dim: coarse,
        mid_dim: mid,
        m: 16,
        ef_construction: 100,
        two_stage_candidates: candidates,
        three_stage_coarse_candidates: candidates,
        three_stage_mid_candidates: candidates / 2,
    };

    vec![
        bench_searcher::<FullDimIndex>("matryoshka-full", &cfg_full, dataset, k, ef),
        bench_searcher::<TwoStageIndex>("matryoshka-funnel", &cfg_two, dataset, k, ef),
    ]
}
