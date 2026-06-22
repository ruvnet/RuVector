//! Benchmark runner for ruvector-lsm-ann: streaming insert + search.
//!
//! Targets the BigANN NeurIPS'23 Streaming track: measures recall during
//! and after active insertions — the key metric the NeurIPS winner used
//! to demonstrate DiskANN + 8-bit quantization at 0.887 averaged recall.
use crate::metrics::{LatencyMetrics, RecallMetrics};
use crate::runners::core_hnsw::{HNSW_BASELINE_MEM_MB, HNSW_BASELINE_P99_MS, HNSW_BASELINE_QPS};
use crate::{claim_sota, darwin_score, BenchScore, Dataset};
use ruvector_lsm_ann::{FullLsm, LsmConfig, LsmIndex};
use std::time::Instant;

/// Benchmark the FullLsm index: insert all corpus, compact, then query.
pub fn run_lsm_ann(dataset: &Dataset, k: usize, l0_max: usize) -> anyhow::Result<BenchScore> {
    let cfg = LsmConfig {
        dims: dataset.dims,
        m: 16,
        ef_construction: 200,
        ef_search: 200,
        l0_max,
        l1_merge_threshold: 5,
    };

    let t_build = Instant::now();
    let mut idx = FullLsm::new(cfg);
    for (i, v) in dataset.corpus.iter().enumerate() {
        idx.insert(i as u64, v.clone());
    }
    idx.compact(); // flush remaining L0 → L1/L2
    let build_secs = t_build.elapsed().as_secs_f64();

    let insert_rate = dataset.corpus.len() as f64 / build_secs;
    let memory_mb = idx.memory_bytes() as f64 / (1024.0 * 1024.0);

    // Query
    let mut latencies: Vec<u128> = Vec::with_capacity(dataset.queries.len());
    let mut r10s = Vec::new();

    for (qi, q) in dataset.queries.iter().enumerate() {
        let t = Instant::now();
        let results = idx.search(q, k.max(10));
        latencies.push(t.elapsed().as_nanos());
        let ids: Vec<u64> = results.iter().map(|&(id, _)| id).collect();
        r10s.push(dataset.recall_at_k(qi, &ids, 10));
    }

    let n_q = dataset.queries.len() as f64;
    let mr10 = r10s.iter().sum::<f64>() / n_q;
    let total_s = latencies.iter().sum::<u128>() as f64 / 1e9;
    let qps = n_q / total_s;
    let latency = LatencyMetrics::from_nanos(latencies);
    let p99_s = latency.p99_us / 1_000.0;

    Ok(BenchScore {
        index: format!("lsm-ann(l0={l0_max},insert={:.0}/s)", insert_rate),
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
            p99_s,
            HNSW_BASELINE_P99_MS,
        ),
        sota: claim_sota(mr10, qps, HNSW_BASELINE_QPS),
        params: [
            ("l0_max".to_string(), l0_max.to_string()),
            ("insert_rate".to_string(), format!("{insert_rate:.0}")),
        ]
        .into(),
    })
}

/// Streaming benchmark: measure recall@10 at 3 checkpoints during insertion.
/// Models the BigANN streaming track where recall must stay high during writes.
pub fn run_lsm_streaming(dataset: &Dataset, k: usize) -> anyhow::Result<Vec<(f64, f64, f64)>> {
    let cfg = LsmConfig {
        dims: dataset.dims,
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        l0_max: 500,
        l1_merge_threshold: 3,
    };

    let mut idx = FullLsm::new(cfg);
    let n = dataset.corpus.len();
    let checkpoints = [n / 4, n / 2, n]; // 25%, 50%, 100% fill
    let mut results = Vec::new();

    let mut inserted = 0;
    for &cp in &checkpoints {
        while inserted < cp {
            idx.insert(inserted as u64, dataset.corpus[inserted].clone());
            inserted += 1;
        }

        // Checkpoint-local ground truth: only the inserted subset.
        // This matches the BigANN streaming track semantics — recall is measured
        // against vectors already in the index, not the full future corpus.
        let inserted_pairs: Vec<(u64, Vec<f32>)> = (0..inserted)
            .map(|i| (i as u64, dataset.corpus[i].clone()))
            .collect();

        let n_queries = 50.min(dataset.queries.len());
        let total_recall: f64 = dataset
            .queries
            .iter()
            .take(n_queries)
            .map(|q| {
                // True top-k among inserted vectors
                let mut dists: Vec<(u64, f32)> = inserted_pairs
                    .iter()
                    .map(|(id, v)| {
                        (
                            *id,
                            v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
                        )
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let gt: Vec<u64> = dists
                    .into_iter()
                    .take(k.max(10))
                    .map(|(id, _)| id)
                    .collect();

                let res = idx.search(q, k.max(10));
                let found: std::collections::HashSet<u64> = res.iter().map(|&(id, _)| id).collect();
                let gt_set: std::collections::HashSet<u64> = gt.iter().take(10).cloned().collect();
                let hits = gt_set.intersection(&found).count();
                hits as f64 / 10.min(gt_set.len()) as f64
            })
            .sum::<f64>()
            / n_queries as f64;

        let fill_pct = inserted as f64 / n as f64 * 100.0;
        results.push((
            fill_pct,
            total_recall,
            idx.memory_bytes() as f64 / (1024.0 * 1024.0),
        ));
    }

    Ok(results)
}
