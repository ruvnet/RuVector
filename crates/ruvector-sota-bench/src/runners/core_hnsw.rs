//! Benchmark runner for ruvector-core HNSW (the primary baseline).
use crate::{Dataset, BenchScore, claim_sota, darwin_score};
use crate::metrics::{RecallMetrics, LatencyMetrics};
use ruvector_core::{
    VectorDB, VectorEntry,
    types::{DbOptions, HnswConfig, SearchQuery},
    DistanceMetric,
};
use std::time::Instant;

/// Baseline QPS for darwin_score normalization (HNSWlib reference on SIFT-128).
pub const HNSW_BASELINE_QPS: f64 = 500.0;
pub const HNSW_BASELINE_MEM_MB: f64 = 200.0;
pub const HNSW_BASELINE_P99_MS: f64 = 5.0;

/// Run the ruvector-core HNSW index on a dataset and return a BenchScore.
pub fn run_core_hnsw(
    dataset: &Dataset,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    k: usize,
) -> anyhow::Result<BenchScore> {
    // ── Build ─────────────────────────────────────────────────────────────────
    let opts = DbOptions {
        dimensions: dataset.dims,
        distance_metric: DistanceMetric::Euclidean,
        storage_path: format!("/tmp/ruvector-sota-bench-{}", dataset.name),
        hnsw_config: Some(HnswConfig { m, ef_construction, ..Default::default() }),
        quantization: None,
    };

    let t_build = Instant::now();
    let mut db = VectorDB::new(opts)?;

    for (i, v) in dataset.corpus.iter().enumerate() {
        db.insert(VectorEntry {
            id: Some(i.to_string()),
            vector: v.clone(),
            metadata: Default::default(),
        })?;
    }
    let build_secs = t_build.elapsed().as_secs_f64();

    // ── Query ─────────────────────────────────────────────────────────────────
    let mut latencies: Vec<u128> = Vec::with_capacity(dataset.queries.len());
    let mut recalls_1 = Vec::new();
    let mut recalls_10 = Vec::new();
    let mut recalls_100 = Vec::new();

    for (qi, q) in dataset.queries.iter().enumerate() {
        let t = Instant::now();
        let results = db.search(SearchQuery {
            vector: q.clone(),
            k: k.max(100),
            ef_search: Some(ef_search),
            filter: None,
        })?;
        latencies.push(t.elapsed().as_nanos());

        let ids: Vec<u64> = results.iter()
            .filter_map(|r| r.id.parse::<u64>().ok())
            .collect();
        recalls_1.push(dataset.recall_at_k(qi, &ids, 1));
        recalls_10.push(dataset.recall_at_k(qi, &ids, 10));
        recalls_100.push(dataset.recall_at_k(qi, &ids, 100.min(k)));
    }

    let n_q = dataset.queries.len() as f64;
    let mean_recall_10 = recalls_10.iter().sum::<f64>() / n_q;
    let latency = LatencyMetrics::from_nanos(latencies.clone());
    let total_s = latencies.iter().sum::<u128>() as f64 / 1e9;
    let qps = n_q / total_s;

    // Approximate memory: raw vectors + ~50% overhead for HNSW graph
    let memory_mb = (dataset.corpus.len() * dataset.dims * 4) as f64 / (1024.0 * 1024.0) * 1.5;

    let score = darwin_score(
        mean_recall_10, qps, HNSW_BASELINE_QPS,
        memory_mb, HNSW_BASELINE_MEM_MB,
        latency.p99_us / 1_000.0, HNSW_BASELINE_P99_MS,
    );

    Ok(BenchScore {
        index: format!("core-hnsw(m={m},ef={ef_search})"),
        dataset: dataset.name.clone(),
        recall: RecallMetrics {
            recall_at_1:   recalls_1.iter().sum::<f64>()   / n_q,
            recall_at_10:  mean_recall_10,
            recall_at_100: recalls_100.iter().sum::<f64>() / n_q,
        },
        latency,
        qps,
        build_secs,
        memory_mb,
        darwin_score: score,
        sota: claim_sota(mean_recall_10, qps, HNSW_BASELINE_QPS),
        params: [
            ("m".to_string(), m.to_string()),
            ("ef_construction".to_string(), ef_construction.to_string()),
            ("ef_search".to_string(), ef_search.to_string()),
        ].into(),
    })
}
