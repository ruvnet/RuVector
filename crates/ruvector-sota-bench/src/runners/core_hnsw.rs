//! Benchmark runner for ruvector-core HNSW.
//!
//! Uses HnswIndex directly (bypassing VectorDB) so ef_search is honoured per
//! query — VectorDB::search ignores SearchQuery::ef_search and always uses the
//! config default. Direct index access fixes the recall stall at ~0.51.
use crate::metrics::{LatencyMetrics, RecallMetrics};
use crate::{claim_sota, darwin_score, BenchScore, Dataset};
use ruvector_core::{
    index::{hnsw::HnswIndex, VectorIndex},
    types::HnswConfig,
    DistanceMetric,
};
use std::time::Instant;

/// Baseline QPS for darwin_score normalization (HNSWlib on SIFT-128, single thread).
pub const HNSW_BASELINE_QPS: f64 = 500.0;
pub const HNSW_BASELINE_MEM_MB: f64 = 200.0;
pub const HNSW_BASELINE_P99_MS: f64 = 5.0;

/// Run ruvector-core's HNSW at a specific ef_search.
pub fn run_core_hnsw(
    dataset: &Dataset,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    k: usize,
) -> anyhow::Result<BenchScore> {
    let cfg = HnswConfig {
        m,
        ef_construction,
        ef_search,
        ..Default::default()
    };

    // ── Build ─────────────────────────────────────────────────────────────────
    let t_build = Instant::now();
    let mut idx = HnswIndex::new(dataset.dims, DistanceMetric::Euclidean, cfg)
        .map_err(|e| anyhow::anyhow!("HnswIndex::new: {e}"))?;

    for (i, v) in dataset.corpus.iter().enumerate() {
        idx.add(i.to_string(), v.clone())
            .map_err(|e| anyhow::anyhow!("HnswIndex::add {i}: {e}"))?;
    }
    let build_secs = t_build.elapsed().as_secs_f64();

    // ── Query with explicit ef_search ─────────────────────────────────────────
    let fetch_k = k.max(100); // over-fetch for recall@100 measurement
    let mut latencies: Vec<u128> = Vec::with_capacity(dataset.queries.len());
    let mut r1 = Vec::new();
    let mut r10 = Vec::new();
    let mut r100 = Vec::new();

    for (qi, q) in dataset.queries.iter().enumerate() {
        let t = Instant::now();
        // Use search_with_ef to honour the ef_search parameter
        let results = idx
            .search_with_ef(q, fetch_k, ef_search)
            .map_err(|e| anyhow::anyhow!("search_with_ef: {e}"))?;
        latencies.push(t.elapsed().as_nanos());

        let ids: Vec<u64> = results
            .iter()
            .filter_map(|r| r.id.parse::<u64>().ok())
            .collect();
        r1.push(dataset.recall_at_k(qi, &ids, 1));
        r10.push(dataset.recall_at_k(qi, &ids, 10));
        r100.push(dataset.recall_at_k(qi, &ids, 100.min(fetch_k)));
    }

    let n_q = dataset.queries.len() as f64;
    let mr10 = r10.iter().sum::<f64>() / n_q;
    let latency = LatencyMetrics::from_nanos(latencies.clone());
    let total_s = latencies.iter().sum::<u128>() as f64 / 1e9;
    let qps = n_q / total_s;

    // Rough memory: raw floats × 1.5 for HNSW graph overhead
    let memory_mb = (dataset.corpus.len() * dataset.dims * 4) as f64 / (1024.0 * 1024.0) * 1.5;

    let score = darwin_score(
        mr10,
        qps,
        HNSW_BASELINE_QPS,
        memory_mb,
        HNSW_BASELINE_MEM_MB,
        latency.p99_us / 1_000.0,
        HNSW_BASELINE_P99_MS,
    );

    Ok(BenchScore {
        index: format!("core-hnsw(m={m},ef={ef_search})"),
        dataset: dataset.name.clone(),
        recall: RecallMetrics {
            recall_at_1: r1.iter().sum::<f64>() / n_q,
            recall_at_10: mr10,
            recall_at_100: r100.iter().sum::<f64>() / n_q,
        },
        latency,
        qps,
        build_secs,
        memory_mb,
        darwin_score: score,
        sota: claim_sota(mr10, qps, HNSW_BASELINE_QPS),
        params: [
            ("m".to_string(), m.to_string()),
            ("ef_construction".to_string(), ef_construction.to_string()),
            ("ef_search".to_string(), ef_search.to_string()),
        ]
        .into(),
    })
}
