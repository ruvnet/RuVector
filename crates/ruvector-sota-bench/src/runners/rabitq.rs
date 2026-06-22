//! Benchmark runner for ruvector-rabitq — 1-bit compressed ANN.
//!
//! The IVF-RaBitQ paper (ACM SIGMOD 2024) demonstrates 99.3% recall@10
//! vs IVF-PQ's 79.2% on SIFT1M at comparable QPS — a 20pp gap. This
//! is RuVector's primary SOTA claim against product-quantized baselines.
//!
//! Three variants:
//!   - FlatF32Index     — exact brute-force baseline (recall = 1.0)
//!   - RabitqIndex      — 1-bit RaBitQ (512× compression, high recall)
//!   - RabitqPlusIndex  — RaBitQ + refinement re-rank (highest recall)
use crate::metrics::{LatencyMetrics, RecallMetrics};
use crate::runners::core_hnsw::{HNSW_BASELINE_MEM_MB, HNSW_BASELINE_P99_MS, HNSW_BASELINE_QPS};
use crate::{claim_sota, darwin_score, BenchScore, Dataset};
use ruvector_rabitq::index::{AnnIndex, FlatF32Index, RabitqIndex, RabitqPlusIndex, SearchResult};
use ruvector_rabitq::rotation::RandomRotationKind;
use std::time::Instant;

fn to_bench_score(
    label: &str,
    dataset: &Dataset,
    results_per_query: Vec<Vec<SearchResult>>,
    latencies: Vec<u128>,
    build_secs: f64,
    memory_mb: f64,
    k: usize,
) -> BenchScore {
    let n_q = dataset.queries.len() as f64;
    let mut r1 = Vec::new();
    let mut r10 = Vec::new();
    let mut r100 = Vec::new();

    for (qi, results) in results_per_query.iter().enumerate() {
        let ids: Vec<u64> = results.iter().map(|r| r.id as u64).collect();
        r1.push(dataset.recall_at_k(qi, &ids, 1));
        r10.push(dataset.recall_at_k(qi, &ids, 10));
        r100.push(dataset.recall_at_k(qi, &ids, 100.min(k)));
    }

    let mr10 = r10.iter().sum::<f64>() / n_q;
    let total_s = latencies.iter().sum::<u128>() as f64 / 1e9;
    let qps = n_q / total_s;
    let latency = LatencyMetrics::from_nanos(latencies);
    let p99_s = latency.p99_us / 1_000.0;

    BenchScore {
        index: label.to_string(),
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
        params: [("index".to_string(), label.to_string())].into(),
    }
}

fn bench_index<I: AnnIndex>(
    label: &str,
    mut idx: I,
    dataset: &Dataset,
    k: usize,
) -> anyhow::Result<BenchScore> {
    // Build
    let t_build = Instant::now();
    for (i, v) in dataset.corpus.iter().enumerate() {
        idx.add(i, v.clone())
            .map_err(|e| anyhow::anyhow!("add: {e}"))?;
    }
    let build_secs = t_build.elapsed().as_secs_f64();

    // Approximate memory for 1-bit codes: dim/8 bytes per vector + overhead
    let memory_mb = (dataset.corpus.len() * (dataset.dims / 8 + 16)) as f64 / (1024.0 * 1024.0);

    // Query
    let mut latencies = Vec::with_capacity(dataset.queries.len());
    let mut results_per_query = Vec::with_capacity(dataset.queries.len());

    for q in &dataset.queries {
        let t = Instant::now();
        let res = idx
            .search(q, k.max(100))
            .map_err(|e| anyhow::anyhow!("search: {e}"))?;
        latencies.push(t.elapsed().as_nanos());
        results_per_query.push(res);
    }

    Ok(to_bench_score(
        label,
        dataset,
        results_per_query,
        latencies,
        build_secs,
        memory_mb,
        k,
    ))
}

/// Run all three RaBitQ variants: exact baseline, 1-bit RaBitQ, RaBitQ+.
pub fn run_rabitq_suite(dataset: &Dataset, k: usize) -> Vec<anyhow::Result<BenchScore>> {
    let seed = 42u64;
    let rerank = 10; // over-fetch 10× candidates, rerank by exact f32
    vec![
        // Exact brute-force baseline (recall = 1.0 by definition)
        bench_index(
            "rabitq-flat-f32",
            FlatF32Index::new(dataset.dims),
            dataset,
            k,
        ),
        // 1-bit RaBitQ with HadamardSigned rotation (highest QPS)
        bench_index(
            "rabitq-1bit",
            RabitqIndex::new_with_rotation(dataset.dims, seed, RandomRotationKind::HadamardSigned),
            dataset,
            k,
        ),
        // RaBitQ+ with re-rank (highest recall, matches paper's 99.3%)
        bench_index(
            "rabitq-plus",
            RabitqPlusIndex::new(dataset.dims, seed, rerank),
            dataset,
            k,
        ),
    ]
}
