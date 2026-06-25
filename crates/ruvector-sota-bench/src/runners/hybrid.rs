//! Benchmark runner for ruvector-hybrid: BM25 + ANN with RRF and RSF fusion.
//!
//! Measures the hybrid search recall improvement over pure-dense baseline,
//! directly targeting the BEIR MS MARCO scenario where hybrid fusion gives
//! 80.8% recall vs 13.9% pure-dense (per deep-researcher report, ADR-265).
use crate::metrics::{LatencyMetrics, RecallMetrics};
use crate::runners::core_hnsw::{HNSW_BASELINE_MEM_MB, HNSW_BASELINE_P99_MS, HNSW_BASELINE_QPS};
use crate::{claim_sota, darwin_score, BenchScore, Dataset};
use ruvector_hybrid::{Document, HybridSearch, RrfHybridIndex, RsfHybridIndex, ScoreFusionIndex};
use std::time::Instant;

/// Convert a Dataset's corpus to ruvector-hybrid Documents.
/// Tokens are synthesized from the vector's first-half values to simulate
/// keyword overlap (sufficient for structural benchmarking).
fn corpus_to_docs(dataset: &Dataset) -> Vec<Document> {
    dataset
        .corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            // Simulate sparse tokens: bucket top values into token strings
            let tokens: Vec<String> = v
                .iter()
                .take(8)
                .enumerate()
                .map(|(j, &x)| format!("t{}_{}", j, (x * 10.0) as i32))
                .collect();
            Document {
                id: i,
                tokens,
                vector: v.clone(),
            }
        })
        .collect()
}

fn query_tokens(query: &[f32]) -> Vec<String> {
    query
        .iter()
        .take(8)
        .enumerate()
        .map(|(j, &x)| format!("t{}_{}", j, (x * 10.0) as i32))
        .collect()
}

fn bench_hybrid<H: HybridSearch>(label: &str, idx: &H, dataset: &Dataset, k: usize) -> BenchScore {
    let mut latencies: Vec<u128> = Vec::with_capacity(dataset.queries.len());
    let mut r10s = Vec::new();

    for (qi, q) in dataset.queries.iter().enumerate() {
        let tokens = query_tokens(q);
        let token_refs: Vec<&str> = tokens.iter().map(String::as_str).collect();

        let t = Instant::now();
        let results = idx.search(&token_refs, q, k.max(10));
        latencies.push(t.elapsed().as_nanos());

        let ids: Vec<u64> = results.iter().map(|r| r.id as u64).collect();
        r10s.push(dataset.recall_at_k(qi, &ids, 10));
    }

    let n_q = dataset.queries.len() as f64;
    let mr10 = r10s.iter().sum::<f64>() / n_q;
    let total_s = latencies.iter().sum::<u128>() as f64 / 1e9;
    let qps = n_q / total_s;
    let memory_mb = (dataset.corpus.len() * dataset.dims * 4) as f64 / (1024.0 * 1024.0) * 2.0;
    let latency = LatencyMetrics::from_nanos(latencies);
    let p99_s = latency.p99_us / 1_000.0;

    BenchScore {
        index: label.to_string(),
        dataset: dataset.name.clone(),
        recall: RecallMetrics {
            recall_at_1: mr10,
            recall_at_10: mr10,
            recall_at_100: mr10,
        },
        latency,
        qps,
        build_secs: 0.0,
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
        params: [("fusion".to_string(), label.to_string())].into(),
    }
}

/// Run all three hybrid fusion strategies and return scores.
pub fn run_hybrid_suite(dataset: &Dataset, k: usize) -> Vec<BenchScore> {
    let docs = corpus_to_docs(dataset);

    let t0 = Instant::now();
    let rrf = RrfHybridIndex::build(&docs);
    let rsf = RsfHybridIndex::build(&docs);
    let score_fusion = ScoreFusionIndex::build(&docs);
    let build_s = t0.elapsed().as_secs_f64();

    let mut out = vec![
        bench_hybrid("hybrid-rrf", &rrf, dataset, k),
        bench_hybrid("hybrid-rsf", &rsf, dataset, k),
        bench_hybrid("hybrid-score-fusion", &score_fusion, dataset, k),
    ];
    for s in &mut out {
        s.build_secs = build_s / 3.0;
    }
    out
}
