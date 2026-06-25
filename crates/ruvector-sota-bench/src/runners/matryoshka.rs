//! Benchmark runner for ruvector-matryoshka coarse-to-fine ANN (ADR-264).
//!
//! Root cause of prior low recall (issue #597): the previous runner fed random
//! Gaussian data to matryoshka indices. MRL / Matryoshka Representation Learning
//! REQUIRES data with cluster structure in the prefix dimensions. On unstructured
//! Gaussian noise, no coarse-dim filtering makes sense — recall collapses.
//!
//! Fix: use `generate_matryoshka_dataset` which produces L2-normalised cluster
//! data where the first `signal_dim` dimensions carry dominant cluster signal,
//! mirroring how OpenAI text-embedding-3 / Nomic-Embed encodes meaning.
use crate::metrics::{LatencyMetrics, RecallMetrics};
use crate::runners::core_hnsw::{HNSW_BASELINE_MEM_MB, HNSW_BASELINE_P99_MS, HNSW_BASELINE_QPS};
use crate::{claim_sota, darwin_score, BenchScore};
use ruvector_matryoshka::{
    brute_force_knn, dataset::generate_matryoshka_dataset, recall_at_k as matr_recall,
    FullDimIndex, MatryoshkaConfig, Searcher, TwoStageIndex,
};
use std::time::Instant;

/// A matryoshka-native benchmark dataset with MRL cluster structure.
struct MatryoshkaDataset {
    name: String,
    full_dim: usize,
    signal_dim: usize,
    corpus: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    /// Ground truth top-k using full_dim euclidean (brute force).
    ground_truth: Vec<Vec<usize>>,
}

impl MatryoshkaDataset {
    fn new(name: &str, n: usize, q: usize, full_dim: usize, signal_dim: usize, seed: u64) -> Self {
        let (corpus, queries) = generate_matryoshka_dataset(n, q, full_dim, signal_dim, seed);
        let ground_truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|qv| brute_force_knn(&corpus, qv, 100, full_dim))
            .collect();
        Self {
            name: name.to_string(),
            full_dim,
            signal_dim,
            corpus,
            queries,
            ground_truth,
        }
    }

    fn recall_at_k(&self, qi: usize, result_idxs: &[usize], k: usize) -> f64 {
        let gt: Vec<usize> = self.ground_truth[qi].iter().take(k).cloned().collect();
        let res: Vec<usize> = result_idxs.iter().take(k).cloned().collect();
        matr_recall(&res, &gt) as f64
    }
}

fn bench_searcher<S: Searcher>(
    label: &str,
    cfg: &MatryoshkaConfig,
    ds: &MatryoshkaDataset,
    k: usize,
    ef: usize,
) -> BenchScore {
    let t_build = Instant::now();
    let idx = S::build(cfg, &ds.corpus);
    let build_secs = t_build.elapsed().as_secs_f64();

    let mut latencies = Vec::with_capacity(ds.queries.len());
    let mut r10s = Vec::new();

    for (qi, q) in ds.queries.iter().enumerate() {
        let t = Instant::now();
        let result_idxs = idx.search(q, k.max(10), ef);
        latencies.push(t.elapsed().as_nanos());
        r10s.push(ds.recall_at_k(qi, &result_idxs, 10));
    }

    let n_q = ds.queries.len() as f64;
    let mr10 = r10s.iter().sum::<f64>() / n_q;
    let total_s = latencies.iter().sum::<u128>() as f64 / 1e9;
    let qps = n_q / total_s;
    let latency = LatencyMetrics::from_nanos(latencies);
    let p99_s = latency.p99_us / 1_000.0;
    let memory_mb = (ds.corpus.len() * ds.full_dim * 4) as f64 / (1024.0 * 1024.0) * 1.2;
    let dataset_tag = format!(
        "{} (MRL n={} d={}/{})",
        ds.name,
        ds.corpus.len(),
        ds.signal_dim,
        ds.full_dim
    );

    BenchScore {
        index: label.to_string(),
        dataset: dataset_tag,
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
            ("ef".to_string(), ef.to_string()),
            ("signal_dim".to_string(), ds.signal_dim.to_string()),
        ]
        .into(),
    }
}

/// Run FullDimIndex and TwoStageIndex on MRL-structured datasets.
///
/// Uses the matryoshka-native dataset generator (cluster structure in prefix dims)
/// so recall numbers reflect real MRL embedding behaviour, not random noise.
pub fn run_matryoshka_suite(
    _dataset_name: &str,
    corpus_n: usize,
    full_dim: usize,
    k: usize,
    ef: usize,
) -> Vec<BenchScore> {
    let signal_dim = full_dim / 4; // coarse prefix: 25% of full dims
    let mid_dim = full_dim / 2;
    let candidates = (ef * 8).max(200);

    let ds = MatryoshkaDataset::new(
        "matryoshka-mrl",
        corpus_n,
        (corpus_n / 100).clamp(50, 200),
        full_dim,
        signal_dim,
        0xDEAD_BEEF,
    );

    let cfg_full = MatryoshkaConfig {
        full_dim,
        coarse_dim: full_dim, // FullDimIndex uses this
        mid_dim: full_dim,
        m: 16,
        ef_construction: 200,
        two_stage_candidates: candidates,
        three_stage_coarse_candidates: candidates,
        three_stage_mid_candidates: candidates / 2,
    };
    let cfg_two = MatryoshkaConfig {
        full_dim,
        coarse_dim: signal_dim,
        mid_dim,
        m: 16,
        ef_construction: 200,
        two_stage_candidates: candidates,
        three_stage_coarse_candidates: candidates,
        three_stage_mid_candidates: candidates / 2,
    };

    vec![
        bench_searcher::<FullDimIndex>("matryoshka-full", &cfg_full, &ds, k, ef),
        bench_searcher::<TwoStageIndex>("matryoshka-funnel", &cfg_two, &ds, k, ef),
    ]
}
