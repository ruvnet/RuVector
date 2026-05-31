// ruvector-hybrid benchmark binary.
// Measures mean/p50/p95 latency, QPS, memory, and recall@10 for four search variants.
// All numbers are measured from a real cargo run — no aspirational values.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_hybrid::{
    dense::l2_normalise, index::HybridIndex, recall_at_k, sparse::bm25_weights, DenseVec,
    HybridDoc, HybridQuery, HybridSearch, SparseVec,
};
use std::time::Instant;

// ─── Dataset parameters ───────────────────────────────────────────────────────
const N: usize = 5_000;
const DIMS: usize = 128;
const VOCAB: u32 = 1_000;
const DOC_TERMS: usize = 20;
const QUERY_TERMS: usize = 5;
const K: usize = 10;
const CANDIDATE_K: usize = 50; // candidates fetched before fusion
const N_QUERIES: usize = 500;
const WARMUP: usize = 20;
const SEED: u64 = 2026;

// Acceptance thresholds.
// Oracle = exact linear fusion α=0.5 over ALL N docs.
// candidate_k=50 means we search only the top 50 per channel before fusing,
// so recall vs the exact oracle will be << 100%: that is the expected tradeoff.
// The meaningful acceptance criteria are:
//   (a) Hybrid must beat the worst single-channel baseline.
//   (b) HybridRRF must not hurt recall vs SparseOnly (the stronger single channel here).
//   (c) Fusion overhead must be bounded.
const MAX_FUSION_OVERHEAD_US: f64 = 500.0; // fusion overhead per query ≤ 500 µs

fn random_dense(rng: &mut StdRng) -> DenseVec {
    let dist = Normal::new(0.0f32, 1.0).unwrap();
    let mut v = DenseVec::new((0..DIMS).map(|_| dist.sample(rng)).collect());
    l2_normalise(&mut v);
    v
}

fn random_sparse(rng: &mut StdRng, n_terms: usize) -> SparseVec {
    let mut ids: Vec<u32> = (0..n_terms).map(|_| rng.gen_range(0..VOCAB)).collect();
    ids.sort_unstable();
    ids.dedup();
    let tf: Vec<f32> = ids.iter().map(|_| rng.gen_range(1.0f32..4.0)).collect();
    let df: Vec<u32> = ids
        .iter()
        .map(|&t| {
            if t < VOCAB / 5 {
                rng.gen_range(1u32..50)
            } else {
                rng.gen_range(100..500)
            }
        })
        .collect();
    bm25_weights(
        &ids,
        &tf,
        &df,
        N as u32,
        n_terms as f32,
        DOC_TERMS as f32,
        1.5,
        0.75,
    )
}

fn percentile(sorted_us: &[f64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_us.len() - 1) as f64).round() as usize;
    sorted_us[idx.min(sorted_us.len() - 1)]
}

struct BenchResult {
    name: &'static str,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    recall: f64,
    mem_kb: f64,
}

fn run_variant<F>(
    name: &'static str,
    queries: &[HybridQuery],
    oracles: &[Vec<ruvector_hybrid::Scored>],
    mem_kb: f64,
    mut search_fn: F,
) -> BenchResult
where
    F: FnMut(&HybridQuery) -> Vec<ruvector_hybrid::Scored>,
{
    // Warmup
    for q in queries.iter().take(WARMUP) {
        let _ = search_fn(q);
    }

    let mut latencies_us = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f64;

    for (q, oracle) in queries.iter().zip(oracles.iter()) {
        let t0 = Instant::now();
        let result = search_fn(q);
        let elapsed = t0.elapsed().as_secs_f64() * 1e6;
        latencies_us.push(elapsed);
        total_recall += recall_at_k(&result, oracle);
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_us = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;
    let p50_us = percentile(&latencies_us, 50.0);
    let p95_us = percentile(&latencies_us, 95.0);
    let qps = 1e6 / mean_us;
    let recall = total_recall / queries.len() as f64;

    BenchResult {
        name,
        mean_us,
        p50_us,
        p95_us,
        qps,
        recall,
        mem_kb,
    }
}

fn print_result(r: &BenchResult) {
    println!(
        "  {:14} | mean={:6.1}µs  p50={:6.1}µs  p95={:7.1}µs  QPS={:8.0}  recall={:5.1}%  mem={:.1}KB",
        r.name, r.mean_us, r.p50_us, r.p95_us, r.qps, r.recall * 100.0, r.mem_kb
    );
}

fn main() {
    // ─── Environment info ─────────────────────────────────────────────────────
    println!("════════════════════════════════════════════════════════════════════");
    println!("  ruvector-hybrid benchmark");
    println!("════════════════════════════════════════════════════════════════════");
    println!("  OS      : {}", std::env::consts::OS);
    println!("  Arch    : {}", std::env::consts::ARCH);
    println!("  Rustc   : {}", rustc_version());
    println!("  Dataset : N={N}  D={DIMS}  vocab={VOCAB}  doc_terms={DOC_TERMS}");
    println!("  Queries : {N_QUERIES}  K={K}  candidate_K={CANDIDATE_K}  warmup={WARMUP}");
    println!("════════════════════════════════════════════════════════════════════");

    let mut rng = StdRng::seed_from_u64(SEED);

    // ─── Build index ──────────────────────────────────────────────────────────
    let t_build = Instant::now();
    let mut idx = HybridIndex::new(DIMS);
    for i in 0..N as u32 {
        let dense = random_dense(&mut rng);
        let sparse = random_sparse(&mut rng, DOC_TERMS);
        idx.insert(HybridDoc {
            id: i,
            dense,
            sparse,
        });
    }
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;

    let dense_mem_kb = (N * DIMS * 4) as f64 / 1024.0;
    let sparse_mem_kb = idx.memory_bytes() as f64 / 1024.0 - dense_mem_kb;
    let total_mem_kb = idx.memory_bytes() as f64 / 1024.0;

    println!("  Build   : {build_ms:.1}ms");
    println!("  Mem     : dense={dense_mem_kb:.0}KB  sparse={sparse_mem_kb:.0}KB  total={total_mem_kb:.0}KB");
    println!();

    // ─── Generate queries and oracle results ──────────────────────────────────
    let queries: Vec<HybridQuery> = (0..N_QUERIES)
        .map(|_| HybridQuery {
            dense: random_dense(&mut rng),
            sparse: random_sparse(&mut rng, QUERY_TERMS),
        })
        .collect();

    // Oracle: linear fusion α=0.5 over ALL N docs (exact)
    let oracles: Vec<Vec<ruvector_hybrid::Scored>> = queries
        .iter()
        .map(|q| idx.search_linear(q, K, N, 0.5))
        .collect();

    // ─── Run variants ─────────────────────────────────────────────────────────
    println!("  Variant        | Mean latency   p50 latency   p95 latency   QPS         Recall@10  Memory");
    println!("  ─────────────────────────────────────────────────────────────────────────────────────────");

    let r_dense = run_variant("DenseOnly", &queries, &oracles, dense_mem_kb, |q| {
        idx.search_dense(q, K)
    });
    let r_sparse = run_variant("SparseOnly", &queries, &oracles, sparse_mem_kb, |q| {
        idx.search_sparse(q, K)
    });
    let r_rrf = run_variant("HybridRRF", &queries, &oracles, total_mem_kb, |q| {
        idx.search_rrf(q, K, CANDIDATE_K)
    });
    let r_linear = run_variant("HybridLinear", &queries, &oracles, total_mem_kb, |q| {
        idx.search_linear(q, K, CANDIDATE_K, 0.5)
    });

    print_result(&r_dense);
    print_result(&r_sparse);
    print_result(&r_rrf);
    print_result(&r_linear);

    // ─── Acceptance tests ─────────────────────────────────────────────────────
    println!();
    println!("═══ Acceptance Tests ════════════════════════════════════════════════");

    // Fusion overhead = hybrid mean latency - max(dense, sparse) mean latency
    let max_baseline_us = r_dense.mean_us.max(r_sparse.mean_us);
    let rrf_overhead_us = (r_rrf.mean_us - max_baseline_us).max(0.0);
    let linear_overhead_us = (r_linear.mean_us - max_baseline_us).max(0.0);
    let overhead_ok =
        rrf_overhead_us <= MAX_FUSION_OVERHEAD_US && linear_overhead_us <= MAX_FUSION_OVERHEAD_US;

    // Hybrid must beat the min single-channel recall (proves fusion adds value).
    let min_single_recall = r_dense.recall.min(r_sparse.recall);
    let rrf_beats_min = r_rrf.recall > min_single_recall;
    let linear_beats_min = r_linear.recall > min_single_recall;

    // Hybrid must not substantially hurt the max single-channel recall (≤2% regression).
    let max_single_recall = r_dense.recall.max(r_sparse.recall);
    let rrf_no_regression = r_rrf.recall >= max_single_recall - 0.02;
    let linear_no_regression = r_linear.recall >= max_single_recall - 0.02;

    println!(
        "  [{}] HybridRRF recall > min(Dense,Sparse) recall  ({:.1}% > {:.1}%)",
        pass(rrf_beats_min),
        r_rrf.recall * 100.0,
        min_single_recall * 100.0
    );
    println!(
        "  [{}] HybridLinear recall > min(Dense,Sparse) recall  ({:.1}% > {:.1}%)",
        pass(linear_beats_min),
        r_linear.recall * 100.0,
        min_single_recall * 100.0
    );
    println!(
        "  [{}] HybridRRF no recall regression vs best single  ({:.1}% >= {:.1}%-2%)",
        pass(rrf_no_regression),
        r_rrf.recall * 100.0,
        max_single_recall * 100.0
    );
    println!(
        "  [{}] HybridLinear no recall regression vs best single  ({:.1}% >= {:.1}%-2%)",
        pass(linear_no_regression),
        r_linear.recall * 100.0,
        max_single_recall * 100.0
    );
    println!(
        "  [{}] Fusion overhead <= {:.0}µs  (RRF={:.1}µs  Linear={:.1}µs)",
        pass(overhead_ok),
        MAX_FUSION_OVERHEAD_US,
        rrf_overhead_us,
        linear_overhead_us
    );

    println!();
    println!("  Note: oracle = exact linear fusion α=0.5 over ALL {N} docs.");
    println!(
        "  Hybrid variants search only top {CANDIDATE_K} candidates per channel before fusing."
    );
    println!("  Recall gap vs oracle is expected and reflects the candidate_k approximation.");

    println!();
    let all_pass = rrf_beats_min
        && linear_beats_min
        && rrf_no_regression
        && linear_no_regression
        && overhead_ok;
    if all_pass {
        println!("  ✓ ALL ACCEPTANCE TESTS PASSED");
    } else {
        println!("  ✗ SOME ACCEPTANCE TESTS FAILED");
        std::process::exit(1);
    }
    println!("════════════════════════════════════════════════════════════════════");
}

fn pass(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}
