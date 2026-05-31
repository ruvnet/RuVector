//! DABS benchmark: fixed-ef vs DABS at multiple gamma values.
//!
//! Measures recall@10, QPS, and distance computations per query on
//! synthetic Gaussian data (D=128, N=10_000, queries=200).
//!
//! Run: cargo run --release -p ruvector-dabs

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_dabs::{recall_at_k, DabsIndex, SearchMode};

const N: usize = 10_000;
const DIM: usize = 128;
const N_QUERIES: usize = 200;
const K: usize = 10;
const M: usize = 16; // graph neighbors per node

fn generate_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

fn ground_truth(index: &DabsIndex, queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|q| {
            index
                .search(q, k, SearchMode::Flat)
                .unwrap()
                .0
        })
        .collect()
}

fn run_benchmark(
    label: &str,
    index: &DabsIndex,
    queries: &[Vec<f32>],
    gts: &[Vec<u32>],
    mode: SearchMode,
    k: usize,
) -> BenchResult {
    // Warm-up pass (not timed)
    for q in queries.iter().take(5) {
        let _ = index.search(q, k, mode).unwrap();
    }

    let t0 = Instant::now();
    let mut total_ops: usize = 0;
    let mut total_recall: f64 = 0.0;

    for (q, gt) in queries.iter().zip(gts.iter()) {
        let (ids, stats) = index.search(q, k, mode).unwrap();
        total_ops += stats.dist_computations;
        total_recall += recall_at_k(&ids, gt);
    }

    let elapsed = t0.elapsed();
    let nq = queries.len() as f64;
    let qps = nq / elapsed.as_secs_f64();
    let mean_recall = total_recall / nq;
    let mean_ops = total_ops as f64 / nq;

    println!(
        "  {label:<30}  recall@{K}={mean_recall:.4}  QPS={qps:>8.1}  dist_ops/q={mean_ops:>7.1}"
    );

    BenchResult { label: label.to_string(), recall: mean_recall, qps, mean_dist_ops: mean_ops }
}

struct BenchResult {
    label: String,
    recall: f64,
    qps: f64,
    mean_dist_ops: f64,
}

fn main() {
    println!("=== ruvector-dabs benchmark ===");
    println!(
        "Dataset: N={N}, D={DIM}, queries={N_QUERIES}, k={K}, M={M}"
    );

    // Generate data
    print!("Building index ({N} vectors × {DIM} dims, M={M})... ");
    let t_build = Instant::now();
    let data = generate_data(N, DIM, 1234);
    let queries = generate_data(N_QUERIES, DIM, 5678);
    let index = DabsIndex::build(data, M).expect("build failed");
    println!("done in {:.2}s", t_build.elapsed().as_secs_f64());

    // Ground truth via exhaustive flat scan
    print!("Computing ground truth (flat scan)... ");
    let t_gt = Instant::now();
    let gts = ground_truth(&index, &queries, K);
    println!("done in {:.2}s", t_gt.elapsed().as_secs_f64());

    println!();
    println!(
        "  {:<30}  {:<14}  {:<12}  {}",
        "Mode", "recall@10", "QPS", "dist_ops/query"
    );
    println!("  {}", "-".repeat(72));

    let mut results: Vec<BenchResult> = Vec::new();

    // ── Baseline: flat exhaustive ──────────────────────────────────────────
    results.push(run_benchmark(
        "flat (exact baseline)",
        &index,
        &queries,
        &gts,
        SearchMode::Flat,
        K,
    ));

    // ── Standard fixed-ef at various ef values ──────────────────────────
    for ef in [20, 40, 64, 128, 256] {
        let label = format!("fixed_ef ef={ef}");
        results.push(run_benchmark(
            &label,
            &index,
            &queries,
            &gts,
            SearchMode::FixedEf { ef },
            K,
        ));
    }

    // ── DABS at various gamma values ──────────────────────────────────────
    for gamma in [0.05_f32, 0.1, 0.2, 0.5, 1.0, 2.0] {
        let label = format!("dabs γ={gamma:.2}");
        results.push(run_benchmark(
            &label,
            &index,
            &queries,
            &gts,
            SearchMode::Dabs { gamma },
            K,
        ));
    }

    println!();
    println!("=== Summary: DABS vs fixed-ef ===");
    println!();

    // Best fixed_ef result (highest recall in the set)
    let best_ef = results.iter().filter(|r| r.label.contains("fixed_ef"))
        .max_by(|a, b| a.recall.total_cmp(&b.recall));
    if let Some(r) = best_ef {
        println!(
            "  Best fixed-ef:  {} → recall={:.4}  QPS={:.1}  ops/q={:.1}",
            r.label, r.recall, r.qps, r.mean_dist_ops
        );
    }

    // DABS result that first matches or exceeds best_ef's recall
    if let Some(ref_r) = best_ef {
        let dabs_beat = results.iter()
            .filter(|r| r.label.contains("dabs") && r.recall >= ref_r.recall)
            .min_by(|a, b| a.mean_dist_ops.total_cmp(&b.mean_dist_ops));
        if let Some(d) = dabs_beat {
            println!(
                "  DABS matching ≥{:.4} recall: {} → recall={:.4}  QPS={:.1}  ops/q={:.1}",
                ref_r.recall, d.label, d.recall, d.qps, d.mean_dist_ops
            );
        }
    }

    // Highest-recall DABS variant
    let best_dabs = results.iter().filter(|r| r.label.contains("dabs"))
        .max_by(|a, b| a.recall.total_cmp(&b.recall).then(b.mean_dist_ops.total_cmp(&a.mean_dist_ops)));
    if let Some(r) = best_dabs {
        println!(
            "  Best-recall DABS: {} → recall={:.4}  QPS={:.1}  ops/q={:.1}",
            r.label, r.recall, r.qps, r.mean_dist_ops
        );
    }

    println!();
    println!("Hardware: {} CPUs, rustc release build", num_cpus());
    println!("Index memory: ~{:.1} MB", index.len() * index.dim() * 4 / 1024 / 1024);
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
