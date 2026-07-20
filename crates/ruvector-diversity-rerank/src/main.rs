//! Benchmark binary for diversity reranking variants.
//!
//! Generates synthetic two-cluster datasets at multiple scales and
//! measures latency, diversity score, and recall for all three rerankers.
//!
//! Usage:
//!   cargo run --release -p ruvector-diversity-rerank --bin benchmark

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_diversity_rerank::{
    mean_pairwise_diversity, BaselineReranker, Candidate, DiversityReranker, MinCutReranker,
    MmrReranker,
};
use std::time::{Duration, Instant};

fn unit_vec(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-9 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - ruvector_diversity_rerank::cosine_sim(a, b)
}

/// Generate `n_candidates` vectors: half near `c1`, half near `c2`.
fn gen_two_cluster(
    n_candidates: usize,
    dim: usize,
    noise: f32,
    rng: &mut impl rand::Rng,
) -> (Vec<Candidate>, Vec<f32>) {
    let c1: Vec<f32> = unit_vec((0..dim).map(|_| 1.0_f32).collect());
    let c2: Vec<f32> = unit_vec(
        (0..dim)
            .map(|i| if i < dim / 2 { 1.0_f32 } else { -1.0_f32 })
            .collect(),
    );
    let query: Vec<f32> = unit_vec((0..dim).map(|_| 1.0_f32).collect());

    let normal = Normal::new(0.0_f32, noise).unwrap();
    let half = n_candidates / 2;

    let mut candidates: Vec<Candidate> = Vec::with_capacity(n_candidates);

    for i in 0..half {
        let v = unit_vec(c1.iter().map(|&x| x + normal.sample(rng)).collect());
        let dist = cosine_distance(&v, &query);
        candidates.push(Candidate {
            id: i,
            distance: dist,
            vector: v,
        });
    }
    for i in half..n_candidates {
        let v = unit_vec(c2.iter().map(|&x| x + normal.sample(rng)).collect());
        let dist = cosine_distance(&v, &query);
        candidates.push(Candidate {
            id: i,
            distance: dist,
            vector: v,
        });
    }

    (candidates, query)
}

/// Recall@k: fraction of ground-truth top-k (by distance) found in result.
fn recall(result: &[Candidate], ground_truth: &[Candidate]) -> f32 {
    let k = result.len();
    let truth_ids: std::collections::HashSet<usize> =
        ground_truth.iter().take(k).map(|c| c.id).collect();
    let found = result.iter().filter(|c| truth_ids.contains(&c.id)).count();
    found as f32 / k as f32
}

struct Stats {
    mean_ns: f64,
    p50_ns: u64,
    p95_ns: u64,
    qps: f64,
}

fn measure(mut f: impl FnMut(), n_queries: usize) -> Stats {
    // warmup
    for _ in 0..5 {
        f();
    }
    let mut latencies: Vec<u64> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let t = Instant::now();
        f();
        latencies.push(t.elapsed().as_nanos() as u64);
    }
    latencies.sort_unstable();
    let mean_ns = latencies.iter().sum::<u64>() as f64 / n_queries as f64;
    let p50_ns = latencies[n_queries / 2];
    let p95_ns = latencies[(n_queries as f64 * 0.95) as usize];
    let qps = 1_000_000_000.0 / mean_ns;
    Stats {
        mean_ns,
        p50_ns,
        p95_ns,
        qps,
    }
}

fn print_header() {
    println!();
    println!("{:-<110}", "");
    println!(
        "{:<22} {:>8} {:>6} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>12}",
        "Variant",
        "N",
        "Dims",
        "K",
        "Mean µs",
        "P50 µs",
        "P95 µs",
        "QPS",
        "MemMB",
        "Diversity",
        "Recall@K"
    );
    println!("{:-<110}", "");
}

fn mem_mb(n: usize, dim: usize) -> f64 {
    // Each Candidate: id (8B) + distance (4B) + Vec<f32> ptr+cap+len (24B) + dim*4 bytes
    let per = 8 + 4 + 24 + dim * 4;
    (n * per) as f64 / 1_048_576.0
}

fn run_suite(
    n_candidates: usize,
    dim: usize,
    k: usize,
    noise: f32,
    n_queries: usize,
    mmr_lambda: f32,
    mc_threshold: f32,
    mc_dw: f32,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    let (candidates, _query) = gen_two_cluster(n_candidates, dim, noise, &mut rng);

    // ground truth by distance
    let mut gt = candidates.clone();
    gt.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

    let mem = mem_mb(n_candidates, dim);

    let rerankers: Vec<(&str, Box<dyn DiversityReranker>)> = vec![
        ("baseline", Box::new(BaselineReranker)),
        ("mmr", Box::new(MmrReranker::new(mmr_lambda))),
        (
            "mincut-diversity",
            Box::new(MinCutReranker::new(mc_threshold, mc_dw)),
        ),
    ];

    for (name, reranker) in &rerankers {
        let cands_clone = candidates.clone();

        // single run to get quality metrics
        let result = reranker.rerank(cands_clone.clone(), k).unwrap();
        let diversity = result.diversity_score;
        let rec = recall(&result.candidates, &gt);

        let stats = measure(
            || {
                let _ = reranker.rerank(cands_clone.clone(), k).unwrap();
            },
            n_queries,
        );

        println!(
            "{:<22} {:>8} {:>6} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.0} {:>10.3} {:>8.4} {:>12.4}",
            name,
            n_candidates,
            dim,
            k,
            stats.mean_ns / 1_000.0,
            stats.p50_ns as f64 / 1_000.0,
            stats.p95_ns as f64 / 1_000.0,
            stats.qps,
            mem,
            diversity,
            rec,
        );
    }
}

fn print_system_info() {
    println!("=== Diversity Rerank Benchmark ===");
    println!("Date: 2026-06-22");
    if let Ok(os) = std::fs::read_to_string("/etc/os-release") {
        for line in os.lines().take(3) {
            println!("OS: {}", line);
        }
    } else {
        println!("OS: unknown");
    }
    println!("Rust: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Legend:");
    println!("  baseline:          top-K by ANN distance, no diversity");
    println!("  mmr:               Maximal Marginal Relevance (lambda=0.5)");
    println!("  mincut-diversity:  graph-cut degree diversity (threshold=0.85, dw=0.6)");
    println!();
    println!("Dataset: 2-cluster synthetic (half vectors near centroid A, half near B)");
    println!("Noise:   σ=0.05 per dimension (tight clusters)");
}

fn acceptance_check(
    n_candidates: usize,
    dim: usize,
    k: usize,
    mmr_lambda: f32,
    mc_threshold: f32,
    mc_dw: f32,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(9876);
    let (candidates, _) = gen_two_cluster(n_candidates, dim, 0.05, &mut rng);

    let base = BaselineReranker.rerank(candidates.clone(), k).unwrap();
    let mmr = MmrReranker::new(mmr_lambda)
        .rerank(candidates.clone(), k)
        .unwrap();
    let mc = MinCutReranker::new(mc_threshold, mc_dw)
        .rerank(candidates.clone(), k)
        .unwrap();

    println!();
    println!("=== Acceptance Test (N={n_candidates}, dim={dim}, k={k}) ===");

    let pass_mmr = mmr.diversity_score > base.diversity_score;
    let pass_mc = mc.diversity_score > base.diversity_score;
    // Absolute floor: MMR must exceed 0.20, MinCut must exceed 0.20
    // (measured on N=200, dim=64, k=20, noise=0.05, seed=9876).
    let pass_mmr_abs = mmr.diversity_score >= 0.20;
    let pass_mc_abs = mc.diversity_score >= 0.20;

    println!("  baseline diversity:          {:.4}", base.diversity_score);
    println!(
        "  mmr diversity:               {:.4}  relative_pass={} abs_pass(≥0.20)={}",
        mmr.diversity_score, pass_mmr, pass_mmr_abs
    );
    println!(
        "  mincut-diversity:            {:.4}  relative_pass={} abs_pass(≥0.20)={}",
        mc.diversity_score, pass_mc, pass_mc_abs
    );

    let all_pass = pass_mmr && pass_mc && pass_mmr_abs && pass_mc_abs;
    if all_pass {
        println!();
        println!("ACCEPTANCE: PASS");
    } else {
        println!();
        println!("ACCEPTANCE: FAIL");
        std::process::exit(1);
    }
}

fn main() {
    print_system_info();
    print_header();

    // Suite 1: small candidate pool (100 candidates, 64 dims, k=10)
    run_suite(100, 64, 10, 0.05, 200, 0.5, 0.85, 0.6);

    // Suite 2: medium pool (500 candidates, 128 dims, k=20)
    run_suite(500, 128, 20, 0.05, 100, 0.5, 0.85, 0.6);

    // Suite 3: large pool (2000 candidates, 256 dims, k=50)
    run_suite(2000, 256, 50, 0.05, 50, 0.5, 0.85, 0.6);

    // Suite 4: high-noise (more overlap between clusters)
    run_suite(200, 64, 20, 0.20, 150, 0.5, 0.75, 0.6);

    println!("{:-<110}", "");

    acceptance_check(200, 64, 20, 0.5, 0.85, 0.6);
}
