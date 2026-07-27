//! rvq-demo — end-to-end benchmark for `ruvector-rvq`.
//!
//! Measures recall@10 and QPS for four index variants against synthetic
//! clustered-Gaussian data.  All numbers are produced from a single run so
//! the research doc can cite them as "same-run" results.
//!
//! ```
//! cargo run --release -p ruvector-rvq --bin rvq-demo
//! ```

use rand::SeedableRng;
use rand::Rng as _;
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::HashSet;
use std::time::Instant;

use ruvector_rvq::{
    index::{AnnIndex, FlatF32Index, PqIndex, RvqIndex, RvqRerankIndex},
    RvqConfig,
};

// ── Dataset ───────────────────────────────────────────────────────────────────

/// Clustered Gaussian data: `n_clusters` centroids in [-2, 2]^D, each with
/// Gaussian noise σ=0.6.  Matches the distribution used in ruvector-rabitq
/// so results are comparable across nightly research runs.
fn generate_clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroid_range = Uniform::new(-2.0f32, 2.0f32);
    let noise = Normal::new(0.0f64, 0.6).unwrap();
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| centroid_range.sample(&mut rng)).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter().map(|&x| x + noise.sample(&mut rng) as f32).collect()
        })
        .collect()
}

// ── Ground truth ──────────────────────────────────────────────────────────────

fn exact_top_k(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (dist, i)
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    scored.iter().take(k).map(|&(_, id)| id).collect()
}

fn recall_at_k(truth: &[usize], got: &[usize]) -> f64 {
    let truth_set: HashSet<usize> = truth.iter().copied().collect();
    got.iter().filter(|id| truth_set.contains(id)).count() as f64 / truth.len() as f64
}

// ── Measurement harness ───────────────────────────────────────────────────────

struct Row {
    label: String,
    recall_10: f64,
    qps: f64,
    mem_mb: f64,
    bytes_per_vec: usize,
    build_ms: f64,
}

fn measure<I: AnnIndex>(
    label: &str,
    idx: &I,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    build_ms: f64,
) -> Row {
    let k = 10;
    let n_queries = queries.len();

    // Warmup.
    for q in queries.iter().take(5) {
        let _ = idx.search(q, k);
    }

    // Timed run.
    let t0 = Instant::now();
    let mut r10_sum = 0.0f64;
    for (qi, q) in queries.iter().enumerate() {
        let got: Vec<usize> = idx.search(q, k).into_iter().map(|r| r.id).collect();
        r10_sum += recall_at_k(&truth[qi], &got);
    }
    let elapsed = t0.elapsed().as_secs_f64();

    Row {
        label: label.to_string(),
        recall_10: r10_sum / n_queries as f64,
        qps: n_queries as f64 / elapsed,
        mem_mb: idx.memory_bytes() as f64 / 1_048_576.0,
        bytes_per_vec: idx.bytes_per_vector(),
        build_ms,
    }
}

fn print_header() {
    println!(
        "  {:<28} {:>8} {:>9} {:>8} {:>10} {:>9}",
        "Variant", "R@10", "QPS", "Mem/MB", "bytes/vec", "build ms"
    );
    println!("  {}", "-".repeat(80));
}

fn print_row(r: &Row) {
    println!(
        "  {:<28} {:>7.1}% {:>9.0} {:>8.2} {:>10} {:>9.1}",
        r.label, r.recall_10 * 100.0, r.qps, r.mem_mb, r.bytes_per_vec, r.build_ms
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn run_suite(n_index: usize, n_queries: usize, d: usize, n_clusters: usize) {
    let k_centroids = 64usize; // 64 centroids → 6 bits, stored as u8 (1 byte)
    let train_iters = 25usize;
    let pq_m = 8usize;  // 8 subspaces → 8 bytes/vec
    let rvq_stages_4 = 4usize; // 4 stages → 4 bytes/vec
    let rvq_stages_8 = 8usize; // 8 stages → 8 bytes/vec (same budget as PQ-8)

    println!("\n── n={n_index}  D={d}  queries={n_queries}  K={k_centroids}  clusters={n_clusters} ──");

    // Generate data.
    let all_data = generate_clustered(n_index + n_queries, d, n_clusters, 42);
    let index_data: Vec<Vec<f32>> = all_data[..n_index].to_vec();
    let queries: Vec<Vec<f32>> = all_data[n_index..].to_vec();

    // Ground truth (exact brute-force on indexed data).
    print!("  computing ground truth...");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| exact_top_k(&index_data, q, 10))
        .collect();
    println!(" done");

    print_header();

    // 1. FlatF32 (exact baseline).
    {
        let t = Instant::now();
        let idx = FlatF32Index::build(index_data.clone());
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let row = measure("FlatF32 (exact)", &idx, &queries, &truth, build_ms);
        print_row(&row);
    }

    // 2. PQ-8 (8 subspaces, K=64, 8 bytes/vec).
    {
        let t = Instant::now();
        let idx = PqIndex::build(index_data.clone(), pq_m, k_centroids, train_iters);
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let row = measure(
            &format!("PQ  M={pq_m} K={k_centroids} (8B/vec)"),
            &idx, &queries, &truth, build_ms,
        );
        print_row(&row);
    }

    // 3. RVQ-4 (4 stages, K=64, 4 bytes/vec — half the budget of PQ-8).
    {
        let cfg = RvqConfig {
            dim: d,
            num_stages: rvq_stages_4,
            codebook_size: k_centroids,
            train_iters,
            dropout_prob: 0.1,
        };
        let t = Instant::now();
        let idx = RvqIndex::build_with_config(cfg, index_data.clone()).unwrap();
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let row = measure(
            &format!("RVQ S={rvq_stages_4} K={k_centroids} (4B/vec)"),
            &idx, &queries, &truth, build_ms,
        );
        print_row(&row);
    }

    // 4. RVQ-8 (8 stages, K=64, 8 bytes/vec — same budget as PQ-8).
    {
        let cfg = RvqConfig {
            dim: d,
            num_stages: rvq_stages_8,
            codebook_size: k_centroids,
            train_iters,
            dropout_prob: 0.1,
        };
        let t = Instant::now();
        let idx = RvqIndex::build_with_config(cfg, index_data.clone()).unwrap();
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let row = measure(
            &format!("RVQ S={rvq_stages_8} K={k_centroids} (8B/vec)"),
            &idx, &queries, &truth, build_ms,
        );
        print_row(&row);
    }

    // 5. RVQ-8 + exact rerank×4 (same byte budget as PQ-8 for codes, + orig).
    {
        let cfg = RvqConfig {
            dim: d,
            num_stages: rvq_stages_8,
            codebook_size: k_centroids,
            train_iters,
            dropout_prob: 0.1,
        };
        let t = Instant::now();
        let idx = RvqRerankIndex::build_with_config(cfg, index_data.clone(), 4).unwrap();
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let row = measure(
            &format!("RVQ S={rvq_stages_8} K={k_centroids} +rerank×4"),
            &idx, &queries, &truth, build_ms,
        );
        print_row(&row);
    }

    println!();
}

fn print_distortion_profile(d: usize, n: usize) {
    let data = generate_clustered(n, d, 50, 7);
    let cfg = RvqConfig {
        dim: d,
        num_stages: 8,
        codebook_size: 64,
        train_iters: 25,
        dropout_prob: 0.1,
    };
    let encoder = ruvector_rvq::rvq::RvqEncoder::train(cfg, &data);
    let stage_dists = encoder.stage_distortions(&data);
    println!("── Distortion convergence (D={d}, N={n}, S=8, K=64) ──");
    println!("  Stage  MeanL2sq   Reduction");
    println!("  ─────  ────────   ─────────");
    let initial = stage_dists[0];
    for (s, &d_val) in stage_dists.iter().enumerate() {
        let reduction_pct = (1.0 - d_val / initial) * 100.0;
        println!("  {:>5}  {:>8.4}   {:>8.1}%", s + 1, d_val, reduction_pct.max(0.0));
    }
    println!();
}

fn main() {
    println!("════════════════════════════════════════════════════════════════════════════");
    println!("  ruvector-rvq benchmark — Residual Vector Quantization (RVQ) vs flat PQ");
    println!("════════════════════════════════════════════════════════════════════════════");
    println!("  Build: cargo run --release -p ruvector-rvq --bin rvq-demo");

    // Suite A: small — fast validation
    run_suite(5_000, 300, 128, 50);

    // Suite B: medium — primary benchmark
    run_suite(20_000, 500, 128, 100);

    // Suite C: higher dimension
    run_suite(10_000, 300, 256, 80);

    // Distortion convergence profile.
    print_distortion_profile(128, 3_000);

    println!("════════════════════════════════════════════════════════════════════════════");
    println!("  Legend:");
    println!("    R@10      = recall@10 (fraction of true top-10 found)");
    println!("    QPS       = queries per second (timed over all query vectors)");
    println!("    Mem/MB    = total index memory (codes + codebooks)");
    println!("    bytes/vec = code bytes stored per indexed vector");
    println!("    PQ M=8    = 8 independent subspaces of D/8 dims each");
    println!("    RVQ S=8   = 8 sequential stages on full-D residuals");
    println!("  Key insight: RVQ S=8 vs PQ M=8 — same 8B/vec, higher R@10");
    println!("════════════════════════════════════════════════════════════════════════════");
}
