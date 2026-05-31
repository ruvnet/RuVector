//! SymphonyQG unified benchmark harness.
//!
//! Measures recall@10, QPS, and memory for three index variants on
//! Gaussian-clustered datasets at multiple scales.
//!
//! Usage:
//!   cargo run --release -p ruvector-symphony-qg -- [--fast]
//!
//! --fast: smoke mode (n ≤ 1K, ~3 s)
//! default: full mode (n ∈ {1K, 2K, 5K}, ~30 s)

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::HashSet;
use std::time::Instant;

use ruvector_symphony_qg::{
    AnnIndex, FlatF32Index, GraphConfig, GraphExact, SymphonyIndex,
};

struct BenchResult {
    name: &'static str,
    n: usize,
    r: usize,
    ef: usize,
    build_ms: f64,
    recall_at_10: f64,
    qps: f64,
    mem_bytes: usize,
}

fn generate_clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let cr = Uniform::new(-2.0f32, 2.0);
    let centroids: Vec<Vec<f32>> =
        (0..n_clusters).map(|_| (0..d).map(|_| cr.sample(&mut rng)).collect()).collect();
    let noise = Normal::new(0.0f64, 0.4).unwrap();
    (0..n)
        .map(|_| {
            use rand::Rng as _;
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter().map(|&x| x + noise.sample(&mut rng) as f32).collect()
        })
        .collect()
}

fn recall_at_k(
    truth: &[Vec<usize>],
    found: &[Vec<usize>],
    k: usize,
) -> f64 {
    let n = truth.len().min(found.len());
    if n == 0 { return 0.0; }
    let sum: f64 = truth.iter().zip(found.iter()).map(|(t, f)| {
        let t_set: HashSet<usize> = t.iter().copied().collect();
        let hits = f.iter().take(k).filter(|id| t_set.contains(id)).count();
        hits as f64 / k.min(t.len()) as f64
    }).sum();
    sum / n as f64
}

fn bench_flat(vectors: &[Vec<f32>], queries: &[Vec<f32>], truth: &[Vec<usize>]) -> BenchResult {
    let n = vectors.len();
    let t0 = Instant::now();
    let idx = FlatF32Index::build(vectors.to_vec()).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mem = idx.memory_bytes();
    let n_q = queries.len();

    let t0 = Instant::now();
    let found: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| idx.search(q, 10).into_iter().map(|r| r.id).collect())
        .collect();
    let elapsed = t0.elapsed().as_secs_f64();
    let qps = n_q as f64 / elapsed;
    let recall = recall_at_k(truth, &found, 10);

    BenchResult {
        name: "FlatF32",
        n,
        r: 0,
        ef: 0,
        build_ms,
        recall_at_10: recall,
        qps,
        mem_bytes: mem,
    }
}

fn bench_graph_exact(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    r: usize,
    ef: usize,
) -> BenchResult {
    let n = vectors.len();
    let dim = vectors[0].len();
    let cfg = GraphConfig::new(dim).with_r(r).with_ef(ef);

    let t0 = Instant::now();
    let idx = GraphExact::build(vectors.to_vec(), cfg).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mem = idx.memory_bytes();

    let n_q = queries.len();
    let t0 = Instant::now();
    let found: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| idx.search(q, 10).into_iter().map(|r| r.id).collect())
        .collect();
    let elapsed = t0.elapsed().as_secs_f64();
    let qps = n_q as f64 / elapsed;
    let recall = recall_at_k(truth, &found, 10);

    BenchResult {
        name: "GraphExact",
        n,
        r,
        ef,
        build_ms,
        recall_at_10: recall,
        qps,
        mem_bytes: mem,
    }
}

fn bench_symphony(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    r: usize,
    ef: usize,
) -> BenchResult {
    let n = vectors.len();
    let dim = vectors[0].len();
    let cfg = GraphConfig::new(dim).with_r(r).with_ef(ef);

    let t0 = Instant::now();
    let idx = SymphonyIndex::build(vectors.to_vec(), cfg).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mem = idx.memory_bytes();

    let n_q = queries.len();
    let t0 = Instant::now();
    let found: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| idx.search(q, 10).into_iter().map(|r| r.id).collect())
        .collect();
    let elapsed = t0.elapsed().as_secs_f64();
    let qps = n_q as f64 / elapsed;
    let recall = recall_at_k(truth, &found, 10);

    BenchResult {
        name: "SymphonyQG",
        n,
        r,
        ef,
        build_ms,
        recall_at_10: recall,
        qps,
        mem_bytes: mem,
    }
}

fn print_table(rows: &[BenchResult]) {
    println!(
        "\n{:<14} {:>6} {:>4} {:>4} {:>10} {:>10} {:>10} {:>10}",
        "Index", "n", "R", "ef", "Build(ms)", "Recall@10", "QPS", "Memory"
    );
    println!("{}", "-".repeat(80));
    for r in rows {
        println!(
            "{:<14} {:>6} {:>4} {:>4} {:>10.1} {:>10.3} {:>10.0} {:>10}",
            r.name,
            r.n,
            r.r,
            r.ef,
            r.build_ms,
            r.recall_at_10,
            r.qps,
            human_bytes(r.mem_bytes),
        );
    }
    println!();
}

fn human_bytes(b: usize) -> String {
    if b < 1024 { format!("{b} B") }
    else if b < 1024 * 1024 { format!("{:.1} KB", b as f64 / 1024.0) }
    else { format!("{:.2} MB", b as f64 / (1024.0 * 1024.0)) }
}

fn run_suite(n: usize, dim: usize, n_clusters: usize, n_queries: usize, fast: bool) {
    println!("=== n={n}, D={dim}, clusters={n_clusters}, queries={n_queries} ===");

    let corpus = generate_clustered(n, dim, n_clusters, 42);
    let queries = generate_clustered(n_queries, dim, n_clusters, 99);

    // Compute ground truth using brute force
    let flat_ref = FlatF32Index::build(corpus.clone()).unwrap();
    let truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| flat_ref.search(q, 10).into_iter().map(|r| r.id).collect())
        .collect();

    let mut rows = Vec::new();

    // 1. FlatF32 baseline
    rows.push(bench_flat(&corpus, &queries, &truth));

    // 2-4. Graph variants at different ef
    let params: &[(usize, usize)] = if fast {
        &[(16, 32)]
    } else {
        &[(16, 32), (16, 64), (32, 64)]
    };

    for &(r, ef) in params {
        rows.push(bench_graph_exact(&corpus, &queries, &truth, r, ef));
        rows.push(bench_symphony(&corpus, &queries, &truth, r, ef));
    }

    print_table(&rows);

    // Print speedup analysis
    if let (Some(flat), Some(sym)) = (
        rows.iter().find(|r| r.name == "FlatF32"),
        rows.iter().filter(|r| r.name == "SymphonyQG").last(),
    ) {
        let qps_speedup = sym.qps / flat.qps;
        let recall_delta = sym.recall_at_10 - flat.recall_at_10;
        println!(
            "  SymphonyQG (R={}, ef={}) vs FlatF32: {:.2}× QPS, recall delta {:+.3}",
            sym.r, sym.ef, qps_speedup, recall_delta
        );
        if let Some(gex) = rows.iter().filter(|r| r.name == "GraphExact" && r.r == sym.r && r.ef == sym.ef).next() {
            let vs_exact = sym.qps / gex.qps;
            println!(
                "  SymphonyQG vs GraphExact (same R/ef): {:.2}× QPS, recall delta {:+.3}",
                vs_exact, sym.recall_at_10 - gex.recall_at_10
            );
        }
    }
    println!();
}

fn main() {
    let fast = std::env::args().any(|a| a == "--fast");

    println!("SymphonyQG Benchmark Harness");
    println!("  arXiv:2411.12229 · SIGMOD 2025");
    println!("  Co-located RaBitQ codes + batch asymmetric distance on k-NN graph");
    println!();

    if fast {
        println!("[fast mode: n≤1K]");
        run_suite(1_000, 128, 50, 200, true);
    } else {
        run_suite(1_000, 128, 50, 200, false);
        run_suite(2_000, 128, 80, 300, false);
        run_suite(5_000, 128, 100, 500, false);
    }

    println!("Done.");
}
