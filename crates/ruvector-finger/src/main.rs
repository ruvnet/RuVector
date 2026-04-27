// FINGER benchmark binary.
// Measures QPS, recall@10, memory, and FINGER prune rate across three variants.
//
// Usage:  cargo run --release -p ruvector-finger
//         cargo run --release -p ruvector-finger -- --n 10000 --dim 256

use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_finger::{recall_at_k, FingerIndex, FlatGraph, GraphWalk};

struct Config {
    n_docs: usize,
    dim: usize,
    m: usize,   // neighbors per node
    k: usize,   // top-k to retrieve
    ef: usize,  // beam width
    n_queries: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config { n_docs: 5_000, dim: 128, m: 16, k: 10, ef: 200, n_queries: 200 }
    }
}

fn gen_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| normal.sample_iter(&mut rng).take(dim).collect()).collect()
}

struct BenchResult {
    name: &'static str,
    build_ms: u128,
    qps: f64,
    recall: f64,
    prune_rate: f64,
    mem_kb: usize,
}

fn run_benchmark(cfg: &Config, seed: u64) -> Vec<BenchResult> {
    println!("Generating {} vectors (D={}) …", cfg.n_docs, cfg.dim);
    let corpus = gen_dataset(cfg.n_docs, cfg.dim, seed);
    let queries = gen_dataset(cfg.n_queries, cfg.dim, seed + 1);

    println!("Building flat k-NN graph (M={}) …", cfg.m);
    let t0 = Instant::now();
    let graph = FlatGraph::build(&corpus, cfg.m).expect("graph build failed");
    let graph_build_ms = t0.elapsed().as_millis();
    println!("  graph built in {} ms", graph_build_ms);

    // Ground truth via brute-force kNN.
    let gt: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| graph.brute_force_knn(q, cfg.k))
        .collect();

    let mut results = Vec::new();

    // ── Exact beam search (no FINGER) ────────────────────────────────────────
    {
        let t0 = Instant::now();
        let idx = FingerIndex::exact(&graph);
        let build_ms = t0.elapsed().as_millis();

        // Warm-up
        for q in queries.iter().take(10) {
            let _ = idx.search(q, cfg.k, cfg.ef);
        }

        let t0 = Instant::now();
        let mut recall_sum = 0.0f64;
        let mut prune_sum = 0.0f64;
        for (qi, q) in queries.iter().enumerate() {
            let (res, stats) = idx.search(q, cfg.k, cfg.ef).unwrap();
            let ids: Vec<u32> = res.iter().map(|(id, _)| *id).collect();
            recall_sum += recall_at_k(&ids, &gt[qi], cfg.k);
            prune_sum += stats.prune_rate();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        results.push(BenchResult {
            name: "ExactBeam",
            build_ms,
            qps: cfg.n_queries as f64 / elapsed,
            recall: recall_sum / cfg.n_queries as f64,
            prune_rate: prune_sum / cfg.n_queries as f64,
            mem_kb: 0,
        });
    }

    // ── FINGER k=4 ───────────────────────────────────────────────────────────
    {
        let t0 = Instant::now();
        let idx = FingerIndex::finger_k4(&graph).expect("finger-k4 build failed");
        let build_ms = t0.elapsed().as_millis();
        let mem_kb = idx.bytes_used() / 1024;

        for q in queries.iter().take(10) {
            let _ = idx.search(q, cfg.k, cfg.ef);
        }

        let t0 = Instant::now();
        let mut recall_sum = 0.0f64;
        let mut prune_sum = 0.0f64;
        for (qi, q) in queries.iter().enumerate() {
            let (res, stats) = idx.search(q, cfg.k, cfg.ef).unwrap();
            let ids: Vec<u32> = res.iter().map(|(id, _)| *id).collect();
            recall_sum += recall_at_k(&ids, &gt[qi], cfg.k);
            prune_sum += stats.prune_rate();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        results.push(BenchResult {
            name: "FINGER-K4",
            build_ms,
            qps: cfg.n_queries as f64 / elapsed,
            recall: recall_sum / cfg.n_queries as f64,
            prune_rate: prune_sum / cfg.n_queries as f64,
            mem_kb,
        });
    }

    // ── FINGER k=8 ───────────────────────────────────────────────────────────
    {
        let t0 = Instant::now();
        let idx = FingerIndex::finger_k8(&graph).expect("finger-k8 build failed");
        let build_ms = t0.elapsed().as_millis();
        let mem_kb = idx.bytes_used() / 1024;

        for q in queries.iter().take(10) {
            let _ = idx.search(q, cfg.k, cfg.ef);
        }

        let t0 = Instant::now();
        let mut recall_sum = 0.0f64;
        let mut prune_sum = 0.0f64;
        for (qi, q) in queries.iter().enumerate() {
            let (res, stats) = idx.search(q, cfg.k, cfg.ef).unwrap();
            let ids: Vec<u32> = res.iter().map(|(id, _)| *id).collect();
            recall_sum += recall_at_k(&ids, &gt[qi], cfg.k);
            prune_sum += stats.prune_rate();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        results.push(BenchResult {
            name: "FINGER-K8",
            build_ms,
            qps: cfg.n_queries as f64 / elapsed,
            recall: recall_sum / cfg.n_queries as f64,
            prune_rate: prune_sum / cfg.n_queries as f64,
            mem_kb,
        });
    }

    results
}

fn print_table(cfg: &Config, results: &[BenchResult]) {
    println!(
        "\nN={}, D={}, M={}, k={}, ef={}, queries={}",
        cfg.n_docs, cfg.dim, cfg.m, cfg.k, cfg.ef, cfg.n_queries
    );
    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "Variant", "Build(ms)", "QPS", "Recall@10", "PruneRate%", "Basis(KB)"
    );
    println!("{:-<72}", "");
    for r in results {
        println!(
            "{:<14} {:>10} {:>10.0} {:>9.1}% {:>11.1}% {:>10}",
            r.name,
            r.build_ms,
            r.qps,
            r.recall * 100.0,
            r.prune_rate * 100.0,
            r.mem_kb,
        );
    }
    // Print speedup relative to exact
    if let Some(exact) = results.iter().find(|r| r.name == "ExactBeam") {
        println!();
        for r in results.iter().filter(|r| r.name != "ExactBeam") {
            let speedup = r.qps / exact.qps;
            println!(
                "  {} vs ExactBeam: {:.2}× QPS, recall={:.1}%, prune={:.1}%",
                r.name,
                speedup,
                r.recall * 100.0,
                r.prune_rate * 100.0,
            );
        }
    }
    println!();
}

fn main() {
    println!("ruvector-finger: FINGER approximate distance skipping benchmark");
    println!("arXiv:2206.11408 — Chen et al., WWW 2023\n");

    // Parse minimal CLI args for N and dim overrides.
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => { cfg.n_docs = args[i + 1].parse().unwrap_or(cfg.n_docs); i += 2; }
            "--dim" => { cfg.dim = args[i + 1].parse().unwrap_or(cfg.dim); i += 2; }
            "--m" => { cfg.m = args[i + 1].parse().unwrap_or(cfg.m); i += 2; }
            "--ef" => { cfg.ef = args[i + 1].parse().unwrap_or(cfg.ef); i += 2; }
            _ => { i += 1; }
        }
    }

    let results = run_benchmark(&cfg, 42);
    print_table(&cfg, &results);

    // Second run at smaller scale to show scaling behaviour.
    if cfg.n_docs == Config::default().n_docs {
        println!("--- Scaling: N=1000 ---");
        let cfg2 = Config { n_docs: 1_000, ..Config::default() };
        let r2 = run_benchmark(&cfg2, 99);
        print_table(&cfg2, &r2);

        println!("--- Scaling: N=10000 ---");
        let cfg3 = Config { n_docs: 10_000, n_queries: 100, ..Config::default() };
        let r3 = run_benchmark(&cfg3, 17);
        print_table(&cfg3, &r3);
    }
}
