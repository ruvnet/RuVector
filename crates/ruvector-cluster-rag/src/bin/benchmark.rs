//! Benchmark binary for ruvector-cluster-rag.
//!
//! Runs three search variants over a deterministic synthetic dataset and prints
//! latency, throughput, memory, and recall statistics. All numbers are real.
//!
//! Usage:
//!   cargo run --release -p ruvector-cluster-rag --bin benchmark
//!
//! Optional env overrides:
//!   N=20000 DIM=128 NQ=1000 K=10 K_CLUSTERS=64 NPROBE=8 LAMBDA=0.7

use ruvector_cluster_rag::{
    bench::{format_bytes, run_bench, BenchResult},
    cluster::kmeans,
    dataset::{generate_queries, generate_vectors},
    search::{ClusterSearch, CoherenceTree, FlatBrute},
    tree::ClusterTree,
    AnnVariant, Hit,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn print_row(r: &BenchResult) {
    println!(
        "{:<18} {:>8.1} {:>8.1} {:>8.1} {:>10.0} {:>12} {:>9.3}",
        r.variant,
        r.mean_us,
        r.p50_us,
        r.p95_us,
        r.qps,
        format_bytes(r.mem_bytes),
        r.mean_recall,
    );
}

fn main() {
    let n = env_usize("N", 10_000);
    let dim = env_usize("DIM", 128);
    let nq = env_usize("NQ", 500);
    let k = env_usize("K", 10);
    let k_clusters = env_usize("K_CLUSTERS", 40);
    let nprobe = env_usize("NPROBE", 20);
    let lambda = env_f32("LAMBDA", 0.70);

    // ── system info ──────────────────────────────────────────────────────────
    println!("=== ruvector-cluster-rag benchmark ===");
    println!("OS      : {}", std::env::consts::OS);
    println!("Arch    : {}", std::env::consts::ARCH);
    println!();
    println!("Config");
    println!("  N           = {n}  (corpus vectors)");
    println!("  DIM         = {dim}  (dimensions)");
    println!("  NQ          = {nq}  (query vectors)");
    println!("  K           = {k}  (top-k)");
    println!("  K_CLUSTERS  = {k_clusters}");
    println!(
        "  NPROBE      = {nprobe}  ({:.0}% of clusters searched)",
        nprobe as f64 / k_clusters as f64 * 100.0
    );
    println!("  LAMBDA      = {lambda:.2}  (CoherenceTree query-sim weight)");
    println!();

    // ── dataset ──────────────────────────────────────────────────────────────
    let seed: u64 = 20260807;
    let corpus = generate_vectors(n, dim, seed);
    let queries = generate_queries(nq, dim, seed);
    println!("Dataset: {n} × {dim} f32 vectors | {nq} queries");
    println!("Raw corpus memory: {}", format_bytes(n * dim * 4));
    println!();

    // ── k-means ──────────────────────────────────────────────────────────────
    print!("k-means (k={k_clusters}, 20 iters) ... ");
    let t0 = std::time::Instant::now();
    let km_cs = kmeans(&corpus, k_clusters, 20);
    let km_ct = kmeans(&corpus, k_clusters, 20);
    println!("done in {:.2}s", t0.elapsed().as_secs_f64());

    // ── build indexes ─────────────────────────────────────────────────────────
    let flat = FlatBrute::new(corpus.clone());
    let cs = ClusterSearch::new(ClusterTree::new(corpus.clone(), km_cs), nprobe);
    let ct = CoherenceTree::new(ClusterTree::new(corpus, km_ct), nprobe, lambda);

    // ── ground truth ─────────────────────────────────────────────────────────
    let ground_truth: Vec<Vec<Hit>> = queries.iter().map(|q| flat.search(q, k)).collect();

    // ── benchmark each variant ────────────────────────────────────────────────
    struct Named {
        name: &'static str,
        idx: Box<dyn AnnVariant>,
    }
    let variants: Vec<Named> = vec![
        Named {
            name: "FlatBrute",
            idx: Box::new(FlatBrute::new(generate_vectors(n, dim, seed))),
        },
        Named {
            name: "ClusterSearch",
            idx: Box::new(cs),
        },
        Named {
            name: "CoherenceTree",
            idx: Box::new(ct),
        },
    ];

    let mut results: Vec<BenchResult> = Vec::new();
    for v in &variants {
        print!("  Benchmarking {} ...", v.name);
        let r = run_bench(v.idx.as_ref(), &queries, k, &ground_truth, n, dim);
        println!(" {:.1} µs/query", r.mean_us);
        results.push(r);
    }

    // ── results table ─────────────────────────────────────────────────────────
    println!();
    println!(
        "Results  (n={n}, dim={dim}, nq={nq}, k={k}, k_clusters={k_clusters}, nprobe={nprobe})"
    );
    println!();
    println!(
        "{:<18} {:>8} {:>8} {:>8} {:>10} {:>12} {:>9}",
        "Variant", "Mean µs", "p50 µs", "p95 µs", "QPS", "Memory", "Recall@K"
    );
    println!("{}", "─".repeat(82));
    for r in &results {
        print_row(r);
    }
    println!();

    // ── memory breakdown ──────────────────────────────────────────────────────
    let leaf_bytes = n * dim * 4;
    let centroid_bytes = k_clusters * dim * 4;
    let inv_bytes = n * 8;
    println!("Memory breakdown:");
    println!("  Leaf vectors       : {}", format_bytes(leaf_bytes));
    println!("  Centroids (level-1): {}", format_bytes(centroid_bytes));
    println!("  Inverted lists     : {}", format_bytes(inv_bytes));
    println!(
        "  Overhead           : {:.1}%",
        (centroid_bytes + inv_bytes) as f64 / leaf_bytes as f64 * 100.0
    );
    println!();

    // ── acceptance gate ───────────────────────────────────────────────────────
    // FlatBrute must achieve recall = 1.0 (it is ground truth).
    // ClusterSearch and CoherenceTree must achieve ≥ 0.70 recall@10
    // with nprobe/k_clusters = 20% of the corpus searched.
    let min_recall_cluster = 0.70;

    let flat_r = results.iter().find(|r| r.variant == "FlatBrute").unwrap();
    assert!(
        flat_r.mean_recall >= 0.999,
        "FlatBrute recall {:.4} must equal 1.0",
        flat_r.mean_recall
    );

    let mut all_pass = true;
    for r in results.iter().filter(|r| r.variant != "FlatBrute") {
        if r.mean_recall >= min_recall_cluster {
            println!(
                "ACCEPTANCE PASS: {} recall {:.3} ≥ {min_recall_cluster:.2}",
                r.variant, r.mean_recall
            );
        } else {
            println!(
                "ACCEPTANCE FAIL: {} recall {:.3} < {min_recall_cluster:.2}",
                r.variant, r.mean_recall
            );
            all_pass = false;
        }
    }
    println!();
    if all_pass {
        println!("All acceptance criteria met. Benchmark complete.");
    } else {
        eprintln!("One or more variants failed the acceptance gate.");
        std::process::exit(1);
    }
}
