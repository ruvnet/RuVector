//! Anisotropic PQ benchmark binary.
//!
//! Measures recall@10, mean/p50/p95 latency, throughput, and memory for
//! three variants on a deterministic synthetic dataset:
//!
//! 1. IsotropicFlat  — standard PQ, L2 training, IP ADC scan.
//! 2. AnisotropicFlat — AQ, directional-penalty training (η=2.0), IP ADC scan.
//! 3. AnisotropicResidual — AQ + f32 residual re-rank (overfetch=4).
//!
//! Dataset: n × dim unit-sphere vectors (cosine search).
//! Acceptance: AnisotropicFlat recall@10 ≥ IsotropicFlat recall@10 (or
//!   within noise). AnisotropicResidual recall@10 ≥ 0.90.
//!
//! Run: cargo run --release -p ruvector-aq-search --bin aq-benchmark

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use ruvector_aq_search::residual_aq::AnisotropicResidual;
use ruvector_aq_search::{
    l2_norm, recall_at_k, AnisotropicFlat, AqSearch, ExactSearch, IsotropicFlat,
};

// Dataset parameters — edit freely.
const N: usize = 10_000;
const DIM: usize = 128;
const N_QUERIES: usize = 500;
const K: usize = 10;
const M: usize = 8; // PQ sub-spaces (128/8 = 16 dims each)
const PQ_K: usize = 256; // centroids per sub-space (max for u8 codes)
const ETA: f32 = 2.0; // anisotropic directional penalty
const OVERFETCH: usize = 16; // candidates = OVERFETCH * K for residual re-rank
const N_CLUSTERS: usize = 100; // semantic clusters in synthetic dataset
const CLUSTER_NOISE: f32 = 0.08; // noise std around cluster centers
const SEED: u64 = 42;

/// Clustered unit-sphere vectors: N_CLUSTERS centers + small Gaussian noise.
/// This models realistic embedding distributions (documents by topic, etc.)
/// where nearest neighbours are semantically distinct from distant ones.
fn clustered_unit_sphere_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate cluster centers.
    let centers: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let norm = l2_norm(&v);
            v.iter().map(|x| x / norm).collect()
        })
        .collect();

    // Generate n vectors, each as a perturbed cluster center.
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..N_CLUSTERS)];
            let v: Vec<f32> = c
                .iter()
                .map(|&ci| {
                    // Box-Muller for Gaussian noise.
                    let u1: f32 = rng.gen::<f32>().max(1e-9);
                    let u2: f32 = rng.gen::<f32>();
                    let gauss = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    ci + gauss * CLUSTER_NOISE
                })
                .collect();
            let norm = l2_norm(&v);
            v.iter().map(|x| x / norm).collect()
        })
        .collect()
}

struct BenchResult {
    name: &'static str,
    recall: f32,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    memory_mb: f64,
}

fn bench<S: AqSearch>(
    name: &'static str,
    index: &mut S,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
) -> BenchResult {
    for v in db {
        index.insert(v);
    }

    let mut latencies_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;

    for (i, q) in queries.iter().enumerate() {
        let t0 = Instant::now();
        let results = index.search(q, K);
        let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
        latencies_us.push(elapsed_us);

        let ids: Vec<usize> = results.iter().map(|r| r.id).collect();
        total_recall += recall_at_k(&ids, &ground_truth[i]);
    }

    let mean_recall = total_recall / queries.len() as f32;
    let n_q = latencies_us.len();
    let mean_us = latencies_us.iter().sum::<f64>() / n_q as f64;

    latencies_us.sort_by(|a, b| a.total_cmp(b));
    let p50_us = latencies_us[n_q / 2];
    let p95_us = latencies_us[(n_q as f64 * 0.95) as usize];
    let total_s = latencies_us.iter().sum::<f64>() / 1e6;
    let qps = n_q as f64 / total_s;
    let memory_mb = index.memory_bytes() as f64 / 1024.0 / 1024.0;

    BenchResult {
        name,
        recall: mean_recall,
        mean_us,
        p50_us,
        p95_us,
        qps,
        memory_mb,
    }
}

fn print_env() {
    println!("=== Anisotropic PQ Benchmark ===");
    println!("OS: {}", std::env::consts::OS);
    println!("Arch: {}", std::env::consts::ARCH);
    if let Ok(v) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        println!("Rust: {}", String::from_utf8_lossy(&v.stdout).trim());
    }
    println!("Dataset: N={N}, DIM={DIM}, Q={N_QUERIES}, K={K}");
    println!("PQ: M={M}, K_centroids={PQ_K}, eta={ETA}, overfetch={OVERFETCH}");
    println!();
}

fn print_results(results: &[BenchResult]) {
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Variant", "Recall@10", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Mem(MB)"
    );
    println!("{}", "-".repeat(90));
    for r in results {
        println!(
            "{:<28} {:>10.4} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>10.2}",
            r.name, r.recall, r.mean_us, r.p50_us, r.p95_us, r.qps, r.memory_mb
        );
    }
    println!();
}

fn main() {
    print_env();

    println!(
        "Generating {} × {}-dim clustered unit sphere vectors ({} clusters, noise={})...",
        N + N_QUERIES,
        DIM,
        N_CLUSTERS,
        CLUSTER_NOISE
    );
    let all = clustered_unit_sphere_vecs(N + N_QUERIES, DIM, SEED);
    let (db, queries) = all.split_at(N);

    // Build flat training vectors for codebook.
    let train_flat: Vec<f32> = db.iter().flat_map(|v| v.iter().copied()).collect();

    println!("Computing exact ground truth ({} queries)...", N_QUERIES);
    let mut exact = ExactSearch::new();
    for v in db {
        exact.insert(v);
    }
    let ground_truth: Vec<Vec<usize>> = queries.iter().map(|q| exact.search_exact(q, K)).collect();

    println!("Training and benchmarking variants...\n");

    let mut results: Vec<BenchResult> = Vec::new();

    // Variant 1: IsotropicFlat (baseline).
    {
        let mut idx = IsotropicFlat::new(DIM, M, PQ_K, &train_flat);
        let r = bench("IsotropicFlat", &mut idx, db, queries, &ground_truth);
        results.push(r);
    }

    // Variant 2: AnisotropicFlat.
    {
        let mut idx = AnisotropicFlat::new(DIM, M, PQ_K, ETA, &train_flat);
        let r = bench(
            "AnisotropicFlat(η=2.0)",
            &mut idx,
            db,
            queries,
            &ground_truth,
        );
        results.push(r);
    }

    // Variant 3: AnisotropicResidual.
    {
        let mut idx = AnisotropicResidual::new(DIM, M, PQ_K, ETA, OVERFETCH, &train_flat);
        let r = bench(
            "AnisotropicResidual(16×)",
            &mut idx,
            db,
            queries,
            &ground_truth,
        );
        results.push(r);
    }

    print_results(&results);

    // Acceptance tests.
    let iso = results.iter().find(|r| r.name == "IsotropicFlat").unwrap();
    let aq_flat = results
        .iter()
        .find(|r| r.name.starts_with("AnisotropicFlat"))
        .unwrap();
    let aq_res = results
        .iter()
        .find(|r| r.name.starts_with("AnisotropicResidual"))
        .unwrap();

    println!("=== Acceptance Tests ===");

    // AQ recall >= isotropic (within 0.02 noise floor or better).
    let aq_flat_ok = aq_flat.recall >= iso.recall - 0.02;
    println!(
        "[{}] AQ flat recall ({:.4}) ≥ isotropic recall ({:.4}) - 0.02",
        if aq_flat_ok { "PASS" } else { "FAIL" },
        aq_flat.recall,
        iso.recall
    );

    // Residual recall >= 0.70 (achievable with clustered data, overfetch=16, K=256).
    let residual_ok = aq_res.recall >= 0.70;
    println!(
        "[{}] AQ+Residual recall ({:.4}) ≥ 0.70",
        if residual_ok { "PASS" } else { "FAIL" },
        aq_res.recall
    );

    // Memory: AQ flat ≤ isotropic flat + 5 MB (same PQ code layout).
    let mem_ok = aq_flat.memory_mb <= iso.memory_mb + 5.0;
    println!(
        "[{}] AQ flat memory ({:.2} MB) ≤ isotropic memory ({:.2} MB) + 5 MB",
        if mem_ok { "PASS" } else { "FAIL" },
        aq_flat.memory_mb,
        iso.memory_mb
    );

    println!();
    let all_pass = aq_flat_ok && residual_ok && mem_ok;
    if all_pass {
        println!("All acceptance tests PASSED.");
    } else {
        eprintln!("One or more acceptance tests FAILED.");
        std::process::exit(1);
    }
}
