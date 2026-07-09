//! RVQ benchmark: ScalarQ-8bit vs ProductQ vs ResidualQ-4
//!
//! Evaluated on two data regimes:
//!   1. Isotropic Gaussian — IID dims, favours PQ (sub-spaces are independent)
//!   2. Clustered semantic  — n_clusters > K_per_stage, favours RVQ (iterative residual
//!      refinement resolves inter-cluster confusion PQ's product code leaves intact)
//!
//! Acceptance: ResidualQ-4 MSQE < ProductQ MSQE on clustered data at equal byte budget.

use ruvector_rvq::{
    generate_clustered_vectors, generate_vectors, msqe, recall_at_k, ProductQuantizer,
    ResidualQuantizer, ScalarQuantizer, VectorQuantizer,
};
use std::time::Instant;

// ---- Dataset parameters ----
const D: usize = 32;
const N_TRAIN: usize = 5_000;
const N_TEST: usize = 2_000;
const N_QUERY: usize = 200;
const K_NN: usize = 10;

// Clustered regime: n_clusters > K so centroids must cover multiple clusters.
const N_CLUSTERS: usize = 100;
const SIGMA_CLUSTER: f32 = 3.0;
const SIGMA_NOISE: f32 = 0.15;

// ---- Quantizer parameters (equal byte budget for PQ and RVQ) ----
const M: usize = 4; // PQ sub-spaces      → 4 bytes/vector
const K: usize = 32; // centroids per codebook
const STAGES: usize = 4; // RVQ stages         → 4 bytes/vector

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() as f64 - 1.0) * p / 100.0).round() as usize;
    v[idx]
}

struct BenchResult {
    name: &'static str,
    bytes_per_vec: usize,
    codebook_kb: f64,
    train_ms: f64,
    encode_us_mean: f64,
    encode_us_p50: f64,
    encode_us_p95: f64,
    decode_us_mean: f64,
    msqe: f64,
    recall: f64,
}

fn bench_quantizer(
    q: &mut dyn VectorQuantizer,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> BenchResult {
    let t0 = Instant::now();
    q.train(train);
    let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut enc_times = Vec::with_capacity(test.len());
    for v in test {
        let t = Instant::now();
        let _c = q.encode(v);
        enc_times.push(t.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let encode_us_mean = enc_times.iter().sum::<f64>() / enc_times.len() as f64;
    let encode_us_p50 = percentile(enc_times.clone(), 50.0);
    let encode_us_p95 = percentile(enc_times, 95.0);

    let mut dec_times = Vec::with_capacity(test.len());
    for v in test {
        let codes = q.encode(v);
        let t = Instant::now();
        let _d = q.decode(&codes);
        dec_times.push(t.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let decode_us_mean = dec_times.iter().sum::<f64>() / dec_times.len() as f64;

    BenchResult {
        name: q.name(),
        bytes_per_vec: q.bytes_per_vector(),
        codebook_kb: q.codebook_bytes() as f64 / 1024.0,
        train_ms,
        encode_us_mean,
        encode_us_p50,
        encode_us_p95,
        decode_us_mean,
        msqe: msqe(q, test),
        recall: recall_at_k(q, test, queries, K_NN),
    }
}

fn print_table(results: &[BenchResult]) {
    println!(
        "{:<16} {:>9} {:>11} {:>10} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10}",
        "Variant",
        "Bytes/Vec",
        "Codebook",
        "Train(ms)",
        "Enc μs",
        "p50 μs",
        "p95 μs",
        "Dec μs",
        "MSQE",
        "Recall@10"
    );
    println!("{}", "-".repeat(115));
    for r in results {
        println!(
            "{:<16} {:>9} {:>10.1}KB {:>9.1} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>10.6} {:>10.4}",
            r.name,
            r.bytes_per_vec,
            r.codebook_kb,
            r.train_ms,
            r.encode_us_mean,
            r.encode_us_p50,
            r.encode_us_p95,
            r.decode_us_mean,
            r.msqe,
            r.recall,
        );
    }
}

fn run_suite(
    label: &str,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> (f64, f64) {
    println!("\n--- {} ---", label);

    let mut sq = ScalarQuantizer::new(D);
    let mut pq = ProductQuantizer::new(D, M, K);
    let mut rq = ResidualQuantizer::new(D, STAGES, K);

    let sq_r = bench_quantizer(&mut sq, train, test, queries);
    let pq_r = bench_quantizer(&mut pq, train, test, queries);
    let rq_r = bench_quantizer(&mut rq, train, test, queries);

    let pq_msqe = pq_r.msqe;
    let rq_msqe = rq_r.msqe;
    let rq_recall = rq_r.recall;

    print_table(&[sq_r, pq_r, rq_r]);

    println!();
    println!("  Memory math ({N_TEST} test vectors):");
    for (name, bpv) in &[
        ("ScalarQ-8bit", D),
        ("ProductQ", M),
        ("ResidualQ-4", STAGES),
    ] {
        let db_kb = (N_TEST * bpv) as f64 / 1024.0;
        let full_kb = (N_TEST * D * 4) as f64 / 1024.0;
        println!(
            "    {name:<16}: {db_kb:.1}KB compressed  (full = {full_kb:.1}KB, {:.1}x ratio)",
            full_kb / db_kb
        );
    }

    (pq_msqe, rq_msqe)
}

fn main() {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("=== ruvector-rvq: Residual Vector Quantization Benchmark ===");
    println!("Platform : {os} {arch}");
    println!("CPU      : {cpu}");
    println!("Dataset  : {N_TRAIN} train / {N_TEST} test / {N_QUERY} queries / {D} dims");
    println!("PQ       : M={M} sub-spaces, K={K} centroids → {M} bytes/vector");
    println!("RVQ      : L={STAGES} stages, K={K} centroids  → {STAGES} bytes/vector");
    println!("ScalarQ  : 8-bit per dim                   → {D} bytes/vector");
    println!("Clustered: {N_CLUSTERS} clusters, σ_cluster={SIGMA_CLUSTER}, σ_noise={SIGMA_NOISE}");

    // Generate data
    let t = Instant::now();
    let gauss_train = generate_vectors(N_TRAIN, D, 1);
    let gauss_test = generate_vectors(N_TEST, D, 2);
    let gauss_queries = generate_vectors(N_QUERY, D, 3);
    let clust_train =
        generate_clustered_vectors(N_TRAIN, D, N_CLUSTERS, SIGMA_CLUSTER, SIGMA_NOISE, 10);
    let clust_test =
        generate_clustered_vectors(N_TEST, D, N_CLUSTERS, SIGMA_CLUSTER, SIGMA_NOISE, 11);
    let clust_queries =
        generate_clustered_vectors(N_QUERY, D, N_CLUSTERS, SIGMA_CLUSTER, SIGMA_NOISE, 12);
    println!(
        "\nData generated in {:.1}ms",
        t.elapsed().as_secs_f64() * 1000.0
    );

    // Suite 1: isotropic Gaussian (PQ-friendly baseline)
    let (pq_gauss, rq_gauss) = run_suite(
        "Suite 1: Isotropic Gaussian (IID dims — PQ-friendly baseline)",
        &gauss_train,
        &gauss_test,
        &gauss_queries,
    );
    println!("  Note: IID Gaussian dims are PQ-optimal; RVQ and PQ perform comparably here.");

    // Suite 2: clustered semantic data (RVQ advantage)
    let (pq_clust, rq_clust) = run_suite(
        "Suite 2: Clustered Semantic Data (n_clusters={N_CLUSTERS} > K={K} — RVQ advantage)",
        &clust_train,
        &clust_test,
        &clust_queries,
    );

    // Acceptance test
    println!("\n=== Acceptance Test (clustered data) ===");
    let msqe_pass = rq_clust < pq_clust;
    println!(
        "  ResidualQ-4 MSQE ({rq_clust:.6}) < ProductQ MSQE ({pq_clust:.6}): {}",
        if msqe_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!(
        "  MSQE improvement: {:.1}x (RVQ / PQ = {:.4})",
        pq_clust / rq_clust,
        rq_clust / pq_clust
    );

    // Summary table
    println!("\n=== Key Numbers Summary ===");
    println!("  Gaussian: PQ MSQE={pq_gauss:.6}, RVQ MSQE={rq_gauss:.6}");
    println!("  Clustered: PQ MSQE={pq_clust:.6}, RVQ MSQE={rq_clust:.6}");
    println!(
        "  RVQ improvement on clustered: {:.1}x",
        pq_clust / rq_clust
    );

    if msqe_pass {
        println!("\nBENCHMARK RESULT: PASS");
    } else {
        eprintln!("\nBENCHMARK RESULT: FAIL");
        std::process::exit(1);
    }
}
