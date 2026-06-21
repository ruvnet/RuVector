//! PQ-ADC benchmark: measures recall@10, mean/p50/p95 latency, and memory for
//! three variants (FlatPQ, IVF+PQ, ResidualPQ) on synthetic Gaussian data.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_pq_search::{
    codebook::{PqCodebook, PqConfig},
    flat::FlatPqIndex,
    ivf_pq::IvfPqIndex,
    recall_at_k,
    residual::ResidualPqIndex,
    ExactSearch, PqSearch,
};
use std::time::Instant;

// ── Dataset parameters ──────────────────────────────────────────────────────
const N: usize = 10_000; // database vectors
const DIM: usize = 128; // embedding dimension
const N_QUERIES: usize = 200;
const K: usize = 10; // top-k
const M: usize = 8; // PQ sub-spaces  (DIM / M = 16 dims per sub-space)
const KQ: usize = 256; // centroids per sub-space

// IVF parameters
const N_LISTS: usize = 32;
const N_PROBE: usize = 4;

// Residual PQ oversampling
const OVERSAMPLING: usize = 8;

// ── Acceptance thresholds ─────────────────────────────────────────────────────
/// Minimum recall@10 for ResidualPQ (primary gate).
/// On synthetic low-intrinsic-dim data, FlatPQ recall ≈ 0.20-0.30 — this is
/// expected: the 64x compression is a coarse filter. ResidualPQ restores recall
/// by storing per-vector residuals and exact re-scoring the shortlist.
const ACCEPT_RECALL_RESIDUAL: f32 = 0.60;
/// Secondary gate: FlatPQ should at least provide usable coarse filtering.
const ACCEPT_RECALL_FLAT: f32 = 0.20;

fn main() {
    print_env();

    println!("Dataset:  n={N}, dim={DIM}, queries={N_QUERIES}, k={K}");
    println!("PQ config: M={M}, K={KQ}  (sub_dim={})", DIM / M);
    println!("IVF config: n_lists={N_LISTS}, n_probe={N_PROBE}");
    println!("ResidualPQ oversampling: {OVERSAMPLING}x");
    println!();

    // Generate dataset.
    let (db_vecs, query_vecs) = gen_dataset(N, N_QUERIES, DIM, 99);

    // Ground truth via brute-force.
    println!("Computing brute-force ground truth …");
    let mut exact = ExactSearch::new();
    for i in 0..N {
        exact.insert(&db_vecs[i * DIM..(i + 1) * DIM]);
    }
    let ground_truth: Vec<Vec<usize>> = (0..N_QUERIES)
        .map(|qi| exact.search_exact(&query_vecs[qi * DIM..(qi + 1) * DIM], K))
        .collect();
    println!("Ground truth ready.\n");

    // Train PQ codebook once; all three variants share it.
    println!("Training PQ codebook (M={M}, K={KQ}, dim={DIM}) …");
    let t0 = Instant::now();
    let config = PqConfig::new(M, KQ);
    let codebook = PqCodebook::train(config, &db_vecs, DIM);
    let train_ms = t0.elapsed().as_millis();
    let codebook_bytes = codebook.memory_bytes();
    println!(
        "  Codebook trained in {train_ms} ms, size: {} KB\n",
        codebook_bytes / 1024
    );

    // ── Variant 1: Flat PQ ──────────────────────────────────────────────────
    let result1 = {
        let mut idx = FlatPqIndex::new(codebook.clone());
        for i in 0..N {
            idx.insert(&db_vecs[i * DIM..(i + 1) * DIM]);
        }
        bench_variant(&mut idx, &query_vecs, &ground_truth, N_QUERIES, DIM, K)
    };

    // ── Variant 2: IVF+PQ ──────────────────────────────────────────────────
    let result2 = {
        let mut idx = IvfPqIndex::new(codebook.clone(), N_LISTS, N_PROBE, &db_vecs);
        for i in 0..N {
            idx.insert(&db_vecs[i * DIM..(i + 1) * DIM]);
        }
        bench_variant(&mut idx, &query_vecs, &ground_truth, N_QUERIES, DIM, K)
    };

    // ── Variant 3: Residual PQ ──────────────────────────────────────────────
    let result3 = {
        let mut idx = ResidualPqIndex::new(codebook.clone(), OVERSAMPLING);
        for i in 0..N {
            idx.insert(&db_vecs[i * DIM..(i + 1) * DIM]);
        }
        bench_variant(&mut idx, &query_vecs, &ground_truth, N_QUERIES, DIM, K)
    };

    // ── Print table ─────────────────────────────────────────────────────────
    println!("{:-<100}", "");
    println!(
        "{:<14} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "Variant", "Recall@10", "Mean(µs)", "P50(µs)", "P95(µs)", "QPS", "Mem(KB)", "Pass"
    );
    println!("{:-<100}", "");
    for (i, r) in [&result1, &result2, &result3].iter().enumerate() {
        let threshold = if i == 0 {
            ACCEPT_RECALL_FLAT
        } else if i == 2 {
            ACCEPT_RECALL_RESIDUAL
        } else {
            0.0
        };
        let pass = if threshold == 0.0 {
            "—"
        } else if r.recall >= threshold {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "{:<14} {:>8.3} {:>10.1} {:>10.1} {:>10.1} {:>12.0} {:>12} {:>8}",
            r.name, r.recall, r.mean_us, r.p50_us, r.p95_us, r.qps, r.mem_kb, pass
        );
    }
    println!("{:-<100}", "");

    // Compression ratio.
    let raw_bytes = N * DIM * 4;
    let codes_bytes = N * M;
    println!(
        "\nCompression:  raw={} KB  PQ codes={} KB  ratio={}x",
        raw_bytes / 1024,
        codes_bytes / 1024,
        raw_bytes / codes_bytes.max(1)
    );

    // Acceptance gates.
    println!("\nAcceptance: FlatPQ recall@{K} ≥ {ACCEPT_RECALL_FLAT}  (coarse-filter gate)");
    println!("Acceptance: ResidualPQ recall@{K} ≥ {ACCEPT_RECALL_RESIDUAL}  (production gate)");

    let flat_pass = result1.recall >= ACCEPT_RECALL_FLAT;
    let residual_pass = result3.recall >= ACCEPT_RECALL_RESIDUAL;

    if flat_pass {
        println!(
            "  FlatPQ   : PASS ({:.3} ≥ {ACCEPT_RECALL_FLAT})",
            result1.recall
        );
    } else {
        eprintln!(
            "  FlatPQ   : FAIL ({:.3} < {ACCEPT_RECALL_FLAT})",
            result1.recall
        );
    }
    if residual_pass {
        println!(
            "  ResidualPQ: PASS ({:.3} ≥ {ACCEPT_RECALL_RESIDUAL})",
            result3.recall
        );
    } else {
        eprintln!(
            "  ResidualPQ: FAIL ({:.3} < {ACCEPT_RECALL_RESIDUAL})",
            result3.recall
        );
    }

    if flat_pass && residual_pass {
        println!("\nRESULT: PASS — all acceptance thresholds met");
    } else {
        eprintln!("\nRESULT: FAIL — one or more acceptance thresholds missed");
        std::process::exit(1);
    }
}

// ── Types ────────────────────────────────────────────────────────────────────

struct BenchResult {
    name: String,
    recall: f32,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    mem_kb: usize,
}

fn bench_variant<I: PqSearch>(
    idx: &mut I,
    query_vecs: &[f32],
    ground_truth: &[Vec<usize>],
    n_queries: usize,
    dim: usize,
    k: usize,
) -> BenchResult {
    let name = idx.name().to_string();
    let mem_kb = idx.memory_bytes() / 1024;

    let mut latencies_us = Vec::with_capacity(n_queries);
    let mut total_recall = 0.0f32;

    for qi in 0..n_queries {
        let query = &query_vecs[qi * dim..(qi + 1) * dim];
        let t0 = Instant::now();
        let results = idx.search(query, k);
        let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
        latencies_us.push(elapsed_us);
        total_recall += recall_at_k(&results, &ground_truth[qi], k);
    }

    latencies_us.sort_by(|a, b| a.total_cmp(b));
    let mean_us = latencies_us.iter().sum::<f64>() / n_queries as f64;
    let p50_us = percentile(&latencies_us, 50.0);
    let p95_us = percentile(&latencies_us, 95.0);
    let total_s: f64 = latencies_us.iter().sum::<f64>() / 1e6;
    let qps = n_queries as f64 / total_s.max(1e-9);
    let recall = total_recall / n_queries as f32;

    println!("  {name}: recall={recall:.3}, mean={mean_us:.1}µs, p50={p50_us:.1}µs, p95={p95_us:.1}µs, mem={mem_kb}KB");

    BenchResult {
        name,
        recall,
        mean_us,
        p50_us,
        p95_us,
        qps,
        mem_kb,
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Generate a low-intrinsic-dimension embedding dataset.
///
/// Vectors are formed as: v = A * z + ε, where:
///   A ∈ ℝ^{dim × k_factors} is a random projection matrix,
///   z ∈ ℝ^k_factors is a low-dimensional latent code (standard Gaussian),
///   ε ∈ ℝ^dim is small isotropic noise (σ = 0.05).
///
/// This mirrors real embedding distributions from encoders: a few dozen
/// effective dimensions explain most variance, making PQ sub-space
/// quantization highly discriminative. With k_factors = dim/M, each
/// PQ sub-space of width `sub_dim` captures roughly one latent factor.
fn gen_dataset(n: usize, q: usize, dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let k_factors = dim / 2; // latent dimension (64 for dim=128)
    let noise_sigma = 0.05f32;
    let mut rng = StdRng::seed_from_u64(seed);

    // Random projection matrix A (dim × k_factors), normalised by √k.
    let scale = 1.0 / (k_factors as f32).sqrt();
    let proj: Vec<f32> = (0..dim * k_factors)
        .map(|_| gauss_sample(&mut rng) * scale)
        .collect();

    let embed = |rng: &mut StdRng| -> Vec<f32> {
        // Sample latent z.
        let z: Vec<f32> = (0..k_factors).map(|_| gauss_sample(rng)).collect();
        // v = A*z.
        let mut v = vec![0.0f32; dim];
        for d in 0..dim {
            for k in 0..k_factors {
                v[d] += proj[d * k_factors + k] * z[k];
            }
            // Add isotropic noise.
            v[d] += gauss_sample(rng) * noise_sigma;
        }
        v
    };

    let mut db = Vec::with_capacity(n * dim);
    for _ in 0..n {
        db.extend(embed(&mut rng));
    }
    let mut qv = Vec::with_capacity(q * dim);
    for _ in 0..q {
        qv.extend(embed(&mut rng));
    }
    (db, qv)
}

/// Box-Muller Gaussian sample.
#[inline]
fn gauss_sample(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn print_env() {
    println!("=== PQ-ADC Search Benchmark ===");
    println!("OS: {}", std::env::consts::OS);
    println!("Arch: {}", std::env::consts::ARCH);
    if let Ok(v) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        println!("Rustc: {}", String::from_utf8_lossy(&v.stdout).trim());
    }
    println!();
}
