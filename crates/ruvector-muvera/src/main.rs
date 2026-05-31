//! MUVERA demo: brute-force MaxSim vs FDE flat-scan at three configs.
//!
//! Section A — i.i.d. Gaussian data: worst case for FDE (no geometric structure).
//!   Recall approaches k/n (random baseline); speedup is the key metric.
//!
//! Section B — Clustered data: realistic case where token sets share semantic
//!   neighborhoods.  Documents are perturbations of 50 cluster centers; recall
//!   rises substantially, demonstrating the algorithm's approximation quality.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_muvera::{FdeConfig, FdeEncoder, MuveraIndex};
use std::time::Instant;

const DIM: usize = 128;
const TOP_K: usize = 10;

// ── Section A constants ───────────────────────────────────────────────────────
const N_DOCS_IID: usize = 5_000;
const N_TOKENS_IID: usize = 32;
const N_QUERIES_IID: usize = 200;

// ── Section B constants ───────────────────────────────────────────────────────
const N_CLUSTERS: usize = 50;
const DOCS_PER_CLUSTER: usize = 100; // 5 000 total
const N_TOKENS_CLUST: usize = 16;
const NOISE_SIGMA: f32 = 0.25; // perturbation around cluster center
const N_QUERIES_CLUST: usize = 100;

fn random_unit_vec(rng: &mut impl Rng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / norm.max(f32::EPSILON)).collect()
}

fn normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let n = norm.max(f32::EPSILON);
    for x in v.iter_mut() {
        *x /= n;
    }
}

fn maxsim_score(doc: &[Vec<f32>], query: &[Vec<f32>]) -> f32 {
    query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

fn brute_force_search(docs: &[Vec<Vec<f32>>], query: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| (i, maxsim_score(doc, query)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.iter().take(k).map(|(i, _)| *i).collect()
}

fn recall_at_k(predicted: &[String], ground_truth: &[usize]) -> f32 {
    let gt: std::collections::HashSet<usize> = ground_truth.iter().cloned().collect();
    let hits = predicted
        .iter()
        .filter(|id| id.parse::<usize>().map(|i| gt.contains(&i)).unwrap_or(false))
        .count();
    hits as f32 / gt.len() as f32
}

fn print_table_header() {
    println!(
        "{:<38} {:>10} {:>10} {:>10} {:>10}",
        "Variant", "Recall@10", "QPS", "FDE-dim", "Mem (MB)"
    );
    println!("{}", "─".repeat(82));
}

fn run_variants(
    label: &str,
    docs: &[Vec<Vec<f32>>],
    queries: &[Vec<Vec<f32>>],
    ground_truths: &[Vec<usize>],
    bf_qps: f64,
    bf_mem: f64,
    raw_dim: usize,
) {
    println!(
        "{:<38} {:>10} {:>10.0} {:>10} {:>9.2}MB",
        "BruteForce-MaxSim",
        "1.000",
        bf_qps,
        raw_dim,
        bf_mem
    );

    let variants: &[(&str, FdeConfig)] = &[
        (
            "FDE-small  (B=8,  dp=8,  R=4)",
            FdeConfig { dim: DIM, buckets: 8, d_proj: 8, reps: 4 },
        ),
        (
            "FDE-medium (B=16, dp=16, R=4)",
            FdeConfig { dim: DIM, buckets: 16, d_proj: 16, reps: 4 },
        ),
        (
            "FDE-large  (B=32, dp=16, R=4)",
            FdeConfig { dim: DIM, buckets: 32, d_proj: 16, reps: 4 },
        ),
    ];

    let _ = label;
    for (vname, cfg) in variants {
        let fde_dim = cfg.fde_dim();
        let mut enc_rng = StdRng::seed_from_u64(7);
        let encoder = FdeEncoder::new(cfg.clone(), &mut enc_rng).unwrap();
        let mut index = MuveraIndex::new(encoder);

        let build_start = Instant::now();
        for (i, doc) in docs.iter().enumerate() {
            index.insert(i.to_string(), doc).unwrap();
        }
        let build_ms = build_start.elapsed().as_millis();

        let t0 = Instant::now();
        let mut all_results: Vec<Vec<String>> = Vec::with_capacity(queries.len());
        for q in queries {
            let res = index.search(q, TOP_K).unwrap();
            all_results.push(res.into_iter().map(|r| r.id).collect());
        }
        let qps = queries.len() as f64 / t0.elapsed().as_secs_f64();

        let recall: f32 = all_results
            .iter()
            .zip(ground_truths.iter())
            .map(|(pred, gt)| recall_at_k(pred, gt))
            .sum::<f32>()
            / queries.len() as f32;

        let mem_mb = index.fde_memory_bytes() as f64 / 1_048_576.0;
        println!(
            "{:<38} {:>10.3} {:>10.0} {:>10} {:>9.2}MB  [build {}ms]",
            vname, recall, qps, fde_dim, mem_mb, build_ms
        );
    }
}

fn section_a(rng: &mut StdRng) {
    println!("=== Section A — i.i.d. Gaussian unit vectors (worst case) ===");
    println!(
        "    N={N_DOCS_IID} docs, {N_TOKENS_IID} tokens/doc, d={DIM}, {N_QUERIES_IID} queries"
    );
    println!("    Expected recall: k/N = {:.4} (random baseline)", TOP_K as f32 / N_DOCS_IID as f32);
    println!();

    let docs: Vec<Vec<Vec<f32>>> = (0..N_DOCS_IID)
        .map(|_| (0..N_TOKENS_IID).map(|_| random_unit_vec(rng, DIM)).collect())
        .collect();
    let queries: Vec<Vec<Vec<f32>>> = (0..N_QUERIES_IID)
        .map(|_| (0..N_TOKENS_IID).map(|_| random_unit_vec(rng, DIM)).collect())
        .collect();

    let ground_truths: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_search(&docs, q, TOP_K))
        .collect();

    let t0 = Instant::now();
    for q in &queries {
        let _ = brute_force_search(&docs, q, TOP_K);
    }
    let bf_qps = N_QUERIES_IID as f64 / t0.elapsed().as_secs_f64();
    let bf_mem = N_DOCS_IID as f64 * N_TOKENS_IID as f64 * DIM as f64 * 4.0 / 1_048_576.0;

    print_table_header();
    run_variants("iid", &docs, &queries, &ground_truths, bf_qps, bf_mem, N_TOKENS_IID * DIM);
}

fn section_b(rng: &mut StdRng) {
    println!();
    println!("=== Section B — Clustered embeddings (realistic structured data) ===");
    println!(
        "    {N_CLUSTERS} clusters × {DOCS_PER_CLUSTER} docs, {N_TOKENS_CLUST} tokens/doc, d={DIM}"
    );
    println!("    Each token = cluster_center + N(0, {NOISE_SIGMA}²)");
    println!();

    let normal = Normal::new(0.0f32, NOISE_SIGMA).unwrap();
    let centers: Vec<Vec<f32>> = (0..N_CLUSTERS).map(|_| random_unit_vec(rng, DIM)).collect();

    let n_docs = N_CLUSTERS * DOCS_PER_CLUSTER;
    let mut docs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_docs);
    let mut doc_cluster: Vec<usize> = Vec::with_capacity(n_docs);

    for (ci, center) in centers.iter().enumerate() {
        for _ in 0..DOCS_PER_CLUSTER {
            let tokens: Vec<Vec<f32>> = (0..N_TOKENS_CLUST)
                .map(|_| {
                    let mut v: Vec<f32> = center
                        .iter()
                        .map(|&c| c + normal.sample(rng))
                        .collect();
                    normalize(&mut v);
                    v
                })
                .collect();
            docs.push(tokens);
            doc_cluster.push(ci);
        }
    }

    // Each query belongs to a cluster; ground truth = all docs in that cluster.
    let queries: Vec<Vec<Vec<f32>>> = (0..N_QUERIES_CLUST)
        .map(|i| {
            let ci = i % N_CLUSTERS;
            (0..N_TOKENS_CLUST)
                .map(|_| {
                    let mut v: Vec<f32> = centers[ci]
                        .iter()
                        .map(|&c| c + normal.sample(rng))
                        .collect();
                    normalize(&mut v);
                    v
                })
                .collect()
        })
        .collect();

    let ground_truths: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_search(&docs, q, TOP_K))
        .collect();

    let t0 = Instant::now();
    for q in &queries {
        let _ = brute_force_search(&docs, q, TOP_K);
    }
    let bf_qps = N_QUERIES_CLUST as f64 / t0.elapsed().as_secs_f64();
    let bf_mem = n_docs as f64 * N_TOKENS_CLUST as f64 * DIM as f64 * 4.0 / 1_048_576.0;

    print_table_header();
    run_variants("clustered", &docs, &queries, &ground_truths, bf_qps, bf_mem, N_TOKENS_CLUST * DIM);
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    section_a(&mut rng);
    section_b(&mut rng);

    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!();
    println!("Hardware: {cpus} logical CPU(s), cargo --release, rustc 1.94");
}
