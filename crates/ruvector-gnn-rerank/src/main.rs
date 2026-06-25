//! ruvector-gnn-rerank benchmark
//!
//! Simulates a quantised-ANN retrieval pipeline and compares four reranking
//! strategies on a synthetic multi-Gaussian corpus.
//!
//! Gaussian noise is added to true similarity scores to simulate the ranking
//! errors produced by 1-bit (RaBitQ-style) or low-bit quantised indexes.
//! All four rerankers receive the same noisy candidate set; the only difference
//! is how they score and sort those candidates.
//!
//! Run:
//!   cargo run --release -p ruvector-gnn-rerank --bin benchmark

use std::collections::HashSet;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use ruvector_gnn_rerank::{
    Candidate, CandidateReranker, ExactL2Reranker, GnnDiffusionReranker, GnnMincutReranker,
    NoisyScoreReranker,
};

// ── configuration ─────────────────────────────────────────────────────────────

const N: usize = 5_000;
const DIM: usize = 128;
const N_CLUSTERS: usize = 20;
const CLUSTER_SIGMA: f32 = 0.5;
const N_QUERIES: usize = 100;
const K: usize = 10;
const RETRIEVAL_K: usize = 80;
// Noise is added to negative-L2 scores.  With typical intra-cluster L2 gap
// of ~0.5, sigma=0.4 causes frequent rank inversions near the k boundary
// while keeping candidate coverage high (true top-K remain in top-RETRIEVAL_K
// because the gap to rank-81 is ~3-4).
const NOISE_SIGMA: f32 = 0.40;
const K_GRAPH: usize = 8;
const SEED: u64 = 42;

// ── data generation ───────────────────────────────────────────────────────────

fn gen_corpus(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-4.0_f32..4.0)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter()
                .map(|&x| x + rng.gen_range(-CLUSTER_SIGMA..CLUSTER_SIGMA))
                .collect()
        })
        .collect()
}

fn gen_queries(corpus: &[Vec<f32>], n_queries: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_queries)
        .map(|_| {
            let base = &corpus[rng.gen_range(0..corpus.len())];
            base.iter()
                .map(|&x| x + rng.gen_range(-0.1_f32..0.1))
                .collect()
        })
        .collect()
}

// ── distance helpers ──────────────────────────────────────────────────────────

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

// ── ground truth ──────────────────────────────────────────────────────────────

fn exact_topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> HashSet<usize> {
    let mut dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2sq(query, v)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

// ── approximate retrieval ─────────────────────────────────────────────────────

/// Noisy retrieval: compute true negative-L2 scores, add Gaussian noise, return top-`retrieval_k`.
///
/// Uses negative L2 distance as the base score (higher = closer to query).
/// Gaussian noise is added to simulate quantised distance estimation errors.
///
/// This is a more realistic model than similarity compression (1/(1+L2)):
/// true top-K items have gaps of ~0.5–2.0 to rank-(K+1) items (intra-cluster),
/// while their gap to rank-(RETRIEVAL_K+1) items is ~3–8 (inter-cluster).
/// A noise sigma of 0.40 therefore causes rank inversions near the K boundary
/// without pushing true top-K items out of the candidate set.
fn noisy_retrieve(
    query: &[f32],
    corpus: &[Vec<f32>],
    retrieval_k: usize,
    noise_sigma: f32,
    rng: &mut StdRng,
) -> Vec<Candidate> {
    let noise = Normal::new(0.0_f32, noise_sigma).unwrap();
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let true_l2 = l2sq(query, v).sqrt();
            // Score: higher = closer. Use negative L2 + noise.
            let noisy_score = -true_l2 + noise.sample(rng);
            (i, noisy_score)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
        .into_iter()
        .take(retrieval_k)
        .map(|(id, noisy_score)| Candidate {
            id: id as u32,
            vector: corpus[id].clone(),
            noisy_score,
        })
        .collect()
}

// ── metrics ───────────────────────────────────────────────────────────────────

fn recall_at_k(results: &[ruvector_gnn_rerank::RankedResult], gt: &HashSet<usize>) -> f64 {
    results
        .iter()
        .filter(|r| gt.contains(&(r.id as usize)))
        .count() as f64
        / gt.len() as f64
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((values.len() as f64 * p / 100.0) as usize).min(values.len() - 1);
    values[idx]
}

// ── benchmark runner ──────────────────────────────────────────────────────────

struct BenchResult {
    name: &'static str,
    mean_recall: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
}

fn run_bench<R: CandidateReranker>(
    name: &'static str,
    reranker: &R,
    queries: &[Vec<f32>],
    cands_per_query: &[Vec<Candidate>],
    ground_truth: &[HashSet<usize>],
    k: usize,
) -> BenchResult {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut lats = Vec::with_capacity(queries.len());

    for (qi, (query, gt)) in queries.iter().zip(ground_truth.iter()).enumerate() {
        let cands = &cands_per_query[qi];
        let t0 = Instant::now();
        let results = reranker.rerank(query, cands, k).expect("rerank failed");
        let us = t0.elapsed().as_nanos() as f64 / 1_000.0;
        recalls.push(recall_at_k(&results, gt));
        lats.push(us);
    }

    let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
    let mean_us = lats.iter().sum::<f64>() / lats.len() as f64;
    let p50_us = percentile(&mut lats.clone(), 50.0);
    let p95_us = percentile(&mut lats, 95.0);
    let qps = 1_000_000.0 / mean_us;

    BenchResult {
        name,
        mean_recall,
        mean_us,
        p50_us,
        p95_us,
        qps,
    }
}

// ── acceptance test ───────────────────────────────────────────────────────────

fn acceptance_test(results: &[BenchResult]) -> bool {
    let noisy = results
        .iter()
        .find(|r| r.name.starts_with("NoisyScore"))
        .unwrap();
    let gnn = results
        .iter()
        .find(|r| r.name.starts_with("GnnDiffusion"))
        .unwrap();
    let exact = results
        .iter()
        .find(|r| r.name.starts_with("ExactL2"))
        .unwrap();

    // GNN diffusion must strictly improve over the noisy baseline.
    let gnn_beats_noisy = gnn.mean_recall > noisy.mean_recall;
    // The exact oracle must be at least as good as GNN (sanity check).
    let exact_at_least_gnn = exact.mean_recall >= gnn.mean_recall;

    if !gnn_beats_noisy {
        eprintln!(
            "FAIL: GnnDiffusion ({:.1}%) did not beat NoisyScore ({:.1}%)",
            gnn.mean_recall * 100.0,
            noisy.mean_recall * 100.0
        );
    }
    if !exact_at_least_gnn {
        eprintln!(
            "FAIL: ExactL2 ({:.1}%) not ≥ GnnDiffusion ({:.1}%)",
            exact.mean_recall * 100.0,
            gnn.mean_recall * 100.0
        );
    }

    gnn_beats_noisy && exact_at_least_gnn
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    // ── header ───────────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         ruvector-gnn-rerank  ·  benchmark                        ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  OS  : {:<57} ║", std::env::consts::OS);
    println!("║  arch: {:<57} ║", std::env::consts::ARCH);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  N={N:<5}  DIM={DIM:<4}  clusters={N_CLUSTERS:<3}  queries={N_QUERIES:<4}  K={K:<3}               ║"
    );
    println!(
        "║  retrieval_k={RETRIEVAL_K:<3}  noise_σ={NOISE_SIGMA:.2}  k_graph={K_GRAPH:<3}                     ║"
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // ── corpus & queries ─────────────────────────────────────────────────────
    println!("\nGenerating corpus (N={N}, D={DIM}, clusters={N_CLUSTERS}) …");
    let corpus = gen_corpus(N, DIM, N_CLUSTERS, SEED);

    println!("Generating {N_QUERIES} queries …");
    let queries = gen_queries(&corpus, N_QUERIES, SEED + 1);

    // ── ground truth ─────────────────────────────────────────────────────────
    println!("Computing exact ground truth (brute-force) …");
    let t0 = Instant::now();
    let ground_truth: Vec<HashSet<usize>> =
        queries.iter().map(|q| exact_topk(q, &corpus, K)).collect();
    println!("  done in {:.1}ms", t0.elapsed().as_millis());

    // ── noisy retrieval ───────────────────────────────────────────────────────
    println!("Simulating noisy retrieval (noise_σ={NOISE_SIGMA}) …");
    let mut rng = StdRng::seed_from_u64(SEED + 99);
    let cands_per_query: Vec<Vec<Candidate>> = queries
        .iter()
        .map(|q| noisy_retrieve(q, &corpus, RETRIEVAL_K, NOISE_SIGMA, &mut rng))
        .collect();

    // Coverage: fraction of true top-K present in the candidate set.
    let coverage: f64 = queries
        .iter()
        .zip(ground_truth.iter())
        .zip(cands_per_query.iter())
        .map(|((_, gt), cands)| {
            let ids: HashSet<usize> = cands.iter().map(|c| c.id as usize).collect();
            gt.intersection(&ids).count() as f64 / gt.len() as f64
        })
        .sum::<f64>()
        / N_QUERIES as f64;
    println!(
        "  candidate coverage of true top-{K}: {:.1}%",
        coverage * 100.0
    );

    // ── run benchmarks ────────────────────────────────────────────────────────
    println!("\nRunning reranker benchmarks …");

    let noisy_r = NoisyScoreReranker;
    let gnn_r = GnnDiffusionReranker {
        alpha: 0.60,
        hops: 1,
        k_graph: K_GRAPH,
    };
    let mincut_r = GnnMincutReranker {
        alpha: 0.60,
        coherence_threshold: 0.50,
        k_graph: K_GRAPH,
    };
    let exact_r = ExactL2Reranker;

    let results = vec![
        run_bench(
            "NoisyScore (baseline)",
            &noisy_r,
            &queries,
            &cands_per_query,
            &ground_truth,
            K,
        ),
        run_bench(
            "GnnDiffusion (1-hop, α=0.60)",
            &gnn_r,
            &queries,
            &cands_per_query,
            &ground_truth,
            K,
        ),
        run_bench(
            "GnnMincut (coh≥0.50, α=0.60)",
            &mincut_r,
            &queries,
            &cands_per_query,
            &ground_truth,
            K,
        ),
        run_bench(
            "ExactL2 (oracle)",
            &exact_r,
            &queries,
            &cands_per_query,
            &ground_truth,
            K,
        ),
    ];

    // ── results table ─────────────────────────────────────────────────────────
    println!();
    println!(
        "{:<35}  {:>10}  {:>10}  {:>10}  {:>12}",
        "Variant", "recall@10", "mean µs", "p50 µs", "p95 µs"
    );
    println!("{}", "─".repeat(82));
    for r in &results {
        println!(
            "{:<35}  {:>9.1}%  {:>10.1}  {:>10.1}  {:>12.1}",
            r.name,
            r.mean_recall * 100.0,
            r.mean_us,
            r.p50_us,
            r.p95_us,
        );
    }

    // ── throughput ────────────────────────────────────────────────────────────
    println!("\nThroughput (single-threaded, reranking step only):");
    for r in &results {
        println!("  {:<35}  {:>10.0} QPS", r.name, r.qps);
    }

    // ── memory model ─────────────────────────────────────────────────────────
    println!("\nMemory model (per query):");
    let vec_bytes = RETRIEVAL_K * (4 + DIM * 4 + 4);
    let graph_bytes = RETRIEVAL_K * K_GRAPH * 8; // (usize, f32) = 8 bytes
    println!(
        "  candidate vectors : {RETRIEVAL_K} × (4B id + {}B vec + 4B score) = {:.1} KB",
        DIM * 4,
        vec_bytes as f64 / 1024.0
    );
    println!(
        "  candidate graph   : {RETRIEVAL_K} × {K_GRAPH} × 8B               = {:.1} KB",
        graph_bytes as f64 / 1024.0
    );
    println!(
        "  total             :                                   = {:.1} KB",
        (vec_bytes + graph_bytes) as f64 / 1024.0
    );

    // ── recall improvement summary ────────────────────────────────────────────
    let noisy_recall = results[0].mean_recall;
    let gnn_recall = results[1].mean_recall;
    let mincut_recall = results[2].mean_recall;
    let exact_recall = results[3].mean_recall;
    println!(
        "\nRecall improvement from GNN diffusion : {:+.1} pp",
        (gnn_recall - noisy_recall) * 100.0
    );
    println!(
        "Recall improvement from GNN mincut    : {:+.1} pp",
        (mincut_recall - noisy_recall) * 100.0
    );
    println!(
        "Gap to oracle (ExactL2)               : {:.1} pp",
        (exact_recall - gnn_recall) * 100.0
    );

    // ── acceptance ────────────────────────────────────────────────────────────
    println!("\n{}", "─".repeat(82));
    println!("Acceptance: GnnDiffusion recall > NoisyScore recall");
    if acceptance_test(&results) {
        println!("RESULT: PASS ✓");
    } else {
        println!("RESULT: FAIL ✗");
        std::process::exit(1);
    }
}
