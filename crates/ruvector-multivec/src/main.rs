//! MUVERA multi-vector late-interaction benchmark harness.
//!
//! Produces the recall + QPS + memory numbers quoted in the research document.
//!
//! Three index variants are compared on the **same** synthetic ColBERT-style
//! corpus (seeded Gaussian token embeddings):
//!
//!   1. CentroidIndex     — pool tokens → centroid dot product (cheapest, lowest recall)
//!   2. MaxSimIndex       — exact ColBERT MaxSim (oracle)
//!   3. MuveraFdeIndex    — MUVERA FDE approximation (fast + accurate)
//!
//! Run:
//!   cargo run --release -p ruvector-multivec
//!   cargo run --release -p ruvector-multivec -- --fast

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::HashSet;
use std::time::Instant;

use ruvector_multivec::{
    index::{
        CentroidIndex, MaxSimIndex, MultiVecIndex, MuveraFdeIndex, MuveraFdeRerankIndex,
        SearchResult,
    },
    scoring::l2_normalize,
};

// ---------------------------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------------------------

/// Simulate ColBERT token embeddings: each document is a set of `t` L2-normalised
/// unit vectors drawn from a clustered Gaussian (100 clusters). Documents within
/// the same cluster share a centroid — search must distinguish them using
/// multi-token overlap, not just proximity.
fn generate_corpus(
    n_docs: usize,
    tokens_per_doc: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> Vec<(usize, Vec<Vec<f32>>)> {
    use rand::Rng as _;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroid_range = Uniform::new(-2.0f32, 2.0);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| centroid_range.sample(&mut rng)).collect())
        .collect();
    let noise = Normal::new(0.0f64, 0.3).unwrap();

    (0..n_docs)
        .map(|id| {
            let c_idx = rng.gen_range(0..n_clusters);
            let c = &centroids[c_idx];
            let tokens = (0..tokens_per_doc)
                .map(|_| {
                    let mut v: Vec<f32> = c
                        .iter()
                        .map(|&x| x + noise.sample(&mut rng) as f32)
                        .collect();
                    l2_normalize(&mut v);
                    v
                })
                .collect();
            (id, tokens)
        })
        .collect()
}

/// Generate query token sets drawn from the same distribution.
fn generate_queries(
    n_queries: usize,
    tokens_per_query: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> Vec<Vec<Vec<f32>>> {
    generate_corpus(n_queries, tokens_per_query, dim, n_clusters, seed)
        .into_iter()
        .map(|(_, tokens)| tokens)
        .collect()
}

// ---------------------------------------------------------------------------
// Ground-truth
// ---------------------------------------------------------------------------

fn ground_truth_maxsim(
    corpus: &[(usize, Vec<Vec<f32>>)],
    queries: &[Vec<Vec<f32>>],
    k: usize,
) -> Vec<Vec<usize>> {
    use ruvector_multivec::scoring::{l2_normalize, maxsim_exact};
    queries
        .iter()
        .map(|qt| {
            let mut qt_norm = qt.clone();
            for t in &mut qt_norm {
                l2_normalize(t);
            }
            let mut scores: Vec<(usize, f32)> = corpus
                .iter()
                .map(|(id, dt)| {
                    let mut dt_norm = dt.clone();
                    for t in &mut dt_norm {
                        l2_normalize(t);
                    }
                    (*id, maxsim_exact(&qt_norm, &dt_norm))
                })
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scores.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Recall computation
// ---------------------------------------------------------------------------

fn recall_at_k(truth: &[usize], got: &[SearchResult], k: usize) -> f64 {
    let truth_k: HashSet<usize> = truth.iter().take(k).copied().collect();
    let got_k: HashSet<usize> = got.iter().take(k).map(|r| r.id).collect();
    truth_k.intersection(&got_k).count() as f64 / truth_k.len().max(1) as f64
}

// ---------------------------------------------------------------------------
// Per-variant measurement
// ---------------------------------------------------------------------------

struct BenchRow {
    name: String,
    r1: f64,
    r10: f64,
    qps: f64,
    mem_mb: f64,
    build_s: f64,
    lat_ms: f64,
}

fn measure<I: MultiVecIndex>(
    idx: &I,
    queries: &[Vec<Vec<f32>>],
    truth: &[Vec<usize>],
    k: usize,
    build_s: f64,
) -> BenchRow {
    let t = Instant::now();
    let results: Vec<Vec<SearchResult>> = queries
        .iter()
        .map(|q| idx.search(q, k).unwrap())
        .collect();
    let elapsed = t.elapsed();
    let nq = queries.len() as f64;

    let r1: f64 = results
        .iter()
        .zip(truth.iter())
        .map(|(r, t)| recall_at_k(t, r, 1))
        .sum::<f64>()
        / nq;
    let r10: f64 = results
        .iter()
        .zip(truth.iter())
        .map(|(r, t)| recall_at_k(t, r, 10.min(truth[0].len())))
        .sum::<f64>()
        / nq;

    BenchRow {
        name: idx.name().to_string(),
        r1,
        r10,
        qps: nq / elapsed.as_secs_f64(),
        mem_mb: idx.memory_bytes() as f64 / 1_048_576.0,
        build_s,
        lat_ms: elapsed.as_secs_f64() / nq * 1000.0,
    }
}

fn print_header() {
    println!(
        "  {:<36} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8}",
        "variant", "r@1", "r@10", "QPS", "mem/MB", "build/s", "lat/ms"
    );
    println!("  {}", "-".repeat(90));
}

fn print_row(r: &BenchRow) {
    println!(
        "  {:<36} {:>6.1}% {:>6.1}% {:>8.0} {:>8.2} {:>8.3} {:>8.3}",
        r.name,
        r.r1 * 100.0,
        r.r10 * 100.0,
        r.qps,
        r.mem_mb,
        r.build_s,
        r.lat_ms
    );
}

// ---------------------------------------------------------------------------
// Scale sweep
// ---------------------------------------------------------------------------

fn run_scale(
    n_docs: usize,
    tokens_per_doc: usize,
    dim: usize,
    n_queries: usize,
    tokens_per_query: usize,
    fde_m: usize,
    fde_r: usize,
    seed: u64,
) {
    println!(
        "\n── n={n_docs} docs · T={tokens_per_doc} tokens/doc · D={dim} · nq={n_queries} · FDE(M={fde_m},R={fde_r}) ──"
    );

    let corpus = generate_corpus(n_docs, tokens_per_doc, dim, 50, seed);
    let queries = generate_queries(n_queries, tokens_per_query, dim, 50, seed.wrapping_add(1));
    let k = 10.min(n_docs);

    println!("  Computing MaxSim ground-truth (brute-force oracle)...");
    let truth = ground_truth_maxsim(&corpus, &queries, k);

    // Build all three indexes.
    let t = Instant::now();
    let mut centroid_idx = CentroidIndex::new(dim);
    for (id, tokens) in &corpus {
        centroid_idx.add(*id, tokens.clone()).unwrap();
    }
    let build_centroid = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut maxsim_idx = MaxSimIndex::new(dim);
    for (id, tokens) in &corpus {
        maxsim_idx.add(*id, tokens.clone()).unwrap();
    }
    let build_maxsim = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut chamfer_idx = MaxSimIndex::new(dim).with_chamfer();
    for (id, tokens) in &corpus {
        chamfer_idx.add(*id, tokens.clone()).unwrap();
    }
    let build_chamfer = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut fde_idx = MuveraFdeIndex::new(dim, fde_m, fde_r, 42).unwrap();
    for (id, tokens) in &corpus {
        fde_idx.add(*id, tokens.clone()).unwrap();
    }
    let build_fde = t.elapsed().as_secs_f64();

    // FDE+Rerank with rerank_factor=5 (fetch 5k candidates, rerank with MaxSim).
    let t = Instant::now();
    let mut fde_rr_idx = MuveraFdeRerankIndex::new(dim, fde_m, fde_r, 5, 42).unwrap();
    for (id, tokens) in &corpus {
        fde_rr_idx.add(*id, tokens.clone()).unwrap();
    }
    let build_fde_rr = t.elapsed().as_secs_f64();

    print_header();

    // MaxSim is the oracle — use its results as ground truth for recall computation.
    let rows = [
        measure(&centroid_idx, &queries, &truth, k, build_centroid),
        measure(&maxsim_idx, &queries, &truth, k, build_maxsim),
        measure(&chamfer_idx, &queries, &truth, k, build_chamfer),
        measure(&fde_idx, &queries, &truth, k, build_fde),
        measure(&fde_rr_idx, &queries, &truth, k, build_fde_rr),
    ];
    for r in &rows {
        print_row(r);
    }

    // Memory breakdown.
    let raw_tokens_bytes = n_docs * tokens_per_doc * dim * 4;
    let fde_bytes = fde_idx.memory_bytes();
    let fde_rr_bytes = fde_rr_idx.memory_bytes();
    println!("\n  Memory comparison (n={n_docs}, T={tokens_per_doc}, D={dim}):");
    println!(
        "    Raw token storage (MaxSim)     : {:.2} MB ({} bytes/doc)",
        raw_tokens_bytes as f64 / 1_048_576.0,
        tokens_per_doc * dim * 4
    );
    println!(
        "    FDE-only storage               : {:.2} MB ({} bytes/doc, {:.1}× overhead vs 1-vec)",
        fde_bytes as f64 / 1_048_576.0,
        fde_bytes / n_docs,
        fde_bytes as f64 / (n_docs * dim * 4) as f64
    );
    println!(
        "    FDE+token storage (rerank)     : {:.2} MB ({} bytes/doc)",
        fde_rr_bytes as f64 / 1_048_576.0,
        fde_rr_bytes / n_docs
    );
    println!(
        "    Centroid storage               : {:.2} MB ({} bytes/doc)",
        centroid_idx.memory_bytes() as f64 / 1_048_576.0,
        dim * 4
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let fast = std::env::args().any(|a| a == "--fast");

    println!("=== ruvector-multivec: MUVERA FDE benchmark harness ===");
    println!("Paper: 'MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings'");
    println!("       Karpukhin et al., NeurIPS 2024 (arXiv:2405.19504)");
    println!();
    println!("Recall is measured against the exact MaxSim (ColBERT oracle) top-10.");
    println!("All variants run on the same seeded Gaussian ColBERT-style corpus.");
    println!(
        "{}",
        if fast { "-- fast mode (small n)" } else { "-- full mode" }
    );

    if fast {
        // Quick smoke test — small n, FDE(M=8,R=4).
        run_scale(500, 8, 64, 50, 4, 8, 4, 42);
        run_scale(2_000, 16, 128, 100, 8, 8, 4, 99);
    } else {
        // Full benchmark suite. Reduce nq for larger n to keep oracle fast.
        // FDE(M=8,R=4): FDE_dim = 4×8×64=2048 or 4×8×128=4096.
        run_scale(1_000, 8, 64, 100, 4, 8, 4, 10);
        run_scale(5_000, 16, 128, 100, 8, 8, 4, 20);
        run_scale(10_000, 32, 128, 50, 16, 8, 4, 30);
        run_scale(20_000, 32, 128, 30, 16, 8, 4, 40);
    }

    println!("\nAll numbers reproducible: cargo run --release -p ruvector-multivec");
}
