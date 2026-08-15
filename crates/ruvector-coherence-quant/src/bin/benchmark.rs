//! Benchmark: Coherence-Adaptive Quantization
//!
//! Compares uniform 8-bit (baseline), uniform 4-bit (candidate A), and
//! mutual-kNN-coherence-adaptive 4/8-bit (candidate B) scalar quantization
//! on a clustered synthetic corpus.
//!
//! Run:
//!   cargo run --release -p ruvector-coherence-quant --bin benchmark

use ruvector_coherence_quant::{
    coherence::{build_knn_graph, mutual_knn_coherence},
    dataset::{clustered_vectors, ground_truth, jittered_queries},
    metrics::LatencyStats,
    quantize::{quantize, Bits, QuantizedVector},
    recall_at_k,
    search::QuantizedIndex,
};
use std::time::Instant;

// ─── Pre-registered experiment configuration ──────────────────────────────
// (fixed before benchmarking; see docs/research/nightly/2026-08-15-coherence-adaptive-quant)
const N: usize = 4_000;
const DIM: usize = 32;
const N_CLUSTERS: usize = 12;
const CLUSTER_NOISE: f32 = 0.18;
const CORPUS_SEED: u64 = 42;

const K_COHERENCE: usize = 12; // k for the mutual-kNN coherence graph
const COHERENCE_THRESHOLD: f32 = 0.5; // >= threshold -> core (4-bit); < threshold -> boundary (8-bit)

const K_RECALL: usize = 10;
const N_QUERIES: usize = 300;
const QUERY_JITTER: f32 = 0.12;
const QUERY_SEED: u64 = 777;

// ─── Pre-registered acceptance thresholds ─────────────────────────────────
// ACCEPT requires all of:
//   1. candidate_A recall is meaningfully worse than baseline (>1.5pp) --
//      otherwise uniform 4-bit already suffices and there is nothing to fix.
//   2. candidate_B recall is within 1.5pp of baseline (recovers most of the
//      precision loss uniform 4-bit incurs).
//   3. candidate_B total memory <= 65% of baseline memory (meaningfully
//      compressed, not just "pretend to adapt but store everything at 8-bit").
const RECALL_GAP_MIN_FOR_SIGNAL: f32 = 0.015;
const RECALL_RECOVERY_TOLERANCE: f32 = 0.015;
const MEMORY_BUDGET_FRACTION: f32 = 0.65;

fn build_index(corpus: &[Vec<f32>], bits: &[Bits]) -> QuantizedIndex {
    let vectors: Vec<QuantizedVector> = corpus
        .iter()
        .zip(bits.iter())
        .map(|(v, &b)| quantize(v, b))
        .collect();
    QuantizedIndex { vectors }
}

fn compute_recall(
    index: &QuantizedIndex,
    queries: &[Vec<f32>],
    corpus: &[Vec<f32>],
    k: usize,
) -> f32 {
    let total: f32 = queries
        .iter()
        .map(|q| {
            let gt = ground_truth(q, corpus, k);
            let hits = index.search(q, k);
            recall_at_k(&gt, &hits, k)
        })
        .sum();
    total / queries.len() as f32
}

struct VariantResult {
    name: &'static str,
    recall: f32,
    stats: LatencyStats,
    mem_bytes: usize,
    mean_bits: f32,
}

fn print_row(r: &VariantResult) {
    println!(
        "  {:<28} recall={:.4}  mean_bits={:.2}  mem={:>7}KB  \
        mean={:6.1}us  p50={:6.1}us  p95={:7.1}us  {:>8.0} qps",
        r.name,
        r.recall,
        r.mean_bits,
        r.mem_bytes / 1024,
        r.stats.mean_us,
        r.stats.p50_us,
        r.stats.p95_us,
        r.stats.throughput_qps,
    );
}

fn main() {
    println!("=== Coherence-Adaptive Quantization Benchmark ===");
    println!();
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let ncpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    println!("OS:           {os} / {arch}");
    println!("CPU threads:  {ncpu}");
    println!("Rust:         (see: rustc --version)");
    println!();

    println!("Dataset:");
    println!("  N (corpus)   : {N}");
    println!("  Dimensions   : {DIM}");
    println!("  Clusters     : {N_CLUSTERS}  noise={CLUSTER_NOISE}");
    println!("  Queries      : {N_QUERIES} (jitter={QUERY_JITTER}, held-out)");
    println!("  k (recall)   : {K_RECALL}");
    println!("  k (coherence): {K_COHERENCE}  threshold={COHERENCE_THRESHOLD}");
    println!();

    println!("Building corpus...");
    let t0 = Instant::now();
    let corpus = clustered_vectors(N, DIM, N_CLUSTERS, CLUSTER_NOISE, CORPUS_SEED);
    println!("  corpus built in {:.1}ms", t0.elapsed().as_millis());

    let queries = jittered_queries(&corpus, N_QUERIES, QUERY_JITTER, QUERY_SEED);

    println!("Building mutual-kNN coherence graph (k={K_COHERENCE})...");
    let t1 = Instant::now();
    let knn = build_knn_graph(&corpus, K_COHERENCE);
    let coherence = mutual_knn_coherence(&knn);
    let coherence_build_ms = t1.elapsed().as_millis();
    println!("  coherence graph + scores built in {coherence_build_ms}ms");

    let mut sorted_coherence = coherence.clone();
    sorted_coherence.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_coherence = coherence.iter().sum::<f32>() / coherence.len() as f32;
    let p50_coherence = sorted_coherence[sorted_coherence.len() / 2];
    let n_core = coherence
        .iter()
        .filter(|&&c| c >= COHERENCE_THRESHOLD)
        .count();
    let n_boundary = coherence.len() - n_core;
    println!(
        "  coherence: mean={mean_coherence:.3} p50={p50_coherence:.3}  \
        core(>= {COHERENCE_THRESHOLD})={n_core} ({:.1}%)  boundary={n_boundary} ({:.1}%)",
        100.0 * n_core as f32 / coherence.len() as f32,
        100.0 * n_boundary as f32 / coherence.len() as f32,
    );
    println!();

    // ── Variant bit assignments ────────────────────────────────────────────
    let bits_baseline: Vec<Bits> = vec![Bits::Eight; N];
    let bits_candidate_a: Vec<Bits> = vec![Bits::Four; N];
    let bits_candidate_b: Vec<Bits> = coherence
        .iter()
        .map(|&c| {
            if c >= COHERENCE_THRESHOLD {
                Bits::Four
            } else {
                Bits::Eight
            }
        })
        .collect();

    let index_baseline = build_index(&corpus, &bits_baseline);
    let index_a = build_index(&corpus, &bits_candidate_a);
    let index_b = build_index(&corpus, &bits_candidate_b);

    println!("Benchmarking variants ({N_QUERIES} held-out queries)...");
    println!();

    let mut results = Vec::new();
    for (name, index) in [
        ("baseline_uniform_8bit", &index_baseline),
        ("candidate_A_uniform_4bit", &index_a),
        ("candidate_B_coherence_adaptive", &index_b),
    ] {
        let recall = compute_recall(index, &queries, &corpus, K_RECALL);
        let (_, stats) = LatencyStats::measure(N_QUERIES, |i| index.search(&queries[i], K_RECALL));
        let r = VariantResult {
            name,
            recall,
            stats,
            mem_bytes: index.total_bytes(),
            mean_bits: index.mean_bits(),
        };
        print_row(&r);
        results.push(r);
    }

    println!();
    println!("─── Acceptance Evaluation ───");
    println!();
    let baseline = &results[0];
    let cand_a = &results[1];
    let cand_b = &results[2];

    let a_gap = baseline.recall - cand_a.recall;
    let b_gap = baseline.recall - cand_b.recall;
    let b_mem_fraction = cand_b.mem_bytes as f32 / baseline.mem_bytes as f32;

    let signal_present = a_gap > RECALL_GAP_MIN_FOR_SIGNAL;
    let recall_recovered = b_gap <= RECALL_RECOVERY_TOLERANCE;
    let memory_compressed = b_mem_fraction <= MEMORY_BUDGET_FRACTION;

    println!(
        "  1. Uniform 4-bit recall gap vs baseline : {:.4} (need > {:.4} for a real signal) [{}]",
        a_gap,
        RECALL_GAP_MIN_FOR_SIGNAL,
        if signal_present { "PASS" } else { "FAIL" }
    );
    println!(
        "  2. Candidate B recall gap vs baseline   : {:.4} (need <= {:.4})            [{}]",
        b_gap,
        RECALL_RECOVERY_TOLERANCE,
        if recall_recovered { "PASS" } else { "FAIL" }
    );
    println!(
        "  3. Candidate B memory / baseline memory : {:.3} (need <= {:.2})              [{}]",
        b_mem_fraction,
        MEMORY_BUDGET_FRACTION,
        if memory_compressed { "PASS" } else { "FAIL" }
    );
    println!();

    let verdict = if !signal_present {
        "INCONCLUSIVE — uniform 4-bit already matches baseline recall on this dataset; no precision-loss signal for coherence-adaptive allocation to fix"
    } else if recall_recovered && memory_compressed {
        "ACCEPT — coherence-adaptive bit allocation recovers baseline recall at compressed memory"
    } else {
        "REJECT — coherence-adaptive bit allocation does not clear both thresholds"
    };
    println!("VERDICT: {verdict}");

    println!();
    println!("─── Memory Summary ───");
    println!(
        "  baseline (8-bit uniform)   : {} KB ({:.2} bits/dim avg)",
        baseline.mem_bytes / 1024,
        baseline.mean_bits
    );
    println!(
        "  candidate A (4-bit uniform): {} KB ({:.2} bits/dim avg)",
        cand_a.mem_bytes / 1024,
        cand_a.mean_bits
    );
    println!(
        "  candidate B (adaptive)     : {} KB ({:.2} bits/dim avg)",
        cand_b.mem_bytes / 1024,
        cand_b.mean_bits
    );
    println!();
    println!("Coherence graph build overhead: {coherence_build_ms}ms (one-time, amortized at index build)");

    if verdict.starts_with("REJECT") {
        std::process::exit(1);
    }
}
