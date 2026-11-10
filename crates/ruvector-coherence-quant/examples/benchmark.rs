//! Nightly research benchmark for ADR-353 / `docs/research/nightly/2026-10-24-coherence-gated-quant`.
//!
//! Run with: `cargo run --release -p ruvector-coherence-quant --example benchmark`
//!
//! Hypothesis: given a clustered agent-memory corpus, allocating Turbo4 vs.
//! 1-bit quantization precision per vector by a mincut-derived local
//! coherence score (candidate_A) beats uniform Turbo4 (baseline) on
//! recall@10-per-byte, without more than a modest absolute recall drop or a
//! latency regression. candidate_B additionally exact-caches the most
//! incoherent outliers.
//!
//! This binary prints only numbers it actually measured this run — no
//! numbers are hand-edited into the research doc; they are copy-pasted from
//! this program's stdout.

use ruvector_attn_mincut::MinCutConfig;
use ruvector_coherence_quant::{allocator, coherence, dataset, metrics};
use ruvector_turboquant::{Bits1Query, Metric, Turbo4Codec};
use std::time::Instant;

const DIM: usize = 128;
const N_CLUSTERS: usize = 40;
const PER_CLUSTER: usize = 150;
const N_QUERIES: usize = 200;
const K: usize = 10;
const HIGH_PRECISION_FRAC: f32 = 0.30;
const RESIDUAL_FRAC: f32 = 0.05;
const CORPUS_SEED: u64 = 42;
const QUERY_SEED: u64 = 99;
const ROTATION_SEED: u64 = 7;

fn brute_force_topk(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut d: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let sq: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (i, sq)
        })
        .collect();
    d.sort_by(|a, b| a.1.total_cmp(&b.1));
    d.into_iter().take(k).map(|(i, _)| i).collect()
}

struct VariantResult {
    name: &'static str,
    mean_bytes: f64,
    turbo4_frac: f64,
    bit1_frac: f64,
    exact_cached_frac: f64,
    encode_ms: f64,
    recall_at_10: f32,
    mean_query_us: f64,
    p50_query_us: f64,
    p95_query_us: f64,
}

fn run_variant(
    name: &'static str,
    variant: allocator::Variant,
    corpus: &dataset::Corpus,
    coherence_scores: &[f32],
    codec: &Turbo4Codec,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
) -> VariantResult {
    let t0 = Instant::now();
    let encoded = allocator::encode_corpus(
        &corpus.vectors,
        coherence_scores,
        &corpus.cluster_id,
        corpus.centroids.len(),
        variant,
        codec,
        HIGH_PRECISION_FRAC,
        RESIDUAL_FRAC,
    );
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let total_bytes: usize = encoded.iter().map(|e| e.bytes()).sum();
    let mean_bytes = total_bytes as f64 / encoded.len() as f64;
    let n = encoded.len() as f64;
    let turbo4_frac = encoded.iter().filter(|e| e.turbo4.is_some()).count() as f64 / n;
    let bit1_frac = encoded.iter().filter(|e| e.bits1.is_some()).count() as f64 / n;
    let exact_cached_frac = encoded.iter().filter(|e| e.exact.is_some()).count() as f64 / n;

    let mut hits = 0f32;
    let mut lat_us = Vec::with_capacity(queries.len());
    for (qi, q) in queries.iter().enumerate() {
        let qt0 = Instant::now();
        let tq = codec.encode_query(q).expect("dim");
        let bq = Bits1Query::new(&tq.rotated).expect("dim");
        let mut d: Vec<(usize, f32)> = encoded
            .iter()
            .enumerate()
            .map(|(i, e)| {
                (
                    i,
                    allocator::estimate_distance(Metric::Euclidean, q, &tq, &bq, e, DIM),
                )
            })
            .collect();
        d.sort_by(|a, b| a.1.total_cmp(&b.1));
        let approx_top: Vec<usize> = d.into_iter().take(K).map(|(i, _)| i).collect();
        lat_us.push(qt0.elapsed().as_nanos() as u64 / 1000);
        hits += metrics::recall_at_k(&approx_top, &ground_truth[qi]);
    }
    let lat_ns: Vec<u64> = lat_us.iter().map(|&us| us * 1000).collect();

    VariantResult {
        name,
        mean_bytes,
        turbo4_frac,
        bit1_frac,
        exact_cached_frac,
        encode_ms,
        recall_at_10: hits / queries.len() as f32,
        mean_query_us: lat_us.iter().sum::<u64>() as f64 / lat_us.len() as f64,
        p50_query_us: metrics::percentile_ns(&lat_ns, 0.50) as f64 / 1000.0,
        p95_query_us: metrics::percentile_ns(&lat_ns, 0.95) as f64 / 1000.0,
    }
}

fn main() {
    println!("ruvector-coherence-quant nightly benchmark (ADR-353)");
    println!(
        "corpus: {N_CLUSTERS} clusters x {PER_CLUSTER} = {} vectors, dim={DIM}, seed={CORPUS_SEED}",
        N_CLUSTERS * PER_CLUSTER
    );
    println!("queries: {N_QUERIES}, seed={QUERY_SEED}, k={K}");
    println!(
        "high_precision_frac={HIGH_PRECISION_FRAC}, residual_frac={RESIDUAL_FRAC}, rotation_seed={ROTATION_SEED}"
    );

    let corpus = dataset::generate_corpus(N_CLUSTERS, PER_CLUSTER, DIM, CORPUS_SEED);
    let queries = dataset::generate_queries(&corpus, N_QUERIES, QUERY_SEED);

    let t_coh = Instant::now();
    let coherence_scores = coherence::corpus_coherence(
        &corpus.vectors,
        &corpus.cluster_id,
        &corpus.centroids,
        &MinCutConfig::default(),
    );
    println!(
        "coherence scoring (mincut over {N_CLUSTERS} blocks): {:.1} ms",
        t_coh.elapsed().as_secs_f64() * 1000.0
    );
    let mean_coh: f32 = coherence_scores.iter().sum::<f32>() / coherence_scores.len() as f32;
    println!("mean coherence score: {mean_coh:.4}");

    let codec = Turbo4Codec::new(DIM, ROTATION_SEED).expect("valid dim");

    println!("computing ground truth (exact brute-force L2 over full corpus)...");
    let t_gt = Instant::now();
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_topk(&corpus.vectors, q, K))
        .collect();
    println!(
        "ground truth: {:.1} ms",
        t_gt.elapsed().as_secs_f64() * 1000.0
    );

    let variants = [
        ("baseline (uniform Turbo4)", allocator::Variant::Baseline),
        (
            "candidate_A (coherence-gated)",
            allocator::Variant::CandidateA,
        ),
        (
            "candidate_B (coherence-gated + residual)",
            allocator::Variant::CandidateB,
        ),
    ];

    let mut results = Vec::new();
    for (name, variant) in variants {
        let r = run_variant(
            name,
            variant,
            &corpus,
            &coherence_scores,
            &codec,
            &queries,
            &ground_truth,
        );
        results.push(r);
    }

    println!();
    println!(
        "{:<42} {:>10} {:>8} {:>8} {:>9} {:>10} {:>12} {:>10} {:>10}",
        "variant",
        "bytes/vec",
        "t4_frac",
        "b1_frac",
        "exact_frac",
        "encode_ms",
        "recall@10",
        "mean_us",
        "p95_us"
    );
    for r in &results {
        println!(
            "{:<42} {:>10.1} {:>8.2} {:>8.2} {:>9.2} {:>10.1} {:>12.3} {:>10.1} {:>10.1}",
            r.name,
            r.mean_bytes,
            r.turbo4_frac,
            r.bit1_frac,
            r.exact_cached_frac,
            r.encode_ms,
            r.recall_at_10,
            r.mean_query_us,
            r.p95_query_us
        );
    }

    let baseline = &results[0];
    let a = &results[1];
    let b = &results[2];

    let rpb = |r: &VariantResult| r.recall_at_10 as f64 / r.mean_bytes;
    let rpb_baseline = rpb(baseline);
    let rpb_a = rpb(a);
    let rpb_b = rpb(b);
    let rpb_gain_a = (rpb_a / rpb_baseline - 1.0) * 100.0;
    let rpb_gain_b = (rpb_b / rpb_baseline - 1.0) * 100.0;

    println!();
    println!("recall@10 per byte: baseline={rpb_baseline:.6} candidate_A={rpb_a:.6} ({rpb_gain_a:+.1}%) candidate_B={rpb_b:.6} ({rpb_gain_b:+.1}%)");

    // Acceptance thresholds, fixed before this run per the research doc:
    // candidate_A must beat baseline recall-per-byte by >=15%, keep absolute
    // recall@10 >= 0.9x baseline, and not regress mean query latency by
    // more than 2x.
    let recall_floor_ok = a.recall_at_10 >= 0.9 * baseline.recall_at_10;
    let rpb_ok = rpb_gain_a >= 15.0;
    let latency_ok = a.mean_query_us <= 2.0 * baseline.mean_query_us;

    println!();
    println!("acceptance checks (candidate_A vs baseline):");
    println!(
        "  recall@10 >= 0.9x baseline: {} ({:.3} vs {:.3})",
        recall_floor_ok, a.recall_at_10, baseline.recall_at_10
    );
    println!(
        "  recall-per-byte gain >= 15%: {} ({:+.1}%)",
        rpb_ok, rpb_gain_a
    );
    println!(
        "  mean query latency <= 2x baseline: {} ({:.1}us vs {:.1}us)",
        latency_ok, a.mean_query_us, baseline.mean_query_us
    );

    let verdict = if recall_floor_ok && rpb_ok && latency_ok {
        "ACCEPT"
    } else if !recall_floor_ok {
        "REJECT"
    } else {
        "INCONCLUSIVE"
    };
    println!();
    println!("ACCEPTANCE RESULT: {verdict}");
}
