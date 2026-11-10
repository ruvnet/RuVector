//! Benchmark: coherence-gated ReasoningBank pattern retrieval vs. cosine-only
//! baseline vs. a quality-weighted ablation.
//!
//! See `docs/research/nightly/2026_10_31_coherence_gated_reasoning_bank/README.md`
//! for the fixed hypothesis and acceptance thresholds this benchmark was built
//! to test, and `raw_benchmark_output.txt` in that directory for the verbatim
//! captured output of this exact run.
//!
//! Run: `cargo run -p ruvector-sona --release --example coherence_gate_bench`

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_sona::types::LearnedPattern;
use ruvector_sona::{find_similar_coherence_gated, find_similar_quality_weighted, PatternCoherenceGraph};
use std::time::Instant;

const DIMS: usize = 64;
const NUM_CLUSTERS: usize = 8;
const MEMBERS_PER_CLUSTER: usize = 6;
const QUERIES_PER_SEED: usize = 10;
const SEEDS: &[u64] = &[1, 2, 3];
const EDGE_THRESHOLD: f32 = 0.90;
const ALPHA: f32 = 0.5;
const TOP_K: usize = 10;
const SIGMA_MEMBER: f32 = 0.05;
const SIGMA_QUERY: f32 = 0.45;
const SIGMA_ADV: f32 = 0.03;

/// A corpus pattern plus ground truth: `Some(cluster_id)` for a true,
/// trajectory-cluster-derived pattern; `None` for plain noise or an
/// adversarial near-duplicate (neither is ever a "relevant" result for any
/// query).
struct Labeled {
    pattern: LearnedPattern,
    cluster: Option<usize>,
}

fn random_unit(rng: &mut StdRng, dims: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    normalize(v)
}

fn perturb(rng: &mut StdRng, base: &[f32], sigma: f32) -> Vec<f32> {
    let v: Vec<f32> = base
        .iter()
        .map(|&b| b + rng.gen_range(-1.0f32..1.0) * sigma)
        .collect();
    normalize(v)
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 1e-8 && nb > 1e-8 {
        dot / (na * nb)
    } else {
        0.0
    }
}

/// Baseline: the same algorithm as `ReasoningBank::find_similar` (rank all
/// patterns by raw cosine similarity, descending, take top-k), applied
/// directly to a pattern slice for apples-to-apples timing against the
/// coherence-gated and quality-weighted candidates.
fn find_similar_baseline(patterns: &[LearnedPattern], query: &[f32], k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = patterns
        .iter()
        .map(|p| (p.id, p.similarity(query)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

struct Corpus {
    labeled: Vec<Labeled>,
    queries: Vec<(Vec<f32>, usize)>, // (query vector, source cluster id)
    adv_ids: Vec<u64>,               // adv_ids[i] is the decoy injected for queries[i]
}

fn build_corpus(seed: u64, target_total_patterns: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut labeled = Vec::new();
    let mut next_id = 0u64;

    let centroids: Vec<Vec<f32>> = (0..NUM_CLUSTERS)
        .map(|_| random_unit(&mut rng, DIMS))
        .collect();

    for (cluster_id, centroid) in centroids.iter().enumerate() {
        for _ in 0..MEMBERS_PER_CLUSTER {
            let centroid_vec = perturb(&mut rng, centroid, SIGMA_MEMBER);
            let mut p = LearnedPattern::new(next_id, centroid_vec);
            next_id += 1;
            p.cluster_size = MEMBERS_PER_CLUSTER;
            p.avg_quality = rng.gen_range(0.3f32..0.9); // label-uninformative
            p.access_count = rng.gen_range(0u32..20);
            labeled.push(Labeled {
                pattern: p,
                cluster: Some(cluster_id),
            });
        }
    }

    // Queries: one per trial, drawn near a (cyclically chosen) cluster
    // centroid but not identical to any stored member.
    let mut queries = Vec::new();
    for i in 0..QUERIES_PER_SEED {
        let cluster_id = i % NUM_CLUSTERS;
        let query = perturb(&mut rng, &centroids[cluster_id], SIGMA_QUERY);
        queries.push((query, cluster_id));
    }

    // Adversarial near-duplicates: one per query, isolated from every true
    // cluster (see module docs / research write-up for why sigma_adv <<
    // sigma_query keeps it a near-perfect cosine match for its query while
    // the calibration below keeps it below edge_threshold from real
    // cluster members).
    let mut adv_ids = Vec::with_capacity(queries.len());
    for (query, _cluster_id) in &queries {
        let adv_vec = perturb(&mut rng, query, SIGMA_ADV);
        let mut p = LearnedPattern::new(next_id, adv_vec);
        let adv_id = next_id;
        next_id += 1;
        p.cluster_size = 1;
        p.avg_quality = rng.gen_range(0.3f32..0.9);
        p.access_count = rng.gen_range(0u32..20);
        labeled.push(Labeled {
            pattern: p,
            cluster: None,
        });
        adv_ids.push(adv_id);
    }

    // Pad with plain, uncorrelated noise singletons to reach the target
    // bank size for this scale point.
    let true_and_adv = labeled.len();
    let noise_count = target_total_patterns.saturating_sub(true_and_adv);
    for _ in 0..noise_count {
        let v = random_unit(&mut rng, DIMS);
        let mut p = LearnedPattern::new(next_id, v);
        next_id += 1;
        p.cluster_size = 1;
        p.avg_quality = rng.gen_range(0.3f32..0.9);
        p.access_count = rng.gen_range(0u32..20);
        labeled.push(Labeled {
            pattern: p,
            cluster: None,
        });
    }

    Corpus {
        labeled,
        queries,
        adv_ids,
    }
}

fn precision_at_k(results: &[(u64, f32)], cluster_lookup: &std::collections::HashMap<u64, Option<usize>>, want_cluster: usize) -> f32 {
    if results.is_empty() {
        return 0.0;
    }
    let hits = results
        .iter()
        .filter(|(id, _)| cluster_lookup.get(id).copied().flatten() == Some(want_cluster))
        .count();
    hits as f32 / results.len() as f32
}

#[derive(Default)]
struct LatencyStats {
    samples_us: Vec<f64>,
}

impl LatencyStats {
    fn record(&mut self, d: std::time::Duration) {
        self.samples_us.push(d.as_secs_f64() * 1e6);
    }

    fn summary(&mut self) -> (f64, f64, f64) {
        self.samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = self.samples_us.len();
        let mean = self.samples_us.iter().sum::<f64>() / n as f64;
        let p50 = self.samples_us[n / 2];
        let p95 = self.samples_us[(n * 95 / 100).min(n - 1)];
        (mean, p50, p95)
    }
}

fn run_scale(bank_size: usize) {
    println!("\n=== bank_size={bank_size} edge_threshold={EDGE_THRESHOLD} alpha={ALPHA} top_k={TOP_K} ===");

    let mut precision_baseline = Vec::new();
    let mut precision_gated = Vec::new();
    let mut precision_quality = Vec::new();

    let mut lat_baseline = LatencyStats::default();
    let mut lat_gated_query_only = LatencyStats::default();
    let mut lat_quality = LatencyStats::default();
    let mut graph_build_us = Vec::new();

    // Decoy-specific diagnostic: did the adversarial near-duplicate make it
    // into the baseline's top-k, and if so did gating push it out?
    let mut adv_in_baseline_top_k = 0usize;
    let mut adv_in_gated_top_k = 0usize;
    let mut adv_demoted_by_gating = 0usize;
    let mut total_trials = 0usize;

    for &seed in SEEDS {
        let corpus = build_corpus(seed, bank_size);
        let patterns: Vec<LearnedPattern> = corpus.labeled.iter().map(|l| l.pattern.clone()).collect();
        let cluster_lookup: std::collections::HashMap<u64, Option<usize>> = corpus
            .labeled
            .iter()
            .map(|l| (l.pattern.id, l.cluster))
            .collect();

        // Graph build: amortized once per corpus change, not per query.
        let t0 = Instant::now();
        let graph = PatternCoherenceGraph::build(&patterns, EDGE_THRESHOLD);
        graph_build_us.push(t0.elapsed().as_secs_f64() * 1e6);

        // Warmup (JIT/branch predictor/caches), not counted.
        for (q, _) in corpus.queries.iter().take(3) {
            let _ = find_similar_baseline(&patterns, q, TOP_K);
            let _ = find_similar_coherence_gated(&patterns, &graph, q, TOP_K, ALPHA);
            let _ = find_similar_quality_weighted(&patterns, q, TOP_K, ALPHA);
        }

        for (i, (query, cluster_id)) in corpus.queries.iter().enumerate() {
            let t0 = Instant::now();
            let r_base = find_similar_baseline(&patterns, query, TOP_K);
            lat_baseline.record(t0.elapsed());

            let t0 = Instant::now();
            let r_gated = find_similar_coherence_gated(&patterns, &graph, query, TOP_K, ALPHA);
            lat_gated_query_only.record(t0.elapsed());

            let t0 = Instant::now();
            let r_qual = find_similar_quality_weighted(&patterns, query, TOP_K, ALPHA);
            lat_quality.record(t0.elapsed());

            precision_baseline.push(precision_at_k(&r_base, &cluster_lookup, *cluster_id));
            precision_gated.push(precision_at_k(&r_gated, &cluster_lookup, *cluster_id));
            precision_quality.push(precision_at_k(&r_qual, &cluster_lookup, *cluster_id));

            let adv_id = corpus.adv_ids[i];
            let in_base = r_base.iter().any(|(id, _)| *id == adv_id);
            let in_gated = r_gated.iter().any(|(id, _)| *id == adv_id);
            total_trials += 1;
            if in_base {
                adv_in_baseline_top_k += 1;
            }
            if in_gated {
                adv_in_gated_top_k += 1;
            }
            if in_base && !in_gated {
                adv_demoted_by_gating += 1;
            }
        }
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (mb, p50b, p95b) = lat_baseline.summary();
    let (mg, p50g, p95g) = lat_gated_query_only.summary();
    let (mq, p50q, p95q) = lat_quality.summary();
    let graph_build_mean = graph_build_us.iter().sum::<f64>() / graph_build_us.len() as f64;

    println!(
        "precision@{TOP_K}  baseline_mean={:.4}  coherence_gated_mean={:.4}  quality_weighted_mean={:.4}",
        mean(&precision_baseline),
        mean(&precision_gated),
        mean(&precision_quality)
    );
    println!(
        "precision_lift_pp  gated_vs_baseline={:+.2}  quality_vs_baseline={:+.2}",
        (mean(&precision_gated) - mean(&precision_baseline)) * 100.0,
        (mean(&precision_quality) - mean(&precision_baseline)) * 100.0
    );
    println!(
        "latency_us(query-only)  baseline  mean={mb:.2} p50={p50b:.2} p95={p95b:.2}"
    );
    println!(
        "latency_us(query-only)  coherence_gated  mean={mg:.2} p50={p50g:.2} p95={p95g:.2}  (ratio_vs_baseline_mean={:.3}x)",
        mg / mb
    );
    println!(
        "latency_us(query-only)  quality_weighted  mean={mq:.2} p50={p50q:.2} p95={p95q:.2}  (ratio_vs_baseline_mean={:.3}x)",
        mq / mb
    );
    println!("graph_build_us(amortized once per corpus, n={bank_size})  mean={graph_build_mean:.2}");
    println!(
        "decoy_diagnostic  trials={total_trials}  adv_in_baseline_top{TOP_K}={adv_in_baseline_top_k} \
         adv_in_gated_top{TOP_K}={adv_in_gated_top_k}  adv_demoted_by_gating={adv_demoted_by_gating}  \
         max_possible_precision_lift_pp={:.2}",
        (adv_in_baseline_top_k as f32 / total_trials as f32) * 100.0 / TOP_K as f32
    );
}

fn main() {
    println!("coherence-gated ReasoningBank retrieval benchmark");
    println!(
        "config: dims={DIMS} clusters={NUM_CLUSTERS} members_per_cluster={MEMBERS_PER_CLUSTER} \
         queries_per_seed={QUERIES_PER_SEED} seeds={SEEDS:?} sigma_member={SIGMA_MEMBER} \
         sigma_query={SIGMA_QUERY} sigma_adv={SIGMA_ADV}"
    );

    // Calibration diagnostic (printed, not used to tune the hypothesis
    // after the fact): confirms the embedding geometry actually separates
    // "corroborated cluster member" similarity from "isolated adversarial
    // near-duplicate" similarity at the fixed edge_threshold.
    {
        let mut rng = StdRng::seed_from_u64(42);
        let centroid = random_unit(&mut rng, DIMS);
        let members: Vec<Vec<f32>> = (0..MEMBERS_PER_CLUSTER)
            .map(|_| perturb(&mut rng, &centroid, SIGMA_MEMBER))
            .collect();
        let query = perturb(&mut rng, &centroid, SIGMA_QUERY);
        let adv = perturb(&mut rng, &query, SIGMA_ADV);

        let intra_cluster_sim: f32 = {
            let mut s = 0.0;
            let mut n = 0;
            for i in 0..members.len() {
                for j in (i + 1)..members.len() {
                    s += cosine(&members[i], &members[j]);
                    n += 1;
                }
            }
            s / n as f32
        };
        let adv_to_member_sim: f32 =
            members.iter().map(|m| cosine(&adv, m)).sum::<f32>() / members.len() as f32;
        let query_to_adv_sim = cosine(&query, &adv);

        println!(
            "calibration: intra_cluster_sim={intra_cluster_sim:.4} adv_to_member_sim={adv_to_member_sim:.4} \
             query_to_adv_sim={query_to_adv_sim:.4} edge_threshold={EDGE_THRESHOLD}"
        );
        println!(
            "calibration_check: cluster_above_threshold={} adv_below_threshold={}",
            intra_cluster_sim >= EDGE_THRESHOLD,
            adv_to_member_sim < EDGE_THRESHOLD
        );
    }

    for &n in &[200usize, 1000, 2000, 5000] {
        run_scale(n);
    }
}
