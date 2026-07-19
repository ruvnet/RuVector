//! Adaptive semantic tiering benchmark.
//!
//! Measures three tiering strategies on a 5,000-vector, 64-dim workload:
//!
//! * AccessOnly   — tier by access count (baseline)
//! * Coherence    — tier by intra-cluster L2 coherence
//! * SemanticTemp — tier by recency + coherence + centrality
//!
//! Workload: 200 warmup queries targeting the Noise cluster, then 500 eval
//! queries targeting the Important cluster.  The key metric is hot-tier hit
//! rate for the eval phase — higher means better placement quality.

use std::time::Instant;

use ruvector_adaptive_tiering::{
    dataset::{cluster_query, generate_clustered, seeded_rng},
    store::TieredStore,
    AccessOnlyScorer, BatchStats, CoherenceScorer, Scorer, SemanticTempScorer, TieringConfig,
};

const N_VECS: usize = 5_000;
const DIMS: usize = 64;
const HOT_CAP: usize = 500; // 10 % of N_VECS
const WARM_CAP: usize = 1_500; // 30 % of N_VECS
const K: usize = 10;
const WARMUP_QUERIES: usize = 200;
const EVAL_QUERIES: usize = 500;
const WARMUP_EPOCH: u64 = WARMUP_QUERIES as u64;

// ─── Benchmark driver ────────────────────────────────────────────────────────

struct RunResult {
    variant: &'static str,
    hot_hit_rate: f64,
    hot_warm_hit_rate: f64,
    hot_count: usize,
    warm_count: usize,
    cold_count: usize,
    mean_ns: u64,
    p50_ns: u64,
    p95_ns: u64,
    throughput_qps: f64,
}

fn run_variant<S: Scorer>(
    scorer: S,
    vecs: &[Vec<f32>],
    labels: &[u8],
    cfg: &TieringConfig,
) -> RunResult {
    let variant = scorer.name();
    let mut store = TieredStore::new(cfg.clone(), scorer);

    // Insert all vectors
    for (id, v) in vecs.iter().enumerate() {
        store.insert(id as u64, v.clone());
    }

    // Warmup: queries targeting the Noise cluster (label 2) to bias access counts
    let noise_q = cluster_query(2, DIMS);
    for epoch in 0..WARMUP_QUERIES {
        let results = store.search(&noise_q, K);
        for r in &results {
            store.record_access(r.id, epoch as u64 + 1);
        }
    }

    // Tier evaluation after warmup
    store.evaluate_tiers(WARMUP_EPOCH);

    let (hot, warm, cold) = store.tier_counts();

    // Evaluation: queries targeting the Important cluster (label 0)
    let eval_q = cluster_query(0, DIMS);
    let mut stats = BatchStats::default();
    let mut latencies_ns: Vec<u64> = Vec::with_capacity(EVAL_QUERIES);

    for _ in 0..EVAL_QUERIES {
        let t0 = Instant::now();
        let results = store.search(&eval_q, K);
        let elapsed = t0.elapsed().as_nanos() as u64;
        latencies_ns.push(elapsed);
        store.accumulate_stats(&results, &mut stats);
    }

    latencies_ns.sort_unstable();
    let mean_ns = latencies_ns.iter().sum::<u64>() / latencies_ns.len() as u64;
    let p50_ns = latencies_ns[latencies_ns.len() * 50 / 100];
    let p95_ns = latencies_ns[latencies_ns.len() * 95 / 100];
    let total_s = latencies_ns.iter().sum::<u64>() as f64 / 1e9;
    let throughput_qps = EVAL_QUERIES as f64 / total_s;

    RunResult {
        variant,
        hot_hit_rate: stats.hot_hit_rate(),
        hot_warm_hit_rate: stats.hot_warm_hit_rate(),
        hot_count: hot,
        warm_count: warm,
        cold_count: cold,
        mean_ns,
        p50_ns,
        p95_ns,
        throughput_qps,
    }
}

// ─── Recall@K (brute-force ground truth) ────────────────────────────────────

/// Computes fraction of true top-k ids found in the result set.
fn recall_at_k(results: &[(u64, f32)], ground_truth: &[(u64, f32)], k: usize) -> f32 {
    let k = k.min(results.len()).min(ground_truth.len());
    let gt_ids: std::collections::HashSet<u64> = ground_truth.iter().take(k).map(|x| x.0).collect();
    let found = results
        .iter()
        .take(k)
        .filter(|r| gt_ids.contains(&r.0))
        .count();
    found as f32 / k as f32
}

fn brute_force_knn(query: &[f32], vecs: &[Vec<f32>], k: usize) -> Vec<(u64, f32)> {
    let mut dists: Vec<(u64, f32)> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (i as u64, d)
        })
        .collect();
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    // System info
    println!("═══ Adaptive Semantic Tiering Benchmark ═══");
    println!("OS:   {}", std::env::consts::OS);
    println!("Arch: {}", std::env::consts::ARCH);
    println!();
    println!("Dataset:    {N_VECS} vectors × {DIMS} dims");
    println!("  Cluster 0 (Important): {}  (tight σ=0.05)", N_VECS / 10);
    println!("  Cluster 1 (Moderate):  {}  (σ=0.25)", N_VECS * 3 / 10);
    println!(
        "  Cluster 2 (Noise):     {}  (sparse σ=1.20)",
        N_VECS - N_VECS / 10 - N_VECS * 3 / 10
    );
    println!(
        "Hot capacity:  {HOT_CAP} ({:.0}%)",
        100.0 * HOT_CAP as f64 / N_VECS as f64
    );
    println!(
        "Warm capacity: {WARM_CAP} ({:.0}%)",
        100.0 * WARM_CAP as f64 / N_VECS as f64
    );
    println!("Warmup queries (→ Noise): {WARMUP_QUERIES}");
    println!("Eval queries   (→ Important): {EVAL_QUERIES}");
    println!("k = {K}");
    println!();

    let mut rng = seeded_rng(0xdeadbeef);
    let (vecs, labels) = generate_clustered(N_VECS, DIMS, &mut rng);

    let cfg = TieringConfig {
        hot_capacity: HOT_CAP,
        warm_capacity: WARM_CAP,
        ..TieringConfig::default()
    };

    // Ground truth for Important cluster query
    let eval_q = cluster_query(0, DIMS);
    let gt = brute_force_knn(&eval_q, &vecs, K);

    let variants: Vec<RunResult> = vec![
        run_variant(AccessOnlyScorer, &vecs, &labels, &cfg),
        run_variant(CoherenceScorer, &vecs, &labels, &cfg),
        run_variant(SemanticTempScorer, &vecs, &labels, &cfg),
    ];

    // ── Tier placement table ──────────────────────────────────────────────
    println!("── Tier Placement After Warmup ──────────────────────────────────");
    println!("{:<15} {:>6} {:>6} {:>6}", "Variant", "Hot", "Warm", "Cold");
    println!("{}", "─".repeat(40));
    for r in &variants {
        println!(
            "{:<15} {:>6} {:>6} {:>6}",
            r.variant, r.hot_count, r.warm_count, r.cold_count
        );
    }
    println!();

    // ── Hit rate table ───────────────────────────────────────────────────
    println!("── Eval Hit Rates (eval_q → Important cluster) ─────────────────");
    println!(
        "{:<15} {:>12} {:>16}",
        "Variant", "Hot hit %", "Hot+Warm hit %"
    );
    println!("{}", "─".repeat(46));
    for r in &variants {
        println!(
            "{:<15} {:>11.1}% {:>15.1}%",
            r.variant,
            r.hot_hit_rate * 100.0,
            r.hot_warm_hit_rate * 100.0
        );
    }
    println!();

    // ── Latency table ────────────────────────────────────────────────────
    println!("── Latency (brute-force over all tiers, {EVAL_QUERIES} eval queries) ──");
    println!(
        "{:<15} {:>12} {:>12} {:>12} {:>14}",
        "Variant", "Mean ns", "p50 ns", "p95 ns", "Throughput QPS"
    );
    println!("{}", "─".repeat(68));
    for r in &variants {
        println!(
            "{:<15} {:>12} {:>12} {:>12} {:>14.0}",
            r.variant, r.mean_ns, r.p50_ns, r.p95_ns, r.throughput_qps
        );
    }
    println!();

    // ── Memory estimate ──────────────────────────────────────────────────
    let vec_bytes = N_VECS * DIMS * 4;
    let meta_bytes = N_VECS * 32; // rough: id(8) + meta(24)
    println!("── Memory Estimate ──────────────────────────────────────────────");
    println!("  Vectors:  {} MB", vec_bytes / 1_048_576 + 1);
    println!("  Metadata: {} KB", meta_bytes / 1024 + 1);
    println!("  Hot tier: {} KB", (HOT_CAP * DIMS * 4) / 1024 + 1);
    println!();

    // ── Recall@K (ground truth comparison) ──────────────────────────────
    println!("── Recall@{K} (ground truth over full dataset) ────────────────────");
    // Use the last eval variant (SemanticTemp) for a point-in-time recall check
    // by running one more search and comparing to GT
    let mut rng2 = seeded_rng(0xdeadbeef);
    let (vecs2, _) = generate_clustered(N_VECS, DIMS, &mut rng2);
    let cfg2 = TieringConfig {
        hot_capacity: HOT_CAP,
        warm_capacity: WARM_CAP,
        ..TieringConfig::default()
    };
    let mut store = TieredStore::new(cfg2, SemanticTempScorer);
    for (id, v) in vecs2.iter().enumerate() {
        store.insert(id as u64, v.clone());
    }
    store.evaluate_tiers(0);
    let results = store.search(&eval_q, K);
    let result_pairs: Vec<(u64, f32)> = results.iter().map(|r| (r.id, r.dist_sq)).collect();
    let recall = recall_at_k(&result_pairs, &gt, K);
    println!(
        "  SemanticTemp recall@{K} (cold start, no warmup): {:.1}%",
        recall * 100.0
    );
    println!("  (Full brute-force search — recall should equal 1.0 when k == K)");
    println!();

    // ── Acceptance test ──────────────────────────────────────────────────
    println!("── Acceptance Test ──────────────────────────────────────────────");

    let access_only = variants.iter().find(|r| r.variant == "AccessOnly").unwrap();
    let coherence = variants.iter().find(|r| r.variant == "Coherence").unwrap();
    let semantic = variants
        .iter()
        .find(|r| r.variant == "SemanticTemp")
        .unwrap();

    let mut passed = true;

    // Test 1: CoherenceScorer outperforms AccessOnly on cold-start hot-tier hit rate
    let coh_beats_ao = coherence.hot_hit_rate > access_only.hot_hit_rate;
    println!(
        "  [{}] CoherenceScorer hot hit rate ({:.1}%) > AccessOnly ({:.1}%)",
        if coh_beats_ao { "PASS" } else { "FAIL" },
        coherence.hot_hit_rate * 100.0,
        access_only.hot_hit_rate * 100.0
    );
    passed &= coh_beats_ao;

    // Test 2: SemanticTemp outperforms AccessOnly on cold-start hot-tier hit rate
    let sem_beats_ao = semantic.hot_hit_rate > access_only.hot_hit_rate;
    println!(
        "  [{}] SemanticTemp hot hit rate ({:.1}%) > AccessOnly ({:.1}%)",
        if sem_beats_ao { "PASS" } else { "FAIL" },
        semantic.hot_hit_rate * 100.0,
        access_only.hot_hit_rate * 100.0
    );
    passed &= sem_beats_ao;

    // Test 3: Hot tier never exceeds configured capacity
    let hot_capped = variants.iter().all(|r| r.hot_count <= HOT_CAP);
    println!(
        "  [{}] All variants: hot_count ≤ {HOT_CAP}",
        if hot_capped { "PASS" } else { "FAIL" }
    );
    passed &= hot_capped;

    // Test 4: Throughput > 100 QPS (brute-force baseline)
    let fast_enough = variants.iter().all(|r| r.throughput_qps > 100.0);
    println!(
        "  [{}] All variants throughput > 100 QPS (min: {:.0})",
        if fast_enough { "PASS" } else { "FAIL" },
        variants
            .iter()
            .map(|r| r.throughput_qps)
            .fold(f64::INFINITY, f64::min)
    );
    passed &= fast_enough;

    // Test 5: SemanticTemp or Coherence achieves ≥ 50% hot-tier hit rate on eval queries
    let quality_ok = semantic.hot_hit_rate >= 0.50 || coherence.hot_hit_rate >= 0.50;
    println!(
        "  [{}] SemanticTemp or Coherence hot hit rate ≥ 50% (sem={:.1}%, coh={:.1}%)",
        if quality_ok { "PASS" } else { "FAIL" },
        semantic.hot_hit_rate * 100.0,
        coherence.hot_hit_rate * 100.0
    );
    passed &= quality_ok;

    println!();
    if passed {
        println!("  ✓ ALL ACCEPTANCE TESTS PASSED");
    } else {
        println!("  ✗ ONE OR MORE ACCEPTANCE TESTS FAILED");
        std::process::exit(1);
    }
}
