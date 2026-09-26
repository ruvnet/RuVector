//! Nightly research benchmark (2026-09-06): structural-time recency for agent
//! memory compaction.
//!
//! Hypothesis (fixed before this run; see
//! docs/research/nightly/2026-09-06-structural-time-agent-memory/README.md):
//!
//! Given a synthetic agent memory stream of 40 "regime-shift" events (fresh,
//! mutually-unrelated 32-dim topic vectors) each immediately followed by 8
//! near-duplicate "churn" memories (small perturbations of the same vector,
//! simulating redundant retries/re-observations that rack up logical-clock
//! ticks without carrying new information), for a 360-entry store,
//!
//! when the store is aggressively compacted to exactly 40 entries (Experiment
//! 1, recency-only weights, no context window — isolating the recency signal)
//! using CoherencePolicy (baseline, per-event tick recency),
//! DedupGatedRecency (candidate A, fair cheap competitor: ticks only on
//! non-duplicate transitions), StructuralTimeRecency (candidate B1,
//! emergent-time's Structural Proper Time clock turned into a `[0,1]` recency
//! *score*), and StructuralKeyframeRetention (candidate B2, the same clock
//! used via emergent-time's own `keyframes()` budget-sampling primitive
//! instead of a rank score),
//!
//! then candidate B2's regime-shift-memory survival rate should exceed
//! baseline's by at least 15 percentage points,
//!
//! subject to: (a) Experiment 2 (50% compaction, production default weights,
//! a 5-vector context window of the most recent regime centroids) shows no
//! more than a 3pp Recall@10 regression for any candidate relative to
//! baseline; (b) every candidate's compaction wall-clock stays within 20x
//! baseline's; (c) candidate B2's survivor selection is deterministic across
//! repeated runs with the same seed.
//!
//! Candidate B1 is included and reported even though a first exploratory run
//! showed it does not beat the baseline (see `structural_recency`'s module
//! docs for why: cumulative structural time is still monotone non-decreasing
//! in insertion order, so ranking by it barely changes the top-K set versus
//! ranking by raw tick count). That negative result is *why* B2 exists — it
//! is retained here as evidence, not deleted once B2 was found.
//!
//! This benchmark also reports each structural candidate vs candidate A (the
//! fair, cheap, emergent-time-free competitor) as a first-class result, not a
//! footnote: per `emergent-time::agentic_time`'s own discipline, a
//! physics-flavoured clock that only beats the wall-clock null hypothesis and
//! not a fair cheap baseline has not demonstrated an edge worth its extra
//! dependency.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example structural_time_recency_bench --features structural-time

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, CoherencePolicy, CoherenceWeights, CompactionPolicy, DedupGatedRecency,
    DedupGatedWeights, MemoryStore, StructuralKeyframeRetention, StructuralTimeRecency,
    StructuralTimeWeights,
};
use std::time::{Duration, Instant};

// ── Dataset parameters ──────────────────────────────────────────────────────
const N_SEGMENTS: usize = 40;
const CHURN_PER_SEGMENT: usize = 8;
const N_MEMORIES: usize = N_SEGMENTS * (1 + CHURN_PER_SEGMENT); // 360
const DIMS: usize = 32;
const CHURN_NOISE: f32 = 0.05;

const TARGET_SIZE_H1: usize = N_SEGMENTS; // Experiment 1: aggressive compaction
const TARGET_SIZE_H2: usize = N_MEMORIES / 2; // Experiment 2: 50% compaction
const CONTEXT_WINDOW_SIZE: usize = 5;
const N_QUERIES: usize = 20;
const K: usize = 10;

const DEDUP_THRESHOLD: f32 = 0.02;

// Acceptance thresholds, fixed before running.
const SURVIVAL_GAP_THRESHOLD_PP: f32 = 15.0;
const RECALL_TOLERANCE_PP: f32 = 3.0;
const MAX_SLOWDOWN: f64 = 20.0;

fn unit_gaussian(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    normalize_vec(&v)
}

fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}

fn perturb(centroid: &[f32], noise: f32, rng: &mut StdRng) -> Vec<f32> {
    let n = unit_gaussian(rng, centroid.len());
    let combined: Vec<f32> = centroid
        .iter()
        .zip(n.iter())
        .map(|(c, x)| c + noise * x)
        .collect();
    normalize_vec(&combined)
}

/// A generated dataset: the store plus bookkeeping needed to score it.
struct Dataset {
    store: MemoryStore,
    /// Entry id of each segment's regime-shift memory (ground truth "signal").
    regime_shift_ids: Vec<u64>,
    /// Each segment's regime centroid vector.
    centroids: Vec<Vec<f32>>,
}

/// Generate the churn/regime-shift trace deterministically from `seed`.
fn generate_dataset(seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut store = MemoryStore::new(DIMS);
    let mut regime_shift_ids = Vec::with_capacity(N_SEGMENTS);
    let mut centroids = Vec::with_capacity(N_SEGMENTS);

    for _ in 0..N_SEGMENTS {
        let regime_vec = unit_gaussian(&mut rng, DIMS);
        let id = store.insert(regime_vec.clone());
        regime_shift_ids.push(id);
        centroids.push(regime_vec.clone());
        for _ in 0..CHURN_PER_SEGMENT {
            store.insert(perturb(&regime_vec, CHURN_NOISE, &mut rng));
        }
    }

    Dataset {
        store,
        regime_shift_ids,
        centroids,
    }
}

/// Regime-shift survival rate: fraction of `regime_shift_ids` still present
/// in `store` after compaction.
fn survival_rate(store: &MemoryStore, regime_shift_ids: &[u64]) -> f32 {
    let present: std::collections::HashSet<u64> = store.entries().iter().map(|e| e.id).collect();
    let hits = regime_shift_ids
        .iter()
        .filter(|id| present.contains(id))
        .count();
    hits as f32 / regime_shift_ids.len() as f32
}

/// Run one policy on a fresh copy of the Experiment-1 dataset (recency-only
/// weights, no context, aggressive compaction). Returns (survival_rate, compaction_time).
fn run_experiment1(policy: &dyn CompactionPolicy, seed: u64) -> (f32, Duration, Vec<u64>) {
    let dataset = generate_dataset(seed);
    let mut store = dataset.store;
    assert_eq!(store.len(), N_MEMORIES);

    let t0 = Instant::now();
    compact(&mut store, policy, TARGET_SIZE_H1, &[]);
    let elapsed = t0.elapsed();

    assert_eq!(store.len(), TARGET_SIZE_H1);
    let rate = survival_rate(&store, &dataset.regime_shift_ids);
    let mut survivor_ids: Vec<u64> = store.entries().iter().map(|e| e.id).collect();
    survivor_ids.sort_unstable();
    (rate, elapsed, survivor_ids)
}

/// Run one policy on a fresh copy of the Experiment-2 dataset (production
/// weights, context window, 50% compaction). Returns (recall@10, compaction_time).
fn run_experiment2(policy: &dyn CompactionPolicy, seed: u64) -> (f32, Duration) {
    let dataset = generate_dataset(seed);
    let mut store = dataset.store;

    // Context window: the most recent segments' regime centroids.
    let start = dataset.centroids.len().saturating_sub(CONTEXT_WINDOW_SIZE);
    let context_window: Vec<Vec<f32>> = dataset.centroids[start..].to_vec();

    // Test queries: perturbations of the last few centroids (the agent's
    // "current focus"), ground truth computed against the pre-compaction store.
    let mut rng = StdRng::seed_from_u64(seed + 7);
    let mut queries: Vec<(Vec<f32>, Vec<u64>)> = Vec::with_capacity(N_QUERIES);
    for i in 0..N_QUERIES {
        let c = &dataset.centroids[start + (i % context_window.len())];
        let q = perturb(c, 0.15, &mut rng);
        let truth: Vec<u64> = store.search(&q, K).into_iter().map(|r| r.id).collect();
        queries.push((q, truth));
    }

    let t0 = Instant::now();
    compact(&mut store, policy, TARGET_SIZE_H2, &context_window);
    let elapsed = t0.elapsed();
    assert_eq!(store.len(), TARGET_SIZE_H2);

    let mut total = 0.0f32;
    for (q, truth) in &queries {
        let candidates: Vec<u64> = store.search(q, K).into_iter().map(|r| r.id).collect();
        total += recall_at_k(truth, &candidates);
    }
    (total / queries.len() as f32, elapsed)
}

fn main() {
    let seed: u64 = 2026_0906;

    println!("================================================================");
    println!(" Structural-Time Recency for Agent Memory Compaction — Nightly");
    println!(" 2026-09-06 · crates/ruvector-agent-memory::structural_recency");
    println!("================================================================\n");
    println!("Platform  : {}", std::env::consts::OS);
    println!("Arch      : {}", std::env::consts::ARCH);
    println!();

    println!("Dataset (Experiment 1 & 2, regenerated per-policy from the same seed)");
    println!("  Segments             : {N_SEGMENTS}");
    println!("  Churn per segment    : {CHURN_PER_SEGMENT}");
    println!("  Total memories       : {N_MEMORIES}");
    println!("  Dimensions           : {DIMS}");
    println!("  Churn noise          : {CHURN_NOISE}");
    println!();

    // ── Experiment 1: recency-only ablation, aggressive compaction ─────────
    println!("── Experiment 1: recency-only ablation (alpha=1, beta=0, gamma=0), no context");
    println!("   Compact {N_MEMORIES} -> {TARGET_SIZE_H1} (exactly the regime-shift budget)\n");

    let recency_only = CoherenceWeights {
        alpha: 1.0,
        beta: 0.0,
        gamma: 0.0,
    };
    let baseline_policy = CoherencePolicy::new(recency_only.clone());
    let dedup_policy = DedupGatedRecency::new(DedupGatedWeights {
        base: recency_only.clone(),
        dedup_threshold: DEDUP_THRESHOLD,
    });
    let structural_policy = StructuralTimeRecency::new(StructuralTimeWeights {
        base: recency_only.clone(),
        ..StructuralTimeWeights::default()
    });
    let keyframe_policy = StructuralKeyframeRetention::new(
        recency_only.clone(),
        StructuralTimeWeights::default().metric,
    );

    let (base_rate, base_dur, _) = run_experiment1(&baseline_policy, seed);
    let (dedup_rate, dedup_dur, _) = run_experiment1(&dedup_policy, seed);
    let (struct_rate, struct_dur, _) = run_experiment1(&structural_policy, seed);
    let (kf_rate, kf_dur, kf_survivors_1) = run_experiment1(&keyframe_policy, seed);
    let (_, _, kf_survivors_2) = run_experiment1(&keyframe_policy, seed);
    let deterministic = kf_survivors_1 == kf_survivors_2;

    println!(
        "{:<28} {:>18} {:>16}",
        "Policy", "Survival rate", "Compaction (us)"
    );
    println!("{}", "-".repeat(64));
    println!(
        "{:<28} {:>17.1}% {:>16}",
        baseline_policy.name(),
        base_rate * 100.0,
        base_dur.as_micros()
    );
    println!(
        "{:<28} {:>17.1}% {:>16}",
        dedup_policy.name(),
        dedup_rate * 100.0,
        dedup_dur.as_micros()
    );
    println!(
        "{:<28} {:>17.1}% {:>16} (B1 — score-based, negative result)",
        structural_policy.name(),
        struct_rate * 100.0,
        struct_dur.as_micros()
    );
    println!(
        "{:<28} {:>17.1}% {:>16} (B2 — keyframes(), primary candidate)",
        keyframe_policy.name(),
        kf_rate * 100.0,
        kf_dur.as_micros()
    );
    println!();

    let b1_gap_vs_baseline_pp = (struct_rate - base_rate) * 100.0;
    let gap_vs_baseline_pp = (kf_rate - base_rate) * 100.0;
    let gap_vs_dedup_pp = (kf_rate - dedup_rate) * 100.0;
    let h1_pass = gap_vs_baseline_pp >= SURVIVAL_GAP_THRESHOLD_PP;
    let beats_fair_baseline = gap_vs_dedup_pp > 0.0;

    println!("Experiment 1 result");
    println!(
        "  B1 StructuralTimeRecency vs baseline : {b1_gap_vs_baseline_pp:+.1}pp (score-based; documented negative result, not gating)"
    );
    println!(
        "  B2 StructuralKeyframeRetention vs baseline (tick recency) : {gap_vs_baseline_pp:+.1}pp  (need >= +{SURVIVAL_GAP_THRESHOLD_PP:.0}pp) : {}",
        if h1_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  B2 StructuralKeyframeRetention vs DedupGated (fair cheap competitor) : {gap_vs_dedup_pp:+.1}pp : {}",
        if beats_fair_baseline {
            "structural has an edge over the fair baseline"
        } else {
            "NO EDGE over the fair cheap baseline — reported honestly, not hidden"
        }
    );
    let slowdown_vs_baseline = kf_dur.as_secs_f64() / base_dur.as_secs_f64().max(1e-9);
    let perf_pass_e1 = slowdown_vs_baseline <= MAX_SLOWDOWN;
    println!(
        "  B2 compaction slowdown vs baseline : {slowdown_vs_baseline:.2}x  (need <= {MAX_SLOWDOWN:.0}x) : {}",
        if perf_pass_e1 { "PASS" } else { "FAIL" }
    );
    println!(
        "  B2 determinism (2 runs, same seed) : {}",
        if deterministic { "PASS" } else { "FAIL" }
    );
    println!();

    // ── Experiment 2: production weights, moderate compaction, recall check ─
    println!(
        "── Experiment 2: production default weights, {CONTEXT_WINDOW_SIZE}-vector context window"
    );
    println!(
        "   Compact {N_MEMORIES} -> {TARGET_SIZE_H2} (50%), Recall@{K} over {N_QUERIES} queries\n"
    );

    let prod_weights = CoherenceWeights::default();
    let baseline_prod = CoherencePolicy::new(prod_weights.clone());
    let dedup_prod = DedupGatedRecency::new(DedupGatedWeights {
        base: prod_weights.clone(),
        dedup_threshold: DEDUP_THRESHOLD,
    });
    let keyframe_prod = StructuralKeyframeRetention::new(
        prod_weights.clone(),
        StructuralTimeWeights::default().metric,
    );

    let (base_recall, base_dur2) = run_experiment2(&baseline_prod, seed);
    let (dedup_recall, dedup_dur2) = run_experiment2(&dedup_prod, seed);
    let (kf_recall, kf_dur2) = run_experiment2(&keyframe_prod, seed);

    println!(
        "{:<28} {:>12} {:>16}",
        "Policy", "Recall@10", "Compaction (us)"
    );
    println!("{}", "-".repeat(59));
    println!(
        "{:<28} {:>11.1}% {:>16}",
        baseline_prod.name(),
        base_recall * 100.0,
        base_dur2.as_micros()
    );
    println!(
        "{:<28} {:>11.1}% {:>16}",
        dedup_prod.name(),
        dedup_recall * 100.0,
        dedup_dur2.as_micros()
    );
    println!(
        "{:<28} {:>11.1}% {:>16}",
        keyframe_prod.name(),
        kf_recall * 100.0,
        kf_dur2.as_micros()
    );
    println!();

    let recall_delta_kf_pp = (kf_recall - base_recall) * 100.0;
    let recall_delta_dedup_pp = (dedup_recall - base_recall) * 100.0;
    let h2_struct_pass = recall_delta_kf_pp >= -RECALL_TOLERANCE_PP;
    let h2_dedup_pass = recall_delta_dedup_pp >= -RECALL_TOLERANCE_PP;
    let slowdown_vs_baseline2 = kf_dur2.as_secs_f64() / base_dur2.as_secs_f64().max(1e-9);
    let perf_pass_e2 = slowdown_vs_baseline2 <= MAX_SLOWDOWN;

    println!("Experiment 2 result");
    println!(
        "  B2 StructuralKeyframeRetention Recall@10 delta vs baseline : {recall_delta_kf_pp:+.1}pp (need >= -{RECALL_TOLERANCE_PP:.0}pp) : {}",
        if h2_struct_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  DedupGated Recall@10 delta vs baseline     : {recall_delta_dedup_pp:+.1}pp (need >= -{RECALL_TOLERANCE_PP:.0}pp) : {}",
        if h2_dedup_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  B2 compaction slowdown vs baseline : {slowdown_vs_baseline2:.2}x (need <= {MAX_SLOWDOWN:.0}x) : {}",
        if perf_pass_e2 { "PASS" } else { "FAIL" }
    );
    println!();

    // ── Overall acceptance ───────────────────────────────────────────────────
    let overall_pass =
        h1_pass && h2_struct_pass && h2_dedup_pass && perf_pass_e1 && perf_pass_e2 && deterministic;

    println!("================================================================");
    println!(
        " ACCEPTANCE: {}",
        if overall_pass { "ACCEPT" } else { "REJECT" }
    );
    println!("================================================================");
    if !beats_fair_baseline {
        println!(
            "Note: StructuralKeyframeRetention (B2) beat the tick-recency baseline by \
             {gap_vs_baseline_pp:+.1}pp but did NOT beat the fair, dependency-free \
             DedupGatedRecency competitor ({gap_vs_dedup_pp:+.1}pp). Per emergent-time's \
             own benchmarking discipline, this is reported as a genuine limitation, not \
             smoothed over: the extra emergent-time dependency buys a win over the naive \
             tick clock but not (on this synthetic trace) over the cheap dedup heuristic."
        );
    }

    if !overall_pass {
        std::process::exit(1);
    }
}
