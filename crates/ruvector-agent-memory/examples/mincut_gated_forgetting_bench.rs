//! Nightly research benchmark (2026-09-05, ADR-341): mincut-gated forgetting.
//!
//! Hypothesis (fixed before this exact run of the benchmark; see
//! docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md,
//! "Failure modes", for why the corpus size below is much smaller than this
//! experiment originally targeted — a pre-registered scaling probe of
//! `RuVectorGraphAnalyzer::partition()` found call latency ranging from tens
//! of milliseconds to multiple seconds — and in one degenerate small/regular
//! topology, over a minute — for graphs of only 20-400 vertices, making a
//! ~2 000-memory corpus computationally infeasible for this nightly run):
//!
//! Given a synthetic corpus of 6 topic clusters (12 memories each = 72) plus
//! 12 "bridge" memories interpolated 50/50 between two randomly paired
//! clusters, 32-dim, with the same hot-cluster access simulation pattern as
//! the existing `agent-memory-bench` (2 of 6 clusters get proportionally more
//! accesses), and a k-NN (k=5, cosine >= 0.05) similarity graph feeding
//! `ruvector-mincut`'s `RuVectorGraphAnalyzer`,
//!
//! when the 84-entry store is compacted to 50% (42 entries) using
//! MincutGatedForgetting-Soft (candidate A, structural bonus δ=0.5) and
//! MincutGatedForgetting-Hard (candidate B, 20% of the retained budget
//! reserved for boundary vertices) versus the existing CoherencePolicy
//! (baseline, no structural signal),
//!
//! then both candidates retain a bridge-memory survival rate at least 15
//! percentage points higher than the baseline, while Recall@10 over 20
//! hot-cluster test queries stays within 2 percentage points of baseline,
//!
//! subject to: (a) each candidate's compaction wall-clock stays under 100x
//! baseline's on the same corpus (release build; mincut is expected to be
//! substantially more expensive — this bounds it as a "background job", it
//! does not claim parity, and per the scaling probe above IS expected to
//! fail at this corpus size), and (b) 100% tamper-detection across 20
//! independent single-byte-flip trials against the eviction witness chain.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example mincut_gated_forgetting_bench --features mincut-forget

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, compact_witnessed, recall_at_k, CoherencePolicy, CoherenceWeights, CompactionPolicy,
    EvictionWitnessChain, MemoryStore, MemoryWitnessLog, MincutGatedForgetting,
};
use std::collections::HashSet;
use std::time::{Duration, Instant};

// ── Dataset parameters ──────────────────────────────────────────────────────
// Deliberately small (see "Failure modes" / scalability finding in
// docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md):
// RuVectorGraphAnalyzer::partition() was measured at ~0.5-12s per call for
// graphs of 100-400 vertices during this experiment's design (see the
// README's scaling table), so the originally planned 1 950-memory / 20
// hot-cluster corpus is computationally infeasible with today's
// `ruvector-mincut` integration. This benchmark uses the largest corpus that
// completes in a practical nightly-run wall-clock (a few minutes) instead,
// and reports that constraint as part of the result rather than hiding it.
const N_CLUSTERS: usize = 6;
const PER_CLUSTER: usize = 12;
const N_CORE: usize = N_CLUSTERS * PER_CLUSTER; // 72
const N_BRIDGES: usize = 12;
const N_MEMORIES: usize = N_CORE + N_BRIDGES; // 84
const N_HOT_CLUSTERS: usize = 2;
const DIMS: usize = 32;
const N_QUERIES: usize = 20;
const K: usize = 5;
const TARGET_SIZE: usize = N_MEMORIES / 2; // 42, 50% compaction
const CONTEXT_WINDOW_SIZE: usize = 10;

const N_COLD_ERA_ACCESSES: usize = 40;
const N_HOT_ERA_ACCESSES: usize = 80;
const HOT_ERA_HOT_FRAC: f64 = 0.90;

const STRUCTURAL_BONUS: f32 = 0.5;
const PROTECT_FRACTION: f32 = 0.2;
const BRIDGE_SURVIVAL_GAP_THRESHOLD_PP: f32 = 15.0;
const RECALL_TOLERANCE: f32 = 0.02;
// Set from the pre-run scaling probe (README "Failure modes"): mincut
// computation is inherently far more expensive than a scalar sort, so this
// is a "background/offline job" bar, not parity — 100x still separates a
// usable-but-slow compaction pass from a pathological one. This is expected
// to fail at the corpus size below; see the Acceptance section.
const MAX_SLOWDOWN_VS_BASELINE: f64 = 100.0;
const N_TAMPER_TRIALS: usize = 20;
// Retries mitigate the partition() non-determinism finding (graph_forget.rs
// doc comment) but each retry re-pays the full mincut latency measured
// above; 1 keeps the whole nightly run in a practical wall-clock at this
// corpus size, at the documented cost of slightly higher variance.
const MINCUT_TRIALS: usize = 1;

// ── Vector utilities (mirrors src/main.rs) ──────────────────────────────────

fn unit_gaussian(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn scale_vec(v: &[f32], s: f32) -> Vec<f32> {
    v.iter().map(|x| x * s).collect()
}

fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}

fn perturb(centroid: &[f32], noise: f32, rng: &mut StdRng) -> Vec<f32> {
    let n = unit_gaussian(rng, centroid.len());
    normalize_vec(&add_vecs(centroid, &scale_vec(&n, noise)))
}

fn midpoint(a: &[f32], b: &[f32]) -> Vec<f32> {
    normalize_vec(&add_vecs(a, b))
}

// ── Dataset ──────────────────────────────────────────────────────────────────

struct Dataset {
    centroids: Vec<Vec<f32>>,
    cluster_of: Vec<usize>,
    /// Indices (into the store, insertion order) of the `N_BRIDGES` bridge
    /// memories.
    bridge_indices: HashSet<usize>,
    queries: Vec<(Vec<f32>, Vec<u64>)>,
}

fn generate_dataset(store: &mut MemoryStore, rng: &mut StdRng) -> Dataset {
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS).map(|_| unit_gaussian(rng, DIMS)).collect();
    let mut cluster_of = Vec::with_capacity(N_MEMORIES);

    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..PER_CLUSTER {
            let v = perturb(centroid, 0.35, rng);
            store.insert(v);
            cluster_of.push(c);
        }
    }

    let mut bridge_indices = HashSet::new();
    for _ in 0..N_BRIDGES {
        let a = rng.gen_range(0..N_CLUSTERS);
        let mut b = rng.gen_range(0..N_CLUSTERS);
        while b == a {
            b = rng.gen_range(0..N_CLUSTERS);
        }
        let mid = midpoint(&centroids[a], &centroids[b]);
        let v = perturb(&mid, 0.15, rng);
        let idx = store.len();
        store.insert(v);
        bridge_indices.insert(idx);
        // Bridges are not attributed to either cluster for query generation.
        cluster_of.push(usize::MAX);
    }

    let mut queries = Vec::with_capacity(N_QUERIES);
    for i in 0..N_QUERIES {
        let hot_cluster = i % N_HOT_CLUSTERS;
        let q = perturb(&centroids[hot_cluster], 0.30, rng);
        let truth: Vec<u64> = store.search(&q, K).into_iter().map(|r| r.id).collect();
        queries.push((q, truth));
    }

    Dataset {
        centroids,
        cluster_of,
        bridge_indices,
        queries,
    }
}

fn simulate_accesses(
    store: &mut MemoryStore,
    dataset: &Dataset,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    for _ in 0..N_COLD_ERA_ACCESSES {
        let idx = rng.gen_range(0..N_MEMORIES);
        store.access_by_index(idx);
    }

    let mut context_accesses: Vec<Vec<f32>> = Vec::new();
    for _ in 0..N_HOT_ERA_ACCESSES {
        let idx = if rng.gen_bool(HOT_ERA_HOT_FRAC) {
            let hot_c = rng.gen_range(0..N_HOT_CLUSTERS);
            hot_c * PER_CLUSTER + rng.gen_range(0..PER_CLUSTER)
        } else {
            let cold_c = rng.gen_range(N_HOT_CLUSTERS..N_CLUSTERS);
            cold_c * PER_CLUSTER + rng.gen_range(0..PER_CLUSTER)
        };
        store.access_by_index(idx);
        let cluster = dataset.cluster_of[idx];
        if cluster != usize::MAX {
            context_accesses.push(dataset.centroids[cluster].clone());
        }
    }

    let start = context_accesses.len().saturating_sub(CONTEXT_WINDOW_SIZE);
    context_accesses[start..].to_vec()
}

fn measure_recall(queries: &[(Vec<f32>, Vec<u64>)], store: &MemoryStore) -> f32 {
    let mut total = 0.0f32;
    for (q, truth) in queries {
        let candidates: Vec<u64> = store.search(q, K).into_iter().map(|r| r.id).collect();
        total += recall_at_k(truth, &candidates);
    }
    total / queries.len() as f32
}

/// Rebuilds a fresh, identically-seeded store+dataset, runs one compaction
/// policy, and reports (bridge survival rate, recall, compaction wall-clock).
fn run_policy(policy: &dyn CompactionPolicy, seed: u64) -> (f32, f32, Duration) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut store = MemoryStore::new(DIMS);
    let dataset = generate_dataset(&mut store, &mut rng);
    let mut rng2 = StdRng::seed_from_u64(seed + 1);
    let context_window = simulate_accesses(&mut store, &dataset, &mut rng2);
    assert_eq!(store.len(), N_MEMORIES);

    // Bridge *ids* (stable across compaction), captured before compaction.
    let bridge_ids: HashSet<u64> = dataset
        .bridge_indices
        .iter()
        .map(|&i| store.entries()[i].id)
        .collect();

    let t0 = Instant::now();
    compact(&mut store, policy, TARGET_SIZE, &context_window);
    let elapsed = t0.elapsed();

    assert_eq!(store.len(), TARGET_SIZE);
    let surviving_bridges = store
        .entries()
        .iter()
        .filter(|e| bridge_ids.contains(&e.id))
        .count();
    let survival_rate = surviving_bridges as f32 / bridge_ids.len() as f32;

    let recall = measure_recall(&dataset.queries, &store);
    (survival_rate, recall, elapsed)
}

fn run_tamper_trials(seed: u64) -> (usize, usize) {
    let mut detected = 0usize;
    for trial in 0..N_TAMPER_TRIALS {
        let mut rng = StdRng::seed_from_u64(seed + trial as u64);
        let mut store = MemoryStore::new(DIMS);
        let dataset = generate_dataset(&mut store, &mut rng);
        let mut rng2 = StdRng::seed_from_u64(seed + trial as u64 + 1);
        let context_window = simulate_accesses(&mut store, &dataset, &mut rng2);

        let mut policy = MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
        policy.mincut_trials = MINCUT_TRIALS;
        let mut chain = EvictionWitnessChain::new();
        let mut log = MemoryWitnessLog::default();
        compact_witnessed(
            &mut store,
            &policy,
            TARGET_SIZE,
            &context_window,
            "nightly-bench",
            trial as u64,
            &mut chain,
            &mut log,
        )
        .expect("witnessed compaction succeeds");

        assert!(log.verify_chain(), "freshly emitted chain must verify");

        // Flip one random byte of one random field on a random record.
        let n = log.records.len();
        let victim = rng.gen_range(0..n);
        match rng.gen_range(0..3) {
            0 => log.records[victim].payload ^= 1 << rng.gen_range(0..64),
            1 => log.records[victim].target_object_id ^= 1 << rng.gen_range(0..32),
            _ => log.records[victim].timestamp_ns ^= 1 << rng.gen_range(0..64),
        }

        if !log.verify_chain() {
            detected += 1;
        }
    }
    (detected, N_TAMPER_TRIALS)
}

fn main() {
    let seed: u64 = 341;
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-agent-memory — Mincut-Gated Forgetting (ADR-341)        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Platform  : {}", std::env::consts::OS);
    println!("Arch      : {}", std::env::consts::ARCH);
    println!();

    println!("Dataset");
    println!("  Clusters        : {N_CLUSTERS} ({PER_CLUSTER} core memories each = {N_CORE})");
    println!(
        "  Bridge memories : {N_BRIDGES} (interpolated between 2 random clusters, noise=0.15)"
    );
    println!("  Total memories  : {N_MEMORIES}");
    println!("  Dimensions      : {DIMS}");
    println!("  Hot clusters    : {N_HOT_CLUSTERS}");
    println!("  Target size     : {TARGET_SIZE} (50% compaction)");
    println!("  Test queries    : {N_QUERIES}, K={K}");
    println!();

    let cow = CoherencePolicy::default();
    let mut soft = MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
    soft.mincut_trials = MINCUT_TRIALS;
    let mut hard = MincutGatedForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION);
    hard.mincut_trials = MINCUT_TRIALS;

    struct Row {
        name: String,
        survival: f32,
        recall: f32,
        micros: u128,
    }
    let mut rows = Vec::new();
    for policy in [&cow as &dyn CompactionPolicy, &soft, &hard] {
        let (survival, recall, dur) = run_policy(policy, seed);
        rows.push(Row {
            name: policy.name().to_string(),
            survival,
            recall,
            micros: dur.as_micros(),
        });
    }

    println!(
        "{:<28} {:>16} {:>12} {:>16}",
        "Policy", "Bridge Surv.", "Recall@10", "Compaction (us)"
    );
    println!("{}", "-".repeat(76));
    for r in &rows {
        println!(
            "{:<28} {:>15.1}% {:>11.1}% {:>16}",
            r.name,
            r.survival * 100.0,
            r.recall * 100.0,
            r.micros
        );
    }
    println!();

    let baseline = &rows[0];
    let soft_row = &rows[1];
    let hard_row = &rows[2];

    println!("Tamper-detection trials (eviction witness chain)");
    let (detected, total) = run_tamper_trials(seed + 1_000);
    println!("  Detected {detected}/{total} single-byte-flip tampers\n");

    println!("Acceptance test");
    let survival_gap_soft = (soft_row.survival - baseline.survival) * 100.0;
    let survival_gap_hard = (hard_row.survival - baseline.survival) * 100.0;
    let soft_gap_pass = survival_gap_soft >= BRIDGE_SURVIVAL_GAP_THRESHOLD_PP;
    let hard_gap_pass = survival_gap_hard >= BRIDGE_SURVIVAL_GAP_THRESHOLD_PP;
    println!(
        "  Soft bridge-survival gap  ({survival_gap_soft:+.1}pp) >= {BRIDGE_SURVIVAL_GAP_THRESHOLD_PP:.0}pp : {}",
        if soft_gap_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Hard bridge-survival gap  ({survival_gap_hard:+.1}pp) >= {BRIDGE_SURVIVAL_GAP_THRESHOLD_PP:.0}pp : {}",
        if hard_gap_pass { "PASS" } else { "FAIL" }
    );

    let recall_delta_soft = (soft_row.recall - baseline.recall).abs();
    let recall_delta_hard = (hard_row.recall - baseline.recall).abs();
    let soft_recall_pass = recall_delta_soft <= RECALL_TOLERANCE;
    let hard_recall_pass = recall_delta_hard <= RECALL_TOLERANCE;
    println!(
        "  Soft |recall delta| ({:.2}pp) <= {:.0}pp                 : {}",
        recall_delta_soft * 100.0,
        RECALL_TOLERANCE * 100.0,
        if soft_recall_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Hard |recall delta| ({:.2}pp) <= {:.0}pp                 : {}",
        recall_delta_hard * 100.0,
        RECALL_TOLERANCE * 100.0,
        if hard_recall_pass { "PASS" } else { "FAIL" }
    );

    let slowdown_soft = soft_row.micros as f64 / baseline.micros.max(1) as f64;
    let slowdown_hard = hard_row.micros as f64 / baseline.micros.max(1) as f64;
    let soft_speed_pass = slowdown_soft <= MAX_SLOWDOWN_VS_BASELINE;
    let hard_speed_pass = slowdown_hard <= MAX_SLOWDOWN_VS_BASELINE;
    println!(
        "  Soft compaction slowdown  ({slowdown_soft:.1}x) <= {MAX_SLOWDOWN_VS_BASELINE:.0}x                : {}",
        if soft_speed_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Hard compaction slowdown  ({slowdown_hard:.1}x) <= {MAX_SLOWDOWN_VS_BASELINE:.0}x                : {}",
        if hard_speed_pass { "PASS" } else { "FAIL" }
    );

    let tamper_pass = detected == total;
    println!(
        "  Tamper detection ({detected}/{total})                          : {}",
        if tamper_pass { "PASS" } else { "FAIL" }
    );
    println!();

    let all_pass = soft_gap_pass
        && hard_gap_pass
        && soft_recall_pass
        && hard_recall_pass
        && soft_speed_pass
        && hard_speed_pass
        && tamper_pass;

    if all_pass {
        println!("=> ACCEPT: mincut-gated forgetting protects structural bridges at no material recall or witness-integrity cost.");
    } else {
        println!("=> REJECT: one or more mandatory acceptance thresholds failed (see above).");
        std::process::exit(1);
    }
}
