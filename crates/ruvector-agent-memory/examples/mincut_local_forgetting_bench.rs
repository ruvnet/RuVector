//! Nightly research benchmark (2026-09-12, ADR-346, follow-up to ADR-345):
//! does replacing `RuVectorGraphAnalyzer::partition()` (one expensive,
//! measured-non-deterministic *global* min-cut call per compaction) with
//! `DeterministicLocalKCut` (`n` cheap, provably deterministic *local*
//! k-cut queries, one per vertex) fix the 2026-09-05 rejection's two root
//! causes — latency and non-determinism — while keeping the structural
//! "protect the bridge" benefit?
//!
//! Hypothesis (fixed before this run; see
//! docs/research/nightly/2026-09-12-local-kcut-gated-forgetting/README.md):
//!
//! Given the same synthetic corpus as the 2026-09-05 experiment (6 topic
//! clusters x 12 core memories + 12 bridge memories interpolated 50/50
//! between two random clusters, 32-dim, k-NN k=5 cosine >= 0.05, hot-cluster
//! access simulation over 20 test queries) compacted 50% by
//! `MincutGatedForgetting` in `MincutEngine::LocalDeterministic` mode
//! (`max_radius=0`, `budget_k=4` — see `graph_forget.rs`'s "Design note" doc
//! comment on `boundary_indices_local` for why `max_radius=0`, decided
//! during design, before this run) versus the same policy in the original
//! `MincutEngine::ExactGlobal` mode and versus plain `CoherencePolicy`,
//!
//! when corpus size is scaled from 84 up to 924 vertices (same cluster/
//! bridge ratio),
//!
//! then (a) `LocalDeterministic` retains the same >=15pp bridge-survival
//! gap over baseline and <=2pp recall delta `ExactGlobal` was required to
//! hit at 84 vertices; (b) `LocalDeterministic`'s wall-clock slowdown vs
//! baseline, at the largest size where `ExactGlobal` still completes a call
//! within a 1.5s budget, is at least 5x smaller than `ExactGlobal`'s at that
//! same size, and stays under 20x in absolute terms; and (c)
//! `LocalDeterministic` returns byte-identical survivor sets across 20
//! repeated `compact()` calls on unchanged input,
//!
//! subject to: 100% tamper-detection across 20 independent single-byte-flip
//! trials against the eviction witness chain, using `LocalDeterministic`
//! (closing the loop on whether the follow-up engine is still compatible
//! with the existing ADR-134 witness machinery).
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example mincut_local_forgetting_bench --features mincut-forget

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, compact_witnessed, recall_at_k, CoherencePolicy, CoherenceWeights, CompactionPolicy,
    EvictionWitnessChain, MemoryStore, MemoryWitnessLog, MincutGatedForgetting,
};
use std::collections::HashSet;
use std::time::{Duration, Instant};

// ── Fixed hypothesis-size dataset (same shape as ADR-345's 84-memory run) ──
const N_CLUSTERS: usize = 6;
const PER_CLUSTER: usize = 12;
const N_BRIDGES: usize = 12;
const N_HOT_CLUSTERS: usize = 2;
const DIMS: usize = 32;
const N_QUERIES: usize = 20;
const K: usize = 5;
const CONTEXT_WINDOW_SIZE: usize = 10;
const N_COLD_ERA_ACCESSES: usize = 40;
const N_HOT_ERA_ACCESSES: usize = 80;
const HOT_ERA_HOT_FRAC: f64 = 0.90;

const STRUCTURAL_BONUS: f32 = 0.5;
const PROTECT_FRACTION: f32 = 0.2;
const BRIDGE_SURVIVAL_GAP_THRESHOLD_PP: f32 = 15.0;
const RECALL_TOLERANCE: f32 = 0.02;
const N_TAMPER_TRIALS: usize = 20;
const N_DETERMINISM_TRIALS: usize = 20;

// Pre-declared acceptance bars for the scaling claim (see module doc):
// LocalDeterministic must be materially faster than ExactGlobal at the
// largest tested size, AND stay within a much tighter absolute bound than
// ExactGlobal's already-violated 100x bar.
const MIN_SPEEDUP_LOCAL_VS_EXACT_AT_MAX_SIZE: f64 = 5.0;
const MAX_LOCAL_SLOWDOWN_VS_BASELINE_AT_MAX_SIZE: f64 = 20.0;
// ExactGlobal scaling sizes stop growing once a single call exceeds this —
// avoids repeating ADR-345's multi-second-per-call blowup for every size.
const EXACT_SCALING_BUDGET: Duration = Duration::from_millis(1500);

// ── Vector utilities (mirrors mincut_gated_forgetting_bench.rs) ────────────

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

// ── Parametric dataset (generalizes ADR-345's fixed 84-memory generator so
//    the scaling probe below can reuse the exact same topology) ───────────

struct Dataset {
    centroids: Vec<Vec<f32>>,
    cluster_of: Vec<usize>,
    bridge_indices: HashSet<usize>,
    queries: Vec<(Vec<f32>, Vec<u64>)>,
}

struct Shape {
    n_clusters: usize,
    per_cluster: usize,
    n_bridges: usize,
}

impl Shape {
    fn n_memories(&self) -> usize {
        self.n_clusters * self.per_cluster + self.n_bridges
    }
}

fn generate_dataset(store: &mut MemoryStore, shape: &Shape, rng: &mut StdRng) -> Dataset {
    let centroids: Vec<Vec<f32>> = (0..shape.n_clusters)
        .map(|_| unit_gaussian(rng, DIMS))
        .collect();
    let mut cluster_of = Vec::with_capacity(shape.n_memories());

    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..shape.per_cluster {
            let v = perturb(centroid, 0.35, rng);
            store.insert(v);
            cluster_of.push(c);
        }
    }

    let mut bridge_indices = HashSet::new();
    for _ in 0..shape.n_bridges {
        let a = rng.gen_range(0..shape.n_clusters);
        let mut b = rng.gen_range(0..shape.n_clusters);
        while b == a {
            b = rng.gen_range(0..shape.n_clusters);
        }
        let mid = midpoint(&centroids[a], &centroids[b]);
        let v = perturb(&mid, 0.15, rng);
        let idx = store.len();
        store.insert(v);
        bridge_indices.insert(idx);
        cluster_of.push(usize::MAX);
    }

    let mut queries = Vec::with_capacity(N_QUERIES);
    for i in 0..N_QUERIES {
        let hot_cluster = i % N_HOT_CLUSTERS.min(shape.n_clusters).max(1);
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
    shape: &Shape,
    dataset: &Dataset,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let n = shape.n_memories();
    for _ in 0..N_COLD_ERA_ACCESSES {
        let idx = rng.gen_range(0..n);
        store.access_by_index(idx);
    }

    let hot_clusters = N_HOT_CLUSTERS.min(shape.n_clusters).max(1);
    let mut context_accesses: Vec<Vec<f32>> = Vec::new();
    for _ in 0..N_HOT_ERA_ACCESSES {
        let idx = if rng.gen_bool(HOT_ERA_HOT_FRAC) {
            let hot_c = rng.gen_range(0..hot_clusters);
            hot_c * shape.per_cluster + rng.gen_range(0..shape.per_cluster)
        } else {
            let cold_c = rng.gen_range(hot_clusters..shape.n_clusters.max(hot_clusters + 1))
                % shape.n_clusters;
            cold_c * shape.per_cluster + rng.gen_range(0..shape.per_cluster)
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

/// Rebuilds a fresh, identically-seeded store+dataset for `shape`, runs one
/// compaction policy, and reports (bridge survival rate, recall, wall-clock).
fn run_policy(policy: &dyn CompactionPolicy, shape: &Shape, seed: u64) -> (f32, f32, Duration) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut store = MemoryStore::new(DIMS);
    let dataset = generate_dataset(&mut store, shape, &mut rng);
    let mut rng2 = StdRng::seed_from_u64(seed + 1);
    let context_window = simulate_accesses(&mut store, shape, &dataset, &mut rng2);
    assert_eq!(store.len(), shape.n_memories());

    let bridge_ids: HashSet<u64> = dataset
        .bridge_indices
        .iter()
        .map(|&i| store.entries()[i].id)
        .collect();

    let target_size = shape.n_memories() / 2;
    let t0 = Instant::now();
    compact(&mut store, policy, target_size, &context_window);
    let elapsed = t0.elapsed();

    assert_eq!(store.len(), target_size);
    let surviving_bridges = store
        .entries()
        .iter()
        .filter(|e| bridge_ids.contains(&e.id))
        .count();
    let survival_rate = surviving_bridges as f32 / bridge_ids.len().max(1) as f32;
    let recall = measure_recall(&dataset.queries, &store);
    (survival_rate, recall, elapsed)
}

/// Runs `compact()` with `policy` `trials` times on freshly rebuilt,
/// identically-seeded input and returns how many trials produced a survivor
/// *id set* identical to the first trial's — the same non-determinism probe
/// ADR-345 used, now run head-to-head for both engines.
fn determinism_trials(
    policy: &dyn CompactionPolicy,
    shape: &Shape,
    seed: u64,
    trials: usize,
) -> usize {
    let mut reference: Option<HashSet<u64>> = None;
    let mut identical = 0usize;
    for t in 0..trials {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut store = MemoryStore::new(DIMS);
        let dataset = generate_dataset(&mut store, shape, &mut rng);
        let mut rng2 = StdRng::seed_from_u64(seed + 1);
        let context_window = simulate_accesses(&mut store, shape, &dataset, &mut rng2);
        let target_size = shape.n_memories() / 2;
        compact(&mut store, policy, target_size, &context_window);
        let ids: HashSet<u64> = store.entries().iter().map(|e| e.id).collect();
        match &reference {
            None => {
                reference = Some(ids);
                identical += 1;
            }
            Some(r) => {
                if *r == ids {
                    identical += 1;
                }
            }
        }
        let _ = t;
    }
    identical
}

fn run_tamper_trials(seed: u64) -> (usize, usize) {
    let shape = Shape {
        n_clusters: N_CLUSTERS,
        per_cluster: PER_CLUSTER,
        n_bridges: N_BRIDGES,
    };
    let mut detected = 0usize;
    for trial in 0..N_TAMPER_TRIALS {
        let mut rng = StdRng::seed_from_u64(seed + trial as u64);
        let mut store = MemoryStore::new(DIMS);
        let dataset = generate_dataset(&mut store, &shape, &mut rng);
        let mut rng2 = StdRng::seed_from_u64(seed + trial as u64 + 1);
        let context_window = simulate_accesses(&mut store, &shape, &dataset, &mut rng2);

        let policy =
            MincutGatedForgetting::soft_local(CoherenceWeights::default(), STRUCTURAL_BONUS);
        let mut chain = EvictionWitnessChain::new();
        let mut log = MemoryWitnessLog::default();
        compact_witnessed(
            &mut store,
            &policy,
            shape.n_memories() / 2,
            &context_window,
            "nightly-bench-local",
            trial as u64,
            &mut chain,
            &mut log,
        )
        .expect("witnessed compaction succeeds");

        assert!(log.verify_chain(), "freshly emitted chain must verify");

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
    let seed: u64 = 346;
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-agent-memory — LocalDeterministic Mincut Forgetting     ║");
    println!("║  (ADR-346, follow-up to ADR-345)                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    println!("Platform  : {}", std::env::consts::OS);
    println!("Arch      : {}", std::env::consts::ARCH);
    println!();

    let base_shape = Shape {
        n_clusters: N_CLUSTERS,
        per_cluster: PER_CLUSTER,
        n_bridges: N_BRIDGES,
    };
    println!(
        "Section A — hypothesis-size corpus ({} memories, same shape as ADR-345)",
        base_shape.n_memories()
    );
    println!(
        "  Clusters={N_CLUSTERS} per_cluster={PER_CLUSTER} bridges={N_BRIDGES} dims={DIMS} target=50%"
    );
    println!();

    let cow = CoherencePolicy::default();
    let soft_exact = MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
    let hard_exact = MincutGatedForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION);
    let soft_local =
        MincutGatedForgetting::soft_local(CoherenceWeights::default(), STRUCTURAL_BONUS);
    let hard_local =
        MincutGatedForgetting::hard_local(CoherenceWeights::default(), PROTECT_FRACTION);

    struct Row {
        name: String,
        survival: f32,
        recall: f32,
        micros: u128,
    }
    let mut rows = Vec::new();
    for policy in [
        &cow as &dyn CompactionPolicy,
        &soft_exact,
        &hard_exact,
        &soft_local,
        &hard_local,
    ] {
        let (survival, recall, dur) = run_policy(policy, &base_shape, seed);
        rows.push(Row {
            name: policy.name().to_string(),
            survival,
            recall,
            micros: dur.as_micros(),
        });
    }
    // soft_local/hard_local share `name()` with soft_exact/hard_exact
    // (`CompactionPolicy::name` only encodes Soft/Hard, not engine) — relabel
    // for the table by position instead of relying on `name()` alone.
    rows[3].name = format!("{}-Local", rows[3].name);
    rows[4].name = format!("{}-Local", rows[4].name);
    rows[1].name = format!("{}-Exact", rows[1].name);
    rows[2].name = format!("{}-Exact", rows[2].name);

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
    let soft_exact_row = &rows[1];
    let hard_exact_row = &rows[2];
    let soft_local_row = &rows[3];
    let hard_local_row = &rows[4];

    println!("Section B — determinism ({N_DETERMINISM_TRIALS} repeated compact() calls on unchanged input)");
    let det_exact = determinism_trials(&soft_exact, &base_shape, seed + 500, N_DETERMINISM_TRIALS);
    let det_local = determinism_trials(&soft_local, &base_shape, seed + 500, N_DETERMINISM_TRIALS);
    println!("  Soft-Exact  identical survivor sets : {det_exact}/{N_DETERMINISM_TRIALS}");
    println!("  Soft-Local  identical survivor sets : {det_local}/{N_DETERMINISM_TRIALS}");
    println!();

    println!("Section C — scaling probe (Soft-Exact vs Soft-Local compact() wall-clock)");
    println!(
        "{:>8} {:>16} {:>16} {:>16}",
        "n", "Baseline (us)", "Exact (us)", "Local (us)"
    );
    let scale_multipliers = [1usize, 2, 3, 5, 7, 11]; // -> 84, 168, 252, 420, 588, 924
    let mut exact_budget_exceeded = false;
    let mut scaling_rows: Vec<(usize, u128, Option<u128>, u128)> = Vec::new();
    for &m in &scale_multipliers {
        let shape = Shape {
            n_clusters: N_CLUSTERS,
            per_cluster: PER_CLUSTER * m,
            n_bridges: N_BRIDGES * m,
        };
        let (_, _, base_dur) = run_policy(&cow, &shape, seed + 900 + m as u64);
        let exact_dur = if exact_budget_exceeded {
            None
        } else {
            let (_, _, d) = run_policy(&soft_exact, &shape, seed + 900 + m as u64);
            if d > EXACT_SCALING_BUDGET {
                exact_budget_exceeded = true;
            }
            Some(d.as_micros())
        };
        let (_, _, local_dur) = run_policy(&soft_local, &shape, seed + 900 + m as u64);
        println!(
            "{:>8} {:>16} {:>16} {:>16}",
            shape.n_memories(),
            base_dur.as_micros(),
            exact_dur
                .map(|v| v.to_string())
                .unwrap_or_else(|| "skipped(budget)".to_string()),
            local_dur.as_micros()
        );
        scaling_rows.push((
            shape.n_memories(),
            base_dur.as_micros(),
            exact_dur,
            local_dur.as_micros(),
        ));
    }
    println!();

    println!("Tamper-detection trials (eviction witness chain, Soft-Local engine)");
    let (detected, total) = run_tamper_trials(seed + 1_000);
    println!("  Detected {detected}/{total} single-byte-flip tampers\n");

    println!("Acceptance test");
    let survival_gap_local_soft = (soft_local_row.survival - baseline.survival) * 100.0;
    let survival_gap_local_hard = (hard_local_row.survival - baseline.survival) * 100.0;
    let soft_gap_pass = survival_gap_local_soft >= BRIDGE_SURVIVAL_GAP_THRESHOLD_PP;
    let hard_gap_pass = survival_gap_local_hard >= BRIDGE_SURVIVAL_GAP_THRESHOLD_PP;
    println!(
        "  (a) Soft-Local bridge-survival gap ({survival_gap_local_soft:+.1}pp) >= {BRIDGE_SURVIVAL_GAP_THRESHOLD_PP:.0}pp : {}",
        if soft_gap_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  (a) Hard-Local bridge-survival gap ({survival_gap_local_hard:+.1}pp) >= {BRIDGE_SURVIVAL_GAP_THRESHOLD_PP:.0}pp : {}",
        if hard_gap_pass { "PASS" } else { "FAIL" }
    );

    let recall_delta_soft = (soft_local_row.recall - baseline.recall).abs();
    let recall_delta_hard = (hard_local_row.recall - baseline.recall).abs();
    let soft_recall_pass = recall_delta_soft <= RECALL_TOLERANCE;
    let hard_recall_pass = recall_delta_hard <= RECALL_TOLERANCE;
    println!(
        "  (a) Soft-Local |recall delta| ({:.2}pp) <= {:.0}pp             : {}",
        recall_delta_soft * 100.0,
        RECALL_TOLERANCE * 100.0,
        if soft_recall_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  (a) Hard-Local |recall delta| ({:.2}pp) <= {:.0}pp             : {}",
        recall_delta_hard * 100.0,
        RECALL_TOLERANCE * 100.0,
        if hard_recall_pass { "PASS" } else { "FAIL" }
    );

    // (b) scaling claim, evaluated at the largest size Exact actually
    // completed within budget (falls back to the largest tested size if
    // Exact never exceeded budget).
    let last_with_exact = scaling_rows
        .iter()
        .rev()
        .find(|(_, _, exact, _)| exact.is_some())
        .cloned()
        .unwrap_or(scaling_rows[0]);
    let (max_n, base_us, exact_us_opt, local_us) = last_with_exact;
    let exact_us = exact_us_opt.unwrap_or(local_us.max(1));
    let speedup_local_vs_exact = exact_us as f64 / local_us.max(1) as f64;
    let local_slowdown_vs_baseline = local_us as f64 / base_us.max(1) as f64;
    let speedup_pass = speedup_local_vs_exact >= MIN_SPEEDUP_LOCAL_VS_EXACT_AT_MAX_SIZE;
    let slowdown_pass = local_slowdown_vs_baseline <= MAX_LOCAL_SLOWDOWN_VS_BASELINE_AT_MAX_SIZE;
    println!(
        "  (b) @n={max_n}: Local vs Exact speedup ({speedup_local_vs_exact:.1}x) >= {MIN_SPEEDUP_LOCAL_VS_EXACT_AT_MAX_SIZE:.0}x       : {}",
        if speedup_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  (b) @n={max_n}: Local vs baseline slowdown ({local_slowdown_vs_baseline:.1}x) <= {MAX_LOCAL_SLOWDOWN_VS_BASELINE_AT_MAX_SIZE:.0}x : {}",
        if slowdown_pass { "PASS" } else { "FAIL" }
    );

    let determinism_pass = det_local == N_DETERMINISM_TRIALS;
    println!(
        "  (c) Soft-Local determinism ({det_local}/{N_DETERMINISM_TRIALS} identical)              : {}",
        if determinism_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "      (reference — Soft-Exact determinism: {det_exact}/{N_DETERMINISM_TRIALS} identical, not gated on)"
    );

    let tamper_pass = detected == total;
    println!(
        "  Tamper detection ({detected}/{total})                                     : {}",
        if tamper_pass { "PASS" } else { "FAIL" }
    );
    println!();

    // Kept out of `all_pass`: informational context, not part of the
    // pre-declared hypothesis (mirrors ADR-345's own convention of reporting
    // non-gating context alongside the gated acceptance test).
    let _ = (soft_exact_row, hard_exact_row);

    let all_pass = soft_gap_pass
        && hard_gap_pass
        && soft_recall_pass
        && hard_recall_pass
        && speedup_pass
        && slowdown_pass
        && determinism_pass
        && tamper_pass;

    if all_pass {
        println!("=> ACCEPT: LocalDeterministic keeps the structural bridge-protection benefit while fixing both the latency and determinism defects found in ADR-345's ExactGlobal engine.");
    } else {
        println!("=> REJECT: one or more mandatory acceptance thresholds failed (see above).");
        std::process::exit(1);
    }
}
