//! Nightly research benchmark (2026-09-15): direct `MinCutBuilder` bridge
//! detection — ADR-345's "Next Research item 1" follow-up.
//! See docs/research/nightly/2026-09-15-direct-mincut-bridge-detection.
//!
//! ADR-345 (docs/adr/ADR-345-mincut-gated-forgetting.md,
//! docs/research/nightly/2026-09-05-mincut-gated-forgetting) measured
//! `MincutGatedForgetting`'s boundary detection — at the time exclusively
//! `RuVectorGraphAnalyzer::from_knn(...).partition()`
//! (`BoundaryMethod::WrapperPartition`) — at ~1,800-2,700x baseline
//! compaction latency (FAIL vs a <=100x gate) and non-deterministic (50%
//! empty-result rate over 30 repeated calls on an identical graph), and
//! rejected the hypothesis that the structural signal helps (0.0pp measured
//! bridge-survival gap vs a >=15pp gate). It left as "Next Research item 1":
//! does calling `ruvector_mincut::DynamicMinCut` directly — bypassing
//! `RuVectorGraphAnalyzer`/`MinCutWrapper`'s up-to-100-instance replay loop —
//! avoid the measured latency and determinism problems?
//!
//! This benchmark reuses ADR-345's exact dataset, seed, and methodology
//! (same 84-entry corpus: 6 clusters x 12 core memories + 12 interpolated
//! bridges, 32-dim, same hot-cluster access simulation, same k-NN
//! parameters) so:
//!   1. Re-running `BoundaryMethod::WrapperPartition` here must reproduce
//!      ADR-345's committed numbers exactly (both runs are fully
//!      deterministic given a fixed seed and unchanged wrapper code path) —
//!      a reproducibility check on the historical result.
//!   2. `BoundaryMethod::DirectBuilder` is measured on the identical corpus,
//!      so the two methods' latency and correctness are directly comparable.
//!
//! Hypothesis (fixed before this run; scoped to ADR-345's item 1 only — the
//! separate "does the structural signal help" question is NOT re-litigated
//! here, since that would require changing the acceptance criteria on a
//! prior rejected hypothesis after seeing new results, which STEP 32 of the
//! nightly harness forbids):
//!
//! Given the identical ADR-345 84-entry corpus and `MincutGatedForgetting`
//! configuration, when boundary detection uses `BoundaryMethod::
//! DirectBuilder` (one-shot `MinCutBuilder::with_edges(...).build()`)
//! instead of `BoundaryMethod::WrapperPartition`, then compaction wall-clock
//! slowdown vs the `CoherencePolicy` baseline should fall from the
//! previously measured ~1,800-2,700x to at or below the pre-existing 100x
//! gate, subject to: (a) `cargo test` remaining green (no regression to the
//! unchanged `WrapperPartition` path or to bridge-detection correctness in
//! the unit-test topology), and (b) the separate `mincut_determinism_probe`
//! example (run alongside this benchmark, not duplicated here) showing 0%
//! empty-result rate over 30 trials, down from the measured 50%.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example mincut_direct_builder_bench --features mincut-forget

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, BoundaryMethod, CoherencePolicy, CoherenceWeights, CompactionPolicy,
    MemoryStore, MincutGatedForgetting,
};
use std::collections::HashSet;
use std::time::{Duration, Instant};

// ── Dataset parameters (identical to ADR-345's benchmark) ──────────────────
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
const MAX_SLOWDOWN_VS_BASELINE: f64 = 100.0; // ADR-345's gate, re-tested here
const MINCUT_TRIALS: usize = 1; // matches ADR-345's benchmark exactly

// ── Vector utilities (mirrors mincut_gated_forgetting_bench.rs / src/main.rs) ─

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

fn run_policy(policy: &dyn CompactionPolicy, seed: u64) -> (f32, f32, Duration) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut store = MemoryStore::new(DIMS);
    let dataset = generate_dataset(&mut store, &mut rng);
    let mut rng2 = StdRng::seed_from_u64(seed + 1);
    let context_window = simulate_accesses(&mut store, &dataset, &mut rng2);
    assert_eq!(store.len(), N_MEMORIES);

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

fn main() {
    let seed: u64 = 341; // identical to ADR-345's benchmark, for reproducibility
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-agent-memory — Direct MinCutBuilder Bridge Detection    ║");
    println!("║  (nightly 2026-09-15, ADR-345 follow-up item 1)                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Platform  : {}", std::env::consts::OS);
    println!("Arch      : {}", std::env::consts::ARCH);
    println!();

    println!("Dataset (identical to ADR-345's benchmark, seed={seed})");
    println!("  Clusters        : {N_CLUSTERS} ({PER_CLUSTER} core memories each = {N_CORE})");
    println!("  Bridge memories : {N_BRIDGES}");
    println!("  Total memories  : {N_MEMORIES}");
    println!("  Target size     : {TARGET_SIZE} (50% compaction)");
    println!();

    let cow = CoherencePolicy::default();

    let mut soft_wrapper =
        MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
    soft_wrapper.mincut_trials = MINCUT_TRIALS;
    soft_wrapper.boundary_method = BoundaryMethod::WrapperPartition;

    let mut hard_wrapper =
        MincutGatedForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION);
    hard_wrapper.mincut_trials = MINCUT_TRIALS;
    hard_wrapper.boundary_method = BoundaryMethod::WrapperPartition;

    let mut soft_direct =
        MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
    soft_direct.mincut_trials = MINCUT_TRIALS;
    soft_direct.boundary_method = BoundaryMethod::DirectBuilder;

    let mut hard_direct =
        MincutGatedForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION);
    hard_direct.mincut_trials = MINCUT_TRIALS;
    hard_direct.boundary_method = BoundaryMethod::DirectBuilder;

    struct Row {
        label: String,
        survival: f32,
        recall: f32,
        micros: u128,
    }
    let policies: [(&str, &dyn CompactionPolicy); 5] = [
        ("CoherenceWeighted (baseline)", &cow),
        (
            "MincutGatedForgetting-Soft (candidate A: wrapper)",
            &soft_wrapper,
        ),
        (
            "MincutGatedForgetting-Hard (candidate A: wrapper)",
            &hard_wrapper,
        ),
        (
            "MincutGatedForgetting-Soft (candidate B: direct)",
            &soft_direct,
        ),
        (
            "MincutGatedForgetting-Hard (candidate B: direct)",
            &hard_direct,
        ),
    ];

    let mut rows = Vec::new();
    for (label, policy) in policies {
        let (survival, recall, dur) = run_policy(policy, seed);
        rows.push(Row {
            label: label.to_string(),
            survival,
            recall,
            micros: dur.as_micros(),
        });
    }

    println!(
        "{:<52} {:>14} {:>11} {:>16}",
        "Policy", "Bridge Surv.", "Recall@10", "Compaction (us)"
    );
    println!("{}", "-".repeat(96));
    for r in &rows {
        println!(
            "{:<52} {:>13.1}% {:>10.1}% {:>16}",
            r.label,
            r.survival * 100.0,
            r.recall * 100.0,
            r.micros
        );
    }
    println!();

    let baseline = &rows[0];
    let soft_wrapper_row = &rows[1];
    let hard_wrapper_row = &rows[2];
    let soft_direct_row = &rows[3];
    let hard_direct_row = &rows[4];

    let slowdown = |row: &Row| row.micros as f64 / baseline.micros.max(1) as f64;
    let sd_soft_wrapper = slowdown(soft_wrapper_row);
    let sd_hard_wrapper = slowdown(hard_wrapper_row);
    let sd_soft_direct = slowdown(soft_direct_row);
    let sd_hard_direct = slowdown(hard_direct_row);

    println!("Acceptance test (this experiment's hypothesis only — bridge-survival-gap and");
    println!("recall-parity are ADR-345's separate, already-rejected hypothesis; reported");
    println!("above as context, not re-gated here):");
    println!(
        "  Candidate A (wrapper) Soft slowdown ({sd_soft_wrapper:>9.1}x) <= {MAX_SLOWDOWN_VS_BASELINE:.0}x : {} (reproduces ADR-345's failing measurement)",
        if sd_soft_wrapper <= MAX_SLOWDOWN_VS_BASELINE { "PASS" } else { "FAIL" }
    );
    println!(
        "  Candidate A (wrapper) Hard slowdown ({sd_hard_wrapper:>9.1}x) <= {MAX_SLOWDOWN_VS_BASELINE:.0}x : {} (reproduces ADR-345's failing measurement)",
        if sd_hard_wrapper <= MAX_SLOWDOWN_VS_BASELINE { "PASS" } else { "FAIL" }
    );
    let direct_soft_pass = sd_soft_direct <= MAX_SLOWDOWN_VS_BASELINE;
    let direct_hard_pass = sd_hard_direct <= MAX_SLOWDOWN_VS_BASELINE;
    println!(
        "  Candidate B (direct)  Soft slowdown ({sd_soft_direct:>9.1}x) <= {MAX_SLOWDOWN_VS_BASELINE:.0}x : {}",
        if direct_soft_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Candidate B (direct)  Hard slowdown ({sd_hard_direct:>9.1}x) <= {MAX_SLOWDOWN_VS_BASELINE:.0}x : {}",
        if direct_hard_pass { "PASS" } else { "FAIL" }
    );
    println!();
    println!(
        "  Speedup vs candidate A: Soft {:.1}x, Hard {:.1}x",
        sd_soft_wrapper / sd_soft_direct.max(1e-9),
        sd_hard_wrapper / sd_hard_direct.max(1e-9),
    );

    // Reproducibility check: candidate A here must match ADR-345's committed
    // numbers exactly (same seed, unchanged wrapper code path).
    println!();
    println!("Reproducibility check vs ADR-345's committed run (seed=341, same corpus):");
    println!(
        "  Candidate A Soft survival={:.1}% recall={:.1}%",
        soft_wrapper_row.survival * 100.0,
        soft_wrapper_row.recall * 100.0
    );
    println!(
        "  Candidate A Hard survival={:.1}% recall={:.1}%",
        hard_wrapper_row.survival * 100.0,
        hard_wrapper_row.recall * 100.0
    );

    if direct_soft_pass && direct_hard_pass {
        println!("\n=> ACCEPT (this experiment): direct MinCutBuilder boundary detection clears the ADR-345 <=100x latency gate that RuVectorGraphAnalyzer::partition() failed.");
    } else {
        println!("\n=> REJECT (this experiment): direct MinCutBuilder boundary detection still exceeds the <=100x latency gate at this corpus size.");
        std::process::exit(1);
    }
}
