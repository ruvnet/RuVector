//! Nightly research benchmark (2026-09-17, ADR-346): does `MincutBackend::Direct`
//! (`ruvector_mincut::DynamicMinCut` via `MinCutBuilder`) fix the performance
//! rejection from the 2026-09-05 run (ADR-345)?
//!
//! This is the 2026-09-05 run's own "Next Research" item 1, executed
//! without modification to the corpus, hypothesis text, or acceptance
//! thresholds (the "don't move the goalposts" rule in that run's Next
//! Research item 3):
//!
//! Given the same synthetic corpus of 6 topic clusters (12 memories each =
//! 72) plus 12 "bridge" memories interpolated 50/50 between two randomly
//! paired clusters, 32-dim, with the same hot-cluster access simulation
//! pattern (2 of 6 clusters get proportionally more accesses), and the same
//! k-NN (k=5, cosine >= 0.05) similarity graph,
//!
//! when the 84-entry store is compacted to 50% (42 entries) using
//! MincutGatedForgetting-Soft and MincutGatedForgetting-Hard with
//! `MincutBackend::Direct` versus the existing CoherencePolicy (baseline,
//! no structural signal) and versus the same policies run with
//! `MincutBackend::Wrapper` (reproducing the 2026-09-05 numbers unmodified,
//! for a direct before/after comparison in one run),
//!
//! then Direct-backed candidates retain a bridge-memory survival rate at
//! least 15 percentage points higher than baseline, while Recall@10 over 20
//! hot-cluster test queries stays within 2 percentage points of baseline,
//!
//! subject to: each Direct-backed candidate's compaction wall-clock stays
//! under 100x baseline's on the same corpus (release build) — the exact
//! gate the Wrapper backend failed by 18-27x last run.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example mincut_direct_backend_bench --features mincut-forget

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, CoherencePolicy, CoherenceWeights, CompactionPolicy, MemoryStore,
    MincutBackend, MincutGatedForgetting,
};
use std::collections::HashSet;
use std::time::{Duration, Instant};

// ── Dataset parameters (byte-identical to mincut_gated_forgetting_bench.rs) ─
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
const MAX_SLOWDOWN_VS_BASELINE: f64 = 100.0;
// Direct is exact and deterministic (see mincut_direct_determinism_probe),
// so no retry-union mitigation is needed; kept at 1 for both backends so
// this run isolates exactly one variable (the backend).
const MINCUT_TRIALS: usize = 1;

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

// ── Dataset (byte-identical generation to mincut_gated_forgetting_bench.rs) ─

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

struct Row {
    name: String,
    survival: f32,
    recall: f32,
    micros: u128,
}

fn main() {
    let seed: u64 = 341; // identical to mincut_gated_forgetting_bench.rs

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-agent-memory — Direct-Backend Mincut Forgetting (ADR-346)║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    println!("Platform  : {}", std::env::consts::OS);
    println!("Arch      : {}", std::env::consts::ARCH);
    println!();
    println!("Dataset (identical to the 2026-09-05 run, seed={seed})");
    println!("  Total memories  : {N_MEMORIES}  Target size: {TARGET_SIZE}  Dims: {DIMS}");
    println!();

    let cow = CoherencePolicy::default();

    let mut soft_wrapper =
        MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
    soft_wrapper.mincut_trials = MINCUT_TRIALS;
    let mut hard_wrapper =
        MincutGatedForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION);
    hard_wrapper.mincut_trials = MINCUT_TRIALS;

    let mut soft_direct =
        MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS)
            .with_backend(MincutBackend::Direct);
    soft_direct.mincut_trials = MINCUT_TRIALS;
    let mut hard_direct =
        MincutGatedForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION)
            .with_backend(MincutBackend::Direct);
    hard_direct.mincut_trials = MINCUT_TRIALS;

    let policies: [(&str, &dyn CompactionPolicy); 5] = [
        ("CoherencePolicy (baseline)", &cow),
        ("Soft-Wrapper (2026-09-05)", &soft_wrapper),
        ("Hard-Wrapper (2026-09-05)", &hard_wrapper),
        ("Soft-Direct (this run)", &soft_direct),
        ("Hard-Direct (this run)", &hard_direct),
    ];

    let mut rows = Vec::new();
    for (name, policy) in policies {
        let (survival, recall, dur) = run_policy(policy, seed);
        rows.push(Row {
            name: name.to_string(),
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
    let candidates = [
        ("Soft-Wrapper", &rows[1]),
        ("Hard-Wrapper", &rows[2]),
        ("Soft-Direct", &rows[3]),
        ("Hard-Direct", &rows[4]),
    ];

    println!("Acceptance test (thresholds unmodified from the 2026-09-05 run)");
    let mut all_direct_pass = true;
    for (label, row) in &candidates {
        let gap = (row.survival - baseline.survival) * 100.0;
        let gap_pass = gap >= BRIDGE_SURVIVAL_GAP_THRESHOLD_PP;
        let recall_delta = (row.recall - baseline.recall).abs();
        let recall_pass = recall_delta <= RECALL_TOLERANCE;
        let slowdown = row.micros as f64 / baseline.micros.max(1) as f64;
        let speed_pass = slowdown <= MAX_SLOWDOWN_VS_BASELINE;
        println!(
            "  {label:<12} gap={gap:+.1}pp[{}] recall_delta={:.2}pp[{}] slowdown={slowdown:.1}x[{}]",
            if gap_pass { "PASS" } else { "FAIL" },
            recall_delta * 100.0,
            if recall_pass { "PASS" } else { "FAIL" },
            if speed_pass { "PASS" } else { "FAIL" }
        );
        if label.ends_with("Direct") && !(gap_pass && recall_pass && speed_pass) {
            all_direct_pass = false;
        }
    }
    println!();

    let wrapper_speedup_soft = rows[1].micros as f64 / rows[3].micros.max(1) as f64;
    let wrapper_speedup_hard = rows[2].micros as f64 / rows[4].micros.max(1) as f64;
    println!(
        "Direct vs Wrapper speedup: Soft {wrapper_speedup_soft:.1}x, Hard {wrapper_speedup_hard:.1}x"
    );
    println!();

    if all_direct_pass {
        println!("=> ACCEPT: MincutBackend::Direct clears the performance gate the 2026-09-05 run failed, at matched effectiveness.");
    } else {
        println!("=> REJECT: at least one mandatory acceptance threshold still fails for the Direct backend (see above).");
        std::process::exit(1);
    }
}
