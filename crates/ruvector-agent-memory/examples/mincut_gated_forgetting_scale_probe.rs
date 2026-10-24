//! Exploratory follow-up probe (2026-09-28, ADR-350) — NOT part of the
//! pre-registered acceptance gate in `mincut_gated_forgetting_bench.rs`.
//!
//! The 2026-09-05 nightly (ADR-345) capped its corpus at 84 memories because
//! `RuVectorGraphAnalyzer::partition()` made anything larger computationally
//! infeasible (see `mincut_scaling_probe.rs`: 11.4s/call at 400 vertices).
//! `MincutBackend::DynamicMinCut` removes that ceiling (286ms at 2,000
//! vertices), which raises a question the original nightly could not
//! afford to ask: is the measured 0.0pp bridge-survival gap at n=84 a
//! property of mincut-gated forgetting itself, or an artifact of that
//! particular (small) corpus size?
//!
//! This probe reuses the exact same dataset-generation shape as
//! `mincut_gated_forgetting_bench.rs` (topic clusters + interpolated
//! bridges, hot-cluster access simulation), scaled by a `SCALE` factor, and
//! reports bridge-survival gap / recall / compaction latency at each scale
//! using only the `DynamicMinCut` backend (the `GraphAnalyzer` backend is
//! not run here — its cost at these sizes is already characterized by
//! `mincut_scaling_probe.rs` and is not the question this probe asks).
//!
//! This is genuinely new evidence, not a re-run of the pre-registered
//! benchmark: it does not change or supersede that benchmark's ACCEPT/REJECT
//! verdict (see "don't move the goalposts", ADR-345 Next Research item 3),
//! it answers a different, explicitly-labeled exploratory question.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example mincut_gated_forgetting_scale_probe --features mincut-forget

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, CoherencePolicy, CoherenceWeights, CompactionPolicy, MemoryStore,
    MincutBackend, MincutGatedForgetting,
};
use std::collections::HashSet;
use std::time::Instant;

const N_CLUSTERS: usize = 6;
const N_HOT_CLUSTERS: usize = 2;
const DIMS: usize = 32;
const N_QUERIES: usize = 20;
const K: usize = 5;
const STRUCTURAL_BONUS: f32 = 0.5;
const PROTECT_FRACTION: f32 = 0.2;
const MINCUT_TRIALS: usize = 1;

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

struct Dataset {
    centroids: Vec<Vec<f32>>,
    cluster_of: Vec<usize>,
    bridge_indices: HashSet<usize>,
    queries: Vec<(Vec<f32>, Vec<u64>)>,
    n_memories: usize,
    per_cluster: usize,
}

fn generate_dataset(store: &mut MemoryStore, rng: &mut StdRng, scale: usize) -> Dataset {
    let per_cluster = 12 * scale;
    let n_bridges = 12 * scale;
    let n_core = N_CLUSTERS * per_cluster;
    let n_memories = n_core + n_bridges;

    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS).map(|_| unit_gaussian(rng, DIMS)).collect();
    let mut cluster_of = Vec::with_capacity(n_memories);

    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..per_cluster {
            let v = perturb(centroid, 0.35, rng);
            store.insert(v);
            cluster_of.push(c);
        }
    }

    let mut bridge_indices = HashSet::new();
    for _ in 0..n_bridges {
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
        n_memories,
        per_cluster,
    }
}

fn simulate_accesses(
    store: &mut MemoryStore,
    dataset: &Dataset,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let n_cold = 40 * (dataset.n_memories / 84).max(1);
    let n_hot = 80 * (dataset.n_memories / 84).max(1);
    for _ in 0..n_cold {
        let idx = rng.gen_range(0..dataset.n_memories);
        store.access_by_index(idx);
    }
    let mut context_accesses: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_hot {
        let idx = if rng.gen_bool(0.90) {
            let hot_c = rng.gen_range(0..N_HOT_CLUSTERS);
            hot_c * dataset.per_cluster + rng.gen_range(0..dataset.per_cluster)
        } else {
            let cold_c = rng.gen_range(N_HOT_CLUSTERS..N_CLUSTERS);
            cold_c * dataset.per_cluster + rng.gen_range(0..dataset.per_cluster)
        };
        store.access_by_index(idx);
        let cluster = dataset.cluster_of[idx];
        if cluster != usize::MAX {
            context_accesses.push(dataset.centroids[cluster].clone());
        }
    }
    let start = context_accesses.len().saturating_sub(10);
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

fn run_policy(
    policy: &dyn CompactionPolicy,
    mincut_policy: Option<&MincutGatedForgetting>,
    seed: u64,
    scale: usize,
) -> (f32, f32, std::time::Duration, Option<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut store = MemoryStore::new(DIMS);
    let dataset = generate_dataset(&mut store, &mut rng, scale);
    let mut rng2 = StdRng::seed_from_u64(seed + 1);
    let context_window = simulate_accesses(&mut store, &dataset, &mut rng2);
    let target_size = dataset.n_memories / 2;

    let bridge_ids: HashSet<u64> = dataset
        .bridge_indices
        .iter()
        .map(|&i| store.entries()[i].id)
        .collect();

    // Only `MincutGatedForgetting` has a structural signal to report;
    // `CoherencePolicy` (the baseline) does not.
    let boundary_size = mincut_policy.map(|p| p.boundary_size(store.entries()));

    let t0 = Instant::now();
    compact(&mut store, policy, target_size, &context_window);
    let elapsed = t0.elapsed();

    let surviving_bridges = store
        .entries()
        .iter()
        .filter(|e| bridge_ids.contains(&e.id))
        .count();
    let survival_rate = surviving_bridges as f32 / bridge_ids.len() as f32;
    let recall = measure_recall(&dataset.queries, &store);
    (survival_rate, recall, elapsed, boundary_size)
}

fn main() {
    let seed: u64 = 341;
    println!("Scale | Memories | Policy                       | BridgeSurv | Recall@10 | Compaction(ms) | Boundary");
    println!("{}", "-".repeat(108));

    for &scale in &[1usize, 10, 50] {
        let n_memories = N_CLUSTERS * 12 * scale + 12 * scale;
        let cow = CoherencePolicy::default();
        let mut soft = MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS)
            .with_backend(MincutBackend::DynamicMinCut);
        soft.mincut_trials = MINCUT_TRIALS;
        let mut hard = MincutGatedForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION)
            .with_backend(MincutBackend::DynamicMinCut);
        hard.mincut_trials = MINCUT_TRIALS;

        let mut baseline_survival = 0.0;
        let runs: [(&str, &dyn CompactionPolicy, Option<&MincutGatedForgetting>); 3] = [
            ("CoherenceWeighted", &cow as &dyn CompactionPolicy, None),
            ("Soft (DynamicMinCut)", &soft, Some(&soft)),
            ("Hard (DynamicMinCut)", &hard, Some(&hard)),
        ];
        for (label, policy, mincut_policy) in runs {
            let (survival, recall, dur, boundary) = run_policy(policy, mincut_policy, seed, scale);
            if label == "CoherenceWeighted" {
                baseline_survival = survival;
            }
            let gap = (survival - baseline_survival) * 100.0;
            let boundary_str = boundary
                .map(|b| b.to_string())
                .unwrap_or_else(|| "-".to_string());
            println!(
                "{scale:>5} | {n_memories:>8} | {label:<28} | {:>9.1}% | {:>8.1}% | {:>13.2} (gap {gap:+.1}pp) | {boundary_str}",
                survival * 100.0,
                recall * 100.0,
                dur.as_secs_f64() * 1000.0,
            );
        }
        println!();
    }
}
