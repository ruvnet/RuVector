//! Exploratory follow-up (2026-09-17 nightly, ADR-346) — NOT part of the
//! registered hypothesis in `mincut_direct_backend_bench.rs` and does not
//! gate this run's ACCEPT/REJECT verdict (see the "don't move the
//! goalposts" rule). That benchmark reproduced the 2026-09-05 run's
//! bridge-survival gap of exactly 0.0pp even with the fast `Direct`
//! backend, at the same 84-memory corpus. This raises an obvious follow-on
//! question the 2026-09-05 report could not afford to ask because
//! `RuVectorGraphAnalyzer` made anything past a few hundred vertices too
//! slow to run: is the zero gap specific to the tiny 84-memory corpus, or
//! does it persist at the corpus size (thousands of memories) the original
//! experiment actually wanted to test? `Direct`'s measured scaling
//! (`mincut_direct_scaling_probe.rs`: ~18.6ms at n=400, k=8) makes that
//! affordable now, so this probe answers it directly.
//!
//! Same generator as `mincut_direct_backend_bench.rs`, scaled by a
//! multiplier applied to `PER_CLUSTER` and `N_BRIDGES` (cluster/bridge
//! *ratios* unchanged), run once per size with `MincutBackend::Direct`
//! only (the backend already shown to reproduce Wrapper's effectiveness
//! numbers exactly at n=84).
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example mincut_direct_scale_effectiveness_probe --features mincut-forget

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, CoherencePolicy, CoherenceWeights, CompactionPolicy, MemoryStore,
    MincutBackend, MincutGatedForgetting,
};
use std::collections::HashSet;
use std::time::Instant;

const N_CLUSTERS: usize = 6;
const DIMS: usize = 32;
const N_QUERIES: usize = 20;
const K: usize = 5;
const N_HOT_CLUSTERS: usize = 2;
const CONTEXT_WINDOW_SIZE: usize = 10;
const STRUCTURAL_BONUS: f32 = 0.5;

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
}

fn generate_dataset(
    store: &mut MemoryStore,
    rng: &mut StdRng,
    per_cluster: usize,
    n_bridges: usize,
) -> Dataset {
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS).map(|_| unit_gaussian(rng, DIMS)).collect();
    let mut cluster_of = Vec::new();
    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..per_cluster {
            store.insert(perturb(centroid, 0.35, rng));
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
        let idx = store.len();
        store.insert(perturb(&mid, 0.15, rng));
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
    n_memories: usize,
    per_cluster: usize,
    n_cold: usize,
    n_hot: usize,
) -> Vec<Vec<f32>> {
    for _ in 0..n_cold {
        let idx = rng.gen_range(0..n_memories);
        store.access_by_index(idx);
    }
    let mut context_accesses: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_hot {
        let idx = if rng.gen_bool(0.90) {
            let hot_c = rng.gen_range(0..N_HOT_CLUSTERS);
            hot_c * per_cluster + rng.gen_range(0..per_cluster)
        } else {
            let cold_c = rng.gen_range(N_HOT_CLUSTERS..N_CLUSTERS);
            cold_c * per_cluster + rng.gen_range(0..per_cluster)
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

/// Rebuilds a fresh, identically-seeded store+dataset (mirrors
/// `mincut_direct_backend_bench.rs::run_policy`, since `MemoryStore` is not
/// `Clone`), runs one compaction policy, and reports (bridge survival rate,
/// recall, compaction wall-clock).
struct ScaleParams {
    seed: u64,
    per_cluster: usize,
    n_bridges: usize,
    n_memories: usize,
    target_size: usize,
    n_cold: usize,
    n_hot: usize,
}

fn run_one(policy: &dyn CompactionPolicy, p: &ScaleParams) -> (f32, f32, std::time::Duration) {
    let mut rng = StdRng::seed_from_u64(p.seed);
    let mut store = MemoryStore::new(DIMS);
    let dataset = generate_dataset(&mut store, &mut rng, p.per_cluster, p.n_bridges);
    let mut rng2 = StdRng::seed_from_u64(p.seed + 1);
    let context_window = simulate_accesses(
        &mut store,
        &dataset,
        &mut rng2,
        p.n_memories,
        p.per_cluster,
        p.n_cold,
        p.n_hot,
    );
    let bridge_ids: HashSet<u64> = dataset
        .bridge_indices
        .iter()
        .map(|&i| store.entries()[i].id)
        .collect();

    let t0 = Instant::now();
    compact(&mut store, policy, p.target_size, &context_window);
    let elapsed = t0.elapsed();

    let survival = store
        .entries()
        .iter()
        .filter(|e| bridge_ids.contains(&e.id))
        .count() as f32
        / bridge_ids.len() as f32;
    let recall = measure_recall(&dataset.queries, &store);
    (survival, recall, elapsed)
}

fn run_at_scale(multiplier: usize, seed: u64) {
    let per_cluster = 12 * multiplier;
    let n_bridges = 12 * multiplier;
    let n_memories = N_CLUSTERS * per_cluster + n_bridges;
    let params = ScaleParams {
        seed,
        per_cluster,
        n_bridges,
        n_memories,
        target_size: n_memories / 2,
        n_cold: 40 * multiplier,
        n_hot: 80 * multiplier,
    };

    let cow = CoherencePolicy::default();
    let (baseline_survival, baseline_recall, _) = run_one(&cow, &params);

    let mut soft = MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS)
        .with_backend(MincutBackend::Direct);
    soft.mincut_trials = 1;
    let (soft_survival, soft_recall, elapsed) = run_one(&soft, &params);

    let gap = (soft_survival - baseline_survival) * 100.0;
    println!(
        "n={n_memories:<6} baseline_surv={:>6.1}% soft_surv={:>6.1}% gap={gap:+7.1}pp recall_delta={:+.2}pp compaction={:>8.1}ms",
        baseline_survival * 100.0,
        soft_survival * 100.0,
        (soft_recall - baseline_recall) * 100.0,
        elapsed.as_secs_f64() * 1000.0,
    );
}

fn main() {
    println!(
        "Exploratory scale-effectiveness probe (Direct backend, Soft policy, mincut_trials=1)"
    );
    println!("Not part of the registered acceptance test — see file header.\n");
    for &multiplier in &[1usize, 2, 4, 8, 16] {
        run_at_scale(multiplier, 341);
    }
}
