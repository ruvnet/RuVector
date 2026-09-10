//! Informational (non-gating) follow-up probe to `cactus_gated_forgetting_bench`:
//! is the single-seed (346) bridge-survival result representative, or an
//! artifact of that specific random corpus? Runs the same dataset generator
//! and both backends (ADR-345's single-trial `MincutGatedForgetting` and this
//! experiment's `CactusGatedForgetting`) across 10 independent seeds and
//! reports the distribution of the bridge-survival gap vs. baseline.
//!
//! Explicitly informational: this does not redefine or re-run the
//! pre-registered acceptance test in `cactus_gated_forgetting_bench` (which
//! stays fixed at seed 346, decided before any benchmark ran). It exists to
//! characterize *why* that test failed on the survival criterion.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example cactus_seed_sensitivity_probe --features mincut-forget-cactus

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, CactusGatedForgetting, CoherencePolicy, CoherenceWeights, CompactionPolicy,
    MemoryStore, MincutGatedForgetting,
};
use std::collections::HashSet;

const N_CLUSTERS: usize = 6;
const PER_CLUSTER: usize = 12;
const N_CORE: usize = N_CLUSTERS * PER_CLUSTER;
const N_BRIDGES: usize = 12;
const N_MEMORIES: usize = N_CORE + N_BRIDGES;
const N_HOT_CLUSTERS: usize = 2;
const DIMS: usize = 32;
const TARGET_SIZE: usize = N_MEMORIES / 2;
const CONTEXT_WINDOW_SIZE: usize = 10;
const N_COLD_ERA_ACCESSES: usize = 40;
const N_HOT_ERA_ACCESSES: usize = 80;
const HOT_ERA_HOT_FRAC: f64 = 0.90;
const STRUCTURAL_BONUS: f32 = 0.5;
const N_SEEDS: u64 = 10;

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
}

fn generate_dataset(store: &mut MemoryStore, rng: &mut StdRng) -> Dataset {
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS).map(|_| unit_gaussian(rng, DIMS)).collect();
    let mut cluster_of = Vec::with_capacity(N_MEMORIES);
    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..PER_CLUSTER {
            store.insert(perturb(centroid, 0.35, rng));
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
        let idx = store.len();
        store.insert(perturb(&mid, 0.15, rng));
        bridge_indices.insert(idx);
        cluster_of.push(usize::MAX);
    }
    Dataset {
        centroids,
        cluster_of,
        bridge_indices,
    }
}

fn simulate_accesses(
    store: &mut MemoryStore,
    dataset: &Dataset,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    for _ in 0..N_COLD_ERA_ACCESSES {
        store.access_by_index(rng.gen_range(0..N_MEMORIES));
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

fn survival_rate(policy: &dyn CompactionPolicy, seed: u64) -> f32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut store = MemoryStore::new(DIMS);
    let dataset = generate_dataset(&mut store, &mut rng);
    let mut rng2 = StdRng::seed_from_u64(seed + 1);
    let context_window = simulate_accesses(&mut store, &dataset, &mut rng2);
    let bridge_ids: HashSet<u64> = dataset
        .bridge_indices
        .iter()
        .map(|&i| store.entries()[i].id)
        .collect();
    compact(&mut store, policy, TARGET_SIZE, &context_window);
    let surviving = store
        .entries()
        .iter()
        .filter(|e| bridge_ids.contains(&e.id))
        .count();
    surviving as f32 / bridge_ids.len() as f32
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32;
    (mean, var.sqrt())
}

fn main() {
    let cow = CoherencePolicy::default();
    let mut old_soft = MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
    old_soft.mincut_trials = 1;
    let new_soft = CactusGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);

    let mut baseline_gaps_old = Vec::new();
    let mut baseline_gaps_new = Vec::new();
    println!(
        "{:<6} {:>14} {:>18} {:>18}",
        "seed", "baseline", "old(mincut) gap pp", "new(cactus) gap pp"
    );
    for i in 0..N_SEEDS {
        let seed = 1000 + i;
        let base = survival_rate(&cow, seed);
        let old = survival_rate(&old_soft, seed);
        let new = survival_rate(&new_soft, seed);
        let gap_old = (old - base) * 100.0;
        let gap_new = (new - base) * 100.0;
        baseline_gaps_old.push(gap_old);
        baseline_gaps_new.push(gap_new);
        println!(
            "{seed:<6} {:>13.1}% {:>17.1}pp {:>17.1}pp",
            base * 100.0,
            gap_old,
            gap_new
        );
    }
    let (mean_old, std_old) = mean_std(&baseline_gaps_old);
    let (mean_new, std_new) = mean_std(&baseline_gaps_new);
    let pass_old = baseline_gaps_old.iter().filter(|&&g| g >= 15.0).count();
    let pass_new = baseline_gaps_new.iter().filter(|&&g| g >= 15.0).count();
    println!();
    println!(
        "old(mincut) gap : mean={mean_old:+.1}pp std={std_old:.1}pp seeds_meeting_15pp_bar={pass_old}/{N_SEEDS}"
    );
    println!(
        "new(cactus) gap : mean={mean_new:+.1}pp std={std_new:.1}pp seeds_meeting_15pp_bar={pass_new}/{N_SEEDS}"
    );
}
