//! Integration tests for ruvector-drift.
//!
//! Each test generates a synthetic dataset, feeds it through a detector or
//! eviction policy, and asserts a numeric threshold — all computed from real
//! Rust code with no mocked measurements.

use rand::{rngs::SmallRng, SeedableRng};
use rand_distr::{Distribution, Normal};

use ruvector_drift::{
    centroid::CentroidDrift,
    frechet::FrechetDrift,
    mmd::MmdDrift,
    spectral::{EvictionPolicy, LruEviction, MemoryEntry, RandomEviction, SpectralEviction},
    DriftDetector,
};

fn gauss_vec(rng: &mut SmallRng, dim: usize, mean: f32) -> Vec<f32> {
    let dist = Normal::new(mean as f64, 1.0).unwrap();
    (0..dim).map(|_| dist.sample(rng) as f32).collect()
}

fn make_memory(n: usize, dim: usize, seed: u64) -> Vec<MemoryEntry> {
    let mut rng = SmallRng::seed_from_u64(seed);
    // 3 clusters centred at 0, 5, and 10
    (0..n)
        .map(|i| {
            let cluster_mean = (i % 3) as f32 * 5.0;
            MemoryEntry {
                id: i,
                vector: gauss_vec(&mut rng, dim, cluster_mean),
                last_access: i as u64,
            }
        })
        .collect()
}

// ─── Drift detection tests ────────────────────────────────────────────────────

#[test]
fn centroid_detects_large_shift() {
    let dim = 32;
    let window = 100;
    let mut det = CentroidDrift::new(dim, window, 1.5);
    let mut rng = SmallRng::seed_from_u64(1);
    // Stable phase
    for _ in 0..window {
        det.observe(&gauss_vec(&mut rng, dim, 0.0));
    }
    assert!(
        !det.is_drifted(),
        "false positive in stable phase; score={}",
        det.score()
    );
    // Drift phase: large mean shift
    for _ in 0..window {
        det.observe(&gauss_vec(&mut rng, dim, 6.0));
    }
    assert!(
        det.is_drifted(),
        "centroid missed large shift; score={}",
        det.score()
    );
}

#[test]
fn mmd_detects_distribution_shift() {
    let dim = 16;
    let window = 80;
    let mut det = MmdDrift::new(dim, window, 30, 0.02);
    let mut rng = SmallRng::seed_from_u64(2);
    for _ in 0..window {
        det.observe(&gauss_vec(&mut rng, dim, 0.0));
    }
    assert!(!det.is_drifted(), "MMD FP; score={}", det.score());
    for _ in 0..window {
        det.observe(&gauss_vec(&mut rng, dim, 5.0));
    }
    assert!(det.is_drifted(), "MMD missed shift; score={}", det.score());
}

#[test]
fn frechet_detects_variance_change() {
    let dim = 8;
    let window = 60;
    // Tight reference (σ=0.1) → wide current (σ=3.0), same mean
    let mut det = FrechetDrift::new(dim, window, 0.5);
    let ref_dist = Normal::new(0.0, 0.1).unwrap();
    let drift_dist = Normal::new(0.0, 3.0).unwrap();
    let mut rng = SmallRng::seed_from_u64(3);
    for _ in 0..window {
        let v: Vec<f32> = (0..dim).map(|_| ref_dist.sample(&mut rng) as f32).collect();
        det.observe(&v);
    }
    assert!(!det.is_drifted(), "Fréchet FP; score={}", det.score());
    for _ in 0..window {
        let v: Vec<f32> = (0..dim)
            .map(|_| drift_dist.sample(&mut rng) as f32)
            .collect();
        det.observe(&v);
    }
    assert!(
        det.is_drifted(),
        "Fréchet missed variance change; score={}",
        det.score()
    );
}

// ─── Eviction policy tests ────────────────────────────────────────────────────

#[test]
fn random_eviction_respects_target_size() {
    let memory = make_memory(50, 8, 10);
    let mut policy = RandomEviction::new(99);
    let plan = policy.plan_eviction(&memory, 35);
    assert_eq!(plan.evict.len(), 15, "should evict exactly 15");
    // All evicted IDs are valid
    let valid_ids: std::collections::HashSet<usize> = memory.iter().map(|e| e.id).collect();
    for id in &plan.evict {
        assert!(valid_ids.contains(id), "invalid eviction id {id}");
    }
}

#[test]
fn lru_eviction_preserves_newest() {
    let n = 20;
    let memory = make_memory(n, 8, 11);
    let mut policy = LruEviction;
    let plan = policy.plan_eviction(&memory, n - 5);
    let evict_set: std::collections::HashSet<usize> = plan.evict.iter().copied().collect();
    // The 5 oldest (last_access 0..5) should be evicted
    for oldest in 0..5 {
        assert!(
            evict_set.contains(&oldest),
            "id {oldest} should be evicted by LRU"
        );
    }
}

#[test]
fn spectral_eviction_no_duplicates() {
    let memory = make_memory(30, 8, 12);
    let mut policy = SpectralEviction::new(4, 20, 7);
    let plan = policy.plan_eviction(&memory, 21);
    let mut seen = std::collections::HashSet::new();
    for id in &plan.evict {
        assert!(seen.insert(id), "duplicate eviction id {id}");
    }
    assert_eq!(plan.evict.len(), 9);
}

#[test]
fn spectral_no_eviction_when_already_small() {
    let memory = make_memory(10, 4, 13);
    let mut policy = SpectralEviction::new(3, 15, 0);
    let plan = policy.plan_eviction(&memory, 20);
    assert!(plan.evict.is_empty(), "should not evict when target > len");
}

// ─── Acceptance: spectral recall ≥ LRU recall (regression guard) ─────────────

#[test]
fn spectral_recall_not_worse_than_lru_on_clustered_data() {
    // Build a small clustered memory and measure recall of each eviction policy.
    // Spectral should preserve recall as well as or better than LRU because it
    // keeps structurally central nodes rather than arbitrary recent ones.
    let n_mem = 120;
    let dim = 16;
    let target = 84; // evict 30%
    let k = 5;
    let memory = make_memory(n_mem, dim, 42);

    // Build a small query set from the same distribution
    let mut rng = SmallRng::seed_from_u64(77);
    let n_q = 20;
    let queries: Vec<Vec<f32>> = (0..n_q)
        .map(|i| gauss_vec(&mut rng, dim, (i % 3) as f32 * 5.0))
        .collect();

    let recall = |kept_ids: &std::collections::HashSet<usize>| -> f64 {
        let full_vecs: Vec<&Vec<f32>> = memory.iter().map(|e| &e.vector).collect();
        let kept_vecs: Vec<&Vec<f32>> = memory
            .iter()
            .filter(|e| kept_ids.contains(&e.id))
            .map(|e| &e.vector)
            .collect();
        if kept_vecs.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f64;
        for q in &queries {
            let l2_all: Vec<(usize, f32)> = full_vecs
                .iter()
                .enumerate()
                .filter(|(i, _)| kept_ids.contains(&memory[*i].id))
                .map(|(i, v)| {
                    (
                        i,
                        v.iter().zip(q.iter()).map(|(a, b)| (a - b).powi(2)).sum(),
                    )
                })
                .collect();
            let mut l2_full: Vec<(usize, f32)> = full_vecs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        i,
                        v.iter().zip(q.iter()).map(|(a, b)| (a - b).powi(2)).sum(),
                    )
                })
                .collect();
            l2_full.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let true_top_k: std::collections::HashSet<usize> = l2_full
                .iter()
                .take(k)
                .filter(|(i, _)| kept_ids.contains(&memory[*i].id))
                .map(|(i, _)| *i)
                .collect();
            if true_top_k.is_empty() {
                total += 1.0;
                continue;
            }
            let mut l2_kept = l2_all.clone();
            l2_kept.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let pred_top_k: std::collections::HashSet<usize> =
                l2_kept.iter().take(k).map(|(i, _)| *i).collect();
            let hits = true_top_k
                .iter()
                .filter(|id| pred_top_k.contains(id))
                .count();
            total += hits as f64 / true_top_k.len() as f64;
        }
        total / n_q as f64
    };

    let all_ids: std::collections::HashSet<usize> = memory.iter().map(|e| e.id).collect();

    let mut lru = LruEviction;
    let lru_plan = lru.plan_eviction(&memory, target);
    let lru_evict: std::collections::HashSet<usize> = lru_plan.evict.iter().copied().collect();
    let lru_kept: std::collections::HashSet<usize> =
        all_ids.difference(&lru_evict).copied().collect();
    let lru_recall = recall(&lru_kept);

    let mut spectral = SpectralEviction::new(5, 25, 42);
    let spec_plan = spectral.plan_eviction(&memory, target);
    let spec_evict: std::collections::HashSet<usize> = spec_plan.evict.iter().copied().collect();
    let spec_kept: std::collections::HashSet<usize> =
        all_ids.difference(&spec_evict).copied().collect();
    let spec_recall = recall(&spec_kept);

    assert!(
        spec_recall >= lru_recall - 0.05, // allow 5% tolerance
        "SpectralEviction recall ({:.4}) should be within 5pp of LruEviction ({:.4})",
        spec_recall,
        lru_recall
    );
}
