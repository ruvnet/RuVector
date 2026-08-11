//! Integration and acceptance tests for bandit-tuned ANN.
//!
//! All tests use deterministic data and must pass with real measurements.

use rand::SeedableRng;
use ruvector_bandit_ann::{
    bandit::{ThompsonBandit, Ucb1Bandit},
    dataset::{ground_truth, random_queries, random_unit_vectors},
    hnsw::Hnsw,
    recall_at_k, AnnVariant, BanditTuned, Hit, StaticDefault, StaticFast,
};

// ─── helper ──────────────────────────────────────────────────────────────────

fn make_gt(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<Hit> {
    ground_truth(data, query, k)
        .into_iter()
        .map(|(id, dist)| Hit { id, dist })
        .collect()
}

// ─── HNSW correctness ────────────────────────────────────────────────────────

#[test]
fn hnsw_recall_at_k10_above_threshold() {
    let n = 5_000;
    let dim = 64;
    let k = 10;
    let data = random_unit_vectors(n, dim, 101);
    let queries = random_queries(50, dim, 101);

    let mut h = Hnsw::new(16, 100);
    for (id, v) in data.iter().enumerate() {
        h.insert(id, v);
    }

    let mut total_recall = 0.0f32;
    for q in &queries {
        let results: Vec<Hit> = h
            .search(q, k, 30)
            .into_iter()
            .map(|(id, dist)| Hit { id, dist })
            .collect();
        let gt = make_gt(&data, q, k);
        total_recall += recall_at_k(&results, &gt, k);
    }
    let recall = total_recall / queries.len() as f32;
    assert!(
        recall >= 0.70,
        "HNSW ef=30 recall@10 = {:.3}, expected >= 0.70",
        recall
    );
}

// ─── StaticDefault vs StaticFast ─────────────────────────────────────────────

#[test]
fn static_default_higher_recall_than_static_fast() {
    let n = 3_000;
    let dim = 64;
    let k = 10;
    let data = random_unit_vectors(n, dim, 200);
    let queries = random_queries(100, dim, 200);

    let v1 = StaticDefault::build(&data, 12, 80);
    let v2 = StaticFast::build(&data, 12, 80);

    let mut r1 = 0.0f32;
    let mut r2 = 0.0f32;
    for q in &queries {
        let gt = make_gt(&data, q, k);
        r1 += recall_at_k(&v1.search(q, k), &gt, k);
        r2 += recall_at_k(&v2.search(q, k), &gt, k);
    }
    r1 /= queries.len() as f32;
    r2 /= queries.len() as f32;

    assert!(
        r1 >= r2,
        "StaticDefault recall ({:.3}) should be >= StaticFast ({:.3})",
        r1,
        r2
    );
}

// ─── Bandit convergence ───────────────────────────────────────────────────────

#[test]
fn ucb1_bandit_converges_away_from_worst_arm() {
    let n = 2_000;
    let dim = 32;
    let k = 5;
    let data = random_unit_vectors(n, dim, 42);
    let queries = random_queries(300, dim, 42);

    let arms = vec![5usize, 15, 30, 50]; // arm 0 (ef=5) should be identified as worst
    let mut bt = BanditTuned::build(&data, 10, 60, arms.clone());

    // Feed 200 queries with feedback.
    for (i, q) in queries.iter().take(200).enumerate() {
        let gt = make_gt(&data, q, k);
        let t = std::time::Instant::now();
        let _ = bt.search_with_feedback(q, k, &gt, t.elapsed().as_nanos() as f64 / 1_000.0);
    }

    let best_arm = bt.bandit.best_arm();
    let best_ef = arms[best_arm];
    // The lowest ef (5) should not win — recall there is too poor.
    assert_ne!(
        best_arm, 0,
        "Bandit should not converge to ef=5 (worst recall); selected ef={}",
        best_ef
    );
}

#[test]
fn bandit_tuned_recall_at_least_80_pct_after_warmup() {
    let n = 3_000;
    let dim = 64;
    let k = 10;
    let data = random_unit_vectors(n, dim, 77);
    let queries = random_queries(200, dim, 77);

    let arms = vec![10usize, 20, 30, 40, 50];
    let mut bt = BanditTuned::build(&data, 12, 100, arms);

    // Warm up.
    for (i, q) in queries.iter().take(100).enumerate() {
        let gt = make_gt(&data, q, k);
        let t = std::time::Instant::now();
        let _ = bt.search_with_feedback(q, k, &gt, t.elapsed().as_nanos() as f64 / 1_000.0);
    }

    // Evaluate on remaining queries.
    let mut total = 0.0f32;
    for q in queries.iter().skip(100) {
        let gt = make_gt(&data, q, k);
        let results = bt.search(q, k);
        total += recall_at_k(&results, &gt, k);
    }
    let recall = total / 100.0;
    assert!(
        recall >= 0.80,
        "BanditTuned recall@10 after warmup = {:.3}, expected >= 0.80",
        recall
    );
}

// ─── Thompson sampling ────────────────────────────────────────────────────────

#[test]
fn thompson_bandit_selects_best_arm_statistically() {
    let mut b = ThompsonBandit::new(4);
    let true_rewards = [0.2, 0.9, 0.5, 0.3]; // arm 1 is best
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);

    // Use rand::Rng trait for gen() call
    use rand::Rng;
    for _ in 0..400 {
        let arm = b.select(&mut rng);
        let r = (true_rewards[arm] + (rng.gen::<f32>() - 0.5) * 0.1).clamp(0.0, 1.0);
        b.update(arm, r);
    }
    assert_eq!(b.best_arm(), 1, "Thompson should identify arm 1 as best");
}

// ─── Memory estimates ─────────────────────────────────────────────────────────

#[test]
fn memory_estimates_are_sane() {
    let n = 1_000;
    let dim = 128;
    let data = random_unit_vectors(n, dim, 9);
    let v = StaticDefault::build(&data, 16, 80);

    // Minimum: n * dim * 4 bytes (just the vectors).
    let min_bytes = n * dim * 4;
    // Maximum: 5x that (generous upper bound including neighbors).
    let max_bytes = min_bytes * 5;
    let reported = v.memory_bytes();
    assert!(
        reported >= min_bytes && reported <= max_bytes,
        "memory_bytes() = {} not in [{}, {}]",
        reported,
        min_bytes,
        max_bytes
    );
}
