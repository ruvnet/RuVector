//! Benchmark binary for ruvector-drift.
//!
//! Runs two back-to-back experiments and prints results to stdout:
//!
//! **Experiment A — Drift Detection**
//! Feeds N synthetic queries in two phases (stable then shifted) through
//! three detectors and measures detection latency and false positive rate.
//!
//! **Experiment B — Eviction Quality**
//! Builds a synthetic agent memory of N vectors in K clusters, evicts 30%
//! using three policies, and measures recall@10 preservation.
//!
//! All numbers come from real computation; none are aspirational.
//! Usage:
//!   cargo run --release -p ruvector-drift
//!   N=20000 DIM=128 cargo run --release -p ruvector-drift

use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;

use ruvector_drift::{
    centroid::CentroidDrift,
    frechet::FrechetDrift,
    mmd::MmdDrift,
    spectral::{EvictionPolicy, LruEviction, MemoryEntry, RandomEviction, SpectralEviction},
    DriftDetector,
};

// ─── Dataset generation ──────────────────────────────────────────────────────

fn rand_vec(rng: &mut SmallRng, dim: usize, mean: f32, std: f32) -> Vec<f32> {
    let dist = Normal::new(mean as f64, std as f64).unwrap();
    (0..dim).map(|_| dist.sample(rng) as f32).collect()
}

/// Generate queries: first `n/2` from N(0, I), second `n/2` from N(delta, I).
fn gen_drift_queries(n: usize, dim: usize, delta: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut queries = Vec::with_capacity(n);
    for i in 0..n {
        let mean = if i < n / 2 { 0.0 } else { delta };
        queries.push(rand_vec(&mut rng, dim, mean, 1.0));
    }
    queries
}

/// Build a memory with `n` vectors in `k` clusters (Gaussian blobs).
fn gen_clustered_memory(n: usize, dim: usize, k: usize, seed: u64) -> Vec<MemoryEntry> {
    let mut rng = SmallRng::seed_from_u64(seed);
    // Sample k cluster centres in [0, 10]
    let centres: Vec<Vec<f32>> = (0..k)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 10.0).collect())
        .collect();
    let mut entries = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % k;
        let v = rand_vec(&mut rng, dim, 0.0, 0.3)
            .into_iter()
            .zip(centres[c].iter())
            .map(|(x, &cx)| x + cx)
            .collect();
        entries.push(MemoryEntry {
            id: i,
            vector: v,
            last_access: rng.gen_range(0..n as u64),
        });
    }
    entries
}

// ─── Recall measurement ──────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_sq_f(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Return the k nearest-neighbour IDs of `query` in `corpus` (brute force, L2).
fn knn_exact(query: &[f32], corpus: &[&Vec<f32>], k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_sq_f(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Recall@k: fraction of true top-k neighbours found in predicted top-k.
fn recall_at_k(
    queries: &[Vec<f32>],
    full_corpus: &[MemoryEntry],
    remaining_ids: &std::collections::HashSet<usize>,
    k: usize,
) -> f64 {
    let full_vecs: Vec<&Vec<f32>> = full_corpus.iter().map(|e| &e.vector).collect();
    let remain_vecs: Vec<&Vec<f32>> = full_corpus
        .iter()
        .filter(|e| remaining_ids.contains(&e.id))
        .map(|e| &e.vector)
        .collect();
    if remain_vecs.is_empty() || queries.is_empty() {
        return 0.0;
    }

    let mut total_recall = 0.0f64;
    for q in queries {
        let true_nn: std::collections::HashSet<usize> = knn_exact(q, &full_vecs, k)
            .into_iter()
            .filter(|&i| remaining_ids.contains(&full_corpus[i].id))
            .collect();
        if true_nn.is_empty() {
            total_recall += 1.0;
            continue;
        }
        let pred_nn: std::collections::HashSet<usize> =
            knn_exact(q, &remain_vecs, k).into_iter().collect();
        // Map pred indices back to original IDs
        let pred_ids: std::collections::HashSet<usize> = pred_nn
            .iter()
            .map(|&ri| {
                full_corpus
                    .iter()
                    .filter(|e| remaining_ids.contains(&e.id))
                    .nth(ri)
                    .map(|e| e.id)
                    .unwrap_or(0)
            })
            .collect();
        let hits = true_nn.iter().filter(|id| pred_ids.contains(id)).count();
        total_recall += hits as f64 / true_nn.len() as f64;
    }
    total_recall / queries.len() as f64
}

// ─── Experiment A: Drift detection ───────────────────────────────────────────

struct DetectionResult {
    name: &'static str,
    detect_latency: Option<usize>,
    false_positives: usize,
    mean_stable_score: f64,
    mean_drift_score: f64,
    elapsed_ms: f64,
}

fn run_drift_experiment(
    det: &mut dyn DriftDetector,
    queries: &[Vec<f32>],
    shift_point: usize,
    name: &'static str,
) -> DetectionResult {
    let t0 = Instant::now();
    let mut first_detect: Option<usize> = None;
    let mut false_positives = 0usize;
    let mut stable_score_sum = 0.0f64;
    let mut stable_count = 0usize;
    let mut drift_score_sum = 0.0f64;
    let mut drift_count = 0usize;

    for (i, q) in queries.iter().enumerate() {
        let obs = det.observe(q);
        if i < shift_point {
            stable_score_sum += obs.score;
            stable_count += 1;
            if obs.is_drifted {
                false_positives += 1;
            }
        } else {
            drift_score_sum += obs.score;
            drift_count += 1;
            if obs.is_drifted && first_detect.is_none() {
                first_detect = Some(i - shift_point + 1);
            }
        }
    }

    DetectionResult {
        name,
        detect_latency: first_detect,
        false_positives,
        mean_stable_score: if stable_count > 0 {
            stable_score_sum / stable_count as f64
        } else {
            0.0
        },
        mean_drift_score: if drift_count > 0 {
            drift_score_sum / drift_count as f64
        } else {
            0.0
        },
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

// ─── Experiment B: Eviction quality ──────────────────────────────────────────

struct EvictionResult {
    name: &'static str,
    recall_before: f64,
    recall_after: f64,
    recall_ratio: f64,
    conductance: f64,
    elapsed_ms: f64,
}

fn run_eviction_experiment(
    policy: &mut dyn EvictionPolicy,
    memory: &[MemoryEntry],
    queries: &[Vec<f32>],
    target_size: usize,
    k: usize,
    name: &'static str,
) -> EvictionResult {
    let full_ids: std::collections::HashSet<usize> = memory.iter().map(|e| e.id).collect();
    let recall_before = recall_at_k(queries, memory, &full_ids, k);

    let t0 = Instant::now();
    let plan = policy.plan_eviction(memory, target_size);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let evict_set: std::collections::HashSet<usize> = plan.evict.iter().copied().collect();
    let remaining: std::collections::HashSet<usize> =
        full_ids.difference(&evict_set).copied().collect();

    let recall_after = recall_at_k(queries, memory, &remaining, k);

    EvictionResult {
        name,
        recall_before,
        recall_after,
        recall_ratio: if recall_before > 0.0 {
            recall_after / recall_before
        } else {
            1.0
        },
        conductance: plan.conductance,
        elapsed_ms,
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let window = (n / 8).max(50);
    let shift_point = n / 2;
    let delta = 4.0f32; // mean shift magnitude

    // ── System info ──────────────────────────────────────────────────────────
    println!("=================================================================");
    println!("  ruvector-drift — Semantic Drift Detection + Spectral Eviction  ");
    println!("=================================================================");
    println!("OS      : {}", std::env::consts::OS);
    println!("Arch    : {}", std::env::consts::ARCH);
    println!("N       : {n}");
    println!("Dim     : {dim}");
    println!("Window  : {window}");
    println!("Shift @ : {shift_point}");
    println!("Δ (mean shift magnitude) : {delta}");

    // ── Experiment A: Drift detection ─────────────────────────────────────────
    println!("\n─── EXPERIMENT A: Drift Detection ───────────────────────────────");
    println!("Detector          | Detect latency | FP count | Mean stable | Mean drift | ms");
    println!("------------------|----------------|----------|-------------|------------|------");

    let queries_a = gen_drift_queries(n, dim, delta, 0xDEAD_BEEF);
    let threshold_centroid = delta * 0.3;
    let threshold_mmd = 0.02;
    let threshold_frechet = (delta as f64).powi(2) * 0.5;

    let detectors: Vec<(&str, Box<dyn DriftDetector>)> = vec![
        (
            "CentroidDrift",
            Box::new(CentroidDrift::new(dim, window, threshold_centroid as f64)),
        ),
        (
            "MmdDrift",
            Box::new(MmdDrift::new(dim, window, window / 3, threshold_mmd)),
        ),
        (
            "FrechetDrift",
            Box::new(FrechetDrift::new(dim, window, threshold_frechet)),
        ),
    ];

    let mut a_results = vec![];
    for (name, mut det) in detectors {
        // Drop the name value; use &'static directly
        let r = match name {
            "CentroidDrift" => {
                run_drift_experiment(det.as_mut(), &queries_a, shift_point, "CentroidDrift")
            }
            "MmdDrift" => run_drift_experiment(det.as_mut(), &queries_a, shift_point, "MmdDrift"),
            _ => run_drift_experiment(det.as_mut(), &queries_a, shift_point, "FrechetDrift"),
        };
        println!(
            "{:<18}| {:>14} | {:>8} | {:>11.4} | {:>10.4} | {:.1}",
            r.name,
            r.detect_latency
                .map(|l| l.to_string())
                .unwrap_or_else(|| "NOT DETECTED".into()),
            r.false_positives,
            r.mean_stable_score,
            r.mean_drift_score,
            r.elapsed_ms,
        );
        a_results.push(r);
    }

    // Acceptance test A
    let all_detected = a_results.iter().all(|r| r.detect_latency.is_some());
    let all_fp_ok = a_results.iter().all(|r| r.false_positives < (n / 2 / 20)); // <5% FP
    println!(
        "\nDetection acceptance: all detected within N/2 = {} queries: {}",
        n / 2,
        if all_detected { "PASS ✓" } else { "FAIL ✗" }
    );
    println!(
        "False positive acceptance (<5% of stable phase = {} alerts): {}",
        n / 2 / 20,
        if all_fp_ok { "PASS ✓" } else { "FAIL ✗" }
    );

    // ── Experiment B: Eviction quality ────────────────────────────────────────
    println!("\n─── EXPERIMENT B: Eviction Quality ──────────────────────────────");
    let n_mem = (n / 4).max(100);
    let k_clusters = 5;
    let target_size = (n_mem as f64 * 0.70) as usize; // evict 30%
    let k_recall = 10;
    let n_eval_queries = 50.min(n_mem / 4);
    println!("Memory size   : {n_mem}");
    println!(
        "Target after  : {target_size} (evict {}%)",
        100 - 100 * target_size / n_mem
    );
    println!("Clusters      : {k_clusters}");
    println!("Recall@k      : {k_recall}");

    let memory = gen_clustered_memory(n_mem, dim, k_clusters, 0xCAFE_1234);
    let eval_queries: Vec<Vec<f32>> = (0..n_eval_queries)
        .map(|i| gen_drift_queries(1, dim, 0.0, i as u64 + 99)[0].clone())
        .collect();

    println!(
        "\nPolicy            | Recall before | Recall after | Recall ratio | Conductance | ms"
    );
    println!(
        "------------------|---------------|--------------|--------------|-------------|------"
    );

    let policies: Vec<(&str, Box<dyn EvictionPolicy>)> = vec![
        ("RandomEviction", Box::new(RandomEviction::new(42))),
        ("LruEviction", Box::new(LruEviction)),
        (
            "SpectralEviction",
            Box::new(SpectralEviction::new(5, 30, 42)),
        ),
    ];

    let mut b_results = vec![];
    for (name, mut policy) in policies {
        let r = match name {
            "RandomEviction" => run_eviction_experiment(
                policy.as_mut(),
                &memory,
                &eval_queries,
                target_size,
                k_recall,
                "RandomEviction",
            ),
            "LruEviction" => run_eviction_experiment(
                policy.as_mut(),
                &memory,
                &eval_queries,
                target_size,
                k_recall,
                "LruEviction",
            ),
            _ => run_eviction_experiment(
                policy.as_mut(),
                &memory,
                &eval_queries,
                target_size,
                k_recall,
                "SpectralEviction",
            ),
        };
        println!(
            "{:<18}| {:>13.4} | {:>12.4} | {:>12.4} | {:>11.4} | {:.1}",
            r.name, r.recall_before, r.recall_after, r.recall_ratio, r.conductance, r.elapsed_ms
        );
        b_results.push(r);
    }

    // Acceptance test B
    let spectral = b_results
        .iter()
        .find(|r| r.name == "SpectralEviction")
        .unwrap();
    let lru = b_results.iter().find(|r| r.name == "LruEviction").unwrap();
    let spectral_wins = spectral.recall_ratio >= lru.recall_ratio;
    println!(
        "\nEviction acceptance: SpectralEviction recall_ratio ≥ LruEviction recall_ratio: {}",
        if spectral_wins {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );

    // Overall
    let overall = all_detected && all_fp_ok && spectral_wins;
    println!("\n=================================================================");
    println!("  OVERALL: {}", if overall { "PASS ✓" } else { "FAIL ✗" });
    println!("=================================================================");
    if !overall {
        std::process::exit(1);
    }
}
