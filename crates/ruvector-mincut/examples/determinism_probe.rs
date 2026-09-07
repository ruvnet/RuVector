//! Nightly research probe (2026-09-07): reproduces and diagnoses the
//! non-determinism in `RuVectorGraphAnalyzer::partition()` first measured by
//! the 2026-09-05 nightly run (ADR-345, Open Question #2). Not part of the
//! shipped API surface -- this is raw measurement evidence for
//! `docs/research/nightly/2026-09-07-deterministic-mincut-witness/README.md`.
//!
//! Two fixed topologies, byte-identical across trials, each queried with a
//! *fresh* `RuVectorGraphAnalyzer` per trial (matching ADR-345's
//! methodology, which builds a new analyzer per trial rather than reusing
//! one with its result cache):
//!
//!  - `two_clique_bridge_19`: the exact 19-vertex topology from
//!    `mincut_determinism_probe.rs` (two 9-vertex near-duplicate clusters
//!    joined by one bridge vector), which measured ~50% empty/degenerate
//!    results in the prior run.
//!  - `two_clique_bridge_84`: a larger 84-entry variant (6 clusters x 12 +
//!    12 bridges) matching the corpus size used in the rejected
//!    `MincutGatedForgetting` benchmark, to check the finding holds at that
//!    scale too.
//!
//! For each topology we report: how many of N trials returned an
//! empty/degenerate partition, how many *distinct* (U, V\U) partitions were
//! observed across trials (byte-identical graph => should be 1 if
//! deterministic), and wall-clock.

use std::collections::HashSet;
use std::time::Instant;

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// Two near-duplicate clusters of `per_cluster` vectors each, joined by one
/// bridge vector interpolated 50/50 between the two cluster axes.
fn two_clique_bridge(per_cluster: usize) -> (Vec<Vec<f32>>, usize) {
    let mut entries: Vec<Vec<f32>> = Vec::new();
    let plain_a = vec![1.0f32, 0.0, 0.0];
    let plain_b = vec![0.0f32, 1.0, 0.0];
    // One gateway per cluster, each interpolated 50/50 toward the shared
    // bridge axis, matching the 2026-09-05 nightly's methodology (a single
    // gateway would leave one cluster with no path to the bridge at all).
    let gateway_a = normalize(&[1.0, 0.0, 0.5]);
    let gateway_b = normalize(&[0.0, 1.0, 0.5]);
    for _ in 0..per_cluster {
        entries.push(plain_a.clone());
    }
    entries.push(gateway_a);
    for _ in 0..per_cluster {
        entries.push(plain_b.clone());
    }
    entries.push(gateway_b);
    entries.push(vec![0.0, 0.0, 1.0]); // bridge
    let bridge_idx = entries.len() - 1;
    (entries, bridge_idx)
}

fn knn_edges(entries: &[Vec<f32>], k: usize, min_sim: f32) -> Vec<(usize, Vec<(usize, f64)>)> {
    let n = entries.len();
    (0..n)
        .map(|i| {
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_sim(&entries[i], &entries[j])))
                .filter(|&(_, s)| s >= min_sim)
                .collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims.truncate(k);
            let dists = sims
                .into_iter()
                .map(|(j, s)| (j, (1.0 - s).max(1e-4) as f64))
                .collect();
            (i, dists)
        })
        .collect()
}

struct TrialStats {
    trials: usize,
    empty_or_degenerate: usize,
    distinct_partitions: usize,
    elapsed_s: f64,
}

fn run_trials(neighbors: &[(usize, Vec<(usize, f64)>)], n: usize, trials: usize) -> TrialStats {
    let mut empty = 0usize;
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let t0 = Instant::now();
    for _ in 0..trials {
        let mut analyzer = ruvector_mincut::RuVectorGraphAnalyzer::from_knn(neighbors);
        match analyzer.partition() {
            None => empty += 1,
            Some((a, b)) => {
                if a.is_empty() || b.is_empty() || a.len() + b.len() != n {
                    empty += 1;
                    continue;
                }
                // Canonical signature: sorted membership of the smaller side.
                let mut sig: Vec<u64> = if a.len() <= b.len() { a } else { b };
                sig.sort_unstable();
                let bytes: Vec<u8> = sig.iter().flat_map(|v| v.to_le_bytes()).collect();
                seen.insert(bytes);
            }
        }
    }
    TrialStats {
        trials,
        empty_or_degenerate: empty,
        distinct_partitions: seen.len(),
        elapsed_s: t0.elapsed().as_secs_f64(),
    }
}

fn main() {
    let trials: usize = std::env::var("TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    for &per_cluster in &[8usize, 9usize, 41usize] {
        let (entries, _bridge_idx) = two_clique_bridge(per_cluster);
        let n = entries.len();
        let neighbors = knn_edges(&entries, 8, 0.05);
        let stats = run_trials(&neighbors, n, trials);
        println!(
            "n={n} trials={} empty_or_degenerate={} ({:.0}%) distinct_partitions_seen={} elapsed={:.2}s avg_ms={:.1}",
            stats.trials,
            stats.empty_or_degenerate,
            100.0 * stats.empty_or_degenerate as f64 / stats.trials as f64,
            stats.distinct_partitions,
            stats.elapsed_s,
            stats.elapsed_s * 1000.0 / stats.trials as f64,
        );
    }
}
