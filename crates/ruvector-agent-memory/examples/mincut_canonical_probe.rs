//! Nightly follow-up probe (2026-09-20), attacking the two open findings
//! left by ADR-345 / docs/research/nightly/2026-09-05-mincut-gated-forgetting
//! ("Next Research" items 1-2): does `ruvector_mincut`'s ADR-117
//! pseudo-deterministic `canonical::source_anchored::SourceAnchoredMinCut`
//! engine avoid the non-determinism and the measured latency overhead of
//! `RuVectorGraphAnalyzer::partition()` (`crate::wrapper::MinCutWrapper`),
//! when driven from the exact same inputs?
//!
//! Reuses, byte-for-byte, the two corpora from the prior nightly's probes so
//! the numbers are directly comparable (same hypothesis, same corpus, same
//! measurement methodology, per the nightly process's "don't move the
//! goalposts" rule):
//!
//! - `mincut_determinism_probe.rs`'s 19-vertex two-clique-plus-bridge graph
//!   (k=8, min_sim=0.05), repeated calls on byte-identical input.
//! - `mincut_scaling_probe.rs`'s fixed-degree ring k-NN graphs at
//!   n = 19, 50, 100, 200, 400 (k=8).
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example mincut_canonical_probe --features mincut-forget

use ruvector_mincut::{SourceAnchoredConfig, SourceAnchoredMinCut};
use std::collections::HashSet;
use std::time::Instant;

fn normalize3(v: [f32; 3]) -> Vec<f32> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    vec![v[0] / n, v[1] / n, v[2] / n]
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

/// `(vertex, [(neighbor, distance)])` k-NN adjacency shape shared with
/// `RuVectorGraphAnalyzer::from_knn`.
type KnnNeighbors = Vec<(usize, Vec<(usize, f64)>)>;

/// `neighbors` in the [`KnnNeighbors`] shape -> flattened, deduplicated
/// undirected edge list `(u, v, weight)` for `SourceAnchoredMinCut::with_edges`,
/// using the identical distance-to-weight inversion (`weight = 1/distance`)
/// so the two backends see the same edge weights.
fn edges_from_knn(neighbors: &[(usize, Vec<(usize, f64)>)]) -> Vec<(u64, u64, f64)> {
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    let mut edges = Vec::new();
    for &(i, ref nbrs) in neighbors {
        for &(j, dist) in nbrs {
            let (u, v) = if i < j {
                (i as u64, j as u64)
            } else {
                (j as u64, i as u64)
            };
            if seen.insert((u, v)) {
                let weight = if dist > 0.0 { 1.0 / dist } else { 1.0 };
                edges.push((u, v, weight));
            }
        }
    }
    edges
}

fn bridge_dataset_neighbors() -> (KnnNeighbors, usize) {
    let mut entries: Vec<Vec<f32>> = Vec::new();
    for axis in 0..2 {
        let plain = if axis == 0 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let gateway = if axis == 0 {
            normalize3([1.0, 0.0, 0.5])
        } else {
            normalize3([0.0, 1.0, 0.5])
        };
        for _ in 0..8 {
            entries.push(plain.to_vec());
        }
        entries.push(gateway);
    }
    entries.push(vec![0.0, 0.0, 1.0]);
    let n = entries.len();
    let bridge_idx = n - 1;

    let k = 8usize;
    let min_sim = 0.05f32;
    let neighbors: Vec<(usize, Vec<(usize, f64)>)> = (0..n)
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
        .collect();
    (neighbors, bridge_idx)
}

fn run_determinism_probe() {
    let (neighbors, bridge_idx) = bridge_dataset_neighbors();
    let edges = edges_from_knn(&neighbors);

    let trials: usize = std::env::var("TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let mut empty = 0usize;
    let mut bridge_detected_boundary = 0usize;
    // Track every distinct `side_vertices` set observed, to directly measure
    // determinism (not just "did the bridge get flagged").
    let mut distinct_partitions: HashSet<Vec<u64>> = HashSet::new();

    let t0 = Instant::now();
    for _ in 0..trials {
        let mut engine =
            SourceAnchoredMinCut::with_edges(edges.clone(), SourceAnchoredConfig::default())
                .expect("build canonical min-cut engine");
        match engine.canonical_cut() {
            None => empty += 1,
            Some(cut) => {
                if cut.side_vertices.is_empty() || cut.side_size == edges_vertex_count(&edges) {
                    empty += 1;
                    continue;
                }
                distinct_partitions.insert(cut.side_vertices.clone());
                let side_a: HashSet<u64> = cut.side_vertices.iter().copied().collect();
                let bridge_in_a = side_a.contains(&(bridge_idx as u64));
                let boundary = cut
                    .cut_edges
                    .iter()
                    .any(|&(u, v)| u == bridge_idx as u64 || v == bridge_idx as u64);
                let _ = bridge_in_a;
                if boundary {
                    bridge_detected_boundary += 1;
                }
            }
        }
    }
    let elapsed = t0.elapsed();
    println!("=== determinism probe (SourceAnchoredMinCut, canonical backend) ===");
    println!(
        "trials={trials} elapsed={:.3}s avg_per_call={:.3}ms empty_or_degenerate={empty} ({:.0}%) \
         bridge_detected_as_boundary={bridge_detected_boundary} ({:.0}%) distinct_partitions={}",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / trials as f64,
        100.0 * empty as f64 / trials as f64,
        100.0 * bridge_detected_boundary as f64 / trials as f64,
        distinct_partitions.len(),
    );
}

fn edges_vertex_count(edges: &[(u64, u64, f64)]) -> usize {
    let mut vs: HashSet<u64> = HashSet::new();
    for &(u, v, _) in edges {
        vs.insert(u);
        vs.insert(v);
    }
    vs.len()
}

fn run_scaling_probe() {
    println!("=== scaling probe (SourceAnchoredMinCut, canonical backend) ===");
    let sizes = [19usize, 50, 100, 200, 400];
    let k = 8usize;
    for &n in &sizes {
        let neighbors: Vec<(usize, Vec<(usize, f64)>)> = (0..n)
            .map(|i| {
                let nbrs: Vec<(usize, f64)> = (1..=k)
                    .map(|d| ((i + d) % n, 0.1 + (d as f64) * 0.01))
                    .collect();
                (i, nbrs)
            })
            .collect();
        let edges = edges_from_knn(&neighbors);

        let t0 = Instant::now();
        let mut engine = SourceAnchoredMinCut::with_edges(edges, SourceAnchoredConfig::default())
            .expect("build canonical min-cut engine");
        let build_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let _ = engine.canonical_cut();
        let cut_elapsed = t1.elapsed();

        println!(
            "n={n:<5} build={:>10.3}ms  canonical_cut={:>10.3}ms",
            build_elapsed.as_secs_f64() * 1000.0,
            cut_elapsed.as_secs_f64() * 1000.0
        );
    }
}

fn main() {
    run_determinism_probe();
    run_scaling_probe();
}
