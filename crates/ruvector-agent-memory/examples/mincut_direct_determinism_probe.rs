//! Throwaway measurement (2026-09-17 nightly, ADR-346) checking whether
//! `MincutBackend::Direct` (`DynamicMinCut` via `MinCutBuilder`) is
//! deterministic on the same fixed, byte-identical 19-vertex two-clique-
//! plus-bridge topology `mincut_determinism_probe.rs` used to characterize
//! `RuVectorGraphAnalyzer`'s non-determinism. Not itself part of the
//! shipped research artifact.

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

fn main() {
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

    let trials: usize = std::env::var("TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let mut empty = 0usize;
    let mut bridge_detected_boundary = 0usize;
    let mut distinct_cut_values: HashSet<u64> = HashSet::new();
    let t0 = Instant::now();
    for _ in 0..trials {
        use ruvector_mincut::{DynamicGraph, MinCutBuilder};
        let graph = DynamicGraph::new();
        for (vertex, nbrs) in &neighbors {
            for &(neighbor, distance) in nbrs {
                let weight = if distance > 0.0 { 1.0 / distance } else { 1.0 };
                let _ = graph.insert_edge(*vertex as u64, neighbor as u64, weight);
            }
        }
        let edges: Vec<(u64, u64, f64)> = graph
            .edges()
            .into_iter()
            .map(|e| (e.source, e.target, e.weight))
            .collect();
        let mincut = MinCutBuilder::new()
            .with_edges(edges)
            .build()
            .expect("valid bridge graph builds");
        distinct_cut_values.insert(mincut.min_cut_value().to_bits());
        if mincut.min_cut_value() <= 0.0 {
            empty += 1;
            continue;
        }
        let (a, b) = mincut.partition();
        if a.is_empty() || b.is_empty() {
            empty += 1;
            continue;
        }
        let a_set: HashSet<u64> = a.iter().copied().collect();
        let mut boundary = false;
        for (i, nbrs) in &neighbors {
            let i_in_a = a_set.contains(&(*i as u64));
            for &(j, _) in nbrs {
                let j_in_a = a_set.contains(&(j as u64));
                if i_in_a != j_in_a && (*i == bridge_idx || j == bridge_idx) {
                    boundary = true;
                }
            }
        }
        if boundary {
            bridge_detected_boundary += 1;
        }
    }
    let elapsed = t0.elapsed();

    println!("MincutBackend::Direct determinism probe ({trials} trials on identical input)");
    println!(
        "  distinct min-cut values observed : {}",
        distinct_cut_values.len()
    );
    println!("  empty/no-signal partitions        : {empty}/{trials}");
    println!("  bridge flagged as boundary        : {bridge_detected_boundary}/{trials}");
    println!(
        "  total wall-clock                  : {:.3}ms ({:.3}ms/call)",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / trials as f64
    );
}
