//! Determinism probe for `ruvector_mincut::CactusGraph::canonical_cut`,
//! directly comparable to ADR-345's `mincut_determinism_probe` (which found
//! `RuVectorGraphAnalyzer::partition()` returned an empty/degenerate result
//! on 50% of repeated calls on byte-identical input). Same 19-vertex
//! two-clique-plus-bridge topology.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example cactus_determinism_probe --features mincut-forget-cactus

use ruvector_mincut::{CactusGraph, DynamicGraph};
use std::collections::HashSet;
use std::sync::Arc;
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

    let trials: usize = std::env::var("TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let mut empty = 0usize;
    let mut bridge_detected_boundary = 0usize;
    let mut distinct_partitions: HashSet<Vec<usize>> = HashSet::new();
    let t0 = Instant::now();
    for _ in 0..trials {
        let graph = Arc::new(DynamicGraph::new());
        for i in 0..n {
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_sim(&entries[i], &entries[j])))
                .filter(|&(_, s)| s >= min_sim)
                .collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims.truncate(k);
            for (j, s) in sims {
                // See graph_forget_cactus.rs: unconditional on i < j, since
                // k-NN truncation is asymmetric and `insert_edge` no-ops on
                // an already-present undirected pair.
                let weight = (1.0 / (1.0 - s).max(1e-4)) as f64;
                let _ = graph.insert_edge(i as u64, j as u64, weight);
            }
        }

        let cactus = CactusGraph::build_from_graph(&graph);
        let cut = cactus.canonical_cut();
        let (side_a, side_b) = &cut.partition;
        if side_a.is_empty() || side_b.is_empty() {
            empty += 1;
            continue;
        }
        let mut key = side_a.clone();
        key.sort_unstable();
        distinct_partitions.insert(key.clone());

        let side_a_set: HashSet<usize> = side_a.iter().copied().collect();
        let mut boundary = false;
        for edge in graph.edges() {
            let u = edge.source as usize;
            let v = edge.target as usize;
            let u_in_a = side_a_set.contains(&u);
            let v_in_a = side_a_set.contains(&v);
            if u_in_a != v_in_a && (u == bridge_idx || v == bridge_idx) {
                boundary = true;
            }
        }
        if boundary {
            bridge_detected_boundary += 1;
        }
    }
    let elapsed = t0.elapsed();
    println!(
        "trials={trials} elapsed={:.4}s avg_per_call={:.3}ms empty_or_degenerate={empty} ({:.0}%) \
         bridge_detected_as_boundary={bridge_detected_boundary} ({:.0}%) distinct_partitions={}",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / trials as f64,
        100.0 * empty as f64 / trials as f64,
        100.0 * bridge_detected_boundary as f64 / trials as f64,
        distinct_partitions.len(),
    );
}
