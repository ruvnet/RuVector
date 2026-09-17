//! Throwaway measurement used only to size/document the real nightly
//! benchmark and the `boundary_indices` doc comment (not part of the shipped
//! research artifact). Measures partition determinism on a fixed,
//! byte-identical 19-vertex graph (the same two-clique-plus-bridge topology
//! as `graph_forget`'s unit tests) across repeated calls.
//!
//! Extended 2026-09-15 (ADR-345 follow-up item 1,
//! docs/research/nightly/2026-09-15-direct-mincut-bridge-detection) to run
//! the same 30-trial measurement against `BoundaryMethod::DirectBuilder`'s
//! underlying `MinCutBuilder::with_edges(...).build()` call, alongside the
//! original `RuVectorGraphAnalyzer::from_knn(...).partition()`
//! (`BoundaryMethod::WrapperPartition`) measurement.

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

/// One trial's outcome: did the call return a usable (non-empty) partition,
/// and if so, was the bridge vertex flagged as boundary?
enum Outcome {
    Empty,
    Boundary(bool),
}

fn wrapper_partition_trial(neighbors: &[(usize, Vec<(usize, f64)>)], bridge_idx: usize) -> Outcome {
    let mut analyzer = ruvector_mincut::RuVectorGraphAnalyzer::from_knn(neighbors);
    match analyzer.partition() {
        None => Outcome::Empty,
        Some((a, b)) => {
            if a.is_empty() || b.is_empty() {
                return Outcome::Empty;
            }
            classify(neighbors, bridge_idx, &a)
        }
    }
}

fn direct_builder_trial(neighbors: &[(usize, Vec<(usize, f64)>)], bridge_idx: usize) -> Outcome {
    // Deduplicate by unordered pair: the k-NN neighbor list is directed and
    // can list both (i,j) and (j,i) with the same weight, but
    // `DynamicGraph::insert_edge` rejects a second insert of the same
    // undirected pair with `EdgeExists`, which would make `MinCutBuilder::
    // build()` fail on the very first duplicate (see graph_forget.rs's
    // `boundary_from_one_partition_direct` for the same dedup, applied there
    // for the identical reason).
    use std::collections::HashMap;
    let mut edge_map: HashMap<(u64, u64), f64> = HashMap::new();
    for (i, nbrs) in neighbors {
        let iu = *i as u64;
        for &(j, dist) in nbrs {
            let ju = j as u64;
            let weight = if dist > 0.0 { 1.0 / dist } else { 1.0 };
            let key = if iu <= ju { (iu, ju) } else { (ju, iu) };
            edge_map.entry(key).or_insert(weight);
        }
    }
    let edges: Vec<(u64, u64, f64)> = edge_map.into_iter().map(|((a, b), w)| (a, b, w)).collect();
    let mincut = match ruvector_mincut::MinCutBuilder::new()
        .with_edges(edges)
        .build()
    {
        Ok(m) => m,
        Err(_) => return Outcome::Empty,
    };
    let (a, b) = mincut.partition();
    if a.is_empty() || b.is_empty() {
        return Outcome::Empty;
    }
    classify(neighbors, bridge_idx, &a)
}

fn classify(
    neighbors: &[(usize, Vec<(usize, f64)>)],
    bridge_idx: usize,
    side_a: &[u64],
) -> Outcome {
    let a_set: HashSet<u64> = side_a.iter().copied().collect();
    let mut boundary = false;
    for (i, nbrs) in neighbors {
        let i_in_a = a_set.contains(&(*i as u64));
        for &(j, _) in nbrs {
            let j_in_a = a_set.contains(&(j as u64));
            if i_in_a != j_in_a && (*i == bridge_idx || j == bridge_idx) {
                boundary = true;
            }
        }
    }
    Outcome::Boundary(boundary)
}

fn run_probe(
    name: &str,
    trials: usize,
    neighbors: &[(usize, Vec<(usize, f64)>)],
    bridge_idx: usize,
    call: impl Fn(&[(usize, Vec<(usize, f64)>)], usize) -> Outcome,
) {
    let mut empty = 0usize;
    let mut bridge_detected_boundary = 0usize;
    let t0 = Instant::now();
    for _ in 0..trials {
        match call(neighbors, bridge_idx) {
            Outcome::Empty => empty += 1,
            Outcome::Boundary(true) => bridge_detected_boundary += 1,
            Outcome::Boundary(false) => {}
        }
    }
    let elapsed = t0.elapsed();
    println!(
        "[{name}] trials={trials} elapsed={:.2}s avg_per_call={:.1}ms empty_or_degenerate={empty} ({:.0}%) bridge_detected_as_boundary={bridge_detected_boundary} ({:.0}%)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / trials as f64,
        100.0 * empty as f64 / trials as f64,
        100.0 * bridge_detected_boundary as f64 / trials as f64,
    );
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

    run_probe(
        "wrapper_partition",
        trials,
        &neighbors,
        bridge_idx,
        wrapper_partition_trial,
    );
    run_probe(
        "direct_builder",
        trials,
        &neighbors,
        bridge_idx,
        direct_builder_trial,
    );
}
