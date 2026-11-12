//! Throwaway measurement used only to size/document the real nightly
//! benchmark and the `boundary_indices` doc comment (not part of the shipped
//! research artifact). Measures `RuVectorGraphAnalyzer::partition()`
//! determinism on a fixed, byte-identical 19-vertex graph (the same
//! two-clique-plus-bridge topology as `graph_forget`'s unit tests) across
//! repeated calls.

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
    let t0 = Instant::now();
    for _ in 0..trials {
        let mut analyzer = ruvector_mincut::RuVectorGraphAnalyzer::from_knn(&neighbors);
        match analyzer.partition() {
            None => empty += 1,
            Some((a, b)) => {
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
        }
    }
    let elapsed = t0.elapsed();
    println!(
        "trials={trials} elapsed={:.2}s avg_per_call={:.1}ms empty_or_degenerate={empty} ({:.0}%) bridge_detected_as_boundary={bridge_detected_boundary} ({:.0}%)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / trials as f64,
        100.0 * empty as f64 / trials as f64,
        100.0 * bridge_detected_boundary as f64 / trials as f64,
    );
}
