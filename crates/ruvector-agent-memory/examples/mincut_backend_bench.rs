//! Nightly research benchmark (2026-10-29): does `ApproxMinCut` (SODA 2025
//! spectral sparsification, `ruvector-mincut::algorithm::approximate`) make
//! a viable, faster drop-in replacement for `RuVectorGraphAnsalyzer`'s exact
//! global min-cut as `MincutGatedForgetting`'s boundary-detection backend?
//!
//! This follows up on two prior nightlies that could not be re-run as
//! originally planned without this work:
//!
//! - 2026-09-05 (`docs/research/nightly/2026-09-05-mincut-gated-forgetting`,
//!   ADR-345): REJECTED `MincutGatedForgetting` for production use because
//!   `RuVectorGraphAnalyzer::partition()` measured 76ms-11.4s per call for
//!   50-400 vertex k-NN graphs — 1,800-2,700x slower than the scalar
//!   baseline even at an 84-memory corpus. Next Research item 1: "Repeat
//!   this experiment against `ruvector_mincut::DynamicMinCut` /
//!   `ClusterHierarchy::boundary_size` directly (bypassing
//!   `RuVectorGraphAnalyzer`) to check whether the lower-level API avoids
//!   the measured overhead."
//! - 2026-09-11 (`docs/research/nightly/2026-09-11-mincut-partition-determinism`,
//!   ADR-346): fixed two independent non-determinism bugs in
//!   `RuVectorGraphAnalyzer`/`DynamicGraph`, but left the latency problem
//!   explicitly open as Next Research item 2, naming
//!   `connectivity::polylog::PolylogConnectivity` as a candidate
//!   replacement backend.
//!
//! Investigating item 1 directly (both `algorithm::DynamicMinCut` and
//! `cluster::ClusterHierarchy`) found neither suitable: `DynamicMinCut`
//! always calls the same exact `exact::minimum_cut` solver regardless of
//! its own `MinCutConfig.approximate` flag (so it has the same cost
//! profile as `RuVectorGraphAnalyzer`), and `ClusterHierarchy` builds its
//! levels by arbitrary vertex-insertion-order pairing, not by minimum cut —
//! fast, but its "boundary_size" is unrelated to which vertices are
//! structurally load-bearing.
//!
//! `ruvector-mincut::ApproxMinCut` (`algorithm::approximate`, already
//! `pub use`d from the crate root, with 9 passing unit tests) looked like
//! the right candidate instead: SODA 2025 spectral sparsification, claimed
//! O(m log^2 n / eps^2) preprocessing / O(n polylog n / eps^2) query. But
//! its `partition` field was a stub: `compute_partition` ignored the
//! computed cut value entirely and returned an arbitrary BFS-order
//! bisection, so it could not have been used as a structural boundary
//! signal before being fixed (see `ruvector-mincut`'s
//! `src/algorithm/approximate.rs`, `stoer_wagner_with_partition`, and its
//! new regression tests `partition_isolates_the_weak_bridge` /
//! `partition_matches_reported_value_on_*`, fixed in this same nightly).
//!
//! This benchmark measures, with the fix in place:
//!
//! Given the same two-clique-plus-bridge 19-vertex correctness topology and
//! the same ring-shaped k-NN scaling probe (19/50/100/200/400 vertices, k=8)
//! used by the 2026-09-05/09-11 nightlies,
//!
//! when `MincutGatedForgetting`'s boundary detection uses
//! `MincutBackend::Approx { epsilon: 0.1 }` or `{ epsilon: 0.3 }` instead of
//! `MincutBackend::Exact` (baseline, unchanged `RuVectorGraphAnalyzer`),
//!
//! then boundary-detection latency at 50-400 vertices should drop
//! substantially relative to `Exact`, while the bridge vertex is still
//! reliably flagged as a boundary vertex and the determinism profile is no
//! worse than baseline,
//!
//! subject to: this is a boundary-*detection* latency/correctness
//! comparison only — it does not re-measure `MincutGatedForgetting`'s
//! bridge-*retention* Recall@10 claim from 2026-09-05 (unaffected by the
//! backend swap: both backends feed the same downstream
//! `select_survivors` scoring) or re-run the eviction-witness mechanism
//! (orthogonal, already covered by `mincut_gated_forgetting_bench`).
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example mincut_backend_bench --features mincut-forget

use ruvector_agent_memory::{boundary_from_knn, MincutBackend};
use std::time::Instant;

/// `(vertex_index, neighbor list of (neighbor_index, distance))`, matching
/// `boundary_from_knn`'s and `ruvector_mincut::RuVectorGraphAnalyzer::from_knn`'s
/// input shape.
type KnnGraph = Vec<(usize, Vec<(usize, f64)>)>;

const BACKENDS: [(&str, MincutBackend); 3] = [
    ("Exact", MincutBackend::Exact),
    ("Approx(eps=0.1)", MincutBackend::Approx { epsilon: 0.1 }),
    ("Approx(eps=0.3)", MincutBackend::Approx { epsilon: 0.3 }),
];

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

/// The exact two-clique-plus-bridge topology used by `graph_forget`'s unit
/// tests and the 2026-09-05/09-11 nightlies' determinism probe: two 9-member
/// orthogonal clusters, each with one "gateway" blended toward a third axis,
/// and a single bridge vertex sitting exactly on that third axis. Isolating
/// the bridge (degree 2) is the unambiguous cheapest cut, independent of
/// tie-breaking among same-score plain members.
fn bridge_knn() -> (KnnGraph, usize, usize) {
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
    let neighbors: KnnGraph = (0..n)
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
    (neighbors, n, bridge_idx)
}

/// Ring-shaped k-NN graph (vertex i connects to the next k vertices mod n):
/// the same regular/symmetric shape used by `mincut_scaling_probe`, as a
/// stand-in for a tightly clustered, roughly evenly spaced embedding.
fn ring_knn(n: usize, k: usize) -> KnnGraph {
    (0..n)
        .map(|i| {
            let nbrs: Vec<(usize, f64)> = (1..=k)
                .map(|d| ((i + d) % n, 0.1 + (d as f64) * 0.01))
                .collect();
            (i, nbrs)
        })
        .collect()
}

fn main() {
    println!("=== Part 1: latency scaling (ring k-NN, k=8) ===");
    println!("{:<18} {:>7} {:>14}", "backend", "n", "latency_ms");
    let sizes = [19usize, 50, 100, 200, 400];
    for &n in &sizes {
        let neighbors = ring_knn(n, 8);
        for &(name, backend) in &BACKENDS {
            let t0 = Instant::now();
            let _ = boundary_from_knn(backend, &neighbors);
            let elapsed = t0.elapsed();
            println!(
                "{:<18} {:>7} {:>14.3}",
                name,
                n,
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    println!();
    println!("=== Part 2: determinism + bridge-boundary correctness (19-vertex bridge dataset) ===");
    let (neighbors, _n, bridge_idx) = bridge_knn();
    let trials: usize = std::env::var("TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    println!(
        "{:<18} {:>8} {:>12} {:>10} {:>18} {:>14}",
        "backend", "trials", "elapsed_s", "empty_pct", "bridge_detect_pct", "avg_ms"
    );
    for &(name, backend) in &BACKENDS {
        let mut empty = 0usize;
        let mut bridge_detected = 0usize;
        let t0 = Instant::now();
        for _ in 0..trials {
            let boundary = boundary_from_knn(backend, &neighbors);
            if boundary.is_empty() {
                empty += 1;
                continue;
            }
            if boundary.contains(&bridge_idx) {
                bridge_detected += 1;
            }
        }
        let elapsed = t0.elapsed();
        println!(
            "{:<18} {:>8} {:>12.3} {:>9.0}% {:>17.0}% {:>14.3}",
            name,
            trials,
            elapsed.as_secs_f64(),
            100.0 * empty as f64 / trials as f64,
            100.0 * bridge_detected as f64 / trials as f64,
            elapsed.as_secs_f64() * 1000.0 / trials as f64,
        );
    }

    println!();
    println!("=== Acceptance ===");
    // Measured, not asserted: this example prints evidence for the nightly
    // research document to report verbatim, it does not itself gate CI.
    println!("See docs/research/nightly/2026-10-29-approx-mincut-boundary-backend/README.md");
}
