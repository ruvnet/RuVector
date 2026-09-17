//! Throwaway scaling probe for `RuVectorGraphAnalyzer::partition()` latency,
//! used to size `mincut_gated_forgetting_bench`'s corpus and document the
//! "Failure modes" scaling table in
//! docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md. Not
//! itself part of the shipped research artifact.
//!
//! Extended 2026-09-15 (ADR-345 follow-up item 1,
//! docs/research/nightly/2026-09-15-direct-mincut-bridge-detection) to add
//! the same measurement for `MinCutBuilder::with_edges(...).build()`
//! (`BoundaryMethod::DirectBuilder`'s underlying call), so the two methods'
//! scaling behavior can be read side by side on identical input graphs.
//!
//! Builds a fixed-degree ring k-NN graph (vertex i connects to the next k
//! vertices mod n) at increasing n and times one full "raw edges -> boundary
//! decision" call at each size, for each method. The ring shape is a
//! stand-in for "a regular, symmetric k-NN graph" — the same shape a k-NN
//! graph over a tightly clustered, roughly evenly spaced embedding tends
//! toward — not a carefully chosen worst case.

use std::time::Instant;

fn wrapper_partition_call(neighbors: &[(usize, Vec<(usize, f64)>)]) -> std::time::Duration {
    let t0 = Instant::now();
    let mut analyzer = ruvector_mincut::RuVectorGraphAnalyzer::from_knn(neighbors);
    let _ = analyzer.partition();
    t0.elapsed()
}

fn direct_builder_call(neighbors: &[(usize, Vec<(usize, f64)>)]) -> std::time::Duration {
    let t0 = Instant::now();
    // Dedup by unordered pair: see mincut_determinism_probe.rs / graph_forget.rs's
    // `boundary_from_one_partition_direct` for why (`insert_edge` rejects a
    // reverse-direction duplicate with `EdgeExists`, failing `build()` fast).
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
    if let Ok(mincut) = ruvector_mincut::MinCutBuilder::new()
        .with_edges(edges)
        .build()
    {
        let _ = mincut.partition();
    }
    t0.elapsed()
}

fn main() {
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

        let wrapper_elapsed = wrapper_partition_call(&neighbors);
        let direct_elapsed = direct_builder_call(&neighbors);

        println!(
            "n={n:<5} wrapper_partition={:>12.3}ms  direct_builder={:>12.3}ms  speedup={:>10.1}x",
            wrapper_elapsed.as_secs_f64() * 1000.0,
            direct_elapsed.as_secs_f64() * 1000.0,
            wrapper_elapsed.as_secs_f64() / direct_elapsed.as_secs_f64().max(1e-9),
        );
    }
}
