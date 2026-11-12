//! Throwaway scaling probe for `RuVectorGraphAnalyzer::partition()` latency,
//! used to size `mincut_gated_forgetting_bench`'s corpus and document the
//! "Failure modes" scaling table in
//! docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md. Not
//! itself part of the shipped research artifact.
//!
//! Builds a fixed-degree ring k-NN graph (vertex i connects to the next k
//! vertices mod n) at increasing n and times one `from_knn` build + one
//! `partition()` call at each size. The ring shape is a stand-in for "a
//! regular, symmetric k-NN graph" — the same shape a k-NN graph over a
//! tightly clustered, roughly evenly spaced embedding tends toward — not a
//! carefully chosen worst case.

use std::time::Instant;

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

        let t0 = Instant::now();
        let mut analyzer = ruvector_mincut::RuVectorGraphAnalyzer::from_knn(&neighbors);
        let build_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let _ = analyzer.partition();
        let partition_elapsed = t1.elapsed();

        println!(
            "n={n:<5} build={:>10.3}ms  partition={:>10.3}ms",
            build_elapsed.as_secs_f64() * 1000.0,
            partition_elapsed.as_secs_f64() * 1000.0
        );
    }
}
