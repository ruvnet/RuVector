//! Throwaway scaling probe for `RuVectorGraphAnalyzer::partition()` latency,
//! used to size `mincut_gated_forgetting_bench`'s corpus and document the
//! "Failure modes" scaling table in
//! docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md. Not
//! itself part of the shipped research artifact.
//!
//! Extended 2026-09-08 (see
//! docs/research/nightly/2026-09-08-static-mincut-forgetting/README.md) with
//! a `partition_static()` column (the new deterministic Stoer-Wagner path,
//! `ruvector_mincut::static_cut`) at the same sizes, plus a larger size (800
//! vertices) since the static engine's O(V^3) cost is the one now worth
//! characterizing.
//!
//! Builds a fixed-degree ring k-NN graph (vertex i connects to the next k
//! vertices mod n) at increasing n and times one `from_knn` build + one
//! `partition()`/`partition_static()` call at each size. The ring shape is a
//! stand-in for "a regular, symmetric k-NN graph" — the same shape a k-NN
//! graph over a tightly clustered, roughly evenly spaced embedding tends
//! toward — not a carefully chosen worst case.

use std::time::Instant;

fn main() {
    let sizes = [19usize, 50, 100, 200, 400, 800];
    let k = 8usize;
    println!(
        "{:<6} {:>12} {:>16} {:>18} {:>10}",
        "n", "build(ms)", "partition(ms)", "partition_static(ms)", "speedup"
    );
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

        // Skip the dynamic engine above 400 vertices: ADR-345 already
        // measured multi-second-to-minutes latency there, and re-measuring
        // it at 800/1600 would make this probe impractically slow for no
        // new information.
        let dynamic_ms = if n <= 400 {
            let t1 = Instant::now();
            let _ = analyzer.partition();
            Some(t1.elapsed().as_secs_f64() * 1000.0)
        } else {
            None
        };

        let t2 = Instant::now();
        let _ = analyzer.partition_static();
        let static_ms = t2.elapsed().as_secs_f64() * 1000.0;

        match dynamic_ms {
            Some(d) => println!(
                "n={n:<4} {:>10.3}ms {:>14.3}ms {:>16.3}ms {:>9.1}x",
                build_elapsed.as_secs_f64() * 1000.0,
                d,
                static_ms,
                d / static_ms.max(1e-6),
            ),
            None => println!(
                "n={n:<4} {:>10.3}ms {:>16} {:>16.3}ms {:>10}",
                build_elapsed.as_secs_f64() * 1000.0,
                "skipped",
                static_ms,
                "n/a",
            ),
        }
    }
}
