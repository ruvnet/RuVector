//! Companion to `mincut_scaling_probe.rs`: same fixed-degree ring k-NN
//! topology and the same five sizes, but timing
//! `RuVectorGraphAnalyzer::{partition_exact, min_cut_exact}` (the sparse
//! Stoer-Wagner solver added in this nightly run) instead of
//! `partition()`/`min_cut()` (the `MinCutWrapper`/`BoundedInstance`
//! LocalKCut-oracle path measured in
//! `docs/research/nightly/2026-09-05-mincut-gated-forgetting`).
//!
//! Also probes larger sizes (800..6400) that the original probe never
//! reached, specifically to find where the *exact* path's own polynomial
//! cost eventually gets expensive — this is a deliberately honest addition:
//! the exact solver is polynomial, not subpolynomial, so it is not claimed to
//! win at every scale, only at the scale this nightly's benchmark actually
//! needs (a few hundred vertices).
//!
//! Not itself part of the shipped research artifact; a reproducible probe
//! referenced by docs/research/nightly/<date>-mincut-exact-partition-backend.

use std::time::Instant;

fn ring_neighbors(n: usize, k: usize) -> Vec<(usize, Vec<(usize, f64)>)> {
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
    let k = 8usize;
    println!("=== same sizes as mincut_scaling_probe.rs (partition()/BoundedInstance) ===");
    for &n in &[19usize, 50, 100, 200, 400] {
        let neighbors = ring_neighbors(n, k);

        let t0 = Instant::now();
        let analyzer = ruvector_mincut::RuVectorGraphAnalyzer::from_knn(&neighbors);
        let build_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let _ = analyzer.partition_exact();
        let partition_elapsed = t1.elapsed();

        println!(
            "n={n:<5} build={:>10.3}ms  partition_exact={:>10.3}ms",
            build_elapsed.as_secs_f64() * 1000.0,
            partition_elapsed.as_secs_f64() * 1000.0
        );
    }

    println!("\n=== extended range: where does the exact path itself get expensive? ===");
    for &n in &[800usize, 1600, 3200, 6400] {
        let neighbors = ring_neighbors(n, k);
        let analyzer = ruvector_mincut::RuVectorGraphAnalyzer::from_knn(&neighbors);

        let t1 = Instant::now();
        let _ = analyzer.partition_exact();
        let partition_elapsed = t1.elapsed();

        println!(
            "n={n:<5} partition_exact={:>10.3}ms",
            partition_elapsed.as_secs_f64() * 1000.0
        );
    }
}
