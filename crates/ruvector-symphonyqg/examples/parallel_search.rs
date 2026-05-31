//! Measure the rayon-parallel `search_batch` speedup.
//!
//! Run with: `cargo run -p ruvector-symphonyqg --release --example parallel_search --features parallel`
//!
//! Builds an n=10K corpus once, then runs 1000 queries via:
//!   (a) sequential `search()` loop
//!   (b) `search_batch()` (parallel when --features parallel)
//! and reports the speedup ratio.

use ruvector_symphonyqg::*;
use std::time::Instant;

fn gaussian_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = (seed as u32) | 1;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    (s as f32 / u32::MAX as f32) - 0.5
                })
                .collect()
        })
        .collect()
}

fn main() {
    let n = 10_000;
    let dim = 128;
    let n_queries = 1000;
    let k = 10;
    let ef = 100;

    let cfg = Config {
        dim,
        m_base: 16,
        ef_construction: 200,
        ..Config::default()
    };
    let vecs = gaussian_vecs(n, dim, 42);
    let probe_data = gaussian_vecs(n_queries, dim, 99);
    let probes: Vec<&[f32]> = probe_data.iter().map(|v| v.as_slice()).collect();

    println!(
        "Building n={} dim={} ...",
        n, dim
    );
    let (_, _, sym) = build_all(&vecs, &cfg);

    println!("Running {} queries (k={}, ef={}) ...", n_queries, k, ef);

    // Warm-up
    let _ = sym.search(&probes[0], k, ef);

    let t0 = Instant::now();
    let seq: Vec<Vec<SearchResult>> = probes.iter().map(|q| sym.search(q, k, ef)).collect();
    let seq_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let bat = sym.search_batch(&probes, k, ef);
    let bat_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Sanity: results must match
    assert_eq!(seq.len(), bat.len());
    for (s, b) in seq.iter().zip(bat.iter()) {
        assert_eq!(s.len(), b.len());
        for (sr, br) in s.iter().zip(b.iter()) {
            assert_eq!(sr.idx, br.idx);
        }
    }

    println!(
        "{:>16} | {:>9}",
        "mode", "wall_ms"
    );
    println!("{}", "-".repeat(30));
    println!("{:>16} | {:>9.1}", "sequential", seq_ms);
    println!("{:>16} | {:>9.1}", "search_batch", bat_ms);
    println!();
    println!("speedup: {:.2}×", seq_ms / bat_ms);
    println!(
        "(parallel feature {})",
        if cfg!(feature = "parallel") {
            "ON"
        } else {
            "OFF — rebuild with --features parallel for speedup"
        }
    );
}
