//! Turbo4 / ACRP ablation harness (ADR-297 phase G).
//!
//! Measures, on one deterministic clustered corpus and identical HNSW
//! parameters:
//!
//! * f32 HNSW baseline
//! * Turbo4 direct — MaxCompression / Balanced / Quality policies
//! * Turbo4 + RaBitQ1 cascade — Balanced
//!
//! Reporting: recall@1 / recall@10 vs brute-force f32 ground truth, build
//! time, P50/P95 query latency, index payload bytes/vector, escalation rate.
//!
//! Usage: `cargo run --release -p ruvector-bench --bin turbo4-ablation
//!         [-- --n 20000 --dim 768 --queries 200]`

use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::turbo4::Turbo4HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig, SearchPolicy, SearchQuantization};
use std::time::Instant;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gauss(state: &mut u64) -> f32 {
    let u1 = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64;
    let u2 = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64;
    ((-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

fn clustered_corpus(n: usize, dim: usize, cluster: usize, spread: f32, seed: u64) -> Vec<Vec<f32>> {
    let n_centers = n.div_ceil(cluster);
    let mut s = seed;
    let centers: Vec<Vec<f32>> = (0..n_centers)
        .map(|_| (0..dim).map(|_| gauss(&mut s)).collect())
        .collect();
    (0..n)
        .map(|i| {
            centers[i / cluster]
                .iter()
                .map(|c| c + spread * gauss(&mut s))
                .collect()
        })
        .collect()
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

struct Report {
    name: String,
    build_ms: f64,
    recall1: f32,
    recall10: f32,
    p50_us: f64,
    p95_us: f64,
    bytes_per_vec: usize,
    escalated_pct: f64,
}

/// Build + query one index; `stats` supplies the escalation counters for
/// Turbo4 configs.
fn measure<F>(
    name: &str,
    index: &mut dyn VectorIndex,
    stats: F,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    bytes_per_vec: usize,
) -> Report
where
    F: FnOnce() -> Option<(u64, u64)>,
{
    let entries: Vec<_> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("v{i}"), v.clone()))
        .collect();
    let t0 = Instant::now();
    index.add_batch(entries).expect("build");
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut lat_us: Vec<f64> = Vec::with_capacity(queries.len());
    let (mut hit1, mut hit10, mut total) = (0usize, 0usize, 0usize);
    for (q, t) in queries.iter().zip(truth) {
        let t0 = Instant::now();
        let res = index.search(q, 10).expect("search");
        lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
        let ids: Vec<usize> = res
            .iter()
            .map(|r| r.id[1..].parse::<usize>().unwrap())
            .collect();
        if !ids.is_empty() && ids[0] == t[0] {
            hit1 += 1;
        }
        let tset: std::collections::HashSet<usize> = t.iter().copied().collect();
        hit10 += ids.iter().filter(|i| tset.contains(i)).count();
        total += 10;
    }
    lat_us.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| lat_us[((lat_us.len() as f64 - 1.0) * p) as usize];
    let escalated_pct = match stats() {
        Some((q, e)) if q > 0 => 100.0 * e as f64 / q as f64,
        _ => 0.0,
    };

    Report {
        name: name.to_string(),
        build_ms,
        recall1: hit1 as f32 / queries.len() as f32,
        recall10: hit10 as f32 / total as f32,
        p50_us: pct(0.50),
        p95_us: pct(0.95),
        bytes_per_vec,
        escalated_pct,
    }
}

fn main() {
    let mut n = 20_000usize;
    let mut dim = 768usize;
    let mut n_queries = 200usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--n" => n = args[i + 1].parse().unwrap(),
            "--dim" => dim = args[i + 1].parse().unwrap(),
            "--queries" => n_queries = args[i + 1].parse().unwrap(),
            _ => {}
        }
        i += 2;
    }

    println!("Turbo4/ACRP ablation: n={n}, dim={dim}, queries={n_queries}, clustered corpus");
    let vectors = clustered_corpus(n, dim, 20, 0.35, 42);
    let mut s = 777u64;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|j| {
            vectors[(j * 97) % n]
                .iter()
                .map(|v| v + 0.2 * gauss(&mut s))
                .collect()
        })
        .collect();

    let t0 = Instant::now();
    let truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let mut d: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(q, v)))
                .collect();
            d.sort_by(|a, b| a.1.total_cmp(&b.1));
            d[..10].iter().map(|(i, _)| *i).collect()
        })
        .collect();
    println!("ground truth: {:.1}s", t0.elapsed().as_secs_f64());

    let config = HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 100,
        max_elements: n + 1,
    };
    let metric = DistanceMetric::Euclidean;
    let mut reports = Vec::new();

    {
        let mut ix = HnswIndex::new(dim, metric, config.clone()).unwrap();
        reports.push(measure(
            "f32 HNSW",
            &mut ix,
            || None,
            &vectors,
            &queries,
            &truth,
            dim * 4,
        ));
    }

    let variants: [(&str, usize, SearchPolicy, SearchQuantization); 4] = [
        (
            "turbo4 maxcomp",
            2,
            SearchPolicy::MaxCompression,
            SearchQuantization::Turbo4Direct,
        ),
        (
            "turbo4 balanced",
            4,
            SearchPolicy::Balanced,
            SearchQuantization::Turbo4Direct,
        ),
        (
            "turbo4 quality",
            8,
            SearchPolicy::Quality,
            SearchQuantization::Turbo4Direct,
        ),
        (
            "cascade balanced",
            4,
            SearchPolicy::Balanced,
            SearchQuantization::RaBitQ1,
        ),
    ];
    for (label, mult, policy, sq) in variants {
        let mut ix =
            Turbo4HnswIndex::new(dim, metric, config.clone(), 42, mult, policy, sq).unwrap();
        let bytes = match sq {
            SearchQuantization::Turbo4Direct => dim / 2 + 8,
            SearchQuantization::RaBitQ1 => dim / 2 + 8 + dim.div_ceil(64) * 8 + 8,
        };
        // Split borrow: measure takes &mut dyn, stats closure reads counters
        // afterwards via a raw copy of the shared atomics through the
        // immutable reborrow inside the closure evaluation (called after
        // searches complete, before ix drops).
        let stats_snapshot: std::cell::Cell<Option<(u64, u64)>> = std::cell::Cell::new(None);
        let report = {
            let r = measure(
                label,
                &mut ix,
                || stats_snapshot.take(),
                &vectors,
                &queries,
                &truth,
                bytes,
            );
            // measure's closure ran before we could snapshot — patch the
            // escalation rate from the live index now.
            let (q, e) = ix.adaptive_stats();
            Report {
                escalated_pct: if q > 0 {
                    100.0 * e as f64 / q as f64
                } else {
                    0.0
                },
                ..r
            }
        };
        reports.push(report);
    }

    // Verification-tier configs (ADR-297 §2): fetch 2k from the quantized
    // index, re-rank those ids against the ORIGINAL f32 vectors (which the
    // storage layer keeps), truncate to k. Tests whether f32 verification
    // recovers the precision-bound tail ranks that wider traversal cannot.
    for (label, mult, policy, sq) in [
        (
            "t4 maxcomp+verify",
            2,
            SearchPolicy::MaxCompression,
            SearchQuantization::Turbo4Direct,
        ),
        (
            "cascade mc+verify",
            2,
            SearchPolicy::MaxCompression,
            SearchQuantization::RaBitQ1,
        ),
    ] {
        let mut ix =
            Turbo4HnswIndex::new(dim, metric, config.clone(), 42, mult, policy, sq).unwrap();
        let bytes = match sq {
            SearchQuantization::Turbo4Direct => dim / 2 + 8,
            SearchQuantization::RaBitQ1 => dim / 2 + 8 + dim.div_ceil(64) * 8 + 8,
        };
        let entries: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();
        let t0 = Instant::now();
        ix.add_batch(entries).expect("build");
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut lat_us: Vec<f64> = Vec::with_capacity(queries.len());
        let (mut hit1, mut hit10, mut total) = (0usize, 0usize, 0usize);
        for (q, t) in queries.iter().zip(&truth) {
            let t0 = Instant::now();
            let res = ix.search(q, 20).expect("search");
            let mut verified: Vec<(usize, f32)> = res
                .iter()
                .map(|r| {
                    let id = r.id[1..].parse::<usize>().unwrap();
                    (id, l2(q, &vectors[id]))
                })
                .collect();
            verified.sort_by(|a, b| a.1.total_cmp(&b.1));
            verified.truncate(10);
            lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
            let ids: Vec<usize> = verified.iter().map(|(i, _)| *i).collect();
            if !ids.is_empty() && ids[0] == t[0] {
                hit1 += 1;
            }
            let tset: std::collections::HashSet<usize> = t.iter().copied().collect();
            hit10 += ids.iter().filter(|i| tset.contains(i)).count();
            total += 10;
        }
        lat_us.sort_by(|a, b| a.total_cmp(b));
        let pct = |p: f64| lat_us[((lat_us.len() as f64 - 1.0) * p) as usize];
        reports.push(Report {
            name: label.to_string(),
            build_ms,
            recall1: hit1 as f32 / queries.len() as f32,
            recall10: hit10 as f32 / total as f32,
            p50_us: pct(0.50),
            p95_us: pct(0.95),
            bytes_per_vec: bytes,
            escalated_pct: 0.0,
        });
    }

    println!();
    println!(
        "{:<18} {:>9} {:>9} {:>10} {:>10} {:>10} {:>8} {:>7}",
        "config", "recall@1", "recall@10", "P50 µs", "P95 µs", "build ms", "B/vec", "esc %"
    );
    for r in &reports {
        println!(
            "{:<18} {:>9.3} {:>9.3} {:>10.1} {:>10.1} {:>10.0} {:>8} {:>7.1}",
            r.name,
            r.recall1,
            r.recall10,
            r.p50_us,
            r.p95_us,
            r.build_ms,
            r.bytes_per_vec,
            r.escalated_pct
        );
    }
    let f32r = reports[0].recall10;
    println!();
    for r in reports.iter().skip(1) {
        println!(
            "{:<18} Δrecall@10 vs f32: {:+.3} pp; payload {:.1}x smaller",
            r.name,
            (r.recall10 - f32r) * 100.0,
            (dim * 4) as f32 / r.bytes_per_vec as f32
        );
    }
}
