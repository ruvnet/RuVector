//! RaBitQ two-stage path per metric: recall@10 and latency vs the exact
//! scan. Seeded and deterministic (uniform i.i.d. components in [-0.5, 0.5),
//! shifted by +0.1 so cosine, inner product and L2 rank differently).
//!
//! Run: cargo run --release --example rabitq_metric_bench -p rvf-runtime -- [n] [dim] [queries]

use std::time::Instant;

use rvf_runtime::options::{DistanceMetric, QueryOptions, RvfOptions};
use rvf_runtime::RvfStore;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn vector(seed: u64, dim: usize) -> Vec<f32> {
    (0..dim as u64)
        .map(|i| {
            let u = (splitmix(seed.wrapping_mul(1_000_003).wrapping_add(i)) >> 40) as f32;
            u / (1u64 << 24) as f32 - 0.4
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let n = args.first().copied().unwrap_or(100_000);
    let dim = args.get(1).copied().unwrap_or(128);
    let queries = args.get(2).copied().unwrap_or(100);
    let k = 10;
    let dir = std::env::temp_dir().join(format!("rvf-rabitq-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    println!("n={n} dim={dim} queries={queries} k={k}\n");
    println!("| metric | exact p50 ms | exact p99 ms | rabitq p50 ms | rabitq p99 ms | speedup p50 | recall@10 |");
    println!("|---|---|---|---|---|---|---|");
    for metric in [
        DistanceMetric::L2,
        DistanceMetric::Cosine,
        DistanceMetric::InnerProduct,
    ] {
        let path = dir.join(format!("{metric:?}.rvf"));
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: dim as u16,
                metric,
                ..Default::default()
            },
        )
        .unwrap();
        for start in (0..n as u64).step_by(10_000) {
            let end = (start + 10_000).min(n as u64);
            let vecs: Vec<Vec<f32>> = (start..end).map(|i| vector(i, dim)).collect();
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            let ids: Vec<u64> = (start..end).collect();
            store.ingest_batch(&refs, &ids, None).unwrap();
        }
        let exact_opts = QueryOptions {
            force_exact: true,
            ..Default::default()
        };
        let rabitq_opts = QueryOptions {
            rabitq: true,
            ..Default::default()
        };
        // Build the code book outside the timed loop.
        store
            .query(&vector(u64::MAX, dim), k, &rabitq_opts)
            .unwrap();
        let (mut te, mut tr, mut recall) = (Vec::new(), Vec::new(), 0.0);
        for qi in 0..queries as u64 {
            let q = vector(u64::MAX - 1 - qi, dim);
            let t = Instant::now();
            let exact = store.query(&q, k, &exact_opts).unwrap();
            te.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let got = store.query(&q, k, &rabitq_opts).unwrap();
            tr.push(t.elapsed().as_secs_f64() * 1e3);
            recall += got
                .iter()
                .filter(|r| exact.iter().any(|e| e.id == r.id))
                .count() as f64
                / k as f64;
        }
        te.sort_by(f64::total_cmp);
        tr.sort_by(f64::total_cmp);
        println!(
            "| {metric:?} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1}x | {:.3} |",
            percentile(&te, 0.5),
            percentile(&te, 0.99),
            percentile(&tr, 0.5),
            percentile(&tr, 0.99),
            percentile(&te, 0.5) / percentile(&tr, 0.5),
            recall / queries as f64
        );
        store.close().unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}
