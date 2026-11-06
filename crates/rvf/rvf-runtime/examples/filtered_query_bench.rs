//! Filtered query benchmark: exact scan (the pre-planner routing, forced
//! with `force_exact`) vs the planner + filtered HNSW path.
//!
//! Seeded and deterministic. Each vector carries field 0 = a pseudo-random
//! bucket in `0..100` (decorrelated from the vector), so `bucket < s` keeps
//! `s`% of the store. Reports recall@10 against the exact result and p50/p99
//! latency for both routes at several selectivities.
//!
//! Run: cargo run --release --example filtered_query_bench -p rvf-runtime -- [n] [dim] [queries]

use std::time::Instant;

use rvf_runtime::filter::{FilterExpr, FilterValue};
use rvf_runtime::options::{
    DistanceMetric, MetadataEntry, MetadataValue, QueryOptions, RvfOptions,
};
use rvf_runtime::RvfStore;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(seed: u64, i: u64) -> f32 {
    (splitmix(seed.wrapping_mul(1_000_003).wrapping_add(i)) >> 40) as f32 / (1u64 << 24) as f32
        - 0.5
}

/// Latent rank of the `lowrank` dataset.
const LATENT: u64 = 16;

/// `uniform`: i.i.d. uniform components (intrinsic dimension = `dim`, the
/// hardest case for any graph index). `lowrank`: `W z + 0.05 * noise` with a
/// fixed random `W` (dim x 16) and Gaussian-ish `z`, which is closer to how
/// real embedding sets behave (low intrinsic dimension).
fn vector(seed: u64, dim: usize) -> Vec<f32> {
    if !LOWRANK.load(std::sync::atomic::Ordering::Relaxed) {
        return (0..dim as u64).map(|i| unit(seed, i)).collect();
    }
    let z: Vec<f32> = (0..LATENT)
        .map(|j| (0..4).map(|r| unit(seed, 1000 + j * 4 + r)).sum::<f32>())
        .collect();
    (0..dim as u64)
        .map(|i| {
            let wz: f32 = (0..LATENT)
                .map(|j| unit(0xC0FFEE, i * LATENT + j) * z[j as usize])
                .sum();
            wz + 0.05 * unit(seed, 5000 + i)
        })
        .collect()
}

static LOWRANK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let n = args.first().copied().unwrap_or(100_000);
    let dim = args.get(1).copied().unwrap_or(128);
    let queries = args.get(2).copied().unwrap_or(200);
    let data = if std::env::args().any(|a| a == "uniform") {
        LOWRANK.store(false, std::sync::atomic::Ordering::Relaxed);
        "uniform"
    } else {
        "lowrank"
    };
    let k = 10;

    let dir = std::env::temp_dir().join(format!("rvf-filtered-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = match std::env::var_os("RVF_BENCH_CACHE") {
        Some(p) => std::path::PathBuf::from(p).join(format!("filtered-{data}-{n}x{dim}.rvf")),
        None => dir.join("bench.rvf"),
    };
    if path.exists() {
        run(RvfStore::open(&path).unwrap(), n, dim, queries, k);
        return;
    }
    let mut store = RvfStore::create(
        &path,
        RvfOptions {
            dimension: dim as u16,
            metric: DistanceMetric::L2,
            ..Default::default()
        },
    )
    .unwrap();

    let t = Instant::now();
    for chunk_start in (0..n as u64).step_by(10_000) {
        let end = (chunk_start + 10_000).min(n as u64);
        let vecs: Vec<Vec<f32>> = (chunk_start..end).map(|i| vector(i, dim)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (chunk_start..end).collect();
        let meta: Vec<MetadataEntry> = ids
            .iter()
            .map(|&id| MetadataEntry {
                field_id: 0,
                value: MetadataValue::U64(splitmix(id ^ 0xB0C4) % 100),
            })
            .collect();
        store.ingest_batch(&refs, &ids, Some(&meta)).unwrap();
    }
    println!(
        "data={data} n={n} dim={dim} queries={queries} k={k}  ingest {:.1}s",
        t.elapsed().as_secs_f64()
    );

    // Build the HNSW index once (lazy, on the first eligible query) and
    // persist it, so the timed section measures queries only.
    let t = Instant::now();
    store
        .query(&vector(u64::MAX, dim), k, &QueryOptions::default())
        .unwrap();
    println!(
        "index build (first unfiltered query) {:.1}s",
        t.elapsed().as_secs_f64()
    );
    store.close().unwrap();
    run(RvfStore::open(&path).unwrap(), n, dim, queries, k);
    if std::env::var_os("RVF_BENCH_CACHE").is_none() {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn run(store: RvfStore, n: usize, dim: usize, queries: usize, k: usize) {
    let _ = n;
    let qs: Vec<Vec<f32>> = (0..queries as u64)
        .map(|i| vector(u64::MAX - i, dim))
        .collect();
    // The first filtered query on a fresh index also pays the one-off graph
    // recall calibration (cached on the index afterwards).
    {
        let opts = QueryOptions {
            filter: Some(FilterExpr::Lt(0, FilterValue::U64(50))),
            ..Default::default()
        };
        let t = Instant::now();
        store.query(&qs[0], k, &opts).unwrap();
        let first = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        store.query(&qs[0], k, &opts).unwrap();
        println!(
            "first filtered query (incl. one-off graph calibration) {first:.1} ms, repeat {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // Unfiltered reference: the HNSW path vs the exact scan.
    {
        let exact_opts = QueryOptions {
            force_exact: true,
            ..Default::default()
        };
        let (mut te, mut tn, mut recall) = (Vec::new(), Vec::new(), 0.0);
        for q in &qs {
            let t = Instant::now();
            let exact = store.query(q, k, &exact_opts).unwrap();
            te.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let got = store.query(q, k, &QueryOptions::default()).unwrap();
            tn.push(t.elapsed().as_secs_f64() * 1e3);
            recall += got
                .iter()
                .filter(|r| exact.iter().any(|e| e.id == r.id))
                .count() as f64
                / exact.len().max(1) as f64;
        }
        te.sort_by(f64::total_cmp);
        tn.sort_by(f64::total_cmp);
        println!(
            "unfiltered: exact p50 {:.3} ms, hnsw p50 {:.3} ms, recall@10 {:.3}\n",
            percentile(&te, 0.5),
            percentile(&tn, 0.5),
            recall / qs.len() as f64
        );
    }

    println!("| selectivity | exact p50 ms | exact p99 ms | new p50 ms | new p99 ms | speedup p50 | recall@10 |");
    println!("|---|---|---|---|---|---|---|");
    for pct in [50u64, 25, 20, 15, 10, 5, 1] {
        let filter = FilterExpr::Lt(0, FilterValue::U64(pct));
        let exact_opts = QueryOptions {
            filter: Some(filter.clone()),
            force_exact: true,
            ..Default::default()
        };
        let new_opts = QueryOptions {
            filter: Some(filter),
            ..Default::default()
        };
        let (mut te, mut tn) = (Vec::new(), Vec::new());
        let mut recall = 0.0;
        for q in &qs {
            let t = Instant::now();
            let exact = store.query(q, k, &exact_opts).unwrap();
            te.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let got = store.query(q, k, &new_opts).unwrap();
            tn.push(t.elapsed().as_secs_f64() * 1e3);
            let hits = got
                .iter()
                .filter(|r| exact.iter().any(|e| e.id == r.id))
                .count();
            recall += hits as f64 / exact.len().max(1) as f64;
        }
        te.sort_by(f64::total_cmp);
        tn.sort_by(f64::total_cmp);
        let (e50, n50) = (percentile(&te, 0.5), percentile(&tn, 0.5));
        println!(
            "| {pct}% | {e50:.3} | {:.3} | {n50:.3} | {:.3} | {:.1}x | {:.3} |",
            percentile(&te, 0.99),
            percentile(&tn, 0.99),
            e50 / n50,
            recall / queries as f64
        );
    }

    // Optional ef sweep of the filtered HNSW walk itself, bypassing the
    // graph-recall gate (PreferLatency), e.g. RVF_BENCH_EF_SWEEP=256,512,1024.
    if let Ok(sweep) = std::env::var("RVF_BENCH_EF_SWEEP") {
        let efs: Vec<u16> = sweep
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        println!("\nfiltered HNSW walk forced (PreferLatency), by ef:");
        println!("| selectivity | ef | p50 ms | p99 ms | recall@10 |");
        println!("|---|---|---|---|---|");
        for pct in [50u64, 25, 20] {
            let filter = FilterExpr::Lt(0, FilterValue::U64(pct));
            let exact_opts = QueryOptions {
                filter: Some(filter.clone()),
                force_exact: true,
                ..Default::default()
            };
            let truth: Vec<Vec<u64>> = qs
                .iter()
                .map(|q| {
                    store
                        .query(q, k, &exact_opts)
                        .unwrap()
                        .iter()
                        .map(|r| r.id)
                        .collect()
                })
                .collect();
            for &ef in &efs {
                let opts = QueryOptions {
                    filter: Some(filter.clone()),
                    ef_search: ef,
                    quality_preference: rvf_types::quality::QualityPreference::PreferLatency,
                    ..Default::default()
                };
                let (mut t_ms, mut recall) = (Vec::new(), 0.0);
                for (q, exact) in qs.iter().zip(&truth) {
                    let t = Instant::now();
                    let got = store.query(q, k, &opts).unwrap();
                    t_ms.push(t.elapsed().as_secs_f64() * 1e3);
                    recall += got.iter().filter(|r| exact.contains(&r.id)).count() as f64
                        / exact.len().max(1) as f64;
                }
                t_ms.sort_by(f64::total_cmp);
                println!(
                    "| {pct}% | {ef} | {:.3} | {:.3} | {:.3} |",
                    percentile(&t_ms, 0.5),
                    percentile(&t_ms, 0.99),
                    recall / qs.len() as f64
                );
            }
        }
    }
    store.close().unwrap();
}
