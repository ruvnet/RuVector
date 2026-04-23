#![allow(clippy::manual_div_ceil)]
#![allow(clippy::needless_range_loop)]

//! RaBitQ unified benchmark harness — produces the *same-run* recall and
//! throughput numbers quoted in `README.md` / `BENCHMARK.md`.
//!
//! Unlike the previous split "recall in tests / QPS in Criterion" setup, this
//! binary runs every index against the same clustered-Gaussian dataset and
//! reports:
//!
//!   - recall@1, @10, @100
//!   - queries-per-second
//!   - actual memory bytes (honest — all allocations included)
//!   - build time
//!   - per-query latency
//!
//! for n ∈ {1 k, 5 k, 50 k, 100 k}. Runs end-to-end in ~20 s on a commodity
//! laptop; emit `--fast` for the sub-5 s smoke version.
//!
//!   cargo run --release -p ruvector-rabitq --bin rabitq-demo
//!   cargo run --release -p ruvector-rabitq --bin rabitq-demo -- --fast

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::HashSet;
use std::time::Instant;

use ruvector_rabitq::{
    index::{AnnIndex, FlatF32Index, RabitqAsymIndex, RabitqIndex, RabitqPlusIndex},
    SearchResult,
};

/// Gaussian-clustered data. 100 centroids in a [-2,2]^D hypercube with
/// σ=0.6 noise per cluster. Approximates embedding distributions — not a
/// substitute for SIFT1M but publishable as an apples-to-apples baseline
/// across all four indexes.
fn generate_clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::Rng as _;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroid_range = Uniform::new(-2.0f32, 2.0);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| centroid_range.sample(&mut rng)).collect())
        .collect();
    let noise = Normal::new(0.0f64, 0.6).unwrap();
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter()
                .map(|&x| x + noise.sample(&mut rng) as f32)
                .collect()
        })
        .collect()
}

fn recall_at_k(truth: &[usize], got: &[usize]) -> f64 {
    let truth_set: HashSet<usize> = truth.iter().copied().collect();
    got.iter().filter(|id| truth_set.contains(id)).count() as f64 / truth.len() as f64
}

struct SearchRow {
    label: String,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    qps: f64,
    mem_mb: f64,
    latency_ms: f64,
}

fn print_header() {
    println!(
        "  {:<26} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "variant", "r@1", "r@10", "r@100", "QPS", "mem/MB", "lat/ms"
    );
}
fn print_row(r: &SearchRow) {
    println!(
        "  {:<26} {:>7.1}% {:>7.1}% {:>7.1}% {:>8.0} {:>8.1} {:>9.3}",
        r.label,
        r.recall_at_1 * 100.0,
        r.recall_at_10 * 100.0,
        r.recall_at_100 * 100.0,
        r.qps,
        r.mem_mb,
        r.latency_ms
    );
}

/// One `search` sweep: computes recall@1/10/100 and QPS against the same
/// queries / ground-truth lists (computed once, outside).
fn measure<I: AnnIndex>(
    label: &str,
    index: &I,
    queries: &[Vec<f32>],
    truth_at_100: &[Vec<usize>],
    k_max: usize,
) -> SearchRow {
    let t = Instant::now();
    let mut r1 = 0.0f64;
    let mut r10 = 0.0f64;
    let mut r100 = 0.0f64;
    for (i, q) in queries.iter().enumerate() {
        let res: Vec<SearchResult> = index.search(q, k_max).unwrap();
        let ids: Vec<usize> = res.into_iter().map(|r| r.id).collect();
        let truth = &truth_at_100[i];
        r1 += recall_at_k(&truth[..1.min(truth.len())], &ids[..1.min(ids.len())]);
        r10 += recall_at_k(&truth[..10.min(truth.len())], &ids[..10.min(ids.len())]);
        r100 += recall_at_k(truth, &ids[..100.min(ids.len())]);
    }
    let elapsed = t.elapsed();
    let nq = queries.len();
    SearchRow {
        label: label.to_string(),
        recall_at_1: r1 / nq as f64,
        recall_at_10: r10 / nq as f64,
        recall_at_100: r100 / nq as f64,
        qps: nq as f64 / elapsed.as_secs_f64(),
        mem_mb: index.memory_bytes() as f64 / 1_048_576.0,
        latency_ms: elapsed.as_secs_f64() / nq as f64 * 1000.0,
    }
}

fn build_flat(d: usize, data: &[Vec<f32>]) -> (FlatF32Index, f64) {
    let t = Instant::now();
    let mut idx = FlatF32Index::new(d);
    for (i, v) in data.iter().enumerate() {
        idx.add(i, v.clone()).unwrap();
    }
    (idx, t.elapsed().as_secs_f64())
}
fn build_rabitq(d: usize, seed: u64, data: &[Vec<f32>]) -> (RabitqIndex, f64) {
    let t = Instant::now();
    let mut idx = RabitqIndex::new(d, seed);
    for (i, v) in data.iter().enumerate() {
        idx.add(i, v.clone()).unwrap();
    }
    (idx, t.elapsed().as_secs_f64())
}
fn build_plus(d: usize, seed: u64, rerank: usize, data: &[Vec<f32>]) -> (RabitqPlusIndex, f64) {
    let t = Instant::now();
    let mut idx = RabitqPlusIndex::new(d, seed, rerank);
    for (i, v) in data.iter().enumerate() {
        idx.add(i, v.clone()).unwrap();
    }
    (idx, t.elapsed().as_secs_f64())
}
fn build_asym(d: usize, seed: u64, rerank: usize, data: &[Vec<f32>]) -> (RabitqAsymIndex, f64) {
    let t = Instant::now();
    let mut idx = RabitqAsymIndex::new(d, seed, rerank);
    for (i, v) in data.iter().enumerate() {
        idx.add(i, v.clone()).unwrap();
    }
    (idx, t.elapsed().as_secs_f64())
}

/// Compute ground-truth top-100 on flat for recall@k comparison.
fn truth_top_100(flat: &FlatF32Index, queries: &[Vec<f32>]) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|q| {
            flat.search(q, 100)
                .unwrap()
                .into_iter()
                .map(|r| r.id)
                .collect()
        })
        .collect()
}

fn run_scale(n: usize, d: usize, n_clusters: usize, nq: usize, seed: u64, k_max: usize) {
    println!("\n── n = {n} · d = {d} · {n_clusters} clusters · nq = {nq} ──");
    let all = generate_clustered(n + nq, d, n_clusters, seed);
    let (db_vecs, query_vecs) = all.split_at(n);

    let (flat, t_flat) = build_flat(d, db_vecs);
    let queries: Vec<Vec<f32>> = query_vecs.to_vec();
    let truth = truth_top_100(&flat, &queries);

    let (rq, t_rq) = build_rabitq(d, seed, db_vecs);
    let (rq_p5, t_rq_p5) = build_plus(d, seed, 5, db_vecs);
    let (rq_p20, t_rq_p20) = build_plus(d, seed, 20, db_vecs);
    let (rq_a1, t_rq_a1) = build_asym(d, seed, 1, db_vecs);
    let (rq_a5, t_rq_a5) = build_asym(d, seed, 5, db_vecs);

    println!(
        "  build times  flat={:.2}s  rabitq={:.2}s  plus×5={:.2}s  plus×20={:.2}s  asym×1={:.2}s  asym×5={:.2}s",
        t_flat, t_rq, t_rq_p5, t_rq_p20, t_rq_a1, t_rq_a5
    );

    print_header();
    let rows = [
        measure("FlatF32 (exact)", &flat, &queries, &truth, k_max),
        measure(
            "RaBitQ 1-bit (sym, no rerank)",
            &rq,
            &queries,
            &truth,
            k_max,
        ),
        measure("RaBitQ+ (sym, rerank×5)", &rq_p5, &queries, &truth, k_max),
        measure("RaBitQ+ (sym, rerank×20)", &rq_p20, &queries, &truth, k_max),
        measure("RaBitQ-Asym (no rerank)", &rq_a1, &queries, &truth, k_max),
        measure("RaBitQ-Asym (rerank×5)", &rq_a5, &queries, &truth, k_max),
    ];
    for r in &rows {
        print_row(r);
    }

    // Compression: codes-only vs flat-only, for honesty.
    let flat_pure = n * d * 4;
    let codes_pure = n * ((d + 63) / 64 * 8);
    println!(
        "  compression (codes vs f32 data): {:.1}×   ({} B vs {} B per vector)",
        flat_pure as f64 / codes_pure as f64,
        (d + 63) / 64 * 8,
        d * 4
    );
}

fn main() {
    let fast = std::env::args().any(|a| a == "--fast");
    println!("=== RaBitQ unified benchmark harness ===");
    println!(
        "build = release  · deterministic seeds  · clustered Gaussian (SIFT-like){}",
        if fast { "  · --fast mode" } else { "" }
    );
    println!(
        "recall is measured against Flat's exact top-100 on the *same* queries so all\nvariants are apples-to-apples. QPS is wall-clock end-to-end search time."
    );

    let d = 128;
    let n_clusters = 100;
    let nq = if fast { 50 } else { 200 };
    let k_max = 100;

    // Scale sweep.
    let scales = if fast {
        vec![(1_000usize, 42u64), (5_000, 55)]
    } else {
        vec![
            (1_000usize, 42u64),
            (5_000, 55),
            (50_000, 77),
            (100_000, 91),
        ]
    };
    for (n, seed) in scales {
        run_scale(n, d, n_clusters, nq, seed, k_max);
    }

    // Non-aligned D regression demo.
    println!("\n── D = 100 (non-aligned regression demo, n=2000) ──");
    let d = 100;
    let n = 2_000;
    let nq = if fast { 50 } else { 100 };
    let all = generate_clustered(n + nq, d, 20, 123);
    let (db, qs) = all.split_at(n);
    let (flat, _) = build_flat(d, db);
    let truth = truth_top_100(&flat, qs);
    let queries: Vec<Vec<f32>> = qs.to_vec();
    let (rq_p5, _) = build_plus(d, 123, 5, db);
    print_header();
    print_row(&measure("FlatF32", &flat, &queries, &truth, k_max));
    print_row(&measure(
        "RaBitQ+ sym ×5 (D=100)",
        &rq_p5,
        &queries,
        &truth,
        k_max,
    ));

    println!("\nAll numbers reproducible with the seeds above.");
}
