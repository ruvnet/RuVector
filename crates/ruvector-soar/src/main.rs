//! SOAR-IVF benchmark harness.
//!
//! Produces the recall@10, QPS, memory, and build-time numbers reported in
//! docs/research/nightly/2026-05-08-soar-ivf/README.md.
//!
//! Usage:
//!   cargo run --release -p ruvector-soar                 # full (n=10k, D=128)
//!   cargo run --release -p ruvector-soar -- --fast       # smoke  (n=2k, D=64)

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

use ruvector_soar::{
    index::{recall_at_k, IndexKind, SoarConfig, SoarIndex},
};

// ── data generation ───────────────────────────────────────────────────────────

/// Clustered-Gaussian corpus: `n_clusters` centroids in [-2,2]^D, σ=0.6 noise.
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

/// Compute exact top-k neighbour IDs for each query (brute force ground truth).
fn ground_truth(corpus: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    use ruvector_soar::kmeans::l2_sq;
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(usize, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2_sq(q, v)))
                .collect();
            dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.into_iter().take(k).map(|(i, _)| i).collect()
        })
        .collect()
}

// ── result row ────────────────────────────────────────────────────────────────

struct Row {
    label: String,
    recall: f64,
    qps: f64,
    mem_kb: f64,
    build_ms: f64,
    lat_ms: f64,
}

fn print_header() {
    println!(
        "  {:<28} {:>8} {:>8} {:>9} {:>10} {:>9}",
        "variant", "recall@10", "QPS", "mem/KB", "build/ms", "lat/ms"
    );
    println!("  {}", "-".repeat(80));
}

fn print_row(r: &Row) {
    println!(
        "  {:<28} {:>7.1}% {:>8.0} {:>9.1} {:>10.1} {:>9.3}",
        r.label,
        r.recall * 100.0,
        r.qps,
        r.mem_kb,
        r.build_ms,
        r.lat_ms
    );
}

// ── benchmark one variant ─────────────────────────────────────────────────────

fn bench(
    label: &str,
    corpus: Vec<Vec<f32>>,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    config: SoarConfig,
    k: usize,
) -> Row {
    // Build
    let t0 = Instant::now();
    let idx = SoarIndex::build(corpus, config).expect("build failed");
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mem_kb = idx.index_bytes() as f64 / 1024.0;

    // Warm-up pass (1 query)
    let _ = idx.search(&queries[0], k);

    // Timed pass
    let nq = queries.len();
    let t1 = Instant::now();
    let mut recall_sum = 0.0;
    for (q, tr) in queries.iter().zip(truth.iter()) {
        let res = idx.search(q, k).expect("search failed");
        recall_sum += recall_at_k(tr, &res, k);
    }
    let elapsed = t1.elapsed().as_secs_f64();
    let qps = nq as f64 / elapsed;
    let lat_ms = elapsed * 1000.0 / nq as f64;
    let recall = recall_sum / nq as f64;

    Row {
        label: label.to_string(),
        recall,
        qps,
        mem_kb,
        build_ms,
        lat_ms,
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let fast = std::env::args().any(|a| a == "--fast");
    let (n, d, nq, nlist, nprobe_values): (usize, usize, usize, usize, &[usize]) = if fast {
        (2_000, 64, 100, 20, &[1, 4, 8])
    } else {
        (10_000, 128, 500, 64, &[2, 8, 16])
    };
    let k = 10;
    let n_clusters = (nlist / 2).max(1);

    println!("\nSOAR-IVF benchmark — ruvector-soar");
    println!("  n={n}, D={d}, queries={nq}, nlist={nlist}, k@{k}");
    println!("  Hardware: {}", hardware_string());
    println!();

    // Generate shared corpus + queries
    let corpus_seed: Vec<Vec<f32>> = generate_clustered(n, d, n_clusters, 1);
    let queries: Vec<Vec<f32>> = generate_clustered(nq, d, n_clusters, 2);
    let truth = ground_truth(&corpus_seed, &queries, k);

    for &nprobe in nprobe_values {
        println!("── nprobe={nprobe} ────────────────────────────────────");
        print_header();

        // 1. Flat exact baseline (nprobe irrelevant)
        let flat_cfg = SoarConfig {
            kind: IndexKind::Flat,
            nlist,
            nprobe,
            ..Default::default()
        };
        let r = bench("Flat-Exact (baseline)", corpus_seed.clone(), &queries, &truth, flat_cfg, k);
        print_row(&r);

        // 2. IVF-PQ (no SOAR)
        let ivf_cfg = SoarConfig {
            kind: IndexKind::IvfPq,
            nlist,
            nprobe,
            m_pq: d / 8,
            ..Default::default()
        };
        let r = bench(
            &format!("IVF-PQ (nprobe={nprobe})"),
            corpus_seed.clone(),
            &queries,
            &truth,
            ivf_cfg,
            k,
        );
        print_row(&r);

        // 3. SOAR-IVF-PQ
        let soar_cfg = SoarConfig {
            kind: IndexKind::SoarIvfPq,
            nlist,
            nprobe,
            m_pq: d / 8,
            lambda: 1.0,
            n_secondary_candidates: 10,
            ..Default::default()
        };
        let r = bench(
            &format!("SOAR-IVF-PQ (nprobe={nprobe})"),
            corpus_seed.clone(),
            &queries,
            &truth,
            soar_cfg,
            k,
        );
        print_row(&r);

        println!();
    }

    println!("Done.");
}

fn hardware_string() -> String {
    // Best-effort: reads /proc/cpuinfo on Linux
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.splitn(2, ':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown CPU".into())
}
