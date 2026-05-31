/// SymphonyQG unified benchmark harness.
///
/// Runs three index variants (FlatExact, GraphExact, SymphonyQG) against the
/// same Gaussian-clustered dataset and reports recall@10, QPS, and memory.
///
///   cargo run --release -p ruvector-symphonyqg -- [--fast]
///
/// `--fast` limits to n=5K only (sub-5 s run).
use std::collections::HashSet;
use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

use ruvector_symphonyqg::{build_all, Config, Metric};

fn main() {
    let fast = std::env::args().any(|a| a == "--fast");
    let ns: &[usize] = if fast {
        &[1_000, 5_000]
    } else {
        &[1_000, 5_000, 50_000]
    };
    let dim = 128usize;
    let n_queries = 500usize;
    let k = 10usize;
    let ef_values: &[usize] = &[50, 100, 200];
    let m_base = 16usize;

    println!("\n╔═══ ruvector-symphonyqg benchmark (SymphonyQG, SIGMOD 2025) ═══╗");
    println!("  dim={dim}, queries={n_queries}, k={k}, m_base={m_base}, BATCH_SIZE=32");
    println!(
        "  Hardware: {} ({})",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("  Rust release build with LLVM auto-vectorisation\n");

    let header = format!(
        "  {:<14} {:>6} {:>5} {:>10} {:>14} {:>12}",
        "Variant", "n", "ef", "Recall@10", "QPS", "Memory"
    );
    let sep = "─".repeat(header.len());
    println!("{sep}");
    println!("{header}");
    println!("{sep}");

    for &n in ns {
        let (data, queries, ground_truth) =
            generate_dataset(n, n_queries, dim, /* seed */ 0xC0FFEE);

        let cfg = Config {
            dim,
            m_base,
            ef_construction: 200.min(n - 1),
            metric: Metric::Euclidean,
            seed: 42,
            // Default benchmark omits Vamana for backward-compatible
            // numbers; the `--vamana` flag below opts into refinement.
            vamana: if std::env::args().any(|a| a == "--vamana") {
                Some(ruvector_symphonyqg::vamana::VamanaConfig::default())
            } else {
                None
            },
        };

        // ── Build ──────────────────────────────────────────────────────────
        let t_build = Instant::now();
        let (flat, graph_exact, symphony) = build_all(&data, &cfg);
        let build_ms = t_build.elapsed().as_millis();

        // ── FlatExact (no ef; always scans all n) ─────────────────────────
        {
            let (recall, qps) = bench_flat(&flat, &queries, &ground_truth, k);
            let mem = fmt_bytes(flat.memory_bytes());
            println!(
                "  {:<14} {:>6} {:>5} {:>9.1}% {:>14.0} {:>12}",
                "FlatExact",
                n,
                "—",
                recall * 100.0,
                qps,
                mem
            );
        }

        // ── GraphExact + SymphonyQG (sweep ef) ────────────────────────────
        for &ef in ef_values {
            let (ge_recall, ge_qps) =
                bench_graph_exact(&graph_exact, &queries, &ground_truth, k, ef);
            let (sq_recall, sq_qps) = bench_symphony(&symphony, &queries, &ground_truth, k, ef);

            let ge_mem = fmt_bytes(graph_exact.memory_bytes());
            let sq_mem = fmt_bytes(symphony.memory_bytes());

            println!(
                "  {:<14} {:>6} {:>5} {:>9.1}% {:>14.0} {:>12}",
                "GraphExact",
                n,
                ef,
                ge_recall * 100.0,
                ge_qps,
                ge_mem
            );
            println!(
                "  {:<14} {:>6} {:>5} {:>9.1}% {:>14.0} {:>12}",
                "SymphonyQG",
                n,
                ef,
                sq_recall * 100.0,
                sq_qps,
                sq_mem
            );

            if ge_qps > 0.0 {
                let speedup = sq_qps / ge_qps;
                println!("    ↳  SymphonyQG speedup over GraphExact at ef={ef}: {speedup:.2}×");
            }
        }
        println!("  [build n={n}: {build_ms} ms]");
        println!("{sep}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────

fn bench_flat(
    flat: &ruvector_symphonyqg::FlatExactIndex,
    queries: &[Vec<f32>],
    gt: &[Vec<usize>],
    k: usize,
) -> (f64, f64) {
    // Warm up.
    let _ = flat.search(&queries[0], k);

    let t = Instant::now();
    let mut hits = 0usize;
    for (i, q) in queries.iter().enumerate() {
        let res = flat.search(q, k);
        let found: HashSet<usize> = res.iter().map(|r| r.idx).collect();
        hits += gt[i].iter().filter(|&&g| found.contains(&g)).count();
    }
    let elapsed = t.elapsed().as_secs_f64();
    let recall = hits as f64 / (queries.len() * k) as f64;
    let qps = queries.len() as f64 / elapsed;
    (recall, qps)
}

fn bench_graph_exact(
    ge: &ruvector_symphonyqg::GraphExactIndex,
    queries: &[Vec<f32>],
    gt: &[Vec<usize>],
    k: usize,
    ef: usize,
) -> (f64, f64) {
    let _ = ge.search(&queries[0], k, ef);
    let t = Instant::now();
    let mut hits = 0usize;
    for (i, q) in queries.iter().enumerate() {
        let res = ge.search(q, k, ef);
        let found: HashSet<usize> = res.iter().map(|r| r.idx).collect();
        hits += gt[i].iter().filter(|&&g| found.contains(&g)).count();
    }
    let elapsed = t.elapsed().as_secs_f64();
    (
        hits as f64 / (queries.len() * k) as f64,
        queries.len() as f64 / elapsed,
    )
}

fn bench_symphony(
    sq: &ruvector_symphonyqg::SymphonyIndex,
    queries: &[Vec<f32>],
    gt: &[Vec<usize>],
    k: usize,
    ef: usize,
) -> (f64, f64) {
    let _ = sq.search(&queries[0], k, ef);
    let t = Instant::now();
    let mut hits = 0usize;
    for (i, q) in queries.iter().enumerate() {
        let res = sq.search(q, k, ef);
        let found: HashSet<usize> = res.iter().map(|r| r.idx).collect();
        hits += gt[i].iter().filter(|&&g| found.contains(&g)).count();
    }
    let elapsed = t.elapsed().as_secs_f64();
    (
        hits as f64 / (queries.len() * k) as f64,
        queries.len() as f64 / elapsed,
    )
}

// ─────────────────────────────────────────────────────────────────────────────

/// Gaussian-clustered dataset: 100 centroids in [-2, 2]^D, σ=0.5 noise.
fn generate_dataset(
    n: usize,
    n_q: usize,
    dim: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<usize>>) {
    let n_clusters = 100usize.min(n / 5).max(1);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let unif = Uniform::new(-2.0f64, 2.0);
    let noise = Normal::new(0.0f64, 0.5).unwrap();

    let centroids: Vec<Vec<f64>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| unif.sample(&mut rng)).collect())
        .collect();

    let sample = |rng: &mut rand::rngs::StdRng| -> Vec<f32> {
        use rand::Rng as _;
        let c = &centroids[rng.gen_range(0..n_clusters)];
        c.iter().map(|&x| (x + noise.sample(rng)) as f32).collect()
    };

    let data: Vec<Vec<f32>> = (0..n).map(|_| sample(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_q).map(|_| sample(&mut rng)).collect();

    // Exact ground truth (brute-force).
    let gt: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let mut ds: Vec<(f32, usize)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, i)
                })
                .collect();
            ds.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            ds.iter().take(10).map(|&(_, i)| i).collect()
        })
        .collect();

    (data, queries, gt)
}

fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 20 {
        format!("{:.2} MB", b as f64 / (1 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KB", b as f64 / (1 << 10) as f64)
    } else {
        format!("{b} B")
    }
}
