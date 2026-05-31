//! IVF-PQ demo: trains on multi-cluster Gaussian data, sweeps nprobe,
//! reports recall@10 / QPS / memory with real cargo-run numbers.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_ivfpq::{IvfPqConfig, IvfPqIndex};
use std::time::Instant;

const N: usize = 10_000;
const DIM: usize = 128;
const NCLUSTERS: usize = 20;
const NQUERIES: usize = 200;
const K: usize = 10;

fn gen_clusters(n: usize, dim: usize, k: usize, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..k)
        .map(|_| (0..dim).map(|_| rng.gen_range(-10.0_f32..10.0)).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = rng.gen_range(0..k);
            (0..dim)
                .map(|d| centers[c][d] + (rng.gen::<f32>() - 0.5) * 2.0 * sigma)
                .collect()
        })
        .collect()
}

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

fn brute_top_k(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut d: Vec<(usize, f32)> =
        corpus.iter().enumerate().map(|(i, v)| (i, l2sq(q, v))).collect();
    d.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    d.iter().take(k).map(|(i, _)| *i).collect()
}

fn recall(gt: &[usize], found: &[usize]) -> f32 {
    let s: std::collections::HashSet<usize> = found.iter().copied().collect();
    gt.iter().filter(|&&id| s.contains(&id)).count() as f32 / gt.len() as f32
}

fn main() {
    println!("=== ruvector IVF-PQ PoC (HAKES filter-refine) ===");
    println!("  N={N}  D={DIM}  clusters={NCLUSTERS}  queries={NQUERIES}  K={K}");
    println!();

    let corpus = gen_clusters(N, DIM, NCLUSTERS, 0.5, 0);
    let queries = gen_clusters(NQUERIES, DIM, NCLUSTERS, 0.5, 1);

    // Ground truth via exact brute-force
    let t0 = Instant::now();
    let gt: Vec<Vec<usize>> = queries.iter().map(|q| brute_top_k(&corpus, q, K)).collect();
    println!("Brute-force ground truth : {:4} ms  ({NQUERIES} queries)", t0.elapsed().as_millis());

    // Build and train the index
    let cfg = IvfPqConfig { nlist: 64, m: 8, ksub: 256, nprobe: 4, rerank_k: 50, max_iter: 30 };
    let mut idx = IvfPqIndex::new(cfg);

    let t0 = Instant::now();
    idx.train(&corpus);
    println!("Train (IVF+PQ k-means)   : {:4} ms", t0.elapsed().as_millis());

    let t0 = Instant::now();
    idx.add(&corpus);
    println!("Add {N} vectors           : {:4} ms", t0.elapsed().as_millis());

    let mem_full_kb = idx.memory_bytes() / 1024;
    let mem_comp_kb = idx.compressed_memory_bytes() / 1024;
    let raw_kb = N * DIM * 4 / 1024;
    println!(
        "Memory (full/comp/raw)   : {mem_full_kb} KB / {mem_comp_kb} KB / {raw_kb} KB"
    );
    println!("Compression ratio        : {:.1}x  (raw / compressed codes only)",
        raw_kb as f32 / mem_comp_kb as f32);
    println!();

    // Search sweep — 3 variants spanning the recall/speed tradeoff.
    // rerank_k is set to ~20% of the probed candidates (nprobe × list_avg)
    // which is empirically sufficient for well-clustered data.
    let list_avg = N / 64;  // ≈156 entries per list at nlist=64
    let configs: &[(&str, usize, usize)] = &[
        ("nprobe=1  rerank_k=50 ", 1,  (list_avg / 2).max(50)),
        ("nprobe=4  rerank_k=200", 4,  (list_avg * 4 / 3).max(200)),
        ("nprobe=16 rerank_k=500", 16, (list_avg * 4).max(500)),
    ];
    println!("{:<28} {:>11} {:>12} {:>10}", "Variant", "Recall@10", "QPS", "Mem(KB)");
    println!("{}", "─".repeat(65));

    for &(label, nprobe, rerank_k) in configs {
        let t0 = Instant::now();
        let found: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| idx.search(q, K, nprobe, rerank_k).into_iter().map(|r| r.id).collect())
            .collect();
        let elapsed = t0.elapsed().as_secs_f64();

        let avg_recall: f32 =
            found.iter().zip(gt.iter()).map(|(f, g)| recall(g, f)).sum::<f32>() / NQUERIES as f32;
        let qps = NQUERIES as f64 / elapsed;

        println!(
            "{label:<28} {:>10.1}%  {:>11.0}  {:>9}",
            avg_recall * 100.0,
            qps,
            mem_full_kb
        );
    }

    println!();
    println!("Hardware: {} ({})", std::env::consts::ARCH, std::env::consts::OS);
    println!("Built with: cargo run --release -p ruvector-ivfpq");
}
