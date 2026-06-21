//! LSM-ANN benchmark binary.
//!
//! Measures insert throughput, search recall@10, and latency for three
//! LSM-ANN variants on a deterministically generated dataset.
//!
//! Usage:
//!   cargo run --release -p ruvector-lsm-ann --bin benchmark
//!
//! Optional env vars:
//!   N_VECS   – dataset size       (default 10000)
//!   DIMS     – vector dimensions  (default 128)
//!   N_QUERY  – number of queries  (default 100)
//!   K        – recall@K           (default 10)

use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_lsm_ann::{
    brute_force_knn, recall_at_k, BaselineLsm, FullLsm, LsmConfig, LsmIndex, TwoTierLsm,
};

fn main() {
    let n_vecs: usize = std::env::var("N_VECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let dims: usize = std::env::var("DIMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let n_query: usize = std::env::var("N_QUERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let k: usize = std::env::var("K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║               ruvector-lsm-ann  Benchmark                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Hardware / environment info
    println!("Environment:");
    println!("  OS      : {}", std::env::consts::OS);
    println!("  Arch    : {}", std::env::consts::ARCH);
    println!("  Rust    : {}", env!("RUSTC_VERSION"));
    println!();

    println!("Dataset:");
    println!("  N_VECS  : {n_vecs}");
    println!("  DIMS    : {dims}");
    println!("  N_QUERY : {n_query}");
    println!("  K       : {k}");
    println!(
        "  Memory  : ~{:.1} MB (raw float32)",
        (n_vecs as f64 * dims as f64 * 4.0) / (1024.0 * 1024.0)
    );
    println!();

    // Generate dataset deterministically (seed = 42).
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();

    let dataset: Vec<Vec<f32>> = (0..n_vecs)
        .map(|_| (0..dims).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    let queries: Vec<Vec<f32>> = (0..n_query)
        .map(|_| (0..dims).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    // Ground truth: brute-force k-NN for each query.
    let all_pairs: Vec<(u64, Vec<f32>)> = dataset
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (i as u64, v))
        .collect();

    let ground_truth: Vec<Vec<(u64, f32)>> = queries
        .iter()
        .map(|q| brute_force_knn(&all_pairs, q, k))
        .collect();

    println!("───────────────────────────────────────────────────────────────────");
    println!(" Variant │ Insert/s │ Mem (MB) │ Recall@{k:<3} │ p50 µs │ p95 µs │ Pass");
    println!("───────────────────────────────────────────────────────────────────");

    let cfg = LsmConfig {
        dims,
        m: 16,
        ef_construction: 200,
        ef_search: 200,
        l0_max: 1_000,
        l1_merge_threshold: 5,
    };

    // Helper to summarise latencies.
    let summary = |latencies_ns: &mut Vec<u128>| -> (f64, f64, f64) {
        latencies_ns.sort_unstable();
        let mean = latencies_ns.iter().sum::<u128>() as f64 / latencies_ns.len() as f64;
        let p50 = latencies_ns[latencies_ns.len() / 2];
        let p95 = latencies_ns[(latencies_ns.len() as f64 * 0.95) as usize];
        (mean / 1_000.0, p50 as f64 / 1_000.0, p95 as f64 / 1_000.0)
    };

    let mut results: Vec<(&str, f64, f64, f64, f64, f64, bool)> = Vec::new();

    // -----------------------------------------------------------------------
    // Variant 1 – Baseline (flat MemTable, brute-force)
    // -----------------------------------------------------------------------
    {
        let mut idx = BaselineLsm::new(cfg.clone());

        let t_insert = Instant::now();
        for (i, v) in dataset.iter().enumerate() {
            idx.insert(i as u64, v.clone());
        }
        let insert_secs = t_insert.elapsed().as_secs_f64();
        let insert_rate = n_vecs as f64 / insert_secs;

        let mem_mb = idx.memory_bytes() as f64 / (1024.0 * 1024.0);

        let mut latencies: Vec<u128> = Vec::with_capacity(n_query);
        let mut total_recall = 0.0_f64;
        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let res = idx.search(q, k);
            latencies.push(t.elapsed().as_nanos());
            total_recall += recall_at_k(&res, &ground_truth[qi], k);
        }
        let recall = total_recall / n_query as f64;
        let (_mean_us, p50_us, p95_us) = summary(&mut latencies);

        // Baseline is the oracle — recall should be exactly 1.0 (brute-force).
        let pass = recall >= 0.999;
        results.push((
            "Baseline (L0 only)",
            insert_rate,
            mem_mb,
            recall,
            p50_us,
            p95_us,
            pass,
        ));
        print_row(
            "Baseline",
            insert_rate,
            mem_mb,
            recall,
            p50_us,
            p95_us,
            k,
            pass,
        );
    }

    // -----------------------------------------------------------------------
    // Variant 2 – TwoTier (MemTable + one frozen NSW segment)
    // -----------------------------------------------------------------------
    {
        let mut idx = TwoTierLsm::new(cfg.clone());

        let t_insert = Instant::now();
        for (i, v) in dataset.iter().enumerate() {
            idx.insert(i as u64, v.clone());
        }
        // Flush remaining L0 into the segment.
        idx.compact();
        let insert_secs = t_insert.elapsed().as_secs_f64();
        let insert_rate = n_vecs as f64 / insert_secs;

        let mem_mb = idx.memory_bytes() as f64 / (1024.0 * 1024.0);

        let mut latencies: Vec<u128> = Vec::with_capacity(n_query);
        let mut total_recall = 0.0_f64;
        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let res = idx.search(q, k);
            latencies.push(t.elapsed().as_nanos());
            total_recall += recall_at_k(&res, &ground_truth[qi], k);
        }
        let recall = total_recall / n_query as f64;
        let (_, p50_us, p95_us) = summary(&mut latencies);

        let pass = recall >= 0.85;
        results.push((
            "TwoTier (L0+L1)",
            insert_rate,
            mem_mb,
            recall,
            p50_us,
            p95_us,
            pass,
        ));
        print_row(
            "TwoTier ",
            insert_rate,
            mem_mb,
            recall,
            p50_us,
            p95_us,
            k,
            pass,
        );
    }

    // -----------------------------------------------------------------------
    // Variant 3 – FullLsm (MemTable + L1 segments + L2 merged segment)
    // -----------------------------------------------------------------------
    {
        let mut idx = FullLsm::new(cfg.clone());

        let t_insert = Instant::now();
        for (i, v) in dataset.iter().enumerate() {
            idx.insert(i as u64, v.clone());
        }
        // Final compaction to promote any residual L0/L1 data.
        idx.compact();
        let insert_secs = t_insert.elapsed().as_secs_f64();
        let insert_rate = n_vecs as f64 / insert_secs;

        let mem_mb = idx.memory_bytes() as f64 / (1024.0 * 1024.0);

        let mut latencies: Vec<u128> = Vec::with_capacity(n_query);
        let mut total_recall = 0.0_f64;
        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let res = idx.search(q, k);
            latencies.push(t.elapsed().as_nanos());
            total_recall += recall_at_k(&res, &ground_truth[qi], k);
        }
        let recall = total_recall / n_query as f64;
        let (_, p50_us, p95_us) = summary(&mut latencies);

        let pass = recall >= 0.85;
        results.push((
            "FullLsm (L0+L1+L2)",
            insert_rate,
            mem_mb,
            recall,
            p50_us,
            p95_us,
            pass,
        ));
        print_row(
            "FullLsm ",
            insert_rate,
            mem_mb,
            recall,
            p50_us,
            p95_us,
            k,
            pass,
        );
    }

    println!("───────────────────────────────────────────────────────────────────");
    println!();

    // -----------------------------------------------------------------------
    // Acceptance criteria
    // -----------------------------------------------------------------------
    println!("Acceptance criteria:");
    println!("  Baseline recall@{k} ≥ 0.999 (brute-force oracle)");
    println!("  TwoTier  recall@{k} ≥ 0.850 (NSW approximation acceptable)");
    println!("  FullLsm  recall@{k} ≥ 0.850 (multi-tier merge acceptable)");
    println!();

    let all_pass = results.iter().all(|r| r.6);
    println!(
        "Overall result: {}",
        if all_pass { "PASS ✓" } else { "FAIL ✗" }
    );

    if !all_pass {
        for (name, _, _, recall, _, _, pass) in &results {
            if !pass {
                eprintln!("  FAIL: {name} recall={recall:.4}");
            }
        }
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn print_row(
    name: &str,
    insert_rate: f64,
    mem_mb: f64,
    recall: f64,
    p50_us: f64,
    p95_us: f64,
    _k: usize,
    pass: bool,
) {
    println!(
        " {name:<8} │ {:>7.0}/s │ {:>7.1}  │  {:.4}   │ {:>6.1} │ {:>6.1} │ {}",
        insert_rate,
        mem_mb,
        recall,
        p50_us,
        p95_us,
        if pass { "PASS" } else { "FAIL" }
    );
}
