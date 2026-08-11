//! Graph-Neighbour Cascade ANN — benchmark binary.
//!
//! Compares four configurations on a deterministic Gaussian dataset:
//!   1. LinearFull            — brute-force f32 (ground truth)
//!   2. QuantizedCascade_1    — INT8 scan, pool = k  (ef_mult=1, tight)
//!   3. QuantizedCascade_4    — INT8 scan, pool = 4k (ef_mult=4, generous)
//!   4. GraphNeighbourCascade — INT8 scan, pool = k, graph expansion of uncertain zone
//!
//! Research claim: GraphNeighbourCascade (ef_mult=1 + graph) achieves recall
//! closer to QuantizedCascade_4 than QuantizedCascade_1, by recovering missed
//! true neighbours via the k-NN graph without a full second scan.
//!
//! Dataset: 5 000 Gaussian random vectors, 128 dims.
//! In 128-D, distance concentration makes consecutive NN scores nearly equal;
//! INT8 rank inversions are measurable and recoverable via graph neighbours.

use std::time::{Duration, Instant};

use ruvector_cascade_ann::{
    dataset::generate_gaussian,
    recall_at_k,
    variants::{GraphNeighbourCascade, LinearFull, QuantizedCascade, DEFAULT_DELTA},
    AnnVariant,
};

// ─── parameters ──────────────────────────────────────────────────────────────

const N: usize = 5_000;
const DIM: usize = 128;
const N_QUERIES: usize = 300;
const K: usize = 10;
const K_GRAPH: usize = 32; // wider graph for more recovery chances
const EF_TIGHT: usize = 1; // pool = k, tight
const EF_WIDE: usize = 4; // pool = 4k, generous
const SEED_CORPUS: u64 = 0x1234_5678_9ABC_DEF0;
const SEED_QUERIES: u64 = 0xFEDC_BA98_7654_3210;

/// Acceptance thresholds.
const RECALL_MIN_TIGHT: f32 = 0.90;
const RECALL_MIN_GRAPH: f32 = 0.95;
/// GraphNeighbourCascade must improve over QuantizedCascade_1 by this margin.
const RECALL_IMPROVEMENT_MIN: f32 = 0.005;

// ─── timing helpers ──────────────────────────────────────────────────────────

fn percentile(sorted_us: &[u64], p: f64) -> u64 {
    let idx = ((p / 100.0) * (sorted_us.len() - 1) as f64).round() as usize;
    sorted_us[idx.min(sorted_us.len() - 1)]
}

fn run(
    variant: &dyn AnnVariant,
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<(Vec<ruvector_cascade_ann::Hit>, Duration)> {
    queries
        .iter()
        .map(|q| {
            let t0 = Instant::now();
            let hits = variant.search(q, k);
            (hits, t0.elapsed())
        })
        .collect()
}

fn print_row(
    name: &str,
    mean_us: u64,
    p50_us: u64,
    p95_us: u64,
    qps: f64,
    mem_mb: f64,
    recall: f32,
    pass: bool,
) {
    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>12.0} {:>10.2} {:>10.3} {:>8}",
        name,
        mean_us,
        p50_us,
        p95_us,
        qps,
        mem_mb,
        recall,
        if pass { "PASS" } else { "FAIL" }
    );
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Graph-Neighbour Cascade ANN Benchmark ===");
    println!("OS     : {}", std::env::consts::OS);
    println!("Arch   : {}", std::env::consts::ARCH);
    println!("Dataset: N={N}, dim={DIM}, queries={N_QUERIES}, k={K}");
    println!("Graph  : K_graph={K_GRAPH} (brute-force k-NN at build time)");
    println!("EF     : tight={EF_TIGHT}  wide={EF_WIDE}  delta={DEFAULT_DELTA}");
    println!();
    println!("Research claim: GNC(ef=1+graph) > QC(ef=1) by ≥{RECALL_IMPROVEMENT_MIN:.3}");
    println!();

    let corpus = generate_gaussian(N, DIM, SEED_CORPUS);
    let queries = generate_gaussian(N_QUERIES, DIM, SEED_QUERIES);

    eprint!("Building LinearFull...");
    let t0 = Instant::now();
    let linear = LinearFull::build(&corpus);
    eprintln!(" {:.2?}", t0.elapsed());

    eprint!("Building QuantizedCascade_1 (ef=1)...");
    let t0 = Instant::now();
    let qc1 = QuantizedCascade::build(&corpus, EF_TIGHT);
    eprintln!(" {:.2?}", t0.elapsed());

    eprint!("Building QuantizedCascade_4 (ef=4)...");
    let t0 = Instant::now();
    let qc4 = QuantizedCascade::build(&corpus, EF_WIDE);
    eprintln!(" {:.2?}", t0.elapsed());

    eprint!("Building GraphNeighbourCascade (ef=1+graph)...");
    let t0 = Instant::now();
    let gnc = GraphNeighbourCascade::build(&corpus, K_GRAPH, EF_TIGHT, DEFAULT_DELTA);
    eprintln!(" {:.2?}", t0.elapsed());

    let ground_truths: Vec<Vec<ruvector_cascade_ann::Hit>> =
        queries.iter().map(|q| linear.search(q, K)).collect();

    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>12} {:>10} {:>10} {:>8}",
        "Variant", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Mem(MB)", "Recall@10", "Result"
    );
    println!("{}", "-".repeat(102));

    let mut pass_count = 0usize;
    let mut fail_count = 0usize;

    struct Row {
        name: String,
        recall: f32,
        mean_us: u64,
    }
    let mut rows: Vec<Row> = Vec::new();

    let configs: Vec<(&dyn AnnVariant, &str, Option<f32>)> = vec![
        (&linear, "LinearFull", None),
        (&qc1, "QuantizedCascade(ef=1)", Some(RECALL_MIN_TIGHT)),
        (&qc4, "QuantizedCascade(ef=4)", Some(RECALL_MIN_TIGHT)),
        (&gnc, "GraphNeighbourCascade", Some(RECALL_MIN_GRAPH)),
    ];

    for (variant, label, threshold) in &configs {
        let results = run(*variant, &queries, K);
        let mut lat_us: Vec<u64> =
            results.iter().map(|(_, d)| d.as_micros() as u64).collect();
        lat_us.sort_unstable();

        let mean_us = lat_us.iter().sum::<u64>() / lat_us.len() as u64;
        let p50_us = percentile(&lat_us, 50.0);
        let p95_us = percentile(&lat_us, 95.0);
        let qps = 1_000_000.0 / mean_us as f64;
        let mem_mb = variant.memory_bytes() as f64 / 1024.0 / 1024.0;

        let recall: f32 = results
            .iter()
            .zip(ground_truths.iter())
            .map(|((hits, _), gt)| recall_at_k(hits, gt, K))
            .sum::<f32>()
            / N_QUERIES as f32;

        let pass = match threshold {
            None => (recall - 1.0).abs() < 0.001,
            Some(min) => recall >= *min,
        };
        if pass { pass_count += 1; } else { fail_count += 1; }

        print_row(label, mean_us, p50_us, p95_us, qps, mem_mb, recall, pass);
        rows.push(Row { name: label.to_string(), recall, mean_us });
    }

    // Cross-variant test: graph cascade beats tight cascade.
    println!();
    let recall_qc1 = rows.iter().find(|r| r.name.contains("ef=1)")).map(|r| r.recall).unwrap_or(0.0);
    let recall_gnc = rows.iter().find(|r| r.name == "GraphNeighbourCascade").map(|r| r.recall).unwrap_or(0.0);
    let recall_qc4 = rows.iter().find(|r| r.name.contains("ef=4)")).map(|r| r.recall).unwrap_or(0.0);
    let mean_qc1   = rows.iter().find(|r| r.name.contains("ef=1)")).map(|r| r.mean_us).unwrap_or(u64::MAX);
    let mean_gnc   = rows.iter().find(|r| r.name == "GraphNeighbourCascade").map(|r| r.mean_us).unwrap_or(u64::MAX);

    let improvement = recall_gnc - recall_qc1;
    let gap_to_wide = recall_qc4 - recall_gnc;

    let cross_pass = improvement >= RECALL_IMPROVEMENT_MIN;
    if cross_pass { pass_count += 1; } else { fail_count += 1; }

    println!(
        "GNC vs QC(ef=1) recall gain : {:.4}  threshold={:.3}  {}",
        improvement, RECALL_IMPROVEMENT_MIN, if cross_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "GNC vs QC(ef=4) recall gap  : {:.4}  (GNC uses 1/{EF_WIDE}th the initial pool)",
        gap_to_wide
    );
    println!(
        "GNC latency vs QC(ef=1)     : {:.1}µs vs {:.1}µs  (graph overhead)",
        mean_gnc, mean_qc1
    );

    println!();
    println!("=== Memory breakdown ===");
    println!("  f32 corpus : {:.2} MB", N * DIM * 4 / 1024 / 1024);
    println!("  u8 corpus  : {:.2} MB", N * DIM / 1024 / 1024);
    println!("  graph adj  : {:.2} MB  ({N} × {K_GRAPH} × 4 B)", N * K_GRAPH * 4 / 1024 / 1024);

    println!();
    println!("=== Summary ===");
    println!("Tests passed: {pass_count}  failed: {fail_count}");

    if fail_count > 0 {
        eprintln!("BENCHMARK FAILED: {fail_count} acceptance test(s) did not pass.");
        std::process::exit(1);
    } else {
        println!("All acceptance tests PASSED.");
    }
}
