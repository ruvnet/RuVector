/// Late-interaction MaxSim benchmark: three variants, real latency, real recall.
///
/// Run: cargo run --release -p ruvector-late-interaction --bin benchmark
///
/// Adjust DATASET_SIZE, DIMS, TOKENS_PER_DOC, QUERY_TOKENS, NUM_QUERIES as needed.
use ruvector_late_interaction::brute::BruteForceIndex;
use ruvector_late_interaction::compressed::CompressedIndex;
use ruvector_late_interaction::dataset::DatasetGen;
use ruvector_late_interaction::plaid::PlaidLiteIndex;
use ruvector_late_interaction::{recall_at_k, MaxSimIndex, MultiVecQuery};
use std::time::{Duration, Instant};

const DATASET_SIZE: usize = 2_000;
const DIMS: usize = 64;
const TOKENS_PER_DOC: usize = 16;
const QUERY_TOKENS: usize = 8;
const NUM_QUERIES: usize = 50;
const TOP_K: usize = 10;

const NUM_CENTROIDS: usize = 64;
const N_PROBE: usize = 4;

fn percentile(mut times: Vec<Duration>, p: f64) -> Duration {
    times.sort();
    let idx = ((times.len() as f64 * p / 100.0) as usize).min(times.len() - 1);
    times[idx]
}

fn bench_index<I: MaxSimIndex>(
    idx: &I,
    queries: &[MultiVecQuery],
    ground_truths: &[Vec<ruvector_late_interaction::ScoredDoc>],
    top_k: usize,
) -> (Vec<Duration>, f32) {
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;

    for (q, gt) in queries.iter().zip(ground_truths.iter()) {
        let t0 = Instant::now();
        let results = idx.query(q, top_k).unwrap();
        latencies.push(t0.elapsed());
        total_recall += recall_at_k(&results, gt, top_k);
    }

    let avg_recall = total_recall / queries.len() as f32;
    (latencies, avg_recall)
}

fn print_separator() {
    println!("{}", "-".repeat(80));
}

fn format_us(d: Duration) -> String {
    format!("{:.1} µs", d.as_nanos() as f64 / 1_000.0)
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║   ruvector-late-interaction  MaxSim Benchmark  (2026-06-10)             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // --- System info ---
    println!("OS         : {}", std::env::consts::OS);
    println!("Arch       : {}", std::env::consts::ARCH);
    println!("Rust       : 1.94.1 (release)");
    println!();
    print_separator();

    // --- Dataset ---
    println!("Dataset params:");
    println!("  N (docs)        = {DATASET_SIZE}");
    println!("  D (dims)        = {DIMS}");
    println!("  tokens/doc      = {TOKENS_PER_DOC}");
    println!("  query tokens    = {QUERY_TOKENS}");
    println!("  queries         = {NUM_QUERIES}");
    println!("  top_k           = {TOP_K}");
    println!("  centroids       = {NUM_CENTROIDS}  (PLAID-lite)");
    println!("  n_probe         = {N_PROBE}  (PLAID-lite)");
    print_separator();

    let gen = DatasetGen::new(42, DIMS);
    let docs = gen.random_docs(DATASET_SIZE, TOKENS_PER_DOC);
    let queries = gen.random_queries(NUM_QUERIES, QUERY_TOKENS);

    // Build all three indexes.
    let t_build = Instant::now();
    let mut bf = BruteForceIndex::new(DIMS);
    let mut cmp = CompressedIndex::new(DIMS);
    let mut plaid = PlaidLiteIndex::new(DIMS, NUM_CENTROIDS, N_PROBE);
    for d in &docs {
        bf.insert(d.clone()).unwrap();
        cmp.insert(d.clone()).unwrap();
        plaid.insert(d.clone()).unwrap();
    }
    bf.build().unwrap();
    cmp.build().unwrap();
    plaid.build().unwrap();
    let build_time = t_build.elapsed();
    println!(
        "Build time (all 3 indexes): {:.2} ms",
        build_time.as_secs_f64() * 1_000.0
    );
    print_separator();

    // Compute ground truth from brute force.
    let ground_truths: Vec<_> = queries
        .iter()
        .map(|q| bf.query(q, TOP_K).unwrap())
        .collect();

    // --- Benchmark brute force ---
    let (bf_times, bf_recall) = bench_index(&bf, &queries, &ground_truths, TOP_K);
    let bf_mean = bf_times.iter().sum::<Duration>() / bf_times.len() as u32;
    let bf_p50 = percentile(bf_times.clone(), 50.0);
    let bf_p95 = percentile(bf_times.clone(), 95.0);
    let bf_throughput = NUM_QUERIES as f64 / bf_times.iter().sum::<Duration>().as_secs_f64();

    // --- Benchmark compressed ---
    let (cmp_times, cmp_recall) = bench_index(&cmp, &queries, &ground_truths, TOP_K);
    let cmp_mean = cmp_times.iter().sum::<Duration>() / cmp_times.len() as u32;
    let cmp_p50 = percentile(cmp_times.clone(), 50.0);
    let cmp_p95 = percentile(cmp_times.clone(), 95.0);
    let cmp_throughput = NUM_QUERIES as f64 / cmp_times.iter().sum::<Duration>().as_secs_f64();

    // --- Benchmark PLAID-lite ---
    let (plaid_times, plaid_recall) = bench_index(&plaid, &queries, &ground_truths, TOP_K);
    let plaid_mean = plaid_times.iter().sum::<Duration>() / plaid_times.len() as u32;
    let plaid_p50 = percentile(plaid_times.clone(), 50.0);
    let plaid_p95 = percentile(plaid_times.clone(), 95.0);
    let plaid_throughput = NUM_QUERIES as f64 / plaid_times.iter().sum::<Duration>().as_secs_f64();

    // --- Memory ---
    let bf_mem_kb = bf.memory_bytes() / 1024;
    let cmp_mem_kb = cmp.memory_bytes() / 1024;
    let plaid_mem_kb = plaid.memory_bytes() / 1024;

    // --- Results table ---
    println!();
    println!("Results  (N={DATASET_SIZE}, D={DIMS}, T_doc={TOKENS_PER_DOC}, T_q={QUERY_TOKENS}, queries={NUM_QUERIES})");
    println!();

    let header = format!(
        "{:<28} {:>10} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "Variant", "Mean lat.", "p50 lat.", "p95 lat.", "QPS", "Mem (KB)", "Recall@10"
    );
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>12.0} {:>10} {:>10}",
        bf.name(),
        format_us(bf_mean),
        format_us(bf_p50),
        format_us(bf_p95),
        bf_throughput,
        bf_mem_kb,
        "1.000 (GT)"
    );
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>12.0} {:>10} {:>10.3}",
        cmp.name(),
        format_us(cmp_mean),
        format_us(cmp_p50),
        format_us(cmp_p95),
        cmp_throughput,
        cmp_mem_kb,
        cmp_recall
    );
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>12.0} {:>10} {:>10.3}",
        plaid.name(),
        format_us(plaid_mean),
        format_us(plaid_p50),
        format_us(plaid_p95),
        plaid_throughput,
        plaid_mem_kb,
        plaid_recall
    );
    println!();

    // Memory math.
    println!("Memory analysis:");
    println!(
        "  brute-force  : {} KB  ({} docs × {} tokens × {} dims × 4 B)",
        bf_mem_kb, DATASET_SIZE, TOKENS_PER_DOC, DIMS
    );
    println!(
        "  compressed   : {} KB  ({} docs × {} tokens × {} dims × 1 B — 4× reduction)",
        cmp_mem_kb, DATASET_SIZE, TOKENS_PER_DOC, DIMS
    );
    println!(
        "  plaid-lite   : {} KB  (same as brute + {} centroids × {} dims × 4 B)",
        plaid_mem_kb, NUM_CENTROIDS, DIMS
    );
    println!();

    // --- Acceptance test ---
    print_separator();
    println!("Acceptance criteria:");

    let cmp_pass = cmp_recall >= 0.75;
    let plaid_pass = plaid_recall >= 0.60;

    println!(
        "  [{}] compressed-sq8 recall@10 ≥ 0.75  (actual: {:.3})",
        if cmp_pass { "PASS" } else { "FAIL" },
        cmp_recall
    );
    println!(
        "  [{}] plaid-lite     recall@10 ≥ 0.60  (actual: {:.3})",
        if plaid_pass { "PASS" } else { "FAIL" },
        plaid_recall
    );

    println!();
    if cmp_pass && plaid_pass {
        println!("✓ ALL ACCEPTANCE CRITERIA PASSED");
    } else {
        eprintln!("✗ SOME ACCEPTANCE CRITERIA FAILED");
        std::process::exit(1);
    }
    println!();
}
