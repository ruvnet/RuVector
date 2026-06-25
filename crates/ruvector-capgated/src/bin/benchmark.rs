/// Capability-Gated ANN Benchmark
///
/// Measures three variants (PostFilter, EagerMask, CapGraph) across two
/// access ratios (high / low) on a synthetic 128-dim dataset.
///
/// Run:
///   cargo run --release -p ruvector-capgated --bin benchmark
use ruvector_capgated::{
    cap_graph::CapGraphIndex,
    dataset::{generate, generate_queries, DatasetConfig},
    eager_mask::EagerMaskIndex,
    oracle::Oracle,
    post_filter::PostFilterIndex,
    recall_at_k, CapGatedIndex, CapMask,
};
use std::time::Instant;

// ─── benchmark config ────────────────────────────────────────────────────────

const N_VECTORS: usize = 5_000;
const DIMS: usize = 64;
const N_CAPS: u8 = 8; // 8 distinct capability bits
const REQUIRED_PER_VEC: u8 = 1; // each vector requires exactly 1 cap bit
const N_QUERIES: usize = 200;
const K: usize = 10;
const GRAPH_DEGREE: usize = 12;
const GRAPH_ENTRY_POINTS: usize = 8;
const SEED: u64 = 0x00c0_ffee_1337;

// ─── scenario: access ratio ───────────────────────────────────────────────────

struct Scenario {
    name: &'static str,
    held_caps: u8, // querier holds this many capability bits
}

// ─── timing helpers ───────────────────────────────────────────────────────────

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * p / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn memory_estimate_mb(n: usize, dims: usize, degree: usize) -> f64 {
    let vec_bytes = n * dims * 4; // f32
    let graph_bytes = n * degree * 8; // usize (8 bytes)
    let cap_bytes = n * 8; // u64
    (vec_bytes + graph_bytes + cap_bytes) as f64 / 1_048_576.0
}

// ─── acceptance test ─────────────────────────────────────────────────────────

const MIN_RECALL_POST_FILTER: f32 = 0.95;
const MIN_RECALL_EAGER_MASK: f32 = 0.95;
const MIN_RECALL_CAP_GRAPH: f32 = 0.70;
const MIN_QPS_POST_FILTER: f64 = 1_000.0;
const MIN_QPS_EAGER_MASK: f64 = 1_000.0;
const MIN_QPS_CAP_GRAPH: f64 = 500.0;

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    print_header();

    let scenarios = [
        Scenario {
            name: "high-access (3/8 caps)",
            held_caps: 3,
        },
        Scenario {
            name: "low-access (1/8 caps)",
            held_caps: 1,
        },
    ];

    let mut all_pass = true;

    for scenario in &scenarios {
        println!("\n╔══════════════════════════════════════════════════════╗");
        println!("║  Scenario: {}  ║", pad(scenario.name, 41));
        println!("╚══════════════════════════════════════════════════════╝");

        let (pass, results) = run_scenario(scenario);
        if !pass {
            all_pass = false;
        }
        print_results_table(&results);
    }

    println!();
    if all_pass {
        println!("ACCEPTANCE RESULT: PASS ✓ — all recall and QPS thresholds met");
    } else {
        println!("ACCEPTANCE RESULT: FAIL ✗ — one or more thresholds not met");
        std::process::exit(1);
    }
}

struct BenchResult {
    variant: &'static str,
    n_vectors: usize,
    dims: usize,
    n_queries: usize,
    mean_us: f64,
    p50_us: u128,
    p95_us: u128,
    qps: f64,
    recall: f32,
    auth_frac: f32,
    mem_mb: f64,
    pass: bool,
}

fn run_scenario(scenario: &Scenario) -> (bool, Vec<BenchResult>) {
    // Build dataset
    let cfg = DatasetConfig {
        n_vectors: N_VECTORS,
        dims: DIMS,
        n_caps: N_CAPS,
        required_per_vector: REQUIRED_PER_VEC,
        seed: SEED,
    };
    let entries = generate(&cfg);
    let (queries, holder) =
        generate_queries(N_QUERIES, DIMS, N_CAPS, scenario.held_caps, SEED ^ 0xabcd);

    // Authorised fraction
    let auth_count = entries
        .iter()
        .filter(|e| holder.satisfies(e.required))
        .count();
    let auth_frac = auth_count as f32 / N_VECTORS as f32;
    println!(
        "\n  Holder mask: 0b{:08b} | Authorised: {}/{} ({:.1}%)",
        holder.0,
        auth_count,
        N_VECTORS,
        auth_frac * 100.0
    );

    // Build oracle
    let mut oracle = Oracle::new(DIMS);
    for e in &entries {
        oracle.insert(e.id, e.clone().vector, e.required);
    }
    let oracle_results: Vec<Vec<_>> = queries
        .iter()
        .map(|q| oracle.search(q, K, holder))
        .collect();

    // ─── PostFilter ───────────────────────────────────────────────────────
    let mut pf = PostFilterIndex::new(DIMS);
    for e in &entries {
        pf.insert(e.id, e.vector.clone(), e.required);
    }
    let (pf_rec, pf_mean, pf_p50, pf_p95, pf_qps) =
        bench_variant(&mut pf, &queries, &oracle_results, holder);
    let mem_flat = memory_estimate_mb(N_VECTORS, DIMS, 0);
    let pf_pass = pf_rec >= MIN_RECALL_POST_FILTER && pf_qps >= MIN_QPS_POST_FILTER;

    // ─── EagerMask ───────────────────────────────────────────────────────
    let mut em = EagerMaskIndex::new(DIMS);
    for e in &entries {
        em.insert(e.id, e.vector.clone(), e.required);
    }
    let (em_rec, em_mean, em_p50, em_p95, em_qps) =
        bench_variant(&mut em, &queries, &oracle_results, holder);
    let em_pass = em_rec >= MIN_RECALL_EAGER_MASK && em_qps >= MIN_QPS_EAGER_MASK;

    // ─── CapGraph ────────────────────────────────────────────────────────
    println!("  Building CapGraph (k-NN graph, n={N_VECTORS})... ");
    let build_start = Instant::now();
    let mut cg = CapGraphIndex::new(DIMS, GRAPH_DEGREE, GRAPH_ENTRY_POINTS);
    // Use batch_build to build the graph once (not O(n³) via per-insert rebuild)
    cg.batch_build(entries.iter().map(|e| (e.id, e.vector.clone(), e.required)));
    println!(
        "  Graph built in {:.2}s",
        build_start.elapsed().as_secs_f64()
    );
    let mem_graph = memory_estimate_mb(N_VECTORS, DIMS, GRAPH_DEGREE);
    let (cg_rec, cg_mean, cg_p50, cg_p95, cg_qps) =
        bench_variant(&mut cg, &queries, &oracle_results, holder);
    let cg_pass = cg_rec >= MIN_RECALL_CAP_GRAPH && cg_qps >= MIN_QPS_CAP_GRAPH;

    let all_pass = pf_pass && em_pass && cg_pass;

    let results = vec![
        BenchResult {
            variant: "PostFilter",
            n_vectors: N_VECTORS,
            dims: DIMS,
            n_queries: N_QUERIES,
            mean_us: pf_mean,
            p50_us: pf_p50,
            p95_us: pf_p95,
            qps: pf_qps,
            recall: pf_rec,
            auth_frac,
            mem_mb: mem_flat,
            pass: pf_pass,
        },
        BenchResult {
            variant: "EagerMask",
            n_vectors: N_VECTORS,
            dims: DIMS,
            n_queries: N_QUERIES,
            mean_us: em_mean,
            p50_us: em_p50,
            p95_us: em_p95,
            qps: em_qps,
            recall: em_rec,
            auth_frac,
            mem_mb: mem_flat,
            pass: em_pass,
        },
        BenchResult {
            variant: "CapGraph",
            n_vectors: N_VECTORS,
            dims: DIMS,
            n_queries: N_QUERIES,
            mean_us: cg_mean,
            p50_us: cg_p50,
            p95_us: cg_p95,
            qps: cg_qps,
            recall: cg_rec,
            auth_frac,
            mem_mb: mem_graph,
            pass: cg_pass,
        },
    ];
    (all_pass, results)
}

fn bench_variant(
    idx: &mut dyn CapGatedIndex,
    queries: &[Vec<f32>],
    oracle_results: &[Vec<ruvector_capgated::SearchResult>],
    holder: CapMask,
) -> (f32, f64, u128, u128, f64) {
    let mut latencies_us: Vec<u128> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;

    for (q, oracle) in queries.iter().zip(oracle_results.iter()) {
        let t0 = Instant::now();
        let res = idx.search(q, K, holder);
        latencies_us.push(t0.elapsed().as_micros());
        total_recall += recall_at_k(oracle, &res, K);
    }

    latencies_us.sort_unstable();
    let mean_us = latencies_us.iter().sum::<u128>() as f64 / latencies_us.len() as f64;
    let p50 = percentile(&latencies_us, 50.0);
    let p95 = percentile(&latencies_us, 95.0);
    let qps = 1_000_000.0 / mean_us;
    let recall = total_recall / queries.len() as f32;
    (recall, mean_us, p50, p95, qps)
}

fn print_results_table(results: &[BenchResult]) {
    println!();
    println!(
        "  {:<12} {:>6} {:>5} {:>7} {:>9} {:>9} {:>9} {:>10} {:>7} {:>8} {:>6}",
        "Variant",
        "N",
        "Dims",
        "Queries",
        "Mean(μs)",
        "p50(μs)",
        "p95(μs)",
        "QPS",
        "Recall",
        "Mem(MB)",
        "Pass"
    );
    println!("  {}", "-".repeat(100));
    for r in results {
        println!(
            "  {:<12} {:>6} {:>5} {:>7} {:>9.1} {:>9} {:>9} {:>10.0} {:>7.3} {:>8.2} {:>6}",
            r.variant,
            r.n_vectors,
            r.dims,
            r.n_queries,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.qps,
            r.recall,
            r.mem_mb,
            if r.pass { "PASS" } else { "FAIL" }
        );
    }
    println!();
    println!("  Thresholds:");
    println!(
        "    PostFilter: recall ≥ {MIN_RECALL_POST_FILTER:.2}, QPS ≥ {MIN_QPS_POST_FILTER:.0}"
    );
    println!("    EagerMask:  recall ≥ {MIN_RECALL_EAGER_MASK:.2}, QPS ≥ {MIN_QPS_EAGER_MASK:.0}");
    println!("    CapGraph:   recall ≥ {MIN_RECALL_CAP_GRAPH:.2}, QPS ≥ {MIN_QPS_CAP_GRAPH:.0}");
}

fn print_header() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ruvector-capgated: Capability-Gated ANN Benchmark");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  OS:      {}", std::env::consts::OS);
    println!("  Arch:    {}", std::env::consts::ARCH);
    println!("  Dataset: {N_VECTORS} vectors × {DIMS} dims");
    println!("  Queries: {N_QUERIES}  k={K}");
    println!("  Caps:    {N_CAPS} bits, {REQUIRED_PER_VEC} required/vector");
    println!("  Graph:   degree={GRAPH_DEGREE}, entry_points={GRAPH_ENTRY_POINTS}");
    println!("═══════════════════════════════════════════════════════════════");
}

fn pad(s: &str, width: usize) -> String {
    format!("{:<width$}", s, width = width)
}
