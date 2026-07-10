/// Benchmark for namespace-partitioned HNSW memory.
///
/// Measures insert throughput, single-namespace search latency, cross-namespace
/// search latency, memory footprint, and recall@10 for three variants:
///
///   GlobalFlat      — one HNSW, filter post-hoc by namespace
///   Partitioned     — per-namespace HNSW
///   HierarchicalNS  — routing index + per-namespace HNSW
use ruvector_ns_partition::{
    hnsw::l2_sq, GlobalFlat, HierarchicalNS, NamespacedIndex, NsResult, Partitioned,
};
use std::time::Instant;

// ── Config ────────────────────────────────────────────────────────────────────

const N_NAMESPACES: usize = 8;
const N_PER_NS: usize = 750; // 750 × 8 = 6 000 total vectors
const DIMS: usize = 128;
const N_QUERIES: usize = 200;
const K: usize = 10;
const EF: usize = 64;
const HNSW_M: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
/// HierarchicalNS: how many namespaces to probe during cross-NS search.
const ROUTE_K: usize = 4;

// ── Deterministic data generation ─────────────────────────────────────────────

fn lcg_next(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 33) as f32) / (u32::MAX as f32) - 0.5
}

fn gen_vectors(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    (0..n)
        .map(|_| (0..dims).map(|_| lcg_next(&mut s)).collect())
        .collect()
}

// ── Brute-force oracle ────────────────────────────────────────────────────────

fn oracle_single(
    corpus: &[Vec<f32>],
    ids: &[u64],
    ns: &str,
    query: &[f32],
    k: usize,
) -> Vec<NsResult> {
    let mut dists: Vec<(f32, u64)> = corpus
        .iter()
        .zip(ids.iter())
        .map(|(v, &id)| (l2_sq(query, v), id))
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists
        .into_iter()
        .take(k)
        .map(|(d, id)| NsResult {
            id,
            namespace: ns.to_owned(),
            dist_sq: d,
        })
        .collect()
}

fn oracle_cross(
    all_ns: &[(&str, Vec<Vec<f32>>, Vec<u64>)],
    query: &[f32],
    k: usize,
) -> Vec<NsResult> {
    let mut all: Vec<NsResult> = all_ns
        .iter()
        .flat_map(|(ns, vecs, ids)| {
            vecs.iter().zip(ids.iter()).map(|(v, &id)| NsResult {
                id,
                namespace: ns.to_string(),
                dist_sq: l2_sq(query, v),
            })
        })
        .collect();
    all.sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
    all.truncate(k);
    all
}

// ── Timing helpers ────────────────────────────────────────────────────────────

fn percentile(sorted: &[u128], p: f64) -> u128 {
    let idx = ((p / 100.0) * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len().saturating_sub(1))]
}

// ── Benchmark runner ──────────────────────────────────────────────────────────

struct BenchResult {
    name: &'static str,
    insert_ms: f64,
    single_ns_mean_us: f64,
    single_ns_p50_us: u128,
    single_ns_p95_us: u128,
    single_ns_tput: f64,
    cross_ns_mean_us: f64,
    cross_ns_p50_us: u128,
    cross_ns_p95_us: u128,
    cross_ns_tput: f64,
    memory_kb: f64,
    single_recall: f64,
    cross_recall: f64,
    accept: bool,
}

fn run_variant(
    idx: &mut dyn NamespacedIndex,
    ns_data: &[(&str, Vec<Vec<f32>>, Vec<u64>)],
    queries: &[Vec<f32>],
    query_ns: &[&str], // which ns each query belongs to
    oracle_single_results: &[Vec<NsResult>],
    oracle_cross_results: &[Vec<NsResult>],
) -> BenchResult {
    let name = idx.name();
    let total_n = N_NAMESPACES * N_PER_NS;

    // ── Insert ───────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    for (ns, vecs, ids) in ns_data.iter() {
        for (v, &id) in vecs.iter().zip(ids.iter()) {
            idx.insert(ns, id, v.clone());
        }
    }
    let insert_ms = t0.elapsed().as_secs_f64() * 1_000.0;

    // ── Single-namespace search ──────────────────────────────────────────────
    let mut single_latencies: Vec<u128> = Vec::with_capacity(N_QUERIES);
    let mut single_recalls = Vec::with_capacity(N_QUERIES);

    for (i, query) in queries.iter().enumerate() {
        let ns = query_ns[i];
        let t = Instant::now();
        let results = idx.search_single(ns, query, K, EF);
        single_latencies.push(t.elapsed().as_micros());
        single_recalls.push(ruvector_ns_partition::recall_at_k(
            &oracle_single_results[i],
            &results,
            K,
        ) as f64);
    }
    single_latencies.sort_unstable();

    let single_mean_us = single_latencies.iter().sum::<u128>() as f64 / N_QUERIES as f64;
    let single_p50 = percentile(&single_latencies, 50.0);
    let single_p95 = percentile(&single_latencies, 95.0);
    let single_tput = N_QUERIES as f64 / (single_mean_us * N_QUERIES as f64 / 1_000_000.0);
    let single_recall_avg = single_recalls.iter().sum::<f64>() / N_QUERIES as f64;

    // ── Cross-namespace search ───────────────────────────────────────────────
    let mut cross_latencies: Vec<u128> = Vec::with_capacity(N_QUERIES);
    let mut cross_recalls = Vec::with_capacity(N_QUERIES);

    for (i, query) in queries.iter().enumerate() {
        let t = Instant::now();
        let results = idx.search_cross(query, K, EF);
        cross_latencies.push(t.elapsed().as_micros());
        cross_recalls
            .push(ruvector_ns_partition::recall_at_k(&oracle_cross_results[i], &results, K) as f64);
    }
    cross_latencies.sort_unstable();

    let cross_mean_us = cross_latencies.iter().sum::<u128>() as f64 / N_QUERIES as f64;
    let cross_p50 = percentile(&cross_latencies, 50.0);
    let cross_p95 = percentile(&cross_latencies, 95.0);
    let cross_tput = N_QUERIES as f64 / (cross_mean_us * N_QUERIES as f64 / 1_000_000.0);
    let cross_recall_avg = cross_recalls.iter().sum::<f64>() / N_QUERIES as f64;

    let memory_kb = idx.memory_bytes() as f64 / 1024.0;

    // Acceptance: both recalls ≥ 0.70 and single-NS latency ≤ 5 ms
    let accept = single_recall_avg >= 0.70 && cross_recall_avg >= 0.60 && single_mean_us <= 5_000.0;

    eprintln!(
        "  [{}] insert={:.1}ms  single_recall={:.0}%  cross_recall={:.0}%  mem={:.0}KB  accept={}",
        name,
        insert_ms,
        single_recall_avg * 100.0,
        cross_recall_avg * 100.0,
        memory_kb,
        if accept { "PASS" } else { "FAIL" }
    );

    BenchResult {
        name,
        insert_ms,
        single_ns_mean_us: single_mean_us,
        single_ns_p50_us: single_p50,
        single_ns_p95_us: single_p95,
        single_ns_tput: single_tput,
        cross_ns_mean_us: cross_mean_us,
        cross_ns_p50_us: cross_p50,
        cross_ns_p95_us: cross_p95,
        cross_ns_tput: cross_tput,
        memory_kb,
        single_recall: single_recall_avg,
        cross_recall: cross_recall_avg,
        accept,
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("=== ruvector-ns-partition benchmark ===");
    println!();
    println!(
        "Hardware : {}",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into())
    );
    println!("OS       : {}", std::env::consts::OS);
    println!("Arch     : {}", std::env::consts::ARCH);
    println!("Rust     : compiled (check `rustc --version` for version)");
    println!();
    println!(
        "Dataset  : {} namespaces × {} vectors = {} total",
        N_NAMESPACES,
        N_PER_NS,
        N_NAMESPACES * N_PER_NS
    );
    println!("Dims     : {}", DIMS);
    println!("Queries  : {}", N_QUERIES);
    println!("k        : {}", K);
    println!("ef       : {}", EF);
    println!("M        : {}", HNSW_M);
    println!("ef_c     : {}", HNSW_EF_CONSTRUCTION);
    println!("RouteK   : {} (HierarchicalNS only)", ROUTE_K);
    println!();

    // ── Generate data ─────────────────────────────────────────────────────────
    let mut ns_data: Vec<(&str, Vec<Vec<f32>>, Vec<u64>)> = Vec::new();
    let ns_names = [
        "agent-alice",
        "agent-bob",
        "agent-carol",
        "agent-dave",
        "agent-eve",
        "agent-frank",
        "agent-grace",
        "agent-heidi",
    ];
    let mut global_id: u64 = 0;
    for (i, &ns) in ns_names.iter().enumerate() {
        let seed = 0xABCD_EF00 + i as u64;
        let vecs = gen_vectors(N_PER_NS, DIMS, seed);
        let ids: Vec<u64> = (0..N_PER_NS as u64)
            .map(|j| {
                let id = global_id;
                global_id += 1;
                id
            })
            .collect();
        ns_data.push((ns, vecs, ids));
    }

    // ── Generate queries (evenly distributed across namespaces) ───────────────
    let queries: Vec<Vec<f32>> = gen_vectors(N_QUERIES, DIMS, 0xDEAD_C0DE);
    let query_ns: Vec<&str> = (0..N_QUERIES).map(|i| ns_names[i % N_NAMESPACES]).collect();

    // ── Build oracle (brute force) ────────────────────────────────────────────
    println!("Building brute-force oracle ...");
    let oracle_single: Vec<Vec<NsResult>> = queries
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let ns = query_ns[i];
            let (_, vecs, ids) = ns_data.iter().find(|(n, _, _)| *n == ns).unwrap();
            oracle_single(vecs, ids, ns, q, K)
        })
        .collect();

    let oracle_cross: Vec<Vec<NsResult>> = queries
        .iter()
        .map(|q| oracle_cross(&ns_data, q, K))
        .collect();
    println!("  done.");
    println!();

    // ── Run variants ──────────────────────────────────────────────────────────
    println!("Running variants ...");
    let mut results: Vec<BenchResult> = Vec::new();

    // Variant 1: GlobalFlat
    {
        let mut idx = GlobalFlat::new(HNSW_M, HNSW_EF_CONSTRUCTION);
        results.push(run_variant(
            &mut idx,
            &ns_data,
            &queries,
            &query_ns,
            &oracle_single,
            &oracle_cross,
        ));
    }

    // Variant 2: Partitioned
    {
        let mut idx = Partitioned::new(HNSW_M, HNSW_EF_CONSTRUCTION);
        results.push(run_variant(
            &mut idx,
            &ns_data,
            &queries,
            &query_ns,
            &oracle_single,
            &oracle_cross,
        ));
    }

    // Variant 3: HierarchicalNS
    {
        let mut idx = HierarchicalNS::new(HNSW_M, HNSW_EF_CONSTRUCTION, DIMS, ROUTE_K);
        results.push(run_variant(
            &mut idx,
            &ns_data,
            &queries,
            &query_ns,
            &oracle_single,
            &oracle_cross,
        ));
    }

    // ── Print table ───────────────────────────────────────────────────────────
    println!();
    println!("=== Results: Single-Namespace Search ===");
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>12} {:>10} {:>8}",
        "Variant", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Recall@10", "Accept"
    );
    println!("{}", "-".repeat(84));
    for r in &results {
        println!(
            "{:<20} {:>10.1} {:>10} {:>10} {:>12.0} {:>9.1}% {:>8}",
            r.name,
            r.single_ns_mean_us,
            r.single_ns_p50_us,
            r.single_ns_p95_us,
            r.single_ns_tput,
            r.single_recall * 100.0,
            if r.accept { "PASS" } else { "FAIL" }
        );
    }

    println!();
    println!("=== Results: Cross-Namespace Search ===");
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>12} {:>10} {:>8}",
        "Variant", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Recall@10", "Memory(KB)"
    );
    println!("{}", "-".repeat(84));
    for r in &results {
        println!(
            "{:<20} {:>10.1} {:>10} {:>10} {:>12.0} {:>9.1}% {:>10.0}",
            r.name,
            r.cross_ns_mean_us,
            r.cross_ns_p50_us,
            r.cross_ns_p95_us,
            r.cross_ns_tput,
            r.cross_recall * 100.0,
            r.memory_kb,
        );
    }

    // ── Acceptance gate ───────────────────────────────────────────────────────
    println!();
    println!("=== Acceptance Gate ===");
    println!("  - Single-NS recall@10 ≥ 70%");
    println!("  - Cross-NS  recall@10 ≥ 60%");
    println!("  - Single-NS mean latency ≤ 5 ms");
    println!();

    let all_pass = results.iter().all(|r| r.accept);
    let any_pass = results.iter().any(|r| r.accept);

    for r in &results {
        println!(
            "  {:20} single_recall={:.0}%  cross_recall={:.0}%  lat={:.0}µs  → {}",
            r.name,
            r.single_recall * 100.0,
            r.cross_recall * 100.0,
            r.single_ns_mean_us,
            if r.accept { "PASS" } else { "FAIL" }
        );
    }

    println!();
    if all_pass {
        println!("ACCEPTANCE: ALL VARIANTS PASS");
    } else if any_pass {
        println!("ACCEPTANCE: PARTIAL — at least one variant passes");
    } else {
        println!("ACCEPTANCE: FAIL — no variant meets thresholds");
        std::process::exit(1);
    }
}
