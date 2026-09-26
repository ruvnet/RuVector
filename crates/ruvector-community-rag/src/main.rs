//! Community-RAG Benchmark Binary
//!
//! Two experiments on synthetic agent-memory datasets (Gaussian task clusters):
//!
//! Exp A — Tight clusters (σ=0.40): all variants have high recall; CommunityRAG
//!          is dramatically faster due to community-scoped scan.
//!
//! Exp B — Overlapping clusters (σ=1.20): FlatScan loses community precision;
//!          CommunityRAG preserves it by routing to the nearest community centroid.
//!
//! Usage: cargo run --release --manifest-path crates/ruvector-community-rag/Cargo.toml

use std::time::Instant;
use ruvector_community_rag::{
    community_rag::CommunityRAG,
    flat_scan::FlatScan,
    graph_hop::GraphHop,
    generate_clustered_dataset,
    recall_at_k,
    community_precision,
    CommunitySearch,
    VectorMeta,
    Hit,
};

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    println!("=== Community-RAG Benchmark ===");
    println!("Platform : {}", std::env::consts::OS);
    println!("Arch     : {}", std::env::consts::ARCH);
    println!();

    let (res_a, flat_a) = experiment("A (tight,  σ=0.40)", 2000, 64, 10, 0.40, 10, 200, 42, 0.80);
    let (res_b, flat_b) = experiment("B (overlap, σ=1.20)", 2000, 64, 10, 1.20, 10, 200, 42, 0.60);

    println!("=== Acceptance Tests ===");
    let mut pass = true;
    pass &= accept("Exp A  FlatScan recall@10 == 1.000",     res_a[0].recall >= 0.999);
    pass &= accept("Exp A  GraphHop recall@10 >= 0.80",      res_a[1].recall >= 0.80);
    pass &= accept("Exp A  CommunityRAG community_prec >= 0.80", res_a[2].comm_prec >= 0.80);
    pass &= accept("Exp A  CommunityRAG ≥3× faster than FlatScan",
        res_a[2].mean_lat_us * 3.0 < res_a[0].mean_lat_us);
    pass &= accept("Exp B  CommunityRAG community_prec >= FlatScan on overlapping clusters",
        res_b[2].comm_prec >= res_b[0].comm_prec);
    pass &= accept("Exp A  GraphHop memory < 4× FlatScan",
        res_a[1].mem_kb < res_a[0].mem_kb * 4);

    // Suppress unused variable warnings for flat_a, flat_b.
    let _ = flat_a;
    let _ = flat_b;

    if pass {
        println!("\nRESULT: PASS — all acceptance tests met.");
    } else {
        eprintln!("\nRESULT: FAIL");
        std::process::exit(1);
    }
}

// ── Experiment ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn experiment(
    label: &str,
    n: usize,
    dims: usize,
    k_comm: usize,
    sigma: f32,
    k: usize,
    n_queries: usize,
    seed: u64,
    threshold: f32,
) -> (Vec<BenchRow>, FlatScan) {
    println!("── Experiment {label} ──────────────────────────────────");
    println!("N={n} D={dims} K_communities={k_comm} σ={sigma:.2} threshold={threshold:.2}");

    let (corpus, labels) = generate_clustered_dataset(n, dims, k_comm, sigma, seed);
    let q_start = n.saturating_sub(n_queries);

    // Build FlatScan (oracle + baseline).
    let t = Instant::now();
    let mut flat = FlatScan::new();
    for (v, l) in corpus.iter().zip(labels.iter()) { flat.insert(v, *l); }
    flat.build();
    eprintln!("[build] FlatScan      {:>4}ms", t.elapsed().as_millis());

    // Pre-compute ground truth for every query.
    let queries: Vec<(&[f32], usize)> = corpus[q_start..]
        .iter()
        .zip(labels[q_start..].iter())
        .map(|(v, &l)| (v.as_slice(), l))
        .collect();
    let oracles: Vec<Vec<Hit>> = queries.iter().map(|(qv, _)| flat.search(qv, k)).collect();
    let metas: &[VectorMeta] = flat.metas();

    // Build GraphHop.
    let t = Instant::now();
    let mut hop = GraphHop::new(6, 40);
    for (v, l) in corpus.iter().zip(labels.iter()) { hop.insert(v, *l); }
    hop.build();
    eprintln!("[build] GraphHop      {:>4}ms", t.elapsed().as_millis());

    // Build CommunityRAG.
    let t = Instant::now();
    let mut rag = CommunityRAG::new(threshold);
    for (v, l) in corpus.iter().zip(labels.iter()) { rag.insert(v, *l); }
    rag.build();
    eprintln!("[build] CommunityRAG  {:>4}ms  ({} communities)", t.elapsed().as_millis(), rag.num_communities());

    // Benchmark each variant.
    let flat_row = bench(&flat,  metas, &queries, &oracles, k);
    let hop_row  = bench(&hop,   metas, &queries, &oracles, k);
    let rag_row  = bench(&rag,   metas, &queries, &oracles, k);

    let rows = vec![flat_row, hop_row, rag_row];
    print_table(&rows);
    (rows, flat)
}

// ── Benchmarking ─────────────────────────────────────────────────────────────

fn bench(
    idx: &dyn CommunitySearch,
    metas: &[VectorMeta],
    queries: &[(&[f32], usize)],
    oracles: &[Vec<Hit>],
    k: usize,
) -> BenchRow {
    let mut lats_ns: Vec<f64> = Vec::with_capacity(queries.len());
    let mut recalls: Vec<f32> = Vec::with_capacity(queries.len());
    let mut comm_precs: Vec<f32> = Vec::with_capacity(queries.len());

    for ((qvec, qtrue), oracle) in queries.iter().zip(oracles.iter()) {
        let t = Instant::now();
        let res = idx.search(qvec, k);
        lats_ns.push(t.elapsed().as_nanos() as f64);
        recalls.push(recall_at_k(&res, oracle, k));
        comm_precs.push(community_precision(&res, metas, *qtrue, k));
    }

    lats_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = lats_ns.iter().sum::<f64>() / lats_ns.len() as f64;
    let p50 = lats_ns[lats_ns.len() / 2];
    let p95 = lats_ns[(lats_ns.len() as f64 * 0.95) as usize];
    let mean_rec = recalls.iter().sum::<f32>() / recalls.len() as f32;
    let mean_comm = comm_precs.iter().sum::<f32>() / comm_precs.len() as f32;

    BenchRow {
        name: idx.name().to_string(),
        mean_lat_us: mean / 1000.0,
        p50_us: p50 / 1000.0,
        p95_us: p95 / 1000.0,
        qps: 1e9 / mean,
        recall: mean_rec,
        comm_prec: mean_comm,
        mem_kb: idx.memory_bytes() / 1024,
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct BenchRow {
    name: String,
    mean_lat_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    recall: f32,
    comm_prec: f32,
    mem_kb: usize,
}

fn print_table(rows: &[BenchRow]) {
    println!("{:<16} {:>11} {:>10} {:>10} {:>11} {:>10} {:>12} {:>9}",
        "Variant", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Recall@10", "CommPrec@10", "Mem(KB)");
    println!("{}", "─".repeat(98));
    for r in rows {
        println!("{:<16} {:>11.2} {:>10.2} {:>10.2} {:>11.0} {:>10.3} {:>12.3} {:>9}",
            r.name, r.mean_lat_us, r.p50_us, r.p95_us, r.qps,
            r.recall, r.comm_prec, r.mem_kb);
    }
    println!();
}

fn accept(label: &str, cond: bool) -> bool {
    println!("  [{}] {label}", if cond { "PASS" } else { "FAIL" });
    cond
}
