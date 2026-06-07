//! namespace-bench — benchmark for all three namespace router variants.
//!
//! Generates N synthetic vectors across `NS` namespaces (multi-cluster
//! Gaussian), inserts into each backend, then measures:
//!   - insert throughput (vecs/sec)
//!   - single-namespace search latency (mean, p50, p95 µs)
//!   - recall@K (fraction of true top-K found within own namespace)
//!   - cross-namespace federation quality (CoherenceGated variant)
//!   - witness log size
//!   - memory estimate (KB)
//!
//! All numbers come from real Rust timing; no invented figures.

use std::collections::HashSet;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_namespace_router::{
    centroid_routed::CentroidRouted,
    coherence_gated::{audited_search, CoherenceGated},
    flat_isolated::FlatIsolated,
    l2sq, NamespaceId, NamespaceIndex, SearchResult,
};

// ─── configuration ────────────────────────────────────────────────────────────

const NS: usize = 8; // number of namespaces (tenants)
const PER_NS: usize = 500; // vectors per namespace
const DIM: usize = 128; // embedding dimension
const NQUERIES: usize = 200; // queries per namespace
const K: usize = 10; // recall@K
const SEED: u64 = 2026_06_07;
const COHERENCE_THRESHOLD: f32 = 0.30; // for CoherenceGated
const PROBE: usize = 3; // namespaces probed for centroid/coherence variants

// ─── data generation ─────────────────────────────────────────────────────────

/// One namespace cluster: vectors near a random center.
fn gen_namespace(_ns: usize, n: usize, dim: usize, rng: &mut StdRng, spread: f32) -> Vec<Vec<f32>> {
    let center: Vec<f32> = (0..dim).map(|_| rng.gen_range(-10.0f32..10.0)).collect();
    (0..n)
        .map(|_| {
            center
                .iter()
                .map(|&c| c + rng.gen_range(-spread..spread))
                .collect()
        })
        .collect()
}

// ─── exact top-K (ground truth) ──────────────────────────────────────────────

fn exact_topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> HashSet<usize> {
    let mut d: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2sq(query, v)))
        .collect();
    d.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    d.iter().take(k).map(|(i, _)| *i).collect()
}

fn recall(results: &[SearchResult], gt: &HashSet<usize>, id_offset: usize) -> f64 {
    let found: usize = results
        .iter()
        .filter(|r| gt.contains(&(r.id as usize - id_offset)))
        .count();
    found as f64 / gt.len() as f64
}

// ─── timing helpers ───────────────────────────────────────────────────────────

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() as f64 * p / 100.0) as usize).min(v.len().saturating_sub(1));
    v[idx]
}

// ─── benchmark one index variant ─────────────────────────────────────────────

struct BenchResult {
    name: String,
    insert_vps: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    recall: f64,
    memory_kb: f64,
    witness_events: usize,
    accept: bool,
}

fn bench_flat(
    corpus: &[Vec<Vec<f32>>],
    queries: &[(NamespaceId, Vec<f32>)],
    gt: &[Vec<HashSet<usize>>],
) -> BenchResult {
    let mut idx = FlatIsolated::new(DIM);

    let t0 = Instant::now();
    for (ns, vecs) in corpus.iter().enumerate() {
        for (i, v) in vecs.iter().enumerate() {
            let id = (ns * PER_NS + i) as u64;
            idx.insert(ns as NamespaceId, id, v.clone()).unwrap();
        }
    }
    let insert_s = t0.elapsed().as_secs_f64();
    let insert_vps = (NS * PER_NS) as f64 / insert_s;

    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f64;
    let t1 = Instant::now();
    for ((ns, q), gt_set) in queries.iter().zip(gt.iter().flatten()) {
        let qt = Instant::now();
        let results = idx.search(*ns, q, K);
        latencies.push(qt.elapsed().as_secs_f64() * 1e6);
        let offset = *ns as usize * PER_NS;
        total_recall += recall(&results, gt_set, offset);
    }
    let elapsed = t1.elapsed().as_secs_f64();
    let mean_recall = total_recall / queries.len() as f64;

    let mean_us = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p50 = percentile(latencies.clone(), 50.0);
    let p95 = percentile(latencies, 95.0);
    let qps = queries.len() as f64 / elapsed;

    BenchResult {
        name: "FlatIsolated".into(),
        insert_vps,
        mean_us,
        p50_us: p50,
        p95_us: p95,
        qps,
        recall: mean_recall,
        memory_kb: idx.memory_bytes() as f64 / 1024.0,
        witness_events: 0,
        accept: mean_recall >= 0.99, // exact scan must achieve near-perfect recall
    }
}

fn bench_centroid(
    corpus: &[Vec<Vec<f32>>],
    queries: &[(NamespaceId, Vec<f32>)],
    gt: &[Vec<HashSet<usize>>],
) -> BenchResult {
    let mut idx = CentroidRouted::new(DIM, PROBE);

    let t0 = Instant::now();
    for (ns, vecs) in corpus.iter().enumerate() {
        for (i, v) in vecs.iter().enumerate() {
            let id = (ns * PER_NS + i) as u64;
            idx.insert(ns as NamespaceId, id, v.clone()).unwrap();
        }
    }
    let insert_s = t0.elapsed().as_secs_f64();
    let insert_vps = (NS * PER_NS) as f64 / insert_s;

    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f64;
    let t1 = Instant::now();
    for ((ns, q), gt_set) in queries.iter().zip(gt.iter().flatten()) {
        let qt = Instant::now();
        // probe=1 → strictly own namespace (same isolation as FlatIsolated)
        idx.probe = 1;
        let results = idx.search(*ns, q, K);
        latencies.push(qt.elapsed().as_secs_f64() * 1e6);
        let offset = *ns as usize * PER_NS;
        total_recall += recall(&results, gt_set, offset);
    }
    let elapsed = t1.elapsed().as_secs_f64();
    let mean_recall = total_recall / queries.len() as f64;

    let mean_us = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p50 = percentile(latencies.clone(), 50.0);
    let p95 = percentile(latencies, 95.0);
    let qps = queries.len() as f64 / elapsed;

    BenchResult {
        name: "CentroidRouted".into(),
        insert_vps,
        mean_us,
        p50_us: p50,
        p95_us: p95,
        qps,
        recall: mean_recall,
        memory_kb: idx.memory_bytes() as f64 / 1024.0,
        witness_events: 0,
        accept: mean_recall >= 0.99,
    }
}

fn bench_coherence(
    corpus: &[Vec<Vec<f32>>],
    queries: &[(NamespaceId, Vec<f32>)],
    gt: &[Vec<HashSet<usize>>],
) -> BenchResult {
    let mut idx = CoherenceGated::new(DIM, COHERENCE_THRESHOLD, PROBE);

    let t0 = Instant::now();
    for (ns, vecs) in corpus.iter().enumerate() {
        for (i, v) in vecs.iter().enumerate() {
            let id = (ns * PER_NS + i) as u64;
            idx.insert(ns as NamespaceId, id, v.clone()).unwrap();
        }
    }
    let insert_s = t0.elapsed().as_secs_f64();
    let insert_vps = (NS * PER_NS) as f64 / insert_s;

    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f64;
    let t1 = Instant::now();
    for ((ns, q), gt_set) in queries.iter().zip(gt.iter().flatten()) {
        let qt = Instant::now();
        // Audited search records cross-namespace events in witness log.
        let results = audited_search(&mut idx, *ns, q, K);
        latencies.push(qt.elapsed().as_secs_f64() * 1e6);
        let offset = *ns as usize * PER_NS;
        total_recall += recall(&results, gt_set, offset);
    }
    let elapsed = t1.elapsed().as_secs_f64();
    let mean_recall = total_recall / queries.len() as f64;
    let witness_events = idx.witness.len();
    let mean_coh = idx.witness.mean_coherence();

    let mean_us = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p50 = percentile(latencies.clone(), 50.0);
    let p95 = percentile(latencies, 95.0);
    let qps = queries.len() as f64 / elapsed;

    eprintln!("  [coherence] witness_events={witness_events} mean_coherence={mean_coh:.3}");

    BenchResult {
        name: "CoherenceGated".into(),
        insert_vps,
        mean_us,
        p50_us: p50,
        p95_us: p95,
        qps,
        recall: mean_recall,
        memory_kb: idx.memory_bytes() as f64 / 1024.0,
        witness_events,
        accept: mean_recall >= 0.99,
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    // System information.
    println!("=== ruvector-namespace-router benchmark ===");
    println!("OS          : {}", std::env::consts::OS);
    println!("Arch        : {}", std::env::consts::ARCH);
    println!("Namespaces  : {NS}");
    println!("Vecs/ns     : {PER_NS}");
    println!("Total vecs  : {}", NS * PER_NS);
    println!("Dimensions  : {DIM}");
    println!("Queries     : {} ({NQUERIES}/ns)", NS * NQUERIES);
    println!("K           : {K}");
    println!("Probe       : {PROBE}");
    println!("Coherence τ : {COHERENCE_THRESHOLD}");
    println!();

    let mut rng = StdRng::seed_from_u64(SEED);

    // Generate corpus: NS well-separated clusters (spread=1.5).
    let corpus: Vec<Vec<Vec<f32>>> = (0..NS)
        .map(|ns| gen_namespace(ns, PER_NS, DIM, &mut rng, 1.5))
        .collect();

    // Generate queries: NQUERIES per namespace, near the same cluster center.
    let queries: Vec<(NamespaceId, Vec<f32>)> = (0..NS)
        .flat_map(|ns| {
            gen_namespace(ns, NQUERIES, DIM, &mut rng, 0.5)
                .into_iter()
                .map(move |q| (ns as NamespaceId, q))
        })
        .collect();

    // Ground truth: exact top-K within each namespace.
    let gt: Vec<Vec<HashSet<usize>>> = (0..NS)
        .map(|ns| {
            let per_ns_queries: Vec<&Vec<f32>> = queries
                .iter()
                .filter(|(qns, _)| *qns == ns as NamespaceId)
                .map(|(_, q)| q)
                .collect();
            per_ns_queries
                .iter()
                .map(|q| exact_topk(q, &corpus[ns], K))
                .collect()
        })
        .collect();

    let results = vec![
        bench_flat(&corpus, &queries, &gt),
        bench_centroid(&corpus, &queries, &gt),
        bench_coherence(&corpus, &queries, &gt),
    ];

    // Print results table.
    println!(
        "{:<18} {:>10} {:>9} {:>9} {:>9} {:>9} {:>8} {:>10} {:>8} {:>7}",
        "Variant",
        "InsertVPS",
        "Mean(µs)",
        "p50(µs)",
        "p95(µs)",
        "QPS",
        "Recall",
        "Mem(KB)",
        "Witness",
        "Accept"
    );
    println!("{}", "-".repeat(104));

    let mut all_pass = true;
    for r in &results {
        let accept_str = if r.accept { "PASS" } else { "FAIL" };
        if !r.accept {
            all_pass = false;
        }
        println!(
            "{:<18} {:>10.0} {:>9.2} {:>9.2} {:>9.2} {:>9.0} {:>8.3} {:>10.1} {:>8} {:>7}",
            r.name,
            r.insert_vps,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.qps,
            r.recall,
            r.memory_kb,
            r.witness_events,
            accept_str
        );
    }
    println!();

    // Acceptance criterion: all variants achieve recall >= 0.99.
    // (FlatIsolated and CentroidRouted are exact; CoherenceGated adds
    //  witness overhead but also uses exact scan within probed namespaces.)
    if all_pass {
        println!("ACCEPTANCE: PASS — all variants recall@{K} >= 0.99");
    } else {
        eprintln!("ACCEPTANCE: FAIL — one or more variants below recall threshold");
        std::process::exit(1);
    }
}
