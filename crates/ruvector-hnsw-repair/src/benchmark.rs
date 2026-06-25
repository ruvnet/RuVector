//! Benchmark for HNSW deletion strategies.
//!
//! Measures recall@10 and latency for three deletion strategies
//! after removing 20% of vectors from a random dataset.

use ruvector_hnsw_repair::{
    brute_force_knn_live,
    graph::{HnswConfig, HnswGraph},
    l2_sq, recall_at_k,
    strategy::{BatchRepair, DeletionStrategy, EagerRepair, TombstoneOnly},
};
use std::time::{Duration, Instant};

fn main() {
    print_header();

    let n: usize = 5_000;
    let dim: usize = 64;
    let n_queries: usize = 100;
    let k: usize = 10;
    let ef_search: usize = 50;
    let delete_frac = 0.20;
    let n_delete = (n as f64 * delete_frac) as usize;

    println!("Dataset        : {} vectors, {} dimensions", n, dim);
    println!("Queries        : {}", n_queries);
    println!("k (recall@k)   : {}", k);
    println!("ef_search      : {}", ef_search);
    println!(
        "Deletion count : {} ({:.0}%)",
        n_delete,
        delete_frac * 100.0
    );
    println!();

    // --- Build index (shared baseline) ---
    let (graph, queries, delete_ids) = build_dataset(n, dim, n_queries, n_delete);
    let baseline_recall = recall_at_k(&graph, &queries, k, ef_search);
    println!(
        "Baseline recall@{} (before deletions): {:.4}",
        k, baseline_recall
    );
    println!();

    // -------------------------------------------------------------------------
    // Strategy 1: TombstoneOnly
    // -------------------------------------------------------------------------
    let (stats_ts, r_ts) = run_strategy(
        "TombstoneOnly",
        &graph,
        &queries,
        &delete_ids,
        k,
        ef_search,
        |g, ids| {
            let s = TombstoneOnly;
            for &id in ids {
                s.delete(g, id);
            }
        },
        |g| {},
    );
    print_stats("TombstoneOnly", &stats_ts, baseline_recall, r_ts);

    // -------------------------------------------------------------------------
    // Strategy 2: BatchRepair (batch_size = 50)
    // -------------------------------------------------------------------------
    let batch_size = 50;
    let (stats_br, r_br) = run_strategy(
        "BatchRepair(50)",
        &graph,
        &queries,
        &delete_ids,
        k,
        ef_search,
        |g, ids| {
            let s = BatchRepair::new(batch_size);
            for &id in ids {
                s.delete(g, id);
            }
            s.flush(g);
        },
        |_g| {},
    );
    print_stats("BatchRepair(50)", &stats_br, baseline_recall, r_br);

    // -------------------------------------------------------------------------
    // Strategy 3: EagerRepair
    // -------------------------------------------------------------------------
    let (stats_er, r_er) = run_strategy(
        "EagerRepair",
        &graph,
        &queries,
        &delete_ids,
        k,
        ef_search,
        |g, ids| {
            let s = EagerRepair;
            for &id in ids {
                s.delete(g, id);
            }
        },
        |_g| {},
    );
    print_stats("EagerRepair", &stats_er, baseline_recall, r_er);

    // -------------------------------------------------------------------------
    // Summary table
    // -------------------------------------------------------------------------
    println!();
    println!("{:-<88}", "");
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Variant", "Delete(ms)", "Search μs", "p50 μs", "p95 μs", "Recall@10", "Pass?"
    );
    println!("{:-<88}", "");
    for (name, stats, recall) in [
        ("TombstoneOnly", &stats_ts, r_ts),
        ("BatchRepair(50)", &stats_br, r_br),
        ("EagerRepair", &stats_er, r_er),
    ] {
        let pass = if recall >= baseline_recall * 0.75 {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "{:<18} {:>10.2} {:>10.1} {:>10.1} {:>10.1} {:>10.4} {:>8}",
            name, stats.delete_ms, stats.mean_search_us, stats.p50_us, stats.p95_us, recall, pass
        );
    }
    println!("{:-<88}", "");
    println!();

    // Acceptance test: at least one strategy maintains recall >= 75% of baseline.
    let best_recall = r_ts.max(r_br).max(r_er);
    let threshold = baseline_recall * 0.75;
    if best_recall >= threshold {
        println!(
            "ACCEPTANCE: PASS — best recall {:.4} >= threshold {:.4}",
            best_recall, threshold
        );
    } else {
        eprintln!(
            "ACCEPTANCE: FAIL — best recall {:.4} < threshold {:.4}",
            best_recall, threshold
        );
        std::process::exit(1);
    }

    // Print memory estimates.
    let vec_bytes = n * dim * 4;
    let edge_bytes = n * 32 * 4; // avg 32 edges, 4 bytes each
    println!();
    println!("Memory estimates (approximate):");
    println!(
        "  Vectors  : {:>8} KB  ({} vecs × {} dim × 4 B)",
        vec_bytes / 1024,
        n,
        dim
    );
    println!(
        "  Edges    : {:>8} KB  ({} vecs × 32 avg edges × 4 B)",
        edge_bytes / 1024,
        n
    );
    println!("  Total    : {:>8} KB", (vec_bytes + edge_bytes) / 1024);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct BenchStats {
    delete_ms: f64,
    mean_search_us: f64,
    p50_us: f64,
    p95_us: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_strategy<D, P>(
    _name: &str,
    base: &HnswGraph,
    queries: &[Vec<f32>],
    delete_ids: &[usize],
    k: usize,
    ef_search: usize,
    do_deletes: D,
    _post: P,
) -> (BenchStats, f32)
where
    D: Fn(&mut HnswGraph, &[usize]),
    P: Fn(&mut HnswGraph),
{
    // Clone the graph so each strategy starts fresh.
    let mut g = clone_graph(base);

    // --- Delete phase ---
    let t0 = Instant::now();
    do_deletes(&mut g, delete_ids);
    let delete_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // --- Search phase ---
    let mut latencies = Vec::with_capacity(queries.len());
    for q in queries {
        let ts = Instant::now();
        let _ = g.search(q, k, ef_search);
        latencies.push(ts.elapsed());
    }

    let mean_us = mean_dur(&latencies) * 1e6;
    let p50_us = percentile_dur(&mut latencies.clone(), 50) * 1e6;
    let p95_us = percentile_dur(&mut latencies.clone(), 95) * 1e6;

    let recall = recall_at_k(&g, queries, k, ef_search);

    (
        BenchStats {
            delete_ms,
            mean_search_us: mean_us,
            p50_us,
            p95_us,
        },
        recall,
    )
}

fn print_stats(name: &str, s: &BenchStats, baseline: f32, recall: f32) {
    println!(
        "{}: delete={:.2}ms  search_mean={:.1}µs  p50={:.1}µs  p95={:.1}µs  recall@10={:.4}  degradation={:+.4}",
        name,
        s.delete_ms,
        s.mean_search_us,
        s.p50_us,
        s.p95_us,
        recall,
        recall - baseline
    );
}

fn build_dataset(
    n: usize,
    dim: usize,
    n_queries: usize,
    n_delete: usize,
) -> (HnswGraph, Vec<Vec<f32>>, Vec<usize>) {
    let config = HnswConfig {
        dim,
        m: 16,
        m0: 32,
        ef_construction: 100,
        ml: 1.0 / (16f64.ln()),
    };
    let mut g = HnswGraph::new(config);
    let mut rng = 0xABCD_1234_EF56_7890u64;

    for _ in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rand_f32(&mut rng)).collect();
        g.insert(v);
    }

    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rand_f32(&mut rng)).collect())
        .collect();

    // Delete the first n_delete nodes (deterministic, evenly spread).
    let step = n / n_delete;
    let delete_ids: Vec<usize> = (0..n_delete).map(|i| i * step).collect();

    (g, queries, delete_ids)
}

fn clone_graph(src: &HnswGraph) -> HnswGraph {
    let config = src.config.clone();
    let mut g = HnswGraph::new(config);
    g.vectors = src.vectors.clone();
    g.deleted = src.deleted.clone();
    g.node_level = src.node_level.clone();
    g.layers = src.layers.clone();
    g.entry = src.entry;
    g
}

fn rand_f32(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 33) as f32 / (u32::MAX as f32)
}

fn mean_dur(v: &[Duration]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().map(|d| d.as_secs_f64()).sum::<f64>() / v.len() as f64
}

fn percentile_dur(v: &mut [Duration], p: usize) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable();
    let idx = (p * v.len() / 100).min(v.len() - 1);
    v[idx].as_secs_f64()
}

fn print_header() {
    println!("==========================================================");
    println!(" ruvector-hnsw-repair  —  Deletion Strategy Benchmark");
    println!("==========================================================");
    println!("OS             : {}", std::env::consts::OS);
    println!("Arch           : {}", std::env::consts::ARCH);
    println!();
}
