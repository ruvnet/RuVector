//! Benchmark: local-min-cut-guided HNSW deletion repair vs. the three
//! baseline strategies from `ruvector-hnsw-repair`.
//!
//! Same dataset generation and dataset size as
//! `ruvector-hnsw-repair`'s own benchmark (5,000 vectors, dim 64, 20%
//! deletion) so the baselines here are not re-tuned in this crate's favour.
//!
//! Acceptance thresholds are fixed before the run (see `main`):
//! 1. MincutGuided recall@10 within 1.0pp (absolute) of EagerRepair.
//! 2. MincutGuided total repaired-edge count <= 60% of EagerRepair's.
//! 3. MincutGuided bookkeeping (find_cut + shadow sync) overhead
//!    <= 25% of its own total delete wall-clock time.
//!
//! `n_delete` is intentionally far smaller than the 20% used by
//! `ruvector-hnsw-repair`'s own benchmark. An initial run at 1,000
//! deletions (20%) did not complete the `LocalCutGuidedRepair` phase in
//! 10+ minutes and was killed; a follow-up run at this dataset's real
//! construction density (m0=32, matching `ruvector-hnsw-repair`'s own
//! benchmark config) measured a **single** `find_cut` call at 163.5
//! seconds (see the nightly research report for the exact run). That one
//! measurement already conclusively fails acceptance criterion 3 by
//! roughly six orders of magnitude, so `n_delete` here is reduced only so
//! the comparison finishes in reproducible, bounded time — not to
//! flatter the result. Extrapolating the 163.5s/call figure, 1,000
//! deletions would take on the order of 45 hours.

use ruvector_hnsw_repair::{
    graph::{HnswConfig, HnswGraph},
    recall_at_k,
    strategy::{BatchRepair, DeletionStrategy, EagerRepair, TombstoneOnly},
};
use ruvector_mincut_repair::LocalCutGuidedRepair;
use std::time::{Duration, Instant};

fn main() {
    print_header();

    let n: usize = 5_000;
    let dim: usize = 64;
    let n_queries: usize = 100;
    let k: usize = 10;
    let ef_search: usize = 50;
    let n_delete: usize = 3; // see module doc: reduced from 20% for tractability
    let delete_frac = n_delete as f64 / n as f64;
    let cut_k = 2usize; // local-cut size bound for LocalCutGuidedRepair

    println!("Dataset        : {n} vectors, {dim} dimensions");
    println!("Queries        : {n_queries}");
    println!("k (recall@k)   : {k}");
    println!("ef_search      : {ef_search}");
    println!("Deletion count : {n_delete} ({:.0}%)", delete_frac * 100.0);
    println!("cut_k          : {cut_k} (LocalCutGuidedRepair local-cut bound)");
    println!();

    let (graph, queries, delete_ids) = build_dataset(n, dim, n_queries, n_delete);
    let baseline_recall = recall_at_k(&graph, &queries, k, ef_search);
    println!("Baseline recall@{k} (before deletions): {baseline_recall:.4}");
    println!();

    let (stats_ts, r_ts) = run_baseline(
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
    );
    print_stats("TombstoneOnly", &stats_ts, baseline_recall, r_ts);

    let batch_size = 50;
    let (stats_br, r_br) = run_baseline(
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
    );
    print_stats("BatchRepair(50)", &stats_br, baseline_recall, r_br);

    let (stats_er, r_er, er_total_edges) = run_eager(&graph, &queries, &delete_ids, k, ef_search);
    print_stats("EagerRepair", &stats_er, baseline_recall, r_er);

    let (stats_mc, r_mc, mc_total_edges, mc_fragile, mc_safe, mc_bookkeeping_ms) =
        run_mincut_guided(&graph, &queries, &delete_ids, k, ef_search, cut_k);
    print_stats("LocalCutGuided", &stats_mc, baseline_recall, r_mc);

    println!();
    println!(
        "LocalCutGuided detail: fragile={mc_fragile} safe={mc_safe} \
         (fragile fraction {:.1}%), bookkeeping={mc_bookkeeping_ms:.2}ms \
         ({:.1}% of delete wall-clock), repaired_edges={mc_total_edges} \
         (EagerRepair repaired_edges={er_total_edges})",
        100.0 * mc_fragile as f64 / n_delete as f64,
        100.0 * mc_bookkeeping_ms / stats_mc.delete_ms,
    );

    println!();
    println!("{:-<96}", "");
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>8}",
        "Variant",
        "Delete(ms)",
        "Search μs",
        "p50 μs",
        "p95 μs",
        "Recall@10",
        "RepairEdges",
        "Pass?"
    );
    println!("{:-<96}", "");
    for (name, stats, recall, edges) in [
        ("TombstoneOnly", &stats_ts, r_ts, 0usize),
        ("BatchRepair(50)", &stats_br, r_br, 0usize),
        ("EagerRepair", &stats_er, r_er, er_total_edges),
        ("LocalCutGuided", &stats_mc, r_mc, mc_total_edges),
    ] {
        let pass = if recall >= baseline_recall * 0.75 {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "{:<18} {:>10.2} {:>10.1} {:>10.1} {:>10.1} {:>10.4} {:>12} {:>8}",
            name,
            stats.delete_ms,
            stats.mean_search_us,
            stats.p50_us,
            stats.p95_us,
            recall,
            edges,
            pass
        );
    }
    println!("{:-<96}", "");
    println!();

    // --- Pre-registered acceptance thresholds ---
    let recall_gap_pp = (r_er - r_mc) * 100.0;
    let recall_ok = recall_gap_pp <= 1.0;
    let edge_ratio = mc_total_edges as f64 / er_total_edges.max(1) as f64;
    let edge_ok = edge_ratio <= 0.60;
    let bookkeeping_frac = mc_bookkeeping_ms / stats_mc.delete_ms;
    let overhead_ok = bookkeeping_frac <= 0.25;

    println!("Acceptance criteria (fixed before this run):");
    println!(
        "  1. recall gap (Eager - MincutGuided) <= 1.0pp     : {recall_gap_pp:+.2}pp  [{}]",
        if recall_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  2. repaired_edges ratio (Mincut/Eager) <= 0.60    : {edge_ratio:.3}  [{}]",
        if edge_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  3. bookkeeping overhead fraction <= 0.25          : {bookkeeping_frac:.3}  [{}]",
        if overhead_ok { "PASS" } else { "FAIL" }
    );
    println!();

    if recall_ok && edge_ok && overhead_ok {
        println!("ACCEPTANCE: ACCEPT — all three criteria met.");
    } else {
        eprintln!("ACCEPTANCE: REJECT — at least one criterion failed.");
        std::process::exit(1);
    }
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

fn run_baseline<D>(
    _name: &str,
    base: &HnswGraph,
    queries: &[Vec<f32>],
    delete_ids: &[usize],
    k: usize,
    ef_search: usize,
    do_deletes: D,
) -> (BenchStats, f32)
where
    D: Fn(&mut HnswGraph, &[usize]),
{
    let mut g = clone_graph(base);
    let t0 = Instant::now();
    do_deletes(&mut g, delete_ids);
    let delete_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (stats, recall) = measure_search(&g, queries, k, ef_search, delete_ms);
    (stats, recall)
}

fn run_eager(
    base: &HnswGraph,
    queries: &[Vec<f32>],
    delete_ids: &[usize],
    k: usize,
    ef_search: usize,
) -> (BenchStats, f32, usize) {
    let mut g = clone_graph(base);
    let s = EagerRepair;
    let t0 = Instant::now();
    let mut total_edges = 0usize;
    for &id in delete_ids {
        total_edges += s.delete(&mut g, id).repaired_edges;
    }
    let delete_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (stats, recall) = measure_search(&g, queries, k, ef_search, delete_ms);
    (stats, recall, total_edges)
}

#[allow(clippy::too_many_arguments)]
fn run_mincut_guided(
    base: &HnswGraph,
    queries: &[Vec<f32>],
    delete_ids: &[usize],
    k: usize,
    ef_search: usize,
    cut_k: usize,
) -> (BenchStats, f32, usize, usize, usize, f64) {
    let mut g = clone_graph(base);
    let strat = LocalCutGuidedRepair::new(&g, cut_k);
    let t0 = Instant::now();
    let mut total_edges = 0usize;
    for &id in delete_ids {
        let call_t0 = Instant::now();
        let (result, fragile) = strat.delete_with_diagnostics(&mut g, id);
        eprintln!(
            "  delete({id}): fragile={fragile} repaired_edges={} took={:?}",
            result.repaired_edges,
            call_t0.elapsed()
        );
        total_edges += result.repaired_edges;
    }
    let delete_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let s = strat.stats();
    let bookkeeping_ms = s.bookkeeping_ns as f64 / 1e6;
    let (stats, recall) = measure_search(&g, queries, k, ef_search, delete_ms);
    (
        stats,
        recall,
        total_edges,
        s.fragile_count,
        s.safe_count,
        bookkeeping_ms,
    )
}

fn measure_search(
    g: &HnswGraph,
    queries: &[Vec<f32>],
    k: usize,
    ef_search: usize,
    delete_ms: f64,
) -> (BenchStats, f32) {
    let mut latencies = Vec::with_capacity(queries.len());
    for q in queries {
        let ts = Instant::now();
        let _ = g.search(q, k, ef_search);
        latencies.push(ts.elapsed());
    }
    let mean_us = mean_dur(&latencies) * 1e6;
    let p50_us = percentile_dur(&mut latencies.clone(), 50) * 1e6;
    let p95_us = percentile_dur(&mut latencies.clone(), 95) * 1e6;
    let recall = recall_at_k(g, queries, k, ef_search);
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
        "{name}: delete={:.2}ms  search_mean={:.1}µs  p50={:.1}µs  p95={:.1}µs  recall@10={:.4}  degradation={:+.4}",
        s.delete_ms, s.mean_search_us, s.p50_us, s.p95_us, recall, recall - baseline
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
    println!(" ruvector-mincut-repair  —  Local-Cut-Guided Repair Benchmark");
    println!("==========================================================");
    println!("OS             : {}", std::env::consts::OS);
    println!("Arch           : {}", std::env::consts::ARCH);
    println!();
}
