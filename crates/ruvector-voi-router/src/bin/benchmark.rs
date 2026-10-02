//! VoI-Gated Coherence Routing — benchmark binary.
//!
//! Compares three search policies on the same clustered flat k-NN proximity
//! graph used by `ruvector-coherence-hnsw`'s own (accepted) benchmark:
//!
//! 1. **Baseline**       — always plain beam search.
//! 2. **AlwaysGated**    — always coherence-gated beam search (the currently
//!    accepted default from the 2026-06-16 nightly).
//! 3. **VoiRouted**      — per query, routes to Baseline or AlwaysGated based
//!    on the free entry→query distance signal, with the distance threshold
//!    picked by a small bounded search (3 generations × 4 candidates, one
//!    promotion) run *only* on a calibration query set.
//!
//! The calibration query set and the evaluation query set are generated from
//! different seeds so the threshold search never sees the queries it is
//! later judged against.
//!
//! ## Usage
//!
//!   cargo run --release -p ruvector-voi-router --bin benchmark

use std::time::Instant;

use ruvector_coherence_hnsw::{
    dataset::{clustered_queries, clustered_unit_vectors, ground_truth},
    graph::{l2_sq, FlatGraph, GraphConfig},
    metrics::{memory_estimate_bytes, recall_at_k, LatencyStats},
    search::{BaselineSearch, CoherenceGatedSearch, Searcher},
};
use ruvector_voi_router::{calibrate::percentile_threshold, router::VoiRoutedSearch};

// ─── Dataset parameters (identical to the accepted coherence-hnsw bench,
//     so results are directly comparable) ─────────────────────────────────
const N_CLUSTERS: usize = 8;
const N_PER_CLUSTER: usize = 250;
const N: usize = N_CLUSTERS * N_PER_CLUSTER;
const DIMS: usize = 32;
const CLUSTER_STD: f32 = 0.15;

const M: usize = 16;
const M_LONGJUMP: usize = 6;
const K: usize = 10;
const EF: usize = 80;
const ENTRY: usize = 0;
const GATE_THRESHOLD: f32 = 0.50;

const CALIB_QUERIES: usize = 300;
const EVAL_QUERIES: usize = 400;
const WARMUP_PASSES: usize = 1;
const TIMED_REPEATS: usize = 7;

// ─── Acceptance thresholds (fixed before running; not adjusted afterward) ──
const MIN_BASELINE_RECALL: f32 = 0.85;
const MIN_GATED_RECALL: f32 = 0.82;
const RECALL_SAFETY_MARGIN: f32 = 0.01; // routed must stay within 1pp of parent
const GATED_LATENCY_NOISE_MARGIN: f32 = 1.02; // 2% tolerance vs parent latency
const EASY_LATENCY_TARGET_RATIO: f32 = 1.10; // routed easy-group vs baseline easy-group

fn main() {
    println!("# VoI-Gated Coherence Routing — Benchmark\n");
    println!("## Dataset\n");
    println!(
        "- Clusters: {N_CLUSTERS} × {N_PER_CLUSTER} = {N} vectors, D={DIMS}, std={CLUSTER_STD}"
    );
    println!("- Graph: M={M} local + {M_LONGJUMP} long-jump, ef={EF}, k={K}, fixed entry={ENTRY}");
    println!("- Calibration queries: {CALIB_QUERIES} (seed 0x0C4B_CA11, disjoint from eval)");
    println!("- Evaluation queries: {EVAL_QUERIES} (seed 0xCAFE_BABE)\n");

    let (data, assignments) =
        clustered_unit_vectors(N_CLUSTERS, N_PER_CLUSTER, DIMS, CLUSTER_STD, 0xDEAD_BEEF);
    let entry_cluster = assignments[ENTRY];

    let build_start = Instant::now();
    let graph = FlatGraph::build(
        data.clone(),
        GraphConfig {
            m: M,
            m_longjump: M_LONGJUMP,
            dims: DIMS,
        },
    );
    let build_ms = build_start.elapsed().as_millis();
    let mem_bytes = memory_estimate_bytes(N, DIMS, M + M_LONGJUMP);

    let calib_queries = clustered_queries(
        CALIB_QUERIES,
        DIMS,
        &data,
        N_PER_CLUSTER,
        CLUSTER_STD,
        0x0C4B_CA11,
    );
    let calib_gt = ground_truth(&data, &calib_queries, DIMS, K);

    let eval_queries = clustered_queries(
        EVAL_QUERIES,
        DIMS,
        &data,
        N_PER_CLUSTER,
        CLUSTER_STD,
        0xCAFE_BABE,
    );
    let eval_gt = ground_truth(&data, &eval_queries, DIMS, K);

    // ─── Calibration: bounded percentile search (Darwin-lite) ──────────────
    let calib_d0s: Vec<f32> = calib_queries
        .iter()
        .map(|q| l2_sq(q, graph.row(ENTRY)))
        .collect();

    let calib_gated = run_policy(
        &graph,
        &CoherenceGatedSearch {
            threshold: GATE_THRESHOLD,
        },
        &calib_queries,
        &calib_gt,
        ENTRY,
        K,
        EF,
    );

    println!(
        "## Calibration (threshold search, {} queries, never used for evaluation)\n",
        CALIB_QUERIES
    );
    println!("| Gen | Pct | Threshold | Recall | Mean (µs) | Fitness | Hard-constraint |");
    println!("|-----|-----|-----------|--------|-----------|---------|------------------|");

    let mut evaluated: Vec<(f32, f32, f32, f64, f64, bool)> = Vec::new(); // (pct, threshold, recall, mean_us, fitness, ok)
    let mut gen_pcts = vec![10.0f32, 35.0, 60.0, 85.0];
    // Fallback if no candidate ever satisfies the recall hard-constraint below:
    // pct 0.0 -> threshold = min(calib_d0s), so d0 <= threshold is (almost) never
    // true and the router routes (almost) every query to AlwaysGated, i.e. "keep
    // the parent". (Routing is `d0 <= threshold -> Baseline`, so a *low*
    // threshold means *few* queries qualify for Baseline, not many.)
    let mut best_pct = 0.0f32;

    for gen in 1..=3 {
        let mut gen_best: Option<(f32, f64)> = None; // (pct, fitness) among candidates satisfying the hard constraint
        for &pct in &gen_pcts {
            if evaluated.iter().any(|e| (e.0 - pct).abs() < 1e-6) {
                continue; // skip re-evaluating a percentile already tried in an earlier generation
            }
            let threshold = percentile_threshold(&calib_d0s, pct);
            let router = VoiRoutedSearch::new(threshold, GATE_THRESHOLD);
            let stats = run_policy(&graph, &router, &calib_queries, &calib_gt, ENTRY, K, EF);

            let recall_ratio = if calib_gated.recall > 0.0 {
                stats.recall / calib_gated.recall
            } else {
                1.0
            };
            let latency_ratio = if calib_gated.mean_us > 0.0 {
                stats.mean_us / calib_gated.mean_us
            } else {
                1.0
            };
            let hard_ok = recall_ratio >= 1.0 - RECALL_SAFETY_MARGIN;
            let fitness = recall_ratio as f64 - 0.5 * latency_ratio;

            println!(
                "| {gen} | {pct:.0} | {threshold:.4} | {:.1}% | {:.2} | {fitness:.4} | {} |",
                stats.recall * 100.0,
                stats.mean_us,
                if hard_ok { "PASS" } else { "FAIL" }
            );

            evaluated.push((
                pct,
                threshold,
                stats.recall,
                stats.mean_us,
                fitness,
                hard_ok,
            ));
            if hard_ok && gen_best.map(|(_, f)| fitness > f).unwrap_or(true) {
                gen_best = Some((pct, fitness));
            }
        }
        if let Some((pct, _)) = gen_best {
            best_pct = pct;
        }
        gen_pcts = match gen {
            1 => [
                best_pct - 15.0,
                best_pct - 5.0,
                best_pct + 5.0,
                best_pct + 15.0,
            ]
            .map(|p| p.clamp(0.0, 100.0))
            .to_vec(),
            2 => [
                best_pct - 5.0,
                best_pct - 2.0,
                best_pct + 2.0,
                best_pct + 5.0,
            ]
            .map(|p| p.clamp(0.0, 100.0))
            .to_vec(),
            _ => vec![],
        };
    }

    // Promote the single best candidate across all evaluated generations (maximum_promotions = 1).
    let promoted = evaluated
        .iter()
        .filter(|e| e.5)
        .max_by(|a, b| a.4.total_cmp(&b.4));
    let (final_pct, final_threshold) = match promoted {
        Some(&(pct, threshold, ..)) => (pct, threshold),
        None => {
            println!("\n  No calibration candidate met the recall hard-constraint — keeping the parent (AlwaysGated).");
            (0.0, f32::NEG_INFINITY) // d0 <= threshold is never true: every query routes to Gated
        }
    };
    println!("\n  Promoted: percentile={final_pct:.0} → distance_threshold={final_threshold:.4}\n");

    // ─── Evaluation (held-out queries, never seen during calibration) ──────
    let router = VoiRoutedSearch::new(final_threshold, GATE_THRESHOLD);
    let baseline_eval = run_policy(
        &graph,
        &BaselineSearch,
        &eval_queries,
        &eval_gt,
        ENTRY,
        K,
        EF,
    );
    let gated_eval = run_policy(
        &graph,
        &CoherenceGatedSearch {
            threshold: GATE_THRESHOLD,
        },
        &eval_queries,
        &eval_gt,
        ENTRY,
        K,
        EF,
    );
    let routed_eval = run_policy(&graph, &router, &eval_queries, &eval_gt, ENTRY, K, EF);

    let groups: Vec<bool> = eval_gt
        .iter()
        .map(|nn| assignments[nn[0] as usize] == entry_cluster) // true = "easy" (near entry's own cluster)
        .collect();
    let n_easy = groups.iter().filter(|&&e| e).count();

    let baseline_easy = run_policy_grouped(
        &graph,
        &BaselineSearch,
        &eval_queries,
        &eval_gt,
        &groups,
        true,
        ENTRY,
        K,
        EF,
    );
    let baseline_hard = run_policy_grouped(
        &graph,
        &BaselineSearch,
        &eval_queries,
        &eval_gt,
        &groups,
        false,
        ENTRY,
        K,
        EF,
    );
    let gated_easy = run_policy_grouped(
        &graph,
        &CoherenceGatedSearch {
            threshold: GATE_THRESHOLD,
        },
        &eval_queries,
        &eval_gt,
        &groups,
        true,
        ENTRY,
        K,
        EF,
    );
    let gated_hard = run_policy_grouped(
        &graph,
        &CoherenceGatedSearch {
            threshold: GATE_THRESHOLD,
        },
        &eval_queries,
        &eval_gt,
        &groups,
        false,
        ENTRY,
        K,
        EF,
    );
    let routed_easy = run_policy_grouped(
        &graph,
        &router,
        &eval_queries,
        &eval_gt,
        &groups,
        true,
        ENTRY,
        K,
        EF,
    );
    let routed_hard = run_policy_grouped(
        &graph,
        &router,
        &eval_queries,
        &eval_gt,
        &groups,
        false,
        ENTRY,
        K,
        EF,
    );

    let route_fraction_gated = eval_queries
        .iter()
        .filter(|q| l2_sq(q, graph.row(ENTRY)) > final_threshold)
        .count() as f64
        / EVAL_QUERIES as f64;

    println!(
        "## Evaluation ({} queries, {n_easy} easy / {} hard by true-nearest-cluster)\n",
        EVAL_QUERIES,
        EVAL_QUERIES - n_easy
    );
    println!("| Policy | Recall@{K} | Mean (µs) | p95 (µs) | QPS |");
    println!("|--------|-----------|-----------|----------|-----|");
    for (name, s) in [
        ("Baseline", &baseline_eval),
        ("AlwaysGated", &gated_eval),
        ("VoiRouted", &routed_eval),
    ] {
        println!(
            "| {name} | {:.1}% | {:.2} | {:.2} | {:.0} |",
            s.recall * 100.0,
            s.mean_us,
            s.p95_us,
            s.qps
        );
    }
    println!(
        "\n- VoiRouted routed {:.1}% of eval queries to AlwaysGated, {:.1}% to Baseline\n",
        route_fraction_gated * 100.0,
        (1.0 - route_fraction_gated) * 100.0
    );

    println!("### Breakdown by true difficulty group\n");
    println!("| Policy | Group | Recall@{K} | Mean (µs) |");
    println!("|--------|-------|-----------|-----------|");
    println!(
        "| Baseline | easy | {:.1}% | {:.2} |",
        baseline_easy.recall * 100.0,
        baseline_easy.mean_us
    );
    println!(
        "| Baseline | hard | {:.1}% | {:.2} |",
        baseline_hard.recall * 100.0,
        baseline_hard.mean_us
    );
    println!(
        "| AlwaysGated | easy | {:.1}% | {:.2} |",
        gated_easy.recall * 100.0,
        gated_easy.mean_us
    );
    println!(
        "| AlwaysGated | hard | {:.1}% | {:.2} |",
        gated_hard.recall * 100.0,
        gated_hard.mean_us
    );
    println!(
        "| VoiRouted | easy | {:.1}% | {:.2} |",
        routed_easy.recall * 100.0,
        routed_easy.mean_us
    );
    println!(
        "| VoiRouted | hard | {:.1}% | {:.2} |\n",
        routed_hard.recall * 100.0,
        routed_hard.mean_us
    );

    println!(
        "- Graph build: {build_ms} ms; memory: {:.1} KB\n",
        mem_bytes as f64 / 1024.0
    );

    // ─── Acceptance ──────────────────────────────────────────────────────────
    println!("## Acceptance Tests\n");
    let t1 = baseline_eval.recall >= MIN_BASELINE_RECALL;
    println!(
        "  [{}] Baseline recall@{K} >= {:.0}%: {:.1}%",
        tag(t1),
        MIN_BASELINE_RECALL * 100.0,
        baseline_eval.recall * 100.0
    );
    let t2 = gated_eval.recall >= MIN_GATED_RECALL;
    println!(
        "  [{}] AlwaysGated recall@{K} >= {:.0}%: {:.1}%",
        tag(t2),
        MIN_GATED_RECALL * 100.0,
        gated_eval.recall * 100.0
    );
    let t3 = routed_eval.recall >= gated_eval.recall - RECALL_SAFETY_MARGIN;
    println!(
        "  [{}] VoiRouted recall within {:.0}pp of AlwaysGated: {:.1}% vs {:.1}%",
        tag(t3),
        RECALL_SAFETY_MARGIN * 100.0,
        routed_eval.recall * 100.0,
        gated_eval.recall * 100.0
    );
    let t4 = routed_eval.mean_us <= gated_eval.mean_us as f64 * GATED_LATENCY_NOISE_MARGIN as f64;
    println!("  [{}] VoiRouted mean latency <= AlwaysGated * {GATED_LATENCY_NOISE_MARGIN}: {:.2}us vs {:.2}us", tag(t4), routed_eval.mean_us, gated_eval.mean_us);
    let t5 = routed_easy.mean_us <= baseline_easy.mean_us * EASY_LATENCY_TARGET_RATIO as f64
        && routed_easy.recall >= baseline_easy.recall - RECALL_SAFETY_MARGIN;
    println!("  [{}] VoiRouted matches Baseline on easy group (latency <= {EASY_LATENCY_TARGET_RATIO}x, recall within {:.0}pp): {:.2}us/{:.1}% vs {:.2}us/{:.1}%", tag(t5), RECALL_SAFETY_MARGIN * 100.0, routed_easy.mean_us, routed_easy.recall * 100.0, baseline_easy.mean_us, baseline_easy.recall * 100.0);
    let t6 = routed_hard.recall >= gated_hard.recall - RECALL_SAFETY_MARGIN;
    println!("  [{}] VoiRouted hard-group recall within {:.0}pp of AlwaysGated hard-group: {:.1}% vs {:.1}%\n", tag(t6), RECALL_SAFETY_MARGIN * 100.0, routed_hard.recall * 100.0, gated_hard.recall * 100.0);

    println!("## Overall\n");
    if !(t1 && t2 && t3 && t4) {
        println!("  REJECT — VoiRouted is either unsafe (recall) or strictly worse than the AlwaysGated parent.");
        std::process::exit(1);
    } else if t5 && t6 {
        println!("  ACCEPT — VoiRouted preserves AlwaysGated's safety and demonstrates the claimed easy-group benefit.");
        std::process::exit(0);
    } else {
        println!("  INCONCLUSIVE — VoiRouted is safe and no worse than the parent, but did not demonstrate the specific easy-group benefit the hypothesis predicted.");
        std::process::exit(2);
    }
}

fn tag(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}

struct PolicyStats {
    recall: f32,
    mean_us: f64,
    p95_us: f64,
    qps: f64,
}

/// Runs `searcher` over every query, with `WARMUP_PASSES` untimed passes
/// first and latency pooled across `TIMED_REPEATS` timed passes — a single
/// pass at these query counts is a few hundred nanoseconds of real work per
/// query, well within OS scheduling jitter, so pooling repeats is needed for
/// a latency comparison to mean anything. Recall is deterministic given a
/// fixed graph/query/config (no randomness in the search algorithms), so it
/// is computed once, not repeated.
fn run_policy(
    graph: &FlatGraph,
    searcher: &dyn Searcher,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    entry: usize,
    k: usize,
    ef: usize,
) -> PolicyStats {
    for q in queries {
        for _ in 0..WARMUP_PASSES {
            searcher.search(graph, q, k, ef, entry);
        }
    }

    let mut latencies_ns = Vec::with_capacity(queries.len() * TIMED_REPEATS);
    let mut total_recall = 0.0f32;
    for (qi, q) in queries.iter().enumerate() {
        let mut res = None;
        for _ in 0..TIMED_REPEATS {
            let t0 = Instant::now();
            let r = searcher.search(graph, q, k, ef, entry);
            latencies_ns.push(t0.elapsed().as_nanos() as u64);
            res = Some(r);
        }
        total_recall += recall_at_k(&res.unwrap(), &gt[qi]);
    }
    let stats = LatencyStats::compute(latencies_ns);
    PolicyStats {
        recall: total_recall / queries.len() as f32,
        mean_us: stats.mean_us(),
        p95_us: stats.p95_us(),
        qps: stats.throughput_qps(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_policy_grouped(
    graph: &FlatGraph,
    searcher: &dyn Searcher,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    groups: &[bool],
    want_easy: bool,
    entry: usize,
    k: usize,
    ef: usize,
) -> PolicyStats {
    for (qi, q) in queries.iter().enumerate() {
        if groups[qi] != want_easy {
            continue;
        }
        for _ in 0..WARMUP_PASSES {
            searcher.search(graph, q, k, ef, entry);
        }
    }

    let mut latencies_ns = Vec::new();
    let mut total_recall = 0.0f32;
    let mut n = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        if groups[qi] != want_easy {
            continue;
        }
        let mut res = None;
        for _ in 0..TIMED_REPEATS {
            let t0 = Instant::now();
            let r = searcher.search(graph, q, k, ef, entry);
            latencies_ns.push(t0.elapsed().as_nanos() as u64);
            res = Some(r);
        }
        total_recall += recall_at_k(&res.unwrap(), &gt[qi]);
        n += 1;
    }
    if n == 0 {
        return PolicyStats {
            recall: 1.0,
            mean_us: 0.0,
            p95_us: 0.0,
            qps: 0.0,
        };
    }
    let stats = LatencyStats::compute(latencies_ns);
    PolicyStats {
        recall: total_recall / n as f32,
        mean_us: stats.mean_us(),
        p95_us: stats.p95_us(),
        qps: stats.throughput_qps(),
    }
}
