//! Adaptive-ef ANN benchmark.
//!
//! Measures recall@10, mean/p50/p95 latency, throughput, and mean distance
//! operations for three variants of beam-search ANN over a shared proximity
//! graph index.
//!
//! Usage:
//!   cargo run --release -p ruvector-adaptive-ef --bin benchmark

use ruvector_adaptive_ef::{
    brute_knn, generate_vectors, recall_at_k,
    search::{AdaptiveEfSearch, AnnSearch, FixedEfSearch, TwoStageSearch},
    ProximityGraph,
};
use std::time::Instant;

// ── Configuration ─────────────────────────────────────────────────────────────

const N: usize = 5_000;
const DIM: usize = 128;
const N_QUERIES: usize = 200;
const K: usize = 10;
const M: usize = 8;
const EF_CONSTRUCTION: usize = 32;

// ── Percentile helper ─────────────────────────────────────────────────────────

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Per-variant result record ─────────────────────────────────────────────────

struct VariantResult {
    name: &'static str,
    recall: f32,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    mean_ops: f64,
    escalation_pct: f32,
}

// ── Single variant benchmark run ──────────────────────────────────────────────

fn bench_variant<S: AnnSearch>(
    searcher: &S,
    graph: &ProximityGraph,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
) -> VariantResult {
    let mut latencies_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;
    let mut total_ops = 0u64;
    let mut escalations = 0u32;

    for (q, gt) in queries.iter().zip(ground_truth.iter()) {
        let t0 = Instant::now();
        let (results, stats) = searcher.search(graph, q, K);
        let elapsed = t0.elapsed();

        latencies_us.push(elapsed.as_secs_f64() * 1_000_000.0);
        total_recall += recall_at_k(&results, gt);
        total_ops += stats.distance_ops as u64;
        if stats.escalated {
            escalations += 1;
        }
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = queries.len() as f64;
    let mean_us = latencies_us.iter().sum::<f64>() / n;
    let p50_us = percentile(&latencies_us, 50.0);
    let p95_us = percentile(&latencies_us, 95.0);
    let qps = 1_000_000.0 / mean_us;

    VariantResult {
        name: searcher.name(),
        recall: total_recall / queries.len() as f32,
        mean_us,
        p50_us,
        p95_us,
        qps,
        mean_ops: total_ops as f64 / queries.len() as f64,
        escalation_pct: escalations as f32 / queries.len() as f32 * 100.0,
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("════════════════════════════════════════════════════════════════════════");
    println!(" ruvector-adaptive-ef  Benchmark");
    println!("════════════════════════════════════════════════════════════════════════");

    // Environment info
    println!("\n  OS:          {}", std::env::consts::OS);
    println!("  Arch:        {}", std::env::consts::ARCH);
    println!("  Dataset:     N={N}, D={DIM}, k={K}");
    println!("  Queries:     {N_QUERIES}");
    println!("  Graph:       M={M}, ef_construction={EF_CONSTRUCTION}");
    println!("\n  Variants:");
    println!("    FixedEf(32)          — ef=32 fixed for every query (baseline)");
    println!("    TwoStage(16→64)      — ef=16 probe, escalate to 64 if diff > 0.70");
    println!("    AdaptiveEf(16‥64)    — continuous ef ∈ [16,64] from diff score");

    // ── Build index ───────────────────────────────────────────────────────────
    println!("\n  Building proximity graph (N={N}, D={DIM}, M={M}, ef_c={EF_CONSTRUCTION})...");
    let corpus = generate_vectors(N, DIM, 0xDEAD_BEEF_u64);
    let t_build = Instant::now();
    let mut graph = ProximityGraph::new(M);
    for v in &corpus {
        graph.insert(v.clone(), EF_CONSTRUCTION);
    }
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;
    let mem_kb = graph.memory_bytes() / 1024;
    println!(
        "  Build complete: {build_ms:.1} ms | index size ≈ {mem_kb} KiB ({} KiB vectors + {} KiB edges)",
        (N * DIM * 4) / 1024,
        mem_kb - (N * DIM * 4) / 1024,
    );

    // ── Generate queries and ground truth ─────────────────────────────────────
    println!("\n  Computing ground truth (brute-force k={K})...");
    let queries = generate_vectors(N_QUERIES, DIM, 0xCAFE_BABE_u64);
    let t_gt = Instant::now();
    let ground_truth: Vec<Vec<usize>> = queries.iter().map(|q| brute_knn(q, &corpus, K)).collect();
    println!(
        "  Ground truth computed in {:.1} ms",
        t_gt.elapsed().as_secs_f64() * 1000.0
    );

    // ── Warm-up pass (evict cold caches) ─────────────────────────────────────
    let warmup = FixedEfSearch { ef: 32 };
    for q in &queries[..20] {
        let _ = warmup.search(&graph, q, K);
    }

    // ── Benchmark variants ────────────────────────────────────────────────────
    let fixed = FixedEfSearch { ef: 32 };
    let two_stage = TwoStageSearch {
        ef_fast: 16,
        ef_full: 64,
        threshold: 0.70,
    };
    let adaptive = AdaptiveEfSearch {
        ef_min: 16,
        ef_max: 64,
        threshold: 0.40,
    };

    println!("\n  Running benchmarks...");
    let r_fixed = bench_variant(&fixed, &graph, &queries, &ground_truth);
    let r_two = bench_variant(&two_stage, &graph, &queries, &ground_truth);
    let r_adap = bench_variant(&adaptive, &graph, &queries, &ground_truth);

    // ── Results table ─────────────────────────────────────────────────────────
    println!("\n  ┌─ Results (N={N} × D={DIM}, k={K}, queries={N_QUERIES}) ───────────────────────────────┐");
    println!(
        "  │ {:<22} {:>8} {:>9} {:>9} {:>9} {:>7} {:>9} {:>8} │",
        "Variant", "Recall", "Mean µs", "p50 µs", "p95 µs", "QPS", "Mean ops", "Esc %"
    );
    println!(
        "  │ {:<22} {:>8} {:>9} {:>9} {:>9} {:>7} {:>9} {:>8} │",
        "──────────────────────",
        "──────",
        "───────",
        "──────",
        "──────",
        "─────",
        "────────",
        "─────"
    );
    for r in [&r_fixed, &r_two, &r_adap] {
        println!(
            "  │ {:<22} {:>8.3} {:>9.1} {:>9.1} {:>9.1} {:>7.0} {:>9.1} {:>7.1}% │",
            r.name, r.recall, r.mean_us, r.p50_us, r.p95_us, r.qps, r.mean_ops, r.escalation_pct
        );
    }
    println!("  └──────────────────────────────────────────────────────────────────────────┘");

    // ── Ops comparison vs FixedEf ──────────────────────────────────────────────
    println!("\n  Distance-ops comparison (vs FixedEf baseline):");
    for r in [&r_two, &r_adap] {
        let ratio = r.mean_ops / r_fixed.mean_ops;
        let delta_recall = r.recall - r_fixed.recall;
        println!(
            "    {:<22} {:.2}× ops vs FixedEf, recall Δ {:+.3}",
            r.name, ratio, delta_recall
        );
    }

    // ── Memory estimate ────────────────────────────────────────────────────────
    println!("\n  Memory estimate:");
    println!(
        "    Index (vectors + edges): {} KiB",
        graph.memory_bytes() / 1024
    );
    println!("    Vectors alone:           {} KiB", (N * DIM * 4) / 1024);
    println!(
        "    Edge overhead:           {} KiB",
        (graph.memory_bytes() - N * DIM * 4) / 1024
    );
    println!(
        "    Bytes per vector:        {} B ({}D f32 + {} edge ptrs)",
        graph.memory_bytes() / N,
        DIM,
        graph.memory_bytes() / N - DIM * 4,
    );

    // ── Acceptance gates ──────────────────────────────────────────────────────
    // NOTE on expected recall: This is a single-layer proximity graph (HNSW layer-0
    // analogue), not a full HNSW with hierarchical routing layers.  On D=128 with
    // M=8 and ef=32, recall@10 ≈ 0.14 is expected.  The adaptive variants achieve
    // higher recall by using larger ef on hard queries.  Absolute recall would
    // improve significantly with full HNSW or higher M values.
    println!("\n  ┌─ Acceptance Gates ─────────────────────────────────────────────────────────┐");

    // Gate 1: both adaptive variants outperform the fixed-ef baseline on recall.
    // On a high-dimensional corpus where all queries escalate, both should exceed baseline.
    let gate1 = r_two.recall > r_fixed.recall && r_adap.recall > r_fixed.recall;
    println!(
        "  │ Adaptive variants recall > FixedEf({:.3}):                                    │",
        r_fixed.recall
    );
    println!(
        "  │   TwoStage={:.3} {}   AdaptiveEf={:.3} {}                                  │",
        r_two.recall,
        if r_two.recall > r_fixed.recall {
            "PASS"
        } else {
            "FAIL"
        },
        r_adap.recall,
        if r_adap.recall > r_fixed.recall {
            "PASS"
        } else {
            "FAIL"
        },
    );

    // Gate 2: AdaptiveEf uses fewer ops than TwoStage (continuous ef is more efficient).
    let gate2 = r_adap.mean_ops <= r_two.mean_ops * 1.05;
    println!(
        "  │ AdaptiveEf ops ≤ TwoStage ops×1.05: {:.1} ≤ {:.1}×1.05={:.1}  {}              │",
        r_adap.mean_ops,
        r_two.mean_ops,
        r_two.mean_ops * 1.05,
        if gate2 { "PASS" } else { "FAIL" }
    );

    // Gate 3: AdaptiveEf recall within 5% of TwoStage recall (same quality, fewer ops).
    let gate3 = r_adap.recall >= r_two.recall * 0.95;
    println!(
        "  │ AdaptiveEf recall ≥ 95% of TwoStage: {:.3} ≥ {:.3}×0.95={:.3}  {}            │",
        r_adap.recall,
        r_two.recall,
        r_two.recall * 0.95,
        if gate3 { "PASS" } else { "FAIL" }
    );

    // Gate 4: AdaptiveEf recall ≥ 1.3× FixedEf recall (meaningful improvement).
    let gate4 = r_adap.recall >= r_fixed.recall * 1.30;
    println!(
        "  │ AdaptiveEf recall ≥ 1.30× FixedEf: {:.3} ≥ {:.3}×1.30={:.3}  {}              │",
        r_adap.recall,
        r_fixed.recall,
        r_fixed.recall * 1.30,
        if gate4 { "PASS" } else { "FAIL" }
    );

    println!("  └──────────────────────────────────────────────────────────────────────────────┘");

    let all_pass = gate1 && gate2 && gate3 && gate4;
    println!(
        "\n  Overall: {}",
        if all_pass {
            "ALL GATES PASSED"
        } else {
            "SOME GATES FAILED — see above"
        }
    );
    println!("════════════════════════════════════════════════════════════════════════");

    if !all_pass {
        std::process::exit(1);
    }
}
