//! Benchmark binary: baseline vs. two mincut-partitioned retention
//! variants on a synthetic clustered agent-memory corpus.
//!
//! ```text
//! cargo run --release -p ruvector-partition-memory --bin benchmark -- \
//!     [n] [floor_min] [coherence_ratio] [fixed_k] [fixed_k_max_n]
//! ```
//! Defaults: n=4000, floor_min=3, coherence_ratio=0.35, fixed_k=10, fixed_k_max_n=600.
//!
//! `fixed_k_max_n` gates candidate A (`MincutFixedK`, built on the existing
//! `ruvector_mincut::GraphPartitioner`): measured during this nightly's
//! development at 8.4s for n=500 and not finished after 5m42s at n=4000 —
//! see the nightly research doc for the repro. Above this size candidate A
//! is skipped with an explicit note rather than silently hung on or
//! silently omitted.

use ruvector_partition_memory::corpus::{generate, CorpusConfig, MemoryRecord};
use ruvector_partition_memory::graph::build_knn_edges;
use ruvector_partition_memory::metrics::{evaluate, VariantReport};
use ruvector_partition_memory::partition::{adaptive_partition, fixed_k_partition, AdaptiveConfig};
use ruvector_partition_memory::retention::{
    retain_global_top_score, retain_partitioned, RetentionPolicy,
};
use std::time::Instant;

struct Args {
    n: usize,
    floor_min: usize,
    coherence_ratio: f64,
    fixed_k: usize,
    fixed_k_max_n: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    Args {
        n: args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4000),
        floor_min: args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3),
        coherence_ratio: args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.35),
        fixed_k: args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10),
        fixed_k_max_n: args.get(5).and_then(|s| s.parse().ok()).unwrap_or(600),
    }
}

/// The most recently accessed `n_ctx` memories, standing in for "what the
/// agent was just working on" — the realistic, non-strawman context a
/// consolidation event has available, and exactly the scenario in which a
/// pure top-score policy can starve unrelated topics.
fn recency_context(records: &[MemoryRecord], n_ctx: usize) -> Vec<Vec<f32>> {
    let mut sorted: Vec<&MemoryRecord> = records.iter().collect();
    sorted.sort_by(|a, b| b.last_accessed_tick.cmp(&a.last_accessed_tick));
    sorted
        .into_iter()
        .take(n_ctx)
        .map(|r| r.embedding.clone())
        .collect()
}

fn print_report(r: &VariantReport) {
    println!(
        "{:<16} retained={:<6} overall_recall={:.4} worst_cluster_recall={:.4} coverage={:.3} partition_us={:<10} retention_us={:<8}",
        r.name, r.retained_count, r.overall_recall, r.worst_cluster_recall, r.cluster_coverage, r.partition_us, r.retention_us
    );
}

fn main() {
    let args = parse_args();
    let cfg = CorpusConfig {
        n: args.n,
        ..CorpusConfig::default()
    };

    let t0 = Instant::now();
    let corpus = generate(&cfg);
    let corpus_gen_us = t0.elapsed().as_micros();

    let target_size = (corpus.records.len() as f64 * 0.5).round() as usize;
    let context = recency_context(&corpus.records, 20);
    let vertices: Vec<u64> = corpus.records.iter().map(|r| r.id).collect();
    let policy = RetentionPolicy {
        floor_min: args.floor_min,
    };

    let t0 = Instant::now();
    let edges = build_knn_edges(&corpus.records, 10);
    let knn_us = t0.elapsed().as_micros();

    println!("=== ruvector-partition-memory nightly benchmark ===");
    println!(
        "n={} dims={} cluster_sizes={:?} focus_cluster={} target_size={} (50% retention) queries={}",
        corpus.records.len(),
        corpus.dims,
        corpus.cluster_sizes,
        corpus.focus_cluster,
        target_size,
        corpus.queries.len()
    );
    println!(
        "params: floor_min={} coherence_ratio={} fixed_k={} fixed_k_max_n={}",
        args.floor_min, args.coherence_ratio, args.fixed_k, args.fixed_k_max_n
    );
    println!(
        "corpus_gen_us={corpus_gen_us} knn_graph_us={knn_us} knn_edges={}",
        edges.len()
    );
    println!();

    // ---- Baseline: GlobalTopScore (ruvector-agent-memory CoherencePolicy) ----
    let t0 = Instant::now();
    let baseline_ids = retain_global_top_score(&corpus.records, &context, target_size);
    let baseline_us = t0.elapsed().as_micros();
    let baseline_report = evaluate(&corpus, &baseline_ids, "GlobalTopScore", 0, baseline_us);

    // ---- Candidate A: MincutFixedK (existing ruvector_mincut::GraphPartitioner) ----
    let report_a: Option<VariantReport> = if corpus.records.len() <= args.fixed_k_max_n {
        let t0 = Instant::now();
        let parts_a = fixed_k_partition(&vertices, &edges, args.fixed_k);
        let partition_a_us = t0.elapsed().as_micros();
        let t0 = Instant::now();
        let ids_a = retain_partitioned(&corpus.records, &parts_a, &context, target_size, &policy);
        let retention_a_us = t0.elapsed().as_micros();
        println!(
            "MincutFixedK partitions: {} (K={} requested) sizes={:?}",
            parts_a.len(),
            args.fixed_k,
            parts_a.iter().map(|p| p.len()).collect::<Vec<_>>()
        );
        Some(evaluate(
            &corpus,
            &ids_a,
            "MincutFixedK",
            partition_a_us,
            retention_a_us,
        ))
    } else {
        println!(
            "MincutFixedK: SKIPPED (n={} exceeds fixed_k_max_n={}; ruvector_mincut::GraphPartitioner \
             measured at 8.4s for n=500 and did not finish in 5m42s for n=4000 during this nightly's \
             development — see the research doc for the repro)",
            corpus.records.len(),
            args.fixed_k_max_n
        );
        None
    };

    // ---- Candidate B: MincutAdaptive (new adaptive-depth bisection) ----
    let adaptive_cfg = AdaptiveConfig {
        coherence_ratio: args.coherence_ratio,
        ..AdaptiveConfig::default()
    };
    let t0 = Instant::now();
    let result_b = adaptive_partition(&edges, &vertices, &adaptive_cfg);
    let partition_b_us = t0.elapsed().as_micros();
    let t0 = Instant::now();
    let ids_b = retain_partitioned(
        &corpus.records,
        &result_b.clusters,
        &context,
        target_size,
        &policy,
    );
    let retention_b_us = t0.elapsed().as_micros();
    let report_b = evaluate(
        &corpus,
        &ids_b,
        "MincutAdaptive",
        partition_b_us,
        retention_b_us,
    );
    println!(
        "MincutAdaptive partitions: {} (coherence_ratio={}) sizes={:?}",
        result_b.clusters.len(),
        args.coherence_ratio,
        result_b
            .clusters
            .iter()
            .map(|p| p.len())
            .collect::<Vec<_>>()
    );
    println!(
        "MincutAdaptive witness: {} split steps, chain_verify={}, head={}",
        result_b.witness.steps.len(),
        result_b.witness.verify(),
        result_b.witness.head()
    );
    println!();

    println!("variant          retained overall_recall worst_cluster_recall coverage partition_us retention_us");
    print_report(&baseline_report);
    if let Some(ref r) = report_a {
        print_report(r);
    }
    print_report(&report_b);
    println!();

    println!(
        "per_cluster_recall GlobalTopScore  = {:?}",
        baseline_report.per_cluster_recall
    );
    if let Some(ref r) = report_a {
        println!(
            "per_cluster_recall MincutFixedK    = {:?}",
            r.per_cluster_recall
        );
    }
    println!(
        "per_cluster_recall MincutAdaptive  = {:?}",
        report_b.per_cluster_recall
    );
    println!();

    // ---- Pre-declared acceptance thresholds (fixed before the accepted
    // run; not adjusted after seeing its results — see the nightly research
    // doc. The latency budget was set from a real calibration pass on this
    // machine, run before the accepted benchmark: MincutAdaptive measured
    // ~11.1s at n=4000, ratio=0.35, so the threshold below is set with
    // headroom above that measurement, not tightened around it.)
    const WORST_CLUSTER_GAIN_THRESHOLD: f64 = 0.15; // best candidate must beat baseline worst-cluster recall by >= 15pp
    const MAX_OVERALL_REGRESSION: f64 = 0.05; // candidates must not lose more than 5pp overall recall vs baseline
    const MAX_LATENCY_US: u128 = 30_000_000; // partition+retention wall time per candidate, this n

    let best_candidate = match &report_a {
        Some(a) if a.worst_cluster_recall > report_b.worst_cluster_recall => a,
        _ => &report_b,
    };
    let worst_gain = best_candidate.worst_cluster_recall - baseline_report.worst_cluster_recall;
    let gain_ok = worst_gain >= WORST_CLUSTER_GAIN_THRESHOLD;
    let overall_ok = report_a.as_ref().is_none_or(|a| {
        a.overall_recall >= baseline_report.overall_recall - MAX_OVERALL_REGRESSION
    }) && report_b.overall_recall
        >= baseline_report.overall_recall - MAX_OVERALL_REGRESSION;
    let latency_ok = report_a
        .as_ref()
        .is_none_or(|a| (a.partition_us + a.retention_us) < MAX_LATENCY_US)
        && (report_b.partition_us + report_b.retention_us) < MAX_LATENCY_US;
    let witness_ok = result_b.witness.verify();

    println!(
        "acceptance: best_candidate={} worst_cluster_gain_pp={:.2} (threshold_pp={:.2}) gain_ok={gain_ok} overall_ok={overall_ok} latency_ok={latency_ok} witness_ok={witness_ok}",
        best_candidate.name,
        worst_gain * 100.0,
        WORST_CLUSTER_GAIN_THRESHOLD * 100.0
    );

    let verdict = if gain_ok && overall_ok && latency_ok && witness_ok {
        "ACCEPT"
    } else {
        "REJECT"
    };
    println!("ACCEPTANCE_RESULT: {verdict}");
}
