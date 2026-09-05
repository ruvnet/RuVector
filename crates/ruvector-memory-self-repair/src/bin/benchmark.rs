//! Reproducible benchmark: baseline vs monitor-only vs spectral-repair on an
//! identical simulated agent-memory timeline. Run with:
//!
//!   cargo run --release -p ruvector-memory-self-repair --bin benchmark
//!
//! Prints a human-readable table and writes raw JSON evidence next to the
//! research writeup so numbers are never hand-transcribed.

use ruvector_memory_self_repair::drift::{
    apply_arrival, apply_compaction, apply_drift_wave, build_arrivals, build_drift_schedule,
    SimConfig,
};
use ruvector_memory_self_repair::graph::MemoryGraph;
use ruvector_memory_self_repair::repair::{RepairConfig, Runner, Variant};
use ruvector_memory_self_repair::retrieval::mean_recall_at_k;
use serde::Serialize;
use std::time::Instant;

const MONITOR_INTERVAL: u64 = 10;
const QUERY_STRIDE: u64 = 5;
const RECALL_K: usize = 10;
// Deliberately tight: a generous budget (originally 200 on a ~2600-node
// graph) let best-first search route around decayed bridges via redundant
// paths, saturating recall@10 at 1.0 for every variant and making the
// benchmark unable to discriminate baseline from repair. Associative
// recall in a real agent is few-hop and budget constrained, so this also
// better matches what it is standing in for.
const WALK_BUDGET: usize = 20;
const DETECTION_WINDOW: u64 = 2 * MONITOR_INTERVAL;

#[derive(Serialize)]
struct VariantResult {
    variant: String,
    final_alive_nodes: usize,
    final_edge_count: usize,
    mean_recall_at_10: f64,
    recall_queries_counted: usize,
    monitor_calls: usize,
    alert_steps: usize,
    repairs_triggered: usize,
    cumulative_edges_added: usize,
    repair_edge_fraction_of_final_graph: f64,
    wall_time_ms: f64,
    witness_chain_valid: Option<bool>,
    witness_receipt_count: usize,
}

#[derive(Serialize)]
struct DetectionResult {
    true_drift_events: usize,
    detected_within_window: usize,
    detection_recall: f64,
    alerts_total: usize,
    alerts_within_window_of_any_drift: usize,
    alert_precision: f64,
}

#[derive(Serialize)]
struct Evidence {
    config: String,
    seed: u64,
    total_steps: u64,
    baseline: VariantResult,
    monitor_only: VariantResult,
    spectral_repair: VariantResult,
    detection: DetectionResult,
    recall_uplift_pp: f64,
    acceptance_threshold_pp: f64,
    max_repair_edge_fraction: f64,
    acceptance_result: String,
}

fn run_variant(
    cfg: &SimConfig,
    variant: Variant,
) -> (MemoryGraph, VariantResult, Option<Runner>, u128) {
    let (_, arrivals) = build_arrivals(cfg);
    let drift_schedule = build_drift_schedule(cfg);
    let mut graph = MemoryGraph::new();
    let mut runner = if variant == Variant::NoRepair {
        None
    } else {
        Some(Runner::new(RepairConfig::default()))
    };

    let start = Instant::now();
    let mut drift_idx = 0usize;
    for (step, embedding) in arrivals.into_iter().enumerate() {
        let step = step as u64;
        apply_arrival(&mut graph, embedding, step, cfg);

        if step > 0 && step % cfg.compaction_interval == 0 {
            apply_compaction(&mut graph, step, cfg);
        }
        while drift_idx < drift_schedule.len() && drift_schedule[drift_idx].step == step {
            apply_drift_wave(&mut graph, drift_schedule[drift_idx], cfg);
            drift_idx += 1;
        }
        if let Some(r) = runner.as_mut() {
            if step % MONITOR_INTERVAL == 0 {
                r.check_and_repair(&mut graph, step, variant);
            }
        }
    }
    let wall_time = start.elapsed().as_micros();

    let queries: Vec<usize> = (0..cfg.total_steps)
        .step_by(QUERY_STRIDE as usize)
        .map(|s| s as usize)
        .collect();
    let (recall, counted) = mean_recall_at_k(&graph, &queries, RECALL_K, WALK_BUDGET);

    let final_edges = graph.edge_count();
    let (monitor_calls, alert_steps, repairs, cum_edges, witness_valid, witness_count) =
        match &runner {
            Some(r) => (
                r.stats.monitor_calls,
                r.stats.alert_steps.len(),
                r.stats.repairs.len(),
                r.stats.cumulative_edges_added,
                Some(r.witness.verify().is_ok()),
                r.witness.receipts.len(),
            ),
            None => (0, 0, 0, 0, None, 0),
        };
    let repair_fraction = if final_edges == 0 {
        0.0
    } else {
        cum_edges as f64 / final_edges as f64
    };

    let result = VariantResult {
        variant: format!("{variant:?}"),
        final_alive_nodes: graph.alive_ids().len(),
        final_edge_count: final_edges,
        mean_recall_at_10: recall,
        recall_queries_counted: counted,
        monitor_calls,
        alert_steps,
        repairs_triggered: repairs,
        cumulative_edges_added: cum_edges,
        repair_edge_fraction_of_final_graph: repair_fraction,
        wall_time_ms: wall_time as f64 / 1000.0,
        witness_chain_valid: witness_valid,
        witness_receipt_count: witness_count,
    };
    (graph, result, runner, wall_time)
}

fn main() {
    let seed = std::env::var("SIM_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);
    let cfg = SimConfig {
        seed,
        ..Default::default()
    };
    println!("=== ruvector-memory-self-repair benchmark ===");
    println!(
        "config: dim={} n_topics={} total_steps={} semantic_k={} compaction_interval={} evict={} drift_interval={} decay={} seed={}",
        cfg.dim, cfg.n_topics, cfg.total_steps, cfg.semantic_k, cfg.compaction_interval,
        cfg.compaction_evict_count, cfg.drift_interval, cfg.drift_decay_factor, cfg.seed
    );

    let (_, baseline, _, _) = run_variant(&cfg, Variant::NoRepair);
    let (_, monitor_only, monitor_runner, _) = run_variant(&cfg, Variant::MonitorOnly);
    let (_, spectral_repair, _, _) = run_variant(&cfg, Variant::SpectralRepair);

    // Detection quality is scored off the monitor-only run: it observes the
    // exact same graph the baseline would have (no repair mutation), so its
    // alerts are a clean read on "did the spectral signal notice the
    // injected drift" independent of any repair feedback loop.
    let drift_schedule = build_drift_schedule(&cfg);
    let alert_steps: Vec<u64> = monitor_runner
        .as_ref()
        .map(|r| r.stats.alert_steps.clone())
        .unwrap_or_default();
    let mut detected = 0usize;
    for ev in &drift_schedule {
        if alert_steps
            .iter()
            .any(|&a| a >= ev.step && a <= ev.step + DETECTION_WINDOW)
        {
            detected += 1;
        }
    }
    let mut precise = 0usize;
    for &a in &alert_steps {
        if drift_schedule
            .iter()
            .any(|ev| a >= ev.step && a <= ev.step + DETECTION_WINDOW)
        {
            precise += 1;
        }
    }
    let detection = DetectionResult {
        true_drift_events: drift_schedule.len(),
        detected_within_window: detected,
        detection_recall: if drift_schedule.is_empty() {
            0.0
        } else {
            detected as f64 / drift_schedule.len() as f64
        },
        alerts_total: alert_steps.len(),
        alerts_within_window_of_any_drift: precise,
        alert_precision: if alert_steps.is_empty() {
            0.0
        } else {
            precise as f64 / alert_steps.len() as f64
        },
    };

    let recall_uplift_pp = (spectral_repair.mean_recall_at_10 - baseline.mean_recall_at_10) * 100.0;
    const ACCEPTANCE_THRESHOLD_PP: f64 = 10.0;
    const MAX_REPAIR_EDGE_FRACTION: f64 = 0.08;

    let repair_bounded =
        spectral_repair.repair_edge_fraction_of_final_graph <= MAX_REPAIR_EDGE_FRACTION;
    let witness_ok = spectral_repair.witness_chain_valid.unwrap_or(false);
    let recall_met = recall_uplift_pp >= ACCEPTANCE_THRESHOLD_PP;

    let acceptance_result = if !witness_ok {
        "REJECT (witness chain invalid)".to_string()
    } else if !repair_bounded {
        "INCONCLUSIVE (repair exceeded targeted-edit bound, cannot rule out rebuild-in-disguise)"
            .to_string()
    } else if recall_met {
        "ACCEPT".to_string()
    } else {
        "REJECT (recall uplift below acceptance threshold)".to_string()
    };

    println!(
        "\n{:<16} {:>10} {:>8} {:>12} {:>8} {:>7} {:>9} {:>10} {:>9}",
        "variant",
        "alive",
        "edges",
        "recall@10",
        "monCalls",
        "alerts",
        "repairs",
        "edgesAdd",
        "wallMs"
    );
    for r in [&baseline, &monitor_only, &spectral_repair] {
        println!(
            "{:<16} {:>10} {:>8} {:>12.4} {:>8} {:>7} {:>9} {:>10} {:>9.2}",
            r.variant,
            r.final_alive_nodes,
            r.final_edge_count,
            r.mean_recall_at_10,
            r.monitor_calls,
            r.alert_steps,
            r.repairs_triggered,
            r.cumulative_edges_added,
            r.wall_time_ms
        );
    }

    println!("\ndrift events injected: {}", detection.true_drift_events);
    println!(
        "detection recall (alert within {DETECTION_WINDOW} steps of drift): {:.4}",
        detection.detection_recall
    );
    println!(
        "alert precision (alert within window of some drift): {:.4}",
        detection.alert_precision
    );
    println!("\nrecall@10 uplift (spectral_repair - baseline): {recall_uplift_pp:.2} pp (threshold: {ACCEPTANCE_THRESHOLD_PP} pp)");
    println!(
        "repair edge fraction of final graph: {:.4} (max allowed: {MAX_REPAIR_EDGE_FRACTION})",
        spectral_repair.repair_edge_fraction_of_final_graph
    );
    println!(
        "witness chain valid: {witness_ok}, receipts: {}",
        spectral_repair.witness_receipt_count
    );
    println!("\nACCEPTANCE RESULT: {acceptance_result}");

    let evidence = Evidence {
        config: format!("{cfg:?}"),
        seed: cfg.seed,
        total_steps: cfg.total_steps,
        baseline,
        monitor_only,
        spectral_repair,
        detection,
        recall_uplift_pp,
        acceptance_threshold_pp: ACCEPTANCE_THRESHOLD_PP,
        max_repair_edge_fraction: MAX_REPAIR_EDGE_FRACTION,
        acceptance_result,
    };
    let json = serde_json::to_string_pretty(&evidence).unwrap();
    let out_path = format!(
        "/tmp/claude-0/-home-user-ruvector/f7813f58-cd3a-5bbb-9872-9c6f3c070715/scratchpad/spectral-memory-self-repair-evidence-seed{}.json",
        cfg.seed
    );
    let out_path = out_path.as_str();
    if let Some(parent) = std::path::Path::new(out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(out_path, &json).expect("write evidence json");
    println!("\nraw evidence written to {out_path}");
}
