//! End-to-end acceptance checks over a full simulated timeline: witness
//! chain integrity and the "repair stays targeted, not a rebuild" bound.
//! Kept separate from the benchmark binary so `cargo test` always exercises
//! the full pipeline even without anyone running the benchmark by hand.

use ruvector_memory_self_repair::drift::{
    apply_arrival, apply_compaction, apply_drift_wave, build_arrivals, build_drift_schedule,
    SimConfig,
};
use ruvector_memory_self_repair::graph::MemoryGraph;
use ruvector_memory_self_repair::repair::{RepairConfig, Runner, Variant};

fn run(cfg: &SimConfig, variant: Variant) -> (MemoryGraph, Runner) {
    let (_, arrivals) = build_arrivals(cfg);
    let drift_schedule = build_drift_schedule(cfg);
    let mut graph = MemoryGraph::new();
    let mut runner = Runner::new(RepairConfig::default());
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
        if step % 10 == 0 {
            runner.check_and_repair(&mut graph, step, variant);
        }
    }
    (graph, runner)
}

#[test]
fn witness_chain_is_valid_after_a_full_run() {
    let cfg = SimConfig {
        total_steps: 1200,
        ..Default::default()
    };
    let (_, runner) = run(&cfg, Variant::SpectralRepair);
    assert!(
        runner.witness.verify().is_ok(),
        "witness chain must verify end to end"
    );
    assert!(
        !runner.witness.receipts.is_empty(),
        "a drifting graph should trigger at least one repair"
    );
}

#[test]
fn repair_edges_stay_a_small_fraction_of_the_final_graph() {
    let cfg = SimConfig {
        total_steps: 1200,
        ..Default::default()
    };
    let (graph, runner) = run(&cfg, Variant::SpectralRepair);
    let fraction = runner.stats.cumulative_edges_added as f64 / graph.edge_count().max(1) as f64;
    assert!(
        fraction <= 0.08,
        "repair touched {:.4} of final graph edges, expected a targeted repair (<=0.08)",
        fraction
    );
}

#[test]
fn monitor_only_never_changes_edge_count_relative_to_no_monitoring() {
    let cfg = SimConfig {
        total_steps: 600,
        ..Default::default()
    };
    let (graph_baseline, _) = run(&cfg, Variant::NoRepair);
    let (graph_monitor, _) = run(&cfg, Variant::MonitorOnly);
    assert_eq!(graph_baseline.edge_count(), graph_monitor.edge_count());
    assert_eq!(
        graph_baseline.alive_ids().len(),
        graph_monitor.alive_ids().len()
    );
}

#[test]
fn spectral_repair_graph_has_at_least_as_many_edges_as_baseline() {
    let cfg = SimConfig {
        total_steps: 1200,
        ..Default::default()
    };
    let (graph_baseline, _) = run(&cfg, Variant::NoRepair);
    let (graph_repair, runner) = run(&cfg, Variant::SpectralRepair);
    assert!(graph_repair.edge_count() >= graph_baseline.edge_count());
    assert!(runner.stats.monitor_calls > 0);
}
