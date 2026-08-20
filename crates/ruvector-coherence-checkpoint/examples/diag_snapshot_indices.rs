//! Diagnostic supporting the nightly research finding: prints the exact
//! event index of every snapshot `DriftTriggered` takes, showing inter-
//! snapshot spacing grows over the stream (evidence that a whole-history
//! cumulative-mean drift signal loses sensitivity as the store accumulates
//! history, rather than staying responsive to recent bursts).
//!
//! Usage: cargo run --release -p ruvector-coherence-checkpoint --example diag_snapshot_indices

use ruvector_coherence_checkpoint::{
    generate_workload, run_checkpoint_policy, DriftTriggered, WorkloadConfig,
};

fn main() {
    let cfg = WorkloadConfig {
        dims: 48,
        n_events: 6000,
        calm_phase_len: 250,
        burst_phase_len: 50,
        noise: 0.04,
    };
    let events = generate_workload(&cfg, 2026);
    let (run, _gate) = run_checkpoint_policy(&events, cfg.dims, DriftTriggered { threshold: 0.08 });
    let indices: Vec<usize> = run.snapshots.iter().map(|s| s.event_index).collect();
    let gaps: Vec<usize> = indices.windows(2).map(|w| w[1] - w[0]).collect();
    println!("snapshot event indices: {:?}", indices);
    println!("inter-snapshot gaps:    {:?}", gaps);
}
