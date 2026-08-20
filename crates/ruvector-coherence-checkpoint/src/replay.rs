//! Exact-replay reconstruction and verification.
//!
//! A checkpoint mechanism is only useful if recovery is lossless: replaying
//! the write log after the nearest snapshot must reproduce *exactly* the
//! state a fixed-interval or drift-triggered policy would have observed at
//! any target event, vector-for-vector.

use crate::checkpoint::{state_digest, CheckpointRun, Snapshot};
use crate::workload::WorkloadEvent;

/// The latest snapshot at or before `target_index`, or `None` if
/// `target_index` precedes every snapshot (shouldn't happen once event 0 is
/// always snapshotted).
pub fn nearest_snapshot_at_or_before(
    snapshots: &[Snapshot],
    target_index: usize,
) -> Option<&Snapshot> {
    snapshots
        .iter()
        .rev()
        .find(|s| s.event_index <= target_index)
}

/// Reconstruct store state at `target_index` from `snapshot` plus replaying
/// the intervening workload events.
pub fn reconstruct_state(
    snapshot: &Snapshot,
    events: &[WorkloadEvent],
    target_index: usize,
) -> Vec<Vec<f32>> {
    let mut state = snapshot.entries.clone();
    for ev in &events[snapshot.event_index + 1..=target_index] {
        state.push(ev.vector.clone());
    }
    state
}

/// Verify that replaying from the nearest snapshot reproduces the exact
/// ground-truth state at `target_index` — both by vector-for-vector equality
/// and by independently recomputed state digest.
pub fn verify_exact_replay(
    run: &CheckpointRun,
    events: &[WorkloadEvent],
    target_index: usize,
) -> bool {
    let snap = match nearest_snapshot_at_or_before(&run.snapshots, target_index) {
        Some(s) => s,
        None => return false,
    };
    let reconstructed = reconstruct_state(snap, events, target_index);
    let ground_truth: Vec<Vec<f32>> = events[..=target_index]
        .iter()
        .map(|e| e.vector.clone())
        .collect();

    if reconstructed != ground_truth {
        return false;
    }
    state_digest(&reconstructed) == state_digest(&ground_truth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::run_checkpoint_policy;
    use crate::policy::{DriftTriggered, DriftTriggeredCapped, FixedInterval};
    use crate::workload::{generate_workload, WorkloadConfig};

    fn workload() -> (WorkloadConfig, Vec<WorkloadEvent>) {
        let cfg = WorkloadConfig {
            dims: 6,
            n_events: 500,
            calm_phase_len: 40,
            burst_phase_len: 15,
            noise: 0.03,
        };
        let events = generate_workload(&cfg, 99);
        (cfg, events)
    }

    #[test]
    fn fixed_interval_replays_exactly_at_every_target() {
        let (cfg, events) = workload();
        let (run, _gate) = run_checkpoint_policy(&events, cfg.dims, FixedInterval { interval: 30 });
        for target in (0..events.len()).step_by(17) {
            assert!(
                verify_exact_replay(&run, &events, target),
                "fixed-interval replay diverged at event {target}"
            );
        }
    }

    #[test]
    fn drift_triggered_replays_exactly_at_every_target() {
        let (cfg, events) = workload();
        let (run, _gate) =
            run_checkpoint_policy(&events, cfg.dims, DriftTriggered { threshold: 0.05 });
        for target in (0..events.len()).step_by(17) {
            assert!(
                verify_exact_replay(&run, &events, target),
                "drift-triggered replay diverged at event {target}"
            );
        }
    }

    #[test]
    fn capped_drift_triggered_replays_exactly_at_every_target() {
        let (cfg, events) = workload();
        let (run, _gate) = run_checkpoint_policy(
            &events,
            cfg.dims,
            DriftTriggeredCapped {
                threshold: 0.05,
                max_interval: 60,
            },
        );
        for target in (0..events.len()).step_by(17) {
            assert!(
                verify_exact_replay(&run, &events, target),
                "capped drift-triggered replay diverged at event {target}"
            );
        }
    }

    #[test]
    fn corrupted_snapshot_entry_breaks_replay_equality() {
        let (cfg, events) = workload();
        let (mut run, _gate) =
            run_checkpoint_policy(&events, cfg.dims, FixedInterval { interval: 30 });
        run.snapshots[1].entries[0][0] += 1.0; // simulate a corrupted snapshot
        let target = run.snapshots[2].event_index - 1;
        assert!(!verify_exact_replay(&run, &events, target));
    }
}
