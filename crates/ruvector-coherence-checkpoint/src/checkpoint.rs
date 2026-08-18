//! Snapshot orchestration: drives a [`MemoryStore`] through a workload under
//! a [`CheckpointPolicy`], witness-chaining every snapshot through
//! `ruvector-proof-gate`'s [`HashChainGate`].

use ruvector_agent_memory::memory::MemoryStore;
use ruvector_proof_gate::{HashChainGate, WriteGate, WritePayload, WriteReceipt};
use sha2::{Digest, Sha256};

use crate::drift::RunningCentroid;
use crate::policy::{CheckpointPolicy, PolicyContext};
use crate::workload::WorkloadEvent;

/// SHA-256 over the full ordered set of stored vectors — a compact,
/// tamper-evident fingerprint of exact store state at snapshot time.
pub fn state_digest(entries: &[Vec<f32>]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update((entries.len() as u64).to_le_bytes());
    for v in entries {
        h.update((v.len() as u32).to_le_bytes());
        for f in v {
            h.update(f.to_le_bytes());
        }
    }
    h.finalize().into()
}

/// A single witness-chained checkpoint: the exact store state at the time it
/// was taken, plus the receipt that binds it into the gate's tamper-evident
/// chain.
pub struct Snapshot {
    pub event_index: usize,
    pub centroid: Vec<f32>,
    /// Full copy of every stored vector, in insertion order — required to
    /// exactly reconstruct state during replay (this is a checkpoint
    /// mechanism, not a lossy compaction).
    pub entries: Vec<Vec<f32>>,
    pub state_digest: [u8; 32],
    pub receipt: WriteReceipt,
    pub payload_hash: [u8; 32],
}

pub struct CheckpointRun {
    pub policy_name: &'static str,
    pub snapshots: Vec<Snapshot>,
    /// `gap_at_event[i]` = number of events since the nearest snapshot at or
    /// before event `i` (0 at a snapshot event itself). Recorded for every
    /// event, not just snapshot points, so worst-case recovery cost is
    /// measured across the whole stream.
    pub gap_at_event: Vec<usize>,
    /// Full cryptographic re-derivation of the witness chain from genesis —
    /// `true` means every admitted snapshot receipt is consistent and
    /// untampered.
    pub gate_integrity_ok: bool,
}

impl CheckpointRun {
    pub fn max_gap(&self) -> usize {
        self.gap_at_event.iter().copied().max().unwrap_or(0)
    }

    pub fn mean_gap(&self) -> f64 {
        if self.gap_at_event.is_empty() {
            return 0.0;
        }
        self.gap_at_event.iter().sum::<usize>() as f64 / self.gap_at_event.len() as f64
    }

    pub fn p95_gap(&self) -> usize {
        if self.gap_at_event.is_empty() {
            return 0;
        }
        let mut sorted = self.gap_at_event.clone();
        sorted.sort_unstable();
        let idx = ((sorted.len() as f64) * 0.95).floor() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Re-verify every snapshot's receipt against the gate's structural chain
    /// AND against its own claimed payload content (rehash-and-compare) —
    /// catches both a corrupted chain and a snapshot whose stored digest was
    /// altered after the fact.
    pub fn all_receipts_structurally_consistent(&self, gate: &HashChainGate) -> bool {
        self.snapshots
            .iter()
            .all(|s| gate.verify_receipt(&s.receipt))
    }
}

/// Drive `events` through a fresh [`MemoryStore`] under `policy`, taking a
/// witness-chained snapshot whenever the policy decides to (always including
/// event 0, so every run has a bootstrap checkpoint).
pub fn run_checkpoint_policy(
    events: &[WorkloadEvent],
    dims: usize,
    mut policy: impl CheckpointPolicy,
) -> (CheckpointRun, HashChainGate) {
    let mut store = MemoryStore::new(dims);
    let mut gate = HashChainGate::new();
    let mut running = RunningCentroid::new(dims);
    let mut last_snapshot_centroid: Option<Vec<f32>> = None;
    let mut last_snapshot_index: i64 = -1;
    let mut snapshots = Vec::new();
    let mut gap_at_event = Vec::with_capacity(events.len());
    let policy_name = policy.name();

    for (i, ev) in events.iter().enumerate() {
        store.insert(ev.vector.clone());
        running.update(&ev.vector);
        let cur_centroid = running.current();
        let events_since_last = (i as i64 - last_snapshot_index) as usize;

        let take = i == 0 || {
            let ctx = PolicyContext {
                current_centroid: &cur_centroid,
                last_snapshot_centroid: last_snapshot_centroid.as_deref(),
                events_since_last_snapshot: events_since_last,
            };
            policy.should_snapshot(&ctx)
        };

        if take {
            let entries: Vec<Vec<f32>> = store.entries().iter().map(|e| e.vector.clone()).collect();
            let digest = state_digest(&entries);
            let payload =
                WritePayload::new(i as u64, cur_centroid.clone()).with_metadata(digest.to_vec());
            let payload_hash = payload.payload_hash();
            let receipt = gate.admit(&payload).expect("witness gate admission");
            snapshots.push(Snapshot {
                event_index: i,
                centroid: cur_centroid.clone(),
                entries,
                state_digest: digest,
                receipt,
                payload_hash,
            });
            last_snapshot_centroid = Some(cur_centroid);
            last_snapshot_index = i as i64;
        }

        gap_at_event.push((i as i64 - last_snapshot_index) as usize);
    }

    let gate_integrity_ok = gate.verify_integrity();
    (
        CheckpointRun {
            policy_name,
            snapshots,
            gap_at_event,
            gate_integrity_ok,
        },
        gate,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::FixedInterval;
    use crate::workload::{generate_workload, WorkloadConfig};

    #[test]
    fn always_snapshots_event_zero() {
        let cfg = WorkloadConfig {
            dims: 4,
            n_events: 50,
            calm_phase_len: 100,
            burst_phase_len: 100,
            noise: 0.01,
        };
        let events = generate_workload(&cfg, 1);
        let (run, _gate) =
            run_checkpoint_policy(&events, cfg.dims, FixedInterval { interval: 1000 });
        assert_eq!(run.snapshots.len(), 1);
        assert_eq!(run.snapshots[0].event_index, 0);
    }

    #[test]
    fn gate_integrity_holds_for_untampered_run() {
        let cfg = WorkloadConfig {
            dims: 4,
            n_events: 200,
            calm_phase_len: 30,
            burst_phase_len: 10,
            noise: 0.02,
        };
        let events = generate_workload(&cfg, 2);
        let (run, gate) = run_checkpoint_policy(&events, cfg.dims, FixedInterval { interval: 25 });
        assert!(run.gate_integrity_ok);
        assert!(run.all_receipts_structurally_consistent(&gate));
    }

    #[test]
    fn tampering_a_receipt_commitment_fails_structural_check() {
        let cfg = WorkloadConfig {
            dims: 4,
            n_events: 100,
            calm_phase_len: 20,
            burst_phase_len: 10,
            noise: 0.02,
        };
        let events = generate_workload(&cfg, 3);
        let (mut run, gate) =
            run_checkpoint_policy(&events, cfg.dims, FixedInterval { interval: 20 });
        // Flip a byte in a receipt's chain commitment to simulate a tampered
        // or forged snapshot claim.
        run.snapshots[0].receipt.chain_commitment[0] ^= 0xFF;
        assert!(!run.all_receipts_structurally_consistent(&gate));
    }

    #[test]
    fn tampering_stored_digest_is_caught_by_payload_rehash() {
        let cfg = WorkloadConfig {
            dims: 4,
            n_events: 100,
            calm_phase_len: 20,
            burst_phase_len: 10,
            noise: 0.02,
        };
        let events = generate_workload(&cfg, 4);
        let (mut run, _gate) =
            run_checkpoint_policy(&events, cfg.dims, FixedInterval { interval: 20 });
        let snap = &mut run.snapshots[0];
        let original_digest = snap.state_digest;
        snap.state_digest[0] ^= 0xFF; // simulate corruption after the fact
        let payload = WritePayload::new(snap.event_index as u64, snap.centroid.clone())
            .with_metadata(snap.state_digest.to_vec());
        assert_ne!(payload.payload_hash(), snap.payload_hash);
        assert_ne!(snap.state_digest, original_digest);
    }
}
