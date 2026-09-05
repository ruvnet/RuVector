//! Witnessed compaction: certify every evicted entry with a chained
//! [`LedgerWitnessRecord`] (ADR-341, PIR nightly 2026-09-05).
//!
//! [`crate::compaction`] and [`crate::ledger`] cover the two ends of a
//! memory's lifecycle — proof-gated *admission* (`ledger::TransactionalLedger`)
//! and *retrieval* certification lives in the sibling `ruvector-retrieval-receipt`
//! crate — but nothing certifies *deletion*. A compaction pass silently drops
//! entries; there is no way to later prove which ids were evicted, when, by
//! which policy, or that the recorded eviction set has not been tampered
//! with after the fact. This module closes that gap using the same ADR-134
//! witness-record machinery [`crate::ledger`] already uses for admission,
//! rather than introducing a new hash or chaining scheme.
//!
//! Tamper-evidence scope is identical to [`crate::ops`]: keyless FNV-1a
//! chaining detects accidental corruption and naive edits, not a
//! log-writing adversary (see the `ops` module docs).

use crate::compaction::CompactionPolicy;
use crate::memory::{MemoryEntry, MemoryStore};
use crate::ops::{
    action_kind, fnv1a, pack_flags, EvidenceGrade, LedgerError, LedgerWitnessRecord, WitnessSink,
};
use std::collections::HashSet;

/// Running chain cursor for eviction witness records, mirroring the
/// `witness_seq` / `last_witness_hash` pattern in
/// [`crate::ledger::TransactionalLedger`].
#[derive(Debug, Clone, Copy, Default)]
pub struct EvictionWitnessChain {
    sequence: u64,
    last_hash: u64,
}

impl EvictionWitnessChain {
    /// Start a new chain (genesis `prev_hash` = 0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Resume an existing chain at the given cursor (e.g. a `MemoryWitnessLog`'s
    /// [`crate::ops::MemoryWitnessLog::head_commitment`]).
    pub fn resume(sequence: u64, last_hash: u64) -> Self {
        Self {
            sequence,
            last_hash,
        }
    }

    /// Current `(sequence, last_hash)` cursor.
    pub fn head(&self) -> (u64, u64) {
        (self.sequence, self.last_hash)
    }
}

/// Compact `store` using `policy`, emitting one chained [`LedgerWitnessRecord`]
/// per evicted entry to `sink` *before* mutating the store — "no witness, no
/// mutation", the same invariant [`crate::ledger`] enforces for admission.
///
/// Evicted ids are witnessed in ascending order for determinism. On success
/// `chain` and `store` both advance and the emitted records are returned. On
/// sink rejection neither is mutated.
///
/// # Panics
/// Panics if `target_size > store.len()` (same contract as [`crate::compact`]).
#[allow(clippy::too_many_arguments)]
pub fn compact_witnessed(
    store: &mut MemoryStore,
    policy: &dyn CompactionPolicy,
    target_size: usize,
    context_window: &[Vec<f32>],
    actor_id: &str,
    now_ns: u64,
    chain: &mut EvictionWitnessChain,
    sink: &mut dyn WitnessSink,
) -> Result<Vec<LedgerWitnessRecord>, LedgerError> {
    assert!(
        target_size <= store.len(),
        "target_size ({}) must be <= store.len() ({})",
        target_size,
        store.len()
    );

    let entries: &[MemoryEntry] = store.entries();
    let survivor_indices = policy.select_survivors(entries, target_size, context_window);
    let survivor_set: HashSet<usize> = survivor_indices.iter().copied().collect();

    let mut evicted_ids: Vec<u64> = (0..entries.len())
        .filter(|i| !survivor_set.contains(i))
        .map(|i| entries[i].id)
        .collect();
    evicted_ids.sort_unstable();

    let mut records: Vec<LedgerWitnessRecord> = Vec::with_capacity(evicted_ids.len());
    let mut prev = chain.last_hash;
    for (offset, &id) in evicted_ids.iter().enumerate() {
        let mut rec = LedgerWitnessRecord {
            sequence: chain.sequence + offset as u64,
            timestamp_ns: now_ns,
            action_kind: action_kind::LEDGER_COMPACT_EVICT,
            proof_tier: 0,
            flags: pack_flags(None, None, EvidenceGrade::Recomputed),
            actor_partition_id: (fnv1a(actor_id.as_bytes()) & 0xFFFF_FFFF) as u32,
            target_object_id: id as u32,
            capability_hash: 0,
            payload: fnv1a(&id.to_le_bytes()),
            prev_hash: prev,
            record_hash: 0,
            aux: 0,
            evidence_grade: EvidenceGrade::Recomputed,
        };
        rec.record_hash = rec.compute_record_hash();
        prev = rec.chain_hash();
        records.push(rec);
    }

    // Witness first: no witness, no mutation.
    sink.emit_batch(&records)?;

    chain.sequence += records.len() as u64;
    chain.last_hash = prev;

    let mut survivors: Vec<MemoryEntry> = survivor_indices
        .into_iter()
        .map(|i| entries[i].clone())
        .collect();
    survivors.sort_unstable_by_key(|e| e.id);
    store.replace_entries(survivors);

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::LruPolicy;
    use crate::ops::MemoryWitnessLog;

    fn make_store(n: usize) -> MemoryStore {
        let mut store = MemoryStore::new(2);
        for i in 0..n {
            store.insert(vec![i as f32, 0.0]);
        }
        store
    }

    #[test]
    fn witnesses_every_evicted_entry_exactly_once() {
        let mut store = make_store(10);
        let mut chain = EvictionWitnessChain::new();
        let mut log = MemoryWitnessLog::default();

        let records = compact_witnessed(
            &mut store,
            &LruPolicy,
            4,
            &[],
            "bench",
            1_000,
            &mut chain,
            &mut log,
        )
        .unwrap();

        assert_eq!(records.len(), 6, "10 - 4 = 6 evictions");
        assert_eq!(store.len(), 4);
        assert!(log.verify_chain(), "freshly emitted chain must verify");

        let evicted_ids: HashSet<u32> = records.iter().map(|r| r.target_object_id).collect();
        assert_eq!(evicted_ids.len(), 6, "no duplicate eviction witnesses");
    }

    #[test]
    fn tamper_is_detected() {
        let mut store = make_store(10);
        let mut chain = EvictionWitnessChain::new();
        let mut log = MemoryWitnessLog::default();
        compact_witnessed(
            &mut store,
            &LruPolicy,
            4,
            &[],
            "bench",
            1_000,
            &mut chain,
            &mut log,
        )
        .unwrap();
        assert!(log.verify_chain());

        // Flip one bit in a non-tail record's payload; the chain must break.
        log.records[0].payload ^= 1;
        assert!(!log.verify_chain(), "single-bit tamper must be detected");
    }

    #[test]
    fn chain_resumes_across_multiple_batches() {
        let mut store = make_store(20);
        let mut chain = EvictionWitnessChain::new();
        let mut log = MemoryWitnessLog::default();

        compact_witnessed(
            &mut store,
            &LruPolicy,
            15,
            &[],
            "bench",
            1_000,
            &mut chain,
            &mut log,
        )
        .unwrap();
        compact_witnessed(
            &mut store,
            &LruPolicy,
            10,
            &[],
            "bench",
            2_000,
            &mut chain,
            &mut log,
        )
        .unwrap();

        assert_eq!(log.records.len(), 10, "5 + 5 evictions across two batches");
        assert!(log.verify_chain(), "cross-batch chain must verify");
        assert_eq!(chain.head(), log.head_commitment());
    }
}
