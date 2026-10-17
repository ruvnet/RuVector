//! Signed structural forgetting receipts: binding [`crate::graph_forget`]'s
//! mincut-boundary decision signal (ADR-345, 2026-09-05) into the eviction
//! witness `aux` field so an auditor holding only the signed witness log —
//! no `MemoryStore`, no raw vectors — can later answer "how many of the
//! entries this compaction pass evicted were structurally load-bearing
//! (mincut-boundary) at decision time, and under which protection mode",
//! and detect after the fact if that answer was tampered with.
//!
//! [`crate::witnessed_compaction::compact_witnessed`] already certifies
//! *which* ids were evicted (ADR-345). It leaves `aux = 0` — the eviction
//! witness says nothing about *why* a mincut-gated policy made the call it
//! did. [`crate::witness_signing::SignedWitnessSink`] (ADR-347) already
//! signs whatever bytes a `WitnessSink` is handed. Composing them is free;
//! what is missing, and what this module adds, is a structural justification
//! *payload* worth signing in the first place: [`encode_structural_aux`] /
//! [`decode_structural_aux`] pack the boundary flag, protection mode, and
//! scalar [`crate::compaction::weighted_importance`] score of each evicted
//! entry into the unused 64-bit `aux` field.
//!
//! [`compact_witnessed_structural`] takes an already-computed
//! [`crate::graph_forget::ForgetDecision`] (from
//! [`MincutGatedForgetting::decide`]) rather than calling `decide` itself.
//! This is a deliberate API choice, not an oversight: `RuVectorGraphAnalyzer::partition()`
//! is documented (2026-09-05/09-11 nightlies) as non-deterministic across
//! repeated calls on *identical* input, so two independent `decide()` calls
//! for what is meant to be "the same decision" can legitimately disagree —
//! each individually a valid minimum cut, but not necessarily the same one.
//! Requiring the caller to compute `decide()` exactly once and pass the
//! result in makes "one decision, one mincut call" a property of the type
//! signature rather than an invariant this module would otherwise have to
//! take on faith. It also lets a caller (or a test) inspect the same
//! `ForgetDecision` used to produce the records, instead of needing a
//! second, potentially-disagreeing call to recover "ground truth."
//!
//! [`audit_structural_evictions`] is the auditor side: a pure function over
//! `&[LedgerWitnessRecord]` that needs no vectors and no `MemoryStore`.
//!
//! ## Why this needs signing, not just the existing FNV-1a chain
//!
//! `aux` is included in [`crate::ops::LedgerWitnessRecord::to_bytes`] (bytes
//! 56-63) and therefore in [`crate::ops::LedgerWitnessRecord::chain_hash`],
//! but **excluded** from [`crate::ops::LedgerWitnessRecord::compute_record_hash`]
//! (the first-48-bytes ADR-134 preimage). A *naive* tamper — flip an aux bit
//! and touch nothing else — still breaks [`crate::ops::MemoryWitnessLog::verify_chain`],
//! because the next record's `prev_hash` (or the log's `committed_head`, for
//! the tail) was computed from the original 64 bytes. A *diligent* tamper —
//! flip the aux bit, then forward-recompute every downstream `chain_hash`
//! and the head commitment to stay self-consistent — passes the unsigned
//! walk, exactly as the 2026-09-16 witness-signer nightly's "diligent
//! forgery" construction demonstrated for generic records. This module's
//! tests repeat that exact construction against structural `aux` payloads
//! specifically, and confirm [`crate::witness_signing::SignedWitnessSink`]
//! (unmodified, reused as-is) rejects it while the unsigned walk alone does
//! not — see `tests::diligent_forgery_hides_boundary_eviction_from_unsigned_walk`
//! and `tests::signed_chain_rejects_diligent_structural_forgery`.

use crate::graph_forget::{ForgetDecision, ForgetMode};
use crate::memory::{MemoryEntry, MemoryStore};
use crate::ops::{
    action_kind, fnv1a, pack_flags, EvidenceGrade, LedgerError, LedgerWitnessRecord, WitnessSink,
};
use crate::witnessed_compaction::EvictionWitnessChain;
use std::collections::HashSet;

/// Bit 0 of `aux`: this evicted entry had at least one neighbor edge
/// crossing the mincut partition at decision time (it was "structurally
/// load-bearing").
pub const AUX_BOUNDARY_BIT: u64 = 1 << 0;
/// Bit 1 of `aux`: the decision was made under [`crate::graph_forget::ForgetMode::Hard`]
/// (budget-reserving) rather than `Soft` (additive-bonus).
pub const AUX_HARD_MODE_BIT: u64 = 1 << 1;
/// Bits 8-39 of `aux`: the evicted entry's scalar
/// [`crate::compaction::weighted_importance`] score, as `f32::to_bits`.
pub const AUX_SCORE_SHIFT: u32 = 8;

/// Pack a structural forgetting decision into a 64-bit `aux` value.
/// Bits 2-7 and 40-63 are reserved (zero) for future structural fields
/// (e.g. the corpus size or the number of crossing edges) without changing
/// this encoding's bit 0/1/8-39 contract.
pub fn encode_structural_aux(boundary: bool, hard_mode: bool, scalar_score: f32) -> u64 {
    let mut aux = 0u64;
    if boundary {
        aux |= AUX_BOUNDARY_BIT;
    }
    if hard_mode {
        aux |= AUX_HARD_MODE_BIT;
    }
    aux |= (scalar_score.to_bits() as u64) << AUX_SCORE_SHIFT;
    aux
}

/// Inverse of [`encode_structural_aux`]: `(was_boundary, was_hard_mode, scalar_score)`.
pub fn decode_structural_aux(aux: u64) -> (bool, bool, f32) {
    let boundary = aux & AUX_BOUNDARY_BIT != 0;
    let hard_mode = aux & AUX_HARD_MODE_BIT != 0;
    let score_bits = ((aux >> AUX_SCORE_SHIFT) & 0xFFFF_FFFF) as u32;
    (boundary, hard_mode, f32::from_bits(score_bits))
}

/// Apply an already-computed [`ForgetDecision`] to `store`, emitting one
/// chained, structurally-annotated [`LedgerWitnessRecord`] per evicted entry
/// to `sink` *before* mutating the store — identical "no witness, no
/// mutation" ordering to [`crate::witnessed_compaction::compact_witnessed`],
/// which this function otherwise mirrors field-for-field except for `aux`.
///
/// `decision` must have been computed from `store`'s *current* entries
/// (i.e. via `policy.decide(store.entries(), ..)` with no mutation of
/// `store` in between) — this function trusts that precondition and does
/// not re-derive or re-check it; see the module docs for why this function
/// takes a decision rather than a policy + recomputing one itself.
///
/// # Panics
/// Panics if `decision.survivors.len() > store.len()`, or if any survivor
/// or evicted index is out of bounds for `store` — both indicate `decision`
/// was not computed from this exact `store` state.
#[allow(clippy::too_many_arguments)]
pub fn compact_witnessed_structural(
    store: &mut MemoryStore,
    decision: &ForgetDecision,
    hard_mode: bool,
    actor_id: &str,
    now_ns: u64,
    chain: &mut EvictionWitnessChain,
    sink: &mut dyn WitnessSink,
) -> Result<Vec<LedgerWitnessRecord>, LedgerError> {
    assert!(
        decision.survivors.len() <= store.len(),
        "decision.survivors ({}) must be <= store.len() ({}) -- was `decision` computed from a \
         different store state?",
        decision.survivors.len(),
        store.len()
    );

    let entries: Vec<MemoryEntry> = store.entries().to_vec();
    let survivor_set: HashSet<usize> = decision.survivors.iter().copied().collect();

    let mut evicted: Vec<(u64, bool, f32)> = (0..entries.len())
        .filter(|i| !survivor_set.contains(i))
        .map(|i| {
            (
                entries[i].id,
                decision.boundary.contains(&i),
                decision.scalar[i],
            )
        })
        .collect();
    evicted.sort_unstable_by_key(|&(id, _, _)| id);

    let (start_seq, start_hash) = chain.head();
    let mut records: Vec<LedgerWitnessRecord> = Vec::with_capacity(evicted.len());
    let mut prev = start_hash;
    for (offset, &(id, is_boundary, score)) in evicted.iter().enumerate() {
        let mut rec = LedgerWitnessRecord {
            sequence: start_seq + offset as u64,
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
            aux: encode_structural_aux(is_boundary, hard_mode, score),
            evidence_grade: EvidenceGrade::Recomputed,
        };
        rec.record_hash = rec.compute_record_hash();
        prev = rec.chain_hash();
        records.push(rec);
    }

    // Witness first: no witness, no mutation.
    sink.emit_batch(&records)?;

    *chain = EvictionWitnessChain::resume(start_seq + records.len() as u64, prev);

    let mut survivors: Vec<MemoryEntry> = decision
        .survivors
        .iter()
        .map(|&i| entries[i].clone())
        .collect();
    survivors.sort_unstable_by_key(|e| e.id);
    store.replace_entries(survivors);

    Ok(records)
}

/// Convenience: `hard_mode` flag derived from a policy's [`ForgetMode`].
pub fn is_hard_mode(mode: ForgetMode) -> bool {
    matches!(mode, ForgetMode::Hard)
}

/// Tally of structural eviction statistics recovered from a signed or
/// unsigned eviction witness log, with no access to the original vectors or
/// `MemoryStore` — see module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuralAuditReport {
    pub evicted: usize,
    pub boundary_evicted: usize,
    pub hard_mode_evicted: usize,
}

/// Recompute [`StructuralAuditReport`] from `LEDGER_COMPACT_EVICT` records'
/// `aux` fields alone. Callers who need the result to be trustworthy (not
/// just internally consistent) must verify the records' authenticity first
/// — via [`crate::witness_signing::verify_signed_chain`] if the sink that
/// produced them was a [`crate::witness_signing::SignedWitnessSink`], or at
/// minimum [`crate::ops::MemoryWitnessLog::verify_chain`] otherwise — this
/// function itself performs no verification.
pub fn audit_structural_evictions(records: &[LedgerWitnessRecord]) -> StructuralAuditReport {
    let mut report = StructuralAuditReport::default();
    for r in records {
        if r.action_kind != action_kind::LEDGER_COMPACT_EVICT {
            continue;
        }
        let (boundary, hard_mode, _score) = decode_structural_aux(r.aux);
        report.evicted += 1;
        if boundary {
            report.boundary_evicted += 1;
        }
        if hard_mode {
            report.hard_mode_evicted += 1;
        }
    }
    report
}

#[cfg(test)]
#[path = "structural_witness_tests.rs"]
mod tests;
