use super::*;
use crate::compaction::CoherenceWeights as Weights;
use crate::graph_forget::{ForgetDecision, ForgetMode, MincutGatedForgetting};
use crate::ops::MemoryWitnessLog;
use crate::witness_signing::{
    verify_signed_chain, SignedAnchor, SignedWitnessSink, SigningStrategy,
};
use ed25519_dalek::SigningKey;

fn normalize3(v: [f32; 3]) -> Vec<f32> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    vec![v[0] / n, v[1] / n, v[2] / n]
}

/// Same bridge construction as `graph_forget`'s unit tests: two 9-member
/// clusters joined only through a single degree-2 bridge vertex. Plain
/// members get `access_count = 1`; the bridge and both gateways get
/// `access_count = 0` (identical timestamps, empty context), making them
/// the tied-lowest group under plain `weighted_importance` -- exactly what
/// a scalar-only policy would evict first, and what `MincutGatedForgetting`
/// must instead label with a structural signal.
///
/// Note on this topology's minimum cut: as `graph_forget.rs`'s own doc
/// comment on `boundary_indices` documents, this graph has *two* equal-cost
/// minimum cuts (either gateway, paired with the bridge, can be the side
/// that keeps only one crossing edge), not one unique cut isolating the
/// bridge alone. `RuVectorGraphAnalyzer::partition()` is also documented as
/// non-deterministic across repeated calls. Consequently these tests never
/// assert *which* specific vertex set a call returns -- only (a) that the
/// bridge is *always* part of whichever boundary set a given call finds,
/// and (b) that whatever `ForgetDecision` a single `decide()` call
/// produces is exactly what the resulting signed/unsigned witness records
/// report back under audit. That second property -- not "bridge is always
/// alone on the boundary" -- is this module's actual correctness claim.
fn bridge_store(target_bridge_idx_out: &mut usize) -> MemoryStore {
    let mut entries: Vec<MemoryEntry> = Vec::new();
    let mut id = 0u64;
    let mut push = |v: Vec<f32>, access_count: u64, entries: &mut Vec<MemoryEntry>| {
        let mut e = MemoryEntry::new(id, v, 0);
        e.access_count = access_count;
        entries.push(e);
        id += 1;
    };
    for axis in 0..2 {
        let plain = if axis == 0 {
            vec![1.0, 0.0, 0.0]
        } else {
            vec![0.0, 1.0, 0.0]
        };
        let gateway = if axis == 0 {
            normalize3([1.0, 0.0, 0.5])
        } else {
            normalize3([0.0, 1.0, 0.5])
        };
        for _ in 0..8 {
            push(plain.clone(), 1, &mut entries);
        }
        push(gateway, 0, &mut entries);
    }
    push(vec![0.0, 0.0, 1.0], 0, &mut entries); // bridge
    *target_bridge_idx_out = entries.len() - 1;

    let mut store = MemoryStore::new(3);
    store.replace_entries(entries);
    store
}

/// Compute `policy.decide(..)` exactly once against `store`'s current
/// entries, apply it via `compact_witnessed_structural`, and hand back both
/// the resulting records AND the `ForgetDecision` that produced them -- so
/// a test's "ground truth" is always the same decision the function under
/// test actually consumed, never a second, independently-nondeterministic
/// `decide()` call.
fn decide_and_apply(
    store: &mut MemoryStore,
    policy: &MincutGatedForgetting,
    target: usize,
    now_ns: u64,
    chain: &mut EvictionWitnessChain,
    sink: &mut dyn WitnessSink,
) -> (Vec<LedgerWitnessRecord>, ForgetDecision) {
    let entries = store.entries().to_vec();
    let decision = policy.decide(&entries, target, &[]);
    let records = compact_witnessed_structural(
        store,
        &decision,
        is_hard_mode(policy.mode),
        "test",
        now_ns,
        chain,
        sink,
    )
    .expect("structural compaction succeeds");
    (records, decision)
}

/// Ground truth derived from a `ForgetDecision` alone: the stable *ids* of
/// entries that are both evicted and boundary-flagged.
fn ground_truth_boundary_ids(entries: &[MemoryEntry], decision: &ForgetDecision) -> HashSet<u64> {
    let survivor_set: HashSet<usize> = decision.survivors.iter().copied().collect();
    (0..entries.len())
        .filter(|i| !survivor_set.contains(i) && decision.boundary.contains(i))
        .map(|i| entries[i].id)
        .collect()
}

#[test]
fn aux_encoding_round_trips() {
    for &(b, h, s) in &[
        (true, true, 1.5f32),
        (false, false, 0.0),
        (true, false, -3.25),
        (false, true, 1234.5),
    ] {
        let aux = encode_structural_aux(b, h, s);
        assert_eq!(decode_structural_aux(aux), (b, h, s));
    }
}

#[test]
fn reserved_bits_are_zero() {
    // Bits 2-7 and 40-63 must stay zero so a future extension can use them
    // without silently colliding with today's encoding.
    let aux = encode_structural_aux(true, true, f32::from_bits(0xFFFF_FFFF));
    assert_eq!(aux & 0b1111_1100, 0, "bits 2-7 reserved");
    assert_eq!(aux >> 40, 0, "bits 40-63 reserved");
}

#[test]
fn structural_audit_matches_ground_truth_when_bridge_is_evicted() {
    let mut bridge_idx = 0usize;
    let mut store = bridge_store(&mut bridge_idx);
    let total = store.len();
    let entries_snapshot = store.entries().to_vec();

    // No structural bonus/protection: a plain scalar tie-break will include
    // the bridge (and the two gateways, all access_count = 0, tied lowest)
    // among the evicted set once we ask for fewer survivors than the
    // "plain member" count.
    let policy = MincutGatedForgetting {
        weights: Weights::default(),
        mode: ForgetMode::Soft,
        k_neighbors: 8,
        min_similarity: 0.05,
        structural_bonus: 0.0,
        protect_fraction: 0.0,
        mincut_trials: 10,
    };

    let mut chain = EvictionWitnessChain::new();
    let mut log = MemoryWitnessLog::default();
    let target = total - 3; // forces exactly the tied-lowest trio out
    let (records, decision) =
        decide_and_apply(&mut store, &policy, target, 1_000, &mut chain, &mut log);

    assert!(log.verify_chain());
    assert_eq!(records.len(), 3);

    let bridge_evicted = records
        .iter()
        .any(|r| r.target_object_id == bridge_idx as u32);
    assert!(
        bridge_evicted,
        "this policy configuration must evict the tied-lowest trio including the bridge"
    );

    let report = audit_structural_evictions(&records);
    assert_eq!(report.evicted, 3);
    assert!(
        decision.boundary.contains(&bridge_idx),
        "the bridge must be on the boundary side of whichever minimum cut this decide() call \
         found (it always touches 2 crossing edges in the worst case; see bridge_store's doc)"
    );
    assert_eq!(
        report.hard_mode_evicted, 0,
        "Soft mode must never set the hard-mode bit"
    );

    // The real correctness claim: the audit recovered from the witness
    // records alone must exactly match the SAME ForgetDecision's boundary
    // set (not a second, independently-recomputed one).
    let ground_truth = ground_truth_boundary_ids(&entries_snapshot, &decision);
    let flagged_boundary: HashSet<u64> = records
        .iter()
        .filter(|r| decode_structural_aux(r.aux).0)
        .map(|r| r.target_object_id as u64)
        .collect();
    assert_eq!(flagged_boundary, ground_truth);
    assert!(
        ground_truth.contains(&(bridge_idx as u64)),
        "the bridge's id must be in ground truth"
    );
}

#[test]
fn structural_audit_reports_zero_boundary_when_bridge_survives() {
    let mut bridge_idx = 0usize;
    let mut store = bridge_store(&mut bridge_idx);
    let total = store.len();

    let policy = MincutGatedForgetting::soft(Weights::default(), 1.0);
    let mut chain = EvictionWitnessChain::new();
    let mut log = MemoryWitnessLog::default();
    // Evict only plain members (none of the trio); high structural_bonus
    // keeps the bridge (and the two gateways) at the top of the ranking.
    let target = total - 2;
    let (records, _decision) =
        decide_and_apply(&mut store, &policy, target, 1_000, &mut chain, &mut log);
    assert!(log.verify_chain());

    let bridge_evicted = records
        .iter()
        .any(|r| r.target_object_id == bridge_idx as u32);
    assert!(
        !bridge_evicted,
        "high structural_bonus must keep the bridge"
    );

    let report = audit_structural_evictions(&records);
    assert_eq!(report.boundary_evicted, 0);
}

#[test]
fn hard_mode_bit_is_set_for_every_record_under_hard_mode() {
    let mut bridge_idx = 0usize;
    let mut store = bridge_store(&mut bridge_idx);
    let total = store.len();

    let policy = MincutGatedForgetting::hard(Weights::default(), 0.3);
    let mut chain = EvictionWitnessChain::new();
    let mut log = MemoryWitnessLog::default();
    let (records, _decision) =
        decide_and_apply(&mut store, &policy, total - 3, 1_000, &mut chain, &mut log);

    let report = audit_structural_evictions(&records);
    assert_eq!(
        report.hard_mode_evicted, report.evicted,
        "every record must carry the Hard-mode bit"
    );
}

#[test]
fn chain_resumes_across_multiple_structural_batches() {
    let mut bridge_idx = 0usize;
    let mut store = bridge_store(&mut bridge_idx);
    let policy = MincutGatedForgetting::soft(Weights::default(), 1.0);
    let mut chain = EvictionWitnessChain::new();
    let mut log = MemoryWitnessLog::default();

    decide_and_apply(&mut store, &policy, 15, 1_000, &mut chain, &mut log);
    decide_and_apply(&mut store, &policy, 10, 2_000, &mut chain, &mut log);

    assert!(log.verify_chain());
    assert_eq!(chain.head(), log.head_commitment());
}

/// Naive aux tamper: flip the boundary bit on the evicted bridge's record,
/// touch nothing else. The unsigned chain walk must catch it, because
/// `aux` is covered by `chain_hash` (the full 64 bytes), which the next
/// record's `prev_hash` (or the log's head commitment, for the tail) was
/// computed from.
#[test]
fn naive_aux_tamper_is_caught_by_unsigned_chain_walk() {
    let mut bridge_idx = 0usize;
    let mut store = bridge_store(&mut bridge_idx);
    let total = store.len();
    let policy = MincutGatedForgetting {
        weights: Weights::default(),
        mode: ForgetMode::Soft,
        k_neighbors: 8,
        min_similarity: 0.05,
        structural_bonus: 0.0,
        protect_fraction: 0.0,
        mincut_trials: 10,
    };
    let mut chain = EvictionWitnessChain::new();
    let mut log = MemoryWitnessLog::default();
    decide_and_apply(&mut store, &policy, total - 3, 1_000, &mut chain, &mut log);
    assert!(log.verify_chain());

    let target = log
        .records
        .iter()
        .position(|r| r.target_object_id == bridge_idx as u32)
        .expect("bridge must be among the evicted records");
    log.records[target].aux ^= AUX_BOUNDARY_BIT; // hide the boundary flag

    assert!(
        !log.verify_chain(),
        "a naive aux tamper (no forward recompute) must break the unsigned chain walk"
    );
}

/// Diligent forgery: flip the boundary bit on the evicted bridge's record
/// AND forward-recompute every downstream `chain_hash`/`prev_hash` plus the
/// log's head commitment, so the unsigned walk alone cannot tell. This
/// mirrors the 2026-09-16 witness-signer nightly's construction, applied to
/// a structural payload. The unsigned walk is fooled by construction (that
/// IS the scenario); `SignedWitnessSink`/`verify_signed_chain` (reused
/// verbatim, no modification) must not be.
#[test]
fn diligent_forgery_hides_boundary_eviction_from_unsigned_walk_but_not_signed_chain() {
    let mut bridge_idx = 0usize;
    let mut store = bridge_store(&mut bridge_idx);
    let total = store.len();
    let policy = MincutGatedForgetting {
        weights: Weights::default(),
        mode: ForgetMode::Soft,
        k_neighbors: 8,
        min_similarity: 0.05,
        structural_bonus: 0.0,
        protect_fraction: 0.0,
        mincut_trials: 10,
    };

    let key = SigningKey::from_bytes(&[7u8; 32]);
    let mut chain = EvictionWitnessChain::new();
    let mut signed = SignedWitnessSink::new(
        MemoryWitnessLog::default(),
        key.clone(),
        SigningStrategy::BatchTail { batch_size: 64 },
    )
    .unwrap();

    decide_and_apply(
        &mut store,
        &policy,
        total - 3,
        1_000,
        &mut chain,
        &mut signed,
    );
    signed.seal();

    let target = signed
        .inner()
        .records
        .iter()
        .position(|r| r.target_object_id == bridge_idx as u32)
        .expect("bridge must be among the evicted records");

    // --- Build a diligently-forged *copy* of the inner log: flip the
    // bridge's boundary bit, then recompute record_hash/chain_hash forward
    // for every subsequent record, and fix up the head commitment, exactly
    // as a log-writing adversary with no signing key would. ---
    let original_boundary_flag = decode_structural_aux(signed.inner().records[target].aux).0;
    assert!(
        original_boundary_flag,
        "the bridge's own eviction record must honestly carry boundary=true before any forgery"
    );

    let mut forged = signed.inner().clone();
    forged.records[target].aux ^= AUX_BOUNDARY_BIT;
    forged.records[target].record_hash = forged.records[target].compute_record_hash();
    let mut prev = if target == 0 {
        0
    } else {
        forged.records[target - 1].chain_hash()
    };
    for r in forged.records.iter_mut().skip(target) {
        r.prev_hash = prev;
        r.record_hash = r.compute_record_hash();
        prev = r.chain_hash();
    }
    forged.committed_head = prev;
    // committed_count unchanged: same number of records.

    assert!(
        forged.verify_chain(),
        "the diligent forgery must pass the unsigned chain walk by construction \
         (confirms the crate's own documented FNV-1a tamper-evidence scope)"
    );
    assert!(
        !decode_structural_aux(forged.records[target].aux).0,
        "the forged record must now (falsely) claim the bridge was NOT boundary, or this isn't a \
         real forgery"
    );

    // Now check the SIGNED chain against the forged log using the spans
    // signed over the ORIGINAL (honest) content. `verify_signed_chain` must
    // reject it even though the unsigned walk above accepted it.
    let anchor = signed.anchor();
    let result = verify_signed_chain(&forged, signed.spans(), &signed.public_key(), &anchor);
    assert!(
        result.is_err(),
        "a diligently forward-recomputed structural forgery must be rejected by the signed chain, \
         got {:?}",
        result
    );

    // Sanity: the SAME spans against the honest (unforged) log must verify.
    let honest = verify_signed_chain(
        signed.inner(),
        signed.spans(),
        &signed.public_key(),
        &anchor,
    );
    assert!(honest.is_ok(), "the honest log must verify: {:?}", honest);
    assert!(
        audit_structural_evictions(&signed.inner().records).boundary_evicted >= 1,
        "the honest, verified log must show at least one boundary eviction (the bridge)"
    );

    let _ = SignedAnchor::genesis(); // exercise the re-export path in this module
}
