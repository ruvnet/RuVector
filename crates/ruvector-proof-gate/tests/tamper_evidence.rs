//! Tamper-evidence guarantees for the HashChain write gate (productionizes #506).
//!
//! Proves the core property: the chain root is a cryptographic commitment to the
//! *entire ordered write log*, so ANY mutation, insertion, deletion, or reorder
//! of the writes yields a different root — and forged/foreign receipts are
//! rejected. This is the defense against silent memory poisoning (MemoryGraft).

use ruvector_proof_gate::{HashChainGate, WriteGate, WritePayload};

fn payload(id: u64, fill: f32) -> WritePayload {
    WritePayload::new(id, vec![fill, fill * 2.0, 1.0, -fill])
}

fn base_log() -> Vec<WritePayload> {
    (0..6).map(|i| payload(i, i as f32 + 1.0)).collect()
}

fn root_of(log: &[WritePayload]) -> [u8; 32] {
    let mut g = HashChainGate::new();
    for p in log {
        g.admit(p).expect("admit");
    }
    g.chain_root()
}

#[test]
fn deterministic_and_each_write_advances_root() {
    let log = base_log();
    assert_eq!(root_of(&log), root_of(&log), "same log must yield the same root");

    // The root must change after every admitted write (no silent no-ops).
    let mut g = HashChainGate::new();
    let mut prev = g.chain_root();
    for p in &log {
        g.admit(p).unwrap();
        let now = g.chain_root();
        assert_ne!(now, prev, "root must advance on every write");
        prev = now;
    }
    assert_eq!(g.len(), log.len());
}

#[test]
fn mutation_changes_root() {
    let base = base_log();
    let root = root_of(&base);
    // Tamper a single middle write's vector.
    let mut tampered = base.clone();
    tampered[2] = payload(2, 999.0);
    assert_ne!(root_of(&tampered), root, "mutating any write must change the root");
}

#[test]
fn insertion_deletion_reorder_change_root() {
    let base = base_log();
    let root = root_of(&base);

    let mut inserted = base.clone();
    inserted.insert(3, payload(99, 7.0));
    assert_ne!(root_of(&inserted), root, "insertion must change the root");

    let mut deleted = base.clone();
    deleted.remove(2);
    assert_ne!(root_of(&deleted), root, "deletion must change the root");

    let mut reordered = base.clone();
    reordered.swap(1, 4);
    assert_ne!(root_of(&reordered), root, "reorder must change the root");
}

#[test]
fn receipts_verify_then_forgeries_rejected() {
    let mut g = HashChainGate::new();
    let receipts: Vec<_> = base_log().iter().map(|p| g.admit(p).unwrap()).collect();

    // Every genuine receipt verifies against the gate.
    for r in &receipts {
        assert!(g.verify_receipt(r), "genuine receipt must verify");
    }

    // Forge: flip a byte of the chain commitment → rejected.
    let mut forged = receipts[2].clone();
    forged.chain_commitment[0] ^= 0xFF;
    assert!(!g.verify_receipt(&forged), "forged commitment must be rejected");

    // Out-of-range sequence → rejected (no panic).
    let mut oob = receipts[0].clone();
    oob.sequence = 9_999;
    assert!(!g.verify_receipt(&oob), "out-of-range receipt must be rejected");
}

#[test]
fn foreign_receipt_rejected() {
    // A receipt minted by a different chain (different write history) must not
    // verify against an unrelated gate at the same sequence position.
    let mut a = HashChainGate::new();
    let mut b = HashChainGate::new();
    let ra = a.admit(&payload(0, 1.0)).unwrap();
    let _rb = b.admit(&payload(0, 2.0)).unwrap(); // different payload → different commitment
    assert!(!b.verify_receipt(&ra), "a receipt from chain A must not verify on chain B");
}
