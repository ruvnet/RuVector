//! The transparency log and the per-subject witness chain.

mod common;

use common::{reader, Fixture, NOW};
use rvforge_registry::objects::{LogObjectType, Outcome, SubjectType, TrustLevel, WitnessEvent};
use rvforge_registry::RegistryErrorCode;
use std::fs;

#[test]
fn every_logged_object_has_a_verifiable_inclusion_proof() {
    let fx = Fixture::new();
    let first = fx.publish("example-agent", "1.0.0");
    let second = fx.publish("other-agent", "1.0.0");
    fx.registry
        .raise_trust_level(fx.trust_update(&first, TrustLevel::Tested), NOW)
        .unwrap();
    fx.registry
        .revoke(fx.revocation(&second, SubjectType::Release), NOW)
        .unwrap();

    let head = fx.registry.tree_head().unwrap();
    let entries = fx.registry.log_entries().unwrap();
    assert_eq!(head.tree_size as usize, entries.len());
    assert!(
        entries.len() >= 5,
        "publisher + 2 releases + trust + revocation"
    );

    for entry in &entries {
        let proof = fx.registry.inclusion_proof(&entry.object).unwrap();
        assert_eq!(proof.leaf_index, entry.index);
        assert!(
            proof.verify(&head.root_hash),
            "entry {} did not prove inclusion",
            entry.index
        );
    }
}

#[test]
fn a_proof_against_the_wrong_root_fails() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    let proof = fx.registry.inclusion_proof(&id).unwrap();

    assert!(proof.verify(&fx.registry.tree_head().unwrap().root_hash));
    assert!(!proof.verify(&format!("sha256:{}", "0".repeat(64))));
    assert!(!proof.verify("not-a-digest"));
}

#[test]
fn a_stale_proof_stops_verifying_once_the_log_grows() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    let stale = fx.registry.inclusion_proof(&id).unwrap();

    fx.publish("other-agent", "1.0.0");
    let new_head = fx.registry.tree_head().unwrap().root_hash;

    assert!(
        !stale.verify(&new_head),
        "an old root must not satisfy the new head"
    );
    assert!(fx.registry.inclusion_proof(&id).unwrap().verify(&new_head));
}

#[test]
fn a_tampered_log_entry_is_detected() {
    let fx = Fixture::new();
    fx.publish("example-agent", "1.0.0");
    fx.registry.log().verify_consistency().unwrap();

    let entries_path = fx.registry.root().join("log").join("entries.jsonl");
    let text = fs::read_to_string(&entries_path).unwrap();
    let tampered = text.replace(
        "\"timestamp\":\"2026-08-03T00:00:00Z\"",
        "\"timestamp\":\"2020-01-01T00:00:00Z\"",
    );
    assert_ne!(
        tampered, text,
        "the fixture timestamp should appear in the log"
    );
    fs::write(&entries_path, tampered).unwrap();

    let err = fx.registry.log().verify_consistency().unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::Log);
    assert!(err.to_string().contains("hashes to"), "{err}");
}

#[test]
fn a_removed_log_entry_is_detected() {
    let fx = Fixture::new();
    fx.publish("example-agent", "1.0.0");
    fx.publish("other-agent", "1.0.0");

    let entries_path = fx.registry.root().join("log").join("entries.jsonl");
    let text = fs::read_to_string(&entries_path).unwrap();
    let kept: Vec<&str> = text.lines().filter(|l| !l.is_empty()).skip(1).collect();
    fs::write(&entries_path, format!("{}\n", kept.join("\n"))).unwrap();

    let err = fx.registry.log_entries().unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::Log);
    assert!(err.to_string().contains("declares index"), "{err}");
}

#[test]
fn logging_the_same_object_twice_does_not_grow_the_tree() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    let head_before = fx.registry.tree_head().unwrap();

    let again = fx
        .registry
        .log()
        .append(&id, LogObjectType::Release, "2027-01-01T00:00:00Z")
        .unwrap();

    assert_eq!(again.timestamp, NOW, "the original entry is returned");
    assert_eq!(fx.registry.tree_head().unwrap(), head_before);
}

#[test]
fn objects_are_indexed_by_their_logged_kind() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    fx.registry
        .revoke(fx.revocation(&id, SubjectType::Release), NOW)
        .unwrap();

    assert_eq!(
        fx.registry
            .log()
            .objects_of_type(LogObjectType::Release)
            .unwrap(),
        vec![id]
    );
    assert_eq!(
        fx.registry
            .log()
            .objects_of_type(LogObjectType::Publisher)
            .unwrap(),
        vec![fx.publisher.publisher_id.clone()]
    );
    assert_eq!(
        fx.registry
            .log()
            .objects_of_type(LogObjectType::Revocation)
            .unwrap()
            .len(),
        1
    );
}

// --- witness receipts ---

#[test]
fn publish_verify_and_revoke_extend_one_chain_per_subject() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    fx.registry
        .record_verification(&id, Outcome::Pass, reader(), "signatures checked", NOW)
        .unwrap();
    fx.registry
        .revoke(fx.revocation(&id, SubjectType::Release), NOW)
        .unwrap();

    let receipts = fx.registry.receipts_for(&id).unwrap();
    assert_eq!(
        receipts.iter().map(|r| r.event).collect::<Vec<_>>(),
        vec![
            WitnessEvent::Publish,
            WitnessEvent::Verify,
            WitnessEvent::Revoke
        ]
    );
    assert_eq!(receipts[0].prev_receipt, None);
    assert_eq!(
        receipts[1].prev_receipt.as_deref(),
        Some(receipts[0].receipt_id.as_str())
    );
    assert_eq!(
        receipts[2].prev_receipt.as_deref(),
        Some(receipts[1].receipt_id.as_str())
    );
    fx.registry.verify_witness_chain(&id).unwrap();
}

#[test]
fn chains_of_different_subjects_do_not_interleave() {
    let fx = Fixture::new();
    let first = fx.publish("example-agent", "1.0.0");
    let second = fx.publish("other-agent", "1.0.0");

    for id in [&first, &second] {
        let receipts = fx.registry.receipts_for(id).unwrap();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].prev_receipt, None);
        assert_eq!(receipts[0].subject, *id);
        fx.registry.verify_witness_chain(id).unwrap();
    }
}

#[test]
fn a_broken_chain_link_is_detected() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    fx.registry
        .record_verification(&id, Outcome::Pass, reader(), "first check", NOW)
        .unwrap();
    fx.registry
        .record_verification(&id, Outcome::Pass, reader(), "second check", NOW)
        .unwrap();
    fx.registry.verify_witness_chain(&id).unwrap();

    // Drop the middle receipt from the index: the third now names a
    // predecessor that no longer precedes it.
    let receipts = fx.registry.receipts_for(&id).unwrap();
    let index = witness_index_path(&fx, &id);
    let kept: Vec<String> = receipts
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 1)
        .map(|(_, r)| r.receipt_id.clone())
        .collect();
    fs::write(&index, format!("{}\n", kept.join("\n"))).unwrap();

    let err = fx.registry.verify_witness_chain(&id).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::Log);
    assert!(err.to_string().contains("prevReceipt"), "{err}");
}

#[test]
fn receipts_are_unsigned_without_a_signer_and_signed_with_one() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    assert!(fx.registry.receipts_for(&id).unwrap()[0]
        .signatures
        .is_empty());

    let signing = std::sync::Arc::new(rvforge_registry::testkit::signing_key(0x5A));
    let signed_registry = fx.registry.clone().with_receipt_signer(signing.clone());
    let receipt = signed_registry
        .record_verification(&id, Outcome::Pass, reader(), "countersigned", NOW)
        .unwrap();

    assert_eq!(receipt.signatures.len(), 1);
    rvforge_registry::crypto::verify_over_id(
        &receipt.receipt_id,
        &receipt.signatures[0].sig,
        &rvforge_registry::Signer::verifying_key(signing.as_ref()),
    )
    .unwrap();

    // Signing does not change the id, so the chain is unaffected.
    signed_registry.verify_witness_chain(&id).unwrap();
}

#[test]
fn revocation_receipts_record_a_denied_outcome() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    let outcome = fx
        .registry
        .revoke(fx.revocation(&id, SubjectType::Release), NOW)
        .unwrap();

    assert_eq!(outcome.receipt.event, WitnessEvent::Revoke);
    assert_eq!(outcome.receipt.outcome, Outcome::Denied);
    assert!(outcome.receipt.evidence.details.contains("reads preserved"));
}

fn witness_index_path(fx: &Fixture, subject: &str) -> std::path::PathBuf {
    let key = rvf_hex(&rvforge_registry::merkle::sha256(subject.as_bytes()));
    fx.registry
        .root()
        .join("witness")
        .join(format!("{key}.jsonl"))
}

fn rvf_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
