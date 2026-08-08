//! Inspection and verification through `rvf-forge-core`: real identity, real
//! pass/fail, never executing, and no capability card without verification.

use rvf_forge_core::inspect::HEADER_SIZE;
use rvf_forge_core::testkit::{minimal_container, signed_segment, unsigned_segment, TestKeypair};
use rvf_types::SegmentType;
use rvforge_reader::capability::CardStatus;
use rvforge_reader::inspect::{self, SignatureStatus, VerificationStatus};
use rvforge_reader::receipts::{ReceiptLog, RECEIPT_SCHEMA};
use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("rvforge-reader-tests");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir.join(name)
}

/// Write a fixture and return its path plus a receipt log of its own, so no
/// test writes to the user's real state directory.
fn fixture(name: &str, bytes: &[u8]) -> (PathBuf, ReceiptLog) {
    let path = scratch(name);
    std::fs::write(&path, bytes).expect("write fixture");
    let log = ReceiptLog::at(scratch(&format!("{name}.receipts.jsonl")));
    let _ = std::fs::remove_file(log.path());
    (path, log)
}

/// A well-formed container: a META segment declaring capabilities, then a root
/// MANIFEST. Its first segment's payload starts at [`HEADER_SIZE`].
fn container(capabilities: &str) -> Vec<u8> {
    minimal_container(capabilities)
}

/// Flip a byte inside the first segment's payload. The header is untouched, so
/// the container still walks and only the content-hash check fails.
fn tamper(mut data: Vec<u8>) -> Vec<u8> {
    data[HEADER_SIZE + 2] ^= 0xff;
    data
}

#[test]
fn inspection_reports_the_real_identity_and_segments() {
    let (path, _) = fixture("real.rvf", &container("network"));
    let summary = inspect::inspect(&path);

    let identity = summary.identity.expect("a well-formed container has an identity");
    assert_eq!(identity.len(), 64, "sha256 hex: {identity}");
    assert!(identity.chars().all(|c| c.is_ascii_hexdigit()));

    assert_eq!(summary.segments.len(), 2);
    assert_eq!(summary.segments[0].segment_type, "meta");
    assert_eq!(summary.segments[1].segment_type, "manifest");
    assert!(summary.root_manifest_present);
    assert_eq!(summary.declared_capabilities, vec!["network".to_string()]);
    assert!(summary.error.is_none());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn identity_is_the_hash_of_the_bytes_not_of_the_path() {
    let (a, _) = fixture("same-a.rvf", &container("clock"));
    let (b, _) = fixture("same-b.rvf", &container("clock"));
    let (c, _) = fixture("other.rvf", &container("network"));

    assert_eq!(inspect::inspect(&a).identity, inspect::inspect(&b).identity);
    assert_ne!(inspect::inspect(&a).identity, inspect::inspect(&c).identity);

    for p in [a, b, c] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn inspection_alone_never_reports_verified() {
    let (path, _) = fixture("unverified.rvf", &container("network"));
    let summary = inspect::inspect(&path);
    assert_eq!(summary.verification, VerificationStatus::Unverified);
    assert_eq!(summary.signature, SignatureStatus::Unverified);
    assert!(
        summary.publisher.is_none(),
        "naming a publisher needs a trust store the reader does not have"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn inspection_never_executes() {
    let (path, _) = fixture("no-exec.rvf", &container("process"));
    assert!(!inspect::inspect(&path).executed);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_file_that_is_not_a_container_is_reported_not_guessed() {
    let (path, _) = fixture("garbage.rvf", b"not an rvf at all");
    let summary = inspect::inspect(&path);
    assert!(summary.exists);
    assert!(summary.identity.is_none(), "no identity is invented");
    assert!(summary.error.is_some());
    assert_eq!(summary.verification, VerificationStatus::Unverified);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_missing_file_is_reported_not_guessed() {
    let summary = inspect::inspect(&scratch("does-not-exist.rvf"));
    assert!(!summary.exists);
    assert_eq!(summary.size_bytes, 0);
    assert!(summary.extension_ok);
    assert_eq!(summary.verification, VerificationStatus::Unverified);
}

#[test]
fn a_non_rvf_extension_is_flagged() {
    assert!(!inspect::inspect(&scratch("agent.zip")).extension_ok);
}

#[test]
fn verification_of_a_clean_container_passes_and_is_witnessed() {
    let (path, log) = fixture("clean.rvf", &container("clock"));
    let outcome = inspect::verify_with_receipts(&path, &log);

    assert_eq!(outcome.summary.verification, VerificationStatus::Verified);
    assert_eq!(
        outcome.summary.signature,
        SignatureStatus::Unsigned,
        "an unsigned container must not read as signature-verified"
    );
    let report = outcome.report.expect("a parsed container produces a report");
    assert!(report.ok);
    assert_eq!(report.segment_count, 2);
    assert_eq!(report.rvf_identity, outcome.summary.identity.unwrap());

    let receipts = log.read_all().expect("receipt log");
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].schema, RECEIPT_SCHEMA);
    assert!(receipts[0].ok);
    assert!(receipts[0].report.is_some());

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn a_tampered_container_fails_verification() {
    let (path, log) = fixture("tampered.rvf", &tamper(container("network")));
    let outcome = inspect::verify_with_receipts(&path, &log);

    assert_eq!(outcome.summary.verification, VerificationStatus::Failed);
    let report = outcome.report.expect("the container still walks");
    assert!(!report.ok);
    assert!(
        !report.failures().is_empty(),
        "a failure must name the check that failed"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn a_refusal_is_witnessed_exactly_as_a_pass_is() {
    let (path, log) = fixture("refused.rvf", &tamper(container("network")));
    let _ = inspect::verify_with_receipts(&path, &log);

    let receipts = log.read_all().expect("receipt log");
    assert_eq!(receipts.len(), 1, "a refusal is an auditable event");
    assert!(!receipts[0].ok);
    assert!(
        !receipts[0].detail.is_empty(),
        "the receipt must say why it was refused"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn bytes_that_are_not_a_container_are_still_witnessed() {
    let (path, log) = fixture("unparseable.rvf", b"not an rvf at all");
    let outcome = inspect::verify_with_receipts(&path, &log);

    assert_eq!(outcome.summary.verification, VerificationStatus::Failed);
    assert!(outcome.report.is_none(), "no check could run");
    let receipts = log.read_all().expect("receipt log");
    assert_eq!(receipts.len(), 1);
    assert!(!receipts[0].ok);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn every_verification_appends_rather_than_replacing() {
    let (path, log) = fixture("appended.rvf", &container("clock"));
    for _ in 0..3 {
        let _ = inspect::verify_with_receipts(&path, &log);
    }
    assert_eq!(log.read_all().unwrap().len(), 3);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn an_unsigned_executable_segment_is_refused() {
    let mut data = unsigned_segment(SegmentType::Manifest, b"root", 1);
    data.extend(unsigned_segment(SegmentType::Wasm, b"\0asm\x01\0\0\0", 2));
    let (path, log) = fixture("unsigned-exec.rvf", &data);

    let summary = inspect::inspect(&path);
    assert_eq!(summary.executable_segment_count, 1);
    assert_eq!(summary.unsigned_executable_segments, vec![2]);

    let outcome = inspect::verify_with_receipts(&path, &log);
    assert_eq!(outcome.summary.verification, VerificationStatus::Failed);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn a_signature_the_reader_cannot_check_is_not_reported_as_verified() {
    let kp = TestKeypair::deterministic(3);
    let mut data = unsigned_segment(SegmentType::Manifest, b"root", 1);
    data.extend(signed_segment(SegmentType::Wasm, b"\0asm\x01\0\0\0", 2, &kp));
    let (path, log) = fixture("signed.rvf", &data);

    let outcome = inspect::verify_with_receipts(&path, &log);
    assert_eq!(
        outcome.summary.signature,
        SignatureStatus::NotChecked,
        "with no trusted key configured, a signature is unchecked, not valid"
    );
    assert_ne!(outcome.summary.signature, SignatureStatus::Verified);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn the_capability_card_comes_from_the_verified_declaration() {
    let (path, log) = fixture("declared.rvf", &container("network,filesystem"));
    let card = inspect::capability_card_with_receipts(&path, &log);

    assert_eq!(card.status, CardStatus::Derived);
    let classes: Vec<&str> = card.requests.iter().map(|l| l.class.as_str()).collect();
    assert_eq!(classes.len(), 2);
    assert!(classes.contains(&"network"));
    assert!(classes.contains(&"filesystem"));
    assert!(
        card.cannot.iter().any(|l| l.class == "process"),
        "an undeclared class stays denied"
    );
    assert_eq!(
        card.manual_review_triggers.len(),
        2,
        "a class opened with no scope goes to manual review"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn a_tampered_package_gets_no_capability_card() {
    let (path, log) = fixture("no-card.rvf", &tamper(container("network")));
    let card = inspect::capability_card_with_receipts(&path, &log);

    assert_eq!(card.status, CardStatus::ManifestUnavailable);
    assert!(card.requests.is_empty(), "an unverified package grants nothing");
    assert_eq!(card.cannot.len(), rvforge_reader::capability::CAPABILITY_CLASSES.len());
    assert!(
        card.notes.iter().any(|n| n.contains("unverified")),
        "the refusal must be visible: {:?}",
        card.notes
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn a_package_that_is_not_a_container_gets_the_everything_denied_card() {
    let (path, log) = fixture("cardless.rvf", b"bytes");
    let card = inspect::capability_card_with_receipts(&path, &log);
    assert!(card.requests.is_empty());
    assert_eq!(card.status, CardStatus::ManifestUnavailable);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn a_sidecar_narrows_the_declaration_to_specific_scopes() {
    let (path, log) = fixture("scoped.rvf", &container("filesystem"));
    let sidecar = scratch("scoped.rvf.manifest.json");
    std::fs::write(
        &sidecar,
        r#"{"schemaVersion":1,"manifestId":"sha256:aaaa","defaultPolicy":"deny",
            "requests":[{"class":"filesystem","scope":"user-selected"}],"denials":[]}"#,
    )
    .unwrap();

    let card = inspect::capability_card_with_receipts(&path, &log);
    let line = card
        .requests
        .iter()
        .find(|l| l.class == "filesystem")
        .expect("filesystem is declared");
    assert_eq!(line.scope.as_deref(), Some("user-selected"));
    assert_eq!(
        line.text,
        "Read the files and folders you select, and nothing else"
    );
    assert!(
        card.manual_review_triggers.is_empty(),
        "a scoped class no longer needs review for a missing scope"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&sidecar);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn a_sidecar_cannot_grant_a_class_the_container_does_not_declare() {
    let (path, log) = fixture("widening.rvf", &container("filesystem"));
    let sidecar = scratch("widening.rvf.manifest.json");
    std::fs::write(
        &sidecar,
        r#"{"schemaVersion":1,"manifestId":"sha256:bbbb","defaultPolicy":"deny",
            "requests":[{"class":"filesystem","scope":"user-selected"},
                        {"class":"network","scope":"api.example.com"}],"denials":[]}"#,
    )
    .unwrap();

    let card = inspect::capability_card_with_receipts(&path, &log);
    assert!(
        !card.requests.iter().any(|l| l.class == "network"),
        "the sidecar is unsigned; it can only narrow"
    );
    assert!(card.cannot.iter().any(|l| l.class == "network"));
    assert!(card.notes.iter().any(|n| n.contains("network")));

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&sidecar);
    let _ = std::fs::remove_file(log.path());
}

#[test]
fn a_rejected_sidecar_manifest_does_not_produce_a_permissive_card() {
    let (path, log) = fixture("bad-sidecar.rvf", &container("filesystem"));
    let sidecar = scratch("bad-sidecar.rvf.manifest.json");
    std::fs::write(
        &sidecar,
        r#"{"schemaVersion":1,"manifestId":"sha256:c","defaultPolicy":"deny",
            "requests":[{"class":"filesystem","scope":"all-files"}],"denials":[]}"#,
    )
    .unwrap();

    let card = inspect::capability_card_with_receipts(&path, &log);
    assert!(card.requests.is_empty(), "a rejected manifest grants nothing");
    assert!(
        card.notes.iter().any(|n| n.contains("rejected")),
        "the rejection must be visible: {:?}",
        card.notes
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&sidecar);
    let _ = std::fs::remove_file(log.path());
}
