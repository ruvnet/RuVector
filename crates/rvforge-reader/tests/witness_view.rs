//! The witness viewer (requirements P15.11, P8; ADR-295 §2 element 8).
//!
//! Every chain here is built by sealing receipts the way the registry does —
//! recompute the content address, then link the next receipt to it — so the
//! valid cases are valid for the same reason a real chain is, and the tampered
//! cases are tampered in the way a real attack would tamper with them.

use rvforge_reader::dock::WitnessStatus;
use rvforge_reader::receipts::{Receipt, ReceiptLog};
use rvforge_reader::witness_view::{
    chain_at, chain_from_text, recompute_id, Actor, ChainStatus, Evidence, WitnessReceipt,
    WITNESS_RECEIPT_TYPE,
};

fn unsealed(subject: &str, event: &str, prev: Option<&str>) -> WitnessReceipt {
    WitnessReceipt {
        schema_version: 1,
        object_type: WITNESS_RECEIPT_TYPE.to_string(),
        receipt_id: String::new(),
        subject: subject.to_string(),
        event: event.to_string(),
        outcome: "pass".to_string(),
        actor: Actor {
            kind: "registry".to_string(),
            id: "registry.example".to_string(),
        },
        evidence: Evidence {
            details: format!("{event} completed"),
        },
        timestamp: "2026-08-03T00:00:00Z".to_string(),
        prev_receipt: prev.map(str::to_string),
        signatures: Vec::new(),
    }
}

/// The registry's sealing step: the id is the content, so it is computed last.
fn seal(mut receipt: WitnessReceipt) -> WitnessReceipt {
    receipt.receipt_id = recompute_id(&receipt).expect("receipt re-encodes");
    receipt
}

/// A sealed chain of `events.len()` receipts for one subject.
fn chain(subject: &str, events: &[&str]) -> Vec<WitnessReceipt> {
    let mut out: Vec<WitnessReceipt> = Vec::new();
    for event in events {
        let prev = out.last().map(|r: &WitnessReceipt| r.receipt_id.clone());
        out.push(seal(unsealed(subject, event, prev.as_deref())));
    }
    out
}

fn jsonl(receipts: &[WitnessReceipt]) -> String {
    receipts
        .iter()
        .map(|r| serde_json::to_string(r).expect("receipt serializes"))
        .collect::<Vec<_>>()
        .join("\n")
}

const SUBJECT: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";
const OTHER: &str = "sha256:2222222222222222222222222222222222222222222222222222222222222222";

#[test]
fn a_valid_chain_renders_every_receipt_in_order() {
    let receipts = chain(SUBJECT, &["build", "verify", "install"]);
    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts));

    assert_eq!(view.status, ChainStatus::Valid);
    assert_eq!(view.label, "valid");
    assert_eq!(view.receipt_count, 3);
    assert_eq!(view.subjects.len(), 1);

    let subject = &view.subjects[0];
    assert_eq!(subject.subject, SUBJECT);
    assert_eq!(subject.status, ChainStatus::Valid);
    assert_eq!(
        subject
            .receipts
            .iter()
            .map(|r| r.event.as_str())
            .collect::<Vec<_>>(),
        ["build", "verify", "install"]
    );
    assert!(subject.receipts.iter().all(|r| r.chained));
    assert!(subject.receipts.iter().all(|r| r.break_reason.is_none()));
    assert_eq!(subject.receipts[1].outcome, "pass");
    assert_eq!(subject.receipts[1].actor, "registry:registry.example");
    assert_eq!(subject.receipts[1].timestamp, "2026-08-03T00:00:00Z");
}

#[test]
fn a_receipt_id_is_shortened_but_still_recognisable_as_a_digest() {
    let receipts = chain(SUBJECT, &["build"]);
    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts));
    let shown = &view.subjects[0].receipts[0].id_short;

    assert!(shown.starts_with("sha256:"), "{shown}");
    assert!(shown.ends_with('…'), "{shown}");
    assert!(receipts[0]
        .receipt_id
        .starts_with(&shown[..shown.len() - 3]));
}

#[test]
fn each_subject_is_verified_as_its_own_chain() {
    let mut receipts = chain(SUBJECT, &["build", "verify"]);
    receipts.extend(chain(OTHER, &["build"]));
    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts));

    assert_eq!(view.status, ChainStatus::Valid);
    assert_eq!(view.subjects.len(), 2);
    assert_eq!(view.subjects[0].subject, SUBJECT);
    assert_eq!(view.subjects[0].receipts.len(), 2);
    assert_eq!(view.subjects[1].subject, OTHER);
    assert_eq!(view.subjects[1].receipts.len(), 1);
}

#[test]
fn a_tampered_receipt_breaks_the_chain_at_its_own_index() {
    let mut receipts = chain(SUBJECT, &["build", "verify", "install"]);
    // The id is left alone; only the content it commits to is edited. This is
    // the whole point of recomputing: the id no longer describes the record.
    receipts[1].evidence.details = "verify completed with no findings".to_string();

    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts));

    match &view.status {
        ChainStatus::BrokenAt { index, reason } => {
            assert_eq!(*index, 1);
            assert!(
                reason.contains("content does not match receiptId"),
                "{reason}"
            );
        }
        other => panic!("expected broken-at-1, got {other:?}"),
    }
    assert_eq!(view.label, "broken-at-1");
    assert_eq!(view.subjects[0].label, "broken-at-1");
    // The break is marked on the receipt that failed, not on the whole chain.
    assert!(view.subjects[0].receipts[0].break_reason.is_none());
    assert!(view.subjects[0].receipts[1].break_reason.is_some());
    assert!(view.subjects[0].receipts[2].break_reason.is_none());
    // The records that did verify are still shown.
    assert_eq!(view.subjects[0].receipts.len(), 3);
}

#[test]
fn tampering_with_the_last_receipt_is_caught_too() {
    let mut receipts = chain(SUBJECT, &["build", "verify"]);
    receipts[1].outcome = "fail".to_string();
    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts));

    assert!(
        matches!(view.status, ChainStatus::BrokenAt { index: 1, .. }),
        "{:?}",
        view.status
    );
}

#[test]
fn reordered_receipts_are_detected_even_though_each_one_verifies() {
    let receipts = chain(SUBJECT, &["build", "verify", "install"]);
    // Every record is byte-for-byte intact; only their order changed.
    let reordered = vec![
        receipts[0].clone(),
        receipts[2].clone(),
        receipts[1].clone(),
    ];
    for r in &reordered {
        assert_eq!(recompute_id(r).as_deref(), Some(r.receipt_id.as_str()));
    }

    let view = chain_from_text("receipts.jsonl", &jsonl(&reordered));

    match &view.status {
        ChainStatus::BrokenAt { index, reason } => {
            assert_eq!(*index, 1);
            assert!(reason.contains("prevReceipt"), "{reason}");
        }
        other => panic!("expected broken-at-1, got {other:?}"),
    }
}

#[test]
fn a_chain_that_does_not_start_at_the_beginning_is_broken_at_zero() {
    let receipts = chain(SUBJECT, &["build", "verify"]);
    // The first record is dropped, so what remains claims a predecessor that
    // is not in the file.
    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts[1..]));

    match &view.status {
        ChainStatus::BrokenAt { index, reason } => {
            assert_eq!(*index, 0);
            assert!(reason.contains("first receipt"), "{reason}");
        }
        other => panic!("expected broken-at-0, got {other:?}"),
    }
}

#[test]
fn a_broken_subject_is_not_hidden_by_a_valid_one() {
    let mut receipts = chain(SUBJECT, &["build", "verify"]);
    receipts[1].timestamp = "2026-08-04T00:00:00Z".to_string();
    receipts.extend(chain(OTHER, &["build", "verify"]));

    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts));

    assert!(
        matches!(view.status, ChainStatus::BrokenAt { .. }),
        "{:?}",
        view.status
    );
    assert_eq!(view.subjects[1].status, ChainStatus::Valid);
    assert_eq!(view.dock_status, WitnessStatus::Broken);
}

#[test]
fn an_empty_file_is_empty_not_an_error() {
    let view = chain_from_text("receipts.jsonl", "");

    assert_eq!(view.status, ChainStatus::Empty);
    assert_eq!(view.label, "empty");
    assert_eq!(view.receipt_count, 0);
    assert!(view.subjects.is_empty());
    assert!(view.malformed.is_empty());
    assert_eq!(view.summary, "No witness records");
    assert_eq!(view.dock_status, WitnessStatus::Unavailable);
}

#[test]
fn blank_lines_are_not_records() {
    let receipts = chain(SUBJECT, &["build"]);
    let text = format!("\n{}\n\n", jsonl(&receipts));
    let view = chain_from_text("receipts.jsonl", &text);

    assert_eq!(view.status, ChainStatus::Valid);
    assert_eq!(view.receipt_count, 1);
    assert!(view.malformed.is_empty());
}

#[test]
fn a_missing_file_reads_as_empty() {
    let dir = std::env::temp_dir().join(format!("rvforge-witness-{}", std::process::id()));
    let view = chain_at(&dir.join("nothing-here.jsonl")).expect("a missing file is not an error");

    assert_eq!(view.status, ChainStatus::Empty);
}

#[test]
fn an_unreadable_line_is_reported_and_stops_the_file_reading_as_valid() {
    let receipts = chain(SUBJECT, &["build", "verify"]);
    let text = format!("{}\nthis is not JSON\n", jsonl(&receipts));
    let view = chain_from_text("receipts.jsonl", &text);

    assert_eq!(view.malformed.len(), 1);
    assert_eq!(view.malformed[0].line, 3);
    // The chain itself verified, but a line nobody could read is a gap in the
    // evidence, and a gap must not render as a complete chain.
    assert_eq!(view.subjects[0].status, ChainStatus::Valid);
    assert_eq!(view.status, ChainStatus::Unchained);
    assert!(view.summary.contains("unreadable"), "{}", view.summary);
    assert_eq!(view.dock_status, WitnessStatus::Unavailable);
}

#[test]
fn a_receipt_from_a_newer_schema_is_not_guessed_at() {
    let mut receipt = seal(unsealed(SUBJECT, "build", None));
    receipt.schema_version = 2;
    let view = chain_from_text("receipts.jsonl", &jsonl(&[receipt]));

    assert_eq!(view.malformed.len(), 1);
    assert!(
        view.malformed[0].reason.contains("schemaVersion"),
        "{}",
        view.malformed[0].reason
    );
    assert_ne!(view.status, ChainStatus::Valid);
}

#[test]
fn the_readers_own_receipts_are_unchained_never_valid() {
    let log = Receipt::new(
        std::path::Path::new("/packages/demo.rvf"),
        Some("sha256:abcd".to_string()),
        true,
        String::new(),
        None,
    );
    let refused = Receipt::new(
        std::path::Path::new("/packages/demo.rvf"),
        None,
        false,
        "root manifest hash does not match".to_string(),
        None,
    );
    let text = format!(
        "{}\n{}",
        serde_json::to_string(&log).unwrap(),
        serde_json::to_string(&refused).unwrap()
    );

    let view = chain_from_text("receipts.jsonl", &text);

    assert_eq!(view.status, ChainStatus::Unchained);
    assert_eq!(view.label, "unchained");
    assert_eq!(view.dock_status, WitnessStatus::Unavailable);
    assert_eq!(view.subjects.len(), 1);
    assert_eq!(view.subjects[0].subject, "/packages/demo.rvf");

    let rows = &view.subjects[0].receipts;
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| !r.chained));
    assert_eq!(rows[0].outcome, "pass");
    assert_eq!(rows[1].outcome, "fail");
    // A refusal keeps the reason it was refused for.
    assert_eq!(rows[1].detail, "root manifest hash does not match");
}

#[test]
fn a_readers_log_written_to_disk_reads_back_as_a_chain_view() {
    let dir = std::env::temp_dir().join(format!(
        "rvforge-witness-log-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let log = ReceiptLog::at(dir.join("receipts.jsonl"));
    log.append(&Receipt::new(
        std::path::Path::new("/packages/demo.rvf"),
        None,
        true,
        String::new(),
        None,
    ))
    .expect("the log is writable");

    let view = chain_at(log.path()).expect("the log reads back");

    assert_eq!(view.receipt_count, 1);
    assert_eq!(view.status, ChainStatus::Unchained);
    assert_eq!(view.source, log.path().to_string_lossy());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_dock_summary_line_says_which_receipt_broke_the_chain() {
    let mut receipts = chain(SUBJECT, &["build", "verify", "install"]);
    receipts[2].actor.id = "someone.else".to_string();
    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts));

    // One-based in prose, zero-based in the model: "receipt 3 of 3" is what an
    // operator counts, `broken-at-2` is what the model reports.
    assert!(view
        .summary
        .starts_with("Witness chain broken at receipt 3 of 3"));
    assert_eq!(view.label, "broken-at-2");
    assert!(
        view.summary.contains("sha256:111111111111…"),
        "{}",
        view.summary
    );
}

#[test]
fn the_dock_summary_line_reflects_a_valid_chain() {
    let mut receipts = chain(SUBJECT, &["build", "verify"]);
    receipts.extend(chain(OTHER, &["build"]));
    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts));

    assert_eq!(
        view.summary,
        "Witness chain valid — 3 receipts across 2 subjects"
    );
    assert_eq!(view.dock_status, WitnessStatus::Valid);
}

#[test]
fn a_single_receipt_reads_as_one_receipt_one_subject() {
    let view = chain_from_text("receipts.jsonl", &jsonl(&chain(SUBJECT, &["build"])));

    assert_eq!(
        view.summary,
        "Witness chain valid — 1 receipt across 1 subject"
    );
}

#[test]
fn the_status_serializes_with_the_index_it_broke_at() {
    let mut receipts = chain(SUBJECT, &["build", "verify"]);
    receipts[1].evidence.details = "edited".to_string();
    let view = chain_from_text("receipts.jsonl", &jsonl(&receipts));

    let json = serde_json::to_value(&view).expect("the view serializes");
    assert_eq!(json["status"]["kind"], "broken-at");
    assert_eq!(json["status"]["index"], 1);
    assert_eq!(json["label"], "broken-at-1");
    assert_eq!(json["dock_status"], "broken");
    assert_eq!(json["subjects"][0]["receipts"][1]["chained"], true);
}

#[test]
fn signatures_do_not_change_a_receipt_id() {
    let sealed = seal(unsealed(SUBJECT, "build", None));
    let mut countersigned = sealed.clone();
    countersigned.signatures = vec![serde_json::json!({
        "keyId": "ed25519:registry",
        "role": "registry",
        "sig": "AAAA"
    })];

    // A countersignature added after the fact must not invalidate the chain.
    assert_eq!(recompute_id(&countersigned), recompute_id(&sealed));
    let view = chain_from_text("receipts.jsonl", &jsonl(&[countersigned]));
    assert_eq!(view.status, ChainStatus::Valid);
}
