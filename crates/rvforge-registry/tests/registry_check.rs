//! `rvforge-registry-check` against registries this crate wrote.
//!
//! The binary is the arbiter in `scripts/rvforge-parity-check.sh`, which makes
//! two things worth proving here rather than only over there: that it accepts a
//! registry built entirely by this crate — so a parity failure is never the
//! checker misreading its own format — and that it *rejects* damage, since a
//! checker that passes everything would make the parity run vacuous.

mod common;

use common::{Fixture, NOW};
use rvforge_registry::objects::{SubjectType, TrustLevel};
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn check(root: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rvforge-registry-check"))
        .arg(root)
        .output()
        .expect("the checker binary should run")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn assert_clean(root: &Path) {
    let output = check(root);
    assert!(
        output.status.success(),
        "expected a clean registry, got:\n{}",
        stdout(&output)
    );
}

/// Assert the checker fails and names the offending area.
fn assert_violation(root: &Path, area: &str, needle: &str) {
    let output = check(root);
    let report = stdout(&output);
    assert!(
        !output.status.success(),
        "expected a violation, but the registry passed:\n{report}"
    );
    assert!(
        report.contains(&format!("violation [{area}]")) && report.contains(needle),
        "expected a [{area}] violation mentioning {needle:?}, got:\n{report}"
    );
}

#[test]
fn an_empty_registry_is_clean() {
    let dir = tempfile::tempdir().unwrap();
    rvforge_registry::Registry::open(dir.path()).unwrap();
    assert_clean(dir.path());
}

#[test]
fn a_registry_this_crate_wrote_passes_every_check() {
    let fixture = Fixture::new();
    let first = fixture.publish("analyst", "1.0.0");

    // A lineage, so the run covers the predecessor rule and a multi-leaf tree.
    let mut second = fixture.draft("analyst", "1.1.0");
    second.predecessor = Some(first.clone());
    let second = rvforge_registry::testkit::sign(
        second,
        &fixture.publisher_key,
        rvforge_registry::objects::SignatureRole::Publisher,
    )
    .unwrap();
    fixture.registry.publish_release(second, NOW).unwrap();

    // …plus the two privileged operations, which are logged and witnessed too.
    let update = fixture.trust_update(&first, TrustLevel::Tested);
    fixture.registry.raise_trust_level(update, NOW).unwrap();
    let revocation = fixture.revocation(&first, SubjectType::Release);
    fixture.registry.revoke(revocation, NOW).unwrap();

    assert_clean(fixture.registry.root());
}

#[test]
fn an_object_edited_on_disk_is_reported() {
    let fixture = Fixture::new();
    let release_id = fixture.publish("analyst", "1.0.0");
    let path = fixture.registry.store().path_for(&release_id).unwrap();

    let body = fs::read_to_string(&path).unwrap();
    fs::write(&path, body.replace("\"1.0.0\"", "\"9.9.9\"")).unwrap();

    assert_violation(fixture.registry.root(), "objects", &release_id);
}

#[test]
fn a_log_entry_edited_on_disk_is_reported() {
    let fixture = Fixture::new();
    fixture.publish("analyst", "1.0.0");
    let path = fixture.registry.root().join("log").join("entries.jsonl");

    let body = fs::read_to_string(&path).unwrap();
    fs::write(&path, body.replace("2026-08-03", "2027-01-01")).unwrap();

    assert_violation(fixture.registry.root(), "log", "hashes to");
}

#[test]
fn a_stale_tree_head_file_is_reported() {
    let fixture = Fixture::new();
    fixture.publish("analyst", "1.0.0");
    let path = fixture.registry.root().join("log").join("tree-head.json");

    let body = fs::read_to_string(&path).unwrap();
    fs::write(&path, body.replace("\"treeSize\":2", "\"treeSize\":7")).unwrap();

    assert_violation(fixture.registry.root(), "log", "tree-head.json");
}

#[test]
fn a_release_missing_from_its_package_index_is_reported() {
    let fixture = Fixture::new();
    let release_id = fixture.publish("analyst", "1.0.0");

    // The index lives under the publisher's digest; truncating it leaves the
    // release published but unindexed.
    let publisher_hex = &fixture.publisher.publisher_id[7..];
    let path = fixture
        .registry
        .root()
        .join("packages")
        .join(publisher_hex)
        .join("analyst")
        .join("releases.jsonl");
    fs::write(&path, "").unwrap();

    assert_violation(fixture.registry.root(), "packages", &release_id);
}

#[test]
fn a_receipt_removed_from_its_chain_is_reported() {
    let fixture = Fixture::new();
    let release_id = fixture.publish("analyst", "1.0.0");
    fixture
        .registry
        .record_verification(
            &release_id,
            rvforge_registry::objects::Outcome::Pass,
            common::reader(),
            "reader checked the signature",
            NOW,
        )
        .unwrap();

    // Drop the first link: the second receipt now names a predecessor that the
    // chain no longer starts with.
    let witness_dir = fixture.registry.root().join("witness");
    for entry in fs::read_dir(&witness_dir).unwrap() {
        let path = entry.unwrap().path();
        let lines: Vec<String> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        if lines.len() == 2 {
            fs::write(&path, format!("{}\n", lines[1])).unwrap();
        }
    }

    assert_violation(fixture.registry.root(), "witness", "prevReceipt");
}
