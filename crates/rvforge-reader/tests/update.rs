//! Updates: the semantic permission diff, the approval gate on any expansion,
//! lineage binding to the predecessor, and rollback that refuses when the
//! installed release declared it unsafe.

mod common;

use common::{recorded, Sandbox};

use rvforge_reader::capability::CapabilityCard;
use rvforge_reader::install::{InstallRecord, InstallRequest, StateContract};
use rvforge_reader::library::{Library, LibraryState};
use rvforge_reader::update::{self, ChangeKind, ReleaseDescriptor, UpdateApproval, UpdateError};

fn accept(classes: &[&str]) -> Vec<String> {
    classes.iter().map(|c| (*c).to_string()).collect()
}

/// A manifest granting each `(class, scope)` pair, for the scope-diff tests.
fn card(scopes: &[(&str, &str)]) -> CapabilityCard {
    let requests: Vec<String> = scopes
        .iter()
        .map(|(class, scope)| format!(r#"{{"class":"{class}","scope":"{scope}"}}"#))
        .collect();
    let json = format!(
        r#"{{"schemaVersion":1,"manifestId":"m","defaultPolicy":"deny","requests":[{}]}}"#,
        requests.join(",")
    );
    CapabilityCard::from_manifest_json(&json).expect("manifest")
}

/// An installed agent plus a candidate release that supersedes it.
fn installed_with_release(
    sandbox: &Sandbox,
    from: &str,
    to: &str,
    accepted: &[&str],
) -> (Library, InstallRecord, ReleaseDescriptor) {
    let library = sandbox.library();
    let entry = library
        .install(&InstallRequest::new(
            sandbox.package("v1.rvf", from),
            "Agent",
            accept(accepted),
        ))
        .expect("install v1");
    let release = ReleaseDescriptor::new(
        sandbox.package("v2.rvf", to),
        "1.3",
        entry.record.base_identity.clone(),
    );
    (library, entry.record, release)
}

#[test]
fn an_added_class_is_an_expansion_that_needs_approval() {
    let diff = update::diff_cards(&card(&[("network", "api.example")]), &card(&[
        ("network", "api.example"),
        ("filesystem", "selected-folders"),
    ]));

    assert_eq!(diff.changes.len(), 1);
    let change = &diff.changes[0];
    assert_eq!(change.class, "filesystem");
    assert_eq!(change.kind, ChangeKind::Added);
    assert!(change.expansion);
    assert!(diff.requires_approval());
    assert_eq!(diff.expanding_classes(), vec!["filesystem".to_string()]);
}

#[test]
fn a_removed_class_is_not_an_expansion() {
    let diff = update::diff_cards(
        &card(&[("network", "api.example"), ("gpu", "inference")]),
        &card(&[("network", "api.example")]),
    );

    assert_eq!(diff.changes.len(), 1);
    assert_eq!(diff.changes[0].class, "gpu");
    assert_eq!(diff.changes[0].kind, ChangeKind::Removed);
    assert!(!diff.changes[0].expansion);
    assert!(
        !diff.requires_approval(),
        "giving something up does not need the user's permission"
    );
    assert!(diff.lines()[0].contains("Removes gpu"));
}

#[test]
fn a_scope_change_reports_both_scopes_and_counts_as_an_expansion() {
    let diff = update::diff_cards(
        &card(&[("filesystem", "project-folder")]),
        &card(&[("filesystem", "selected-folders")]),
    );

    let change = &diff.changes[0];
    assert_eq!(change.kind, ChangeKind::ScopeChanged);
    assert_eq!(change.from_scope.as_deref(), Some("project-folder"));
    assert_eq!(change.to_scope.as_deref(), Some("selected-folders"));
    assert!(
        change.expansion,
        "the reader cannot prove the new scope is inside the old one, so it does not assume it"
    );
    assert!(change.text.contains("project-folder"));
    assert!(change.text.contains("selected-folders"));
}

#[test]
fn a_scope_that_becomes_stated_is_a_narrowing() {
    let before = CapabilityCard::from_declared_classes(&["filesystem".to_string()]).unwrap();
    let after = card(&[("filesystem", "selected-folders")]);

    let diff = update::diff_cards(&before, &after);
    let change = &diff.changes[0];
    assert_eq!(change.kind, ChangeKind::ScopeChanged);
    assert_eq!(change.from_scope, None);
    assert_eq!(change.to_scope.as_deref(), Some("selected-folders"));
    assert!(!change.expansion, "naming a limit is not asking for more");
    assert!(!diff.requires_approval());
}

#[test]
fn a_scope_that_becomes_unstated_is_a_widening() {
    let before = card(&[("filesystem", "selected-folders")]);
    let after = CapabilityCard::from_declared_classes(&["filesystem".to_string()]).unwrap();

    let diff = update::diff_cards(&before, &after);
    assert!(diff.changes[0].expansion);
    assert!(diff.requires_approval());
}

#[test]
fn an_identical_contract_produces_no_changes() {
    let diff = update::diff_cards(&card(&[("network", "api")]), &card(&[("network", "api")]));

    assert!(diff.changes.is_empty());
    assert!(!diff.requires_approval());
    assert!(
        diff.lines()[0].contains("within the existing contract"),
        "P9: code fixes inside the contract are named as such"
    );
}

#[test]
fn a_schema_change_appears_in_the_rendered_difference() {
    let sandbox = Sandbox::new("update-schema-line");
    let (library, record, mut release) =
        installed_with_release(&sandbox, "network", "network,clock", &["network"]);
    release.state_contract = StateContract {
        state_schema_version: 3,
        rollback_unsafe: false,
    };

    let plan = update::plan(&library, &record.install_id, &release).expect("plan");
    assert_eq!(
        plan.diff.state_schema_change.map(|s| (s.from, s.to)),
        Some((1, 3)),
        "the schema move is part of the difference, not a footnote"
    );

    let lines = plan.diff.lines();
    assert!(
        lines.iter().any(|l| l.contains("memory schema from 1 to 3")),
        "{lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.starts_with("Adds")),
        "the capability change is rendered beside it: {lines:?}"
    );
}

#[test]
fn an_update_whose_lineage_does_not_match_is_rejected() {
    let sandbox = Sandbox::new("update-lineage");
    let (library, record, mut release) =
        installed_with_release(&sandbox, "network", "network,filesystem", &["network"]);

    release.predecessor_identity = format!("sha256:{}", "cd".repeat(32));

    let err = update::plan(&library, &record.install_id, &release)
        .expect_err("a release that does not descend from what is installed");
    assert!(matches!(err, UpdateError::LineageMismatch { .. }));
    assert!(
        format!("{err}").contains(&record.base_identity),
        "the refusal names what is actually installed: {err}"
    );
    assert!(recorded(&library, "update", "rejected"));

    // Nothing moved.
    let after = library.get(&record.install_id).expect("get");
    assert_eq!(after.record.base_identity, record.base_identity);
}

#[test]
fn an_unverified_release_is_rejected_before_its_contract_is_shown() {
    let sandbox = Sandbox::new("update-unverified");
    let library = sandbox.library();
    let entry = library
        .install(&InstallRequest::new(
            sandbox.package("v1.rvf", "network"),
            "Agent",
            accept(&["network"]),
        ))
        .expect("install");

    let release = ReleaseDescriptor::new(
        sandbox.tampered_package("v2.rvf", "network,filesystem"),
        "1.3",
        entry.record.base_identity.clone(),
    );

    let err = update::plan(&library, &entry.record.install_id, &release)
        .expect_err("an unverified release");
    assert!(matches!(err, UpdateError::NotVerified { .. }));
}

#[test]
fn applying_without_approving_the_expansion_is_refused() {
    let sandbox = Sandbox::new("update-approval-gate");
    let (library, record, release) =
        installed_with_release(&sandbox, "network", "network,filesystem", &["network"]);

    let plan = update::plan(&library, &record.install_id, &release).expect("plan");
    assert!(plan.requires_approval);
    assert_eq!(plan.approval_classes, vec!["filesystem".to_string()]);
    assert_eq!(plan.from_identity, record.base_identity);
    assert_ne!(plan.to_identity, record.base_identity);
    assert_eq!(
        library.state_of(&record.install_id),
        LibraryState::Updates,
        "planning flags the entry as having an update available"
    );

    let err = update::apply(&library, &release, &plan, &UpdateApproval::default())
        .expect_err("an expansion nobody approved");
    assert!(matches!(err, UpdateError::ApprovalRequired { ref classes }
        if classes == &vec!["filesystem".to_string()]));
    assert!(recorded(&library, "update", "refused"));

    // The installed release is untouched.
    assert_eq!(
        library.get(&record.install_id).unwrap().record.base_identity,
        record.base_identity
    );
}

#[test]
fn an_approved_update_rebinds_the_install_to_the_new_base() {
    let sandbox = Sandbox::new("update-apply");
    let (library, record, release) =
        installed_with_release(&sandbox, "network", "network,filesystem", &["network"]);

    let plan = update::plan(&library, &record.install_id, &release).expect("plan");
    let updated = update::apply(&library, &release, &plan, &UpdateApproval::granting(&plan))
        .expect("apply an approved update");

    assert_eq!(updated.install_id, record.install_id, "the agent keeps its identity");
    assert_eq!(updated.base_identity, plan.to_identity);
    assert_eq!(
        updated.predecessor_identity.as_deref(),
        Some(record.base_identity.as_str()),
        "the new base records its predecessor (ADR-288 §7)"
    );
    assert_eq!(
        updated.granted_classes,
        vec!["network".to_string(), "filesystem".to_string()],
        "the previous grants plus exactly what was approved"
    );
    assert_eq!(library.state_of(&record.install_id), LibraryState::Installed);
    assert!(recorded(&library, "update", "applied"));
}

#[test]
fn an_update_that_only_narrows_needs_no_approval() {
    let sandbox = Sandbox::new("update-narrowing");
    let (library, record, release) =
        installed_with_release(&sandbox, "network,filesystem", "network", &["network", "filesystem"]);

    let plan = update::plan(&library, &record.install_id, &release).expect("plan");
    assert!(!plan.requires_approval);
    assert!(plan.approval_classes.is_empty());

    let updated = update::apply(&library, &release, &plan, &UpdateApproval::default())
        .expect("no approval is needed to give something up");
    assert_eq!(
        updated.granted_classes,
        vec!["network".to_string()],
        "the removed class is not carried forward"
    );
}

#[test]
fn rollback_returns_to_the_retained_predecessor() {
    let sandbox = Sandbox::new("update-rollback");
    let (library, record, release) =
        installed_with_release(&sandbox, "network", "network,filesystem", &["network"]);

    let plan = update::plan(&library, &record.install_id, &release).expect("plan");
    let updated = update::apply(&library, &release, &plan, &UpdateApproval::granting(&plan))
        .expect("apply");
    assert!(plan.rollback_available_after);

    let restored = update::rollback(&library, &record.install_id).expect("rollback");
    assert_eq!(restored.base_identity, record.base_identity);
    assert_ne!(restored.base_identity, updated.base_identity);
    assert_eq!(
        restored.granted_classes,
        vec!["network".to_string()],
        "rolling back restores the contract that came with the older release"
    );
    assert!(recorded(&library, "rollback", "completed"));

    // And the restored artifact is the one that verifies as the predecessor.
    let installed = std::fs::read(restored.base_artifact()).expect("base.rvf");
    let original = std::fs::read(sandbox.root().join("v1.rvf")).expect("v1");
    assert_eq!(installed, original);
}

#[test]
fn rollback_is_refused_when_the_release_declared_it_unsafe() {
    let sandbox = Sandbox::new("update-rollback-unsafe");
    let (library, record, mut release) =
        installed_with_release(&sandbox, "network", "network,filesystem", &["network"]);

    // Declared before the release is installed, per P9.
    release.state_contract = StateContract {
        state_schema_version: 2,
        rollback_unsafe: true,
    };

    let plan = update::plan(&library, &record.install_id, &release).expect("plan");
    assert!(
        !plan.rollback_available_after,
        "the plan tells the user, before they approve, that this is one-way"
    );

    update::apply(&library, &release, &plan, &UpdateApproval::granting(&plan)).expect("apply");

    let err = update::rollback(&library, &record.install_id).expect_err("declared unsafe");
    assert!(matches!(err, UpdateError::RollbackUnsafe { .. }));
    assert!(recorded(&library, "rollback", "refused"));

    // Still on the new release.
    assert_eq!(
        library.get(&record.install_id).unwrap().record.base_identity,
        plan.to_identity
    );
}

#[test]
fn rollback_with_nothing_retained_is_refused() {
    let sandbox = Sandbox::new("update-rollback-none");
    let library = sandbox.library();
    let entry = library
        .install(&InstallRequest::new(
            sandbox.package("v1.rvf", "network"),
            "Agent",
            accept(&["network"]),
        ))
        .expect("install");

    let err = update::rollback(&library, &entry.record.install_id)
        .expect_err("a first install has no predecessor");
    assert!(matches!(err, UpdateError::NoPredecessor { .. }));
}

#[test]
fn state_from_the_predecessor_is_retained_rather_than_migrated() {
    let sandbox = Sandbox::new("update-state-disposition");
    let (library, record, release) =
        installed_with_release(&sandbox, "memory", "memory,network", &["memory"]);

    // Accumulate some state under the old base.
    let capsule = record.capsule().expect("capsule");
    let sealed = capsule.seal(b"remembered").expect("seal");
    std::fs::write(capsule.delta_path(1), sealed).expect("delta");

    let plan = update::plan(&library, &record.install_id, &release).expect("plan");
    update::apply(&library, &release, &plan, &UpdateApproval::granting(&plan)).expect("apply");

    assert!(
        capsule.delta_path(1).is_file(),
        "the predecessor's capsule is left intact; migration is not implemented here"
    );
    let after = library.get(&record.install_id).expect("get");
    assert_eq!(
        after.state_bytes, 0,
        "the new base starts from an empty capsule rather than silently inheriting state"
    );
}
