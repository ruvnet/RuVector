//! The installed-agent library: state transitions including the illegal ones,
//! state operations, and the rule that removal never silently destroys state.

mod common;

use common::{lifecycle_events, Sandbox};

use rvf_forge_core::inspect::HEADER_SIZE;
use rvforge_reader::install::InstallRequest;
use rvforge_reader::inspect::VerificationStatus;
use rvforge_reader::library::{Library, LibraryEntry, LibraryError, LibraryState, UninstallPolicy};
use rvforge_reader::state::StateCapsule;
use rvforge_reader::witness_view;

fn accept(classes: &[&str]) -> Vec<String> {
    classes.iter().map(|c| (*c).to_string()).collect()
}

/// One installed agent, ready to operate on.
fn installed(sandbox: &Sandbox, capabilities: &str) -> (Library, LibraryEntry) {
    let library = sandbox.library();
    let package = sandbox.package("agent.rvf", capabilities);
    let entry = library
        .install(&InstallRequest::new(
            &package,
            "Agent",
            accept(&capabilities.split(',').collect::<Vec<_>>()),
        ))
        .expect("install");
    (library, entry)
}

/// Write one sealed delta into an agent's capsule.
fn seal_a_delta(entry: &LibraryEntry, seq: u32, payload: &[u8]) {
    let capsule = entry.record.capsule().expect("capsule");
    let sealed = capsule.seal(payload).expect("seal");
    std::fs::write(capsule.delta_path(seq), sealed).expect("write delta");
}

#[test]
fn a_new_install_lists_as_installed() {
    let sandbox = Sandbox::new("lib-list");
    let (library, entry) = installed(&sandbox, "network");

    let rows = library.list().expect("list");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].state, LibraryState::Installed);
    assert_eq!(rows[0].state_label, "Installed");
    assert_eq!(rows[0].record.install_id, entry.record.install_id);
    assert_eq!(rows[0].state_bytes, 0, "a fresh install holds no state");
}

#[test]
fn open_pause_and_terminate_walk_the_expected_states() {
    let sandbox = Sandbox::new("lib-walk");
    let (library, entry) = installed(&sandbox, "network");
    let id = &entry.record.install_id;

    assert_eq!(library.open(id).expect("open"), LibraryState::Running);
    assert_eq!(library.pause(id).expect("pause"), LibraryState::Paused);
    assert_eq!(library.open(id).expect("resume"), LibraryState::Running);
    assert_eq!(
        library.terminate(id).expect("terminate"),
        LibraryState::Installed,
        "stopping an agent ends the run, not the installation"
    );
}

#[test]
fn an_archived_agent_cannot_start_running() {
    let sandbox = Sandbox::new("lib-archived");
    let (library, entry) = installed(&sandbox, "network");
    let id = &entry.record.install_id;

    library.archive(id).expect("archive");
    let err = library
        .open(id)
        .expect_err("coming back from archived is a decision, not a side effect of pressing open");
    assert!(matches!(
        err,
        LibraryError::IllegalTransition {
            from: LibraryState::Archived,
            to: LibraryState::Running,
            ..
        }
    ));

    library.restore(id).expect("restore");
    assert_eq!(library.open(id).expect("open after restore"), LibraryState::Running);
}

#[test]
fn a_quarantined_agent_cannot_start_running() {
    let sandbox = Sandbox::new("lib-quarantined");
    let (library, entry) = installed(&sandbox, "network");
    let id = &entry.record.install_id;

    library.quarantine(id, "policy hold").expect("quarantine");
    let err = library.open(id).expect_err("quarantine blocks execution");
    assert!(matches!(
        err,
        LibraryError::IllegalTransition {
            from: LibraryState::Quarantined,
            ..
        }
    ));

    assert_eq!(
        library.get(id).expect("get").note.as_deref(),
        Some("policy hold"),
        "the reason is kept with the entry"
    );
}

#[test]
fn quarantine_preserves_state_rather_than_deleting_it() {
    let sandbox = Sandbox::new("lib-quarantine-state");
    let (library, entry) = installed(&sandbox, "memory");
    seal_a_delta(&entry, 1, b"accumulated memory");

    library
        .quarantine(&entry.record.install_id, "revoked upstream")
        .expect("quarantine");

    let after = library.get(&entry.record.install_id).expect("get");
    assert_eq!(after.state, LibraryState::Quarantined);
    assert!(
        after.state_bytes > 0,
        "ADR-294: a user whose agent is revoked keeps their state capsule"
    );
}

#[test]
fn opening_an_agent_whose_artifact_changed_quarantines_it() {
    let sandbox = Sandbox::new("lib-tampered-artifact");
    let (library, entry) = installed(&sandbox, "network");
    let id = &entry.record.install_id;

    // Alter the installed artifact behind the reader's back.
    let artifact = entry.record.base_artifact();
    let mut perms = std::fs::metadata(&artifact).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    std::fs::set_permissions(&artifact, perms).unwrap();
    let mut bytes = std::fs::read(&artifact).unwrap();
    bytes[HEADER_SIZE + 2] ^= 0xff;
    std::fs::write(&artifact, &bytes).unwrap();

    let err = library
        .open(id)
        .expect_err("verification precedes every load, including of an already installed package");
    assert!(matches!(err, LibraryError::VerificationFailed { .. }));
    assert_eq!(
        library.state_of(id),
        LibraryState::Quarantined,
        "an artifact that stopped verifying is not left offered as runnable"
    );
}

#[test]
fn verify_passes_for_an_untouched_installation() {
    let sandbox = Sandbox::new("lib-verify-ok");
    let (library, entry) = installed(&sandbox, "network");

    assert_eq!(
        library.verify(&entry.record.install_id).expect("verify"),
        VerificationStatus::Verified
    );
}

#[test]
fn reset_discards_state_and_leaves_the_base_alone() {
    let sandbox = Sandbox::new("lib-reset");
    let (library, entry) = installed(&sandbox, "memory");
    seal_a_delta(&entry, 1, b"one");
    seal_a_delta(&entry, 2, b"two");

    let before = std::fs::read(entry.record.base_artifact()).unwrap();
    let removed = library.reset(&entry.record.install_id).expect("reset");

    assert_eq!(removed, 2);
    assert_eq!(
        library.get(&entry.record.install_id).unwrap().state_bytes,
        0
    );
    assert_eq!(
        std::fs::read(entry.record.base_artifact()).unwrap(),
        before,
        "reset discards state records; it never rewrites the base"
    );
}

#[test]
fn exported_state_can_be_imported_back() {
    let sandbox = Sandbox::new("lib-roundtrip");
    let (library, entry) = installed(&sandbox, "memory");
    let id = &entry.record.install_id;
    seal_a_delta(&entry, 1, b"remembered");

    let dest = sandbox.dir("exported");
    let transfer = library.export_state(id, &dest).expect("export");
    assert_eq!(transfer.files, 1);
    assert!(transfer.bytes > 0);
    assert_eq!(transfer.base_identity, entry.record.base_identity);

    // Exported bytes are still sealed: an export is a copy of the capsule, not
    // a decryption of it.
    let exported = std::fs::read(dest.join("delta-001.wdelta")).expect("exported delta");
    assert!(
        !exported.windows(10).any(|w| w == b"remembered"),
        "exported state must not contain the plaintext"
    );

    library.reset(id).expect("reset");
    assert_eq!(library.get(id).unwrap().state_bytes, 0);

    library.import_state(id, &dest).expect("import");
    assert!(library.get(id).unwrap().state_bytes > 0);

    // And it still opens, so the round trip preserved the sealed bytes exactly.
    let capsule = entry.record.capsule().unwrap();
    let sealed = std::fs::read(capsule.delta_path(1)).unwrap();
    assert_eq!(capsule.unseal(&sealed).expect("unseal"), b"remembered");
}

#[test]
fn state_from_another_lineage_is_refused_on_import() {
    let sandbox = Sandbox::new("lib-foreign-state");
    let (library, entry) = installed(&sandbox, "memory");
    let id = &entry.record.install_id;
    seal_a_delta(&entry, 1, b"mine");
    let own = std::fs::read(entry.record.capsule().unwrap().delta_path(1)).unwrap();

    // Seal a delta under a different base identity, in the same state root.
    let foreign_identity = format!("sha256:{}", "ab".repeat(32));
    let foreign = StateCapsule::new(
        entry.record.install_dir(),
        sandbox.state_root(),
        &foreign_identity,
    )
    .expect("foreign capsule");
    let incoming = sandbox.dir("incoming");
    std::fs::write(
        incoming.join("delta-001.wdelta"),
        foreign.seal(b"someone else's").expect("seal"),
    )
    .unwrap();

    let err = library
        .import_state(id, &incoming)
        .expect_err("state from another lineage is refused (ADR-288 §4)");
    assert!(matches!(err, LibraryError::LineageRejected { .. }));
    assert!(
        format!("{err}").contains("lineage"),
        "the refusal says what was wrong: {err}"
    );

    assert_eq!(
        std::fs::read(entry.record.capsule().unwrap().delta_path(1)).unwrap(),
        own,
        "a rejected import leaves the existing capsule exactly as it was"
    );
}

#[test]
fn uninstall_keeps_state_by_default() {
    let sandbox = Sandbox::new("lib-uninstall-default");
    let (library, entry) = installed(&sandbox, "memory");
    let id = entry.record.install_id.clone();
    seal_a_delta(&entry, 1, b"kept");
    let capsule_dir = entry.record.capsule().unwrap().capsule_dir();

    let outcome = library
        .uninstall(&id, &UninstallPolicy::keep_state())
        .expect("uninstall");

    assert!(outcome.runtime_removed);
    assert!(!outcome.state_removed);
    assert_eq!(
        outcome.state_retained_at.as_deref(),
        Some(capsule_dir.display().to_string().as_str())
    );
    assert!(!entry.record.install_dir().exists(), "the runtime is gone");
    assert!(
        capsule_dir.join("delta-001.wdelta").is_file(),
        "ADR-294: removal preserves the user's state unless they asked otherwise"
    );
    assert!(library.list().expect("list").is_empty());
}

#[test]
fn uninstall_exports_before_it_removes_state() {
    let sandbox = Sandbox::new("lib-uninstall-export");
    let (library, entry) = installed(&sandbox, "memory");
    let id = entry.record.install_id.clone();
    seal_a_delta(&entry, 1, b"kept");
    let capsule_dir = entry.record.capsule().unwrap().capsule_dir();
    let dest = sandbox.dir("rescue");

    let outcome = library
        .uninstall(&id, &UninstallPolicy::export_then_remove(&dest))
        .expect("uninstall");

    assert!(outcome.state_removed);
    assert_eq!(outcome.exported_files, 1);
    assert_eq!(outcome.exported_to.as_deref(), Some(dest.display().to_string().as_str()));
    assert!(
        dest.join("delta-001.wdelta").is_file(),
        "the export happened before the removal"
    );
    assert!(!capsule_dir.exists(), "then the capsule was removed");
}

#[test]
fn removing_state_without_an_export_has_to_be_asked_for_by_name() {
    let sandbox = Sandbox::new("lib-uninstall-discard");
    let (library, entry) = installed(&sandbox, "memory");
    let id = entry.record.install_id.clone();
    seal_a_delta(&entry, 1, b"kept");
    let capsule_dir = entry.record.capsule().unwrap().capsule_dir();

    // `remove_state` with no rescue path is the explicit discard, and it is
    // honoured — the guarantee is that state is never lost *silently*.
    let outcome = library
        .uninstall(
            &id,
            &UninstallPolicy {
                remove_state: true,
                export_state_to: None,
            },
        )
        .expect("an explicit discard is allowed");

    assert!(outcome.state_removed);
    assert!(outcome.exported_to.is_none());
    assert!(!capsule_dir.exists());
}

#[test]
fn a_failed_export_aborts_the_whole_uninstall() {
    let sandbox = Sandbox::new("lib-uninstall-export-fails");
    let (library, entry) = installed(&sandbox, "memory");
    let id = entry.record.install_id.clone();
    seal_a_delta(&entry, 1, b"kept");
    let capsule_dir = entry.record.capsule().unwrap().capsule_dir();

    // A regular file where the export directory should go: the copy cannot run.
    let blocked = sandbox.write("blocked", b"not a directory");

    let err = library
        .uninstall(&id, &UninstallPolicy::export_then_remove(&blocked))
        .expect_err("an export that failed must not be followed by a removal");
    assert!(matches!(err, LibraryError::Io { .. }), "{err}");

    assert!(entry.record.install_dir().exists(), "the runtime is still there");
    assert!(capsule_dir.join("delta-001.wdelta").is_file(), "so is the state");
}

#[test]
fn uninstall_refuses_while_the_agent_may_be_executing() {
    let sandbox = Sandbox::new("lib-uninstall-running");
    let (library, entry) = installed(&sandbox, "network");
    let id = entry.record.install_id.clone();
    library.open(&id).expect("open");

    let err = library
        .uninstall(&id, &UninstallPolicy::keep_state())
        .expect_err("removing a running agent's runtime out from under it");
    assert!(matches!(
        err,
        LibraryError::StillActive {
            state: LibraryState::Running,
            ..
        }
    ));

    library.terminate(&id).expect("terminate");
    library
        .uninstall(&id, &UninstallPolicy::keep_state())
        .expect("uninstall after terminate");
}

#[test]
fn operating_on_an_agent_that_is_not_installed_says_so() {
    let sandbox = Sandbox::new("lib-missing");
    let library = sandbox.library();

    for err in [
        library.open("agent-nope").unwrap_err(),
        library.pause("agent-nope").unwrap_err(),
        library.reset("agent-nope").unwrap_err(),
        library.verify("agent-nope").unwrap_err(),
    ] {
        assert!(matches!(err, LibraryError::NotInstalled { .. }), "{err}");
    }
}

#[test]
fn the_transition_table_is_the_whole_rule() {
    use LibraryState::*;

    assert!(Installed.may_move_to(Running));
    assert!(Running.may_move_to(Paused));
    assert!(Paused.may_move_to(Running));
    assert!(Updates.may_move_to(Running));
    assert!(Quarantined.may_move_to(Installed));
    assert!(Archived.may_move_to(Installed));

    // The two states that block execution release only through Installed.
    assert!(!Quarantined.may_move_to(Running));
    assert!(!Quarantined.may_move_to(Paused));
    assert!(!Archived.may_move_to(Running));
    assert!(!Archived.may_move_to(Paused));
    assert!(!Archived.may_move_to(Quarantined));
    // Archiving something that may be executing skips the stop.
    assert!(!Running.may_move_to(Archived));
    assert!(!Paused.may_move_to(Archived));

    // A move to the state you are already in is not an error.
    for state in [Installed, Running, Paused, Updates, Quarantined, Archived] {
        assert!(state.may_move_to(state));
    }
}

#[test]
fn every_library_operation_leaves_a_verifiable_receipt() {
    let sandbox = Sandbox::new("lib-witness");
    let (library, entry) = installed(&sandbox, "memory");
    let id = entry.record.install_id.clone();
    seal_a_delta(&entry, 1, b"kept");

    library.open(&id).expect("open");
    library.pause(&id).expect("pause");
    library.terminate(&id).expect("terminate");
    library.export_state(&id, &sandbox.dir("out")).expect("export");
    library.reset(&id).expect("reset");
    library
        .uninstall(&id, &UninstallPolicy::keep_state())
        .expect("uninstall");

    let events: Vec<String> = lifecycle_events(&library)
        .into_iter()
        .map(|(_, e, _)| e)
        .collect();
    for expected in [
        "install",
        "verify",
        "library-state",
        "state-export",
        "reset",
        "uninstall",
    ] {
        assert!(
            events.iter().any(|e| e == expected),
            "'{expected}' is not in {events:?}"
        );
    }

    let view = witness_view::chain_at(library.layout().lifecycle_log().path()).expect("chain");
    assert_eq!(view.label, "valid", "{}", view.summary);
}
