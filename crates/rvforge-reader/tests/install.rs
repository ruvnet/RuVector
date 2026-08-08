//! Installation: verification gates it, the declaration bounds it, the base
//! artifact is immutable, and every outcome is witnessed.

mod common;

use common::{lifecycle_events, recorded, Sandbox};

use rvforge_reader::capability::CardStatus;
use rvforge_reader::install::{self, InstallError, InstallRequest, StateContract};
use rvforge_reader::witness_view;

fn accept(classes: &[&str]) -> Vec<String> {
    classes.iter().map(|c| (*c).to_string()).collect()
}

#[test]
fn installing_materializes_the_package_and_records_what_was_accepted() {
    let sandbox = Sandbox::new("install-happy");
    let library = sandbox.library();
    let package = sandbox.package("agent.rvf", "network,filesystem");

    let entry = library
        .install(&InstallRequest::new(
            &package,
            "Research agent",
            accept(&["network"]),
        ))
        .expect("a verified package with a declared grant installs");

    let record = &entry.record;
    assert!(record.base_identity.starts_with("sha256:"));
    assert_eq!(record.agent_name, "Research agent");
    assert_eq!(record.granted_classes, vec!["network".to_string()]);
    assert_eq!(record.predecessor_identity, None);
    assert!(!record.state_contract.rollback_unsafe);

    // The installed artifact is a copy of the bytes that verified.
    let installed = std::fs::read(record.base_artifact()).expect("base.rvf");
    assert_eq!(installed, std::fs::read(&package).expect("source"));

    // The record survives a round trip through disk.
    let reread = library.layout().read_record(&record.install_id).unwrap();
    assert_eq!(&reread, record);

    // Only what was accepted is described as granted; the declared-but-unaccepted
    // class is not carried into the record.
    assert_eq!(record.granted_text.len(), 1);
    assert!(record.granted_text[0].to_lowercase().contains("network"));
}

#[test]
fn state_lives_outside_the_install_root() {
    let sandbox = Sandbox::new("install-state-separate");
    let library = sandbox.library();
    let package = sandbox.package("agent.rvf", "memory");

    let entry = library
        .install(&InstallRequest::new(&package, "Agent", accept(&["memory"])))
        .expect("install");

    let capsule = entry.record.capsule().expect("capsule");
    assert!(capsule.capsule_dir().is_dir(), "the capsule dir is created");
    assert!(
        !capsule.capsule_dir().starts_with(entry.record.install_dir()),
        "deleting state must not reach the immutable base artifact (ADR-288 §2)"
    );
}

#[test]
fn the_installed_artifact_is_read_only() {
    let sandbox = Sandbox::new("install-readonly");
    let library = sandbox.library();
    let package = sandbox.package("agent.rvf", "clock");

    let entry = library
        .install(&InstallRequest::new(&package, "Agent", accept(&["clock"])))
        .expect("install");

    let perms = std::fs::metadata(entry.record.base_artifact())
        .expect("base.rvf")
        .permissions();
    assert!(perms.readonly(), "the base artifact is immutable after signing");
}

#[test]
fn an_unverified_package_is_not_installed() {
    let sandbox = Sandbox::new("install-unverified");
    let library = sandbox.library();
    let package = sandbox.tampered_package("agent.rvf", "network");

    let err = library
        .install(&InstallRequest::new(&package, "Agent", accept(&["network"])))
        .expect_err("a package that fails its content-hash check must not install");

    assert!(
        format!("{err}").contains("unverified"),
        "the refusal says why: {err}"
    );
    assert!(
        library.list().expect("list").is_empty(),
        "nothing is installed after a refusal"
    );
    assert!(
        !sandbox.install_root().join("agent-").exists(),
        "no install directory is created"
    );
}

#[test]
fn a_grant_the_container_did_not_declare_is_refused() {
    let sandbox = Sandbox::new("install-overbroad");
    let library = sandbox.library();
    // The container declares network only.
    let package = sandbox.package("agent.rvf", "network");

    let err = library
        .install(&InstallRequest::new(
            &package,
            "Agent",
            accept(&["network", "filesystem"]),
        ))
        .expect_err("accepting more than was declared is not a narrowing, it is a mismatch");

    assert!(
        format!("{err}").contains("filesystem"),
        "the refusal names the class: {err}"
    );
    assert!(library.list().expect("list").is_empty());
}

#[test]
fn accepting_nothing_installs_with_nothing_granted() {
    let sandbox = Sandbox::new("install-deny-all");
    let library = sandbox.library();
    let package = sandbox.package("agent.rvf", "network,filesystem");

    let entry = library
        .install(&InstallRequest::new(&package, "Agent", Vec::new()))
        .expect("declining every class is a valid contract");

    assert!(entry.record.granted_classes.is_empty());
    assert!(entry.record.granted_text.is_empty());
}

#[test]
fn the_offer_refuses_before_the_user_is_asked_to_consent() {
    let sandbox = Sandbox::new("install-offer-refusal");
    let library = sandbox.library();

    let good = library.offer(&sandbox.package("ok.rvf", "network"));
    assert!(good.verified);
    assert_eq!(good.card.status, CardStatus::Derived);
    assert_eq!(good.grantable_classes, vec!["network".to_string()]);
    assert!(good.refusal.is_none());
    assert!(good.runtime_profile.is_some(), "a runtime was selected");

    let bad = library.offer(&sandbox.tampered_package("bad.rvf", "network"));
    assert!(!bad.verified);
    assert_eq!(bad.card.status, CardStatus::ManifestUnavailable);
    assert!(
        bad.grantable_classes.is_empty(),
        "an unverified package grants nothing"
    );
    assert!(bad.refusal.is_some(), "the refusal is stated, not implied");
    assert_eq!(
        bad.card.requests.len(),
        0,
        "the card shown for a refused package is the everything-denied card"
    );
}

#[test]
fn installing_the_same_package_twice_is_refused() {
    let sandbox = Sandbox::new("install-twice");
    let library = sandbox.library();
    let package = sandbox.package("agent.rvf", "network");
    let request = InstallRequest::new(&package, "Agent", accept(&["network"]));

    library.install(&request).expect("first install");
    let err = library
        .install(&request)
        .expect_err("a second install of the same base must go through the update flow");
    assert!(matches!(err, _ if format!("{err}").contains("already installed")));
}

#[test]
fn the_installation_is_witnessed_as_a_verifiable_chain() {
    let sandbox = Sandbox::new("install-witness");
    let library = sandbox.library();
    let package = sandbox.package("agent.rvf", "network");

    let entry = library
        .install(&InstallRequest::new(&package, "Agent", accept(&["network"])))
        .expect("install");

    assert!(
        entry.record.receipt_id.is_some(),
        "the record names the receipt it was witnessed as"
    );

    let events = lifecycle_events(&library);
    assert!(
        events
            .iter()
            .any(|(s, e, o)| s == &entry.record.install_id && e == "install" && o == "accepted"),
        "the install is recorded: {events:?}"
    );

    // The receipts are chained and the chain verifies here, recomputed rather
    // than reported by the file.
    let view = witness_view::chain_at(library.layout().lifecycle_log().path()).expect("chain");
    assert_eq!(view.label, "valid", "{}", view.summary);
    assert!(view.malformed.is_empty());
}

#[test]
fn a_refused_install_is_witnessed_too() {
    let sandbox = Sandbox::new("install-refusal-witness");
    let library = sandbox.library();
    let package = sandbox.package("agent.rvf", "network");

    let _ = library
        .install(&InstallRequest::new(
            &package,
            "Agent",
            accept(&["network", "gpu"]),
        ))
        .expect_err("over-broad grant");

    assert!(
        recorded(&library, "install", "refused"),
        "a refusal that leaves no record is indistinguishable from an install nobody tried"
    );
}

#[test]
fn the_rollback_contract_is_recorded_at_install_time() {
    let sandbox = Sandbox::new("install-rollback-flag");
    let library = sandbox.library();
    let package = sandbox.package("agent.rvf", "memory");

    let mut request = InstallRequest::new(&package, "Agent", accept(&["memory"]));
    request.state_contract = StateContract {
        state_schema_version: 3,
        rollback_unsafe: true,
    };

    let entry = library.install(&request).expect("install");
    assert!(entry.record.state_contract.rollback_unsafe);
    assert_eq!(entry.record.state_contract.state_schema_version, 3);

    // Declared before installation, per P9 — so it is readable from the record
    // without consulting the release that is being rolled back to.
    let reread = library.layout().read_record(&entry.record.install_id).unwrap();
    assert!(reread.state_contract.rollback_unsafe);
}

#[test]
fn an_install_id_is_derived_from_the_base_identity() {
    let a = install::derive_install_id("sha256:00112233445566778899aabbccddeeff");
    assert_eq!(a, "agent-0011223344556677");
    assert_ne!(
        a,
        install::derive_install_id("sha256:ffeeddccbbaa99887766554433221100"),
        "two different packages never collide on an install directory"
    );
}

#[test]
fn each_refusal_reports_its_own_kind() {
    let sandbox = Sandbox::new("install-typed-errors");
    let layout = sandbox.layout();

    let undeclared = install::install(
        &layout,
        &InstallRequest::new(
            sandbox.package("a.rvf", "network"),
            "Agent",
            accept(&["display"]),
        ),
    )
    .expect_err("undeclared class");
    assert!(matches!(
        undeclared,
        InstallError::GrantNotDeclared { ref class } if class == "display"
    ));

    let missing = install::install(
        &layout,
        &InstallRequest::new(sandbox.root().join("nope.rvf"), "Agent", Vec::new()),
    )
    .expect_err("a path that is not a package");
    assert!(matches!(missing, InstallError::NotVerified { .. }));
}
