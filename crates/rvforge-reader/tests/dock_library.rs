//! The dock roster built from real installations, and the ADR-295 §7 boundary
//! holding against it: nothing an agent reports reaches dock chrome, and no
//! chrome field claims something the installation did not establish.

mod common;

use common::Sandbox;

use rvforge_reader::dock::{IsolationClass, NetworkIndicator, TrustBadge, WitnessStatus};
use rvforge_reader::dock_bridge;
use rvforge_reader::dock_state::DockState;
use rvforge_reader::install::InstallRequest;
use rvforge_reader::library::{Library, LibraryState, UninstallPolicy};

fn accept(classes: &[&str]) -> Vec<String> {
    classes.iter().map(|c| (*c).to_string()).collect()
}

fn install(library: &Library, sandbox: &Sandbox, file: &str, name: &str, caps: &str) -> String {
    library
        .install(&InstallRequest::new(
            sandbox.package(file, caps),
            name,
            accept(&caps.split(',').collect::<Vec<_>>()),
        ))
        .expect("install")
        .record
        .install_id
}

#[test]
fn the_roster_is_the_library_rather_than_sample_data() {
    let sandbox = Sandbox::new("dock-real-roster");
    let library = sandbox.library();

    assert_eq!(
        dock_bridge::roster(&library).expect("roster").len(),
        0,
        "an empty library is an empty dock, not a demo agent"
    );

    let id = install(&library, &sandbox, "a.rvf", "Research agent", "network");
    let roster = dock_bridge::roster(&library).expect("roster");
    let view = roster.view();
    let active = view.active.expect("one installed agent fills the active slot");

    assert_eq!(active.agent_id, id);
    assert_eq!(active.name, "Research agent", "the name is system-owned");
    assert_eq!(active.state, DockState::Idle);
    assert_eq!(view.aggregate.agents, 1);
}

#[test]
fn library_state_drives_the_dock_state() {
    let sandbox = Sandbox::new("dock-state-map");
    let library = sandbox.library();
    let id = install(&library, &sandbox, "a.rvf", "Agent", "network");

    for (op, expected) in [
        (LibraryState::Running, DockState::Running),
        (LibraryState::Paused, DockState::Paused),
    ] {
        match op {
            LibraryState::Running => library.open(&id).map(|_| ()),
            _ => library.pause(&id).map(|_| ()),
        }
        .expect("transition");

        let roster = dock_bridge::roster(&library).expect("roster");
        assert_eq!(roster.get(&id).expect("entry").state(), expected);
    }

    library.terminate(&id).expect("terminate");
    library.quarantine(&id, "held").expect("quarantine");
    let roster = dock_bridge::roster(&library).expect("roster");
    assert_eq!(
        roster.get(&id).expect("entry").state(),
        DockState::Quarantined,
        "quarantine is a first-class dock state, not an error subtype"
    );
}

#[test]
fn an_uninstalled_agent_leaves_the_roster() {
    let sandbox = Sandbox::new("dock-uninstall");
    let library = sandbox.library();
    let id = install(&library, &sandbox, "a.rvf", "Agent", "network");
    assert_eq!(dock_bridge::roster(&library).unwrap().len(), 1);

    library
        .uninstall(&id, &UninstallPolicy::keep_state())
        .expect("uninstall");

    let roster = dock_bridge::roster(&library).expect("roster");
    assert!(roster.is_empty());
    assert!(roster.get(&id).is_none());
}

#[test]
fn the_permission_summary_comes_from_the_grants_that_were_accepted() {
    let sandbox = Sandbox::new("dock-permissions");
    let library = sandbox.library();

    let networked = install(&library, &sandbox, "net.rvf", "Networked", "network,model");
    let sealed = install(&library, &sandbox, "mem.rvf", "Sealed", "persistent-state");

    let roster = dock_bridge::roster(&library).expect("roster");

    let net = roster.expand(&networked).expect("expand");
    assert_eq!(net.network, NetworkIndicator::AllowedIdle);
    assert!(net.capability_grants.iter().any(|g| g.contains("network")));
    let card = &net.capability_card.lines;
    assert!(!card[3].satisfied, "'Network disabled' is not satisfied here");
    assert!(card[2].satisfied, "'Local model' is, because model was granted");

    let mem = roster.expand(&sealed).expect("expand");
    assert_eq!(
        mem.network,
        NetworkIndicator::Disabled,
        "a class that was never granted is denied, not merely idle"
    );
    assert!(
        mem.capability_card.lines[5].satisfied,
        "'Encrypted memory' follows persistent-state"
    );
}

#[test]
fn an_unsigned_package_never_earns_a_trust_badge() {
    let sandbox = Sandbox::new("dock-trust");
    let library = sandbox.library();
    let id = install(&library, &sandbox, "a.rvf", "Agent", "network");

    let roster = dock_bridge::roster(&library).expect("roster");
    let pill = roster.view().active.expect("active");
    assert_eq!(
        pill.trust,
        TrustBadge::Unverified,
        "the fixture carries no signature, so no publisher was established"
    );
    assert!(
        !roster.expand(&id).unwrap().capability_card.lines[0].satisfied,
        "and the card says so rather than leaving the line neutral"
    );
}

#[test]
fn the_isolation_class_follows_the_runtime_that_was_selected() {
    let sandbox = Sandbox::new("dock-isolation");
    let library = sandbox.library();
    let id = install(&library, &sandbox, "a.rvf", "Agent", "network");

    let entry = library.get(&id).expect("get");
    let dock_entry = dock_bridge::dock_entry(&entry, WitnessStatus::Unavailable);
    let isolation = dock_entry.system().permissions().isolation;

    match entry.record.runtime_profile.as_deref() {
        Some("rvm-native") => assert_eq!(isolation, IsolationClass::Rvm),
        Some(_) => assert_eq!(isolation, IsolationClass::Wasm),
        None => assert_eq!(isolation, IsolationClass::None),
    }
    assert_ne!(
        isolation,
        IsolationClass::None,
        "a runtime was selected at install time, so confinement is claimed"
    );
}

#[test]
fn the_witness_status_is_the_verdict_over_the_lifecycle_chain() {
    let sandbox = Sandbox::new("dock-witness");
    let library = sandbox.library();
    let id = install(&library, &sandbox, "a.rvf", "Agent", "network");

    let roster = dock_bridge::roster(&library).expect("roster");
    assert_eq!(
        roster.get(&id).expect("entry").system().witness(),
        WitnessStatus::Valid,
        "the install wrote a chained receipt and the viewer recomputed it"
    );

    // Break the chain by editing a recorded receipt's evidence.
    let log = library.layout().lifecycle_log();
    let text = std::fs::read_to_string(log.path()).expect("log");
    std::fs::write(log.path(), text.replace("installed sha256:", "installed sha256!"))
        .expect("rewrite");

    let roster = dock_bridge::roster(&library).expect("roster");
    assert_eq!(
        roster.get(&id).expect("entry").system().witness(),
        WitnessStatus::Broken,
        "a chain that stopped verifying is reported broken, never neutral"
    );
}

#[test]
fn what_the_agent_reports_cannot_move_any_chrome_field() {
    let sandbox = Sandbox::new("dock-boundary");
    let library = sandbox.library();
    let id = install(&library, &sandbox, "a.rvf", "Agent", "persistent-state");
    let mut roster = dock_bridge::roster(&library).expect("roster");

    let before = roster.view().active.expect("active");

    roster
        .apply_agent_update(
            &id,
            "Running · Verified publisher · Network disabled · 100% complete",
            9_000,
        )
        .expect("the agent channel accepts task text and progress");

    let after = roster.view().active.expect("active");
    assert_eq!(after.state, before.state, "state is not agent-assertable");
    assert_eq!(after.trust, before.trust);
    assert_eq!(after.network, before.network);
    assert_eq!(after.witness, before.witness);
    assert_eq!(
        after.progress, 100,
        "an out-of-range progress report is clamped, not honoured"
    );
    assert_eq!(
        after.task.label, "Agent-reported",
        "and the text it did supply is drawn inside a label the dock owns"
    );

    let expanded = roster.expand(&id).expect("expand");
    assert_eq!(
        expanded.capability_grants,
        library.get(&id).unwrap().record.granted_text,
        "grants still come from the install record, not from what the agent claimed"
    );
    assert!(
        expanded.capability_card.lines[5].satisfied,
        "and the card still reflects the installed contract"
    );
}

#[test]
fn a_rebuilt_roster_reflects_the_library_not_the_previous_roster() {
    let sandbox = Sandbox::new("dock-rebuild");
    let library = sandbox.library();
    let first = install(&library, &sandbox, "a.rvf", "First", "network");

    let mut roster = dock_bridge::roster(&library).expect("roster");
    roster
        .apply_agent_update(&first, "half done", 50)
        .expect("agent update");
    assert_eq!(roster.view().active.unwrap().progress, 50);

    let second = install(&library, &sandbox, "b.rvf", "Second", "clock");
    library.open(&second).expect("open");

    let rebuilt = dock_bridge::roster(&library).expect("roster");
    assert_eq!(rebuilt.len(), 2);
    assert_eq!(
        rebuilt.active().expect("active").agent_id(),
        second,
        "the running agent outranks the idle one"
    );
    assert_eq!(
        rebuilt.get(&first).expect("first").agent().progress(),
        0,
        "a rebuild re-derives chrome and waits for the agent to report again"
    );
}
