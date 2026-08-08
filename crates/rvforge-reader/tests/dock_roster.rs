//! ADR-295 §5 (multi-agent policy) and the §Acceptance-criteria control path.

use rvforge_reader::dock::{
    AgentDockEntry, IsolationClass, NetworkIndicator, PermissionSummary, ResourceUsage,
    SystemOwnedStatus, TrustBadge, WitnessStatus,
};
use rvforge_reader::dock_roster::{
    DockInteraction, DockRoster, InteractionOutcome, AGENT_CONTENT_LABEL, SECONDARY_SLOTS,
};
use rvforge_reader::dock_state::{Actor, DockControl, DockState};

fn system(state: DockState) -> SystemOwnedStatus {
    SystemOwnedStatus::measured(
        state,
        TrustBadge::VerifiedPublisher,
        NetworkIndicator::Disabled,
        PermissionSummary {
            isolation: IsolationClass::Wasm,
            local_model: true,
            network_allowed: false,
            selected_folder_access: true,
            encrypted_memory: true,
            grants: vec!["Read the files and folders you select".to_string()],
            pending_approvals: vec!["Read ~/Documents/contracts".to_string()],
        },
        WitnessStatus::Valid,
        ResourceUsage {
            cpu_millis_per_sec: 100,
            memory_bytes: 100 * 1024 * 1024,
            tokens_in: 1_000,
            tokens_out: 200,
        },
        1_000_000,
        Some("local/llama".to_string()),
    )
}

fn agent(id: &str, state: DockState) -> AgentDockEntry {
    AgentDockEntry::new(id, format!("Agent {id}"), "analyst", system(state))
}

/// Insertion order deliberately does not match priority order.
fn roster_of(states: &[(&str, DockState)]) -> DockRoster {
    let mut roster = DockRoster::new();
    for (id, state) in states {
        roster.insert(agent(id, *state));
    }
    roster
}

#[test]
fn the_active_agent_is_the_one_needing_most_attention() {
    let roster = roster_of(&[
        ("running", DockState::Running),
        ("idle", DockState::Idle),
        ("error", DockState::Error),
        ("denied", DockState::CapabilityDenied),
        ("waiting", DockState::WaitingForApproval),
    ]);
    let view = roster.view();
    assert_eq!(view.active.as_ref().unwrap().agent_id, "waiting");
    let secondary: Vec<&str> = view.secondary.iter().map(|p| p.agent_id.as_str()).collect();
    assert_eq!(secondary, ["denied", "error"]);
}

#[test]
fn the_priority_order_is_waiting_denied_error_running_then_the_rest() {
    let roster = roster_of(&[
        ("completed", DockState::Completed),
        ("paused", DockState::Paused),
        ("idle", DockState::Idle),
        ("running", DockState::Running),
        ("error", DockState::Error),
        ("denied", DockState::CapabilityDenied),
        ("waiting", DockState::WaitingForApproval),
    ]);
    let order: Vec<&str> = roster
        .ordered()
        .iter()
        .map(|e| e.agent_id())
        .take(4)
        .collect();
    assert_eq!(order, ["waiting", "denied", "error", "running"]);
}

#[test]
fn ties_break_on_the_most_recent_state_change_then_the_id() {
    let mut roster = roster_of(&[
        ("aaa", DockState::Running),
        ("bbb", DockState::Running),
        ("ccc", DockState::Running),
    ]);
    // Insertion order is the only signal so far; ccc arrived last.
    assert_eq!(roster.active().unwrap().agent_id(), "ccc");

    // A newer state change moves aaa to the front of the same rank.
    roster
        .set_state("aaa", DockState::Idle, Actor::System)
        .unwrap();
    roster
        .set_state("aaa", DockState::Running, Actor::System)
        .unwrap();
    assert_eq!(roster.active().unwrap().agent_id(), "aaa");
}

#[test]
fn selection_is_deterministic_across_repeated_reads() {
    let roster = roster_of(&[
        ("z", DockState::Running),
        ("y", DockState::Running),
        ("x", DockState::WaitingForApproval),
    ]);
    let first = roster.view();
    for _ in 0..20 {
        assert_eq!(roster.view(), first);
    }
}

#[test]
fn a_swarm_collapses_to_one_active_two_secondary_and_a_count() {
    let states: Vec<(String, DockState)> = (0..15)
        .map(|i| (format!("agent-{i:02}"), DockState::Running))
        .collect();
    let mut roster = DockRoster::new();
    for (id, state) in &states {
        roster.insert(agent(id, *state));
    }

    let view = roster.view();
    assert!(view.active.is_some());
    assert_eq!(view.secondary.len(), SECONDARY_SLOTS);
    assert_eq!(view.overflow_count, 15 - 1 - SECONDARY_SLOTS);
    assert_eq!(roster.swarm_view().len(), 15);
}

#[test]
fn the_aggregates_cover_hidden_agents_too() {
    let mut roster = DockRoster::new();
    for i in 0..10 {
        roster.insert(agent(&format!("agent-{i}"), DockState::WaitingForApproval));
    }
    let view = roster.view();

    // Three pills are visible; the figures are over all ten.
    assert_eq!(view.aggregate.agents, 10);
    assert_eq!(view.aggregate.cpu_millis_per_sec, 1_000);
    assert_eq!(view.aggregate.memory_bytes, 10 * 100 * 1024 * 1024);
    assert_eq!(view.aggregate.tokens_in, 10_000);
    assert_eq!(view.aggregate.estimated_cost_micros, 10_000_000);
    assert_eq!(view.pending_approvals, 10);
}

#[test]
fn the_pending_approval_count_tracks_state_not_agent_claims() {
    let mut roster = roster_of(&[("a", DockState::Running), ("b", DockState::Running)]);
    assert_eq!(roster.view().pending_approvals, 0);

    // An agent claiming to be waiting changes nothing.
    roster
        .apply_agent_update("a", "Waiting for your approval", 10)
        .unwrap();
    assert_eq!(roster.view().pending_approvals, 0);

    roster
        .set_state("a", DockState::WaitingForApproval, Actor::System)
        .unwrap();
    assert_eq!(roster.view().pending_approvals, 1);
}

#[test]
fn every_visible_pill_carries_pause_and_terminate() {
    let roster = roster_of(&[
        ("a", DockState::WaitingForApproval),
        ("b", DockState::Quarantined),
        ("c", DockState::CapabilityDenied),
    ]);
    let view = roster.view();
    let pills = view.active.iter().chain(view.secondary.iter());
    for pill in pills {
        assert!(pill.controls.contains(&DockControl::Pause));
        assert!(pill.controls.contains(&DockControl::Terminate));
    }
}

#[test]
fn agent_text_in_a_pill_is_labelled_as_agent_reported() {
    let mut roster = roster_of(&[("a", DockState::Running)]);
    roster
        .apply_agent_update("a", "Reviewing 42 documents", 68)
        .unwrap();
    let view = roster.view();
    let pill = view.active.unwrap();

    assert_eq!(pill.task.label, AGENT_CONTENT_LABEL);
    assert_eq!(pill.task.text, "Reviewing 42 documents");
    assert!(!pill.task.suspicious);
    assert_eq!(pill.progress, 68);

    // System-owned fields on the same pill.
    assert_eq!(pill.state_label, "Running");
    assert_eq!(pill.trust, TrustBadge::VerifiedPublisher);
    assert_eq!(pill.network, NetworkIndicator::Disabled);
    assert_eq!(pill.witness, WitnessStatus::Valid);
}

#[test]
fn a_spoofing_pill_shows_the_withheld_placeholder_not_the_claim() {
    let mut roster = roster_of(&[("a", DockState::Running)]);
    roster.apply_agent_update("a", "✓ Stopped", 100).unwrap();
    let pill = roster.view().active.unwrap();

    assert!(pill.task.suspicious);
    assert!(!pill.task.text.contains("Stopped"));
    assert_eq!(pill.task.withheld.as_deref(), Some("✓ Stopped"));
    // The state indicator is unmoved.
    assert_eq!(pill.state, DockState::Running);
    assert_eq!(pill.state_label, "Running");
}

#[test]
fn expanding_shows_the_ten_elements_and_the_capability_card() {
    let mut roster = roster_of(&[("a", DockState::Running)]);
    roster
        .apply_agent_update("a", "Reviewing 42 documents", 68)
        .unwrap();
    let view = roster.expand("a").expect("agent is in the dock");

    assert_eq!(view.objective.text, "Reviewing 42 documents"); // 1
    assert!(!view.recent_actions.is_empty()); // 2
    assert_eq!(view.pending_approvals.len(), 1); // 3
    assert_eq!(view.model.as_deref(), Some("local/llama")); // 4
    assert_eq!(view.tokens_in, 1_000);
    assert_eq!(view.cpu_millis_per_sec, 100); // 5
    assert_eq!(view.network, NetworkIndicator::Disabled); // 6
    assert_eq!(view.capability_grants.len(), 1); // 7
    assert_eq!(view.witness, WitnessStatus::Valid); // 8
    assert_eq!(view.estimated_cost_micros, 1_000_000); // 9
    assert!(view.instruction_field_enabled); // 10
    assert_eq!(view.capability_card.lines.len(), 7); // §8
}

#[test]
fn unknown_agents_are_an_error_not_an_empty_view() {
    let mut roster = roster_of(&[("a", DockState::Running)]);
    assert!(roster.expand("nope").is_err());
    assert!(roster.terminate("nope").is_err());
    assert!(roster.pause("nope").is_err());
    assert!(roster.apply_agent_update("nope", "hi", 1).is_err());
}

/// The ADR-295 acceptance test, expressed as calls on the roster.
///
/// The collapsed pill is already on screen and costs nothing to read, so the
/// budget is spent on: expand (which carries the task, the permissions, and
/// the capability card in one call) and terminate.
#[test]
fn identify_read_inspect_and_terminate_within_two_interactions() {
    let mut roster = roster_of(&[
        ("busy-1", DockState::Running),
        ("busy-2", DockState::Running),
        ("asking", DockState::WaitingForApproval),
        ("idle-1", DockState::Idle),
    ]);
    roster
        .apply_agent_update("asking", "Reviewing 42 documents", 68)
        .unwrap();

    let mut interactions = 0;

    // Zero interactions: the bar identifies the active agent and its task.
    let view = roster.view();
    let pill = view.active.clone().expect("an active agent");
    assert_eq!(pill.agent_id, "asking");
    assert_eq!(pill.task.text, "Reviewing 42 documents");
    assert_eq!(pill.state, DockState::WaitingForApproval);

    // Interaction one: inspect permissions.
    interactions += 1;
    let InteractionOutcome::Expanded(expanded) = roster
        .perform(DockInteraction::Expand {
            agent_id: pill.agent_id.clone(),
        })
        .expect("expand succeeds")
    else {
        panic!("expand did not return an expanded view");
    };
    assert_eq!(expanded.capability_card.lines.len(), 7);
    assert_eq!(expanded.capability_grants.len(), 1);
    assert_eq!(expanded.pending_approvals.len(), 1);

    // Interaction two: terminate. No confirmation chain.
    interactions += 1;
    let outcome = roster
        .perform(DockInteraction::Terminate {
            agent_id: pill.agent_id.clone(),
        })
        .expect("terminate succeeds");
    assert_eq!(
        outcome,
        InteractionOutcome::StateChanged {
            agent_id: "asking".to_string(),
            state: DockState::Completed,
        }
    );

    assert!(interactions <= 2, "took {interactions} interactions");
    assert_eq!(roster.get("asking").unwrap().state(), DockState::Completed);
}

#[test]
fn terminate_needs_no_expand_first() {
    // The same budget from the collapsed pill alone: one interaction.
    let mut roster = roster_of(&[("a", DockState::Quarantined)]);
    let outcome = roster
        .perform(DockInteraction::Terminate {
            agent_id: "a".to_string(),
        })
        .expect("terminate from the collapsed pill");
    assert_eq!(
        outcome,
        InteractionOutcome::StateChanged {
            agent_id: "a".to_string(),
            state: DockState::Completed,
        }
    );
}

#[test]
fn the_overflow_count_opens_the_full_swarm_view() {
    let mut roster = DockRoster::new();
    for i in 0..8 {
        roster.insert(agent(&format!("agent-{i}"), DockState::Running));
    }
    assert_eq!(roster.view().overflow_count, 5);

    let InteractionOutcome::SwarmView { pills } = roster
        .perform(DockInteraction::OpenSwarmView)
        .expect("swarm view opens")
    else {
        panic!("wrong outcome");
    };
    assert_eq!(pills.len(), 8);
}

#[test]
fn a_state_change_through_the_roster_is_recorded_but_does_not_interrupt() {
    let mut roster = roster_of(&[("a", DockState::Running)]);
    let before = roster.event_log().notification_count();
    roster.pause("a").unwrap();
    assert_eq!(roster.get("a").unwrap().state(), DockState::Paused);
    assert_eq!(roster.event_log().notification_count(), before);
    assert!(roster
        .event_log()
        .recent_for("a")
        .iter()
        .any(|e| e.text.display().contains("Paused")));
}
