//! ADR-295 §3 and §4: which state changes are legal, and who may make them.

use rvforge_reader::dock_state::{
    pause, terminate, transition, Actor, DockControl, DockState, TransitionError, DOCK_STATES,
};

#[test]
fn there_are_exactly_eight_states_and_each_is_visually_distinct() {
    assert_eq!(DOCK_STATES.len(), 8);

    let mut colors: Vec<&str> = DOCK_STATES.iter().map(|s| s.color_token()).collect();
    colors.sort_unstable();
    colors.dedup();
    assert_eq!(colors.len(), 8, "two states share a color token");

    let mut glyphs: Vec<&str> = DOCK_STATES.iter().map(|s| s.glyph()).collect();
    glyphs.sort_unstable();
    glyphs.dedup();
    assert_eq!(glyphs.len(), 8, "two states share a glyph");
}

#[test]
fn capability_denied_and_quarantined_are_not_error_subtypes() {
    assert_ne!(DockState::CapabilityDenied, DockState::Error);
    assert_ne!(DockState::Quarantined, DockState::Error);
    assert_ne!(
        DockState::CapabilityDenied.color_token(),
        DockState::Error.color_token()
    );
    assert_ne!(
        DockState::Quarantined.color_token(),
        DockState::Error.color_token()
    );
}

#[test]
fn legal_transitions_are_accepted() {
    for (from, to) in [
        (DockState::Idle, DockState::Running),
        (DockState::Running, DockState::WaitingForApproval),
        (DockState::WaitingForApproval, DockState::Running),
        (DockState::Running, DockState::CapabilityDenied),
        (DockState::CapabilityDenied, DockState::Running),
        (DockState::Paused, DockState::Running),
        (DockState::Error, DockState::Running),
        (DockState::Quarantined, DockState::Idle),
        (DockState::Running, DockState::Completed),
    ] {
        assert_eq!(
            transition(from, to, Actor::System),
            Ok(to),
            "{from:?} -> {to:?} should be legal for the system"
        );
    }
}

#[test]
fn illegal_transitions_are_rejected() {
    for (from, to) in [
        // A capability denial implies a request, which implies execution.
        (DockState::Idle, DockState::CapabilityDenied),
        (DockState::Paused, DockState::CapabilityDenied),
        // Only a running agent can ask for approval.
        (DockState::Idle, DockState::WaitingForApproval),
        (DockState::Paused, DockState::WaitingForApproval),
        // Quarantine releases to Idle or Completed, never straight back to work.
        (DockState::Quarantined, DockState::Running),
        (DockState::Quarantined, DockState::Paused),
    ] {
        assert!(
            matches!(
                transition(from, to, Actor::System),
                Err(TransitionError::Illegal { .. })
            ),
            "{from:?} -> {to:?} should be illegal"
        );
    }
}

#[test]
fn completed_is_terminal_for_every_actor() {
    for actor in [Actor::System, Actor::User, Actor::Agent] {
        for to in DOCK_STATES {
            assert!(
                matches!(
                    transition(DockState::Completed, to, actor),
                    Err(TransitionError::Terminal { .. })
                ),
                "{actor:?} moved a completed agent to {to:?}"
            );
        }
    }
    assert!(matches!(
        pause(DockState::Completed),
        Err(TransitionError::Terminal { .. })
    ));
    assert!(matches!(
        terminate(DockState::Completed),
        Err(TransitionError::Terminal { .. })
    ));
}

#[test]
fn an_agent_cannot_claim_it_is_paused() {
    // The exact spoof of ADR-295 §Context: rendering "Paused" while executing.
    assert!(matches!(
        transition(DockState::Running, DockState::Paused, Actor::Agent),
        Err(TransitionError::AgentCannotEnter {
            to: DockState::Paused
        })
    ));
    // The user and the runtime can.
    assert_eq!(
        transition(DockState::Running, DockState::Paused, Actor::User),
        Ok(DockState::Paused)
    );
}

#[test]
fn an_agent_cannot_put_itself_in_a_system_forced_state() {
    for to in [
        DockState::Paused,
        DockState::CapabilityDenied,
        DockState::Quarantined,
    ] {
        assert!(
            matches!(
                transition(DockState::Running, to, Actor::Agent),
                Err(TransitionError::AgentCannotEnter { .. })
            ),
            "an agent asserted {to:?}"
        );
        assert!(to.is_system_forced());
        assert!(!to.is_agent_assertable());
    }
}

#[test]
fn an_agent_cannot_exit_quarantine() {
    for to in DOCK_STATES {
        assert!(
            matches!(
                transition(DockState::Quarantined, to, Actor::Agent),
                Err(TransitionError::AgentCannotLeave {
                    from: DockState::Quarantined
                })
            ),
            "an agent left quarantine for {to:?}"
        );
    }
    // The runtime can release it.
    assert_eq!(
        transition(DockState::Quarantined, DockState::Idle, Actor::System),
        Ok(DockState::Idle)
    );
}

#[test]
fn an_agent_cannot_exit_a_capability_denial_or_a_pause_by_itself() {
    for from in [DockState::CapabilityDenied, DockState::Paused] {
        assert!(matches!(
            transition(from, DockState::Running, Actor::Agent),
            Err(TransitionError::AgentCannotLeave { .. })
        ));
        assert_eq!(
            transition(from, DockState::Running, Actor::User),
            Ok(DockState::Running)
        );
    }
}

#[test]
fn pause_and_terminate_are_legal_from_every_non_terminal_state() {
    for from in DOCK_STATES.iter().copied().filter(|s| !s.is_terminal()) {
        assert_eq!(pause(from), Ok(DockState::Paused), "cannot pause {from:?}");
        assert_eq!(
            terminate(from),
            Ok(DockState::Completed),
            "cannot terminate {from:?}"
        );
    }
}

#[test]
fn pause_and_terminate_are_one_action_in_every_non_terminal_state() {
    for state in DOCK_STATES.iter().copied().filter(|s| !s.is_terminal()) {
        let controls = state.controls();
        assert!(
            controls.contains(&DockControl::Pause),
            "{state:?} hides Pause"
        );
        assert!(
            controls.contains(&DockControl::Terminate),
            "{state:?} hides Terminate"
        );
        assert!(
            controls.contains(&DockControl::Expand),
            "{state:?} hides Expand"
        );
    }
    assert!(DockState::Paused.controls().contains(&DockControl::Resume));
    assert_eq!(DockState::Completed.controls(), vec![DockControl::Expand]);
}

#[test]
fn pausing_an_already_paused_agent_succeeds() {
    // A control that sometimes fails is a control users stop reaching for.
    assert_eq!(pause(DockState::Paused), Ok(DockState::Paused));
}

#[test]
fn no_state_transitions_to_itself() {
    for state in DOCK_STATES {
        assert!(!state.may_precede(state), "{state:?} precedes itself");
    }
}
