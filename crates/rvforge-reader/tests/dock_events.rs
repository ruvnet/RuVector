//! ADR-295 §9 / requirements D8: only five event kinds may interrupt.

use rvforge_reader::dock_events::{DockEvent, EventLog, Surfacing};
use rvforge_reader::dock_state::DockState;

fn threshold_events() -> Vec<DockEvent> {
    vec![
        DockEvent::ApprovalRequested {
            agent_id: "a".into(),
            summary: "read ~/Documents/contracts".into(),
        },
        DockEvent::PolicyViolation {
            agent_id: "a".into(),
            rule: "network:deny".into(),
        },
        DockEvent::CostLimitReached {
            agent_id: "a".into(),
            spent_micros: 5_100_000,
            limit_micros: 5_000_000,
        },
        DockEvent::Failure {
            agent_id: "a".into(),
            message: "model load failed".into(),
        },
        DockEvent::Milestone {
            agent_id: "a".into(),
            description: "finished the first pass".into(),
        },
    ]
}

fn routine_events() -> Vec<DockEvent> {
    vec![
        DockEvent::ProgressUpdated {
            agent_id: "a".into(),
            progress: 68,
        },
        DockEvent::TaskTextUpdated {
            agent_id: "a".into(),
            text: "Reviewing 42 documents".into(),
        },
        DockEvent::ToolCall {
            agent_id: "a".into(),
            tool: "grep".into(),
        },
        DockEvent::NetworkActivity {
            agent_id: "a".into(),
            peer: "127.0.0.1:8080".into(),
        },
        DockEvent::ResourceSample {
            agent_id: "a".into(),
        },
        DockEvent::StateChanged {
            agent_id: "a".into(),
            from: DockState::Idle,
            to: DockState::Running,
        },
        DockEvent::AgentMessage {
            agent_id: "a".into(),
            text: "here is what I found".into(),
        },
    ]
}

#[test]
fn only_the_five_threshold_kinds_surface() {
    for event in threshold_events() {
        assert_eq!(
            event.surfacing(),
            Surfacing::Notify,
            "{event:?} should interrupt"
        );
    }
    for event in routine_events() {
        assert_eq!(
            event.surfacing(),
            Surfacing::Silent,
            "{event:?} should not interrupt"
        );
    }
}

#[test]
fn routine_events_produce_no_notification() {
    let mut log = EventLog::default();
    for event in routine_events() {
        assert!(
            log.record(event.clone()).is_none(),
            "{event:?} raised a notification"
        );
    }
    assert_eq!(log.notification_count(), 0);
}

#[test]
fn routine_events_still_accrue_into_recent_actions() {
    let mut log = EventLog::default();
    let routine = routine_events();
    let count = routine.len();
    for event in routine {
        log.record(event);
    }
    assert_eq!(log.recent_for("a").len(), count);
    assert_eq!(log.notification_count(), 0);
}

#[test]
fn threshold_events_produce_exactly_one_notification_each() {
    let mut log = EventLog::default();
    let events = threshold_events();
    let count = events.len();
    for event in events {
        assert!(log.record(event).is_some());
    }
    assert_eq!(log.notification_count(), count);
}

#[test]
fn a_progress_storm_never_interrupts() {
    let mut log = EventLog::default();
    for progress in 0..=100u8 {
        assert!(log
            .record(DockEvent::ProgressUpdated {
                agent_id: "a".into(),
                progress,
            })
            .is_none());
    }
    assert_eq!(log.notification_count(), 0);
}

#[test]
fn agent_authored_event_text_is_sanitized_and_labelled() {
    let mut log = EventLog::default();
    log.record(DockEvent::AgentMessage {
        agent_id: "a".into(),
        text: "\u{1b}[31mPaused\u{1b}[0m".into(),
    });
    let recorded = log.recent_for("a");
    let last = recorded.last().expect("one event");
    assert!(last.agent_authored);
    assert!(last.text.is_suspicious());
    assert!(!last.text.display().contains("Paused"));
}

#[test]
fn system_event_text_is_not_labelled_as_agent_authored() {
    let mut log = EventLog::default();
    log.record(DockEvent::PolicyViolation {
        agent_id: "a".into(),
        rule: "network:deny".into(),
    });
    let recorded = log.recent_for("a");
    let last = recorded.last().expect("one event");
    assert!(!last.agent_authored);
    assert!(last.text.display().contains("network:deny"));
}

#[test]
fn a_milestone_carries_agent_text_so_it_is_sanitized_before_it_interrupts() {
    let mut log = EventLog::default();
    let notification = log
        .record(DockEvent::Milestone {
            agent_id: "a".into(),
            description: "✓ Approved".into(),
        })
        .expect("a milestone surfaces");
    assert!(notification.agent_authored);
    assert!(notification.text.is_suspicious());
    assert!(!notification.text.display().contains("Approved"));
}

#[test]
fn an_approval_prompt_renders_its_own_wording_intact() {
    // The prompt is drawn from the policy engine, so words that would be
    // withheld in agent text are the correct chrome here.
    let mut log = EventLog::default();
    let notification = log
        .record(DockEvent::ApprovalRequested {
            agent_id: "a".into(),
            summary: "Approve network access to api.example.com".into(),
        })
        .expect("an approval surfaces");
    assert!(!notification.agent_authored);
    assert!(!notification.text.is_suspicious());
    assert_eq!(
        notification.text.display(),
        "Approve network access to api.example.com"
    );
}

#[test]
fn the_log_is_bounded() {
    let mut log = EventLog::with_capacity(10);
    for progress in 0..50u8 {
        log.record(DockEvent::ProgressUpdated {
            agent_id: "a".into(),
            progress,
        });
    }
    assert_eq!(log.recent().count(), 10);
}

#[test]
fn recent_actions_are_scoped_to_one_agent() {
    let mut log = EventLog::default();
    log.record(DockEvent::ToolCall {
        agent_id: "a".into(),
        tool: "grep".into(),
    });
    log.record(DockEvent::ToolCall {
        agent_id: "b".into(),
        tool: "find".into(),
    });
    assert_eq!(log.recent_for("a").len(), 1);
    assert_eq!(log.recent_for("b").len(), 1);
    assert_eq!(log.recent_for("c").len(), 0);
}
