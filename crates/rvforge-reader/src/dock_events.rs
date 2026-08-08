//! Event thresholds (ADR-295 §9, requirements D8).
//!
//! A dock that interrupts constantly gets dismissed, and a dismissed trust
//! surface provides no trust. So only five event kinds are allowed to
//! interrupt: approvals, policy violations, cost limits, failures, and
//! meaningful milestones. Everything else accrues into the expanded panel's
//! recent-actions list, where a user who wants detail can find it.
//!
//! The classification lives on the event type rather than at the call site, so
//! a new event kind has to state its threshold to compile.

use std::collections::VecDeque;

use serde::Serialize;

use crate::dock::{sanitize_agent_text, sanitize_system_text, SanitizedText};
use crate::dock_state::DockState;

/// Whether an event is allowed to interrupt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Surfacing {
    /// Raises a notification.
    Notify,
    /// Accrues into recent actions only.
    Silent,
}

/// Everything the dock records.
///
/// Variants carrying agent-authored strings keep them raw here; the log
/// sanitizes on the way in, so nothing agent-authored is stored unscreened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DockEvent {
    // --- the five that surface (ADR-295 §9) ---
    ApprovalRequested {
        agent_id: String,
        summary: String,
    },
    PolicyViolation {
        agent_id: String,
        rule: String,
    },
    CostLimitReached {
        agent_id: String,
        spent_micros: u64,
        limit_micros: u64,
    },
    Failure {
        agent_id: String,
        message: String,
    },
    Milestone {
        agent_id: String,
        description: String,
    },

    // --- everything else ---
    ProgressUpdated {
        agent_id: String,
        progress: u8,
    },
    TaskTextUpdated {
        agent_id: String,
        text: String,
    },
    ToolCall {
        agent_id: String,
        tool: String,
    },
    NetworkActivity {
        agent_id: String,
        peer: String,
    },
    ResourceSample {
        agent_id: String,
    },
    StateChanged {
        agent_id: String,
        from: DockState,
        to: DockState,
    },
    AgentMessage {
        agent_id: String,
        text: String,
    },
}

impl DockEvent {
    pub fn agent_id(&self) -> &str {
        match self {
            DockEvent::ApprovalRequested { agent_id, .. }
            | DockEvent::PolicyViolation { agent_id, .. }
            | DockEvent::CostLimitReached { agent_id, .. }
            | DockEvent::Failure { agent_id, .. }
            | DockEvent::Milestone { agent_id, .. }
            | DockEvent::ProgressUpdated { agent_id, .. }
            | DockEvent::TaskTextUpdated { agent_id, .. }
            | DockEvent::ToolCall { agent_id, .. }
            | DockEvent::NetworkActivity { agent_id, .. }
            | DockEvent::ResourceSample { agent_id }
            | DockEvent::StateChanged { agent_id, .. }
            | DockEvent::AgentMessage { agent_id, .. } => agent_id,
        }
    }

    /// The D8 threshold. Only the first five variants interrupt.
    pub fn surfacing(&self) -> Surfacing {
        match self {
            DockEvent::ApprovalRequested { .. }
            | DockEvent::PolicyViolation { .. }
            | DockEvent::CostLimitReached { .. }
            | DockEvent::Failure { .. }
            | DockEvent::Milestone { .. } => Surfacing::Notify,

            DockEvent::ProgressUpdated { .. }
            | DockEvent::TaskTextUpdated { .. }
            | DockEvent::ToolCall { .. }
            | DockEvent::NetworkActivity { .. }
            | DockEvent::ResourceSample { .. }
            | DockEvent::StateChanged { .. }
            | DockEvent::AgentMessage { .. } => Surfacing::Silent,
        }
    }

    /// `true` when the text in this event was written by the agent, so the
    /// panel can mark the region as agent-authored (ADR-295 §7).
    pub fn is_agent_authored(&self) -> bool {
        matches!(
            self,
            DockEvent::TaskTextUpdated { .. }
                | DockEvent::AgentMessage { .. }
                | DockEvent::Milestone { .. }
        )
    }

    /// The line shown in recent actions. Agent-authored text goes through the
    /// sanitizer; system text is a fixed sentence built from measured values.
    fn render(&self) -> SanitizedText {
        match self {
            // The approval prompt is the one thing an agent must never draw
            // (ADR-295 §Context), so its summary comes from the policy engine
            // and is rendered as chrome, not screened as agent text.
            DockEvent::ApprovalRequested { summary, .. } => system_line(summary),
            DockEvent::Milestone { description, .. } => sanitize_agent_text(description),
            DockEvent::TaskTextUpdated { text, .. } => sanitize_agent_text(text),
            DockEvent::AgentMessage { text, .. } => sanitize_agent_text(text),
            DockEvent::PolicyViolation { rule, .. } => {
                system_line(&format!("Policy violation: {rule}"))
            }
            DockEvent::CostLimitReached {
                spent_micros,
                limit_micros,
                ..
            } => system_line(&format!(
                "Cost limit reached: {} of {} spent",
                money(*spent_micros),
                money(*limit_micros)
            )),
            DockEvent::Failure { message, .. } => system_line(&format!("Failure: {message}")),
            DockEvent::ProgressUpdated { progress, .. } => {
                system_line(&format!("Progress {progress}%"))
            }
            DockEvent::ToolCall { tool, .. } => system_line(&format!("Called tool {tool}")),
            DockEvent::NetworkActivity { peer, .. } => {
                system_line(&format!("Network connection to {peer}"))
            }
            DockEvent::ResourceSample { .. } => system_line("Resource sample recorded"),
            DockEvent::StateChanged { from, to, .. } => {
                system_line(&format!("{} → {}", from.label(), to.label()))
            }
        }
    }
}

/// System text is cleaned but not screened for mimicry: it *is* the system
/// label, so screening it would withhold the dock's own chrome.
fn system_line(text: &str) -> SanitizedText {
    sanitize_system_text(text)
}

fn money(micros: u64) -> String {
    format!(
        "${}.{:02}",
        micros / 1_000_000,
        (micros % 1_000_000) / 10_000
    )
}

/// An event after recording: rendered, screened, and tagged with its origin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecordedEvent {
    pub seq: u64,
    pub agent_id: String,
    pub surfacing: Surfacing,
    /// `true` when `text` came from the agent, so the panel can confine it to
    /// an agent-authored region.
    pub agent_authored: bool,
    pub text: SanitizedText,
}

/// A surfaced event. Only threshold events produce one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Notification {
    pub seq: u64,
    pub agent_id: String,
    pub agent_authored: bool,
    pub text: SanitizedText,
}

/// Bounded recent-actions log with D8 thresholding.
#[derive(Debug, Clone)]
pub struct EventLog {
    recent: VecDeque<RecordedEvent>,
    notifications: Vec<Notification>,
    capacity: usize,
    seq: u64,
}

impl Default for EventLog {
    fn default() -> Self {
        Self::with_capacity(200)
    }
}

impl EventLog {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            recent: VecDeque::new(),
            notifications: Vec::new(),
            capacity: capacity.max(1),
            seq: 0,
        }
    }

    /// Record an event. Every event lands in recent actions; only threshold
    /// events return a [`Notification`].
    pub fn record(&mut self, event: DockEvent) -> Option<Notification> {
        self.seq += 1;
        let surfacing = event.surfacing();
        let recorded = RecordedEvent {
            seq: self.seq,
            agent_id: event.agent_id().to_string(),
            surfacing,
            agent_authored: event.is_agent_authored(),
            text: event.render(),
        };

        if self.recent.len() == self.capacity {
            self.recent.pop_front();
        }
        self.recent.push_back(recorded.clone());

        match surfacing {
            Surfacing::Silent => None,
            Surfacing::Notify => {
                let notification = Notification {
                    seq: recorded.seq,
                    agent_id: recorded.agent_id,
                    agent_authored: recorded.agent_authored,
                    text: recorded.text,
                };
                self.notifications.push(notification.clone());
                Some(notification)
            }
        }
    }

    /// Recent actions for one agent, oldest first.
    pub fn recent_for(&self, agent_id: &str) -> Vec<&RecordedEvent> {
        self.recent
            .iter()
            .filter(|e| e.agent_id == agent_id)
            .collect()
    }

    pub fn recent(&self) -> impl Iterator<Item = &RecordedEvent> {
        self.recent.iter()
    }

    /// Every notification raised so far.
    pub fn notifications(&self) -> &[Notification] {
        &self.notifications
    }

    pub fn notification_count(&self) -> usize {
        self.notifications.len()
    }
}
