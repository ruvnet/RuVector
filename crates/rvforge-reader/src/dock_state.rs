//! Dock state machine (ADR-295 §3, §4).
//!
//! The eight dock states are **system-owned**. This module is the only place a
//! [`DockState`] may change, and every change carries the [`Actor`] that asked
//! for it, because the actor is what makes a transition legal or not:
//!
//! - `Paused`, `CapabilityDenied`, and `Quarantined` are system-forced. An
//!   agent can neither enter them (the "I have stopped" spoof of ADR-295
//!   §Context) nor leave them (escaping quarantine by asserting a new state).
//! - Pause and terminate are legal for the user from **every** non-terminal
//!   state and are a single call — never behind a confirmation chain
//!   (ADR-295 §4).
//! - `Completed` is terminal. Nothing leaves it, including the runtime.

use serde::Serialize;

/// The eight ADR-295 §3 states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DockState {
    Idle,
    Running,
    WaitingForApproval,
    Paused,
    CapabilityDenied,
    Error,
    Quarantined,
    Completed,
}

/// Every state, in the ADR-295 §3 listing order.
pub const DOCK_STATES: [DockState; 8] = [
    DockState::Idle,
    DockState::Running,
    DockState::WaitingForApproval,
    DockState::Paused,
    DockState::CapabilityDenied,
    DockState::Error,
    DockState::Quarantined,
    DockState::Completed,
];

/// Who is asking for a transition.
///
/// `Agent` is the untrusted channel: it is the only actor whose requests are
/// filtered by anything other than the transition table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Actor {
    /// The runtime (RVM), reporting measured facts.
    System,
    /// The person at the dock.
    User,
    /// The agent itself.
    Agent,
}

/// Controls the dock offers for a state. Pause and Terminate appear in every
/// non-terminal state, collapsed and expanded alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DockControl {
    Pause,
    Resume,
    Terminate,
    Expand,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum TransitionError {
    /// Not in the transition table for any actor.
    Illegal { from: DockState, to: DockState },
    /// The state is terminal; nothing leaves it.
    Terminal { from: DockState },
    /// An agent tried to enter a state only the system may assert.
    AgentCannotEnter { to: DockState },
    /// An agent tried to leave a state the system forced on it.
    AgentCannotLeave { from: DockState },
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransitionError::Illegal { from, to } => {
                write!(f, "{} cannot transition to {}", from.label(), to.label())
            }
            TransitionError::Terminal { from } => {
                write!(f, "{} is terminal", from.label())
            }
            TransitionError::AgentCannotEnter { to } => write!(
                f,
                "an agent cannot put itself into '{}'; that state is asserted by the runtime",
                to.label()
            ),
            TransitionError::AgentCannotLeave { from } => write!(
                f,
                "an agent cannot leave '{}'; only the runtime releases it",
                from.label()
            ),
        }
    }
}

impl std::error::Error for TransitionError {}

impl DockState {
    /// The label rendered in system chrome.
    pub fn label(self) -> &'static str {
        match self {
            DockState::Idle => "Idle",
            DockState::Running => "Running",
            DockState::WaitingForApproval => "Waiting for approval",
            DockState::Paused => "Paused",
            DockState::CapabilityDenied => "Capability denied",
            DockState::Error => "Error",
            DockState::Quarantined => "Quarantined",
            DockState::Completed => "Completed",
        }
    }

    /// Stable per-state color token. Distinguishable at pill size, so
    /// `CapabilityDenied` and `Quarantined` never share a hue with `Error`
    /// (ADR-295 §3).
    pub fn color_token(self) -> &'static str {
        match self {
            DockState::Idle => "state-idle",
            DockState::Running => "state-running",
            DockState::WaitingForApproval => "state-waiting",
            DockState::Paused => "state-paused",
            DockState::CapabilityDenied => "state-denied",
            DockState::Error => "state-error",
            DockState::Quarantined => "state-quarantined",
            DockState::Completed => "state-completed",
        }
    }

    /// Glyph shown beside the color, so the state survives a color-blind
    /// reading and a greyscale screenshot.
    pub fn glyph(self) -> &'static str {
        match self {
            DockState::Idle => "○",
            DockState::Running => "▶",
            DockState::WaitingForApproval => "?",
            DockState::Paused => "❚❚",
            DockState::CapabilityDenied => "⊘",
            DockState::Error => "!",
            DockState::Quarantined => "▧",
            DockState::Completed => "✔",
        }
    }

    /// Nothing leaves a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, DockState::Completed)
    }

    /// States the runtime forces and the agent cannot argue with. Both are
    /// first-class outcomes of the trust system rather than error subtypes
    /// (ADR-295 §3).
    pub fn is_system_forced(self) -> bool {
        matches!(
            self,
            DockState::Paused | DockState::CapabilityDenied | DockState::Quarantined
        )
    }

    /// States an agent is allowed to claim for itself. Everything outside this
    /// set carries a security meaning and is drawn from runtime state only.
    pub fn is_agent_assertable(self) -> bool {
        matches!(
            self,
            DockState::Idle
                | DockState::Running
                | DockState::WaitingForApproval
                | DockState::Error
                | DockState::Completed
        )
    }

    /// The transition table, ignoring actor. `Quarantined` releases only to
    /// `Idle` or `Completed`; `CapabilityDenied` is reachable only from states
    /// in which a capability could have been requested.
    pub fn may_precede(self, to: DockState) -> bool {
        use DockState::*;
        if self == to {
            return false;
        }
        match self {
            Idle => matches!(to, Running | Paused | Error | Quarantined | Completed),
            Running => matches!(
                to,
                Idle | WaitingForApproval
                    | Paused
                    | CapabilityDenied
                    | Error
                    | Quarantined
                    | Completed
            ),
            WaitingForApproval => matches!(
                to,
                Running | Paused | CapabilityDenied | Error | Quarantined | Completed
            ),
            Paused => matches!(to, Running | Idle | Error | Quarantined | Completed),
            CapabilityDenied => matches!(
                to,
                Running | Idle | Paused | Error | Quarantined | Completed
            ),
            Error => matches!(to, Idle | Running | Paused | Quarantined | Completed),
            Quarantined => matches!(to, Idle | Completed),
            Completed => false,
        }
    }

    /// Controls the dock renders for this state. Pause and Terminate are in
    /// every non-terminal list, so terminate is always one action away.
    pub fn controls(self) -> Vec<DockControl> {
        if self.is_terminal() {
            return vec![DockControl::Expand];
        }
        let mut controls = vec![
            DockControl::Pause,
            DockControl::Terminate,
            DockControl::Expand,
        ];
        if self == DockState::Paused {
            controls.insert(1, DockControl::Resume);
        }
        controls
    }
}

/// Apply a transition on behalf of `actor`.
///
/// The actor check runs before the table so an agent gets the specific
/// "you may not assert this" error rather than a generic illegal-transition
/// message that would read as a table gap.
pub fn transition(
    from: DockState,
    to: DockState,
    actor: Actor,
) -> Result<DockState, TransitionError> {
    if from.is_terminal() {
        return Err(TransitionError::Terminal { from });
    }
    if actor == Actor::Agent {
        if from.is_system_forced() {
            return Err(TransitionError::AgentCannotLeave { from });
        }
        if !to.is_agent_assertable() {
            return Err(TransitionError::AgentCannotEnter { to });
        }
    }
    if !from.may_precede(to) {
        return Err(TransitionError::Illegal { from, to });
    }
    Ok(to)
}

/// Pause, in one action, from any non-terminal state.
///
/// Idempotent: pausing an already paused agent succeeds rather than erroring,
/// because a control that sometimes fails is a control users stop trusting.
pub fn pause(from: DockState) -> Result<DockState, TransitionError> {
    if from.is_terminal() {
        return Err(TransitionError::Terminal { from });
    }
    Ok(DockState::Paused)
}

/// Terminate, in one action, from any non-terminal state. Never behind a
/// confirmation chain (ADR-295 §4 and the acceptance test's two-interaction
/// budget).
pub fn terminate(from: DockState) -> Result<DockState, TransitionError> {
    if from.is_terminal() {
        return Err(TransitionError::Terminal { from });
    }
    Ok(DockState::Completed)
}
