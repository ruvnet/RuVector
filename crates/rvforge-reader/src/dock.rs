//! Agent Dock domain model (ADR-295 §1, §2, §7, §8).
//!
//! The load-bearing rule of ADR-295 is §7: dock chrome is RVForge's, content is
//! the agent's, and the two must not be confusable. That rule is enforced here
//! **in the types**, not in the renderer:
//!
//! - [`AgentProvidedStatus`] carries the only two things an agent supplies —
//!   task text and progress — and it can only be built by
//!   [`AgentProvidedStatus::from_agent`], which sanitizes both.
//! - [`SystemOwnedStatus`] carries everything with a security meaning (state,
//!   trust badge, network indicator, permission summary, witness status,
//!   resource usage, cost). Its fields are private, it derives no
//!   `Deserialize`, and it has exactly one constructor —
//!   [`SystemOwnedStatus::measured`] — which takes no agent-derived value.
//! - [`AgentDockEntry`] composes the two. The only mutator that accepts agent
//!   input is [`AgentDockEntry::apply_agent_update`], and it replaces the agent
//!   half wholesale; there is no method, field, or trait impl through which
//!   agent bytes reach the system half.
//!
//! An agent that wants to appear paused therefore has to actually be paused,
//! because the state indicator is a projection of [`SystemOwnedStatus`].
//!
//! Cleaning and screening of agent strings lives in [`crate::dock_text`] and is
//! re-exported here, because the sanitizer is part of this boundary rather than
//! a separate concern.

use serde::Serialize;

use crate::dock_state::DockState;

pub use crate::dock_text::{
    sanitize_agent_text, sanitize_system_text, SanitizedText, SanitizerFinding, MAX_TASK_CHARS,
    WITHHELD_TEXT,
};

/// The agent's half of a dock entry: task text and progress, nothing else.
///
/// Deliberately not `Deserialize`. The only way in is
/// [`Self::from_agent`], so an agent cannot hand the dock a serialized status
/// object with extra fields and hope one of them lands somewhere meaningful.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgentProvidedStatus {
    task: SanitizedText,
    progress: u8,
    /// The agent reported a progress value outside 0–100 and it was clamped.
    progress_clamped: bool,
}

impl AgentProvidedStatus {
    /// Build from raw agent input. `raw_progress` is `i64` so an out-of-range
    /// report is visible and clamped rather than rejected by the wire format.
    pub fn from_agent(raw_task: &str, raw_progress: i64) -> Self {
        Self {
            task: sanitize_agent_text(raw_task),
            progress: raw_progress.clamp(0, 100) as u8,
            progress_clamped: !(0..=100).contains(&raw_progress),
        }
    }

    /// The resting entry for an agent that has not reported yet.
    pub fn none_reported() -> Self {
        Self {
            task: sanitize_agent_text("No task reported"),
            progress: 0,
            progress_clamped: false,
        }
    }

    pub fn task(&self) -> &SanitizedText {
        &self.task
    }
    pub fn progress(&self) -> u8 {
        self.progress
    }
    pub fn progress_was_clamped(&self) -> bool {
        self.progress_clamped
    }
    pub fn is_suspicious(&self) -> bool {
        self.task.is_suspicious()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrustBadge {
    /// Publisher signature checked against the registry.
    VerifiedPublisher,
    /// Signed, publisher not in the verified set.
    SignedUnknownPublisher,
    /// No signature has been checked. Never rendered as neutral.
    Unverified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkIndicator {
    /// The runtime is denying network access.
    Disabled,
    /// Allowed, and currently idle.
    AllowedIdle,
    /// Allowed, with live connections.
    Active,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WitnessStatus {
    Valid,
    Broken,
    /// No witness chain has been read. The scaffold's honest answer.
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum IsolationClass {
    Rvm,
    Wasm,
    /// The adapter could not engage confinement. Reported, never hidden.
    None,
}

/// Capability facts read from the policy, not from the agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PermissionSummary {
    pub isolation: IsolationClass,
    pub local_model: bool,
    pub network_allowed: bool,
    /// Filesystem access is limited to folders the user picked.
    pub selected_folder_access: bool,
    pub encrypted_memory: bool,
    /// One line per granted capability, from the signed manifest.
    pub grants: Vec<String>,
    pub pending_approvals: Vec<String>,
}

/// Measured by RVForge and RVM, never reported by the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ResourceUsage {
    pub cpu_millis_per_sec: u32,
    pub memory_bytes: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
}

impl ResourceUsage {
    pub const ZERO: Self = Self {
        cpu_millis_per_sec: 0,
        memory_bytes: 0,
        tokens_in: 0,
        tokens_out: 0,
    };
}

/// The system half of a dock entry. Private fields, no `Deserialize`, one
/// constructor that takes no agent-derived value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SystemOwnedStatus {
    state: DockState,
    trust: TrustBadge,
    network: NetworkIndicator,
    permissions: PermissionSummary,
    witness: WitnessStatus,
    resources: ResourceUsage,
    estimated_cost_micros: u64,
    model: Option<String>,
    /// Bumped by the runtime on every state change; the roster orders on it.
    state_seq: u64,
}

impl SystemOwnedStatus {
    /// The only constructor. Every argument comes from the capability policy,
    /// the witness chain, or measured runtime telemetry.
    #[allow(clippy::too_many_arguments)]
    pub fn measured(
        state: DockState,
        trust: TrustBadge,
        network: NetworkIndicator,
        permissions: PermissionSummary,
        witness: WitnessStatus,
        resources: ResourceUsage,
        estimated_cost_micros: u64,
        model: Option<String>,
    ) -> Self {
        Self {
            state,
            trust,
            network,
            permissions,
            witness,
            resources,
            estimated_cost_micros,
            model,
            state_seq: 0,
        }
    }

    pub fn state(&self) -> DockState {
        self.state
    }
    pub fn trust(&self) -> TrustBadge {
        self.trust
    }
    pub fn network(&self) -> NetworkIndicator {
        self.network
    }
    pub fn permissions(&self) -> &PermissionSummary {
        &self.permissions
    }
    pub fn witness(&self) -> WitnessStatus {
        self.witness
    }
    pub fn resources(&self) -> ResourceUsage {
        self.resources
    }
    pub fn estimated_cost_micros(&self) -> u64 {
        self.estimated_cost_micros
    }
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
    pub fn state_seq(&self) -> u64 {
        self.state_seq
    }

    /// Record a state the state machine already approved. Private to the crate
    /// so the transition check cannot be skipped from outside.
    pub(crate) fn set_state(&mut self, state: DockState, seq: u64) {
        self.state = state;
        self.state_seq = seq;
    }

    /// Replace measured telemetry. Takes no agent-derived value.
    pub fn observe(&mut self, resources: ResourceUsage, estimated_cost_micros: u64) {
        self.resources = resources;
        self.estimated_cost_micros = estimated_cost_micros;
    }
}

/// One line of the ADR-295 §8 key capability card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapabilityCardLine {
    pub label: String,
    /// `true` when the runtime is enforcing the property right now.
    pub satisfied: bool,
    pub detail: String,
}

/// The seven-line running-agent capability card. Chrome, not content: every
/// line is derived from [`SystemOwnedStatus`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KeyCapabilityCard {
    pub lines: Vec<CapabilityCardLine>,
}

impl KeyCapabilityCard {
    pub fn from_system(system: &SystemOwnedStatus) -> Self {
        let p = system.permissions();
        let line = |label: &str, satisfied: bool, detail: &str| CapabilityCardLine {
            label: label.to_string(),
            satisfied,
            detail: detail.to_string(),
        };
        Self {
            lines: vec![
                line(
                    "Verified publisher",
                    system.trust() == TrustBadge::VerifiedPublisher,
                    match system.trust() {
                        TrustBadge::VerifiedPublisher => "Signature checked against the registry",
                        TrustBadge::SignedUnknownPublisher => "Signed by an unrecognised publisher",
                        TrustBadge::Unverified => "No signature has been checked",
                    },
                ),
                line(
                    "RVM or WASM isolated",
                    p.isolation != IsolationClass::None,
                    match p.isolation {
                        IsolationClass::Rvm => "Running inside RVM",
                        IsolationClass::Wasm => "Running inside a WASM sandbox",
                        IsolationClass::None => "No confinement was engaged",
                    },
                ),
                line(
                    "Local model",
                    p.local_model,
                    if p.local_model {
                        "Inference runs on this computer"
                    } else {
                        "Inference leaves this computer"
                    },
                ),
                line(
                    "Network disabled",
                    !p.network_allowed,
                    match system.network() {
                        NetworkIndicator::Disabled => "The runtime is denying network access",
                        NetworkIndicator::AllowedIdle => {
                            "Network is permitted; no live connections"
                        }
                        NetworkIndicator::Active => "Network is permitted and in use",
                    },
                ),
                line(
                    "Selected folder access",
                    p.selected_folder_access,
                    if p.selected_folder_access {
                        "Only folders you selected are readable"
                    } else {
                        "No folder access is granted"
                    },
                ),
                line(
                    "Encrypted memory",
                    p.encrypted_memory,
                    if p.encrypted_memory {
                        "Persistent state is sealed on disk"
                    } else {
                        "No persistent state is kept"
                    },
                ),
                line(
                    "Witness chain valid",
                    system.witness() == WitnessStatus::Valid,
                    match system.witness() {
                        WitnessStatus::Valid => "Witness chain verified",
                        WitnessStatus::Broken => "Witness chain does not verify",
                        WitnessStatus::Unavailable => "No witness chain has been read",
                    },
                ),
            ],
        }
    }
}

/// One agent in the dock: system-owned identity and status, plus the agent's
/// own two fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgentDockEntry {
    /// System-owned: from the signed manifest, not from the running agent.
    agent_id: String,
    name: String,
    icon: String,
    system: SystemOwnedStatus,
    agent: AgentProvidedStatus,
}

impl AgentDockEntry {
    /// Identity comes from the package manifest; status from the runtime. The
    /// agent half starts empty and is only ever filled by
    /// [`Self::apply_agent_update`].
    pub fn new(
        agent_id: impl Into<String>,
        name: impl Into<String>,
        icon: impl Into<String>,
        system: SystemOwnedStatus,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            name: name.into(),
            icon: icon.into(),
            system,
            agent: AgentProvidedStatus::none_reported(),
        }
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn icon(&self) -> &str {
        &self.icon
    }
    pub fn system(&self) -> &SystemOwnedStatus {
        &self.system
    }
    pub fn agent(&self) -> &AgentProvidedStatus {
        &self.agent
    }
    pub fn state(&self) -> DockState {
        self.system.state()
    }

    /// The one mutator that accepts agent input. It replaces the agent half and
    /// touches nothing else — the system half is not passed to it, so no field
    /// of [`SystemOwnedStatus`] is reachable from this call.
    pub fn apply_agent_update(&mut self, raw_task: &str, raw_progress: i64) {
        self.agent = AgentProvidedStatus::from_agent(raw_task, raw_progress);
    }

    /// System-side telemetry update.
    pub fn observe(&mut self, resources: ResourceUsage, estimated_cost_micros: u64) {
        self.system.observe(resources, estimated_cost_micros);
    }

    pub(crate) fn set_state(&mut self, state: DockState, seq: u64) {
        self.system.set_state(state, seq);
    }

    pub fn capability_card(&self) -> KeyCapabilityCard {
        KeyCapabilityCard::from_system(&self.system)
    }
}
