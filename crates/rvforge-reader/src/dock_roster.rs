//! Multi-agent dock policy (ADR-295 §5) and the two-interaction control path.
//!
//! Fifteen agents rendered as fifteen pills is an unusable bar, so the roster
//! collapses to: one active agent, two secondary icons, an overflow count,
//! aggregate resource usage, and a pending-approval count. The two aggregates
//! are computed over **every** agent, including the ones the dock is hiding —
//! they are the figures that reveal a fleet doing more work or asking for more
//! consent than the user expects, so they must not be a sample of three.
//!
//! Selection of the active agent is deterministic: attention rank first
//! (waiting for approval, then capability denied, then error, then running,
//! then the rest), most recent state change next, agent id last. A dock whose
//! active slot depends on hash order is a dock whose users cannot learn it.
//!
//! The acceptance test of ADR-295 — identify, read, inspect permissions,
//! terminate, in two interactions — is expressed here as
//! [`DockInteraction`]: the collapsed pill is free (it is already on screen),
//! [`DockInteraction::Expand`] is one, [`DockInteraction::Terminate`] is two.

use serde::Serialize;

use crate::dock::{
    AgentDockEntry, KeyCapabilityCard, NetworkIndicator, PermissionSummary, ResourceUsage,
    TrustBadge, WitnessStatus,
};
use crate::dock_events::{DockEvent, EventLog, Notification, RecordedEvent};
use crate::dock_state::{self, Actor, DockControl, DockState, TransitionError};

/// How many agents get a slot next to the active one.
pub const SECONDARY_SLOTS: usize = 2;

/// The label on every region holding agent-authored text. The distinction the
/// agent cannot imitate is that this label is drawn by the dock, outside the
/// region it wraps (ADR-295 §7).
pub const AGENT_CONTENT_LABEL: &str = "Agent-reported";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum RosterError {
    UnknownAgent { agent_id: String },
    Transition { detail: String },
}

impl std::fmt::Display for RosterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RosterError::UnknownAgent { agent_id } => {
                write!(f, "no agent '{agent_id}' in the dock")
            }
            RosterError::Transition { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for RosterError {}

impl From<TransitionError> for RosterError {
    fn from(e: TransitionError) -> Self {
        RosterError::Transition {
            detail: e.to_string(),
        }
    }
}

/// Agent-authored text, always paired with its label and its suspicion flag so
/// a renderer cannot accidentally emit it bare.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgentContent {
    pub label: &'static str,
    pub text: String,
    pub suspicious: bool,
    /// Present when the text was withheld: what the agent actually said.
    pub withheld: Option<String>,
}

/// The collapsed pill: the eight ADR-295 §1 elements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgentPill {
    pub agent_id: String,
    /// System-owned, from the manifest.
    pub name: String,
    pub icon: String,
    /// Agent-owned.
    pub task: AgentContent,
    pub progress: u8,
    pub state: DockState,
    pub state_label: &'static str,
    pub state_color: &'static str,
    pub state_glyph: &'static str,
    /// Always contains Pause and Terminate outside terminal states.
    pub controls: Vec<DockControl>,
    pub trust: TrustBadge,
    pub network: NetworkIndicator,
    pub witness: WitnessStatus,
}

/// Aggregates over every agent, hidden ones included.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AggregateUsage {
    pub agents: usize,
    pub cpu_millis_per_sec: u32,
    pub memory_bytes: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub estimated_cost_micros: u64,
}

/// What the bar shows however many agents exist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RosterView {
    pub active: Option<AgentPill>,
    /// At most [`SECONDARY_SLOTS`].
    pub secondary: Vec<AgentPill>,
    /// Agents with no slot. Selecting the count opens the swarm view.
    pub overflow_count: usize,
    pub aggregate: AggregateUsage,
    pub pending_approvals: usize,
}

/// The expanded panel: the ten ADR-295 §2 elements plus the §8 capability card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpandedAgentView {
    pub pill: AgentPill,
    /// 1 — agent-authored.
    pub objective: AgentContent,
    /// 2 — mixed; each entry carries its own `agent_authored` flag.
    pub recent_actions: Vec<RecordedEvent>,
    /// 3 — system-owned, from the policy engine.
    pub pending_approvals: Vec<String>,
    /// 4 — measured.
    pub model: Option<String>,
    pub tokens_in: u64,
    pub tokens_out: u64,
    /// 5 — measured.
    pub cpu_millis_per_sec: u32,
    pub memory_bytes: u64,
    /// 6 — measured.
    pub network: NetworkIndicator,
    /// 7 — from the signed manifest.
    pub capability_grants: Vec<String>,
    /// 8 — from the witness chain.
    pub witness: WitnessStatus,
    /// 9 — measured.
    pub estimated_cost_micros: u64,
    /// 10 — the one place the user addresses the agent.
    pub instruction_field_enabled: bool,
    /// ADR-295 §8.
    pub capability_card: KeyCapabilityCard,
}

/// One user action against the dock. The acceptance test's budget is counted
/// in these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DockInteraction {
    Expand { agent_id: String },
    Pause { agent_id: String },
    Terminate { agent_id: String },
    OpenSwarmView,
}

/// What an interaction produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum InteractionOutcome {
    Expanded(Box<ExpandedAgentView>),
    StateChanged { agent_id: String, state: DockState },
    SwarmView { pills: Vec<AgentPill> },
}

/// Attention rank; lower sorts first (ADR-295 §5).
fn attention_rank(state: DockState) -> u8 {
    match state {
        DockState::WaitingForApproval => 0,
        DockState::CapabilityDenied => 1,
        DockState::Error => 2,
        DockState::Running => 3,
        _ => 4,
    }
}

/// Every agent the dock knows about.
#[derive(Debug, Default)]
pub struct DockRoster {
    entries: Vec<AgentDockEntry>,
    log: EventLog,
    seq: u64,
}

impl DockRoster {
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit an agent. Its arrival counts as a state change, so a newly
    /// arrived agent outranks an older one at the same attention rank.
    pub fn insert(&mut self, mut entry: AgentDockEntry) {
        self.seq += 1;
        let state = entry.state();
        entry.set_state(state, self.seq);
        match self
            .entries
            .iter_mut()
            .find(|e| e.agent_id() == entry.agent_id())
        {
            Some(slot) => *slot = entry,
            None => self.entries.push(entry),
        }
    }

    pub fn remove(&mut self, agent_id: &str) {
        self.entries.retain(|e| e.agent_id() != agent_id);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, agent_id: &str) -> Option<&AgentDockEntry> {
        self.entries.iter().find(|e| e.agent_id() == agent_id)
    }

    pub fn event_log(&self) -> &EventLog {
        &self.log
    }

    /// Record an event through the D8 thresholds.
    pub fn record(&mut self, event: DockEvent) -> Option<Notification> {
        self.log.record(event)
    }

    /// The agent-supplied channel. Task text and progress, nothing else.
    pub fn apply_agent_update(
        &mut self,
        agent_id: &str,
        raw_task: &str,
        raw_progress: i64,
    ) -> Result<(), RosterError> {
        let entry = self
            .entries
            .iter_mut()
            .find(|e| e.agent_id() == agent_id)
            .ok_or_else(|| RosterError::UnknownAgent {
                agent_id: agent_id.to_string(),
            })?;
        entry.apply_agent_update(raw_task, raw_progress);
        let progress = entry.agent().progress();
        self.log.record(DockEvent::TaskTextUpdated {
            agent_id: agent_id.to_string(),
            text: raw_task.to_string(),
        });
        self.log.record(DockEvent::ProgressUpdated {
            agent_id: agent_id.to_string(),
            progress,
        });
        Ok(())
    }

    /// Change state on behalf of `actor`, through the state machine.
    pub fn set_state(
        &mut self,
        agent_id: &str,
        to: DockState,
        actor: Actor,
    ) -> Result<DockState, RosterError> {
        let from = self
            .get(agent_id)
            .ok_or_else(|| RosterError::UnknownAgent {
                agent_id: agent_id.to_string(),
            })?
            .state();
        let next = dock_state::transition(from, to, actor)?;
        self.commit_state(agent_id, from, next);
        Ok(next)
    }

    /// Pause in one action, from any non-terminal state.
    pub fn pause(&mut self, agent_id: &str) -> Result<DockState, RosterError> {
        let from = self
            .get(agent_id)
            .ok_or_else(|| RosterError::UnknownAgent {
                agent_id: agent_id.to_string(),
            })?
            .state();
        let next = dock_state::pause(from)?;
        self.commit_state(agent_id, from, next);
        Ok(next)
    }

    /// Terminate in one action, from any non-terminal state.
    pub fn terminate(&mut self, agent_id: &str) -> Result<DockState, RosterError> {
        let from = self
            .get(agent_id)
            .ok_or_else(|| RosterError::UnknownAgent {
                agent_id: agent_id.to_string(),
            })?
            .state();
        let next = dock_state::terminate(from)?;
        self.commit_state(agent_id, from, next);
        Ok(next)
    }

    fn commit_state(&mut self, agent_id: &str, from: DockState, to: DockState) {
        self.seq += 1;
        let seq = self.seq;
        if let Some(entry) = self.entries.iter_mut().find(|e| e.agent_id() == agent_id) {
            entry.set_state(to, seq);
        }
        self.log.record(DockEvent::StateChanged {
            agent_id: agent_id.to_string(),
            from,
            to,
        });
    }

    /// System-side telemetry.
    pub fn observe(
        &mut self,
        agent_id: &str,
        resources: ResourceUsage,
        estimated_cost_micros: u64,
    ) -> Result<(), RosterError> {
        let entry = self
            .entries
            .iter_mut()
            .find(|e| e.agent_id() == agent_id)
            .ok_or_else(|| RosterError::UnknownAgent {
                agent_id: agent_id.to_string(),
            })?;
        entry.observe(resources, estimated_cost_micros);
        Ok(())
    }

    /// Agents in dock order: attention rank, then most recent state change,
    /// then agent id.
    pub fn ordered(&self) -> Vec<&AgentDockEntry> {
        let mut ordered: Vec<&AgentDockEntry> = self.entries.iter().collect();
        ordered.sort_by(|a, b| {
            attention_rank(a.state())
                .cmp(&attention_rank(b.state()))
                .then(b.system().state_seq().cmp(&a.system().state_seq()))
                .then(a.agent_id().cmp(b.agent_id()))
        });
        ordered
    }

    /// The active agent under the §5 selection rule.
    pub fn active(&self) -> Option<&AgentDockEntry> {
        self.ordered().into_iter().next()
    }

    pub fn pending_approval_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.state() == DockState::WaitingForApproval)
            .count()
    }

    pub fn aggregate(&self) -> AggregateUsage {
        let mut agg = AggregateUsage {
            agents: self.entries.len(),
            cpu_millis_per_sec: 0,
            memory_bytes: 0,
            tokens_in: 0,
            tokens_out: 0,
            estimated_cost_micros: 0,
        };
        for entry in &self.entries {
            let r = entry.system().resources();
            agg.cpu_millis_per_sec = agg.cpu_millis_per_sec.saturating_add(r.cpu_millis_per_sec);
            agg.memory_bytes = agg.memory_bytes.saturating_add(r.memory_bytes);
            agg.tokens_in = agg.tokens_in.saturating_add(r.tokens_in);
            agg.tokens_out = agg.tokens_out.saturating_add(r.tokens_out);
            agg.estimated_cost_micros = agg
                .estimated_cost_micros
                .saturating_add(entry.system().estimated_cost_micros());
        }
        agg
    }

    /// The bar. Free to read — it is already on screen, so it costs no
    /// interaction in the acceptance test.
    pub fn view(&self) -> RosterView {
        let ordered = self.ordered();
        let mut pills = ordered.iter().map(|e| pill_of(e));
        let active = pills.next();
        let secondary: Vec<AgentPill> = pills.by_ref().take(SECONDARY_SLOTS).collect();
        let shown = usize::from(active.is_some()) + secondary.len();
        RosterView {
            active,
            secondary,
            overflow_count: self.entries.len().saturating_sub(shown),
            aggregate: self.aggregate(),
            pending_approvals: self.pending_approval_count(),
        }
    }

    /// Interaction one: everything the acceptance test needs to read, in a
    /// single call — task, permissions, capability card, cost, witness.
    pub fn expand(&self, agent_id: &str) -> Result<ExpandedAgentView, RosterError> {
        let entry = self
            .get(agent_id)
            .ok_or_else(|| RosterError::UnknownAgent {
                agent_id: agent_id.to_string(),
            })?;
        let system = entry.system();
        let permissions: &PermissionSummary = system.permissions();
        let resources = system.resources();
        Ok(ExpandedAgentView {
            pill: pill_of(entry),
            objective: agent_content(entry),
            recent_actions: self.log.recent_for(agent_id).into_iter().cloned().collect(),
            pending_approvals: permissions.pending_approvals.clone(),
            model: system.model().map(str::to_string),
            tokens_in: resources.tokens_in,
            tokens_out: resources.tokens_out,
            cpu_millis_per_sec: resources.cpu_millis_per_sec,
            memory_bytes: resources.memory_bytes,
            network: system.network(),
            capability_grants: permissions.grants.clone(),
            witness: system.witness(),
            estimated_cost_micros: system.estimated_cost_micros(),
            instruction_field_enabled: !entry.state().is_terminal(),
            capability_card: entry.capability_card(),
        })
    }

    /// Every agent, for the swarm view behind the overflow count.
    pub fn swarm_view(&self) -> Vec<AgentPill> {
        self.ordered().iter().map(|e| pill_of(e)).collect()
    }

    /// Apply one user interaction. Terminate is reachable from the collapsed
    /// pill, so it never needs an expand first.
    pub fn perform(
        &mut self,
        interaction: DockInteraction,
    ) -> Result<InteractionOutcome, RosterError> {
        match interaction {
            DockInteraction::Expand { agent_id } => Ok(InteractionOutcome::Expanded(Box::new(
                self.expand(&agent_id)?,
            ))),
            DockInteraction::Pause { agent_id } => {
                let state = self.pause(&agent_id)?;
                Ok(InteractionOutcome::StateChanged { agent_id, state })
            }
            DockInteraction::Terminate { agent_id } => {
                let state = self.terminate(&agent_id)?;
                Ok(InteractionOutcome::StateChanged { agent_id, state })
            }
            DockInteraction::OpenSwarmView => Ok(InteractionOutcome::SwarmView {
                pills: self.swarm_view(),
            }),
        }
    }
}

fn agent_content(entry: &AgentDockEntry) -> AgentContent {
    let task = entry.agent().task();
    AgentContent {
        label: AGENT_CONTENT_LABEL,
        text: task.display().to_string(),
        suspicious: task.is_suspicious(),
        withheld: task.withheld().map(str::to_string),
    }
}

fn pill_of(entry: &AgentDockEntry) -> AgentPill {
    let state = entry.state();
    let system = entry.system();
    AgentPill {
        agent_id: entry.agent_id().to_string(),
        name: entry.name().to_string(),
        icon: entry.icon().to_string(),
        task: agent_content(entry),
        progress: entry.agent().progress(),
        state,
        state_label: state.label(),
        state_color: state.color_token(),
        state_glyph: state.glyph(),
        controls: state.controls(),
        trust: system.trust(),
        network: system.network(),
        witness: system.witness(),
    }
}
