//! The reasoning engine — wires capability grounding, the thought map, the
//! causal gate, quarantine, and dead-end topology adaptation into one decision
//! surface (ADR-326).
//!
//! Two responsibilities matter for the WP23 security properties:
//!
//! * [`ReasoningEngine::incorporate`] is the **single choke point** through which
//!   any peer message reaches the thought graph. It enforces, in order:
//!   integrity (ADR-311) → quarantine dampening (ADR-311) → causal attribution
//!   (ADR-310). A message that fails any check is **never** written to the graph.
//! * [`ReasoningEngine::handle_dead_end`] applies the swappable, explicitly
//!   *unverified* dead-end [`DeadEndPolicy`] (ADR-326 §6), and **refuses any
//!   topology change prompted by a quarantined peer** — a misbehaving peer
//!   cannot churn the topology.

use crate::capability::{LocalView, PeerId, PeerProfile, Query};
use crate::causal::{CausalVerdict, CausalVerifier};
use crate::quarantine::{Anomaly, Quarantine};
use crate::thoughtmap::{StepId, ThoughtGraphBackend, ThoughtMap};
use crate::topology::{DeadEndPolicy, TopologyCause, TopologyChange};
use crate::transport::{integrity_tag, LatentMessage};

/// Root step every reasoning process starts from; `Restart` resumes here.
pub const ROOT_STEP: StepId = StepId(0);

/// Weight a freshly incorporated peer-interaction edge starts at, before any
/// success/failure feedback reinforces or weakens it.
const INITIAL_EDGE_WEIGHT: f32 = 0.5;

/// Result of attempting to incorporate a message.
#[derive(Debug, Clone, PartialEq)]
pub enum Incorporation {
    /// Accepted: a new reasoning step was written and connected to its cause.
    Accepted { step: StepId },
    /// Dropped because the sender is quarantined; the map is untouched.
    Dampened,
    /// Rejected by a gate; the map is untouched. Carries the reason.
    Rejected(RejectionKind),
}

/// Why a message was rejected (distinct from being dampened for quarantine).
#[derive(Debug, Clone, PartialEq)]
pub enum RejectionKind {
    /// The latent payload contained a non-finite (NaN/±inf) element. Such a
    /// message is malformed/hostile and is rejected at the choke point *before*
    /// the causal gate — a NaN would otherwise make every `<`-comparison in the
    /// gate false and silently fall through to Accept, then poison the map-wide
    /// control population. This is the real fix for that class of attack.
    NonFinite,
    /// Integrity tag did not match (tampering) — ADR-311.
    Integrity,
    /// Failed the causal-attribution gate — ADR-310.
    Causal(crate::causal::RejectReason),
}

/// The engine. Generic over the [`CausalVerifier`] so WP6 can substitute a
/// production audit without touching the choke-point logic.
#[derive(Debug, Clone)]
pub struct ReasoningEngine<V: CausalVerifier> {
    pub map: ThoughtMap,
    pub quarantine: Quarantine,
    /// **Unverified design choice** (ADR-326 §6) — see [`crate::topology`].
    pub policy: DeadEndPolicy,
    verifier: V,
}

impl<V: CausalVerifier> ReasoningEngine<V> {
    pub fn new(
        map: ThoughtMap,
        verifier: V,
        quarantine: Quarantine,
        policy: DeadEndPolicy,
    ) -> Self {
        Self {
            map,
            quarantine,
            policy,
            verifier,
        }
    }

    /// **The choke point.** Incorporate a received latent message into the
    /// thought graph iff it passes every gate. Order is deliberate: cheap
    /// tamper-evidence first, then quarantine, then the causal audit.
    pub fn incorporate(&mut self, msg: &LatentMessage) -> Incorporation {
        // 0. Finiteness (the real NaN-bypass fix). A non-finite latent would make
        //    the causal gate's `<`-comparisons false and fall through to Accept,
        //    then poison the map-wide control. Reject BEFORE any gate, never write
        //    it, and strike the sender — a non-finite injection IS map degradation
        //    (this is where the previously-unrecorded MapDegradation anomaly is
        //    wired), so a repeat offender quarantines instead of retrying forever.
        if !crate::vecmath::is_finite(&msg.latent) {
            self.quarantine.record(msg.from, Anomaly::MapDegradation);
            return Incorporation::Rejected(RejectionKind::NonFinite);
        }

        // 1. Integrity (ADR-311). A tampered channel is the strongest anomaly.
        let expected = integrity_tag(msg.from, msg.to, &msg.latent, msg.causal.antecedent);
        if msg.integrity != expected {
            self.quarantine.record(msg.from, Anomaly::IntegrityFailure);
            return Incorporation::Rejected(RejectionKind::Integrity);
        }

        // 2. Quarantine dampening (ADR-311). A quarantined peer's messages are
        //    never incorporated — this is what stops a bad actor reshaping the map.
        if self.quarantine.is_quarantined(msg.from) {
            return Incorporation::Dampened;
        }

        // 3. Causal attribution (ADR-310). Must be attributable to its claimed
        //    cause under controlled replacement before it may touch the graph.
        match self.verifier.verify(&self.map, msg) {
            CausalVerdict::Accept => {
                let step = self
                    .map
                    .add_step(msg.from, msg.latent.clone(), msg.summary.clone());
                self.map
                    .connect(msg.causal.antecedent, step, msg.from, INITIAL_EDGE_WEIGHT);
                Incorporation::Accepted { step }
            }
            CausalVerdict::Reject(reason) => {
                // A message that fails causal attribution is an anomaly signal;
                // sustained failures quarantine the sender.
                self.quarantine.record(msg.from, Anomaly::CausalRejection);
                Incorporation::Rejected(RejectionKind::Causal(reason))
            }
        }
    }

    /// Reinforce a confirmed-good reasoning path (ADR-326 §5).
    pub fn reinforce(&mut self, path: &[StepId], delta: f32) {
        self.map.reinforce_path(path, delta);
    }

    /// Weaken a confirmed-bad reasoning path (ADR-326 §5).
    pub fn weaken(&mut self, path: &[StepId], delta: f32) {
        self.map.weaken_path(path, delta);
    }

    /// Apply the dead-end policy at `node`.
    ///
    /// `triggered_by` is the peer/agent whose situation prompted the adaptation;
    /// **if that peer is quarantined the change is refused (`None`)** so a
    /// misbehaving peer cannot force topology churn. Every returned
    /// [`TopologyChange`] is attributable (`cause` + `triggered_by`).
    ///
    /// For [`DeadEndPolicy::ContinueWithTopologyChange`] (the **unverified**
    /// default, ADR-326 §6) reasoning resumes from `node` itself, re-routed to a
    /// freshly selected non-quarantined peer — progress is preserved. For
    /// [`DeadEndPolicy::Restart`] reasoning resumes from [`ROOT_STEP`].
    pub fn handle_dead_end(
        &mut self,
        node: StepId,
        view: &LocalView,
        candidates: &[PeerProfile],
        query: &Query,
        visited: &[PeerId],
        triggered_by: PeerId,
    ) -> Option<TopologyChange> {
        // Refuse churn prompted by a quarantined peer (ADR-311 / acceptance).
        if self.quarantine.is_quarantined(triggered_by) {
            return None;
        }

        match self.policy {
            DeadEndPolicy::ContinueWithTopologyChange => {
                // Re-route locally, excluding already-visited and quarantined peers.
                let excluded = self.excluded_peers(visited, candidates);
                let new_peer = view
                    .select_peer(candidates, query, &excluded)
                    .map(|s| s.peer);
                Some(TopologyChange {
                    resume_from: node, // continue from current state, NOT root
                    new_peer,
                    cause: TopologyCause::DeadEnd,
                    triggered_by,
                    continued: true,
                })
            }
            DeadEndPolicy::Restart => Some(TopologyChange {
                resume_from: ROOT_STEP, // discard progress
                new_peer: None,
                cause: TopologyCause::DeadEnd,
                triggered_by,
                continued: false,
            }),
        }
    }

    /// Visited peers plus every currently quarantined candidate — the exclusion
    /// set for re-routing.
    fn excluded_peers(&self, visited: &[PeerId], candidates: &[PeerProfile]) -> Vec<PeerId> {
        let mut excluded = visited.to_vec();
        for c in candidates {
            if self.quarantine.is_quarantined(c.id) && !excluded.contains(&c.id) {
                excluded.push(c.id);
            }
        }
        excluded
    }
}
