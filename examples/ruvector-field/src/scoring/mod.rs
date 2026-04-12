//! Scoring primitives: resonance, coherence, retrieval, routing, drift signal.

pub mod coherence;
pub mod resonance;
pub mod retrieval;
pub mod routing;

use core::fmt;

use crate::model::{HintId, NodeId, Shell};

pub use coherence::{effective_resistance_proxy, local_coherence};
pub use resonance::resonance_score;
pub use retrieval::score_candidate;
pub use routing::score_route;

/// Drift signal across four channels. Spec section 12.
///
/// # Example
///
/// ```
/// use ruvector_field::scoring::DriftSignal;
/// let d = DriftSignal {
///     semantic: 0.3,
///     structural: 0.2,
///     policy: 0.0,
///     identity: 0.0,
///     total: 0.5,
/// };
/// assert!(d.agreement_fires(0.4, 0.1));
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriftSignal {
    /// Centroid shift vs reference.
    pub semantic: f32,
    /// Jaccard distance over edge set between snapshots.
    pub structural: f32,
    /// Mean movement in policy fit across nodes.
    pub policy: f32,
    /// Agent/role assignment distribution change.
    pub identity: f32,
    /// Sum of all four channels.
    pub total: f32,
}

impl DriftSignal {
    /// Four-channel agreement rule: total > `total_threshold` AND at least
    /// two individual channels above `per_channel_threshold`.
    pub fn agreement_fires(&self, total_threshold: f32, per_channel_threshold: f32) -> bool {
        let agreeing = [self.semantic, self.structural, self.policy, self.identity]
            .iter()
            .filter(|c| **c > per_channel_threshold)
            .count();
        self.total > total_threshold && agreeing >= 2
    }
}

impl fmt::Display for DriftSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "drift[sem={:.3} struct={:.3} pol={:.3} ident={:.3} total={:.3}]",
            self.semantic, self.structural, self.policy, self.identity, self.total
        )
    }
}

/// Retrieval result.
///
/// # Example
///
/// ```
/// use ruvector_field::scoring::RetrievalResult;
/// let r = RetrievalResult::default();
/// assert!(r.selected.is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct RetrievalResult {
    /// Selected nodes ordered by descending final score.
    pub selected: Vec<NodeId>,
    /// Rejected candidates also scored but not returned.
    pub rejected: Vec<NodeId>,
    /// Contradiction frontier discovered during the 2-hop walk.
    pub contradiction_frontier: Vec<NodeId>,
    /// `max - min` confidence over the contradiction frontier.
    pub confidence_spread: f32,
    /// Explanation trace lines.
    pub explanation: Vec<String>,
}

impl fmt::Display for RetrievalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "retrieval[selected={} frontier={} spread={:.3}]",
            self.selected.len(),
            self.contradiction_frontier.len(),
            self.confidence_spread
        )
    }
}

/// Routing hint — advisory until committed through a proof gate.
///
/// # Example
///
/// ```
/// use ruvector_field::scoring::RoutingHint;
/// use ruvector_field::model::HintId;
/// let h = RoutingHint::demo(HintId(1), false);
/// assert_eq!(h.id, HintId(1));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingHint {
    /// Stable id.
    pub id: HintId,
    /// Target partition, if any.
    pub target_partition: Option<u64>,
    /// Target agent, if any.
    pub target_agent: Option<u64>,
    /// Target shell depth match used for shell_fit.
    pub target_shell: Option<Shell>,
    /// Capability fit factor in `[0, 1]`.
    pub capability_fit: f32,
    /// Role fit factor in `[0, 1]`.
    pub role_fit: f32,
    /// Locality fit factor in `[0, 1]`.
    pub locality_fit: f32,
    /// Shell fit factor in `[0, 1]`.
    pub shell_fit: f32,
    /// Expected gain (resonance delta).
    pub gain_estimate: f32,
    /// Expected cost.
    pub cost_estimate: f32,
    /// TTL in scheduler epochs.
    pub ttl_epochs: u16,
    /// If true the hint must pass a proof gate before commit.
    pub requires_proof: bool,
    /// Committed flag — flipped by `commit`.
    pub committed: bool,
    /// Human-readable reason.
    pub reason: String,
}

impl RoutingHint {
    /// Construct a demo hint (used in tests and docs).
    pub fn demo(id: HintId, requires_proof: bool) -> Self {
        Self {
            id,
            target_partition: None,
            target_agent: Some(1),
            target_shell: Some(Shell::Concept),
            capability_fit: 0.8,
            role_fit: 0.9,
            locality_fit: 0.7,
            shell_fit: 0.9,
            gain_estimate: 0.6,
            cost_estimate: 0.2,
            ttl_epochs: 4,
            requires_proof,
            committed: false,
            reason: "demo".to_string(),
        }
    }

    /// Compute the raw route score. Spec 8.4.
    pub fn score(&self) -> f32 {
        let cost = self.cost_estimate.max(1e-3);
        self.capability_fit
            * self.role_fit
            * self.locality_fit
            * self.shell_fit
            * (self.gain_estimate / cost)
    }

    /// Commit the hint through a [`crate::proof::ProofGate`]. Marks it as
    /// committed on success. The caller is responsible for emitting the
    /// `RoutingHintCommitted` witness event.
    pub fn commit<G: crate::proof::ProofGate>(
        &mut self,
        gate: &mut G,
    ) -> Result<crate::proof::ProofToken, crate::proof::ProofError> {
        if !self.requires_proof {
            self.committed = true;
            return Ok(crate::proof::ProofToken {
                hint: self.id,
                sequence: 0,
            });
        }
        let token = gate.authorize(self)?;
        self.committed = true;
        Ok(token)
    }
}

impl fmt::Display for RoutingHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} agent={:?} gain={:.3} cost={:.3} score={:.3} ttl={} committed={} reason={}",
            self.id,
            self.target_agent,
            self.gain_estimate,
            self.cost_estimate,
            self.score(),
            self.ttl_epochs,
            self.committed,
            self.reason
        )
    }
}
