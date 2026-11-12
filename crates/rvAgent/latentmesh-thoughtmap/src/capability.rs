//! Capability vectors replacing named agent roles (ADR-326 Decision §1–§2).
//!
//! DeAR's confirmed mechanism (arXiv:2608.17282, part 1) is *query-dependent
//! agent specialization*. We operationalize it by replacing fixed labels like
//! `"coder"` / `"reviewer"` with a capability vector recomputed **per query**:
//!
//! ```text
//! C(i, j, q) = competence × trust × locality × freshness ÷ cost
//! ```
//!
//! (ruv's formula in RuVector#882 / ADR-326.) `competence`, `locality`, and
//! `freshness` are grounded against the specific query `q`, so an agent's
//! effective specialization is a function of its measured profile against that
//! query — not a static role.
//!
//! **Local peer selection is decentralized (ADR-326 §2).** [`LocalView::select_peer`]
//! takes only the *selecting* agent's own view plus the candidate profiles it
//! can observe. There is deliberately **no central planner / dispatcher / global
//! registry** argument — the contrast point with `radio-moe`'s centralized
//! reputation-weighted routing called out in ADR-326. Each agent chooses its
//! next collaboration peer by locally comparing `C(i,j,q)`.

use serde::{Deserialize, Serialize};

use crate::vecmath::cosine;

/// Stable identifier for an agent / peer participating in the reasoning fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PeerId(pub u64);

/// A query the reasoning fabric is collaborating on. Carries a small feature
/// embedding so competence/locality/freshness can be grounded query-dependently
/// (`C(i,j,q)` depends on `q`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub id: u64,
    /// Feature/embedding vector describing what the query needs.
    pub features: Vec<f32>,
}

impl Query {
    pub fn new(id: u64, features: Vec<f32>) -> Self {
        Self { id, features }
    }
}

/// An observable profile of a candidate peer. This is what an agent can *see*
/// about another agent locally — its advertised skill embedding, its
/// witness-logged trust score (ADR-312 tamper-evident inputs), its cost, and
/// the last step at which it was observed active (drives freshness).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeerProfile {
    pub id: PeerId,
    /// Capability embedding — dotted against the query to get query-dependent
    /// competence.
    pub skill: Vec<f32>,
    /// Witnessed reliability in `[0, 1]`. MUST be a tamper-evident, witness-logged
    /// input per ADR-326's capability-vector-integrity gate — a peer inflating
    /// its own trust to attract traffic is a modeled attack, not assumed-honest.
    pub trust: f32,
    /// Relative resource cost of collaborating with this peer (`> 0`).
    pub cost: f32,
    /// Step index at which this peer was last observed contributing. Older ⇒
    /// less fresh.
    pub last_active_step: u64,
}

/// The resolved five-factor capability vector for one `(i, j, q)` triple.
///
/// This is exactly ADR-326's formula operands; [`CapabilityVector::score`] is
/// the `C(i,j,q)` product.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapabilityVector {
    /// Query-dependent skill match in `[0, 1]`.
    pub competence: f32,
    /// Witnessed trust in `[0, 1]`.
    pub trust: f32,
    /// Selecting agent's local affinity to the peer in `[0, 1]`.
    pub locality: f32,
    /// Recency of the peer's last contribution in `[0, 1]`.
    pub freshness: f32,
    /// Resource cost (`> 0`); the divisor.
    pub cost: f32,
}

impl CapabilityVector {
    /// `C(i,j,q) = competence × trust × locality × freshness ÷ cost`.
    ///
    /// Cost is clamped to a small positive floor so a zero/negative advertised
    /// cost cannot manufacture an infinite score (a modeled gaming attack).
    pub fn score(&self) -> f32 {
        let cost = self.cost.max(1e-6);
        (self.competence * self.trust * self.locality * self.freshness) / cost
    }
}

/// The selecting agent's *local* view of the fabric — its identity, a locality
/// kernel over peers, and the current step clock for freshness decay.
///
/// Everything peer-selection needs is carried here or passed as observable
/// candidate profiles. There is no field pointing at a central router.
#[derive(Debug, Clone)]
pub struct LocalView {
    pub me: PeerId,
    /// Local affinity embedding — cosine against a peer's skill gives a
    /// decentralized `locality` term (agents near in capability space are
    /// "local" to each other). This is the *selecting* agent's own view.
    pub locality_kernel: Vec<f32>,
    /// Current logical step, used to decay `freshness`.
    pub now_step: u64,
    /// Freshness half-life in steps.
    pub freshness_half_life: u64,
}

impl LocalView {
    pub fn new(me: PeerId, locality_kernel: Vec<f32>, now_step: u64) -> Self {
        Self {
            me,
            locality_kernel,
            now_step,
            freshness_half_life: 8,
        }
    }

    /// Ground the five factors of `C(i,j,q)` for a single candidate peer against
    /// a specific query. This is where query-dependence enters: `competence` and
    /// `locality` are derived from the query and the selecting agent's own view.
    pub fn ground(&self, peer: &PeerProfile, query: &Query) -> CapabilityVector {
        // Query-dependent competence: how well the peer's skill matches what the
        // query needs. cosine ∈ [-1,1] → clamp to [0,1] (negative match = none).
        let competence = cosine(&peer.skill, &query.features).max(0.0);

        // Decentralized locality: closeness of the peer to the selecting agent's
        // own locality kernel. No global topology consulted.
        let locality = cosine(&self.locality_kernel, &peer.skill).max(0.0);

        // Freshness: exponential decay of staleness by half-life.
        let age = self.now_step.saturating_sub(peer.last_active_step) as f32;
        let hl = self.freshness_half_life.max(1) as f32;
        let freshness = 0.5f32.powf(age / hl);

        CapabilityVector {
            competence,
            trust: peer.trust.clamp(0.0, 1.0),
            locality,
            freshness,
            cost: peer.cost,
        }
    }

    /// **Local, decentralized peer selection.** Returns the candidate maximizing
    /// `C(i,j,q)`, computed purely from this agent's own view and the observable
    /// candidate profiles. No central planner participates.
    ///
    /// The selecting agent never picks itself. `excluded` lets callers skip peers
    /// already visited on the current path or peers that are quarantined.
    pub fn select_peer(
        &self,
        candidates: &[PeerProfile],
        query: &Query,
        excluded: &[PeerId],
    ) -> Option<PeerSelection> {
        candidates
            .iter()
            .filter(|p| p.id != self.me && !excluded.contains(&p.id))
            .map(|p| {
                let cap = self.ground(p, query);
                PeerSelection {
                    peer: p.id,
                    capability: cap,
                    score: cap.score(),
                }
            })
            // Highest score wins; total_cmp gives a deterministic order for the
            // NaN-free scores we produce.
            .max_by(|a, b| a.score.total_cmp(&b.score))
    }
}

/// Result of a local peer-selection round.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeerSelection {
    pub peer: PeerId,
    pub capability: CapabilityVector,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(id: u64, skill: Vec<f32>, trust: f32, cost: f32, last: u64) -> PeerProfile {
        PeerProfile {
            id: PeerId(id),
            skill,
            trust,
            cost,
            last_active_step: last,
        }
    }

    #[test]
    fn score_is_the_adr_formula() {
        let cap = CapabilityVector {
            competence: 0.8,
            trust: 0.5,
            locality: 1.0,
            freshness: 0.5,
            cost: 2.0,
        };
        // 0.8 * 0.5 * 1.0 * 0.5 / 2.0 = 0.1
        assert!((cap.score() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn competence_is_query_dependent() {
        // Same peer, two different queries → different competence, therefore a
        // different capability vector. This is the "no static role" property.
        let view = LocalView::new(PeerId(0), vec![1.0, 1.0, 1.0], 10);
        let peer = profile(1, vec![1.0, 0.0, 0.0], 1.0, 1.0, 10);

        let q_match = Query::new(1, vec![1.0, 0.0, 0.0]);
        let q_miss = Query::new(2, vec![0.0, 1.0, 0.0]);

        let c_match = view.ground(&peer, &q_match).competence;
        let c_miss = view.ground(&peer, &q_miss).competence;

        assert!(c_match > 0.9, "aligned query → high competence");
        assert!(c_miss < 0.1, "orthogonal query → ~zero competence");
    }

    #[test]
    fn select_peer_is_local_and_picks_best() {
        let view = LocalView::new(PeerId(0), vec![1.0, 0.0, 0.0], 10);
        // Query wants the first skill dimension.
        let query = Query::new(1, vec![1.0, 0.0, 0.0]);
        let candidates = vec![
            profile(1, vec![1.0, 0.0, 0.0], 0.9, 1.0, 10), // strong match
            profile(2, vec![0.0, 1.0, 0.0], 0.9, 1.0, 10), // wrong skill
            profile(3, vec![1.0, 0.0, 0.0], 0.9, 5.0, 10), // match but expensive
        ];
        let sel = view.select_peer(&candidates, &query, &[]).unwrap();
        assert_eq!(sel.peer, PeerId(1), "best competence/cost wins locally");
    }

    #[test]
    fn select_peer_never_returns_self_or_excluded() {
        let view = LocalView::new(PeerId(1), vec![1.0, 0.0, 0.0], 10);
        let query = Query::new(1, vec![1.0, 0.0, 0.0]);
        let candidates = vec![
            profile(1, vec![1.0, 0.0, 0.0], 1.0, 1.0, 10), // self
            profile(2, vec![1.0, 0.0, 0.0], 1.0, 1.0, 10), // excluded below
            profile(3, vec![0.9, 0.1, 0.0], 0.8, 1.0, 10),
        ];
        let sel = view.select_peer(&candidates, &query, &[PeerId(2)]).unwrap();
        assert_eq!(sel.peer, PeerId(3));
    }

    #[test]
    fn zero_cost_cannot_manufacture_infinite_score() {
        let cap = CapabilityVector {
            competence: 1.0,
            trust: 1.0,
            locality: 1.0,
            freshness: 1.0,
            cost: 0.0,
        };
        assert!(cap.score().is_finite());
    }
}
