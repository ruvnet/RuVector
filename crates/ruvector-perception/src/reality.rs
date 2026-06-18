//! Reality graph — grounding layer for agents.
//!
//! Agents hallucinate because they reason from prompts, not physical state. This
//! module lets an agent **query reality**: every answer is backed by witnessed
//! evidence (the [`DeltaWitness`] evidence hashes that justify it), not by text
//! inference. The agent asks "is anyone in the room? what changed since last
//! hour? which sensor is lying? is this action allowed?" and the reality graph
//! answers from physical memory.

use crate::witness::{Action, DeltaWitness};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A grounding question an agent can ask the physical world.
#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// Is something currently happening / present in a zone?
    Presence { zone: String },
    /// Which zones changed (acted-upon) at or after time `t`?
    ChangedSince { t: u64 },
    /// Which zones carry contradicted / untrusted evidence right now?
    WhichUntrusted,
    /// Is escalation (Alert/Mutate) currently permitted in a zone?
    ActionAllowed { zone: String },
    /// The most recent witness for a zone.
    LastWitness { zone: String },
}

/// A witness-grounded answer. `evidence` lists the SHA-256 evidence hashes that
/// justify the answer — provenance, not prose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundedAnswer {
    /// Boolean verdict (for yes/no queries; `false` when not applicable).
    pub yes: bool,
    /// Human-readable, fully grounded explanation.
    pub detail: String,
    /// Zones relevant to the answer (sorted).
    pub zones: Vec<String>,
    /// Supporting evidence hashes (the witnesses backing this answer).
    pub evidence: Vec<String>,
    /// Aggregate coherence of the supporting evidence, `[0, 1]`.
    pub coherence: f32,
}

impl GroundedAnswer {
    fn none(detail: impl Into<String>) -> Self {
        Self {
            yes: false,
            detail: detail.into(),
            zones: Vec::new(),
            evidence: Vec::new(),
            coherence: 0.0,
        }
    }
}

/// Physical-memory graph queried by agents. Holds the latest witness per zone.
#[derive(Debug, Clone, Default)]
pub struct RealityGraph {
    latest: BTreeMap<String, DeltaWitness>,
}

impl RealityGraph {
    /// Create an empty reality graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold a witness into physical memory (keyed by its changed boundary zone).
    pub fn ingest(&mut self, w: &DeltaWitness) {
        if w.changed_boundary.is_empty() {
            return;
        }
        self.latest.insert(w.changed_boundary.clone(), w.clone());
    }

    /// Zones known to the reality graph (sorted).
    pub fn zones(&self) -> Vec<String> {
        self.latest.keys().cloned().collect()
    }

    /// Answer a grounding query from physical memory.
    pub fn query(&self, q: &Query) -> GroundedAnswer {
        match q {
            Query::Presence { zone } => match self.latest.get(zone) {
                Some(w) if w.action != Action::Ignore => GroundedAnswer {
                    yes: true,
                    detail: format!(
                        "activity in {zone}: {} supporting modality(ies), novelty {:.2}, action {:?}",
                        w.supporting_modalities.len(),
                        w.novelty,
                        w.action
                    ),
                    zones: vec![zone.clone()],
                    evidence: vec![w.evidence_hash.clone()],
                    coherence: w.coherence,
                },
                Some(w) => GroundedAnswer {
                    yes: false,
                    detail: format!("{zone} quiet (last action Ignore)"),
                    zones: vec![zone.clone()],
                    evidence: vec![w.evidence_hash.clone()],
                    coherence: w.coherence,
                },
                None => GroundedAnswer::none(format!("no physical memory for {zone}")),
            },
            Query::ChangedSince { t } => {
                let mut zones = Vec::new();
                let mut evidence = Vec::new();
                let mut coh = 0.0f32;
                for (z, w) in &self.latest {
                    if w.t >= *t && w.action != Action::Ignore {
                        zones.push(z.clone());
                        evidence.push(w.evidence_hash.clone());
                        coh = coh.max(w.coherence);
                    }
                }
                GroundedAnswer {
                    yes: !zones.is_empty(),
                    detail: format!("{} zone(s) changed since t={t}", zones.len()),
                    zones,
                    evidence,
                    coherence: coh,
                }
            }
            Query::WhichUntrusted => {
                let mut zones = Vec::new();
                let mut evidence = Vec::new();
                let mut coh = 0.0f32;
                for (z, w) in &self.latest {
                    if w.contradiction > 0.0 {
                        zones.push(z.clone());
                        evidence.push(w.evidence_hash.clone());
                        coh = coh.max(w.contradiction);
                    }
                }
                GroundedAnswer {
                    yes: !zones.is_empty(),
                    detail: format!(
                        "{} zone(s) carry contradicted evidence (a modality that usually reacts stayed silent)",
                        zones.len()
                    ),
                    zones,
                    evidence,
                    coherence: coh,
                }
            }
            Query::ActionAllowed { zone } => match self.latest.get(zone) {
                Some(w) => {
                    let allowed = matches!(w.action, Action::Alert | Action::Mutate);
                    GroundedAnswer {
                        yes: allowed,
                        detail: if allowed {
                            format!("escalation permitted in {zone}: evidence is novel, coherent, uncontradicted ({:?})", w.action)
                        } else {
                            format!("escalation NOT permitted in {zone}: action capped at {:?} (contradiction {:.2})", w.action, w.contradiction)
                        },
                        zones: vec![zone.clone()],
                        evidence: vec![w.evidence_hash.clone()],
                        coherence: w.coherence,
                    }
                }
                None => GroundedAnswer::none(format!("no physical memory for {zone}; action denied by default")),
            },
            Query::LastWitness { zone } => match self.latest.get(zone) {
                Some(w) => GroundedAnswer {
                    yes: true,
                    detail: format!("last witness for {zone} at t={}, action {:?}", w.t, w.action),
                    zones: vec![zone.clone()],
                    evidence: vec![w.evidence_hash.clone()],
                    coherence: w.coherence,
                },
                None => GroundedAnswer::none(format!("no physical memory for {zone}")),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modality::Modality;

    fn witness(zone: &str, t: u64, action: Action, contradiction: f32, hash: &str) -> DeltaWitness {
        DeltaWitness {
            t,
            changed_boundary: zone.to_string(),
            supporting_modalities: vec![Modality::Rf, Modality::Vibration],
            contradicting_modalities: if contradiction > 0.0 {
                vec![Modality::Thermal]
            } else {
                vec![]
            },
            novelty: 0.8,
            coherence: 0.9,
            contradiction,
            action,
            evidence_hash: hash.to_string(),
            prev_hash: None,
        }
    }

    #[test]
    fn presence_is_grounded_in_a_witness() {
        let mut rg = RealityGraph::new();
        rg.ingest(&witness("kitchen", 5, Action::Alert, 0.0, "h1"));
        let a = rg.query(&Query::Presence {
            zone: "kitchen".into(),
        });
        assert!(a.yes);
        assert_eq!(a.evidence, vec!["h1".to_string()]);
        // A zone with no memory is honestly unknown, not hallucinated.
        let b = rg.query(&Query::Presence {
            zone: "garage".into(),
        });
        assert!(!b.yes);
        assert!(b.evidence.is_empty());
    }

    #[test]
    fn untrusted_and_action_gate() {
        let mut rg = RealityGraph::new();
        rg.ingest(&witness("door", 1, Action::Observe, 0.8, "hc")); // contradicted -> Observe
        rg.ingest(&witness("hall", 2, Action::Mutate, 0.0, "hm")); // clean -> Mutate
        let untrusted = rg.query(&Query::WhichUntrusted);
        assert_eq!(untrusted.zones, vec!["door".to_string()]);
        // Contradicted zone: escalation denied. Clean zone: allowed.
        assert!(
            !rg.query(&Query::ActionAllowed {
                zone: "door".into()
            })
            .yes
        );
        assert!(
            rg.query(&Query::ActionAllowed {
                zone: "hall".into()
            })
            .yes
        );
    }

    #[test]
    fn changed_since_filters_by_time() {
        let mut rg = RealityGraph::new();
        rg.ingest(&witness("a", 1, Action::Alert, 0.0, "ha"));
        rg.ingest(&witness("b", 9, Action::Alert, 0.0, "hb"));
        let a = rg.query(&Query::ChangedSince { t: 5 });
        assert_eq!(a.zones, vec!["b".to_string()]);
    }
}
