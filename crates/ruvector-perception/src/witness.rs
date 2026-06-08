//! Proof-gated perception. A physical change may only drive an action if it
//! passes a proof gate — an auditable evidence chain (raw hash, feature hash,
//! novelty, coherence, contradiction, boundary, policy), not a confidence score.

use crate::modality::Modality;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bounded authority the engine may exercise on a witnessed change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    /// Nothing changed worth noting.
    Ignore,
    /// Real but ambiguous/contradicted — keep watching, do not escalate.
    Observe,
    /// Coherent, novel, uncontradicted — raise an alert.
    Alert,
    /// Strong, clean, uncontradicted — allowed to mutate persistent memory.
    Mutate,
}

/// Thresholds that turn scores into bounded authority.
#[derive(Debug, Clone, Copy)]
pub struct ProofGate {
    /// Below this novelty, ignore (it's business as usual).
    pub novelty_min: f32,
    /// Novelty at/above this is "high".
    pub novelty_high: f32,
    /// Minimum boundary coherence to trust the localisation.
    pub coherence_min: f32,
    /// At/above this contradiction, never escalate beyond Observe.
    pub contradiction_max: f32,
}

impl Default for ProofGate {
    fn default() -> Self {
        Self {
            novelty_min: 0.25,
            novelty_high: 0.6,
            coherence_min: 0.5,
            contradiction_max: 0.34,
        }
    }
}

impl ProofGate {
    /// Decide bounded authority from the three scores. Contradiction caps
    /// authority at Observe; only clean, novel, uncontradicted evidence escalates.
    pub fn decide(&self, novelty: f32, coherence: f32, contradiction: f32) -> Action {
        if novelty < self.novelty_min {
            return Action::Ignore;
        }
        if contradiction >= self.contradiction_max {
            return Action::Observe; // evidence is internally inconsistent
        }
        if coherence < self.coherence_min {
            return Action::Observe; // can't trust the localisation
        }
        if novelty >= self.novelty_high {
            // Strong, clean, uncontradicted: highest authority only when
            // contradiction is essentially absent.
            if contradiction <= self.contradiction_max * 0.25 {
                Action::Mutate
            } else {
                Action::Alert
            }
        } else {
            Action::Observe
        }
    }
}

/// Qualitative novelty bucket for human-readable witnesses.
pub fn novelty_level(n: f32, gate: &ProofGate) -> &'static str {
    if n >= gate.novelty_high {
        "high"
    } else if n >= gate.novelty_min {
        "medium"
    } else {
        "low"
    }
}

/// The structured output of perception — a delta, not a label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltaWitness {
    /// Time window index.
    pub t: u64,
    /// The zone whose physical state moved.
    pub changed_boundary: String,
    /// Modalities that responded coherently.
    pub supporting_modalities: Vec<Modality>,
    /// Modalities that should have responded (historically responsive) but
    /// stayed silent — first-class disagreement.
    pub contradicting_modalities: Vec<Modality>,
    /// Novelty vs prior physical states, `[0, 1]`.
    pub novelty: f32,
    /// Boundary coherence (localisation cleanliness), `[0, 1]`.
    pub coherence: f32,
    /// Contradiction strength, `[0, 1]`.
    pub contradiction: f32,
    /// Bounded authority granted by the proof gate.
    pub action: Action,
    /// SHA-256 evidence hash for this witness (hex).
    pub evidence_hash: String,
    /// Previous witness hash — forms an auditable chain of custody.
    pub prev_hash: Option<String>,
}

/// Compute the evidence hash binding raw signal, features, scores, boundary,
/// policy, and the prior witness into one auditable digest.
#[allow(clippy::too_many_arguments)]
pub fn evidence_hash(
    raw: &[u8],
    features: &[u8],
    boundary: &str,
    novelty: f32,
    coherence: f32,
    contradiction: f32,
    action: Action,
    prev: Option<&str>,
) -> String {
    let mut h = Sha256::new();
    h.update(b"rvperception-v1");
    h.update((raw.len() as u64).to_le_bytes());
    h.update(raw);
    h.update((features.len() as u64).to_le_bytes());
    h.update(features);
    h.update(boundary.as_bytes());
    h.update(novelty.to_le_bytes());
    h.update(coherence.to_le_bytes());
    h.update(contradiction.to_le_bytes());
    h.update([action as u8]);
    if let Some(p) = prev {
        h.update(p.as_bytes());
    }
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contradiction_caps_authority_at_observe() {
        let g = ProofGate::default();
        // High novelty, clean boundary, but contradicted -> Observe, never Alert.
        assert_eq!(g.decide(0.9, 0.9, 0.5), Action::Observe);
        // Clean, novel, uncontradicted -> escalates.
        assert_eq!(g.decide(0.9, 0.9, 0.0), Action::Mutate);
        // Below novelty floor -> Ignore.
        assert_eq!(g.decide(0.1, 0.9, 0.0), Action::Ignore);
    }

    #[test]
    fn evidence_hash_is_deterministic_and_chains() {
        let a = evidence_hash(b"raw", b"feat", "zoneA", 0.9, 0.8, 0.1, Action::Alert, None);
        let b = evidence_hash(b"raw", b"feat", "zoneA", 0.9, 0.8, 0.1, Action::Alert, None);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        // Chaining changes the hash.
        let c = evidence_hash(
            b"raw",
            b"feat",
            "zoneA",
            0.9,
            0.8,
            0.1,
            Action::Alert,
            Some(&a),
        );
        assert_ne!(a, c);
    }
}
