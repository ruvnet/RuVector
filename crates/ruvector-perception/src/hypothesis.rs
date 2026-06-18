//! Multi-modal disagreement engine — *disagreement is information, not noise*.
//!
//! When modalities contradict each other, the classical reflex is to fuse them
//! into a single agreed-upon answer (and throw away the conflict). This module
//! does the opposite: it treats a contradiction as a *question* — "why do these
//! sensors disagree?" — and answers with **ranked hypotheses** instead of a
//! forced consensus.
//!
//! The same raw disagreement can mean very different things:
//!
//! - a **real event** the slow channels haven't caught up to yet,
//! - a single channel **drifting** out of calibration,
//! - a sensor that was physically **relocated** so its readings no longer fit
//!   the spatial field,
//! - an **adversarial replay** where the easy-to-spoof channels were faked while
//!   the hard-to-spoof physical channels stayed silent,
//! - or a transient **environmental artifact** (an echo / reflection).
//!
//! Each candidate gets an `evidence` score in `[0, 1]` derived from *typed*
//! physics ([`Modality::physics`]) plus the qualitative shape of the
//! disagreement. We always return all five, sorted by evidence descending, so a
//! caller can inspect the full ranked field rather than a single label.

use crate::modality::Modality;
use serde::{Deserialize, Serialize};

/// A candidate explanation for *why* the modalities disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hypothesis {
    /// A genuine physical change; supporting (hard-to-spoof) channels fired
    /// coherently and the few contradictions are explainable (e.g. latency).
    RealEvent,
    /// A single low-spoof-resistance channel slowly wandering out of calibration
    /// — persistent, lone, and spatially incoherent.
    SensorDrift,
    /// A sensor was physically moved: it responds strongly but its readings no
    /// longer fit the neighbours' spatial pattern (sudden + novel + incoherent).
    SensorRelocation,
    /// Easy-to-spoof channels (RF/Optical) report a strong event while the
    /// hard-to-spoof physical channels (Vibration/Thermal) stayed silent.
    AdversarialReplay,
    /// A transient reflection/echo: familiar (low novelty), short-lived, mixed
    /// support — present but not a durable, coherent event.
    EnvironmentalArtifact,
}

/// One scored explanation. `evidence` is normalised to `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedHypothesis {
    /// The candidate explanation.
    pub hypothesis: Hypothesis,
    /// Strength of support for this explanation, in `[0, 1]`.
    pub evidence: f32,
}

/// Inputs describing a single witnessed disagreement.
#[derive(Debug, Clone)]
pub struct DisagreementInput {
    /// Modalities that *did* respond to the change.
    pub supporting: Vec<Modality>,
    /// Modalities that *should* have responded in this zone but stayed silent.
    pub contradicting: Vec<Modality>,
    /// How surprising the signal is, `[0, 1]`.
    pub novelty: f32,
    /// Cleanliness of the spatial boundary, `[0, 1]` (1 = crisp, 0 = smeared).
    pub coherence: f32,
    /// How long the signal has persisted across windows, `[0, 1]`.
    pub persistence: f32,
}

impl DisagreementInput {
    /// Mean spoof-resistance of the supporting set (0 if empty).
    fn supporting_spoof_resistance(&self) -> f32 {
        mean_spoof_resistance(&self.supporting)
    }

    /// Mean spoof-resistance of the contradicting set (0 if empty).
    fn contradicting_spoof_resistance(&self) -> f32 {
        mean_spoof_resistance(&self.contradicting)
    }

    /// Fraction of involved modalities that are contradicting, `[0, 1]`.
    /// 0 when nothing is involved at all.
    fn contradiction_fraction(&self) -> f32 {
        let total = self.supporting.len() + self.contradicting.len();
        if total == 0 {
            return 0.0;
        }
        self.contradicting.len() as f32 / total as f32
    }
}

/// Mean spoof-resistance of a modality set; 0.0 for an empty set.
fn mean_spoof_resistance(set: &[Modality]) -> f32 {
    if set.is_empty() {
        return 0.0;
    }
    let sum: f32 = set.iter().map(|m| m.physics().spoof_resistance).sum();
    sum / set.len() as f32
}

/// Clamp a raw score into `[0, 1]`.
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Rank all candidate explanations for a disagreement by evidence (descending).
///
/// Always returns exactly five [`RankedHypothesis`] entries. Ties keep the
/// canonical declaration order ([`Hypothesis::RealEvent`] first) because the
/// sort is stable and the candidates are pushed in that order.
pub fn rank_hypotheses(input: &DisagreementInput) -> Vec<RankedHypothesis> {
    let mut ranked = vec![
        RankedHypothesis {
            hypothesis: Hypothesis::RealEvent,
            evidence: score_real_event(input),
        },
        RankedHypothesis {
            hypothesis: Hypothesis::SensorDrift,
            evidence: score_sensor_drift(input),
        },
        RankedHypothesis {
            hypothesis: Hypothesis::SensorRelocation,
            evidence: score_sensor_relocation(input),
        },
        RankedHypothesis {
            hypothesis: Hypothesis::AdversarialReplay,
            evidence: score_adversarial_replay(input),
        },
        RankedHypothesis {
            hypothesis: Hypothesis::EnvironmentalArtifact,
            evidence: score_environmental_artifact(input),
        },
    ];

    // Stable, descending by evidence. `total_cmp` keeps this deterministic even
    // for NaN-free f32s and never panics. Stability preserves declaration order
    // on ties.
    ranked.sort_by(|a, b| b.evidence.total_cmp(&a.evidence));
    ranked
}

/// **RealEvent**: many trustworthy supporting channels, a crisp boundary, some
/// novelty and decent persistence, with few contradictions. We weight by the
/// supporting set's mean spoof-resistance (hard-to-spoof agreement is the
/// strongest signal a thing actually happened) and penalise by the contradiction
/// fraction.
fn score_real_event(input: &DisagreementInput) -> f32 {
    if input.supporting.is_empty() {
        return 0.0;
    }
    let trust = input.supporting_spoof_resistance();
    // Reward breadth of support: two trustworthy channels beat one.
    let breadth = (input.supporting.len() as f32 / 3.0).min(1.0);
    let persistence_term = 0.5 + 0.5 * clamp01(input.persistence);
    let novelty_term = 0.5 + 0.5 * clamp01(input.novelty);
    let base = trust * clamp01(input.coherence) * persistence_term * novelty_term;
    let breadth_boosted = base * (0.6 + 0.4 * breadth);
    // Contradictions erode a "real event" reading.
    clamp01(breadth_boosted * (1.0 - input.contradiction_fraction()))
}

/// **SensorDrift**: a lone low-spoof-resistance channel slowly wandering. High
/// when exactly one modality supports, that modality is easy to spoof / noisy
/// (low spoof-resistance), the boundary is *incoherent* (drift is not a clean
/// spatial edge), and persistence is high (drift is slow and sustained). Novelty
/// should be modest — drift creeps, it does not jump.
fn score_sensor_drift(input: &DisagreementInput) -> f32 {
    if input.supporting.len() != 1 {
        return 0.0;
    }
    let weak_channel = 1.0 - input.supporting_spoof_resistance();
    let incoherence = 1.0 - clamp01(input.coherence);
    let persistent = clamp01(input.persistence);
    // Gradual: penalise high novelty (that points at relocation instead).
    let gradual = 1.0 - clamp01(input.novelty);
    clamp01(weak_channel * incoherence * persistent * (0.5 + 0.5 * gradual))
}

/// **SensorRelocation**: a sensor moved, so it still responds strongly but its
/// readings no longer fit the spatial field. Distinguished from drift by being
/// *sudden and novel* rather than gradual: support present, coherence LOW
/// (doesn't fit neighbours), novelty HIGH, and at least one contradiction
/// (neighbours that should agree don't). Persistence is not required — a
/// relocation is a step change.
fn score_sensor_relocation(input: &DisagreementInput) -> f32 {
    if input.supporting.is_empty() || input.contradicting.is_empty() {
        return 0.0;
    }
    let responding = clamp01(input.supporting_spoof_resistance().max(0.2));
    let incoherence = 1.0 - clamp01(input.coherence);
    let sudden = clamp01(input.novelty);
    let mismatch = input.contradiction_fraction();
    clamp01(responding * incoherence * sudden * (0.5 + 0.5 * mismatch))
}

/// **AdversarialReplay**: the supporting set is dominated by easy-to-spoof
/// channels (RF/Optical) while the *hard-to-spoof* physical channels
/// (Vibration/Thermal) are in the contradicting set (silent). A fake can drive
/// radios and light but cannot reproduce structural vibration or thermal mass.
/// Persistence is usually low/static for a replayed snippet, so low persistence
/// adds a little weight.
fn score_adversarial_replay(input: &DisagreementInput) -> f32 {
    if input.supporting.is_empty() || input.contradicting.is_empty() {
        return 0.0;
    }
    // Supporting must be *easy* to spoof; contradicting must be *hard* to spoof.
    let support_spoofability = 1.0 - input.supporting_spoof_resistance();
    let silent_trust = input.contradicting_spoof_resistance();
    // Only meaningful when the hard channels are the silent ones.
    if silent_trust <= input.supporting_spoof_resistance() {
        return 0.0;
    }
    let static_signal = 1.0 - clamp01(input.persistence);
    let core = support_spoofability * silent_trust;
    clamp01(core * (0.7 + 0.3 * static_signal))
}

/// **EnvironmentalArtifact**: a transient reflection/echo. Familiar rather than
/// novel (low novelty), short-lived (low persistence), with moderate coherence
/// and mixed support — it shows up but never settles into a durable, trustworthy
/// event.
fn score_environmental_artifact(input: &DisagreementInput) -> f32 {
    if input.supporting.is_empty() {
        return 0.0;
    }
    let familiar = 1.0 - clamp01(input.novelty);
    let transient = 1.0 - clamp01(input.persistence);
    // Moderate coherence peaks at ~0.5 (an echo is neither crisp nor formless).
    let moderate_coherence = 1.0 - (clamp01(input.coherence) - 0.5).abs() * 2.0;
    // Low-trust support is more echo-like than a hard physical channel.
    let soft_support = 1.0 - input.supporting_spoof_resistance();
    clamp01(familiar * transient * (0.4 + 0.6 * moderate_coherence) * (0.5 + 0.5 * soft_support))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first(input: &DisagreementInput) -> Hypothesis {
        rank_hypotheses(input)[0].hypothesis
    }

    #[test]
    fn returns_all_five_sorted_descending() {
        let input = DisagreementInput {
            supporting: vec![Modality::Vibration],
            contradicting: vec![],
            novelty: 0.5,
            coherence: 0.5,
            persistence: 0.5,
        };
        let ranked = rank_hypotheses(&input);
        assert_eq!(ranked.len(), 5);
        for w in ranked.windows(2) {
            assert!(w[0].evidence >= w[1].evidence);
            assert!((0.0..=1.0).contains(&w[0].evidence));
        }
    }

    #[test]
    fn many_trustworthy_supporters_imply_real_event() {
        // Hard-to-spoof channels agree, boundary is crisp, contradictions are
        // absent — this is what a genuine physical event looks like.
        let input = DisagreementInput {
            supporting: vec![Modality::Vibration, Modality::Thermal, Modality::Acoustic],
            contradicting: vec![],
            novelty: 0.7,
            coherence: 0.9,
            persistence: 0.7,
        };
        assert_eq!(first(&input), Hypothesis::RealEvent);
    }

    #[test]
    fn lone_weak_persistent_channel_implies_drift() {
        // A single easy-to-spoof channel, no clean boundary, sustained over time,
        // creeping (low novelty): the signature of calibration drift.
        let input = DisagreementInput {
            supporting: vec![Modality::Rf],
            contradicting: vec![],
            novelty: 0.2,
            coherence: 0.1,
            persistence: 0.9,
        };
        assert_eq!(first(&input), Hypothesis::SensorDrift);
    }

    #[test]
    fn easy_channels_loud_hard_channels_silent_imply_replay() {
        // RF + Optical (easy to spoof) report a strong event, while Vibration +
        // Thermal (hard to spoof) are silent — a classic replayed/faked signal.
        let input = DisagreementInput {
            supporting: vec![Modality::Rf, Modality::Optical],
            contradicting: vec![Modality::Vibration, Modality::Thermal],
            novelty: 0.6,
            coherence: 0.6,
            persistence: 0.1,
        };
        assert_eq!(first(&input), Hypothesis::AdversarialReplay);
    }

    #[test]
    fn sudden_novel_incoherent_with_contradiction_implies_relocation() {
        // A trustworthy sensor still responds strongly, but suddenly (high
        // novelty), incoherently, and its neighbours contradict it: it moved.
        let input = DisagreementInput {
            supporting: vec![Modality::Vibration],
            contradicting: vec![Modality::Acoustic],
            novelty: 0.95,
            coherence: 0.1,
            persistence: 0.2,
        };
        let ranked = rank_hypotheses(&input);
        // Relocation should out-rank drift here because the change is sudden.
        let reloc = ranked
            .iter()
            .find(|r| r.hypothesis == Hypothesis::SensorRelocation)
            .unwrap()
            .evidence;
        let drift = ranked
            .iter()
            .find(|r| r.hypothesis == Hypothesis::SensorDrift)
            .unwrap()
            .evidence;
        assert!(reloc > drift);
    }

    #[test]
    fn familiar_transient_implies_environmental_artifact() {
        // Low novelty, short-lived, moderate coherence, soft support: an echo.
        let input = DisagreementInput {
            supporting: vec![Modality::Optical],
            contradicting: vec![],
            novelty: 0.05,
            coherence: 0.5,
            persistence: 0.05,
        };
        assert_eq!(first(&input), Hypothesis::EnvironmentalArtifact);
    }

    #[test]
    fn empty_supporting_is_deterministic_and_bounded() {
        let input = DisagreementInput {
            supporting: vec![],
            contradicting: vec![Modality::Thermal],
            novelty: 0.5,
            coherence: 0.5,
            persistence: 0.5,
        };
        let ranked = rank_hypotheses(&input);
        assert_eq!(ranked.len(), 5);
        for r in &ranked {
            assert!((0.0..=1.0).contains(&r.evidence));
        }
    }
}
