//! Physical CAPTCHA — proof-of-reality via active challenge–response.
//!
//! A replayed or statically-spoofed sensor stream can mimic *passive* readings,
//! but it cannot answer a *fresh* physical challenge. This module models that
//! interaction: the device emits a [`Stimulus`] (a chirp, an RF pulse, a tap,
//! …) and expects a multi-modal [`ObservedResponse`] with characteristic
//! per-modality delays and magnitudes.
//!
//! [`CaptchaVerifier`] learns the expected response *profile* for each stimulus
//! from known-good challenges (an EWMA over delay and magnitude per modality),
//! then [`CaptchaVerifier::verify`] scores a fresh observation against it. The
//! score is weighted by each modality's [`spoof_resistance`]: a missing
//! hard-to-fake modality (e.g. vibration or thermal) costs far more than a
//! missing easy-to-fake one (e.g. RF or optical).
//!
//! [`spoof_resistance`]: crate::modality::Physics::spoof_resistance

use crate::modality::Modality;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// EWMA smoothing factor for profile learning. Newer observations get weight
/// `ALPHA`, the running estimate keeps `1 - ALPHA`.
const ALPHA: f32 = 0.3;

/// Minimum fraction of the expected magnitude an observed response must reach to
/// count as a valid (non-spoofed, non-attenuated) reply.
const MAGNITUDE_FLOOR_FRACTION: f32 = 0.5;

/// An active physical challenge emitted by the device.
///
/// Each variant maps to a distinct emission whose echoes/responses propagate
/// across several [`Modality`]s with modality-specific delays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stimulus {
    /// A swept-frequency acoustic tone ("chirp").
    AcousticChirp,
    /// A short radio-frequency burst.
    RfPulse,
    /// A mechanical tap exciting structural vibration.
    VibrationTap,
    /// A modulated-light flash.
    LightModulation,
    /// A brief thermal/IR pulse.
    ThermalPulse,
}

impl Stimulus {
    /// Stable short name (useful for logs and witnesses).
    pub fn name(self) -> &'static str {
        match self {
            Stimulus::AcousticChirp => "acoustic-chirp",
            Stimulus::RfPulse => "rf-pulse",
            Stimulus::VibrationTap => "vibration-tap",
            Stimulus::LightModulation => "light-modulation",
            Stimulus::ThermalPulse => "thermal-pulse",
        }
    }
}

/// One observed response on a single modality to an emitted [`Stimulus`].
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedResponse {
    /// Which modality this reading came from.
    pub modality: Modality,
    /// Time from stimulus emission to observed response (seconds).
    pub delay: f32,
    /// Response magnitude in arbitrary, modality-normalised units (`>= 0`).
    pub magnitude: f32,
}

/// A complete challenge–response record: one stimulus, many modality responses.
#[derive(Debug, Clone)]
pub struct ChallengeResponse {
    /// The stimulus that was emitted.
    pub stimulus: Stimulus,
    /// All observed responses (at most one expected per modality).
    pub responses: Vec<ObservedResponse>,
}

/// The verdict produced by [`CaptchaVerifier::verify`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealityProof {
    /// `true` when the weighted score met the verifier's `min_score`.
    pub trusted: bool,
    /// Weighted fraction of expected modalities that responded correctly, in `[0, 1]`.
    pub score: f32,
    /// Expected modalities that were missing or out of tolerance.
    pub missing: Vec<Modality>,
    /// Human-readable explanation of the verdict.
    pub reason: String,
}

/// Expected per-modality response, maintained as a running EWMA.
#[derive(Debug, Clone, Copy)]
struct Expected {
    delay: f32,
    magnitude: f32,
}

/// Learns expected challenge-response profiles and verifies fresh observations
/// against them.
#[derive(Debug, Clone)]
pub struct CaptchaVerifier {
    /// Per `(Stimulus, Modality)` expected response, learned via EWMA.
    profiles: HashMap<(Stimulus, Modality), Expected>,
    /// Allowed absolute deviation in delay (seconds) for a response to count.
    delay_tolerance: f32,
    /// Minimum weighted score to mark a proof as `trusted`.
    min_score: f32,
}

impl CaptchaVerifier {
    /// Create a verifier.
    ///
    /// * `delay_tolerance` — max absolute delay error (seconds) tolerated per modality.
    /// * `min_score` — weighted score threshold (`[0, 1]`) for `trusted`.
    pub fn new(delay_tolerance: f32, min_score: f32) -> Self {
        Self {
            profiles: HashMap::new(),
            delay_tolerance,
            min_score,
        }
    }

    /// Learn (or update) the expected response profile for a stimulus from a
    /// known-good challenge response.
    ///
    /// For every observed modality the expected `{delay, magnitude}` is folded
    /// in with an EWMA (`ALPHA`). The first time a `(stimulus, modality)` pair
    /// is seen it is initialised directly to the observed values.
    pub fn learn(&mut self, cr: &ChallengeResponse) {
        for r in &cr.responses {
            let key = (cr.stimulus, r.modality);
            self.profiles
                .entry(key)
                .and_modify(|e| {
                    e.delay = ewma(e.delay, r.delay);
                    e.magnitude = ewma(e.magnitude, r.magnitude);
                })
                .or_insert(Expected {
                    delay: r.delay,
                    magnitude: r.magnitude,
                });
        }
    }

    /// Verify an observed challenge response against the learned profile.
    ///
    /// Each expected modality contributes weight equal to its
    /// [`spoof_resistance`]; a modality passes only when an observed response
    /// exists whose delay is within `delay_tolerance` and whose magnitude is at
    /// least [`MAGNITUDE_FLOOR_FRACTION`] of the expected magnitude. The
    /// returned `score` is `passed_weight / total_weight`.
    ///
    /// [`spoof_resistance`]: crate::modality::Physics::spoof_resistance
    pub fn verify(&self, cr: &ChallengeResponse) -> RealityProof {
        // Gather everything we expect for this stimulus.
        let expected: Vec<(Modality, Expected)> = Modality::ALL
            .iter()
            .filter_map(|&m| self.profiles.get(&(cr.stimulus, m)).map(|e| (m, *e)))
            .collect();

        if expected.is_empty() {
            return RealityProof {
                trusted: false,
                score: 0.0,
                missing: Vec::new(),
                reason: "unknown stimulus profile".to_string(),
            };
        }

        let mut total_weight = 0.0_f32;
        let mut passed_weight = 0.0_f32;
        let mut missing: Vec<Modality> = Vec::new();

        for (modality, exp) in &expected {
            let weight = modality.physics().spoof_resistance;
            total_weight += weight;

            if self.responded_correctly(cr, *modality, exp) {
                passed_weight += weight;
            } else {
                missing.push(*modality);
            }
        }

        // `total_weight` is > 0 because `expected` is non-empty and every
        // modality has a positive spoof_resistance.
        let score = passed_weight / total_weight;
        let trusted = score >= self.min_score;
        let reason = self.explain(trusted, &missing);

        RealityProof {
            trusted,
            score,
            missing,
            reason,
        }
    }

    /// Does `cr` contain a correct response for `modality` matching `exp`?
    fn responded_correctly(
        &self,
        cr: &ChallengeResponse,
        modality: Modality,
        exp: &Expected,
    ) -> bool {
        let floor = exp.magnitude * MAGNITUDE_FLOOR_FRACTION;
        cr.responses.iter().any(|r| {
            r.modality == modality
                && (r.delay - exp.delay).abs() <= self.delay_tolerance
                && r.magnitude >= floor
        })
    }

    /// Build a human-readable reason for the verdict.
    fn explain(&self, trusted: bool, missing: &[Modality]) -> String {
        if missing.is_empty() {
            return "all expected modalities responded within tolerance".to_string();
        }

        // Surface the hardest-to-spoof missing modality first — that is the one
        // an attacker is least likely to fake.
        let worst = missing
            .iter()
            .copied()
            .max_by(|a, b| {
                a.physics()
                    .spoof_resistance
                    .partial_cmp(&b.physics().spoof_resistance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("missing is non-empty here");

        let names: Vec<&str> = missing.iter().map(|m| m.name()).collect();
        let prefix = if trusted {
            "degraded but trusted"
        } else {
            "rejected"
        };
        format!(
            "{prefix}: missing high-spoof-resistance modality: {} (absent/out-of-tolerance: {})",
            worst.name(),
            names.join(", ")
        )
    }
}

/// Exponentially-weighted moving-average update.
fn ewma(prev: f32, sample: f32) -> f32 {
    (1.0 - ALPHA) * prev + ALPHA * sample
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative known-good response for an acoustic chirp across the
    /// fast modalities plus vibration (the hard-to-spoof one).
    fn good_chirp() -> ChallengeResponse {
        ChallengeResponse {
            stimulus: Stimulus::AcousticChirp,
            responses: vec![
                ObservedResponse {
                    modality: Modality::Acoustic,
                    delay: 0.030,
                    magnitude: 1.0,
                },
                ObservedResponse {
                    modality: Modality::Vibration,
                    delay: 0.050,
                    magnitude: 0.8,
                },
                ObservedResponse {
                    modality: Modality::Rf,
                    delay: 0.010,
                    magnitude: 0.5,
                },
            ],
        }
    }

    #[test]
    fn matching_response_is_trusted() {
        let mut v = CaptchaVerifier::new(0.01, 0.8);
        // Learn the profile a few times so the EWMA settles.
        for _ in 0..5 {
            v.learn(&good_chirp());
        }

        let proof = v.verify(&good_chirp());
        assert!(
            proof.trusted,
            "matching response should be trusted: {proof:?}"
        );
        assert!(
            proof.score > 0.99,
            "score should be near 1.0: {}",
            proof.score
        );
        assert!(proof.missing.is_empty());
        assert_eq!(
            proof.reason,
            "all expected modalities responded within tolerance"
        );
    }

    #[test]
    fn replayed_missing_vibration_is_rejected() {
        let mut v = CaptchaVerifier::new(0.01, 0.8);
        for _ in 0..5 {
            v.learn(&good_chirp());
        }

        // A replay that drops vibration entirely and zeroes the rest's delays.
        let replay = ChallengeResponse {
            stimulus: Stimulus::AcousticChirp,
            responses: vec![
                ObservedResponse {
                    modality: Modality::Acoustic,
                    delay: 0.0,
                    magnitude: 1.0,
                },
                ObservedResponse {
                    modality: Modality::Rf,
                    delay: 0.0,
                    magnitude: 0.5,
                },
            ],
        };

        let proof = v.verify(&replay);
        assert!(!proof.trusted, "replay should be rejected: {proof:?}");
        assert!(
            proof.missing.contains(&Modality::Vibration),
            "vibration must be flagged missing: {:?}",
            proof.missing
        );
        // Acoustic delay (0.0 vs ~0.03) is out of the 0.01 tolerance too.
        assert!(proof.missing.contains(&Modality::Acoustic));
        assert!(proof.score < 0.8);
    }

    #[test]
    fn unknown_stimulus_is_rejected() {
        let v = CaptchaVerifier::new(0.01, 0.8);
        let probe = ChallengeResponse {
            stimulus: Stimulus::ThermalPulse,
            responses: vec![ObservedResponse {
                modality: Modality::Thermal,
                delay: 2.0,
                magnitude: 1.0,
            }],
        };

        let proof = v.verify(&probe);
        assert!(!proof.trusted);
        assert_eq!(proof.score, 0.0);
        assert_eq!(proof.reason, "unknown stimulus profile");
        assert!(proof.missing.is_empty());
    }

    #[test]
    fn weak_magnitude_fails_tolerance() {
        let mut v = CaptchaVerifier::new(0.01, 0.9);
        for _ in 0..5 {
            v.learn(&good_chirp());
        }

        // Vibration arrives on time but with collapsed magnitude (< 50%).
        let attenuated = ChallengeResponse {
            stimulus: Stimulus::AcousticChirp,
            responses: vec![
                ObservedResponse {
                    modality: Modality::Acoustic,
                    delay: 0.030,
                    magnitude: 1.0,
                },
                ObservedResponse {
                    modality: Modality::Vibration,
                    delay: 0.050,
                    magnitude: 0.1,
                },
                ObservedResponse {
                    modality: Modality::Rf,
                    delay: 0.010,
                    magnitude: 0.5,
                },
            ],
        };

        let proof = v.verify(&attenuated);
        assert!(
            !proof.trusted,
            "weak vibration should fail high threshold: {proof:?}"
        );
        assert!(proof.missing.contains(&Modality::Vibration));
    }
}
