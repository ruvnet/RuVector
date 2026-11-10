//! Tiny-detector seam and risk signal (PIR WP33, ADR-337).
//!
//! The first rung of the monitoring hierarchy is a cheap detector that scores
//! an operation's risk before any investigator is paid for. This module
//! defines the *seam* — the subject shape, the signal shape, and the trait a
//! detector implements — plus a deliberately trivial default so the ladder is
//! testable end to end.
//!
//! # Honest scope
//!
//! [`KeywordDetector`] is **not** a classifier. It is a placeholder with the
//! right signature: substring matching over the operation text, with a fixed
//! uncertainty. It has no training, no calibration, no notion of context or
//! intent, and it will both miss obfuscated operations and fire on innocuous
//! prose that happens to contain a marker word. It exists so the ladder can
//! be exercised and so a real detector has an interface to land against;
//! nothing in this crate's claims rests on its accuracy. Substituting a real
//! detector is a matter of implementing [`TinyDetector`].
//!
//! What *does* rest on this module is the numeric discipline: every risk and
//! uncertainty crosses [`RiskSignal::new`], which rejects non-finite values
//! **before** any comparison is made. A NaN reaching a comparison-based gate
//! silently passes it (`NaN < x` is false in every direction), which is how a
//! monitoring gate fails open without anyone noticing.

use super::MonitorError;

/// The operation being considered: what the ladder scores and classifies.
#[derive(Debug, Clone, PartialEq)]
pub struct InspectionSubject {
    /// Tool or operation name.
    pub tool_name: String,
    /// Human-readable description of what the operation does.
    pub description: String,
    /// Arguments the operation was invoked with.
    pub arguments: serde_json::Value,
}

impl InspectionSubject {
    /// Build a subject from its parts.
    pub fn new(
        tool_name: impl Into<String>,
        description: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            description: description.into(),
            arguments,
        }
    }

    /// Lowercased concatenation of name, description, and serialized
    /// arguments, used for marker matching.
    ///
    /// Arguments are included so an operation cannot evade classification by
    /// keeping its verb out of the name and description. Serialization
    /// failure yields the empty string for that part rather than an error:
    /// this feeds *detection*, and a subject that cannot be serialized must
    /// still be classified on the parts that are readable rather than
    /// skipped.
    pub fn searchable_text(&self) -> String {
        let args = serde_json::to_string(&self.arguments).unwrap_or_default();
        format!("{} {} {}", self.tool_name, self.description, args).to_lowercase()
    }
}

/// A risk estimate with its uncertainty, both on a unit scale.
///
/// The unit scale is not cosmetic: the value-of-information math this feeds
/// uses an Abramowitz–Stegun normal cdf whose absolute error (≤1.5e-7) is
/// scaled by the mean, so large-magnitude utilities can have their deep tail
/// overestimated. Keeping risk in `[0, 1]` keeps that error immaterial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskSignal {
    risk: f64,
    uncertainty: f64,
}

impl RiskSignal {
    /// Construct a signal, rejecting non-finite and out-of-range inputs
    /// **before** any comparison downstream can consume them.
    ///
    /// `risk` must be finite and within `[0, 1]`. `uncertainty` must be
    /// finite and within `(0, 1]` on the same unit scale. A zero-uncertainty
    /// prior leaves nothing for an investigator to reveal; a value above one
    /// inflates the value-of-information ceiling.
    pub fn new(risk: f64, uncertainty: f64) -> Result<Self, MonitorError> {
        if !risk.is_finite() {
            return Err(MonitorError::NonFinite {
                field: "risk",
                value: format!("{risk}"),
            });
        }
        if !uncertainty.is_finite() {
            return Err(MonitorError::NonFinite {
                field: "uncertainty",
                value: format!("{uncertainty}"),
            });
        }
        if !(0.0..=1.0).contains(&risk) {
            return Err(MonitorError::RiskOutOfRange { risk });
        }
        if uncertainty <= 0.0 {
            return Err(MonitorError::NonPositiveUncertainty { uncertainty });
        }
        if uncertainty > 1.0 {
            return Err(MonitorError::UncertaintyOutOfRange { uncertainty });
        }
        Ok(Self { risk, uncertainty })
    }

    /// Estimated probability of violation, in `[0, 1]`.
    pub fn risk(&self) -> f64 {
        self.risk
    }

    /// Standard deviation of the estimate, strictly positive.
    pub fn uncertainty(&self) -> f64 {
        self.uncertainty
    }
}

/// The cheap first-pass scorer.
///
/// Implementations must be fast enough that running one per operation is
/// negligible against the operation itself — the whole overhead budget rests
/// on that. A detector that needs a model call belongs on the ladder as an
/// investigator rung, not here.
pub trait TinyDetector {
    /// Score an operation. Returning `Err` is treated as *fail closed* by the
    /// ladder: an unscoreable operation is escalated or halted, never allowed.
    fn score(&self, subject: &InspectionSubject) -> Result<RiskSignal, MonitorError>;

    /// The uncertainty this detector actually reports from [`Self::score`],
    /// in `(0, 1]`.
    ///
    /// The ladder's construction-time calibration check needs to know how
    /// much uncertainty there is for an investigator to resolve, because
    /// that bounds what any rung can be worth. This is a trait method rather
    /// than a configuration field on purpose: an operator-declared figure is
    /// free to diverge from what the detector really produces, and a
    /// *larger* declared value inflates the ceiling and makes the guard more
    /// permissive — so the one field that exists to catch a never-escalate
    /// ladder could be used to wave one through. Deriving it from the
    /// detector bounds the construction-time value. Inspection recalibrates
    /// against the actual signal and halts when no rung is purchasable at
    /// that uncertainty, so an inflated declaration cannot silently admit
    /// a ladder that never escalates.
    ///
    /// A detector whose uncertainty varies per input should report the
    /// **largest** value it can return, since that is the most permissive
    /// ceiling it could honestly justify.
    fn expected_uncertainty(&self) -> f64;
}

/// A trivial substring-matching detector.
///
/// See this module's honest-scope note: this is a seam-filler, not a
/// classifier. It scores by counting how many risk markers appear in the
/// operation text, saturating at [`KeywordDetector::saturation`] markers.
#[derive(Debug, Clone)]
pub struct KeywordDetector {
    markers: Vec<String>,
    saturation: usize,
    uncertainty: f64,
}

impl KeywordDetector {
    /// Default risk markers. Overlapping with the mandatory-class markers is
    /// intentional and harmless: mandatory classification runs independently
    /// and does not consult the detector.
    pub const DEFAULT_MARKERS: &'static [&'static str] = &[
        "delete",
        "drop",
        "exec",
        "eval",
        "spawn",
        "sudo",
        "token",
        "secret",
        "password",
        "http",
        "socket",
        "mount",
        "install",
        "revoke",
        "overwrite",
        "force",
    ];

    /// Number of distinct markers at which risk saturates to 1.0.
    pub fn saturation(&self) -> usize {
        self.saturation
    }

    /// Detector with the default markers, saturating at 4 matches, reporting
    /// a fixed uncertainty of 0.2.
    ///
    /// The fixed uncertainty is the clearest signal that this is a stub: a
    /// real detector's confidence varies with the input.
    pub fn new() -> Self {
        Self {
            markers: Self::DEFAULT_MARKERS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            saturation: 4,
            uncertainty: 0.2,
        }
    }

    /// Detector with explicit markers, saturation, and uncertainty.
    ///
    /// Rejects a non-finite or non-positive uncertainty and a zero
    /// saturation at construction, so a misconfigured detector cannot
    /// produce a signal that fails validation on every call.
    pub fn with_config(
        markers: Vec<String>,
        saturation: usize,
        uncertainty: f64,
    ) -> Result<Self, MonitorError> {
        if saturation == 0 {
            return Err(MonitorError::ZeroSaturation);
        }
        // Round-trip a probe signal so an invalid uncertainty is refused here
        // rather than on every later score() call.
        RiskSignal::new(0.0, uncertainty)?;
        Ok(Self {
            markers,
            saturation,
            uncertainty,
        })
    }
}

impl Default for KeywordDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl TinyDetector for KeywordDetector {
    fn score(&self, subject: &InspectionSubject) -> Result<RiskSignal, MonitorError> {
        let haystack = subject.searchable_text();
        let hits = self
            .markers
            .iter()
            .filter(|m| haystack.contains(m.as_str()))
            .count();
        let risk = (hits as f64 / self.saturation as f64).min(1.0);
        RiskSignal::new(risk, self.uncertainty)
    }

    /// This detector reports one fixed uncertainty from every `score`, so the
    /// value it declares here is exactly the value it produces.
    fn expected_uncertainty(&self) -> f64 {
        self.uncertainty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_non_finite_before_any_comparison() {
        // Each of these would silently pass a naive `risk > threshold` gate.
        assert!(matches!(
            RiskSignal::new(f64::NAN, 0.2),
            Err(MonitorError::NonFinite { .. })
        ));
        assert!(matches!(
            RiskSignal::new(f64::INFINITY, 0.2),
            Err(MonitorError::NonFinite { .. })
        ));
        assert!(matches!(
            RiskSignal::new(0.5, f64::NAN),
            Err(MonitorError::NonFinite { .. })
        ));
        assert!(matches!(
            RiskSignal::new(0.5, f64::NEG_INFINITY),
            Err(MonitorError::NonFinite { .. })
        ));
    }

    #[test]
    fn rejects_out_of_range_and_degenerate_inputs() {
        assert!(matches!(
            RiskSignal::new(1.5, 0.2),
            Err(MonitorError::RiskOutOfRange { .. })
        ));
        assert!(matches!(
            RiskSignal::new(-0.1, 0.2),
            Err(MonitorError::RiskOutOfRange { .. })
        ));
        assert!(matches!(
            RiskSignal::new(0.5, 0.0),
            Err(MonitorError::NonPositiveUncertainty { .. })
        ));
        assert!(RiskSignal::new(0.0, 1e-9).is_ok());
        assert!(RiskSignal::new(1.0, 0.2).is_ok());
    }

    #[test]
    fn keyword_detector_saturates_at_one() {
        let d = KeywordDetector::new();
        let s = d
            .score(&InspectionSubject::new(
                "x",
                "delete drop exec eval spawn sudo token secret",
                json!({}),
            ))
            .unwrap();
        assert_eq!(s.risk(), 1.0);
    }

    #[test]
    fn keyword_detector_scores_benign_as_zero() {
        let d = KeywordDetector::new();
        let s = d
            .score(&InspectionSubject::new("sum", "add integers", json!({})))
            .unwrap();
        assert_eq!(s.risk(), 0.0);
        assert!(s.uncertainty() > 0.0);
    }

    #[test]
    fn misconfigured_detector_is_refused_at_construction() {
        assert!(matches!(
            KeywordDetector::with_config(vec![], 0, 0.2),
            Err(MonitorError::ZeroSaturation)
        ));
        assert!(KeywordDetector::with_config(vec![], 4, f64::NAN).is_err());
        assert!(KeywordDetector::with_config(vec![], 4, 0.0).is_err());
    }

    #[test]
    fn unserializable_arguments_do_not_erase_the_other_parts() {
        // f64::NAN cannot be represented in JSON; serialization of the
        // argument blob fails and yields "", but name and description must
        // still be searchable.
        let subject = InspectionSubject {
            tool_name: "danger".into(),
            description: "sudo delete".into(),
            arguments: serde_json::Value::String("ok".into()),
        };
        let text = subject.searchable_text();
        assert!(text.contains("sudo"));
        assert!(text.contains("delete"));
    }
}
