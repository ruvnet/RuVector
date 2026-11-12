//! Tunable engine knobs (ADR-004 loops 2–3: the head/probe/calibration/abstain
//! parameters a campaign proposes over). Every field is defaulted so that
//! `EngineOptions::default()` reproduces the engine's original behaviour
//! bit-for-bit — the probe hyper-parameters, the `not_for` penalty, the abstain
//! logit constants and the calibration slice all match the pre-options code.
//! Options round-trip through serde (camelCase on the wire) so the bindings and
//! the campaign can carry `{ "engine": { ... } }`.

use serde::{Deserialize, Serialize};

/// Which head to use for a `choice` / `score` question. `Auto` keeps the
/// original rule (probe when every class clears the minimum, else prototype);
/// `Prototype` / `Probe` pin it, for a campaign arm that wants to isolate one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum HeadChoice {
    #[default]
    Auto,
    Prototype,
    Probe,
}

/// How `meta.abstain` is reported for `choice` / `score` answers.
///
/// `Softmax` (default, original behaviour) reports the abstain logit's share of
/// a (K+1)-way softmax with the option scores, after the fitted temperature. Its
/// scale shrinks as the option count K grows, and after training a sharp fitted
/// temperature pushes it towards zero, so one threshold does not carry across
/// questions. `Sigmoid` reports `sigmoid(abstain_logit)`: the same
/// out-of-scope signal (distance to the nearest prototype, or the best
/// `not_for` match) on a fixed 0–1 scale that does not depend on K or on the
/// head's temperature. `choice`, `probabilities` and `confidence` are identical
/// in both modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AbstainMode {
    #[default]
    Softmax,
    Sigmoid,
}

/// The engine's tunable parameters. Defaults reproduce the original constants:
/// probe `lr=0.8, l2=1e-3, iters=400`; `not_for` λ = 0.5; abstain τ = 0.35,
/// scale = 0.5; logit scale = 1.0 (no sharpening); calibration slice = 20 %
/// with a floor of 20 (ADR-003/ADR-006).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct EngineOptions {
    /// Linear-probe L2 regularisation strength.
    pub probe_l2: f32,
    /// Linear-probe full-batch gradient-descent iterations.
    pub probe_iterations: usize,
    /// Linear-probe learning rate.
    pub probe_learning_rate: f32,
    /// Whether the probe loss is class-balanced (each class weighted to an equal
    /// total), matching the binary head. Off by default (original behaviour).
    pub probe_class_balanced: bool,
    /// Penalty weight on the `not_for` hard-negative similarity (prototype head).
    pub not_for_lambda: f32,
    /// Similarity below which a state starts to look out-of-scope (abstain).
    pub abstain_tau: f32,
    /// Scale of the abstain distance term.
    pub abstain_scale: f32,
    /// Inverse-temperature prior applied to the class logits *before*
    /// temperature fitting and the abstain softmax. `1.0` is a no-op; a larger
    /// value (e.g. 20.0) sharpens flat cosine logits so calibration has range to
    /// work with. Applied consistently in fitting and in `decide`.
    pub logit_scale: f32,
    /// Fraction of a question's training examples reserved for the calibration
    /// slice, in `(0, 1)`. Drives the default bank split for the plain `train`
    /// path: `calibration = round(fraction·100) %`, the rest `Train`.
    pub calibration_fraction: f32,
    /// Calibration-slice size below which `confidence` stays uncalibrated
    /// (temperature / Platt is not fitted). ADR-006 sets the floor.
    pub min_calibration: usize,
    /// Head selection for class questions.
    pub head: HeadChoice,
    /// Prefix a `choice` question's `instructions` to each option's `what`,
    /// examples and `not_for` before embedding, as `score` already does for
    /// its legend. Off by default (original behaviour: `choice` embeds only
    /// the criteria, so its instructions do not affect the answer).
    pub choice_instructions: bool,
    /// How `meta.abstain` is reported (see [`AbstainMode`]). Default `Softmax`.
    pub abstain_mode: AbstainMode,
    /// When the held-out calibration slice is below `min_calibration`, fit the
    /// temperature (class heads) or Platt layer (`noul`) on 5-fold out-of-fold
    /// scores over train ∪ calibration instead of leaving the answer
    /// uncalibrated. Needs only `min_calibration` labels in total rather than
    /// in the slice (about 5× fewer by default). Off by default (original
    /// behaviour); the served head is unchanged, only its calibration layer.
    pub crossfit_calibration: bool,
    /// Key of a catch-all option (e.g. `"other"`) in `choice` questions. Off
    /// by default (`None`: every option, including one named "other", is an
    /// ordinary option). When set and the question has that key, the option's
    /// own text is not scored; its probability is `sigmoid` of the
    /// out-of-scope logit over the other options (distance to their nearest
    /// prototype, or the best `not_for` match), and it is chosen when that
    /// probability reaches `catch_all_threshold`.
    pub catch_all: Option<String>,
    /// Probability at which the catch-all option is chosen. Tune it on
    /// labelled in-scope and off-topic examples; the useful range depends on
    /// the embedder (about 0.34 for bge-small with the default τ and scale).
    pub catch_all_threshold: f32,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            probe_l2: 1e-3,
            probe_iterations: 400,
            probe_learning_rate: 0.8,
            probe_class_balanced: false,
            not_for_lambda: 0.5,
            abstain_tau: 0.35,
            abstain_scale: 0.5,
            logit_scale: 1.0,
            calibration_fraction: 0.2,
            min_calibration: 20,
            head: HeadChoice::Auto,
            choice_instructions: false,
            abstain_mode: AbstainMode::Softmax,
            crossfit_calibration: false,
            catch_all: None,
            catch_all_threshold: 0.5,
        }
    }
}

impl EngineOptions {
    /// Bank split ratios for the plain `train` path: everything is admitted as
    /// `Train`, and the calibration slice is carved positionally at fit time
    /// (see [`calib_stride`](Self::calib_stride)). This keeps the calibration
    /// slice class-stratified and reproducible for small few-shot samples — the
    /// original every-`N`th behaviour — and reserves the bank's frozen five-way
    /// splits for a campaign, which supplies them explicitly.
    #[must_use]
    pub fn train_ratios(&self) -> crate::bank::SplitRatios {
        crate::bank::SplitRatios::new(100, 0, 0, 0, 0).unwrap_or_default()
    }

    /// Positional calibration stride: every `stride`-th admitted example (in
    /// insertion order) is held out for calibration. Derived from
    /// `calibration_fraction` (0.2 → every 5th), floored at 2 so a slice always
    /// exists, and `usize::MAX` when the fraction is 0 (no calibration slice).
    #[must_use]
    pub fn calib_stride(&self) -> usize {
        let f = self.calibration_fraction.clamp(0.0, 1.0);
        if f <= 0.0 {
            return usize::MAX;
        }
        ((1.0 / f).round() as usize).max(2)
    }

    /// Parse from JSON, tolerating a missing object (returns the default).
    pub fn from_json_opt(value: Option<&serde_json::Value>) -> crate::Result<Self> {
        match value {
            None => Ok(Self::default()),
            Some(v) => serde_json::from_value(v.clone())
                .map_err(|e| crate::TypesafeError::Invalid(format!("engine options: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_through_json() {
        let opts = EngineOptions::default();
        let json = serde_json::to_string(&opts).unwrap();
        let back: EngineOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn partial_json_keeps_defaults_for_absent_fields() {
        let opts: EngineOptions =
            serde_json::from_str(r#"{"logitScale":20.0,"head":"probe"}"#).unwrap();
        assert_eq!(opts.logit_scale, 20.0);
        assert_eq!(opts.head, HeadChoice::Probe);
        // Untouched fields keep their defaults.
        assert_eq!(opts.probe_iterations, 400);
        assert_eq!(opts.min_calibration, 20);
    }

    #[test]
    fn plain_train_admits_everything_as_train() {
        let r = EngineOptions::default().train_ratios();
        assert_eq!((r.train, r.calibration), (100, 0));
    }

    #[test]
    fn default_calib_stride_is_every_fifth() {
        assert_eq!(EngineOptions::default().calib_stride(), 5);
    }
}
