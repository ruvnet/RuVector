//! **Adaptive change-point detection** — the M3b deliverable.
//!
//! M3 ran the agentic-clock defensibility gate on real recorded agent traces
//! with a **fixed-window** `mean + kσ` change-point alarm ([`alarm_step`]) and
//! got an *honest null*: the agentic-honest clock never alarms, because the
//! agent's early-exploration churn sets a high fixed baseline that later genuine
//! movement can't clear (0 win / 1 tie / 1 loss vs the fair baseline). M3 itself
//! diagnosed the cause — *a fixed baseline poisoned by early churn* — and named
//! the fix: an **adaptive-window detector** whose reference statistic keeps
//! moving instead of freezing on the first `BASELINE_WINDOW` increments.
//!
//! This module implements the simplest defensible such detector — the
//! **Page–Hinkley test** — as a pure, dependency-free clock-agnostic alarm. It
//! is applied **equally** to the agentic clock *and* the fair baseline (and the
//! constant-rate clocks), exactly as the fixed-window alarm was, so any change in
//! verdict is a fair same-detector-both-sides result, not a manufactured win.
//!
//! [`alarm_step`]: crate::agentic_time::alarm_step
//!
//! ## The Page–Hinkley test (math)
//!
//! Page's cumulative-sum (CUSUM) change-point test (Page, 1954) detects a shift
//! in the mean of a stream `x₁, x₂, …`. The *Hinkley* one-sided form tracks the
//! cumulative deviation of each sample from the **running mean** `x̄_T`, minus a
//! tolerance `δ` for changes considered "normal":
//!
//! ```text
//!   x̄_T = (1/T) Σ_{t≤T} x_t                 (running mean — adapts every step)
//!
//!   upward drift accumulator (detect an *increase* in the mean):
//!     U_T = Σ_{t≤T} (x_t − x̄_t − δ)
//!     m_T = min_{t≤T} U_t
//!     PH_T = U_T − m_T                       (rise above the running minimum)
//!     alarm when PH_T > λ
//!
//!   downward drift accumulator (detect a *decrease*, symmetric):
//!     D_T = Σ_{t≤T} (x_t − x̄_t + δ)
//!     M_T = max_{t≤T} D_t
//!     PH⁻_T = M_T − D_T
//!     alarm when PH⁻_T > λ
//! ```
//!
//! Intuition: while the stream is stationary, `x_t − x̄_t` averages to ≈ 0 and the
//! `−δ` tolerance pulls `U_T` steadily *down*, so `U_T` keeps re-touching its
//! running minimum and `PH_T` stays near 0 — **no alarm on stationary noise**.
//! When the mean genuinely steps **up**, `x_t − x̄_t − δ` turns positive for a run
//! of samples, `U_T` climbs away from its minimum, and once the accumulated rise
//! exceeds `λ` the test **alarms**. Crucially the reference `x̄_t` is a *running*
//! mean over the whole observed prefix, so — unlike `mean + kσ` over a frozen
//! early window — a high-variance early phase does **not** permanently raise the
//! bar; it is averaged into a reference the later drift is measured against.
//!
//! Because an early-exploration phase makes `x̄_t` *larger* early (so later
//! genuine movement clears a smaller relative bar), and because the test responds
//! to a *sustained directional shift* rather than a single-sample threshold
//! crossing, Page–Hinkley is precisely the detector M3 argued for.
//!
//! ## Parameters (PRE-REGISTERED — see `real_trace_eval.rs`)
//!
//! * `delta` (`δ`) — magnitude of change tolerated as "normal" before the
//!   accumulator starts counting it. Small `δ` ⇒ more sensitive.
//! * `lambda` (`λ`) — detection threshold on the cumulative deviation. Large `λ`
//!   ⇒ fewer false alarms, later detection.
//!
//! The real-trace harness fixes `δ` and `λ` **before** any lead is computed and
//! prints them, keeping M3's pre-registration discipline.
//!
//! ## Literature
//!
//! * E. S. Page, *"Continuous Inspection Schemes"*, Biometrika **41**(1/2), 1954,
//!   pp. 100–115 — the original CUSUM change-point test.
//! * D. V. Hinkley, *"Inference about the change-point in a sequence of random
//!   variables"*, Biometrika **57**(1), 1970 — the change-point estimator the
//!   one-sided "Page–Hinkley" running form is named for.
//! * For the streaming/concept-drift framing (same family as the fixed-window
//!   baseline): A. Bifet & R. Gavaldà, *"Learning from Time-Changing Data with
//!   Adaptive Windowing" (ADWIN)*, SDM 2007.

use crate::agentic_time::{AgentClock, AgentState};

/// Pre-registered Page–Hinkley parameters. Fixed before any lead is computed.
#[derive(Clone, Copy, Debug)]
pub struct PageHinkley {
    /// `δ` — magnitude of change considered normal (tolerance). The accumulator
    /// only counts deviations beyond this, so it suppresses stationary jitter.
    pub delta: f64,
    /// `λ` — alarm threshold on the cumulative deviation from the running mean.
    pub lambda: f64,
    /// Detect upward shifts (an *increase* in the mean increment) when `true`.
    /// The agentic-drift event we predict is a *rise* in structural movement, so
    /// the upward form is the natural one; the downward form is provided for
    /// completeness and symmetry of the test.
    pub upward: bool,
}

impl PageHinkley {
    /// Construct an upward (increase-detecting) Page–Hinkley test.
    pub fn upward(delta: f64, lambda: f64) -> Self {
        PageHinkley {
            delta,
            lambda,
            upward: true,
        }
    }

    /// Construct a downward (decrease-detecting) Page–Hinkley test.
    pub fn downward(delta: f64, lambda: f64) -> Self {
        PageHinkley {
            delta,
            lambda,
            upward: false,
        }
    }

    /// Run the Page–Hinkley statistic over a raw scalar stream and return the
    /// **first index** at which it alarms, or `None` if it never does.
    ///
    /// Index `0` is treated as **padding**, not a sample (consistent with the
    /// per-transition increment convention used across the crate, where slot 0 is
    /// a padded `0.0`): it is excluded from the running mean and from the
    /// accumulator, so the artificial `0 → first-real-increment` jump cannot
    /// itself trip the detector. The same exclusion is what lets a constant-rate
    /// clock (increments `[0, 1, 1, …]`) stay un-alarmed, exactly as it does
    /// under the fixed-window `mean + kσ` alarm (whose baseline is `inc[1..]`).
    /// Alarms are therefore reported only from index 1 onward, evaluated over the
    /// real increment stream.
    pub fn first_alarm(&self, stream: &[f64]) -> Option<usize> {
        if stream.len() < 2 {
            return None;
        }
        // Running mean accumulators.
        let mut count: f64 = 0.0;
        let mut sum: f64 = 0.0;
        // Cumulative accumulator and its running extremum.
        let mut cum: f64 = 0.0;
        let mut extreme: f64 = if self.upward {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };

        for (i, &x) in stream.iter().enumerate() {
            // Slot 0 is padding (the per-transition convention pads it with 0.0);
            // exclude it from the running statistics entirely so the artificial
            // first jump from the pad into the real stream cannot trip the test.
            if i == 0 {
                continue;
            }
            // Update the running mean to INCLUDE the current sample, so the
            // reference adapts every step (the whole point vs a frozen window).
            count += 1.0;
            sum += x;
            let mean = sum / count;

            if self.upward {
                // U_T = Σ (x_t − x̄_t − δ); alarm on rise above running min.
                cum += x - mean - self.delta;
                if cum < extreme {
                    extreme = cum;
                }
                let ph = cum - extreme;
                if i >= 1 && ph > self.lambda {
                    return Some(i);
                }
            } else {
                // D_T = Σ (x_t − x̄_t + δ); alarm on drop below running max.
                cum += x - mean + self.delta;
                if cum > extreme {
                    extreme = cum;
                }
                let ph = extreme - cum;
                if i >= 1 && ph > self.lambda {
                    return Some(i);
                }
            }
        }
        None
    }
}

/// The first step at which the Page–Hinkley test, applied to a **clock's own
/// per-step increment stream**, alarms. This is the adaptive counterpart of
/// [`alarm_step`](crate::agentic_time::alarm_step): same input (`clock.increments`),
/// a different — adaptive — detector. Applying it to *any* clock keeps the
/// agentic-vs-baseline comparison fair (same detector on both sides).
pub fn adaptive_alarm_step(
    clock: &dyn AgentClock,
    trace: &[AgentState],
    ph: &PageHinkley,
) -> Option<usize> {
    let inc = clock.increments(trace);
    ph.first_alarm(&inc)
}

/// Adaptive early-warning lead: steps between the Page–Hinkley alarm and the
/// failure (0 if no alarm or the alarm is after the failure). Mirrors
/// [`early_warning_lead`](crate::agentic_time::early_warning_lead) but uses the
/// adaptive detector.
pub fn adaptive_early_warning_lead(
    clock: &dyn AgentClock,
    trace: &[AgentState],
    fail_index: usize,
    ph: &PageHinkley,
) -> usize {
    match adaptive_alarm_step(clock, trace, ph) {
        Some(a) if a <= fail_index => fail_index - a,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic_time::{
        early_warning_lead, generate_failing_trace, AgenticTime, AgenticWeights, StepCountClock,
        WindowedDeltaClock,
    };

    // -- Core Page–Hinkley behaviour on synthetic scalar streams ------------

    /// A clean step-change: stationary noise, then a sustained level shift. The
    /// detector MUST fire, and fire after the step, not before it.
    #[test]
    fn detects_a_real_step_change() {
        // 40 samples around 0.0 (±0.05), then 40 samples around 2.0 (±0.05).
        let mut stream = vec![0.0]; // slot-0 padding (per-transition convention)
        let mut seed = 1u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.05
        };
        for _ in 0..40 {
            stream.push(0.0 + noise());
        }
        let step_at = stream.len();
        for _ in 0..40 {
            stream.push(2.0 + noise());
        }
        // δ tolerates the 0.05 noise; λ requires a sustained rise.
        let ph = PageHinkley::upward(0.1, 1.0);
        let alarm = ph
            .first_alarm(&stream)
            .expect("must detect the level shift");
        assert!(
            alarm >= step_at,
            "alarm {alarm} must come at/after the step at {step_at}, never before"
        );
        // And it must fire promptly — within a handful of samples of the step.
        assert!(
            alarm <= step_at + 5,
            "alarm {alarm} should follow the step ({step_at}) promptly"
        );
    }

    /// Stationary noise only: the detector MUST NOT fire (no false alarm).
    #[test]
    fn does_not_fire_on_stationary_noise() {
        let mut stream = vec![0.0];
        let mut seed = 99u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.1
        };
        for _ in 0..200 {
            stream.push(1.0 + noise());
        }
        let ph = PageHinkley::upward(0.2, 1.0);
        assert_eq!(
            ph.first_alarm(&stream),
            None,
            "Page–Hinkley must not alarm on stationary noise (δ tolerates jitter)"
        );
    }

    /// A constant stream cannot trip the detector (zero deviation from its mean).
    #[test]
    fn constant_stream_never_alarms() {
        let stream = vec![0.0; 100];
        let ph = PageHinkley::upward(0.0, 0.5);
        assert_eq!(ph.first_alarm(&stream), None);

        let flat = vec![3.3_f64; 100];
        assert_eq!(ph.first_alarm(&flat), None);
    }

    /// Build a slot-0-padded step-change stream: `n_low` samples at `low`, then
    /// `n_high` samples at `high`.
    fn step_stream(low: f64, n_low: usize, high: f64, n_high: usize) -> Vec<f64> {
        let mut s = vec![0.0]; // slot-0 padding (per-transition convention)
        s.extend(std::iter::repeat_n(low, n_low));
        s.extend(std::iter::repeat_n(high, n_high));
        s
    }

    /// The downward form catches a level DROP and ignores a level RISE.
    #[test]
    fn downward_form_detects_a_drop_not_a_rise() {
        let down = step_stream(2.0, 30, 0.0, 30);
        let ph_down = PageHinkley::downward(0.1, 1.0);
        assert!(
            ph_down.first_alarm(&down).is_some(),
            "downward form must detect a level drop"
        );

        // A pure rise should NOT trip the downward detector.
        let up = step_stream(0.0, 30, 2.0, 30);
        assert_eq!(
            ph_down.first_alarm(&up),
            None,
            "downward form must ignore a level rise"
        );
    }

    /// A larger λ delays (or suppresses) detection; a smaller λ detects sooner.
    /// Monotonicity in the threshold is a basic correctness property.
    #[test]
    fn larger_lambda_detects_later_or_never() {
        let stream = step_stream(0.0, 30, 1.0, 60);
        let sensitive = PageHinkley::upward(0.05, 0.5).first_alarm(&stream);
        let strict = PageHinkley::upward(0.05, 5.0).first_alarm(&stream);
        assert!(sensitive.is_some());
        match (sensitive, strict) {
            (Some(a), Some(b)) => assert!(b >= a, "stricter λ must not detect earlier"),
            (Some(_), None) => {} // stricter suppressed entirely — also valid
            _ => panic!("sensitive detector should have fired"),
        }
    }

    /// A larger δ tolerance makes the test less sensitive: a small drift that the
    /// sensitive setting catches can be tolerated away by a big δ.
    #[test]
    fn larger_delta_tolerates_small_drift() {
        // A SMALL sustained drift of +0.3 after a flat phase.
        let stream = step_stream(0.0, 30, 0.3, 60);
        let sensitive = PageHinkley::upward(0.05, 0.5).first_alarm(&stream);
        let tolerant = PageHinkley::upward(0.5, 0.5).first_alarm(&stream);
        assert!(sensitive.is_some(), "small δ should catch the small drift");
        assert!(
            tolerant.is_none(),
            "δ (0.5) larger than the drift (0.3) must tolerate it away"
        );
    }

    /// Slot-0 padding is never itself reported as an alarm.
    #[test]
    fn never_alarms_on_padding_index_zero() {
        // Construct a stream whose only "movement" is the padded 0 vs a big first
        // real sample; ensure the detector does not return index 0.
        let stream = vec![0.0, 5.0, 5.0, 5.0, 5.0];
        let ph = PageHinkley::upward(0.0, 0.1);
        if let Some(a) = ph.first_alarm(&stream) {
            assert!(a >= 1, "alarm index must be ≥ 1 (never the slot-0 pad)");
        }
    }

    /// Short streams degrade gracefully (no panic, no alarm).
    #[test]
    fn short_streams_are_safe() {
        let ph = PageHinkley::upward(0.1, 1.0);
        assert_eq!(ph.first_alarm(&[]), None);
        assert_eq!(ph.first_alarm(&[0.0]), None);
        // A 2-sample stream must not panic; whatever it returns is index ≥ 1.
        if let Some(a) = ph.first_alarm(&[0.0, 1.0]) {
            assert!(a >= 1);
        }
    }

    // -- Clock-wired adaptive alarms ----------------------------------------

    /// The adaptive alarm wired to a real clock fires on the synthetic failing
    /// trace's agentic signal, and the constant-rate step clock (flat increments)
    /// never fires — same structural blindness the fixed-window alarm has, so the
    /// adaptive detector is not silently rescuing strawmen.
    #[test]
    fn adaptive_alarm_fires_on_agentic_not_on_constant_clock() {
        let tr = generate_failing_trace(0xA9E1);
        let agentic = AgenticTime::new(AgenticWeights::default());
        let ph = PageHinkley::upward(0.1, 1.0);

        let agentic_alarm = adaptive_alarm_step(&agentic, &tr.states, &ph);
        assert!(
            agentic_alarm.is_some(),
            "adaptive detector must fire on the agentic signal of the failing trace"
        );

        // The step-count clock emits a constant 1.0 per step: zero deviation from
        // its running mean, so Page–Hinkley cannot fire on it (just like the
        // fixed-window mean+kσ alarm). Confirms the adaptive detector is not a
        // free pass for constant-rate strawmen.
        let step_alarm = adaptive_alarm_step(&StepCountClock, &tr.states, &ph);
        assert_eq!(
            step_alarm, None,
            "constant-rate step clock must not alarm even under the adaptive detector"
        );
    }

    /// The adaptive lead is well-formed: a non-zero lead means the alarm preceded
    /// the failure; it is bounded by the failure index.
    #[test]
    fn adaptive_lead_is_well_formed() {
        let tr = generate_failing_trace(0xA9E1);
        let agentic = AgenticTime::new(AgenticWeights::default());
        let ph = PageHinkley::upward(0.1, 1.0);

        let lead = adaptive_early_warning_lead(&agentic, &tr.states, tr.fail_index, &ph);
        assert!(
            lead <= tr.fail_index,
            "lead cannot exceed the failure index"
        );
        if let Some(a) = adaptive_alarm_step(&agentic, &tr.states, &ph) {
            if a <= tr.fail_index {
                assert_eq!(lead, tr.fail_index - a);
            } else {
                assert_eq!(lead, 0);
            }
        }
    }

    /// Same-detector fairness: the adaptive detector can be applied to BOTH the
    /// agentic clock and the fair windowed baseline with identical parameters —
    /// the property that makes any verdict change a fair comparison, not an
    /// artifact. We only assert the mechanism runs identically on both; the
    /// outcome (who wins) is reported by the example, not asserted here.
    #[test]
    fn adaptive_detector_applies_to_both_sides_identically() {
        let tr = generate_failing_trace(0xA9E1);
        let agentic = AgenticTime::new(AgenticWeights::default());
        let baseline = WindowedDeltaClock::belief_shift(tr.baseline_window);
        let ph = PageHinkley::upward(0.1, 1.0);

        // Both calls take the SAME detector instance — same δ, same λ.
        let _a = adaptive_early_warning_lead(&agentic, &tr.states, tr.fail_index, &ph);
        let _b = adaptive_early_warning_lead(&baseline, &tr.states, tr.fail_index, &ph);
        // (Mechanism check: identical detector, both produce a defined lead.)
    }

    /// Cross-check against the fixed-window alarm on the synthetic trace: the
    /// adaptive detector is a genuinely *different* detector, so it need not agree
    /// with the fixed-window alarm — but both must be live (able to fire) on the
    /// agentic signal. This documents that we swapped the detector, not the clock.
    #[test]
    fn adaptive_and_fixed_window_are_distinct_live_detectors() {
        let tr = generate_failing_trace(0xA9E1);
        let agentic = AgenticTime::new(AgenticWeights::default());
        let bw = tr.baseline_window;

        let fixed_lead = early_warning_lead(&agentic, &tr.states, tr.fail_index, bw, 4.0);
        let ph = PageHinkley::upward(0.1, 1.0);
        let adaptive_lead = adaptive_early_warning_lead(&agentic, &tr.states, tr.fail_index, &ph);

        // Both are live on this designed trace (both can produce a lead > 0).
        assert!(
            fixed_lead > 0,
            "fixed-window alarm fires on the synthetic trace"
        );
        assert!(
            adaptive_lead > 0,
            "adaptive alarm also fires on the synthetic trace"
        );
    }
}
