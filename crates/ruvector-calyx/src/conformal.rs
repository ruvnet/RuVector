//! Conformal cross-lens abstention.
//!
//! The hand-tuned guard threshold in [`crate::ward`] answers "when the lenses
//! agree enough" — but *how much* is enough is a magic number with no promise
//! attached. This module replaces it with **conformal risk control**
//! (Angelopoulos, Bates, Fisch, Lei & Schuster, 2022): given a calibration set,
//! it picks a threshold on the cross-lens agreement score that carries a
//! **distribution-free, finite-sample guarantee** — the expected rate of
//! confident-but-wrong answers on future queries is `≤ α`.
//!
//! ## The guarantee
//!
//! Let each query's nonconformity be captured by its top candidate's cross-lens
//! agreement `a ∈ [0,1]`, and let `wrong` mean "answering would be an error".
//! Define the per-query loss at threshold `t` as `L(t) = 1{a ≥ t ∧ wrong}` —
//! non-increasing in `t`. With `n` exchangeable calibration points, choose
//!
//! ```text
//! t̂ = inf { t : (n/(n+1))·R̂ₙ(t) + 1/(n+1) ≤ α }
//! ```
//!
//! Then for an exchangeable test query, `E[L(t̂)] ≤ α`: at most an `α` fraction
//! of queries result in a confident wrong answer. The router *answers* iff
//! `a ≥ t̂` and *abstains* otherwise — "fail closed" with a number behind it.

/// A calibrated conformal abstention rule: answer iff agreement ≥ `threshold`.
#[derive(Debug, Clone, Copy)]
pub struct ConformalSelector {
    /// Agreement threshold above which answering is permitted.
    pub threshold: f32,
    /// The risk level the threshold was calibrated to guarantee.
    pub alpha: f32,
}

impl ConformalSelector {
    /// Answer iff the top candidate's cross-lens agreement clears the threshold.
    #[inline]
    pub fn admits(&self, agreement: f32) -> bool {
        agreement >= self.threshold
    }
}

/// Calibrate a conformal abstention threshold via conformal risk control.
///
/// `cal` holds `(agreement, wrong_if_answered)` for calibration queries.
/// `alpha` is the target ceiling on the confident-wrong-answer rate.
/// Returns a selector whose threshold gives `E[loss] ≤ alpha` on exchangeable
/// test data. If no finite threshold achieves the bound, it abstains on
/// everything (threshold `> 1`).
pub fn calibrate(cal: &[(f32, bool)], alpha: f32) -> ConformalSelector {
    let n = cal.len();
    let abstain_all = ConformalSelector {
        threshold: 1.000_001,
        alpha,
    };
    if n == 0 {
        return abstain_all;
    }

    // Candidate thresholds: each distinct agreement value (answering the set
    // {a_i ≥ t}); evaluated ascending = increasingly strict = lower loss.
    let mut candidates: Vec<f32> = cal.iter().map(|&(a, _)| a).collect();
    candidates.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup();

    let nf = n as f32;
    let slack = 1.0 / (nf + 1.0); // B/(n+1) with 0/1 loss, B = 1
    for &t in &candidates {
        let wrong_admitted = cal
            .iter()
            .filter(|&&(a, wrong)| a >= t && wrong)
            .count();
        let r_hat = wrong_admitted as f32 / nf;
        if (nf / (nf + 1.0)) * r_hat + slack <= alpha {
            return ConformalSelector { threshold: t, alpha };
        }
    }
    abstain_all
}

/// Empirical loss of a selector on `data`: the fraction of *all* queries that
/// are confident wrong answers (`admitted ∧ wrong`). This is the quantity the
/// guarantee bounds by `α`.
pub fn empirical_risk(sel: &ConformalSelector, data: &[(f32, bool)]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let bad = data
        .iter()
        .filter(|&&(a, wrong)| sel.admits(a) && wrong)
        .count();
    bad as f32 / data.len() as f32
}

/// Fraction of queries the selector answers (coverage).
pub fn coverage(sel: &ConformalSelector, data: &[(f32, bool)]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let admitted = data.iter().filter(|&&(a, _)| sel.admits(a)).count();
    admitted as f32 / data.len() as f32
}

/// Selective error: among *answered* queries, the fraction that are wrong.
/// (`empirical_risk / coverage`; `0` when nothing is answered.)
pub fn selective_error(sel: &ConformalSelector, data: &[(f32, bool)]) -> f32 {
    let admitted: Vec<bool> = data
        .iter()
        .filter(|&&(a, _)| sel.admits(a))
        .map(|&(_, wrong)| wrong)
        .collect();
    if admitted.is_empty() {
        return 0.0;
    }
    admitted.iter().filter(|w| **w).count() as f32 / admitted.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Build a sample where high-agreement queries are usually right and
    /// low-agreement queries are usually wrong (the realistic regime).
    fn sample(rng: &mut Rng, n: usize) -> Vec<(f32, bool)> {
        (0..n)
            .map(|_| {
                let a = rng.f32();
                // P(wrong) decreases with agreement.
                let wrong = rng.f32() > a; // higher a → less likely wrong
                (a, wrong)
            })
            .collect()
    }

    #[test]
    fn stricter_alpha_lowers_coverage() {
        let mut rng = Rng::new(1);
        let cal = sample(&mut rng, 4000);
        let s01 = calibrate(&cal, 0.01);
        let s10 = calibrate(&cal, 0.10);
        // A tighter risk budget must not answer *more*.
        assert!(s01.threshold >= s10.threshold);
    }

    #[test]
    fn guarantee_holds_on_held_out_data_monte_carlo() {
        // Across many splits, the mean test risk must stay ≤ α (+ tiny slack).
        let alpha = 0.05;
        let trials = 200;
        let mut violations_mean = 0.0f32;
        let mut rng = Rng::new(7);
        for _ in 0..trials {
            let mut all = sample(&mut rng, 1200);
            // Shuffle then split cal/test (exchangeable).
            for i in (1..all.len()).rev() {
                let j = rng.range(i + 1);
                all.swap(i, j);
            }
            let (cal, test) = all.split_at(600);
            let sel = calibrate(cal, alpha);
            violations_mean += empirical_risk(&sel, test);
        }
        let mean_risk = violations_mean / trials as f32;
        assert!(
            mean_risk <= alpha + 0.01,
            "mean test risk {mean_risk} exceeded alpha {alpha}"
        );
    }

    #[test]
    fn abstains_on_everything_when_all_wrong() {
        let cal: Vec<(f32, bool)> = (0..100).map(|i| (i as f32 / 100.0, true)).collect();
        let sel = calibrate(&cal, 0.01);
        assert!(sel.threshold > 1.0); // cannot answer anything safely
        assert_eq!(coverage(&sel, &cal), 0.0);
    }
}
