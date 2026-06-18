//! Learned agentic-time channel weights.
//!
//! The hand-set [`crate::agentic_time::AgenticWeights`] (contradiction 1.5,
//! belief 1.0, …) are a guess. This module **learns** the per-channel weights
//! from labelled outcomes and measures, honestly, whether a learned composition
//! of the channels beats two fair competitors:
//!
//! 1. the hand-set weights used as a fixed linear scorer, and
//! 2. the single best individual channel (the fair "one scalar" baseline).
//!
//! The learner is a plain L2-regularized logistic regression (batch gradient
//! descent, feature standardization) — no external deps. The fitted
//! coefficients double as **interpretable** channel importances, and their
//! non-negative part yields clock weights for [`crate::agentic_time::AgenticTime`].
//!
//! ## Honesty guards
//!
//! * **Held-out evaluation.** Weights are fit on a train split of trace seeds
//!   and every reported number is computed on a disjoint validation split.
//! * **Circularity guard.** [`FeatureMode::Honest`] drops the contradiction
//!   channel, because in these synthetic traces failure correlates with rising
//!   contradiction by construction; the meaningful question is whether the
//!   *other* channels (plan thrash, belief jitter, retrieval instability, goal
//!   reopening) predict failure on their own.
//! * **Negative results are reported, not hidden.** The verdict prints whether
//!   learning actually beats the baselines, even when it does not.
//!
//! This is a synthetic-data harness: a positive result here is *evidence that
//! the channel composition carries signal worth pursuing on real traces*, not a
//! production claim. Real labelled traces are required to clear ADR-251
//! invariant §4 (baseline dominance).

use crate::agentic_time::AgentState;

/// Which channels feed the learner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureMode {
    /// All six channels (belief, memory, retrieval, goal, contradiction, plan).
    Full,
    /// Drop the contradiction channel (circularity guard).
    Honest,
}

impl FeatureMode {
    /// Human-readable channel names in feature order.
    pub fn channel_names(self) -> &'static [&'static str] {
        match self {
            FeatureMode::Full => &[
                "belief",
                "memory",
                "retrieval",
                "goal_graph",
                "contradiction",
                "plan",
            ],
            FeatureMode::Honest => &["belief", "memory", "retrieval", "goal_graph", "plan"],
        }
    }

    pub fn dim(self) -> usize {
        self.channel_names().len()
    }
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Per-step channel-movement feature vector for a transition `prev -> cur`.
pub fn step_features(prev: &AgentState, cur: &AgentState, mode: FeatureMode) -> Vec<f64> {
    let belief = l2(&prev.belief, &cur.belief);
    let memory = l2(&prev.memory, &cur.memory);
    let retrieval = l2(&prev.retrieval, &cur.retrieval);
    let goal = (cur.goal_graph - prev.goal_graph).abs();
    let contradiction = (cur.contradiction - prev.contradiction).abs();
    let plan = l2(&prev.plan, &cur.plan);
    match mode {
        FeatureMode::Full => vec![belief, memory, retrieval, goal, contradiction, plan],
        FeatureMode::Honest => vec![belief, memory, retrieval, goal, plan],
    }
}

// ---------------------------------------------------------------------------
// Labelled synthetic traces (deterministic).
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    /// Uniform in (0, 1].
    fn unit01(&mut self) -> f64 {
        (self.unit() + 1.0) * 0.5 + 1e-12
    }
    /// Standard normal via Box–Muller.
    fn gaussian(&mut self) -> f64 {
        let u1 = self.unit01();
        let u2 = self.unit01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A labelled trace: the agent states plus the failure step (if any).
pub struct LabeledTrace {
    pub states: Vec<AgentState>,
    pub fail_index: Option<usize>,
}

/// Generate one labelled synthetic trace. `will_fail` traces thrash (plan
/// oscillation, belief jitter, retrieval instability, goal reopening, rising
/// contradiction) before a failure step; healthy traces converge steadily.
pub fn synth_trace(seed: u64, will_fail: bool) -> LabeledTrace {
    let dim = 6;
    let steps = 100;
    let mut rng = Rng::new(seed);
    let target: Vec<f64> = (0..dim)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    // Seed-varied schedule for failing traces.
    let fail_index = if will_fail {
        Some(70 + (seed % 13) as usize)
    } else {
        None
    };
    let onset = fail_index.map(|f| f.saturating_sub(35));

    let mut states = Vec::with_capacity(steps);
    let mut tokens = 0u64;
    let mut prev_belief = vec![0.0; dim];

    for i in 0..steps {
        tokens += 120 + (rng.unit().abs() * 12.0) as u64;

        let thrashing = matches!((onset, fail_index), (Some(o), Some(f)) if i >= o && i < f + 4);

        let (belief, plan, retrieval, contradiction, goal) = if thrashing {
            let osc = if i % 2 == 0 { 1.0 } else { -1.0 };
            let o = onset.unwrap();
            let f = fail_index.unwrap();
            let p = (i - o) as f64 / (f - o).max(1) as f64;
            let belief: Vec<f64> = target
                .iter()
                .map(|&t| t + 0.3 * osc + 0.06 * rng.unit())
                .collect();
            let plan: Vec<f64> = target
                .iter()
                .map(|&t| t + 0.8 * osc + 0.10 * rng.unit())
                .collect();
            let retrieval: Vec<f64> = target.iter().map(|&t| t + 0.4 * rng.unit()).collect();
            let contradiction = (0.05 + 0.9 * p).min(0.95);
            let goal = 1.0 + (i - o) as f64 * 0.05;
            (belief, plan, retrieval, contradiction, goal)
        } else {
            // Healthy convergence (also the early phase of failing traces).
            let frac = i as f64 / steps as f64;
            let belief: Vec<f64> = target
                .iter()
                .map(|&t| frac * t + 0.02 * rng.unit())
                .collect();
            let plan = belief.clone();
            let retrieval: Vec<f64> = target
                .iter()
                .map(|&t| frac * t + 0.02 * rng.unit())
                .collect();
            let contradiction = 0.05 + 0.02 * rng.unit().abs();
            let goal = (1.0 - frac).max(0.0);
            (belief, plan, retrieval, contradiction, goal)
        };

        let memory = prev_belief.clone();
        prev_belief = belief.clone();

        states.push(AgentState {
            belief,
            memory,
            retrieval,
            goal_graph: goal,
            contradiction,
            plan,
            tokens,
        });
    }

    LabeledTrace { states, fail_index }
}

/// Build a per-step classification dataset from labelled traces. A step is
/// positive iff a failure occurs within `horizon` steps ahead; steps at or after
/// the failure are dropped (we predict the *approach*, not the aftermath).
pub fn build_dataset(
    traces: &[LabeledTrace],
    horizon: usize,
    mode: FeatureMode,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for tr in traces {
        for i in 1..tr.states.len() {
            match tr.fail_index {
                Some(f) => {
                    if i >= f {
                        continue; // at/after failure: drop
                    }
                    let label = if f - i <= horizon { 1.0 } else { 0.0 };
                    x.push(step_features(&tr.states[i - 1], &tr.states[i], mode));
                    y.push(label);
                }
                None => {
                    x.push(step_features(&tr.states[i - 1], &tr.states[i], mode));
                    y.push(0.0);
                }
            }
        }
    }
    (x, y)
}

// ---------------------------------------------------------------------------
// Logistic regression with feature standardization.
// ---------------------------------------------------------------------------

/// A fitted logistic-regression scorer over standardized channel features.
#[derive(Clone, Debug)]
pub struct LearnedWeights {
    /// Feature dimensionality.
    pub dim: usize,
    /// Coefficients in standardized-feature space (interpretable importances).
    pub coef: Vec<f64>,
    pub bias: f64,
    /// Per-feature training mean (standardization).
    pub mean: Vec<f64>,
    /// Per-feature training std (standardization).
    pub std: Vec<f64>,
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

impl LearnedWeights {
    /// Fit by L2-regularized logistic regression (batch GD) over `dim` features.
    pub fn fit(
        x: &[Vec<f64>],
        y: &[f64],
        dim: usize,
        iters: usize,
        lr: f64,
        l2_reg: f64,
    ) -> LearnedWeights {
        let d = dim;
        let n = x.len().max(1);
        // Standardize columns.
        let mut mean = vec![0.0; d];
        for row in x {
            for j in 0..d {
                mean[j] += row[j];
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mut std = vec![0.0; d];
        for row in x {
            for j in 0..d {
                std[j] += (row[j] - mean[j]).powi(2);
            }
        }
        for s in &mut std {
            *s = (*s / n as f64).sqrt().max(1e-9);
        }
        let z = |row: &[f64], coef: &[f64], bias: f64| -> f64 {
            let mut acc = bias;
            for j in 0..d {
                acc += coef[j] * (row[j] - mean[j]) / std[j];
            }
            acc
        };

        let mut coef = vec![0.0; d];
        let mut bias = 0.0;
        for _ in 0..iters {
            let mut g = vec![0.0; d];
            let mut gb = 0.0;
            for (row, &label) in x.iter().zip(y) {
                let p = sigmoid(z(row, &coef, bias));
                let err = p - label;
                for j in 0..d {
                    g[j] += err * (row[j] - mean[j]) / std[j];
                }
                gb += err;
            }
            for j in 0..d {
                coef[j] -= lr * (g[j] / n as f64 + l2_reg * coef[j]);
            }
            bias -= lr * gb / n as f64;
        }

        LearnedWeights {
            dim,
            coef,
            bias,
            mean,
            std,
        }
    }

    /// Predicted failure-approach probability for a raw feature vector.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let d = self.dim;
        let mut acc = self.bias;
        for j in 0..d {
            acc += self.coef[j] * (features[j] - self.mean[j]) / self.std[j];
        }
        sigmoid(acc)
    }

    /// Non-negative clock weights derived from the learned coefficients (the
    /// positive part, since a clock increment must stay non-negative).
    pub fn clock_weights(&self) -> Vec<f64> {
        self.coef.iter().map(|c| c.max(0.0)).collect()
    }

    /// Reconstruct a model from persisted parameters (used when loading a sealed
    /// artifact for verification).
    pub fn from_params(
        dim: usize,
        coef: Vec<f64>,
        bias: f64,
        mean: Vec<f64>,
        std: Vec<f64>,
    ) -> LearnedWeights {
        LearnedWeights {
            dim,
            coef,
            bias,
            mean,
            std,
        }
    }
}

/// A controlled **diffuse weak-signal** benchmark: `dim` Gaussian features where
/// the positive class shifts each feature `j` by `mus[j]` standard deviations.
/// Some channels carry weak signal, some are pure noise (`mu = 0`). This is the
/// regime the composition thesis targets — no single channel separates the
/// classes well, but their *weighted* combination does, and because the per-
/// channel strengths differ, the optimal weights are non-uniform (so learning
/// beats an equal-weight guess too).
///
/// Returns `(X, y)` with `n_per_class` positives and `n_per_class` negatives.
/// Deterministic in `seed`. This is explicitly a synthetic signal-composition
/// benchmark, NOT agent traces — it proves the *learner* works when its
/// assumption (distributed weak signal of varying strength) holds.
pub fn diffuse_dataset(n_per_class: usize, mus: &[f64], seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let d = mus.len();
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(2 * n_per_class);
    let mut y = Vec::with_capacity(2 * n_per_class);
    for k in 0..2 * n_per_class {
        let label = if k % 2 == 0 { 1.0 } else { 0.0 };
        let row: Vec<f64> = (0..d).map(|j| label * mus[j] + rng.gaussian()).collect();
        x.push(row);
        y.push(label);
    }
    (x, y)
}

/// Rank-based ROC AUC (Mann–Whitney). 0.5 = chance, 1.0 = perfect ranking.
pub fn auc(scores: &[f64], labels: &[f64]) -> f64 {
    let pos: Vec<f64> = scores
        .iter()
        .zip(labels)
        .filter(|(_, &l)| l > 0.5)
        .map(|(&s, _)| s)
        .collect();
    let neg: Vec<f64> = scores
        .iter()
        .zip(labels)
        .filter(|(_, &l)| l <= 0.5)
        .map(|(&s, _)| s)
        .collect();
    if pos.is_empty() || neg.is_empty() {
        return 0.5;
    }
    let mut wins = 0.0;
    for &p in &pos {
        for &nn in &neg {
            if p > nn {
                wins += 1.0;
            } else if (p - nn).abs() < 1e-12 {
                wins += 0.5;
            }
        }
    }
    wins / (pos.len() * neg.len()) as f64
}

/// Score a dataset with a fixed non-negative weight vector (a linear clock-style
/// scorer) — used to evaluate the hand-set weights as a competitor.
pub fn linear_scores(x: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|row| row.iter().zip(weights).map(|(a, b)| a * b).sum())
        .collect()
}

/// AUC of the single best individual channel (the fair "one scalar" baseline).
pub fn best_single_channel_auc(x: &[Vec<f64>], y: &[f64], dim: usize) -> (usize, f64) {
    let mut best = (0usize, 0.0f64);
    for j in 0..dim {
        let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
        let a = auc(&col, y);
        // A channel can be anti-correlated; take the stronger of a and 1-a.
        let a = a.max(1.0 - a);
        if a > best.1 {
            best = (j, a);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build disjoint train/val trace seeds, half failing / half healthy.
    fn split_traces(n_per_class: usize, train_frac: f64) -> (Vec<LabeledTrace>, Vec<LabeledTrace>) {
        let mut train = Vec::new();
        let mut val = Vec::new();
        let cut = (n_per_class as f64 * train_frac) as u64;
        for s in 0..n_per_class as u64 {
            for will_fail in [true, false] {
                let seed = (s + 1) * 2_654_435_761 + will_fail as u64;
                let tr = synth_trace(seed, will_fail);
                if s < cut {
                    train.push(tr);
                } else {
                    val.push(tr);
                }
            }
        }
        (train, val)
    }

    #[test]
    fn learned_weights_beat_chance_on_held_out() {
        let (train, val) = split_traces(40, 0.6);
        let horizon = 12;
        let mode = FeatureMode::Honest;
        let (xtr, ytr) = build_dataset(&train, horizon, mode);
        let (xva, yva) = build_dataset(&val, horizon, mode);

        let model = LearnedWeights::fit(&xtr, &ytr, mode.dim(), 600, 0.3, 1e-3);
        let scores: Vec<f64> = xva.iter().map(|r| model.predict(r)).collect();
        let learned_auc = auc(&scores, &yva);

        // Even WITHOUT the contradiction channel, the composed signal should be
        // clearly better than chance on held-out traces.
        assert!(
            learned_auc > 0.7,
            "held-out honest-mode AUC {learned_auc} should beat chance"
        );
    }

    #[test]
    fn learned_beats_handset_weights() {
        // Learning the weights should be no worse than the hand-set guess: this
        // is the defensible, robust claim. (Whether it beats the *best single
        // channel* is a separate, data-dependent question — see the next test.)
        let (train, val) = split_traces(40, 0.6);
        let horizon = 12;
        let mode = FeatureMode::Honest;
        let (xtr, ytr) = build_dataset(&train, horizon, mode);
        let (xva, yva) = build_dataset(&val, horizon, mode);

        let model = LearnedWeights::fit(&xtr, &ytr, mode.dim(), 600, 0.3, 1e-3);
        let learned: Vec<f64> = xva.iter().map(|r| model.predict(r)).collect();
        let learned_auc = auc(&learned, &yva);

        // Hand-set weights (default AgenticWeights, contradiction dropped for
        // Honest mode): belief 1.0, memory 0.5, retrieval 0.5, goal 1.0, plan 1.0.
        let handset = [1.0, 0.5, 0.5, 1.0, 1.0];
        let handset_auc = auc(&linear_scores(&xva, &handset), &yva);

        assert!(
            learned_auc >= handset_auc - 1e-9,
            "learned {learned_auc} should be >= hand-set {handset_auc}"
        );
    }

    #[test]
    fn honest_finding_single_channel_is_a_strong_baseline() {
        // HONEST NEGATIVE-ish RESULT (documented, not hidden): on this synthetic
        // generator the failure signal is concentrated in ONE planted channel
        // (plan thrash), so the best single channel is a strong baseline that the
        // learned multi-channel composition does NOT clearly beat. This mirrors
        // ADR-251 §4: composition only earns its keep when signal is spread
        // across several weak channels — which is a property of REAL traces, not
        // this single-dominant-signal synthetic. We assert the relationship that
        // actually holds so the test documents the truth rather than a wished-for
        // win.
        let (train, val) = split_traces(40, 0.6);
        let mode = FeatureMode::Honest;
        let (xtr, ytr) = build_dataset(&train, 12, mode);
        let (xva, yva) = build_dataset(&val, 12, mode);

        let model = LearnedWeights::fit(&xtr, &ytr, mode.dim(), 600, 0.3, 1e-3);
        let learned_auc = auc(
            &xva.iter().map(|r| model.predict(r)).collect::<Vec<_>>(),
            &yva,
        );
        let (_, single_auc) = best_single_channel_auc(&xva, &yva, mode.dim());

        // Both are strong; learning is competitive (within a small margin) but
        // does not beat the dominant single channel on synthetic data.
        assert!(learned_auc > 0.7 && single_auc > 0.7);
        assert!(
            (learned_auc - single_auc).abs() < 0.08,
            "learned {learned_auc} and best-single {single_auc} should be close"
        );
    }

    #[test]
    fn clock_weights_are_non_negative() {
        let (train, _val) = split_traces(20, 1.0);
        let (xtr, ytr) = build_dataset(&train, 12, FeatureMode::Full);
        let model = LearnedWeights::fit(&xtr, &ytr, FeatureMode::Full.dim(), 300, 0.3, 1e-3);
        assert!(model.clock_weights().iter().all(|&w| w >= 0.0));
    }

    /// In the regime the composition thesis actually targets — signal spread
    /// weakly across channels of *differing* strength, with pure-noise channels
    /// present — the learned composition beats BOTH the best single channel AND
    /// the equal-weight hand-set guess, on a held-out split. This is a clean
    /// existence proof that the learner earns its keep when its assumption holds.
    #[test]
    fn learned_beats_both_baselines_on_diffuse_signal() {
        // 6 channels: two strong-ish, two weak, two pure noise.
        let mus = [0.7, 0.6, 0.3, 0.3, 0.0, 0.0];
        let d = mus.len();
        let (xtr, ytr) = diffuse_dataset(2000, &mus, 0xD1FF);
        let (xva, yva) = diffuse_dataset(2000, &mus, 0x5EED);

        let model = LearnedWeights::fit(&xtr, &ytr, d, 400, 0.3, 1e-4);
        let learned_auc = auc(
            &xva.iter().map(|r| model.predict(r)).collect::<Vec<_>>(),
            &yva,
        );

        // Equal-weight hand-set guess (a fair "just sum the channels" baseline).
        let equal = vec![1.0; d];
        let handset_auc = auc(&linear_scores(&xva, &equal), &yva);

        let (_, single_auc) = best_single_channel_auc(&xva, &yva, d);

        assert!(
            learned_auc > single_auc + 0.02,
            "learned {learned_auc} should beat best single channel {single_auc}"
        );
        assert!(
            learned_auc > handset_auc + 0.005,
            "learned {learned_auc} should beat equal-weight handset {handset_auc}"
        );
    }
}
