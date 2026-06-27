//! Reward-hacking defenses for evolutionary harness/config search (ADR-271).
//!
//! Borrowed from Ornith-1.0's three-layer defense ("Self-Scaffolding LLMs for
//! Agentic Coding", DeepReinforce 2026). When an evolutionary loop is allowed to
//! evolve its own harness/config, candidates can "win" by gaming the fitness
//! rather than improving — so the search must be screened:
//!
//!   1. **Immutable boundary** — the verifier (the fitness/eval) is frozen and
//!      lives outside what evolves; the genome can only change the *inner* policy.
//!      Modelled here by keeping [`screen`] a pure function of verifier output the
//!      candidate cannot fabricate.
//!   2. **Deterministic monitor** — non-finite metrics, out-of-bounds genes, or a
//!      degenerate/collapsed "win" are flagged and the candidate is **excluded
//!      from the selection statistics** (Pareto front / advantage), NOT merely
//!      zero-scored. A zero-scored hack can still bias selection; an excluded one
//!      cannot. See [`best_accepted`].
//!   3. **Frozen judge veto** — an [`IntentJudge`] (e.g. a frozen LLM) may VETO
//!      intent-level gaming inside the allowed surface, but never *sets* the
//!      reward — it is a veto on top of the verifier, not the reward itself.

/// Outcome of screening one candidate. `Rejected` candidates are dropped from the
/// selection statistics entirely (the "exclude from advantage" rule).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// Passed all layers; carries the verifier fitness.
    Accepted(f32),
    /// Rejected; excluded from Pareto/advantage with a reason.
    Rejected(Reject),
}

/// Why a candidate was rejected (telemetry + auditability).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reject {
    /// A metric or the fitness was NaN/Inf.
    NonFinite,
    /// A gene was outside its declared bounds.
    OutOfBounds,
    /// "Won" via a collapsed/trivial path (caller-defined degeneracy check).
    Degenerate,
    /// The frozen intent-judge vetoed it.
    JudgeVeto,
}

/// Layer 3: a frozen judge that may only VETO a candidate, never set its reward.
pub trait IntentJudge {
    /// Return `true` to veto (reject) the candidate.
    fn veto(&self, fitness: f32) -> bool;
}

/// Deterministic-only screening (no judge).
#[derive(Clone, Copy, Debug, Default)]
pub struct NoJudge;
impl IntentJudge for NoJudge {
    fn veto(&self, _fitness: f32) -> bool {
        false
    }
}

/// The reward-hacking guard.
#[derive(Clone, Copy, Debug)]
pub struct Guard<J: IntentJudge = NoJudge> {
    judge: J,
}

impl Guard<NoJudge> {
    /// Deterministic-monitor-only guard (layers 1–2).
    #[must_use]
    pub fn deterministic() -> Self {
        Self { judge: NoJudge }
    }
}

impl<J: IntentJudge> Guard<J> {
    /// Guard with a layer-3 intent judge.
    pub fn with_judge(judge: J) -> Self {
        Self { judge }
    }

    /// Screen one candidate. `fitness`/`finite_metrics` come from the IMMUTABLE
    /// verifier (the candidate cannot fabricate them); `in_bounds`/`degenerate`
    /// are caller-supplied deterministic checks over the genome + its metrics.
    pub fn screen(
        &self,
        fitness: f32,
        finite_metrics: bool,
        in_bounds: bool,
        degenerate: bool,
    ) -> Verdict {
        if !finite_metrics || !fitness.is_finite() {
            return Verdict::Rejected(Reject::NonFinite);
        }
        if !in_bounds {
            return Verdict::Rejected(Reject::OutOfBounds);
        }
        if degenerate {
            return Verdict::Rejected(Reject::Degenerate);
        }
        if self.judge.veto(fitness) {
            return Verdict::Rejected(Reject::JudgeVeto);
        }
        Verdict::Accepted(fitness)
    }
}

/// Best ACCEPTED candidate, EXCLUDING every rejected one from the comparison
/// (the Ornith "exclude from advantage" rule). `None` if all were rejected.
/// NaN-safe: rejected non-finite candidates never reach the comparator.
#[must_use]
pub fn best_accepted(verdicts: &[Verdict]) -> Option<(usize, f32)> {
    verdicts
        .iter()
        .enumerate()
        .filter_map(|(i, v)| match v {
            Verdict::Accepted(f) => Some((i, *f)),
            Verdict::Rejected(_) => None,
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// Rejection counts by reason: `[non_finite, out_of_bounds, degenerate, judge_veto]`.
#[must_use]
pub fn reject_summary(verdicts: &[Verdict]) -> [usize; 4] {
    let mut c = [0usize; 4];
    for v in verdicts {
        if let Verdict::Rejected(r) = v {
            c[match r {
                Reject::NonFinite => 0,
                Reject::OutOfBounds => 1,
                Reject::Degenerate => 2,
                Reject::JudgeVeto => 3,
            }] += 1;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_is_excluded_not_zeroed() {
        let g = Guard::deterministic();
        // A NaN-producing candidate must be REJECTED (excluded), not scored 0 —
        // a 0 could still win if all real candidates score negative.
        assert_eq!(
            g.screen(f32::NAN, true, true, false),
            Verdict::Rejected(Reject::NonFinite)
        );
        assert_eq!(
            g.screen(1.0, false, true, false),
            Verdict::Rejected(Reject::NonFinite)
        );
    }

    #[test]
    fn out_of_bounds_and_degenerate_rejected() {
        let g = Guard::deterministic();
        assert_eq!(
            g.screen(5.0, true, false, false),
            Verdict::Rejected(Reject::OutOfBounds)
        );
        assert_eq!(
            g.screen(5.0, true, true, true),
            Verdict::Rejected(Reject::Degenerate)
        );
    }

    #[test]
    fn best_accepted_excludes_rejects_and_is_nan_safe() {
        // The hacked candidate (NonFinite) must NOT win even though its raw value
        // would sort highest; only accepted candidates are compared.
        let vs = [
            Verdict::Accepted(-0.5),
            Verdict::Rejected(Reject::NonFinite),
            Verdict::Accepted(-0.2),
            Verdict::Rejected(Reject::Degenerate),
        ];
        assert_eq!(best_accepted(&vs), Some((2, -0.2)));
        assert_eq!(reject_summary(&vs), [1, 0, 1, 0]);
        // All rejected → no selection (caller must handle, not crash).
        assert_eq!(
            best_accepted(&[Verdict::Rejected(Reject::OutOfBounds)]),
            None
        );
    }

    #[test]
    fn judge_vetoes_but_does_not_set_reward() {
        struct VetoHigh;
        impl IntentJudge for VetoHigh {
            fn veto(&self, fitness: f32) -> bool {
                fitness > 100.0 // an implausibly-good score smells like gaming
            }
        }
        let g = Guard::with_judge(VetoHigh);
        assert_eq!(
            g.screen(999.0, true, true, false),
            Verdict::Rejected(Reject::JudgeVeto)
        );
        assert_eq!(g.screen(1.0, true, true, false), Verdict::Accepted(1.0));
    }
}
