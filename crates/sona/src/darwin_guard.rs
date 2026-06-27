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

// ---------------------------------------------------------------------------
// Contamination guard (weight-eft / ADR-198 borrow). The training-data analog of
// the reward-hacking monitor: training or selecting on instances that appear in
// the eval holdout is *fake lift*. Enforce strict train/eval instance-ID
// disjointness — and surface what was excluded, never silently.
// ---------------------------------------------------------------------------

use std::collections::HashSet;

/// Train IDs that illegally appear in the eval holdout (the contamination set).
#[must_use]
pub fn contamination<'a>(
    train_ids: impl IntoIterator<Item = &'a str>,
    eval_holdout: &[&str],
) -> Vec<String> {
    let holdout: HashSet<&str> = eval_holdout.iter().copied().collect();
    let mut bad: Vec<String> = train_ids
        .into_iter()
        .filter(|id| holdout.contains(id))
        .map(str::to_string)
        .collect();
    bad.sort();
    bad.dedup();
    bad
}

/// `assertTrainEvalDisjoint` analog: `Err(overlapping_ids)` if any training
/// instance is in the eval holdout, else `Ok(())`. Callers should treat `Err` as
/// fatal — a contaminated training set produces fake held-out lift.
///
/// # Errors
/// Returns the sorted, de-duplicated overlapping instance IDs.
pub fn assert_train_eval_disjoint(
    train_ids: &[&str],
    eval_holdout: &[&str],
) -> Result<(), Vec<String>> {
    let bad = contamination(train_ids.iter().copied(), eval_holdout);
    if bad.is_empty() {
        Ok(())
    } else {
        Err(bad)
    }
}

/// Exporter-style contamination filter: split `items` into
/// `(kept, excluded_by_holdout)` by their instance id, so the training set is
/// disjoint from the eval holdout by construction. Pair with the export report
/// (`excluded.len()`), never drop silently.
pub fn filter_holdout<T>(
    items: Vec<T>,
    id_of: impl Fn(&T) -> &str,
    eval_holdout: &[&str],
) -> (Vec<T>, Vec<T>) {
    let holdout: HashSet<&str> = eval_holdout.iter().copied().collect();
    let mut kept = Vec::new();
    let mut excluded = Vec::new();
    for it in items {
        if holdout.contains(id_of(&it)) {
            excluded.push(it);
        } else {
            kept.push(it);
        }
    }
    (kept, excluded)
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

    #[test]
    fn disjoint_train_eval_ok_and_contamination_detected() {
        let eval = ["i-3", "i-9"];
        assert_eq!(assert_train_eval_disjoint(&["i-1", "i-2"], &eval), Ok(()));
        // Overlap is fatal and reports the contaminated ids (sorted, deduped).
        assert_eq!(
            assert_train_eval_disjoint(&["i-1", "i-9", "i-3", "i-9"], &eval),
            Err(vec!["i-3".to_string(), "i-9".to_string()])
        );
    }

    #[test]
    fn filter_holdout_partitions_by_id() {
        let items = vec![("i-1", 10), ("i-3", 20), ("i-5", 30)];
        let (kept, excluded) = filter_holdout(items, |x| x.0, &["i-3"]);
        assert_eq!(
            kept.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec!["i-1", "i-5"]
        );
        assert_eq!(
            excluded.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec!["i-3"]
        );
        // The kept set is now disjoint from the holdout by construction.
        let kept_ids: Vec<&str> = kept.iter().map(|x| x.0).collect();
        assert!(assert_train_eval_disjoint(&kept_ids, &["i-3"]).is_ok());
    }
}
