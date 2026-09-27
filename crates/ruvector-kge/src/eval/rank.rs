//! Filtered ranking with RANDOM tie-breaking (ADR-003 §5, ADR-006 §2).
//!
//! The target's rank is `1 + #(strictly better candidates) + offset`, where
//! `offset` depends on the tie policy. RANDOM places the target uniformly
//! within its tied block: `offset = uniform_int(0, tied)` over the `tied`
//! other candidates sharing its score — so an all-tied scorer yields a mean
//! rank of `(|E|+1)/2`, not 1 (TOP) or `|E|` (BOTTOM). ADR-007 takes its verdict
//! on BOTTOM with RANDOM alongside (see `pair.rs`); TOP is for the CI gate.
//!
//! ## Non-finite scores (ADR-007 M0)
//!
//! IEEE comparisons with NaN are all false, so a naive `s > ts` / `s == ts`
//! count gives a NaN *target* rank 1 — a diverged model would report MRR 1.0.
//! Therefore:
//! - a non-finite **target** score (NaN or ±inf) is an error
//!   ([`KgeError::Scorer`]), never a rank;
//! - a non-finite **candidate** score ranks as the worst possible candidate: it
//!   is counted neither as better than nor as tied with the (finite) target.
//!   This deliberately includes `+inf` — an overflowed score is a defect, not
//!   evidence of plausibility, so it must not push the target down.

use crate::data::Rng;
use crate::{EntityId, KgeError, Result};
use std::collections::BTreeSet;

/// Tie-breaking policy. ADR-007 verdicts use BOTTOM (worst rank in the tied
/// block) with RANDOM reported alongside; TOP exists for the CI tie gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieBreak {
    Random,
    Top,
    Bottom,
}

/// Rank of `target` among all entity scores. When `filter` is `Some`, other
/// true entities in it are removed from the candidate set (filtered ranking);
/// the target is never filtered even if present in the set. `rng` is only
/// consumed for [`TieBreak::Random`], so callers seed it per query.
///
/// # Errors
/// [`KgeError::Scorer`] when the target's own score is non-finite (see the
/// module docs); a non-finite candidate ranks as worst instead.
#[cfg(test)]
pub(crate) fn rank_of(
    scores: &[f32],
    target: EntityId,
    filter: Option<&BTreeSet<EntityId>>,
    tie: TieBreak,
    rng: &mut Rng,
) -> Result<usize> {
    let (greater, tied) = counts_of(scores, target, filter)?;
    Ok(resolve(greater, tied, tie, rng))
}

/// Turn `(greater, tied)` counts into a rank under `tie`. `rng` is consumed
/// only for [`TieBreak::Random`] (exactly one draw), so resolving the same
/// counts as Bottom and then as Random with one seeded RNG yields the same
/// RANDOM rank as a Random-only pass.
pub(crate) fn resolve(greater: usize, tied: usize, tie: TieBreak, rng: &mut Rng) -> usize {
    let offset = match tie {
        TieBreak::Random => rng.range_inclusive(tied as u64) as usize,
        TieBreak::Top => 0,
        TieBreak::Bottom => tied,
    };
    greater + 1 + offset
}

/// `(greater, tied)`: the number of surviving candidates that strictly outrank
/// the target and that share its score. Both tie policies derive from this one
/// scan, so Bottom and RANDOM ranks come from a single pass (ADR-007 §2.2).
///
/// # Errors
/// [`KgeError::Scorer`] when the target's own score is non-finite.
pub(crate) fn counts_of(
    scores: &[f32],
    target: EntityId,
    filter: Option<&BTreeSet<EntityId>>,
) -> Result<(usize, usize)> {
    let ts = scores[target as usize];
    if !ts.is_finite() {
        return Err(KgeError::Scorer(format!(
            "non-finite target score ({ts}); refusing to rank a diverged model"
        )));
    }
    let mut greater = 0usize;
    let mut tied = 0usize;
    for (e, &s) in scores.iter().enumerate() {
        let e = e as EntityId;
        if e == target {
            continue;
        }
        if let Some(f) = filter {
            if f.contains(&e) {
                continue;
            }
        }
        if !s.is_finite() {
            continue; // worst possible candidate: never outranks or ties
        }
        if s > ts {
            greater += 1;
        } else if s == ts {
            tied += 1;
        }
    }
    Ok((greater, tied))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_all_tied_bounds() {
        let scores = vec![0.0f32; 50];
        let mut rng = Rng::seeded(1);
        // TOP => 1, BOTTOM => |E|, RANDOM in [1, |E|].
        assert_eq!(
            rank_of(&scores, 0, None, TieBreak::Top, &mut rng).unwrap(),
            1
        );
        assert_eq!(
            rank_of(&scores, 0, None, TieBreak::Bottom, &mut rng).unwrap(),
            50
        );
        let r = rank_of(&scores, 0, None, TieBreak::Random, &mut rng).unwrap();
        assert!((1..=50).contains(&r));
    }

    #[test]
    fn rank_filter_excludes_higher() {
        // target=0 scores lowest; entity 1 scores highest but is a true fact.
        let scores = vec![0.1f32, 0.9, 0.2];
        let mut rng = Rng::seeded(1);
        let raw = rank_of(&scores, 0, None, TieBreak::Top, &mut rng).unwrap();
        let filt: BTreeSet<EntityId> = [1u32].into_iter().collect();
        let filtered = rank_of(&scores, 0, Some(&filt), TieBreak::Top, &mut rng).unwrap();
        assert!(filtered < raw, "filtering a better true fact lowers rank");
    }

    #[test]
    fn nan_target_is_an_error_not_rank_one() {
        // The pre-M0 bug: NaN compares false with everything, so greater =
        // tied = 0 and the rank came out as 1.
        let mut rng = Rng::seeded(1);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let scores = vec![bad, 0.5, 0.7];
            for tie in [TieBreak::Random, TieBreak::Top, TieBreak::Bottom] {
                let r = rank_of(&scores, 0, None, tie, &mut rng);
                assert!(
                    matches!(r, Err(KgeError::Scorer(_))),
                    "non-finite target {bad} must error, got {r:?}"
                );
            }
        }
    }

    #[test]
    fn non_finite_candidates_rank_worst() {
        // Target 0.5; one finite better (0.9), one finite tie (0.5), and three
        // non-finite candidates that must neither outrank nor tie the target.
        let scores = vec![0.5f32, 0.9, 0.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let mut rng = Rng::seeded(3);
        assert_eq!(
            rank_of(&scores, 0, None, TieBreak::Top, &mut rng).unwrap(),
            2
        );
        assert_eq!(
            rank_of(&scores, 0, None, TieBreak::Bottom, &mut rng).unwrap(),
            3
        );
        for _ in 0..50 {
            let r = rank_of(&scores, 0, None, TieBreak::Random, &mut rng).unwrap();
            assert!(
                (2..=3).contains(&r),
                "RANDOM rank {r} outside the finite tied block"
            );
        }
    }
}
