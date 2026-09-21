//! Filtered ranking with RANDOM tie-breaking (ADR-003 §5, ADR-006 §2).
//!
//! The target's rank is `1 + #(strictly better candidates) + offset`, where
//! `offset` depends on the tie policy. RANDOM places the target uniformly
//! within its tied block: `offset = uniform_int(0, tied)` over the `tied`
//! other candidates sharing its score — so an all-tied scorer yields a mean
//! rank of `(|E|+1)/2`, not 1 (TOP) or `|E|` (BOTTOM). TOP/BOTTOM exist only
//! for the CI assertion that pins that behaviour.

use crate::data::Rng;
use crate::EntityId;
use std::collections::BTreeSet;

/// Tie-breaking policy. RANDOM is the only correct one for reported numbers;
/// TOP and BOTTOM are for the CI gate that asserts RANDOM sits between them.
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
pub(crate) fn rank_of(
    scores: &[f32],
    target: EntityId,
    filter: Option<&BTreeSet<EntityId>>,
    tie: TieBreak,
    rng: &mut Rng,
) -> usize {
    let ts = scores[target as usize];
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
        if s > ts {
            greater += 1;
        } else if s == ts {
            tied += 1;
        }
    }
    let offset = match tie {
        TieBreak::Random => rng.range_inclusive(tied as u64) as usize,
        TieBreak::Top => 0,
        TieBreak::Bottom => tied,
    };
    greater + 1 + offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_all_tied_bounds() {
        let scores = vec![0.0f32; 50];
        let mut rng = Rng::seeded(1);
        // TOP => 1, BOTTOM => |E|, RANDOM in [1, |E|].
        assert_eq!(rank_of(&scores, 0, None, TieBreak::Top, &mut rng), 1);
        assert_eq!(rank_of(&scores, 0, None, TieBreak::Bottom, &mut rng), 50);
        let r = rank_of(&scores, 0, None, TieBreak::Random, &mut rng);
        assert!((1..=50).contains(&r));
    }

    #[test]
    fn rank_filter_excludes_higher() {
        // target=0 scores lowest; entity 1 scores highest but is a true fact.
        let scores = vec![0.1f32, 0.9, 0.2];
        let mut rng = Rng::seeded(1);
        let raw = rank_of(&scores, 0, None, TieBreak::Top, &mut rng);
        let filt: BTreeSet<EntityId> = [1u32].into_iter().collect();
        let filtered = rank_of(&scores, 0, Some(&filt), TieBreak::Top, &mut rng);
        assert!(filtered < raw, "filtering a better true fact lowers rank");
    }
}
