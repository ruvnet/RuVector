//! Bottom and RANDOM per-query ranks from one scoring pass (ADR-007 §2.2, M0).
//!
//! The ADR-007 verdict is taken on [`TieBreak::Bottom`] (worst rank within a
//! tied block) with RANDOM reported alongside. Both are resolved from the same
//! `(greater, tied)` counts, so they cannot disagree about which candidates
//! outrank the target, and `random[i] <= bottom[i]` holds for every query by
//! construction — asserted in CI below.

use super::{counts, qseed, rank, EvalConfig, TieBreak};
use crate::data::{Rng, TripleStore};
use crate::{Result, Scorer, Tables, Triple};
use serde::{Deserialize, Serialize};

/// Element-aligned per-query filtered ranks under both tie policies. Index
/// layout matches [`super::evaluate_ranks`]: for each triple, its tail-query
/// `(s, r, ?)` rank then its head-query `(?, r, o)` rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankPair {
    /// Worst rank in the tied block (`greater + 1 + tied`) — the verdict ranks.
    pub bottom: Vec<usize>,
    /// Seeded uniform rank in the tied block (`greater + 1 + U[0, tied]`).
    /// Bit-identical to `evaluate_ranks` with `EvalConfig::random(seed)`
    /// (same `filtered`), because the per-query seeds are the same.
    pub random: Vec<usize>,
}

/// Rank both sides of every `eval_triples` triple once and resolve the ties
/// both ways. `filtered` and `seed` mean what they do in [`EvalConfig`]; head
/// queries are direct (a reciprocal model goes through
/// [`evaluate_rank_pair_with`] with [`EvalConfig::reciprocal`] set).
///
/// # Errors
/// As [`super::evaluate_ranks`]: a non-finite target score is an error.
pub fn evaluate_rank_pair<S: Scorer + ?Sized>(
    tables: &Tables,
    scorer: &S,
    filter_store: &TripleStore,
    eval_triples: &[Triple],
    filtered: bool,
    seed: u64,
) -> Result<RankPair> {
    let mut config = EvalConfig::random(seed);
    config.filtered = filtered;
    evaluate_rank_pair_with(tables, scorer, filter_store, eval_triples, &config)
}

/// The pair for an [`EvalConfig`] (its `tie_break` is ignored — both policies
/// are always produced; `reciprocal` selects the head-query protocol).
pub fn evaluate_rank_pair_with<S: Scorer + ?Sized>(
    tables: &Tables,
    scorer: &S,
    filter_store: &TripleStore,
    eval_triples: &[Triple],
    config: &EvalConfig,
) -> Result<RankPair> {
    let all = counts::all_counts(tables, scorer, filter_store, eval_triples, config)?;
    let cap = eval_triples.len() * 2;
    let mut out = RankPair {
        bottom: Vec::with_capacity(cap),
        random: Vec::with_capacity(cap),
    };
    for (&t, counts) in eval_triples.iter().zip(all) {
        // Tail uses side 1, head side 0 — the same seeds `evaluate_ranks` uses.
        for ((greater, tied), side) in counts.into_iter().zip([1u64, 0]) {
            let mut rng = Rng::seeded(qseed(config.seed, t, side));
            out.bottom
                .push(rank::resolve(greater, tied, TieBreak::Bottom, &mut rng));
            out.random
                .push(rank::resolve(greater, tied, TieBreak::Random, &mut rng));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::evaluate_ranks;
    use super::*;
    use crate::train::grad::testing::DistMult;

    /// A DistMult model with dims 1 whose entity values repeat in a short
    /// cycle, so almost every query has a large tied block — and a filter
    /// store with extra true facts, so filtering actually removes candidates.
    fn tied_fixture() -> (Tables, DistMult, TripleStore, Vec<Triple>) {
        let ne = 60usize;
        let nr = 2usize;
        let mut tables = Tables::new(ne, nr, 1, 4);
        for e in 0..ne as u32 {
            tables.entity_mut(e).unwrap()[0] = ((e % 4) as f32) - 1.0; // -1,0,1,2
        }
        tables.relation_mut(0).unwrap()[0] = 1.0;
        tables.relation_mut(1).unwrap()[0] = -0.5;
        let test: Vec<Triple> = (0..40u32)
            .map(|i| Triple::new(i, i % 2, (i * 7 + 3) % ne as u32))
            .collect();
        let mut all = test.clone();
        all.extend((0..40u32).map(|i| Triple::new(i, i % 2, (i * 11 + 5) % ne as u32)));
        all.extend(
            (0..40u32)
                .map(|i| Triple::new((i * 13 + 1) % ne as u32, i % 2, (i * 7 + 3) % ne as u32)),
        );
        let store = TripleStore::with_counts(all, Some(ne), Some(nr)).unwrap();
        (tables, DistMult::new(1), store, test)
    }

    /// ADR-007 M0 CI assertion: RANDOM ≤ Bottom for every query, over a scorer
    /// with ties and a filtered store; and both vectors agree exactly with the
    /// single-policy entry point, so the one-pass form is not a new metric.
    #[test]
    fn random_le_bottom_for_every_query() {
        let (tables, scorer, store, test) = tied_fixture();
        for filtered in [true, false] {
            for seed in [0u64, 7, 99] {
                let pair =
                    evaluate_rank_pair(&tables, &scorer, &store, &test, filtered, seed).unwrap();
                assert_eq!(pair.bottom.len(), test.len() * 2);
                assert_eq!(pair.random.len(), pair.bottom.len());
                for (i, (&r, &b)) in pair.random.iter().zip(&pair.bottom).enumerate() {
                    assert!(r >= 1 && r <= b, "query {i}: RANDOM {r} > Bottom {b}");
                }
                let cfg = |tie_break| EvalConfig {
                    tie_break,
                    filtered,
                    seed,
                    reciprocal: false,
                };
                let random =
                    evaluate_ranks(&tables, &scorer, &store, &test, &cfg(TieBreak::Random))
                        .unwrap();
                let bottom =
                    evaluate_ranks(&tables, &scorer, &store, &test, &cfg(TieBreak::Bottom))
                        .unwrap();
                assert_eq!(
                    pair.random, random,
                    "one-pass RANDOM must equal the RANDOM-only pass"
                );
                assert_eq!(
                    pair.bottom, bottom,
                    "one-pass Bottom must equal the Bottom-only pass"
                );
            }
        }
        // The fixture really has ties (otherwise the assertion is vacuous) and
        // filtering really changes ranks.
        let f = evaluate_rank_pair(&tables, &scorer, &store, &test, true, 1).unwrap();
        let raw = evaluate_rank_pair(&tables, &scorer, &store, &test, false, 1).unwrap();
        assert!(
            f.random
                .iter()
                .zip(&f.bottom)
                .filter(|(r, b)| r < b)
                .count()
                > test.len() / 2
        );
        assert_ne!(
            f.bottom, raw.bottom,
            "filtered store must remove candidates"
        );
    }
}
