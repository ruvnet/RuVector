//! Filtered link-prediction evaluation (ADR-003 §5, ADR-006).
//!
//! For each test triple both the head `(?, r, o)` and the tail `(s, r, ?)`
//! query are ranked by scoring every entity exactly (through [`Scorer::score`]),
//! filtering out the *other* true entities, and breaking ties with a seeded
//! RANDOM policy. Metrics are reported per side and combined.

mod metrics;
mod pair;
mod rank;

pub use metrics::MetricSet;
pub use pair::{evaluate_rank_pair, evaluate_rank_pair_with, RankPair};
pub use rank::TieBreak;

use crate::data::{Rng, TripleStore};
use crate::{Result, Scorer, Tables, Triple};
use metrics::Accum;
use serde::{Deserialize, Serialize};

/// Evaluation options. `filtered` removes other true facts from the candidate
/// set; `tie_break` picks the tie policy (ADR-007: BOTTOM for verdicts, RANDOM
/// alongside — [`evaluate_rank_pair`] yields both). `seed` makes RANDOM reproducible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalConfig {
    pub tie_break: TieBreak,
    pub filtered: bool,
    pub seed: u64,
}

impl EvalConfig {
    /// The standard configuration: filtered ranking, RANDOM tie-break.
    pub fn random(seed: u64) -> Self {
        Self {
            tie_break: TieBreak::Random,
            filtered: true,
            seed,
        }
    }
}

/// Per-side and combined metrics for one evaluation run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// Head queries `(?, r, o)`.
    pub head: MetricSet,
    /// Tail queries `(s, r, ?)`.
    pub tail: MetricSet,
    /// Head and tail pooled.
    pub combined: MetricSet,
}

/// Rank every entity for both sides of each `eval_triples` triple against the
/// trained `tables`, filtering against every true fact in `filter_store`.
///
/// `scorer` is taken as `&S: Scorer + ?Sized`, so both a concrete scorer and a
/// `&dyn Scorer`/`&dyn Differentiable` are accepted.
pub fn evaluate<S: Scorer + ?Sized>(
    tables: &Tables,
    scorer: &S,
    filter_store: &TripleStore,
    eval_triples: &[Triple],
    config: &EvalConfig,
) -> Result<EvalReport> {
    let n = tables.num_entities();
    let mut head = Accum::default();
    let mut tail = Accum::default();
    let mut scores = vec![0.0f32; n];

    for &t in eval_triples {
        let (tail_rank, head_rank) =
            ranks_for_triple(tables, scorer, filter_store, t, config, &mut scores)?;
        tail.add(tail_rank);
        head.add(head_rank);
    }

    let mut combined = head.clone();
    combined.merge(&tail);
    Ok(EvalReport {
        head: head.finish(),
        tail: tail.finish(),
        combined: combined.finish(),
    })
}

/// Per-query **filtered integer ranks** for `eval_triples`, in a stable order:
/// for each triple, its tail-query rank `(s, r, ?)` then its head-query rank
/// `(?, r, o)`. The vector's length (`2 · eval_triples.len()`) and per-index
/// meaning depend only on `eval_triples`, so two models scored over the *same*
/// `eval_triples` produce element-aligned vectors — exactly what the
/// self-optimization loop pairs candidate-vs-incumbent (ADR-004). Reciprocal
/// ranks (`1 / rank`) recover MRR; the integers keep ties exact for pairing.
pub fn evaluate_ranks<S: Scorer + ?Sized>(
    tables: &Tables,
    scorer: &S,
    filter_store: &TripleStore,
    eval_triples: &[Triple],
    config: &EvalConfig,
) -> Result<Vec<usize>> {
    let n = tables.num_entities();
    let mut scores = vec![0.0f32; n];
    let mut out = Vec::with_capacity(eval_triples.len() * 2);
    for &t in eval_triples {
        let (tail_rank, head_rank) =
            ranks_for_triple(tables, scorer, filter_store, t, config, &mut scores)?;
        out.push(tail_rank);
        out.push(head_rank);
    }
    Ok(out)
}

/// The one scoring loop [`evaluate`], [`evaluate_ranks`] and
/// [`evaluate_rank_pair`] share: filtered tail then head rank of one triple,
/// reusing `scores` as scratch. Identical seeds and candidate ordering to the
/// previous inline form, so reported metrics are unchanged.
fn ranks_for_triple<S: Scorer + ?Sized>(
    tables: &Tables,
    scorer: &S,
    filter_store: &TripleStore,
    t: Triple,
    config: &EvalConfig,
    scores: &mut [f32],
) -> Result<(usize, usize)> {
    let [(tg, tt), (hg, ht)] =
        counts_for_triple(tables, scorer, filter_store, t, config.filtered, scores)?;
    let mut rng = Rng::seeded(qseed(config.seed, t, 1));
    let tail_rank = rank::resolve(tg, tt, config.tie_break, &mut rng);
    let mut rng = Rng::seeded(qseed(config.seed, t, 0));
    let head_rank = rank::resolve(hg, ht, config.tie_break, &mut rng);
    Ok((tail_rank, head_rank))
}

/// `[(greater, tied) tail, (greater, tied) head]` for one triple — the single
/// scoring pass every tie policy is resolved from.
fn counts_for_triple<S: Scorer + ?Sized>(
    tables: &Tables,
    scorer: &S,
    filter_store: &TripleStore,
    t: Triple,
    filtered: bool,
    scores: &mut [f32],
) -> Result<[(usize, usize); 2]> {
    let s = tables.entity(t.s)?;
    let r = tables.relation(t.r)?;
    let o = tables.entity(t.o)?;

    // Tail: (s, r, ?) — vary the object.
    for (e, slot) in scores.iter_mut().enumerate() {
        *slot = scorer.score(s, r, tables.entity(e as u32)?);
    }
    let filt = if filtered {
        filter_store.true_tails(t.s, t.r)
    } else {
        None
    };
    let tail = rank::counts_of(scores, t.o, filt)?;

    // Head: (?, r, o) — vary the subject.
    for (e, slot) in scores.iter_mut().enumerate() {
        *slot = scorer.score(tables.entity(e as u32)?, r, o);
    }
    let filt = if filtered {
        filter_store.true_heads(t.r, t.o)
    } else {
        None
    };
    let head = rank::counts_of(scores, t.s, filt)?;
    Ok([tail, head])
}

/// Well-mixed per-query seed so RANDOM tie-breaking is reproducible yet
/// independent across queries and sides.
fn qseed(base: u64, t: Triple, side: u64) -> u64 {
    base ^ (t.s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (t.r as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (t.o as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ side.wrapping_mul(0xD6E8_FEB8_6659_FD93)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::grad::testing::Constant;

    /// ADR-006 §2 gate: an all-tied scorer must yield a mean rank ≈ (|E|+1)/2
    /// under RANDOM, 1 under TOP, |E| under BOTTOM. Averaged across seeds
    /// (raw ranking, so filtering does not shrink the candidate set).
    #[test]
    fn eval_tie_break_gate() {
        let ne = 200usize;
        let nr = 3usize;
        let tables = Tables::new(ne, nr, 8, 1);
        let scorer = Constant::new(8);
        // 100 test triples; the scores are constant so the triple values only
        // seed the per-query RNG.
        let test: Vec<Triple> = (0..100u32)
            .map(|i| Triple::new(i, i % 3, (i + 1) % 200))
            .collect();
        let store = TripleStore::with_counts(test.clone(), Some(ne), Some(nr)).unwrap();

        let expected = (ne as f32 + 1.0) / 2.0; // 100.5

        // RANDOM, averaged over 3 seeds.
        let mut mr_sum = 0.0f32;
        let seeds = [11u64, 22, 33];
        for &seed in &seeds {
            let cfg = EvalConfig {
                tie_break: TieBreak::Random,
                filtered: false,
                seed,
            };
            let rep = evaluate(&tables, &scorer, &store, &test, &cfg).unwrap();
            mr_sum += rep.combined.mr;
        }
        let mr_random = mr_sum / seeds.len() as f32;
        println!("[eval] all-tied mean rank RANDOM={mr_random:.2} (expected ≈ {expected:.1})");
        assert!(
            (mr_random - expected).abs() < 10.0,
            "RANDOM mean rank {mr_random:.2} should be ≈ {expected:.1}, not 1 (TOP) or {ne} (BOTTOM)"
        );

        // TOP => exactly 1; BOTTOM => exactly |E|.
        let top = evaluate(
            &tables,
            &scorer,
            &store,
            &test,
            &EvalConfig {
                tie_break: TieBreak::Top,
                filtered: false,
                seed: 0,
            },
        )
        .unwrap();
        assert_eq!(top.combined.mr, 1.0, "TOP tie-break must give mean rank 1");
        let bottom = evaluate(
            &tables,
            &scorer,
            &store,
            &test,
            &EvalConfig {
                tie_break: TieBreak::Bottom,
                filtered: false,
                seed: 0,
            },
        )
        .unwrap();
        assert_eq!(
            bottom.combined.mr, ne as f32,
            "BOTTOM tie-break must give mean rank |E|"
        );
    }

    /// A scorer that has diverged: every score is NaN.
    struct NanScorer(usize);
    impl Scorer for NanScorer {
        fn dims(&self) -> usize {
            self.0
        }
        fn score(&self, _s: &[f32], _r: &[f32], _o: &[f32]) -> f32 {
            f32::NAN
        }
        fn query_vector(&self, _r: &[f32], anchor: &[f32], _side: crate::Side) -> Vec<f32> {
            vec![f32::NAN; anchor.len()]
        }
        fn index_vector(&self, e: &[f32]) -> Vec<f32> {
            vec![f32::NAN; e.len()]
        }
        fn index_dims(&self) -> usize {
            self.0
        }
        fn id(&self) -> &str {
            "nan-test"
        }
    }

    /// ADR-007 M0: an all-NaN scorer must never produce MRR 1 (pre-M0 every
    /// NaN target ranked 1). It must fail loudly instead, on every tie policy
    /// and on both public entry points.
    #[test]
    fn all_nan_scorer_does_not_produce_mrr_one() {
        let ne = 20usize;
        let tables = Tables::new(ne, 2, 4, 7);
        let test: Vec<Triple> = (0..10u32).map(|i| Triple::new(i, i % 2, i + 1)).collect();
        let store = TripleStore::with_counts(test.clone(), Some(ne), Some(2)).unwrap();
        for tie_break in [TieBreak::Random, TieBreak::Top, TieBreak::Bottom] {
            for filtered in [true, false] {
                let cfg = EvalConfig {
                    tie_break,
                    filtered,
                    seed: 5,
                };
                let rep = evaluate(&tables, &NanScorer(4), &store, &test, &cfg);
                match rep {
                    Err(crate::KgeError::Scorer(_)) => {}
                    Ok(r) => panic!(
                        "all-NaN scorer reported MRR {} instead of an error",
                        r.combined.mrr
                    ),
                    Err(e) => panic!("unexpected error kind: {e}"),
                }
                assert!(evaluate_ranks(&tables, &NanScorer(4), &store, &test, &cfg).is_err());
            }
        }
    }

    /// ADR-007 M0: the RANDOM tie-break under *filtered* ranking at |E| ≥ 1000.
    /// Every query has exactly one other true entity filtered out, so the
    /// candidate block is |E| − 1 and the expectations are exact for TOP (1)
    /// and BOTTOM (|E| − 1), and ≈ |E|/2 for RANDOM (±10%, the gate tolerance).
    #[test]
    fn eval_filtered_tie_break_at_1000_entities() {
        let ne = 1000usize;
        let tables = Tables::new(ne, 1, 8, 3);
        let scorer = Constant::new(8);
        // test: (i, 0, i+1) for i in 1..=300; extra true facts (j, 0, j+2) for
        // j in 0..=301 give each tail query (i,0,?) one other true tail (i+2)
        // and each head query (?,0,i+1) one other true head (i-1).
        let test: Vec<Triple> = (1..=300u32).map(|i| Triple::new(i, 0, i + 1)).collect();
        let mut all = test.clone();
        all.extend((0..=301u32).map(|j| Triple::new(j, 0, j + 2)));
        let store = TripleStore::with_counts(all, Some(ne), Some(1)).unwrap();

        let run = |tie_break, seed| {
            let cfg = EvalConfig {
                tie_break,
                filtered: true,
                seed,
            };
            evaluate(&tables, &scorer, &store, &test, &cfg).unwrap()
        };
        let block = (ne - 1) as f32; // candidates left after filtering one other
        assert_eq!(run(TieBreak::Top, 0).combined.mr, 1.0);
        assert_eq!(run(TieBreak::Bottom, 0).combined.mr, block);
        let expected = (block + 1.0) / 2.0; // 500
        let mut sum = 0.0f32;
        for seed in [1u64, 2, 3] {
            sum += run(TieBreak::Random, seed).combined.mr;
        }
        let mr = sum / 3.0;
        assert!(
            (mr - expected).abs() <= 0.1 * expected,
            "filtered RANDOM MR {mr:.1} should be ≈ {expected:.1}"
        );
        // And it is not collapsing to either extreme.
        assert!(mr > 0.25 * block && mr < 0.75 * block);
    }

    /// The bench tie-check (`bench/lib/arms.mjs` zeroTables) zeroes a model's
    /// tables *textually* in the FFI `toJson` body and re-hashes it, because
    /// `fromJson` verifies the hash over serde's re-serialization. That only
    /// round-trips if serde_json writes a zero f32 as exactly `0.0` inside
    /// `{"dims":D,"entities":[...],"relations":[...]}`. Pin that format here.
    #[test]
    fn zeroed_tables_serialize_as_literal_zero_point_zero() {
        let mut t = Tables::new(3, 2, 4, 9);
        for e in 0..3u32 {
            t.entity_mut(e).unwrap().iter_mut().for_each(|x| *x = 0.0);
        }
        for r in 0..2u32 {
            t.relation_mut(r).unwrap().iter_mut().for_each(|x| *x = 0.0);
        }
        let json = serde_json::to_string(&t).unwrap();
        let zeros = |n: usize| vec!["0.0"; n].join(",");
        assert_eq!(
            json,
            format!(
                "{{\"dims\":4,\"entities\":[{}],\"relations\":[{}]}}",
                zeros(12),
                zeros(8)
            )
        );
        // An all-zero HolE model scores every triple exactly 0 → all tied.
        let sc = crate::HolE::new(4).unwrap();
        let z = t.entity(0).unwrap();
        assert_eq!(sc.score(z, t.relation(0).unwrap(), z), 0.0);
    }

    /// Filtered ranking must differ from raw when a true fact would outrank the
    /// target. Uses the DistMult test scorer with hand-set embeddings.
    #[test]
    fn eval_filtered_vs_raw_differ() {
        use crate::train::grad::testing::DistMult;
        let ne = 5usize;
        let nr = 1usize;
        let dims = 1usize;
        let mut tables = Tables::new(ne, nr, dims, 1);
        // score(s,r,o) = s0*r0*o0. Set r0=1, s0=1, so score = o0.
        tables.relation_mut(0).unwrap()[0] = 1.0;
        tables.entity_mut(0).unwrap()[0] = 1.0; // subject
        tables.entity_mut(1).unwrap()[0] = 5.0; // a true tail, scores highest
        tables.entity_mut(2).unwrap()[0] = 2.0; // the target tail
        tables.entity_mut(3).unwrap()[0] = 0.0;
        tables.entity_mut(4).unwrap()[0] = 0.0;
        let scorer = DistMult::new(dims);
        // Both (0,0,1) and (0,0,2) are true; target triple is (0,0,2).
        let store = TripleStore::with_counts(
            vec![Triple::new(0, 0, 1), Triple::new(0, 0, 2)],
            Some(ne),
            Some(nr),
        )
        .unwrap();
        let target = [Triple::new(0, 0, 2)];

        let raw = evaluate(
            &tables,
            &scorer,
            &store,
            &target,
            &EvalConfig {
                tie_break: TieBreak::Top,
                filtered: false,
                seed: 0,
            },
        )
        .unwrap();
        let filtered = evaluate(
            &tables,
            &scorer,
            &store,
            &target,
            &EvalConfig {
                tie_break: TieBreak::Top,
                filtered: true,
                seed: 0,
            },
        )
        .unwrap();
        // Raw: entity 1 (score 5) outranks target (score 2) => tail rank 2.
        // Filtered: entity 1 removed => tail rank 1.
        assert!(
            filtered.tail.mr < raw.tail.mr,
            "filtered tail MR {} should be below raw {}",
            filtered.tail.mr,
            raw.tail.mr
        );
        assert_eq!(filtered.tail.mr, 1.0);
        assert_eq!(raw.tail.mr, 2.0);
    }
}
