//! Reciprocal-aware evaluation and the GEMM scoring route (plan M3). All on
//! synthetic tables; no dataset is read.

use super::counts::all_counts;
use super::rank::counts_of;
use super::*;
use crate::scorer::ComplEx;
use crate::train::grad::testing::DistMult;
use crate::train::reciprocal;
use crate::{KgeError, Side};

/// ComplEx with the GEMM route switched off: the exact per-candidate oracle.
struct ExactComplEx(ComplEx);
impl Scorer for ExactComplEx {
    fn dims(&self) -> usize {
        self.0.dims()
    }
    fn score(&self, s: &[f32], r: &[f32], o: &[f32]) -> f32 {
        self.0.score(s, r, o)
    }
    fn query_vector(&self, r: &[f32], a: &[f32], side: Side) -> Vec<f32> {
        self.0.query_vector(r, a, side)
    }
    fn index_vector(&self, e: &[f32]) -> Vec<f32> {
        self.0.index_vector(e)
    }
    fn index_dims(&self) -> usize {
        self.0.index_dims()
    }
    fn id(&self) -> &str {
        "complex-exact-test"
    }
}

const NE: usize = 60;
const NR: usize = 3;

/// `(all true facts, eval triples)` with several true heads/tails per query,
/// so filtering removes real candidates.
fn facts() -> (Vec<Triple>, Vec<Triple>) {
    let ne = NE as u32;
    let eval: Vec<Triple> = (0..40u32)
        .map(|i| Triple::new(i, i % NR as u32, (i * 7 + 3) % ne))
        .collect();
    let mut all = eval.clone();
    all.extend((0..40u32).map(|i| Triple::new(i, i % NR as u32, (i * 11 + 5) % ne)));
    all.extend((0..40u32).map(|i| Triple::new((i * 13 + 1) % ne, i % NR as u32, (i * 7 + 3) % ne)));
    (all, eval)
}

/// Reciprocal tables (`2·NR` relation rows) at `dims`; `tied` quantises every
/// value to {-1, 0, 1} so most queries have large tied blocks.
fn tables(dims: usize, tied: bool) -> Tables {
    let mut t = Tables::new(NE, 2 * NR, dims, 17);
    if tied {
        let q = |x: &mut f32| *x = (*x * 1e3).round().clamp(-1.0, 1.0);
        t.entities_raw_mut().iter_mut().for_each(q);
        t.relations_raw_mut().iter_mut().for_each(q);
    }
    t
}

fn cfg(filtered: bool, seed: u64) -> EvalConfig {
    EvalConfig {
        tie_break: TieBreak::Random,
        filtered,
        seed,
        reciprocal: true,
    }
}

/// The phase-3 sanity runner's construction (`kernel/sanity.rs` before M3):
/// augment filter and queries, rank as plain tail queries, keep even indices;
/// the first half are the forward tails, the second half the inverse tails —
/// i.e. the head queries. Returns `(tail counts, head counts, bottom ranks)`.
type SanityOut = (Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<usize>);
fn sanity_runner<S: Scorer + ?Sized>(
    t: &Tables,
    sc: &S,
    all: &[Triple],
    eval: &[Triple],
    filtered: bool,
) -> SanityOut {
    let filter =
        TripleStore::with_counts(reciprocal::augment(t, all).unwrap(), Some(NE), Some(2 * NR))
            .unwrap();
    let queries = reciprocal::augment(t, eval).unwrap();
    let direct = EvalConfig::random(1);
    let direct = EvalConfig { filtered, ..direct };
    let counts = all_counts(t, sc, &filter, &queries, &direct).unwrap();
    let tails: Vec<_> = counts.iter().map(|c| c[0]).collect();
    let pair = evaluate_rank_pair(t, sc, &filter, &queries, filtered, 1).unwrap();
    let bottom: Vec<usize> = pair.bottom.iter().step_by(2).copied().collect();
    let half = eval.len();
    (tails[..half].to_vec(), tails[half..].to_vec(), bottom)
}

fn check_equals_sanity<S: Scorer + ?Sized>(sc: &S, dims: usize) {
    let (all, eval) = facts();
    let base = TripleStore::with_counts(all.clone(), Some(NE), Some(NR)).unwrap();
    for tied in [false, true] {
        let t = tables(dims, tied);
        for filtered in [true, false] {
            let (tails, heads, bottom) = sanity_runner(&t, sc, &all, &eval, filtered);
            let c = cfg(filtered, 9);
            let counts = all_counts(&t, sc, &base, &eval, &c).unwrap();
            for (i, cs) in counts.iter().enumerate() {
                assert_eq!(cs[0], tails[i], "tail counts, triple {i}");
                assert_eq!(cs[1], heads[i], "head counts, triple {i}");
            }
            let pair = evaluate_rank_pair_with(&t, sc, &base, &eval, &c).unwrap();
            let half = eval.len();
            for i in 0..half {
                assert_eq!(pair.bottom[2 * i], bottom[i], "tail bottom {i}");
                assert_eq!(pair.bottom[2 * i + 1], bottom[half + i], "head bottom {i}");
            }
            if tied {
                let n_tied = counts.iter().filter(|c| c[1].1 > 0).count();
                assert!(n_tied > half / 2, "fixture must have ties ({n_tied})");
            }
        }
    }
}

/// Eval proper equals the sanity-runner construction: identical `(greater,
/// tied)` counts on both sides, hence identical Bottom ranks. RANDOM draws
/// differ by design — eval seeds a head query as `qseed(seed, t, 0)` (the
/// query's identity, shared with non-reciprocal models for ADR-004 pairing),
/// the runner as `qseed(seed, (o, r⁻¹, s), 1)` — so equal counts are the
/// seed-free statement of equality.
#[test]
fn reciprocal_eval_equals_sanity_runner_exact_path() {
    check_equals_sanity(&DistMult::new(4), 4);
    check_equals_sanity(&ExactComplEx(ComplEx::new(4).unwrap()), 4);
}

#[test]
fn reciprocal_eval_equals_sanity_runner_gemm_path() {
    check_equals_sanity(&ComplEx::new(4).unwrap(), 4);
}

/// The filter the runner builds, `true_tails(o, r⁻¹)` of the augmented store,
/// is exactly `true_heads(r, o)` of the base store — which eval proper uses.
#[test]
fn reciprocal_filter_is_base_true_heads() {
    let (all, eval) = facts();
    let t = tables(4, false);
    let base = TripleStore::with_counts(all.clone(), Some(NE), Some(NR)).unwrap();
    let aug = TripleStore::with_counts(
        reciprocal::augment(&t, &all).unwrap(),
        Some(NE),
        Some(2 * NR),
    )
    .unwrap();
    let mut multi = 0;
    for &x in &eval {
        let inv = reciprocal::inverse_relation(&t, x.r).unwrap();
        let heads = base.true_heads(x.r, x.o);
        assert_eq!(aug.true_tails(x.o, inv), heads);
        multi += usize::from(heads.map_or(0, |h| h.len()) > 1);
    }
    assert!(multi > 0, "fixture must have queries with other true heads");
}

/// For ComplEx, an inverse row equal to `conj(r)` makes `(o, r⁻¹, s)` score
/// exactly like `(s, r, o)`, so reciprocal eval must reproduce direct ranks.
#[test]
fn conjugate_inverse_rows_reproduce_direct_ranks() {
    let (all, eval) = facts();
    let base = TripleStore::with_counts(all, Some(NE), Some(NR)).unwrap();
    let d = 8usize;
    let mut t = tables(d, false);
    for r in 0..NR as u32 {
        let mut row = t.relation(r).unwrap().to_vec();
        row[d / 2..].iter_mut().for_each(|x| *x = -*x);
        t.relation_mut(r + NR as u32).unwrap().copy_from_slice(&row);
    }
    let sc = ExactComplEx(ComplEx::new(d).unwrap());
    let direct = evaluate_ranks(&t, &sc, &base, &eval, &cfg(true, 3).with_reciprocal(false));
    let recip = evaluate_ranks(&t, &sc, &base, &eval, &cfg(true, 3)).unwrap();
    assert_eq!(recip, direct.unwrap());
}

/// The pre-M3 inline loop, verbatim, as the reference for "unchanged".
fn reference_ranks<S: Scorer + ?Sized>(
    t: &Tables,
    sc: &S,
    store: &TripleStore,
    eval: &[Triple],
    c: &EvalConfig,
) -> Vec<usize> {
    let mut scores = vec![0.0f32; t.num_entities()];
    let mut out = Vec::new();
    for &x in eval {
        let (s, r, o) = (
            t.entity(x.s).unwrap(),
            t.relation(x.r).unwrap(),
            t.entity(x.o).unwrap(),
        );
        for (e, v) in scores.iter_mut().enumerate() {
            *v = sc.score(s, r, t.entity(e as u32).unwrap());
        }
        let f = c.filtered.then(|| store.true_tails(x.s, x.r)).flatten();
        let (tg, tt) = counts_of(&scores, x.o, f).unwrap();
        for (e, v) in scores.iter_mut().enumerate() {
            *v = sc.score(t.entity(e as u32).unwrap(), r, o);
        }
        let f = c.filtered.then(|| store.true_heads(x.r, x.o)).flatten();
        let (hg, ht) = counts_of(&scores, x.s, f).unwrap();
        let mut rng = Rng::seeded(qseed(c.seed, x, 1));
        out.push(rank::resolve(tg, tt, c.tie_break, &mut rng));
        let mut rng = Rng::seeded(qseed(c.seed, x, 0));
        out.push(rank::resolve(hg, ht, c.tie_break, &mut rng));
    }
    out
}

/// Non-reciprocal models: the exact route is bit-for-bit the pre-M3 loop, and
/// the GEMM route (ComplEx) gives the same ranks — including every tie policy
/// on a heavily tied table, where exact equality of scores matters.
#[test]
fn non_reciprocal_behaviour_unchanged() {
    let (all, eval) = facts();
    let store = TripleStore::with_counts(all, Some(NE), Some(NR)).unwrap();
    for tied in [false, true] {
        let t = tables(4, tied);
        for tie_break in [TieBreak::Random, TieBreak::Top, TieBreak::Bottom] {
            for filtered in [true, false] {
                let c = EvalConfig {
                    tie_break,
                    filtered,
                    seed: 5,
                    reciprocal: false,
                };
                let dm = DistMult::new(4);
                let want = reference_ranks(&t, &dm, &store, &eval, &c);
                assert_eq!(evaluate_ranks(&t, &dm, &store, &eval, &c).unwrap(), want);
                let exact = ExactComplEx(ComplEx::new(4).unwrap());
                let want = reference_ranks(&t, &exact, &store, &eval, &c);
                assert_eq!(evaluate_ranks(&t, &exact, &store, &eval, &c).unwrap(), want);
                let gemm = ComplEx::new(4).unwrap();
                assert_eq!(evaluate_ranks(&t, &gemm, &store, &eval, &c).unwrap(), want);
            }
        }
    }
}

/// GEMM and exact routes agree on more triples than one chunk holds, both
/// protocols; chunk boundaries do not matter.
#[test]
fn gemm_matches_exact_across_chunks() {
    let ne = 50usize;
    let t = Tables::new(ne, 4, 16, 3);
    let eval: Vec<Triple> = (0..600u32)
        .map(|i| Triple::new(i % 50, i % 2, (i * 7 + 1) % 50))
        .collect();
    assert!(eval.len() > super::gemm::chunk_size(ne));
    let store = TripleStore::with_counts(eval.clone(), Some(ne), Some(2)).unwrap();
    for reciprocal in [false, true] {
        let c = cfg(true, 11).with_reciprocal(reciprocal);
        let g = evaluate_rank_pair_with(&t, &ComplEx::new(16).unwrap(), &store, &eval, &c);
        let x = evaluate_rank_pair_with(
            &t,
            &ExactComplEx(ComplEx::new(16).unwrap()),
            &store,
            &eval,
            &c,
        );
        assert_eq!(g.unwrap(), x.unwrap());
    }
}

/// All-zero ComplEx tables are exactly tied under GEMM (every logit is 0.0).
#[test]
fn gemm_all_zero_tables_stay_tied() {
    let (_, eval) = facts();
    let store = TripleStore::with_counts(eval.clone(), Some(NE), Some(NR)).unwrap();
    let mut t = tables(4, false);
    t.entities_raw_mut().fill(0.0);
    t.relations_raw_mut().fill(0.0);
    let sc = ComplEx::new(4).unwrap();
    for reciprocal in [false, true] {
        let mut c = cfg(false, 0).with_reciprocal(reciprocal);
        c.tie_break = TieBreak::Top;
        assert_eq!(
            evaluate(&t, &sc, &store, &eval, &c).unwrap().combined.mr,
            1.0
        );
        c.tie_break = TieBreak::Bottom;
        assert_eq!(
            evaluate(&t, &sc, &store, &eval, &c).unwrap().combined.mr,
            NE as f32
        );
    }
}

/// A diverged (NaN) model still errors through the GEMM route.
#[test]
fn gemm_nan_target_errors() {
    let (_, eval) = facts();
    let store = TripleStore::with_counts(eval.clone(), Some(NE), Some(NR)).unwrap();
    let mut t = tables(4, false);
    t.entities_raw_mut().fill(f32::NAN);
    for reciprocal in [false, true] {
        let c = cfg(true, 0).with_reciprocal(reciprocal);
        let r = evaluate(&t, &ComplEx::new(4).unwrap(), &store, &eval, &c);
        assert!(matches!(r, Err(KgeError::Scorer(_))), "got {r:?}");
    }
}

/// A table that cannot be reciprocal is rejected, never indexed out of range.
#[test]
fn reciprocal_rejects_bad_tables() {
    let (_, eval) = facts();
    let store = TripleStore::with_counts(eval.clone(), Some(NE), Some(NR)).unwrap();
    let odd = Tables::new(NE, 2 * NR + 1, 4, 1);
    for sc in [&DistMult::new(4) as &dyn Scorer, &ComplEx::new(4).unwrap()] {
        let r = evaluate(&odd, sc, &store, &eval, &cfg(true, 0));
        assert!(matches!(r, Err(KgeError::Invalid(_))), "odd rows: {r:?}");
        // R = NR / ... : a table with only 2 relation rows has R = 1, so
        // relation ids 1 and 2 have no inverse row.
        let small = Tables::new(NE, 2, 4, 1);
        let r = evaluate(&small, sc, &store, &eval, &cfg(true, 0));
        assert!(
            matches!(r, Err(KgeError::UnknownRelation(_))),
            "r >= R: {r:?}"
        );
    }
}

/// Wall-clock of the GEMM route vs the exact per-candidate route on a
/// WN18RR-sized synthetic table. `#[ignore]`d (timing only; synthetic data):
/// `cargo test --release -p ruvector-kge --features parallel --lib
/// eval::tests_reciprocal::gemm_eval_speedup -- --ignored --nocapture`.
#[test]
#[ignore]
fn gemm_eval_speedup() {
    let (ne, d, nq) = (40_943usize, 200usize, 300u32);
    let t = Tables::new(ne, 22, d, 1);
    let eval: Vec<Triple> = (0..nq)
        .map(|i| Triple::new(i * 97 % ne as u32, i % 11, i * 131 % ne as u32))
        .collect();
    let store = TripleStore::with_counts(eval.clone(), Some(ne), Some(11)).unwrap();
    let c = cfg(true, 1);
    let time = |sc: &dyn Scorer| {
        let start = std::time::Instant::now();
        let p = evaluate_rank_pair_with(&t, sc, &store, &eval, &c).unwrap();
        (start.elapsed().as_secs_f64(), p)
    };
    let (tg, pg) = time(&ComplEx::new(d).unwrap());
    let (tx, px) = time(&ExactComplEx(ComplEx::new(d).unwrap()));
    println!(
        "[eval] |E|={ne} d={d} queries={}: gemm {tg:.3}s exact {tx:.3}s speedup {:.1}x",
        2 * nq,
        tx / tg
    );
    // fp reassociation may reorder near-ties only: few queries, small shifts.
    let diffs: Vec<i64> = pg
        .bottom
        .iter()
        .zip(&px.bottom)
        .filter(|(a, b)| a != b)
        .map(|(&a, &b)| a as i64 - b as i64)
        .collect();
    let max = diffs.iter().map(|x| x.abs()).max().unwrap_or(0);
    println!(
        "[eval] rank differences: {} of {} queries, max |Δrank| {max}",
        diffs.len(),
        pg.bottom.len()
    );
    assert!(diffs.len() * 20 <= pg.bottom.len() && max <= 5, "{diffs:?}");
}

/// Receipts must not depend on how a caller batches its queries: the GEMM
/// route gives the same per-query ranks whether the triples are evaluated at
/// once or split at an arbitrary (chunk-unaligned) point — checked at a shape
/// where GEMM and exact scoring really do diverge on near-ties, so the
/// invariance is not inherited from equality with the exact route.
#[test]
fn gemm_ranks_do_not_depend_on_query_batching() {
    let (ne, d) = (6_000usize, 128usize);
    // Integer-valued rows (exact sums in any order) plus ~ulp-sized jitter:
    // a dense field of near-ties that reassociation can reorder.
    let mut t = Tables::new(ne, 8, d, 4);
    let mut jitter = crate::data::Rng::seeded(9);
    for x in t.entities_raw_mut() {
        let j = (jitter.range_inclusive(2) as f32 - 1.0) * 1e-6;
        *x = (*x * 1e3).round().clamp(-1.0, 1.0) + j;
    }
    for x in t.relations_raw_mut() {
        *x = (*x * 1e3).round().clamp(-1.0, 1.0);
    }
    let eval: Vec<Triple> = (0..400u32)
        .map(|i| Triple::new(i * 97 % ne as u32, i % 4, i * 131 % ne as u32))
        .collect();
    let store = TripleStore::with_counts(eval.clone(), Some(ne), Some(4)).unwrap();
    for reciprocal in [false, true] {
        let c = cfg(true, 2).with_reciprocal(reciprocal);
        let sc = ComplEx::new(d).unwrap();
        let whole = evaluate_rank_pair_with(&t, &sc, &store, &eval, &c).unwrap();
        let (a, b) = eval.split_at(137);
        let mut split = evaluate_rank_pair_with(&t, &sc, &store, a, &c).unwrap();
        let rest = evaluate_rank_pair_with(&t, &sc, &store, b, &c).unwrap();
        split.bottom.extend(rest.bottom);
        split.random.extend(rest.random);
        assert_eq!(split, whole, "reciprocal={reciprocal}");
        let exact = ExactComplEx(ComplEx::new(d).unwrap());
        let x = evaluate_rank_pair_with(&t, &exact, &store, &eval, &c).unwrap();
        let diverged = x
            .bottom
            .iter()
            .zip(&whole.bottom)
            .filter(|(p, q)| p != q)
            .count();
        println!("[eval] reciprocal={reciprocal}: GEMM vs exact differ on {diverged} queries");
        if !reciprocal {
            assert!(diverged > 0, "shape must exercise GEMM != exact near-ties");
        }
    }
}
