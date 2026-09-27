//! The one scoring pass every evaluation entry point resolves ranks from:
//! per-triple `(greater, tied)` counts for the tail query `(s, r, ?)` and the
//! head query `(?, r, o)`.
//!
//! ## Reciprocal models (ADR-007 §3, plan M3)
//!
//! A model trained with reciprocal relations (`TrainConfig::reciprocal`) only
//! ever learned **tail** queries; row `r + R` of its relation table is `r`'s
//! inverse (`R = num_relations / 2`, the convention of
//! [`crate::train::reciprocal`]). With [`EvalConfig::reciprocal`] set, the head
//! query `(?, r, o)` is therefore scored as the tail query `(o, r⁻¹, ?)`. The
//! target is still `s` and the filter is still `true_heads(r, o)` of the
//! caller's ordinary (base-id) filter store — `true_tails(o, r⁻¹)` of the
//! reciprocal-augmented store is the same set — so callers never augment.
//! Tie-break seeds are those of the head query itself (`qseed(seed, t, 0)`),
//! not of the augmented triple, so a reciprocal and a non-reciprocal model over
//! the same `eval_triples` draw identical per-query tie offsets (ADR-004
//! pairing).
//!
//! ## Scoring route
//!
//! A scorer with [`Scorer::eval_by_gemm`] (ComplEx) scores whole chunks of
//! queries against the raw entity table as one GEMM (`super::gemm`); every
//! other scorer keeps the exact per-candidate [`Scorer::score`] loop, bit for
//! bit what it was before.

use super::{gemm, rank, EvalConfig};
use crate::data::TripleStore;
use crate::train::reciprocal;
use crate::{KgeError, Result, Scorer, Tables, Triple};

/// `(greater, tied)` for the tail then the head query of one triple.
pub(super) type Counts = [(usize, usize); 2];

/// The head query of `t` as `(anchor, relation, side)` under the protocol:
/// `(o, r, Head)` directly, or `(o, r⁻¹, Tail)` for a reciprocal model.
pub(super) fn head_query(
    tables: &Tables,
    t: Triple,
    reciprocal: bool,
) -> Result<(u32, u32, crate::Side)> {
    if reciprocal {
        Ok((
            t.o,
            reciprocal::inverse_relation(tables, t.r)?,
            crate::Side::Tail,
        ))
    } else {
        Ok((t.o, t.r, crate::Side::Head))
    }
}

/// Reject a table that cannot be a reciprocal model: the relation row count
/// must be even and non-zero (`[base 0..R | inverse R..2R]`). Per-triple
/// `r < R` is checked by [`reciprocal::inverse_relation`].
fn validate_reciprocal(tables: &Tables) -> Result<()> {
    let nr = tables.num_relations();
    if nr == 0 || !nr.is_multiple_of(2) {
        return Err(KgeError::Invalid(format!(
            "reciprocal evaluation needs an even, non-zero number of relation rows \
             (row r + R is r's inverse), got {nr}"
        )));
    }
    Ok(())
}

/// Counts for every triple of `eval_triples`, in order.
pub(super) fn all_counts<S: Scorer + ?Sized>(
    tables: &Tables,
    scorer: &S,
    filter_store: &TripleStore,
    eval_triples: &[Triple],
    config: &EvalConfig,
) -> Result<Vec<Counts>> {
    if config.reciprocal {
        validate_reciprocal(tables)?;
    }
    if scorer.eval_by_gemm()
        && scorer.index_dims() == tables.dims()
        && scorer.dims() == tables.dims()
    {
        return gemm::all_counts(tables, scorer, filter_store, eval_triples, config);
    }
    let mut scores = vec![0.0f32; tables.num_entities()];
    eval_triples
        .iter()
        .map(|&t| exact_counts(tables, scorer, filter_store, t, config, &mut scores))
        .collect()
}

/// The filter sets of `t`'s tail and head queries (both `None` when raw).
pub(super) fn filters(
    filter_store: &TripleStore,
    t: Triple,
    filtered: bool,
) -> [Option<&std::collections::BTreeSet<u32>>; 2] {
    if filtered {
        [
            filter_store.true_tails(t.s, t.r),
            filter_store.true_heads(t.r, t.o),
        ]
    } else {
        [None, None]
    }
}

/// Exact per-candidate counts of one triple, reusing `scores` as scratch.
fn exact_counts<S: Scorer + ?Sized>(
    tables: &Tables,
    scorer: &S,
    filter_store: &TripleStore,
    t: Triple,
    config: &EvalConfig,
    scores: &mut [f32],
) -> Result<Counts> {
    let s = tables.entity(t.s)?;
    let r = tables.relation(t.r)?;
    let o = tables.entity(t.o)?;
    let [tail_filter, head_filter] = filters(filter_store, t, config.filtered);

    // Tail: (s, r, ?) — vary the object.
    for (e, slot) in scores.iter_mut().enumerate() {
        *slot = scorer.score(s, r, tables.entity(e as u32)?);
    }
    let tail = rank::counts_of(scores, t.o, tail_filter)?;

    // Head: (?, r, o) — vary the subject; reciprocal: (o, r⁻¹, ?).
    if config.reciprocal {
        let r_inv = tables.relation(reciprocal::inverse_relation(tables, t.r)?)?;
        for (e, slot) in scores.iter_mut().enumerate() {
            *slot = scorer.score(o, r_inv, tables.entity(e as u32)?);
        }
    } else {
        for (e, slot) in scores.iter_mut().enumerate() {
            *slot = scorer.score(tables.entity(e as u32)?, r, o);
        }
    }
    let head = rank::counts_of(scores, t.s, head_filter)?;
    Ok([tail, head])
}
