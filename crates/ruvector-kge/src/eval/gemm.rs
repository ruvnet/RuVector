//! Batched evaluation scoring on the GEMM kernel (plan M2 → M3).
//!
//! For a scorer with [`Scorer::eval_by_gemm`] (index = raw entity row), the
//! scores of every candidate for a chunk of `B` triples are one
//! `[2B × N] = Q · Eᵀ` product ([`crate::kernel::one_n_scores`]) where `Q`
//! holds each triple's tail query row and head query row (the reciprocal
//! `(o, r⁻¹, ?)` row for a reciprocal model) and `E` is
//! [`Tables::entities_raw`] itself — no copy, no transform. Counting then runs
//! per row, in parallel under the `parallel` feature.
//!
//! **Determinism.** The kernel's chunking is fixed (bitwise identical across
//! thread counts) and rows are counted independently, so results do not depend
//! on the thread count or on `B`: every logit is one fixed-order dot product.
//! Against the exact per-candidate [`Scorer::score`] the scores differ only by
//! floating-point reassociation, which can reorder near-ties only (exact ties
//! such as an all-zero table stay exact: every product is `0.0`).

use super::counts::{filters, head_query, Counts};
use super::{rank, EvalConfig};
use crate::data::TripleStore;
use crate::kernel::one_n_scores;
use crate::{KgeError, Result, Scorer, Side, Tables, Triple};

/// Cap on the `2B × N` logits buffer (floats): 16 Mi floats = 64 MiB.
const MAX_LOGITS: usize = 16 << 20;
/// Triples per chunk at most (keeps `Q` small when `N` is tiny).
const MAX_CHUNK: usize = 256;

/// Triples per GEMM chunk for `n` entities.
pub(super) fn chunk_size(n: usize) -> usize {
    (MAX_LOGITS / (2 * n.max(1))).clamp(1, MAX_CHUNK)
}

/// Counts for every triple, in order, through the GEMM kernel.
pub(super) fn all_counts<S: Scorer + ?Sized>(
    tables: &Tables,
    scorer: &S,
    filter_store: &TripleStore,
    eval_triples: &[Triple],
    config: &EvalConfig,
) -> Result<Vec<Counts>> {
    let (n, d) = (tables.num_entities(), tables.dims());
    let entities = tables.entities_raw();
    let b = chunk_size(n);
    let mut q = vec![0.0f32; 2 * b * d];
    let mut logits = vec![0.0f32; 2 * b * n];
    let mut out = Vec::with_capacity(eval_triples.len());

    for chunk in eval_triples.chunks(b) {
        let m = chunk.len();
        for (rows, &t) in q.chunks_mut(2 * d).zip(chunk) {
            let (tail_row, head_row) = rows.split_at_mut(d);
            let s = tables.entity(t.s)?;
            tables.entity(t.o)?; // the target must exist, as on the exact path
            write_row(tail_row, scorer, tables.relation(t.r)?, s, Side::Tail)?;
            let (anchor, rel, side) = head_query(tables, t, config.reciprocal)?;
            let (anchor, rel) = (tables.entity(anchor)?, tables.relation(rel)?);
            write_row(head_row, scorer, rel, anchor, side)?;
        }
        let lg = &mut logits[..2 * m * n];
        one_n_scores(&q[..2 * m * d], entities, d, lg)?;
        out.extend(count_rows(lg, n, chunk, filter_store, config.filtered)?);
    }
    Ok(out)
}

/// `out = scorer.query_vector(rel, anchor, side)`, length-checked.
fn write_row<S: Scorer + ?Sized>(
    out: &mut [f32],
    scorer: &S,
    rel: &[f32],
    anchor: &[f32],
    side: Side,
) -> Result<()> {
    let v = scorer.query_vector(rel, anchor, side);
    if v.len() != out.len() {
        return Err(KgeError::Dims {
            expected: out.len(),
            got: v.len(),
        });
    }
    out.copy_from_slice(&v);
    Ok(())
}

/// `(greater, tied)` of each triple's two logits rows (tail at `2i`, head at
/// `2i + 1`); any row error fails the chunk (which one is reported is
/// unspecified under `parallel`).
fn count_rows(
    logits: &[f32],
    n: usize,
    chunk: &[Triple],
    filter_store: &TripleStore,
    filtered: bool,
) -> Result<Vec<Counts>> {
    let one = |(rows, &t): (&[f32], &Triple)| -> Result<Counts> {
        let (tail_row, head_row) = rows.split_at(n);
        let [tf, hf] = filters(filter_store, t, filtered);
        Ok([
            rank::counts_of(tail_row, t.o, tf)?,
            rank::counts_of(head_row, t.s, hf)?,
        ])
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        logits
            .par_chunks(2 * n)
            .zip(chunk.par_iter())
            .map(one)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        logits.chunks(2 * n).zip(chunk.iter()).map(one).collect()
    }
}
