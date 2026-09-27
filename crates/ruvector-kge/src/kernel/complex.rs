//! ComplEx-form front-end of the 1-N kernel.
//!
//! Rows are `k` complex dims stored split `[re_0..re_{k-1}, im_0..im_{k-1}]`
//! (width `2k`, the layout of the test-only `scorer::complex` and of plan M1's
//! product ComplEx). `score(s, r, o) = Re Σ r·s·conj(o)`, so for a query row
//!
//! ```text
//! tail (s, r, ?):  q = r ⊙ s          score(o) = [Re q; Im q] · [Re o; Im o]
//! head (?, r, o):  q = conj(r) ⊙ o    score(s) = [Re q; Im q] · [Re s; Im s]
//! ```
//!
//! and the entity table itself is the GEMM's `E` — no transform. Backward,
//! with `G = ∂L/∂q` as a complex vector:
//!
//! ```text
//! tail: ∂L/∂s = conj(r) ⊙ G    ∂L/∂r = conj(s) ⊙ G
//! head: ∂L/∂o = r ⊙ G          ∂L/∂r = o ⊙ conj(G)
//! ```
//!
//! Anchor/relation gradients are scattered sequentially in row order
//! (deterministic).

use super::{one_n_softmax_ce, OneNQuery, Reduction, Workspace};
use crate::{KgeError, Result, Side};

/// Scratch for [`complex_one_n_step`].
#[derive(Debug, Default)]
pub struct ComplexWorkspace {
    q: Vec<f32>,
    gq: Vec<f32>,
    targets: Vec<u32>,
    core: Workspace,
}

impl ComplexWorkspace {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Build the ComplEx query row for `(anchor, rel, side)` into `out` (`2k`).
pub fn complex_query(rel: &[f32], anchor: &[f32], side: Side, out: &mut [f32]) {
    let k = rel.len() / 2;
    let (rr, ri) = rel.split_at(k);
    let (ar, ai) = anchor.split_at(k);
    let (qr, qi) = out.split_at_mut(k);
    for i in 0..k {
        // tail: r·a ; head: conj(r)·a  (flip the sign of Im r)
        let ri_s = match side {
            Side::Tail => ri[i],
            Side::Head => -ri[i],
        };
        qr[i] = rr[i] * ar[i] - ri_s * ai[i];
        qi[i] = rr[i] * ai[i] + ri_s * ar[i];
    }
}

/// Pull `gq = ∂L/∂q` back to the anchor and relation rows, *adding* into
/// `g_anchor` and `g_rel` (each `2k`).
pub fn complex_query_backward(
    rel: &[f32],
    anchor: &[f32],
    side: Side,
    gq: &[f32],
    g_anchor: &mut [f32],
    g_rel: &mut [f32],
) {
    let k = rel.len() / 2;
    let (rr, ri) = rel.split_at(k);
    let (ar, ai) = anchor.split_at(k);
    let (gr, gi) = gq.split_at(k);
    let (gar, gai) = g_anchor.split_at_mut(k);
    let (grr, gri) = g_rel.split_at_mut(k);
    for i in 0..k {
        match side {
            Side::Tail => {
                // ∂a = conj(r)·G ; ∂r = conj(a)·G
                gar[i] += rr[i] * gr[i] + ri[i] * gi[i];
                gai[i] += rr[i] * gi[i] - ri[i] * gr[i];
                grr[i] += ar[i] * gr[i] + ai[i] * gi[i];
                gri[i] += ar[i] * gi[i] - ai[i] * gr[i];
            }
            Side::Head => {
                // ∂a = r·G ; ∂r = a·conj(G)
                gar[i] += rr[i] * gr[i] - ri[i] * gi[i];
                gai[i] += rr[i] * gi[i] + ri[i] * gr[i];
                grr[i] += ar[i] * gr[i] + ai[i] * gi[i];
                gri[i] += ai[i] * gr[i] - ar[i] * gi[i];
            }
        }
    }
}

/// One batched 1-N softmax-CE step for a ComplEx model.
///
/// `entities` (`N×2k`) and `relations` (`R×2k`) are the raw row-major tables
/// (e.g. [`crate::Tables::entities_raw`]). Writes dense `grad_entities`
/// (`N×2k`) and `grad_relations` (`R×2k`), both **overwritten**. Returns the
/// reduced loss over `queries.len()` rows.
#[allow(clippy::too_many_arguments)]
pub fn complex_one_n_step(
    entities: &[f32],
    relations: &[f32],
    k: usize,
    queries: &[OneNQuery],
    reduction: Reduction,
    ws: &mut ComplexWorkspace,
    grad_entities: &mut [f32],
    grad_relations: &mut [f32],
) -> Result<f32> {
    let dim = 2 * k;
    if k == 0 || !entities.len().is_multiple_of(dim) || !relations.len().is_multiple_of(dim) {
        return Err(KgeError::Invalid(
            "complex kernel: tables are not whole 2k-wide rows".into(),
        ));
    }
    let (n, nr) = (entities.len() / dim, relations.len() / dim);
    if grad_relations.len() != nr * dim {
        return Err(KgeError::Dims {
            expected: nr * dim,
            got: grad_relations.len(),
        });
    }
    let b = queries.len();
    ws.q.resize(b * dim, 0.0);
    ws.gq.resize(b * dim, 0.0);
    ws.targets.clear();
    for (qrow, qy) in ws.q.chunks_mut(dim).zip(queries) {
        let a = qy.anchor as usize;
        let r = qy.relation as usize;
        if a >= n {
            return Err(KgeError::UnknownEntity(qy.anchor));
        }
        if r >= nr {
            return Err(KgeError::UnknownRelation(qy.relation));
        }
        complex_query(
            &relations[r * dim..(r + 1) * dim],
            &entities[a * dim..(a + 1) * dim],
            qy.side,
            qrow,
        );
        ws.targets.push(qy.target);
    }

    let loss = one_n_softmax_ce(
        &ws.q,
        entities,
        dim,
        &ws.targets,
        reduction,
        &mut ws.core,
        &mut ws.gq,
        grad_entities,
    )?;

    grad_relations.fill(0.0);
    for (gq, qy) in ws.gq.chunks(dim).zip(queries) {
        let (a, r) = (qy.anchor as usize, qy.relation as usize);
        let (ga, gr) = (
            &mut grad_entities[a * dim..(a + 1) * dim],
            &mut grad_relations[r * dim..(r + 1) * dim],
        );
        complex_query_backward(
            &relations[r * dim..(r + 1) * dim],
            &entities[a * dim..(a + 1) * dim],
            qy.side,
            gq,
            ga,
            gr,
        );
    }
    Ok(loss)
}
